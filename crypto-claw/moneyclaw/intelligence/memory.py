"""
NeuralMemorySystem — Observations with confidence, evidence tracking,
and recency-weighted decay.

Memories are the raw building blocks of intelligence. Every trade outcome,
volume anomaly, regime shift, and pattern confirmation creates or updates
a memory. Over time, unused memories decay; frequently confirmed ones
grow stronger.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from moneyclaw.persistence.database import Database

logger = logging.getLogger(__name__)

# Decay constants
DECAY_HALF_LIFE_HOURS = 72  # memory halves in relevance every 3 days
DECAY_LAMBDA = math.log(2) / (DECAY_HALF_LIFE_HOURS * 3600)
MIN_RECENCY_WEIGHT = 0.05  # floor — below this, memory is prunable


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


class NeuralMemorySystem:
    """Store, confirm, refute, and recall observations with decay.

    Parameters
    ----------
    db:
        The shared Database instance (schema already initialised).
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        category: str,
        subject: str,
        observation: str,
        confidence: float = 0.5,
        tags: str = "",
        metadata: dict | None = None,
        ttl_hours: float | None = None,
    ) -> int:
        """Create a new memory or strengthen an existing one.

        If a memory with the same category + subject + observation already
        exists, we confirm it instead of creating a duplicate.

        Returns the memory id.
        """
        # Check for existing
        existing = self._find_existing(category, subject, observation)
        if existing:
            self.confirm(existing["id"])
            return existing["id"]

        now = _utcnow_iso()
        expires_at = None
        if ttl_hours is not None:
            expires_at = (
                datetime.utcnow() + timedelta(hours=ttl_hours)
            ).isoformat()

        meta_str = json.dumps(metadata or {})
        confidence = max(0.0, min(1.0, confidence))

        with self.db._cursor() as cur:
            cur.execute(
                """
                INSERT INTO intel_memories
                    (category, subject, observation, confidence, evidence,
                     recency_weight, tags, metadata, created_at, updated_at,
                     expires_at)
                VALUES (?, ?, ?, ?, 1, 1.0, ?, ?, ?, ?, ?)
                """,
                (
                    category, subject, observation, confidence,
                    tags, meta_str, now, now, expires_at,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Confirm / Refute
    # ------------------------------------------------------------------

    def confirm(self, memory_id: int, confidence_boost: float = 0.05) -> None:
        """Strengthen a memory — increase evidence count and confidence."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                UPDATE intel_memories
                SET evidence = evidence + 1,
                    confidence = MIN(1.0, confidence + ?),
                    recency_weight = 1.0,
                    updated_at = ?
                WHERE id = ?
                """,
                (confidence_boost, _utcnow_iso(), memory_id),
            )

    def refute(self, memory_id: int, confidence_penalty: float = 0.1) -> None:
        """Weaken a memory — decrease confidence."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                UPDATE intel_memories
                SET confidence = MAX(0.0, confidence - ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (confidence_penalty, _utcnow_iso(), memory_id),
            )

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        category: str | None = None,
        subject: str | None = None,
        tags: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Retrieve memories filtered by category, subject, or tags.

        Results are ordered by (confidence * recency_weight) descending.
        """
        query = "SELECT * FROM intel_memories WHERE 1=1"
        params: list[Any] = []

        if category:
            query += " AND category = ?"
            params.append(category)
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        if tags:
            query += " AND tags LIKE ?"
            params.append(f"%{tags}%")
        if min_confidence > 0:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        # Exclude expired memories
        query += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(_utcnow_iso())

        query += " ORDER BY (confidence * recency_weight) DESC LIMIT ?"
        params.append(limit)

        with self.db._cursor() as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def recall_strongest(
        self,
        category: str,
        limit: int = 10,
    ) -> list[dict]:
        """Get the strongest memories in a category."""
        return self.recall(category=category, min_confidence=0.3, limit=limit)

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def apply_decay(self) -> int:
        """Apply recency decay to all memories. Returns count of decayed.

        Uses batch updates to avoid N+1 query pattern.
        """
        now = datetime.utcnow()
        decayed = 0

        with self.db._cursor() as cur:
            cur.execute("SELECT id, updated_at, recency_weight FROM intel_memories")
            rows = cur.fetchall()

            # Collect all updates and apply in a single transaction
            updates: list[tuple[float, int]] = []
            for row in rows:
                updated_at = datetime.fromisoformat(row["updated_at"])
                elapsed_seconds = (now - updated_at).total_seconds()
                new_weight = math.exp(-DECAY_LAMBDA * elapsed_seconds)
                new_weight = max(MIN_RECENCY_WEIGHT, new_weight)

                if abs(new_weight - row["recency_weight"]) > 0.01:
                    updates.append((new_weight, row["id"]))

            if updates:
                cur.executemany(
                    "UPDATE intel_memories SET recency_weight = ? WHERE id = ?",
                    updates,
                )
                decayed = len(updates)

        return decayed

    def prune_expired(self) -> int:
        """Remove expired and near-zero-weight memories. Returns count deleted."""
        now = _utcnow_iso()
        with self.db._cursor() as cur:
            cur.execute(
                """
                DELETE FROM intel_memories
                WHERE (expires_at IS NOT NULL AND expires_at < ?)
                   OR (recency_weight < ? AND confidence < 0.2)
                """,
                (now, MIN_RECENCY_WEIGHT + 0.01),
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return memory system statistics."""
        with self.db._cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM intel_memories")
            total = cur.fetchone()["total"]

            cur.execute(
                "SELECT COUNT(*) as strong FROM intel_memories WHERE confidence >= 0.7"
            )
            strong = cur.fetchone()["strong"]

            cur.execute(
                """SELECT category, COUNT(*) as cnt
                   FROM intel_memories
                   GROUP BY category
                   ORDER BY cnt DESC"""
            )
            by_category = {r["category"]: r["cnt"] for r in cur.fetchall()}

        return {
            "total_memories": total,
            "strong_memories": strong,
            "by_category": by_category,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_existing(
        self, category: str, subject: str, observation: str,
    ) -> dict | None:
        """Find a memory with same category + subject + similar observation.

        Uses exact match on category+subject first, then checks if
        observation matches or is a minor variant (e.g. different pnl_pct
        formatting). This prevents near-duplicate memories.
        """
        with self.db._cursor() as cur:
            # Try exact match first
            cur.execute(
                """
                SELECT * FROM intel_memories
                WHERE category = ? AND subject = ? AND observation = ?
                LIMIT 1
                """,
                (category, subject, observation),
            )
            row = cur.fetchone()
            if row:
                return dict(row)

            # For trade_outcome/big_win/big_loss categories, match on
            # category+subject only (pnl formatting creates false negatives)
            if category in ("trade_outcome", "big_win", "big_loss"):
                cur.execute(
                    """
                    SELECT * FROM intel_memories
                    WHERE category = ? AND subject = ?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (category, subject),
                )
                row = cur.fetchone()
                return dict(row) if row else None

            return None
