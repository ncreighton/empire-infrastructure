"""
SkillEngine — Learned trading behaviors that influence signal confidence.

A skill has:
  - trigger: conditions that must match (strategy, regime, indicator ranges)
  - action:  confidence delta to apply (positive = boost, negative = reduce)
  - context: coins, timeframes, etc.

Safety bounds:
  - Single skill: [-0.25, +0.25]
  - Cumulative per evaluation: [-0.35, +0.35]
  - Skills CANNOT override risk manager, force trades, or change SL/TP

Lifecycle: dormant -> active (after validation) -> deprecated (if <30% after 20 uses)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from moneyclaw.persistence.database import Database

logger = logging.getLogger(__name__)

# Safety bounds
MAX_SINGLE_DELTA = 0.25
MAX_CUMULATIVE_DELTA = 0.35
MIN_SUCCESS_RATE_DEPRECATION = 0.30
MIN_USES_FOR_DEPRECATION = 20
MIN_USES_FOR_ACTIVATION = 5


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


class SkillEngine:
    """Manage learned trading skills with hot-path evaluation.

    Skills are loaded into memory on init and refreshed periodically
    to keep the hot-path (get_adjustments) allocation-free and fast.
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self._active_skills: list[dict] = []
        self.refresh_cache()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def refresh_cache(self) -> None:
        """Reload all active skills from DB into memory."""
        with self.db._cursor() as cur:
            cur.execute(
                "SELECT * FROM intel_skills WHERE status = 'active'"
            )
            rows = cur.fetchall()
            self._active_skills = [dict(r) for r in rows]

        logger.debug("SkillEngine cache refreshed: %d active skills", len(self._active_skills))

    # ------------------------------------------------------------------
    # Hot-path evaluation
    # ------------------------------------------------------------------

    def get_adjustments(
        self,
        product_id: str,
        strategy_name: str,
        regime: str,
        indicators: Any,
    ) -> float:
        """Evaluate all active skills and return cumulative confidence delta.

        This runs in the hot path of every coin evaluation, so it must be fast.

        Parameters
        ----------
        product_id : str
            The coin being evaluated (e.g. "BTC-USD").
        strategy_name : str
            Strategy name (e.g. "momentum").
        regime : str
            Current market regime (e.g. "trending_up").
        indicators : Any
            Indicators dataclass with RSI, MACD, etc.

        Returns
        -------
        float
            Cumulative delta, clamped to [-0.35, +0.35].
        """
        if not self._active_skills:
            return 0.0

        cumulative = 0.0

        for skill in self._active_skills:
            try:
                if not self._matches_trigger(skill, product_id, strategy_name, regime, indicators):
                    continue

                action = json.loads(skill["action_json"])
                delta = float(action.get("boost", 0.0))

                # Clamp single skill delta
                delta = max(-MAX_SINGLE_DELTA, min(MAX_SINGLE_DELTA, delta))
                cumulative += delta

            except Exception:
                # Never let a bad skill crash the hot path
                continue

        # Clamp cumulative
        cumulative = max(-MAX_CUMULATIVE_DELTA, min(MAX_CUMULATIVE_DELTA, cumulative))
        return cumulative

    # ------------------------------------------------------------------
    # Skill lifecycle
    # ------------------------------------------------------------------

    def create_skill(
        self,
        name: str,
        trigger: dict,
        action: dict,
        context: dict | None = None,
        source_pattern_id: int | None = None,
    ) -> int | None:
        """Create a new skill in dormant status.

        Returns the skill id, or None if a skill with that name exists.
        """
        now = _utcnow_iso()

        # Enforce action bounds
        boost = action.get("boost", 0.0)
        action["boost"] = max(-MAX_SINGLE_DELTA, min(MAX_SINGLE_DELTA, boost))

        try:
            with self.db._cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO intel_skills
                        (name, trigger_json, action_json, context_json,
                         status, source_pattern_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'dormant', ?, ?, ?)
                    """,
                    (
                        name,
                        json.dumps(trigger),
                        json.dumps(action),
                        json.dumps(context or {}),
                        source_pattern_id,
                        now, now,
                    ),
                )
                return cur.lastrowid
        except Exception:
            # UNIQUE constraint violation = skill already exists
            logger.debug("Skill '%s' already exists", name)
            return None

    def activate_skill(self, skill_id: int) -> None:
        """Move a dormant skill to active status."""
        now = _utcnow_iso()
        with self.db._cursor() as cur:
            cur.execute(
                """
                UPDATE intel_skills
                SET status = 'active', activated_at = ?, updated_at = ?
                WHERE id = ? AND status = 'dormant'
                """,
                (now, now, skill_id),
            )
        self.refresh_cache()

    def deprecate_skill(self, skill_id: int, reason: str = "") -> None:
        """Move a skill to deprecated status."""
        now = _utcnow_iso()
        with self.db._cursor() as cur:
            cur.execute(
                """
                UPDATE intel_skills
                SET status = 'deprecated', deprecated_at = ?,
                    deprecation_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, reason, now, skill_id),
            )
        self.refresh_cache()

    # ------------------------------------------------------------------
    # Outcome tracking
    # ------------------------------------------------------------------

    def record_outcome(self, skill_id: int, success: bool) -> None:
        """Record whether a skill-influenced trade was successful.

        Automatically deprecates skills below 30% success after 20 uses.
        """
        with self.db._cursor() as cur:
            if success:
                cur.execute(
                    """
                    UPDATE intel_skills
                    SET total_uses = total_uses + 1,
                        success_count = success_count + 1,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (_utcnow_iso(), skill_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE intel_skills
                    SET total_uses = total_uses + 1,
                        failure_count = failure_count + 1,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (_utcnow_iso(), skill_id),
                )

            # Update success rate
            cur.execute("SELECT * FROM intel_skills WHERE id = ?", (skill_id,))
            row = cur.fetchone()
            if row:
                total = row["total_uses"]
                if total > 0:
                    rate = row["success_count"] / total
                    cur.execute(
                        "UPDATE intel_skills SET success_rate = ? WHERE id = ?",
                        (rate, skill_id),
                    )

                    # Auto-deprecation check
                    if (
                        total >= MIN_USES_FOR_DEPRECATION
                        and rate < MIN_SUCCESS_RATE_DEPRECATION
                        and row["status"] == "active"
                    ):
                        self.deprecate_skill(
                            skill_id,
                            f"Auto-deprecated: {rate:.1%} success after {total} uses",
                        )
                        logger.info(
                            "Skill '%s' auto-deprecated (%.1f%% success, %d uses)",
                            row["name"], rate * 100, total,
                        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_skills(self, status: str | None = None) -> list[dict]:
        """Return all skills, optionally filtered by status."""
        query = "SELECT * FROM intel_skills"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"

        with self.db._cursor() as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def get_stats(self) -> dict:
        """Return skill engine statistics."""
        with self.db._cursor() as cur:
            stats: dict[str, Any] = {}
            for s in ("dormant", "active", "deprecated"):
                cur.execute(
                    "SELECT COUNT(*) as cnt FROM intel_skills WHERE status = ?",
                    (s,),
                )
                stats[f"{s}_skills"] = cur.fetchone()["cnt"]

            cur.execute(
                """SELECT AVG(success_rate) as avg_rate
                   FROM intel_skills WHERE status = 'active' AND total_uses > 0"""
            )
            row = cur.fetchone()
            stats["avg_active_success_rate"] = row["avg_rate"] if row["avg_rate"] else 0.0

        return stats

    # ------------------------------------------------------------------
    # Trigger matching
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_trigger(
        skill: dict,
        product_id: str,
        strategy_name: str,
        regime: str,
        indicators: Any,
    ) -> bool:
        """Check if current conditions match a skill's trigger.

        Trigger JSON format:
        {
            "strategy": "mean_reversion",     // optional
            "regime": "breakout",             // optional
            "rsi_below": 35,                  // optional
            "rsi_above": 70,                  // optional
            "volume_ratio_above": 1.5,        // optional
        }

        Context JSON format:
        {
            "coins": ["DOGE-USD", "SHIB-USD"],  // optional — empty = all coins
        }
        """
        try:
            trigger = json.loads(skill["trigger_json"])
            context = json.loads(skill["context_json"])
        except (json.JSONDecodeError, TypeError):
            return False

        # Check context: coin filter
        coins = context.get("coins", [])
        if coins and product_id not in coins:
            return False

        # Check trigger: strategy
        if "strategy" in trigger and trigger["strategy"] != strategy_name:
            return False

        # Check trigger: regime
        if "regime" in trigger and trigger["regime"] != regime:
            return False

        # Check trigger: indicator conditions
        if indicators is not None:
            rsi = getattr(indicators, "rsi", None)
            if rsi is not None:
                if "rsi_below" in trigger and rsi >= trigger["rsi_below"]:
                    return False
                if "rsi_above" in trigger and rsi <= trigger["rsi_above"]:
                    return False

            # Volume ratio check
            volume = getattr(indicators, "volume", None)
            volume_sma = getattr(indicators, "volume_sma", None)
            if "volume_ratio_above" in trigger and volume is not None and volume_sma and volume_sma > 0:
                ratio = volume / volume_sma
                if ratio < trigger["volume_ratio_above"]:
                    return False

            # MACD checks
            macd = getattr(indicators, "macd", None)
            if macd is not None:
                if "macd_above_zero" in trigger and trigger["macd_above_zero"] and macd <= 0:
                    return False
                if "macd_below_zero" in trigger and trigger["macd_below_zero"] and macd >= 0:
                    return False

            # Bollinger Band checks
            bb_lower = getattr(indicators, "bb_lower", None)
            bb_upper = getattr(indicators, "bb_upper", None)
            close = getattr(indicators, "close", None)
            if close is not None:
                if "below_bb_lower" in trigger and bb_lower is not None:
                    if not (close < bb_lower):
                        return False
                if "above_bb_upper" in trigger and bb_upper is not None:
                    if not (close > bb_upper):
                        return False

            # ATR check
            atr = getattr(indicators, "atr", None)
            if "atr_above" in trigger and atr is not None:
                if atr < trigger["atr_above"]:
                    return False

        return True
