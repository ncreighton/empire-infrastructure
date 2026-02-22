"""CodexAdvisor — bridges PracticeCodex history into generation logic.

Reads the practitioner's logged practice sessions, correspondences used,
effectiveness ratings, and moon patterns from PracticeCodex. Provides
methods that SpellSmith, MysticEnhancer, and GrimoireEngine use to
personalize output:

  - Preferred herbs/crystals (highest-rated for a given intention)
  - Discovery candidates (knowledge-base items the user has never tried)
  - Auto-difficulty scaling based on session count
  - Best moon phase based on personal effectiveness data
  - One-call aggregation of all personalization context

All methods gracefully return empty/default values when the Codex has
no data, so the system behaves identically for brand-new users.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grimoire.forge.practice_codex import PracticeCodex


class CodexAdvisor:
    """Bridges PracticeCodex user history into generation logic."""

    def __init__(self, codex: PracticeCodex) -> None:
        self.codex = codex

    # ── Preferred materials ──────────────────────────────────────────────

    def get_preferred_herbs(self, intention: str = "", limit: int = 5) -> list[str]:
        """Return the user's top-rated herbs for *intention*.

        Queries ``correspondences_used`` filtered by type='herb' and
        optionally by intention, ordered by average effectiveness then
        usage count.

        Returns an empty list if no data exists.
        """
        return self._get_preferred("herb", intention, limit)

    def get_preferred_crystals(self, intention: str = "", limit: int = 5) -> list[str]:
        """Return the user's top-rated crystals for *intention*."""
        return self._get_preferred("crystal", intention, limit)

    def _get_preferred(
        self, corr_type: str, intention: str, limit: int
    ) -> list[str]:
        """Generic preferred-correspondence query."""
        try:
            with self.codex._connect() as conn:
                if intention:
                    rows = conn.execute(
                        """
                        SELECT correspondence_name AS name,
                               COUNT(*) AS cnt,
                               ROUND(AVG(CASE WHEN effectiveness_rating > 0
                                         THEN effectiveness_rating END), 1) AS avg_eff
                        FROM correspondences_used
                        WHERE correspondence_type = ?
                          AND LOWER(intention) LIKE ?
                        GROUP BY correspondence_name
                        ORDER BY avg_eff DESC, cnt DESC
                        LIMIT ?
                        """,
                        (corr_type, f"%{intention.lower()}%", limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT correspondence_name AS name,
                               COUNT(*) AS cnt,
                               ROUND(AVG(CASE WHEN effectiveness_rating > 0
                                         THEN effectiveness_rating END), 1) AS avg_eff
                        FROM correspondences_used
                        WHERE correspondence_type = ?
                        GROUP BY correspondence_name
                        ORDER BY avg_eff DESC, cnt DESC
                        LIMIT ?
                        """,
                        (corr_type, limit),
                    ).fetchall()
                return [r["name"] for r in rows]
        except Exception:
            return []

    # ── Discovery candidates ─────────────────────────────────────────────

    def get_discovery_candidates(
        self, intention: str = "", category: str = "herb", limit: int = 3
    ) -> list[str]:
        """Return knowledge-base items the user has NEVER used.

        Compares the full herb/crystal knowledge base against the user's
        ``correspondences_used`` history and returns items that have never
        appeared, optionally filtered to those relevant to *intention*.
        """
        try:
            # Get all items user has used of this type
            with self.codex._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT DISTINCT LOWER(correspondence_name) AS name
                    FROM correspondences_used
                    WHERE correspondence_type = ?
                    """,
                    (category,),
                ).fetchall()
            used_names = {r["name"] for r in rows}

            # Get full knowledge base
            if category == "herb":
                from grimoire.knowledge.correspondences import HERBS
                all_items = HERBS
            elif category == "crystal":
                from grimoire.knowledge.correspondences import CRYSTALS
                all_items = CRYSTALS
            else:
                return []

            # Filter to unused items
            candidates = []
            for name, data in all_items.items():
                if name.lower() not in used_names:
                    # If intention specified, only include if relevant
                    if intention:
                        props = data.get("magical_properties", [])
                        intentions = data.get("intentions", [])
                        all_text = " ".join(props + intentions).lower()
                        if intention.lower() not in all_text:
                            continue
                    candidates.append(name)

            # Return a subset
            return candidates[:limit]
        except Exception:
            return []

    # ── Auto-difficulty ──────────────────────────────────────────────────

    def get_auto_difficulty(self) -> str:
        """Determine difficulty level based on total session count.

        - 0-10 sessions: beginner
        - 11-50 sessions: intermediate
        - 51+ sessions: advanced
        """
        try:
            total = self.codex.get_total_sessions()
            if total <= 10:
                return "beginner"
            elif total <= 50:
                return "intermediate"
            else:
                return "advanced"
        except Exception:
            return "beginner"

    # ── Best moon phase ──────────────────────────────────────────────────

    def get_best_moon_phase(self, intention: str = "") -> str | None:
        """Return the moon phase where the user has highest effectiveness.

        Only returns a result if there are at least 3 sessions with
        effectiveness ratings. Returns None otherwise.
        """
        try:
            with self.codex._connect() as conn:
                if intention:
                    rows = conn.execute(
                        """
                        SELECT moon_phase,
                               ROUND(AVG(effectiveness_rating), 1) AS avg_eff,
                               COUNT(*) AS cnt
                        FROM practice_log
                        WHERE effectiveness_rating > 0
                          AND moon_phase IS NOT NULL AND moon_phase != ''
                          AND LOWER(intention) LIKE ?
                        GROUP BY moon_phase
                        HAVING cnt >= 2
                        ORDER BY avg_eff DESC
                        LIMIT 1
                        """,
                        (f"%{intention.lower()}%",),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT moon_phase,
                               ROUND(AVG(effectiveness_rating), 1) AS avg_eff,
                               COUNT(*) AS cnt
                        FROM practice_log
                        WHERE effectiveness_rating > 0
                          AND moon_phase IS NOT NULL AND moon_phase != ''
                        GROUP BY moon_phase
                        HAVING cnt >= 3
                        ORDER BY avg_eff DESC
                        LIMIT 1
                        """,
                    ).fetchall()
                if rows:
                    return rows[0]["moon_phase"]
                return None
        except Exception:
            return None

    # ── One-call aggregation ─────────────────────────────────────────────

    def get_personalization_context(self, intention: str = "") -> dict:
        """Return all personalization data in a single call.

        Returns a dict with keys:
          - session_count: total sessions logged
          - streak: current consecutive-day streak
          - auto_difficulty: beginner/intermediate/advanced
          - preferred_herbs: top herbs for this intention
          - preferred_crystals: top crystals for this intention
          - discovery_herb: one herb the user hasn't tried
          - discovery_crystal: one crystal the user hasn't tried
          - best_moon_phase: most effective moon phase (or None)
          - top_method: most-used practice type
          - favorite_correspondences: overall top 5 correspondences

        All fields gracefully default to empty values when the Codex
        has no data.
        """
        try:
            session_count = self.codex.get_total_sessions()
        except Exception:
            session_count = 0

        try:
            streak = self.codex.get_practice_streak()
        except Exception:
            streak = 0

        preferred_herbs = self.get_preferred_herbs(intention)
        preferred_crystals = self.get_preferred_crystals(intention)

        discovery_herbs = self.get_discovery_candidates(intention, "herb", 1)
        discovery_crystals = self.get_discovery_candidates(intention, "crystal", 1)

        best_phase = self.get_best_moon_phase(intention)

        try:
            by_type = self.codex.get_sessions_by_type()
            top_method = list(by_type.keys())[0] if by_type else ""
        except Exception:
            top_method = ""

        try:
            fav_raw = self.codex.get_favorite_correspondences(limit=5)
            favorite_correspondences = [c["name"] for c in fav_raw]
        except Exception:
            favorite_correspondences = []

        return {
            "session_count": session_count,
            "streak": streak,
            "auto_difficulty": self.get_auto_difficulty(),
            "preferred_herbs": preferred_herbs,
            "preferred_crystals": preferred_crystals,
            "discovery_herb": discovery_herbs[0] if discovery_herbs else "",
            "discovery_crystal": discovery_crystals[0] if discovery_crystals else "",
            "best_moon_phase": best_phase,
            "top_method": top_method,
            "favorite_correspondences": favorite_correspondences,
        }
