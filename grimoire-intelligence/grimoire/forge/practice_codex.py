"""
PracticeCodex -- SQLite Learning Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Part of the FORGE intelligence layer for the Grimoire Intelligence System.

Every practice session, tarot reading, moon journal entry, and intention
is recorded in a local SQLite database. Over time the Codex learns what
correspondences work best for the practitioner, which moon phases feel
most powerful, and where the journey wants to grow next.

The voice here is quiet and supportive -- this is a practitioner's
private grimoire, and every entry matters.

Follows the Codex pattern from VelvetVeil's forge_pdf_engine.py:
SQLite-based learning from every interaction.
"""

from __future__ import annotations

import sqlite3
import json
import datetime
from pathlib import Path

from grimoire.models import PracticeEntry, TarotReading, JourneyInsight


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS practice_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    practice_type TEXT NOT NULL,
    title TEXT NOT NULL,
    intention TEXT,
    category TEXT,
    moon_phase TEXT,
    zodiac_sign TEXT,
    day_ruler TEXT,
    correspondences_used TEXT,
    notes TEXT,
    mood_before TEXT,
    mood_after TEXT,
    effectiveness_rating INTEGER,
    duration_minutes INTEGER
);

CREATE TABLE IF NOT EXISTS moon_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    moon_phase TEXT NOT NULL,
    zodiac_sign TEXT,
    entry TEXT NOT NULL,
    mood TEXT,
    dreams TEXT,
    energy_level INTEGER
);

CREATE TABLE IF NOT EXISTS tarot_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    spread_name TEXT NOT NULL,
    question TEXT,
    cards TEXT NOT NULL,
    interpretation TEXT,
    follow_up_actions TEXT,
    accuracy_rating INTEGER
);

CREATE TABLE IF NOT EXISTS correspondences_used (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    practice_id INTEGER,
    correspondence_type TEXT,
    correspondence_name TEXT,
    intention TEXT,
    effectiveness_rating INTEGER,
    FOREIGN KEY (practice_id) REFERENCES practice_log(id)
);

CREATE TABLE IF NOT EXISTS intention_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    intention TEXT NOT NULL,
    category TEXT,
    method TEXT,
    correspondences TEXT,
    moon_phase TEXT,
    outcome TEXT,
    outcome_date TEXT,
    notes TEXT,
    days_to_manifest INTEGER
);

CREATE TABLE IF NOT EXISTS sabbat_celebrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    sabbat TEXT NOT NULL,
    year INTEGER,
    activities TEXT,
    altar_description TEXT,
    offerings TEXT,
    reflections TEXT,
    photos_path TEXT
);

CREATE TABLE IF NOT EXISTS growth_milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    milestone_type TEXT,
    description TEXT,
    celebration_note TEXT
);

CREATE TABLE IF NOT EXISTS prompt_evolution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query_type TEXT,
    original_query TEXT,
    enhanced_query TEXT,
    score_before REAL,
    score_after REAL,
    improvement REAL
);
"""

# ── Milestone definitions ─────────────────────────────────────────────────────

_FIRST_TYPE_MILESTONES: dict[str, str] = {
    "spell": "first_spell",
    "ritual": "first_ritual",
    "meditation": "first_meditation",
    "divination": "first_divination",
    "journaling": "first_journal",
}

_STREAK_MILESTONES: dict[int, str] = {
    7: "streak_7",
    14: "streak_14",
    30: "streak_30",
    100: "streak_100",
}

_SESSION_MILESTONES: dict[int, str] = {
    10: "sessions_10",
    50: "sessions_50",
    100: "sessions_100",
    500: "sessions_500",
}

_ALL_PRACTICE_TYPES = {"spell", "ritual", "meditation", "divination", "journaling"}

_ALL_SABBATS = {
    "samhain", "yule", "imbolc", "ostara",
    "beltane", "litha", "lughnasadh", "mabon",
}


# ===========================================================================
# PracticeCodex
# ===========================================================================

class PracticeCodex:
    """SQLite-backed learning engine that grows with the practitioner.

    Every practice session, tarot reading, moon observation, intention,
    and sabbat celebration is logged and analyzed. The Codex surfaces
    patterns -- which herbs and crystals serve you best, which moon
    phases feel most powerful, where your journey is thriving, and where
    it wants to stretch.

    Usage::

        codex = PracticeCodex()
        entry = PracticeEntry(
            practice_type="spell",
            title="Protection Jar",
            intention="Ward my home",
            correspondences_used=["rosemary", "black tourmaline", "salt"],
            effectiveness_rating=4,
        )
        practice_id = codex.log_practice(entry)
        summary = codex.get_growth_summary()
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(
                Path(__file__).resolve().parent.parent / "data" / "grimoire.db"
            )

        self.db_path = db_path
        self._is_memory = db_path == ":memory:"
        self._shared_conn: sqlite3.Connection | None = None

        # Ensure the parent directory exists (skip for in-memory).
        if not self._is_memory:
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    # ------------------------------------------------------------------
    # Database bootstrap
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create all tables if they do not already exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """Return a connection with row_factory set to sqlite3.Row.

        For file-backed databases each call opens a new connection.
        For ``:memory:`` databases a single persistent connection is
        reused so that tables survive across calls.
        """
        if self._is_memory:
            if self._shared_conn is None:
                self._shared_conn = sqlite3.connect(":memory:")
                self._shared_conn.row_factory = sqlite3.Row
            return self._shared_conn

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        """ISO-8601 timestamp for the current moment."""
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    # ==================================================================
    # Logging Methods
    # ==================================================================

    def log_practice(self, entry: PracticeEntry) -> int:
        """Record a practice session and check for milestone triggers.

        Inserts into ``practice_log`` and creates a row in
        ``correspondences_used`` for each correspondence the practitioner
        listed. Then checks whether this session triggers any growth
        milestones (first of type, streak thresholds, session counts).

        Args:
            entry: A populated PracticeEntry dataclass.

        Returns:
            The ``practice_log.id`` of the newly inserted row.
        """
        ts = entry.date if entry.date else self._now()
        correspondences_json = json.dumps(entry.correspondences_used)

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO practice_log
                    (timestamp, practice_type, title, intention, category,
                     moon_phase, zodiac_sign, day_ruler,
                     correspondences_used, notes, mood_before, mood_after,
                     effectiveness_rating, duration_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    entry.practice_type,
                    entry.title,
                    entry.intention,
                    "",  # category -- to be filled by caller or analysis
                    entry.moon_phase,
                    entry.zodiac_sign,
                    "",  # day_ruler -- optional enrichment
                    correspondences_json,
                    entry.notes,
                    entry.mood_before,
                    entry.mood_after,
                    entry.effectiveness_rating,
                    0,  # duration_minutes -- not on PracticeEntry currently
                ),
            )
            practice_id = cursor.lastrowid

            # Log individual correspondences.
            for corr_name in entry.correspondences_used:
                conn.execute(
                    """
                    INSERT INTO correspondences_used
                        (timestamp, practice_id, correspondence_type,
                         correspondence_name, intention, effectiveness_rating)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        practice_id,
                        self._guess_correspondence_type(corr_name),
                        corr_name,
                        entry.intention,
                        entry.effectiveness_rating,
                    ),
                )

            conn.commit()

        # Check milestones (outside the main transaction for clarity).
        self._check_milestones(entry.practice_type)

        return practice_id

    def log_moon_journal(
        self,
        moon_phase: str,
        entry: str,
        mood: str = "",
        dreams: str = "",
        energy_level: int = 0,
        zodiac_sign: str = "",
    ) -> int:
        """Record a moon journal entry.

        Args:
            moon_phase: Current moon phase name.
            entry: The practitioner's journal text.
            mood: Optional mood descriptor.
            dreams: Optional dream notes.
            energy_level: 1-10 energy scale, 0 if not rated.
            zodiac_sign: Optional current zodiac transit.

        Returns:
            The ``moon_journal.id`` of the new row.
        """
        ts = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO moon_journal
                    (timestamp, moon_phase, zodiac_sign, entry,
                     mood, dreams, energy_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, moon_phase, zodiac_sign, entry, mood, dreams, energy_level),
            )
            conn.commit()
            return cursor.lastrowid

    def log_tarot_reading(self, reading: TarotReading) -> int:
        """Record a tarot reading.

        Args:
            reading: A populated TarotReading dataclass.

        Returns:
            The ``tarot_log.id`` of the new row.
        """
        ts = reading.date if reading.date else self._now()
        cards_json = json.dumps(reading.cards)
        actions_json = json.dumps(reading.follow_up_actions)

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tarot_log
                    (timestamp, spread_name, question, cards,
                     interpretation, follow_up_actions, accuracy_rating)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    reading.spread_name,
                    reading.question,
                    cards_json,
                    reading.interpretation,
                    actions_json,
                    None,  # accuracy_rating added later
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def log_intention(
        self,
        intention: str,
        category: str,
        method: str,
        correspondences: list[str],
        moon_phase: str = "",
    ) -> int:
        """Create a new intention with outcome='pending'.

        Args:
            intention: What the practitioner intends to manifest.
            category: Intention category (protection, love, etc.).
            method: Spell or ritual type used.
            correspondences: List of correspondence names used.
            moon_phase: Moon phase when the intention was set.

        Returns:
            The ``intention_outcomes.id`` of the new row.
        """
        ts = self._now()
        corr_json = json.dumps(correspondences)

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO intention_outcomes
                    (timestamp, intention, category, method,
                     correspondences, moon_phase, outcome,
                     outcome_date, notes, days_to_manifest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, intention, category, method,
                    corr_json, moon_phase, "pending",
                    None, "", None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def update_intention_outcome(
        self,
        intention_id: int,
        outcome: str,
        notes: str = "",
    ) -> None:
        """Update an intention's outcome and calculate manifestation time.

        Valid outcomes: manifested, partial, redirected, released.

        Args:
            intention_id: The ``intention_outcomes.id`` to update.
            outcome: One of the valid outcome strings.
            notes: Optional practitioner notes about the outcome.
        """
        now = self._now()

        with self._connect() as conn:
            # Fetch the original timestamp to compute days_to_manifest.
            row = conn.execute(
                "SELECT timestamp FROM intention_outcomes WHERE id = ?",
                (intention_id,),
            ).fetchone()

            days_to_manifest = None
            if row:
                try:
                    created = datetime.datetime.fromisoformat(row["timestamp"])
                    resolved = datetime.datetime.fromisoformat(now)
                    days_to_manifest = (resolved - created).days
                except (ValueError, TypeError):
                    days_to_manifest = None

            conn.execute(
                """
                UPDATE intention_outcomes
                SET outcome = ?,
                    outcome_date = ?,
                    notes = ?,
                    days_to_manifest = ?
                WHERE id = ?
                """,
                (outcome, now, notes, days_to_manifest, intention_id),
            )
            conn.commit()

    def log_sabbat(
        self,
        sabbat: str,
        activities: list[str],
        reflections: str = "",
        altar_description: str = "",
        offerings: str = "",
    ) -> int:
        """Record a sabbat celebration.

        Args:
            sabbat: Sabbat name (samhain, yule, etc.).
            activities: List of things done during the celebration.
            reflections: Personal reflections on the celebration.
            altar_description: Description of altar setup.
            offerings: Offerings made.

        Returns:
            The ``sabbat_celebrations.id`` of the new row.
        """
        ts = self._now()
        now = datetime.datetime.now(datetime.timezone.utc)
        activities_json = json.dumps(activities)

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sabbat_celebrations
                    (timestamp, sabbat, year, activities,
                     altar_description, offerings, reflections, photos_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    sabbat.lower().strip(),
                    now.year,
                    activities_json,
                    altar_description,
                    offerings,
                    reflections,
                    None,
                ),
            )
            conn.commit()

            # Check if all sabbats have been celebrated.
            self._check_all_sabbats_milestone()

            return cursor.lastrowid

    def log_prompt_evolution(
        self,
        query_type: str,
        original: str,
        enhanced: str,
        score_before: float,
        score_after: float,
    ) -> int:
        """Record a prompt enhancement for learning.

        Args:
            query_type: The type of query (spell_request, etc.).
            original: The original user query.
            enhanced: The enhanced query after processing.
            score_before: Quality score of the original.
            score_after: Quality score of the enhanced version.

        Returns:
            The ``prompt_evolution.id`` of the new row.
        """
        ts = self._now()
        improvement = score_after - score_before

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO prompt_evolution
                    (timestamp, query_type, original_query, enhanced_query,
                     score_before, score_after, improvement)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, query_type, original, enhanced,
                 score_before, score_after, improvement),
            )
            conn.commit()
            return cursor.lastrowid

    # ==================================================================
    # Analytics Methods
    # ==================================================================

    def get_favorite_correspondences(self, limit: int = 10) -> list[dict]:
        """Return the most-used correspondences ranked by usage count.

        Each entry includes the correspondence name, type, usage count,
        and average effectiveness rating across all uses.

        Args:
            limit: Maximum number of results to return.

        Returns:
            A list of dicts with keys: name, type, count, avg_effectiveness.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    correspondence_name AS name,
                    correspondence_type AS type,
                    COUNT(*) AS count,
                    ROUND(AVG(CASE WHEN effectiveness_rating > 0
                              THEN effectiveness_rating END), 1) AS avg_effectiveness
                FROM correspondences_used
                GROUP BY correspondence_name
                ORDER BY count DESC, avg_effectiveness DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_most_effective_methods(self, limit: int = 5) -> list[dict]:
        """Return practice types with the highest average effectiveness.

        Args:
            limit: Maximum number of results to return.

        Returns:
            A list of dicts with keys: practice_type, avg_effectiveness,
            session_count.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    practice_type,
                    ROUND(AVG(CASE WHEN effectiveness_rating > 0
                              THEN effectiveness_rating END), 1) AS avg_effectiveness,
                    COUNT(*) AS session_count
                FROM practice_log
                WHERE effectiveness_rating > 0
                GROUP BY practice_type
                ORDER BY avg_effectiveness DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_practice_streak(self) -> int:
        """Return the current consecutive-day practice streak.

        Counts backward from today. Each calendar day (UTC) with at
        least one ``practice_log`` entry extends the streak by one.

        Returns:
            Number of consecutive days, starting from today or yesterday.
        """
        return self._compute_streak(current=True)

    def get_longest_streak(self) -> int:
        """Return the longest consecutive-day practice streak ever.

        Returns:
            The maximum streak length across all history.
        """
        return self._compute_streak(current=False)

    def get_total_sessions(self) -> int:
        """Return the total number of practice log entries."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM practice_log").fetchone()
            return row["c"] if row else 0

    def get_sessions_by_type(self) -> dict[str, int]:
        """Return session counts broken down by practice_type.

        Returns:
            A dict mapping practice_type strings to their counts.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT practice_type, COUNT(*) AS c
                FROM practice_log
                GROUP BY practice_type
                ORDER BY c DESC
                """
            ).fetchall()
            return {r["practice_type"]: r["c"] for r in rows}

    def get_growth_summary(self) -> JourneyInsight:
        """Build a comprehensive journey summary.

        Gathers total sessions, streak info, favorite correspondences,
        most effective methods, moon-phase patterns, milestones, the
        next upcoming sabbat, and personalized recommendations.

        Returns:
            A fully populated JourneyInsight dataclass.
        """
        total = self.get_total_sessions()
        streak = self.get_practice_streak()
        longest = self.get_longest_streak()

        # Favorite practice types (top 3).
        by_type = self.get_sessions_by_type()
        favorite_types = list(by_type.keys())[:3]

        # Favorite correspondences (top 5 names).
        fav_corr_raw = self.get_favorite_correspondences(limit=5)
        favorite_correspondences = [c["name"] for c in fav_corr_raw]

        # Most effective methods (top 3 type names).
        eff_raw = self.get_most_effective_methods(limit=3)
        most_effective = [
            f"{m['practice_type']} ({m['avg_effectiveness']}/5)"
            for m in eff_raw
        ]

        # Moon patterns.
        moon_patterns = self.get_moon_patterns()

        # Milestones.
        milestones = self._get_all_milestones()

        # Next sabbat.
        next_sabbat_name, days_until = self._next_sabbat_info()

        # Recommendations.
        recommendations = self.get_personalized_recommendations(limit=5)

        return JourneyInsight(
            total_sessions=total,
            practice_streak=streak,
            longest_streak=longest,
            favorite_types=favorite_types,
            favorite_correspondences=favorite_correspondences,
            most_effective_methods=most_effective,
            moon_patterns=moon_patterns.get("activity_by_phase", {}),
            growth_milestones=milestones,
            recommendations=recommendations,
            next_sabbat=next_sabbat_name,
            days_until_sabbat=days_until,
        )

    def get_personalized_recommendations(self, limit: int = 5) -> list[str]:
        """Generate practice recommendations based on history.

        Examines what the practitioner has and has not tried, looks at
        patterns, and offers gentle, encouraging suggestions.

        Args:
            limit: Maximum number of recommendations.

        Returns:
            A list of recommendation strings.
        """
        recs: list[str] = []
        by_type = self.get_sessions_by_type()
        total = self.get_total_sessions()
        tried_types = set(by_type.keys())

        # Suggest untried practice types.
        untried = _ALL_PRACTICE_TYPES - tried_types
        type_suggestions = {
            "meditation": (
                "You have not logged a meditation session yet. Even five "
                "minutes of stillness can deepen your connection to your "
                "craft. Consider a simple grounding meditation to start."
            ),
            "divination": (
                "Divination has not appeared in your practice log yet. "
                "Drawing a single tarot card each morning is a gentle way "
                "to build your intuitive muscles."
            ),
            "spell": (
                "You have not logged any spellwork yet. A simple candle "
                "spell is a beautiful first step -- choose a color that "
                "matches your intention and let the flame carry your will."
            ),
            "ritual": (
                "A ritual has not been logged yet. Rituals can be as "
                "simple as lighting a candle and speaking your intention "
                "aloud under the moonlight."
            ),
            "journaling": (
                "Journaling has not appeared in your practice log. Writing "
                "even a few sentences after each session helps you notice "
                "patterns and honor your growth."
            ),
        }
        for ptype in untried:
            if ptype in type_suggestions and len(recs) < limit:
                recs.append(type_suggestions[ptype])

        # Suggest branching out with correspondences.
        fav_corr = self.get_favorite_correspondences(limit=3)
        if len(fav_corr) >= 3:
            top_names = [c["name"] for c in fav_corr[:3]]
            recs.append(
                f"You have been working frequently with {', '.join(top_names)}. "
                f"Consider branching out -- try a new herb, crystal, or color "
                f"to discover fresh resonances in your practice."
            )

        # Check tarot readings.
        with self._connect() as conn:
            tarot_count = conn.execute(
                "SELECT COUNT(*) AS c FROM tarot_log"
            ).fetchone()["c"]
        if tarot_count == 0 and "divination" not in untried and len(recs) < limit:
            recs.append(
                "You have not logged any tarot readings yet. Recording your "
                "spreads and returning to them later reveals how your intuition "
                "grows over time."
            )

        # Check sabbat celebrations.
        celebrated = self._get_celebrated_sabbats()
        missing_sabbats = _ALL_SABBATS - celebrated
        if missing_sabbats and len(recs) < limit:
            next_name, days = self._next_sabbat_info()
            if next_name.lower() in missing_sabbats:
                recs.append(
                    f"{next_name} is coming up in {days} days. Planning a "
                    f"small celebration -- even just setting an intention "
                    f"and lighting a candle -- connects you to the Wheel "
                    f"of the Year."
                )
            elif missing_sabbats:
                sample = next(iter(missing_sabbats)).title()
                recs.append(
                    f"You have not celebrated {sample} yet. Marking the "
                    f"sabbats, even simply, weaves your practice into the "
                    f"rhythm of the seasons."
                )

        # Check for low-effectiveness methods.
        low_eff = self._get_low_effectiveness_methods()
        for method_info in low_eff:
            if len(recs) >= limit:
                break
            ptype = method_info["practice_type"]
            avg = method_info["avg_effectiveness"]
            if avg and avg < 3.0:
                recs.append(
                    f"Your {ptype} sessions average {avg}/5 effectiveness. "
                    f"This could mean the approach is not resonating yet. "
                    f"Try adjusting your timing, adding new correspondences, "
                    f"or simplifying the practice to find what clicks."
                )

        # Gentle encouragement if long gap since last practice.
        last_date = self._get_last_practice_date()
        if last_date and len(recs) < limit:
            try:
                last_dt = datetime.datetime.fromisoformat(last_date)
                now_dt = datetime.datetime.now(datetime.timezone.utc)
                gap_days = (now_dt - last_dt).days
                if gap_days >= 7:
                    recs.append(
                        f"It has been {gap_days} days since your last logged "
                        f"practice. There is no judgment here -- life has its "
                        f"seasons. When you are ready, even a one-minute "
                        f"grounding exercise counts. Your grimoire is always "
                        f"here for you."
                    )
            except (ValueError, TypeError):
                pass

        # If no practice at all, welcome them.
        if total == 0 and not recs:
            recs.append(
                "Welcome to your Grimoire. You have not logged any practice "
                "sessions yet. Start with whatever calls to you -- a candle, "
                "a crystal, a quiet moment under the moon. Every journey "
                "begins with a single step."
            )

        return recs[:limit]

    def get_moon_patterns(self) -> dict:
        """Analyze which moon phases the practitioner is most active during.

        Returns a dict with:
          - activity_by_phase: {phase_name: session_count}
          - most_active_phase: the phase with the most sessions
          - most_effective_phase: the phase with highest avg effectiveness
        """
        with self._connect() as conn:
            # Activity by moon phase.
            activity_rows = conn.execute(
                """
                SELECT moon_phase, COUNT(*) AS c
                FROM practice_log
                WHERE moon_phase IS NOT NULL AND moon_phase != ''
                GROUP BY moon_phase
                ORDER BY c DESC
                """
            ).fetchall()

            activity_by_phase = {r["moon_phase"]: r["c"] for r in activity_rows}
            most_active = activity_rows[0]["moon_phase"] if activity_rows else ""

            # Effectiveness by moon phase.
            eff_rows = conn.execute(
                """
                SELECT moon_phase,
                       ROUND(AVG(effectiveness_rating), 1) AS avg_eff
                FROM practice_log
                WHERE moon_phase IS NOT NULL
                  AND moon_phase != ''
                  AND effectiveness_rating > 0
                GROUP BY moon_phase
                ORDER BY avg_eff DESC
                """
            ).fetchall()

            most_effective = eff_rows[0]["moon_phase"] if eff_rows else ""

        return {
            "activity_by_phase": activity_by_phase,
            "most_active_phase": most_active,
            "most_effective_phase": most_effective,
        }

    def get_intention_success_rate(self) -> dict:
        """Return success statistics by intention category.

        Returns a dict with:
          - by_category: {category: {total, manifested, partial, ...}}
          - overall_manifested: total manifested count
          - overall_total: total intention count
          - overall_rate: manifested / total as a percentage
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT category, outcome, COUNT(*) AS c
                FROM intention_outcomes
                GROUP BY category, outcome
                """
            ).fetchall()

        by_category: dict[str, dict[str, int]] = {}
        overall_total = 0
        overall_manifested = 0

        for r in rows:
            cat = r["category"] or "uncategorized"
            outcome = r["outcome"] or "unknown"
            count = r["c"]

            if cat not in by_category:
                by_category[cat] = {"total": 0}

            by_category[cat]["total"] += count
            by_category[cat][outcome] = by_category[cat].get(outcome, 0) + count

            overall_total += count
            if outcome == "manifested":
                overall_manifested += count

        overall_rate = (
            round((overall_manifested / overall_total) * 100, 1)
            if overall_total > 0
            else 0.0
        )

        return {
            "by_category": by_category,
            "overall_manifested": overall_manifested,
            "overall_total": overall_total,
            "overall_rate": overall_rate,
        }

    def get_stats(self) -> dict:
        """Quick overview of the practitioner's grimoire.

        Returns a dict with keys: total_sessions, streak,
        types_breakdown, total_tarot_readings, total_moon_journal_entries,
        total_intentions, manifested_intentions, milestones_count.
        """
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) AS c FROM practice_log"
            ).fetchone()["c"]

            tarot = conn.execute(
                "SELECT COUNT(*) AS c FROM tarot_log"
            ).fetchone()["c"]

            moon = conn.execute(
                "SELECT COUNT(*) AS c FROM moon_journal"
            ).fetchone()["c"]

            intentions_total = conn.execute(
                "SELECT COUNT(*) AS c FROM intention_outcomes"
            ).fetchone()["c"]

            manifested = conn.execute(
                "SELECT COUNT(*) AS c FROM intention_outcomes WHERE outcome = 'manifested'"
            ).fetchone()["c"]

            milestones = conn.execute(
                "SELECT COUNT(*) AS c FROM growth_milestones"
            ).fetchone()["c"]

        return {
            "total_sessions": total,
            "streak": self.get_practice_streak(),
            "types_breakdown": self.get_sessions_by_type(),
            "total_tarot_readings": tarot,
            "total_moon_journal_entries": moon,
            "total_intentions": intentions_total,
            "manifested_intentions": manifested,
            "milestones_count": milestones,
        }

    # ==================================================================
    # Milestone Engine
    # ==================================================================

    def _check_milestones(self, practice_type: str) -> list[str]:
        """Check if the latest log triggers any milestones.

        Milestones checked:
          - First of each practice type (first_spell, first_ritual, etc.)
          - Streak thresholds (7, 14, 30, 100 consecutive days)
          - Total session thresholds (10, 50, 100, 500)
          - All practice types tried
          - Full-moon streak of 3

        Args:
            practice_type: The practice type just logged.

        Returns:
            A list of milestone description strings that were triggered.
        """
        triggered: list[str] = []

        # --- First-of-type milestones ---
        milestone_key = _FIRST_TYPE_MILESTONES.get(practice_type)
        if milestone_key and not self._has_milestone(milestone_key):
            with self._connect() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) AS c FROM practice_log WHERE practice_type = ?",
                    (practice_type,),
                ).fetchone()["c"]
            if count == 1:
                desc = f"First {practice_type} session logged!"
                self._record_milestone(
                    milestone_key,
                    desc,
                    f"You completed your very first {practice_type}. "
                    f"Every master was once a beginner.",
                )
                triggered.append(desc)

        # --- Session count milestones ---
        total = self.get_total_sessions()
        for threshold, m_key in _SESSION_MILESTONES.items():
            if total >= threshold and not self._has_milestone(m_key):
                desc = f"Reached {threshold} total practice sessions!"
                self._record_milestone(
                    m_key,
                    desc,
                    f"You have logged {threshold} sessions. Your dedication "
                    f"is weaving real magic into your life.",
                )
                triggered.append(desc)

        # --- Streak milestones ---
        streak = self.get_practice_streak()
        for threshold, m_key in _STREAK_MILESTONES.items():
            if streak >= threshold and not self._has_milestone(m_key):
                desc = f"Practice streak: {threshold} consecutive days!"
                self._record_milestone(
                    m_key,
                    desc,
                    f"You have practiced for {threshold} days in a row. "
                    f"Consistency is its own kind of spell.",
                )
                triggered.append(desc)

        # --- All types tried ---
        if not self._has_milestone("all_types_tried"):
            by_type = self.get_sessions_by_type()
            if _ALL_PRACTICE_TYPES.issubset(set(by_type.keys())):
                desc = "Explored all five practice types!"
                self._record_milestone(
                    "all_types_tried",
                    desc,
                    "You have tried spell, ritual, meditation, divination, "
                    "and journaling. A well-rounded practitioner indeed.",
                )
                triggered.append(desc)

        # --- Full-moon streak of 3 ---
        if not self._has_milestone("full_moon_streak_3"):
            if self._check_full_moon_streak(3):
                desc = "Practiced during 3 consecutive full moons!"
                self._record_milestone(
                    "full_moon_streak_3",
                    desc,
                    "Three full moons, three sessions. The lunar cycle "
                    "recognizes your devotion.",
                )
                triggered.append(desc)

        return triggered

    def _has_milestone(self, milestone_type: str) -> bool:
        """Check whether a milestone has already been recorded."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM growth_milestones WHERE milestone_type = ?",
                (milestone_type,),
            ).fetchone()
            return row["c"] > 0

    def _record_milestone(
        self, milestone_type: str, description: str, celebration_note: str
    ) -> None:
        """Insert a new milestone into growth_milestones."""
        ts = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO growth_milestones
                    (timestamp, milestone_type, description, celebration_note)
                VALUES (?, ?, ?, ?)
                """,
                (ts, milestone_type, description, celebration_note),
            )
            conn.commit()

    def _get_all_milestones(self) -> list[str]:
        """Return all milestone descriptions, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT description FROM growth_milestones ORDER BY timestamp DESC"
            ).fetchall()
            return [r["description"] for r in rows]

    def _check_all_sabbats_milestone(self) -> None:
        """Record a milestone if all 8 sabbats have been celebrated."""
        if self._has_milestone("all_sabbats_celebrated"):
            return
        celebrated = self._get_celebrated_sabbats()
        if _ALL_SABBATS.issubset(celebrated):
            self._record_milestone(
                "all_sabbats_celebrated",
                "Celebrated all 8 sabbats on the Wheel of the Year!",
                "You have honored every turn of the Wheel. The seasons "
                "know your name.",
            )

    def _check_full_moon_streak(self, required: int) -> bool:
        """Check if there are practice entries during N consecutive full moons.

        This is a simplified check: looks for practice_log entries where
        moon_phase contains 'full' and checks if there are at least
        ``required`` distinct months represented.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT substr(timestamp, 1, 7) AS month
                FROM practice_log
                WHERE LOWER(moon_phase) LIKE '%full%'
                ORDER BY month DESC
                LIMIT ?
                """,
                (required,),
            ).fetchall()

        if len(rows) < required:
            return False

        # Check that the months are consecutive.
        months = [r["month"] for r in rows]
        for i in range(len(months) - 1):
            y1, m1 = map(int, months[i].split("-"))
            y2, m2 = map(int, months[i + 1].split("-"))
            expected_prev_month = m1 - 1 if m1 > 1 else 12
            expected_prev_year = y1 if m1 > 1 else y1 - 1
            if m2 != expected_prev_month or y2 != expected_prev_year:
                return False

        return True

    # ==================================================================
    # Private Helpers
    # ==================================================================

    def _compute_streak(self, current: bool) -> int:
        """Compute practice streak.

        Args:
            current: If True, compute the current streak (from today
                backward). If False, compute the longest streak ever.

        Returns:
            Streak length in days.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT date(timestamp) AS d
                FROM practice_log
                ORDER BY d DESC
                """
            ).fetchall()

        if not rows:
            return 0

        dates = []
        for r in rows:
            try:
                dates.append(datetime.date.fromisoformat(r["d"]))
            except (ValueError, TypeError):
                continue

        if not dates:
            return 0

        if current:
            return self._streak_from_today(dates)
        else:
            return self._longest_streak_in(dates)

    @staticmethod
    def _streak_from_today(dates: list[datetime.date]) -> int:
        """Count consecutive days ending at today (or yesterday).

        The list must be sorted descending (newest first).
        """
        today = datetime.date.today()

        # The streak can start from today or yesterday.
        if dates[0] == today:
            start = today
        elif dates[0] == today - datetime.timedelta(days=1):
            start = dates[0]
        else:
            return 0

        streak = 0
        expected = start
        date_set = set(dates)

        while expected in date_set:
            streak += 1
            expected -= datetime.timedelta(days=1)

        return streak

    @staticmethod
    def _longest_streak_in(dates: list[datetime.date]) -> int:
        """Find the longest consecutive-day run in a descending date list."""
        if not dates:
            return 0

        unique_sorted = sorted(set(dates))
        best = 1
        run = 1

        for i in range(1, len(unique_sorted)):
            if unique_sorted[i] - unique_sorted[i - 1] == datetime.timedelta(days=1):
                run += 1
                best = max(best, run)
            else:
                run = 1

        return best

    def _get_last_practice_date(self) -> str | None:
        """Return the timestamp of the most recent practice log entry."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT timestamp FROM practice_log ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return row["timestamp"] if row else None

    def _get_celebrated_sabbats(self) -> set[str]:
        """Return the set of sabbat names that have been celebrated."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT LOWER(sabbat) AS s FROM sabbat_celebrations"
            ).fetchall()
            return {r["s"] for r in rows}

    def _get_low_effectiveness_methods(self) -> list[dict]:
        """Return practice types with below-average effectiveness."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT practice_type,
                       ROUND(AVG(effectiveness_rating), 1) AS avg_effectiveness
                FROM practice_log
                WHERE effectiveness_rating > 0
                GROUP BY practice_type
                HAVING avg_effectiveness < 3.0
                ORDER BY avg_effectiveness ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def _next_sabbat_info(self) -> tuple[str, int]:
        """Return (sabbat_name, days_until) for the next upcoming sabbat.

        Uses the wheel_of_year module if available, otherwise falls back
        to a simple month-based lookup.
        """
        try:
            from grimoire.knowledge.wheel_of_year import get_next_sabbat
            today = datetime.date.today()
            name, _, days = get_next_sabbat(today.month, today.day)
            return name, days
        except (ImportError, Exception):
            return "the next sabbat", 0

    @staticmethod
    def _guess_correspondence_type(name: str) -> str:
        """Best-effort guess at what type a correspondence is.

        Uses simple keyword matching. This is intentionally imprecise --
        the practitioner's own categorization matters more than ours.
        """
        name_lower = name.lower().strip()

        # Common crystals and stones.
        crystals = {
            "quartz", "amethyst", "tourmaline", "obsidian", "citrine",
            "moonstone", "selenite", "labradorite", "jasper", "agate",
            "carnelian", "malachite", "lapis", "tiger's eye", "rose quartz",
            "fluorite", "garnet", "onyx", "jade", "aventurine", "pyrite",
            "bloodstone", "amber", "jet", "hematite", "sodalite",
            "aquamarine", "sunstone", "peridot", "rhodonite", "turquoise",
        }
        for crystal in crystals:
            if crystal in name_lower:
                return "crystal"

        # Common colors.
        colors = {
            "red", "blue", "green", "yellow", "purple", "black", "white",
            "orange", "pink", "gold", "silver", "brown", "indigo", "violet",
        }
        if name_lower in colors:
            return "color"

        # Elements.
        elements = {"fire", "water", "earth", "air", "spirit"}
        if name_lower in elements:
            return "element"

        # Planets.
        planets = {"sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn"}
        if name_lower in planets:
            return "planet"

        # Default to herb (the most common correspondence type).
        return "herb"
