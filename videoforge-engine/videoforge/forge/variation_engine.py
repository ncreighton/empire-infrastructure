"""VariationEngine — SQLite-backed anti-repetition pool selector.

Adapted from grimoire-intelligence for video content generation.
Ensures hooks, transitions, music, visual styles don't repeat too often.
"""

import os
import random
import sqlite3
import time
from pathlib import Path

_RECENCY_SCHEMA = """
CREATE TABLE IF NOT EXISTS recency_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    pool_name TEXT NOT NULL,
    selected_value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pool_ts ON recency_log(pool_name, timestamp);
"""


class VariationEngine:
    """Weighted random selection with recency-based anti-repetition."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "codex.db")
        if db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_RECENCY_SCHEMA)
        self._conn.commit()

    def pick(self, pool_name: str, variants: list) -> str:
        """Pick one item from variants with recency-weighted randomness.

        Weight tiers:
        - Used < 3 days ago: 0.1 (strongly avoid)
        - Used 4-14 days ago: 0.5 (mild avoidance)
        - Used 15-30 days ago: 1.0 (neutral)
        - Never used / > 30 days: 2.0 (discovery bias)
        """
        if not variants:
            return ""
        if len(variants) == 1:
            self._log_selection(pool_name, variants[0])
            return variants[0]

        weights = self._get_recency_weights(pool_name, variants)
        selected = random.choices(variants, weights=weights, k=1)[0]
        self._log_selection(pool_name, selected)
        return selected

    def pick_n(self, pool_name: str, variants: list, n: int) -> list:
        """Pick n unique items with recency weighting."""
        if not variants:
            return []
        n = min(n, len(variants))
        remaining = list(variants)
        results = []
        for _ in range(n):
            if not remaining:
                break
            weights = self._get_recency_weights(pool_name, remaining)
            selected = random.choices(remaining, weights=weights, k=1)[0]
            results.append(selected)
            remaining.remove(selected)
            self._log_selection(pool_name, selected)
        return results

    def _log_selection(self, pool_name: str, value: str):
        self._conn.execute(
            "INSERT INTO recency_log (timestamp, pool_name, selected_value) VALUES (?, ?, ?)",
            (time.time(), pool_name, str(value)),
        )
        self._conn.commit()

    def _get_recency_weights(self, pool_name: str, variants: list) -> list:
        now = time.time()
        cutoff_30d = now - (30 * 86400)

        rows = self._conn.execute(
            "SELECT selected_value, MAX(timestamp) as last_used "
            "FROM recency_log WHERE pool_name = ? AND timestamp > ? "
            "GROUP BY selected_value",
            (pool_name, cutoff_30d),
        ).fetchall()

        last_used = {row[0]: row[1] for row in rows}
        weights = []

        for v in variants:
            vs = str(v)
            if vs not in last_used:
                weights.append(2.0)  # Discovery bias
            else:
                age_days = (now - last_used[vs]) / 86400
                if age_days < 3:
                    weights.append(0.1)
                elif age_days < 14:
                    weights.append(0.5)
                else:
                    weights.append(1.0)

        return weights

    def close(self):
        self._conn.close()


# ── Video-specific text pools ────────────────────────────────────────

HOOK_OPENING_POOLS = {
    "pattern_interrupt": [
        "STOP scrolling right now.",
        "Wait — you need to see this.",
        "Hold on. This changes everything.",
        "Don't skip this.",
        "POV: You're about to learn something insane.",
    ],
    "curiosity_gap": [
        "Nobody talks about this, but...",
        "There's a secret most people don't know...",
        "I found something that will blow your mind...",
        "What if I told you everything you know is wrong?",
        "You're not going to believe what I discovered...",
    ],
    "story_hook": [
        "Let me tell you a story...",
        "This is the story they don't want you to hear.",
        "Thousands of years ago, something incredible happened.",
        "Picture this...",
        "Once upon a time, this was forbidden knowledge.",
    ],
    "list_authority": [
        "Here are the top {n} things you need to know.",
        "{N} secrets that will change how you see {topic}.",
        "The only {n} {things} that actually matter.",
        "I tested everything. These {n} are the BEST.",
    ],
    "contrarian": [
        "This is going to be controversial, but...",
        "Everyone gets this wrong. Here's the truth.",
        "Unpopular opinion incoming...",
        "The experts are lying to you about {topic}.",
    ],
}

CTA_POOLS = {
    "follow": [
        "Follow for more {niche} content.",
        "Hit follow so you don't miss the next one.",
        "Follow for daily {niche} tips.",
        "I post {niche} content every day — follow!",
    ],
    "engagement": [
        "Comment your thoughts below!",
        "Save this for later — you'll need it.",
        "Share this with someone who needs to see it.",
        "Drop a 🔥 if you learned something new.",
        "Which one was your favorite? Comment!",
    ],
    "subscribe": [
        "Subscribe for more content like this.",
        "Hit subscribe and the bell — new videos weekly.",
        "Don't forget to subscribe!",
    ],
}

TRANSITION_PHRASE_POOLS = [
    "But here's where it gets interesting...",
    "Now watch this...",
    "And the best part?",
    "But that's not all...",
    "Here's the thing nobody mentions...",
    "Let me show you something...",
    "This is where it gets crazy...",
    "Pay attention to this next part...",
]

THUMBNAIL_CONCEPT_POOLS = {
    "witchcraft": [
        "Glowing candles in dark room with mysterious text",
        "Crystal collection with moonlight, title overlay",
        "Hands over cauldron with colorful smoke, bold text",
        "Moon phases with mystical symbols, neon title",
    ],
    "mythology": [
        "Epic god/goddess portrait with gold accents",
        "Ancient temple with dramatic lighting, bold title",
        "Mythical creature silhouette against sunset",
        "Warrior statue with epic text overlay",
    ],
    "tech": [
        "Clean device photo with star rating, bold title",
        "Split screen comparison with VS graphic",
        "Glowing product on dark background, price tag",
        "Hand holding device with reaction emoji",
    ],
    "ai_news": [
        "AI brain graphic with breaking news banner",
        "Robot/AI visual with shocking headline text",
        "Before/after AI comparison, bold stats",
        "Futuristic interface with urgent text overlay",
    ],
    "lifestyle": [
        "Bright, warm photo with clean text overlay",
        "Cozy setup with numbered list preview",
        "Hands doing activity with step indicator",
        "Before/after transformation, split frame",
    ],
}
