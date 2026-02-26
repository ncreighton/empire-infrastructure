"""SQLite persistence for Reddit automation state.

Database at data/reddit/reddit_state.db with tables:
- account_state: warmup phase, karma estimate, cumulative stats
- daily_counts: auto-resetting daily action limits
- comment_history: dedup + ratio tracking
- post_history: Etsy link tracking for 10% rule
- subreddit_rules: cached rules per sub
- sessions: session log for pattern analysis
"""

import json
import sqlite3
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger("reddit_state")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "reddit" / "reddit_state.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS account_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_counts (
    date TEXT NOT NULL,
    action TEXT NOT NULL,
    subreddit TEXT DEFAULT '',
    count INTEGER DEFAULT 0,
    PRIMARY KEY (date, action, subreddit)
);

CREATE TABLE IF NOT EXISTS comment_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    subreddit TEXT NOT NULL,
    post_title TEXT DEFAULT '',
    comment_text TEXT NOT NULL,
    is_promo INTEGER DEFAULT 0,
    ngram_hash TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS post_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    body_snippet TEXT DEFAULT '',
    is_promo INTEGER DEFAULT 0,
    etsy_link_included INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS subreddit_rules (
    subreddit TEXT PRIMARY KEY,
    rules_json TEXT DEFAULT '{}',
    last_checked TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_type TEXT NOT NULL,
    duration_seconds REAL DEFAULT 0,
    actions_json TEXT DEFAULT '{}',
    subreddits_visited TEXT DEFAULT '[]'
);
"""


class RedditState:
    """Persistent state manager for Reddit automation."""

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        # Seed default account state
        defaults = {
            "start_date": date.today().isoformat(),
            "karma_estimate": "1",
            "total_comments": "0",
            "total_posts": "0",
            "total_promo_posts": "0",
            "total_upvotes": "0",
            "total_sessions": "0",
        }
        for key, value in defaults.items():
            self._conn.execute(
                "INSERT OR IGNORE INTO account_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, datetime.now().isoformat()),
            )
        self._conn.commit()

    def close(self):
        self._conn.close()

    # --- Account state ---

    def get(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM account_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO account_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, datetime.now().isoformat()),
        )
        self._conn.commit()

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.get(key, str(default)))
        except ValueError:
            return default

    def increment(self, key: str, amount: int = 1):
        current = self.get_int(key, 0)
        self.set(key, str(current + amount))

    # --- Phase management ---

    def get_phase(self) -> str:
        """Return current warmup phase based on account age.

        Phases: lurk (0-14), comment (15-28), active (29-56), established (57+)
        """
        start = date.fromisoformat(self.get("start_date", date.today().isoformat()))
        age_days = (date.today() - start).days

        if age_days < 15:
            return "lurk"
        elif age_days < 29:
            return "comment"
        elif age_days < 57:
            return "active"
        else:
            return "established"

    def get_account_age_days(self) -> int:
        start = date.fromisoformat(self.get("start_date", date.today().isoformat()))
        return (date.today() - start).days

    # --- Daily counts ---

    def get_daily_count(self, action: str, subreddit: str = "") -> int:
        today = date.today().isoformat()
        row = self._conn.execute(
            "SELECT count FROM daily_counts WHERE date = ? AND action = ? AND subreddit = ?",
            (today, action, subreddit),
        ).fetchone()
        return row["count"] if row else 0

    def increment_daily(self, action: str, subreddit: str = "", amount: int = 1):
        today = date.today().isoformat()
        self._conn.execute(
            """INSERT INTO daily_counts (date, action, subreddit, count)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(date, action, subreddit) DO UPDATE SET count = count + ?""",
            (today, action, subreddit, amount, amount),
        )
        self._conn.commit()

    def get_all_daily_counts(self) -> dict:
        today = date.today().isoformat()
        rows = self._conn.execute(
            "SELECT action, subreddit, count FROM daily_counts WHERE date = ?",
            (today,),
        ).fetchall()
        result = {}
        for row in rows:
            key = row["action"] if not row["subreddit"] else f"{row['action']}:{row['subreddit']}"
            result[key] = row["count"]
        return result

    # --- Comment history ---

    def add_comment(self, subreddit: str, post_title: str, text: str,
                    is_promo: bool = False, ngram_hash: str = ""):
        self._conn.execute(
            """INSERT INTO comment_history (timestamp, subreddit, post_title, comment_text, is_promo, ngram_hash)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), subreddit, post_title, text, int(is_promo), ngram_hash),
        )
        self._conn.commit()
        self.increment("total_comments")

    def get_recent_comments(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM comment_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_ngram_hashes(self, limit: int = 50) -> set[str]:
        rows = self._conn.execute(
            "SELECT ngram_hash FROM comment_history WHERE ngram_hash != '' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {r["ngram_hash"] for r in rows}

    # --- Post history ---

    def add_post(self, subreddit: str, title: str, body_snippet: str = "",
                 is_promo: bool = False, etsy_link: bool = False):
        self._conn.execute(
            """INSERT INTO post_history (timestamp, subreddit, title, body_snippet, is_promo, etsy_link_included)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), subreddit, title, body_snippet[:500],
             int(is_promo), int(etsy_link)),
        )
        self._conn.commit()
        self.increment("total_posts")
        if is_promo:
            self.increment("total_promo_posts")

    def get_last_promo_time(self) -> datetime | None:
        row = self._conn.execute(
            "SELECT timestamp FROM post_history WHERE is_promo = 1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return datetime.fromisoformat(row["timestamp"]) if row else None

    def get_promo_ratio(self) -> float:
        """Calculate self-promo ratio: promo / (total comments + total posts)."""
        total_comments = self.get_int("total_comments", 0)
        total_posts = self.get_int("total_posts", 0)
        total_promo = self.get_int("total_promo_posts", 0)
        total = total_comments + total_posts
        if total == 0:
            return 0.0
        return total_promo / total

    # --- Subreddit rules ---

    def get_sub_rules(self, subreddit: str) -> dict:
        row = self._conn.execute(
            "SELECT rules_json FROM subreddit_rules WHERE subreddit = ?", (subreddit,)
        ).fetchone()
        if row:
            try:
                return json.loads(row["rules_json"])
            except json.JSONDecodeError:
                pass
        return {}

    def set_sub_rules(self, subreddit: str, rules: dict):
        self._conn.execute(
            "INSERT OR REPLACE INTO subreddit_rules (subreddit, rules_json, last_checked) VALUES (?, ?, ?)",
            (subreddit, json.dumps(rules), datetime.now().isoformat()),
        )
        self._conn.commit()

    # --- Sessions ---

    def log_session(self, session_type: str, duration: float,
                    actions: dict, subreddits: list[str]):
        self._conn.execute(
            """INSERT INTO sessions (timestamp, session_type, duration_seconds, actions_json, subreddits_visited)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), session_type, duration,
             json.dumps(actions), json.dumps(subreddits)),
        )
        self._conn.commit()
        self.increment("total_sessions")

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Maintenance ---

    def cleanup_old_data(self, retention_days: int = 90):
        """Remove data older than retention period."""
        cutoff = (date.today() - timedelta(days=retention_days)).isoformat()
        self._conn.execute("DELETE FROM daily_counts WHERE date < ?", (cutoff,))
        self._conn.execute("DELETE FROM comment_history WHERE timestamp < ?", (cutoff,))
        self._conn.execute("DELETE FROM post_history WHERE timestamp < ?", (cutoff,))
        self._conn.execute("DELETE FROM sessions WHERE timestamp < ?", (cutoff,))
        self._conn.execute("VACUUM")
        self._conn.commit()
        logger.info(f"Cleaned up data older than {cutoff}")
