"""Intelligence Amplifier Codex — SQLite storage for article performance and playbooks."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "intelligence.db"


class IntelligenceCodex:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS article_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_slug TEXT NOT NULL,
                    post_id INTEGER,
                    url TEXT,
                    title TEXT NOT NULL,
                    headline_type TEXT,
                    word_count INTEGER,
                    clicks_30d INTEGER DEFAULT 0,
                    impressions_30d INTEGER DEFAULT 0,
                    sessions_30d INTEGER DEFAULT 0,
                    avg_position REAL,
                    ctr REAL,
                    velocity REAL DEFAULT 0,
                    decay_rate REAL DEFAULT 0,
                    grade TEXT DEFAULT 'C',
                    scored_at TEXT DEFAULT (datetime('now')),
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS winning_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_slug TEXT,
                    niche TEXT,
                    pattern_type TEXT NOT NULL,
                    pattern_value TEXT NOT NULL,
                    win_rate REAL DEFAULT 0,
                    sample_size INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    details TEXT,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS niche_playbooks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    niche TEXT NOT NULL UNIQUE,
                    optimal_word_count_min INTEGER,
                    optimal_word_count_max INTEGER,
                    best_headline_formulas TEXT,
                    best_publish_days TEXT,
                    best_ctas TEXT,
                    avg_grade TEXT,
                    top_performing_topics TEXT,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_perf_site ON article_performance(site_slug);
                CREATE INDEX IF NOT EXISTS idx_perf_grade ON article_performance(grade);
                CREATE INDEX IF NOT EXISTS idx_pattern_niche ON winning_patterns(niche);
            """)

    def upsert_article(self, site_slug: str, title: str, data: Dict) -> int:
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM article_performance WHERE site_slug=? AND title=?",
                (site_slug, title)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE article_performance SET clicks_30d=?, impressions_30d=?, "
                    "sessions_30d=?, avg_position=?, ctr=?, velocity=?, decay_rate=?, "
                    "grade=?, headline_type=?, word_count=?, scored_at=datetime('now') WHERE id=?",
                    (data.get("clicks", 0), data.get("impressions", 0),
                     data.get("sessions", 0), data.get("position"),
                     data.get("ctr"), data.get("velocity", 0),
                     data.get("decay_rate", 0), data.get("grade", "C"),
                     data.get("headline_type"), data.get("word_count"),
                     existing["id"])
                )
                return existing["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO article_performance (site_slug, post_id, url, title, "
                    "headline_type, word_count, clicks_30d, impressions_30d, sessions_30d, "
                    "avg_position, ctr, velocity, decay_rate, grade) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (site_slug, data.get("post_id"), data.get("url"), title,
                     data.get("headline_type"), data.get("word_count"),
                     data.get("clicks", 0), data.get("impressions", 0),
                     data.get("sessions", 0), data.get("position"),
                     data.get("ctr"), data.get("velocity", 0),
                     data.get("decay_rate", 0), data.get("grade", "C"))
                )
                return cur.lastrowid

    def save_pattern(self, niche: str, pattern_type: str, pattern_value: str,
                     win_rate: float, sample_size: int, site_slug: str = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO winning_patterns (site_slug, niche, pattern_type, "
                "pattern_value, win_rate, sample_size, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT DO NOTHING",
                (site_slug, niche, pattern_type, pattern_value, win_rate,
                 sample_size, min(1.0, sample_size / 20))
            )

    def save_playbook(self, niche: str, playbook: Dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO niche_playbooks (niche, optimal_word_count_min, "
                "optimal_word_count_max, best_headline_formulas, best_publish_days, "
                "best_ctas, avg_grade, top_performing_topics) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(niche) DO UPDATE SET "
                "optimal_word_count_min=excluded.optimal_word_count_min, "
                "optimal_word_count_max=excluded.optimal_word_count_max, "
                "best_headline_formulas=excluded.best_headline_formulas, "
                "best_publish_days=excluded.best_publish_days, "
                "best_ctas=excluded.best_ctas, avg_grade=excluded.avg_grade, "
                "top_performing_topics=excluded.top_performing_topics, "
                "updated_at=datetime('now')",
                (niche, playbook.get("word_count_min"), playbook.get("word_count_max"),
                 json.dumps(playbook.get("headline_formulas", [])),
                 json.dumps(playbook.get("publish_days", [])),
                 json.dumps(playbook.get("ctas", [])),
                 playbook.get("avg_grade", "C"),
                 json.dumps(playbook.get("top_topics", [])))
            )

    def get_articles(self, site_slug: str = None, grade: str = None,
                     limit: int = 50) -> List[Dict]:
        with self._conn() as conn:
            conditions = []
            params = []
            if site_slug:
                conditions.append("site_slug=?")
                params.append(site_slug)
            if grade:
                conditions.append("grade=?")
                params.append(grade)
            where = " AND ".join(conditions) if conditions else "1=1"
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM article_performance WHERE {where} "
                "ORDER BY clicks_30d DESC LIMIT ?", params
            ).fetchall()
            return [dict(r) for r in rows]

    def get_decaying(self, site_slug: str = None, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            if site_slug:
                rows = conn.execute(
                    "SELECT * FROM article_performance WHERE site_slug=? "
                    "AND decay_rate > 0.1 ORDER BY decay_rate DESC LIMIT ?",
                    (site_slug, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM article_performance WHERE decay_rate > 0.1 "
                    "ORDER BY decay_rate DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_playbook(self, niche: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM niche_playbooks WHERE niche=?", (niche,)
            ).fetchone()
            if row:
                pb = dict(row)
                for field in ("best_headline_formulas", "best_publish_days",
                              "best_ctas", "top_performing_topics"):
                    try:
                        pb[field] = json.loads(pb.get(field, "[]") or "[]")
                    except (json.JSONDecodeError, TypeError):
                        pb[field] = []
                return pb
            return None

    def get_all_playbooks(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM niche_playbooks ORDER BY niche").fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> Dict:
        with self._conn() as conn:
            articles = conn.execute("SELECT COUNT(*) as c FROM article_performance").fetchone()["c"]
            patterns = conn.execute("SELECT COUNT(*) as c FROM winning_patterns").fetchone()["c"]
            playbooks = conn.execute("SELECT COUNT(*) as c FROM niche_playbooks").fetchone()["c"]
            return {"articles_tracked": articles, "patterns": patterns, "playbooks": playbooks}
