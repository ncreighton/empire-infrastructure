"""Cross-Pollination Codex — SQLite storage for overlaps and promotions."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "pollination.db"


class PollinationCodex:
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
                CREATE TABLE IF NOT EXISTS audience_overlaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_a TEXT NOT NULL,
                    site_b TEXT NOT NULL,
                    overlap_keywords INTEGER DEFAULT 0,
                    overlap_score REAL DEFAULT 0,
                    sample_keywords TEXT,
                    updated_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(site_a, site_b)
                );

                CREATE TABLE IF NOT EXISTS cross_promotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_site TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    target_site TEXT NOT NULL,
                    target_url TEXT NOT NULL,
                    anchor_text TEXT,
                    keyword TEXT,
                    status TEXT DEFAULT 'suggested',
                    injected_at TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS niche_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_name TEXT NOT NULL,
                    sites TEXT NOT NULL,
                    shared_keywords INTEGER DEFAULT 0,
                    description TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_overlap_sites ON audience_overlaps(site_a, site_b);
                CREATE INDEX IF NOT EXISTS idx_promo_status ON cross_promotions(status);
            """)

    def upsert_overlap(self, site_a: str, site_b: str, overlap_count: int,
                       score: float, sample_keywords: List[str] = None):
        # Ensure consistent ordering
        if site_a > site_b:
            site_a, site_b = site_b, site_a
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO audience_overlaps (site_a, site_b, overlap_keywords, "
                "overlap_score, sample_keywords) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(site_a, site_b) DO UPDATE SET "
                "overlap_keywords=excluded.overlap_keywords, "
                "overlap_score=excluded.overlap_score, "
                "sample_keywords=excluded.sample_keywords, "
                "updated_at=datetime('now')",
                (site_a, site_b, overlap_count, score,
                 json.dumps(sample_keywords[:10]) if sample_keywords else None)
            )

    def suggest_promotion(self, source_site: str, source_url: str,
                          target_site: str, target_url: str,
                          anchor_text: str, keyword: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO cross_promotions (source_site, source_url, target_site, "
                "target_url, anchor_text, keyword) VALUES (?, ?, ?, ?, ?, ?)",
                (source_site, source_url, target_site, target_url, anchor_text, keyword)
            )
            return cur.lastrowid

    def mark_injected(self, promo_id: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE cross_promotions SET status='injected', injected_at=datetime('now') "
                "WHERE id=?", (promo_id,)
            )

    def save_cluster(self, name: str, sites: List[str], shared: int, desc: str = ""):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO niche_clusters (cluster_name, sites, shared_keywords, description) "
                "VALUES (?, ?, ?, ?)",
                (name, json.dumps(sites), shared, desc)
            )

    def get_overlaps(self, min_score: float = 0) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audience_overlaps WHERE overlap_score >= ? "
                "ORDER BY overlap_score DESC",
                (min_score,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_suggestions(self, status: str = "suggested", limit: int = 50) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cross_promotions WHERE status=? "
                "ORDER BY created_at DESC LIMIT ?",
                (status, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_clusters(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM niche_clusters ORDER BY cluster_name").fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["sites"] = json.loads(d.get("sites", "[]"))
                except (json.JSONDecodeError, TypeError):
                    d["sites"] = []
                result.append(d)
            return result

    def stats(self) -> Dict:
        with self._conn() as conn:
            overlaps = conn.execute("SELECT COUNT(*) as c FROM audience_overlaps").fetchone()["c"]
            promos = conn.execute("SELECT COUNT(*) as c FROM cross_promotions").fetchone()["c"]
            injected = conn.execute(
                "SELECT COUNT(*) as c FROM cross_promotions WHERE status='injected'"
            ).fetchone()["c"]
            clusters = conn.execute("SELECT COUNT(*) as c FROM niche_clusters").fetchone()["c"]
            return {"overlaps": overlaps, "suggestions": promos,
                    "injected": injected, "clusters": clusters}
