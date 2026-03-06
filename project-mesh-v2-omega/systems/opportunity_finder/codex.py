"""Opportunity Finder Codex — SQLite storage for opportunities and seasonal data."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "opportunity.db"


class OpportunityCodex:
    """Persistent storage for discovered opportunities."""

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
                CREATE TABLE IF NOT EXISTS opportunities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_slug TEXT NOT NULL,
                    keyword TEXT NOT NULL,
                    opportunity_type TEXT NOT NULL,
                    score REAL DEFAULT 0,
                    traffic_potential REAL DEFAULT 0,
                    monetization_fit REAL DEFAULT 0,
                    effort REAL DEFAULT 0,
                    cross_site_synergy REAL DEFAULT 0,
                    seasonal_urgency REAL DEFAULT 0,
                    current_position REAL,
                    impressions INTEGER DEFAULT 0,
                    clicks INTEGER DEFAULT 0,
                    existing_url TEXT,
                    status TEXT DEFAULT 'open',
                    details TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    acted_on TEXT
                );

                CREATE TABLE IF NOT EXISTS opportunity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date TEXT NOT NULL,
                    site_slug TEXT,
                    total_found INTEGER DEFAULT 0,
                    top_score REAL DEFAULT 0,
                    summary TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS seasonal_calendar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT NOT NULL,
                    peak_month INTEGER NOT NULL,
                    niche TEXT,
                    traffic_multiplier REAL DEFAULT 1.0,
                    notes TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_opp_site ON opportunities(site_slug);
                CREATE INDEX IF NOT EXISTS idx_opp_status ON opportunities(status);
                CREATE INDEX IF NOT EXISTS idx_opp_score ON opportunities(score DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_opp_unique
                    ON opportunities(site_slug, keyword, opportunity_type)
                    WHERE status='open';
            """)

    def upsert_opportunity(self, site_slug: str, keyword: str, opp_type: str,
                           score: float, dimensions: Dict, details: Dict = None) -> int:
        with self._conn() as conn:
            # Check if open opportunity already exists
            existing = conn.execute(
                "SELECT id FROM opportunities WHERE site_slug=? AND keyword=? "
                "AND opportunity_type=? AND status='open'",
                (site_slug, keyword, opp_type)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE opportunities SET score=?, traffic_potential=?, "
                    "monetization_fit=?, effort=?, cross_site_synergy=?, "
                    "seasonal_urgency=?, current_position=?, impressions=?, "
                    "clicks=?, details=? WHERE id=?",
                    (score, dimensions.get("traffic_potential", 0),
                     dimensions.get("monetization_fit", 0),
                     dimensions.get("effort", 0),
                     dimensions.get("cross_site_synergy", 0),
                     dimensions.get("seasonal_urgency", 0),
                     dimensions.get("current_position"),
                     dimensions.get("impressions", 0),
                     dimensions.get("clicks", 0),
                     json.dumps(details) if details else None,
                     existing["id"])
                )
                return existing["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO opportunities (site_slug, keyword, opportunity_type, "
                    "score, traffic_potential, monetization_fit, effort, "
                    "cross_site_synergy, seasonal_urgency, current_position, "
                    "impressions, clicks, existing_url, details) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (site_slug, keyword, opp_type, score,
                     dimensions.get("traffic_potential", 0),
                     dimensions.get("monetization_fit", 0),
                     dimensions.get("effort", 0),
                     dimensions.get("cross_site_synergy", 0),
                     dimensions.get("seasonal_urgency", 0),
                     dimensions.get("current_position"),
                     dimensions.get("impressions", 0),
                     dimensions.get("clicks", 0),
                     dimensions.get("existing_url"),
                     json.dumps(details) if details else None)
                )
                return cur.lastrowid

    def log_snapshot(self, site_slug: str, total: int, top_score: float, summary: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO opportunity_snapshots (scan_date, site_slug, total_found, top_score, summary) "
                "VALUES (date('now'), ?, ?, ?, ?)",
                (site_slug, total, top_score, summary)
            )

    def get_queue(self, site_slug: str = None, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            if site_slug:
                rows = conn.execute(
                    "SELECT * FROM opportunities WHERE site_slug=? AND status='open' "
                    "ORDER BY score DESC LIMIT ?",
                    (site_slug, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM opportunities WHERE status='open' "
                    "ORDER BY score DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def mark_acted(self, opp_id: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE opportunities SET status='acted', acted_on=datetime('now') WHERE id=?",
                (opp_id,)
            )

    def get_cross_site_keywords(self) -> List[Dict]:
        """Find keywords appearing across multiple sites."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT keyword, GROUP_CONCAT(DISTINCT site_slug) as sites, "
                "COUNT(DISTINCT site_slug) as site_count, MAX(score) as best_score "
                "FROM opportunities WHERE status='open' "
                "GROUP BY keyword HAVING site_count > 1 "
                "ORDER BY site_count DESC, best_score DESC LIMIT 50"
            ).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> Dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM opportunities").fetchone()["c"]
            open_count = conn.execute(
                "SELECT COUNT(*) as c FROM opportunities WHERE status='open'"
            ).fetchone()["c"]
            acted = conn.execute(
                "SELECT COUNT(*) as c FROM opportunities WHERE status='acted'"
            ).fetchone()["c"]
            return {"total": total, "open": open_count, "acted": acted}
