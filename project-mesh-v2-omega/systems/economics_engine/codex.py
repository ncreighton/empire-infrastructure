"""Economics Engine Codex — SQLite storage for revenue, costs, and ROI."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "economics.db"


class EconomicsCodex:
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
                CREATE TABLE IF NOT EXISTS revenue_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_slug TEXT,
                    source TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    description TEXT,
                    reference_id TEXT,
                    event_date TEXT DEFAULT (date('now')),
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS cost_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_slug TEXT,
                    category TEXT NOT NULL,
                    service TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    description TEXT,
                    reference_id TEXT,
                    event_date TEXT DEFAULT (date('now')),
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS article_economics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_slug TEXT NOT NULL,
                    post_id INTEGER,
                    title TEXT NOT NULL,
                    total_cost REAL DEFAULT 0,
                    total_revenue REAL DEFAULT 0,
                    roi REAL DEFAULT 0,
                    cost_breakdown TEXT,
                    revenue_breakdown TEXT,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS site_economics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_slug TEXT NOT NULL UNIQUE,
                    total_cost REAL DEFAULT 0,
                    total_revenue REAL DEFAULT 0,
                    roi REAL DEFAULT 0,
                    article_count INTEGER DEFAULT 0,
                    avg_cost_per_article REAL DEFAULT 0,
                    avg_revenue_per_article REAL DEFAULT 0,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_rev_site ON revenue_events(site_slug);
                CREATE INDEX IF NOT EXISTS idx_cost_site ON cost_events(site_slug);
                CREATE INDEX IF NOT EXISTS idx_art_econ_site ON article_economics(site_slug);
            """)

    def log_revenue(self, site_slug: str, source: str, amount: float,
                    description: str = None, reference_id: str = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO revenue_events (site_slug, source, amount, description, reference_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (site_slug, source, amount, description, reference_id)
            )

    def log_cost(self, site_slug: str, category: str, service: str, amount: float,
                 description: str = None, reference_id: str = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cost_events (site_slug, category, service, amount, description, reference_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (site_slug, category, service, amount, description, reference_id)
            )

    def update_article_economics(self, site_slug: str, title: str,
                                  cost: float, revenue: float,
                                  cost_breakdown: Dict = None,
                                  revenue_breakdown: Dict = None):
        roi = ((revenue - cost) / cost * 100) if cost > 0 else 0
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO article_economics (site_slug, title, total_cost, "
                "total_revenue, roi, cost_breakdown, revenue_breakdown) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (site_slug, title, cost, revenue, roi,
                 json.dumps(cost_breakdown) if cost_breakdown else None,
                 json.dumps(revenue_breakdown) if revenue_breakdown else None)
            )

    def update_site_economics(self, site_slug: str):
        """Recalculate site-level economics from article data."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count, SUM(total_cost) as cost, "
                "SUM(total_revenue) as rev FROM article_economics WHERE site_slug=?",
                (site_slug,)
            ).fetchone()

            count = row["count"] or 0
            cost = row["cost"] or 0
            rev = row["rev"] or 0
            roi = ((rev - cost) / cost * 100) if cost > 0 else 0

            conn.execute(
                "INSERT INTO site_economics (site_slug, total_cost, total_revenue, "
                "roi, article_count, avg_cost_per_article, avg_revenue_per_article) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(site_slug) DO UPDATE SET "
                "total_cost=excluded.total_cost, total_revenue=excluded.total_revenue, "
                "roi=excluded.roi, article_count=excluded.article_count, "
                "avg_cost_per_article=excluded.avg_cost_per_article, "
                "avg_revenue_per_article=excluded.avg_revenue_per_article, "
                "updated_at=datetime('now')",
                (site_slug, cost, rev, roi, count,
                 cost / count if count > 0 else 0,
                 rev / count if count > 0 else 0)
            )

    def get_empire_summary(self) -> Dict:
        with self._conn() as conn:
            sites = conn.execute(
                "SELECT * FROM site_economics ORDER BY roi DESC"
            ).fetchall()
            total_cost = sum(s["total_cost"] for s in sites)
            total_rev = sum(s["total_revenue"] for s in sites)
            total_roi = ((total_rev - total_cost) / total_cost * 100) if total_cost > 0 else 0
            return {
                "total_cost": round(total_cost, 2),
                "total_revenue": round(total_rev, 2),
                "total_roi": round(total_roi, 1),
                "sites": [dict(s) for s in sites],
            }

    def get_site_summary(self, site_slug: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM site_economics WHERE site_slug=?", (site_slug,)
            ).fetchone()
            return dict(row) if row else None

    def get_top_articles(self, site_slug: str = None, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            if site_slug:
                rows = conn.execute(
                    "SELECT * FROM article_economics WHERE site_slug=? "
                    "ORDER BY roi DESC LIMIT ?", (site_slug, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM article_economics ORDER BY roi DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> Dict:
        with self._conn() as conn:
            rev_total = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) as c FROM revenue_events"
            ).fetchone()["c"]
            cost_total = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) as c FROM cost_events"
            ).fetchone()["c"]
            articles = conn.execute(
                "SELECT COUNT(*) as c FROM article_economics"
            ).fetchone()["c"]
            return {
                "total_revenue": round(rev_total, 2),
                "total_cost": round(cost_total, 2),
                "net_profit": round(rev_total - cost_total, 2),
                "articles_tracked": articles,
            }
