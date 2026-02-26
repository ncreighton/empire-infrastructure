"""VideoCodex — SQLite learning engine. Tracks videos, performance, and costs."""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS video_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    topic TEXT NOT NULL,
    niche TEXT NOT NULL,
    platform TEXT NOT NULL,
    format TEXT NOT NULL,
    hook_formula TEXT,
    trending_format TEXT,
    quality_score INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    render_url TEXT,
    status TEXT DEFAULT 'created'
);

CREATE TABLE IF NOT EXISTS performance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    logged_at TEXT NOT NULL,
    views INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    watch_time_seconds REAL DEFAULT 0,
    retention_percent REAL DEFAULT 0,
    ctr_percent REAL DEFAULT 0,
    FOREIGN KEY (video_id) REFERENCES video_log(id)
);

CREATE TABLE IF NOT EXISTS cost_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER,
    logged_at TEXT NOT NULL,
    category TEXT NOT NULL,
    provider TEXT,
    amount REAL NOT NULL,
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_video_niche ON video_log(niche);
CREATE INDEX IF NOT EXISTS idx_video_platform ON video_log(platform);
CREATE INDEX IF NOT EXISTS idx_perf_video ON performance_log(video_id);
CREATE INDEX IF NOT EXISTS idx_cost_video ON cost_log(video_id);
"""


class VideoCodex:
    """SQLite learning engine — tracks what works, what costs, what to try next."""

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
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Logging ───────────────────────────────────────────────────────

    def log_video(self, topic: str, niche: str, platform: str, format: str,
                  hook_formula: str = "", trending_format: str = "",
                  quality_score: int = 0, total_cost: float = 0.0,
                  render_url: str = "", status: str = "created") -> int:
        """Log a created video. Returns video_id."""
        cur = self._conn.execute(
            "INSERT INTO video_log (created_at, topic, niche, platform, format, "
            "hook_formula, trending_format, quality_score, total_cost, render_url, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), topic, niche, platform, format,
             hook_formula, trending_format, quality_score, total_cost, render_url, status),
        )
        self._conn.commit()
        return cur.lastrowid

    def log_performance(self, video_id: int, views: int = 0, likes: int = 0,
                        comments: int = 0, shares: int = 0,
                        watch_time_seconds: float = 0, retention_percent: float = 0,
                        ctr_percent: float = 0):
        """Log performance metrics for a video."""
        self._conn.execute(
            "INSERT INTO performance_log (video_id, logged_at, views, likes, comments, "
            "shares, watch_time_seconds, retention_percent, ctr_percent) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (video_id, datetime.utcnow().isoformat(), views, likes, comments,
             shares, watch_time_seconds, retention_percent, ctr_percent),
        )
        self._conn.commit()

    def log_cost(self, category: str, amount: float, provider: str = "",
                 details: str = "", video_id: int = None):
        """Log a cost entry."""
        self._conn.execute(
            "INSERT INTO cost_log (video_id, logged_at, category, provider, amount, details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (video_id, datetime.utcnow().isoformat(), category, provider, amount, details),
        )
        self._conn.commit()

    def update_video_status(self, video_id: int, status: str, render_url: str = None):
        """Update video status (created, rendered, published, failed)."""
        if render_url:
            self._conn.execute(
                "UPDATE video_log SET status = ?, render_url = ? WHERE id = ?",
                (status, render_url, video_id),
            )
        else:
            self._conn.execute(
                "UPDATE video_log SET status = ? WHERE id = ?",
                (status, video_id),
            )
        self._conn.commit()

    # ── Insights ──────────────────────────────────────────────────────

    def get_video_count(self, niche: str = None, days: int = None) -> int:
        """Get total video count, optionally filtered."""
        query = "SELECT COUNT(*) FROM video_log WHERE 1=1"
        params = []
        if niche:
            query += " AND niche = ?"
            params.append(niche)
        if days:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            query += " AND created_at > ?"
            params.append(cutoff)
        row = self._conn.execute(query, params).fetchone()
        return row[0] if row else 0

    def get_total_cost(self, days: int = None) -> float:
        """Get total cost, optionally for last N days."""
        query = "SELECT COALESCE(SUM(amount), 0) FROM cost_log WHERE 1=1"
        params = []
        if days:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            query += " AND logged_at > ?"
            params.append(cutoff)
        row = self._conn.execute(query, params).fetchone()
        return row[0] if row else 0.0

    def get_avg_cost_per_video(self, days: int = 30) -> float:
        """Get average cost per video over a period."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(AVG(total_cost), 0) FROM video_log WHERE created_at > ?",
            (cutoff,),
        ).fetchone()
        return row[0] if row else 0.0

    def get_best_hooks(self, niche: str = None, limit: int = 5) -> list:
        """Get best-performing hook formulas by average quality score."""
        query = (
            "SELECT hook_formula, COUNT(*) as count, AVG(quality_score) as avg_score "
            "FROM video_log WHERE hook_formula != ''"
        )
        params = []
        if niche:
            query += " AND niche = ?"
            params.append(niche)
        query += " GROUP BY hook_formula ORDER BY avg_score DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [{"hook": r["hook_formula"], "count": r["count"], "avg_score": r["avg_score"]} for r in rows]

    def get_niche_stats(self, niche: str) -> dict:
        """Get aggregate stats for a niche."""
        row = self._conn.execute(
            "SELECT COUNT(*) as total, AVG(quality_score) as avg_quality, "
            "AVG(total_cost) as avg_cost, SUM(total_cost) as total_cost "
            "FROM video_log WHERE niche = ?",
            (niche,),
        ).fetchone()
        if not row or row["total"] == 0:
            return {"total": 0, "avg_quality": 0, "avg_cost": 0, "total_cost": 0}
        return dict(row)

    def get_recent_videos(self, limit: int = 10, niche: str = None) -> list:
        """Get recent videos."""
        query = "SELECT * FROM video_log"
        params = []
        if niche:
            query += " WHERE niche = ?"
            params.append(niche)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(query, params).fetchall()]

    def get_cost_breakdown(self, days: int = 30) -> dict:
        """Get cost breakdown by category."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            "SELECT category, SUM(amount) as total FROM cost_log "
            "WHERE logged_at > ? GROUP BY category ORDER BY total DESC",
            (cutoff,),
        ).fetchall()
        return {r["category"]: r["total"] for r in rows}

    def get_insights(self, niche: str = None) -> dict:
        """Get comprehensive insights."""
        return {
            "total_videos": self.get_video_count(niche=niche),
            "videos_30d": self.get_video_count(niche=niche, days=30),
            "total_cost_30d": round(self.get_total_cost(days=30), 2),
            "avg_cost_per_video": round(self.get_avg_cost_per_video(), 4),
            "best_hooks": self.get_best_hooks(niche=niche),
            "cost_breakdown": self.get_cost_breakdown(),
        }

    def close(self):
        self._conn.close()
