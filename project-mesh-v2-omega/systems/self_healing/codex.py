"""Self-Healing Codex — SQLite storage for healing events and service history."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "healing.db"


class HealingCodex:
    """Persistent storage for healing events, service health, and API key status."""

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
                CREATE TABLE IF NOT EXISTS healing_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    action_taken TEXT,
                    result TEXT,
                    details TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS service_health_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_time_ms REAL,
                    error_message TEXT,
                    checked_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS api_key_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT NOT NULL UNIQUE,
                    key_prefix TEXT,
                    is_valid INTEGER DEFAULT 1,
                    last_checked TEXT,
                    usage_count INTEGER DEFAULT 0,
                    rate_limit_remaining INTEGER,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_healing_type ON healing_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_health_svc ON service_health_history(service_id);
            """)

    def log_healing_event(self, event_type: str, target: str, action: str,
                          result: str, details: Optional[Dict] = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO healing_events (event_type, target, action_taken, result, details) "
                "VALUES (?, ?, ?, ?, ?)",
                (event_type, target, action, result,
                 json.dumps(details) if details else None)
            )
            return cur.lastrowid

    def log_health_check(self, service_id: str, status: str,
                         response_time_ms: float = 0, error: str = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO service_health_history (service_id, status, response_time_ms, error_message) "
                "VALUES (?, ?, ?, ?)",
                (service_id, status, response_time_ms, error)
            )

    def update_api_key(self, service_name: str, is_valid: bool,
                       key_prefix: str = None, rate_limit: int = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO api_key_status (service_name, key_prefix, is_valid, last_checked, rate_limit_remaining) "
                "VALUES (?, ?, ?, datetime('now'), ?) "
                "ON CONFLICT(service_name) DO UPDATE SET "
                "is_valid=excluded.is_valid, key_prefix=excluded.key_prefix, "
                "last_checked=datetime('now'), rate_limit_remaining=excluded.rate_limit_remaining, "
                "usage_count=usage_count+1, updated_at=datetime('now')",
                (service_name, key_prefix, int(is_valid), rate_limit)
            )

    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM healing_events ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_service_history(self, service_id: str, limit: int = 100) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM service_health_history WHERE service_id=? "
                "ORDER BY checked_at DESC LIMIT ?",
                (service_id, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_api_key_status(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM api_key_status ORDER BY service_name"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_service_uptime(self, service_id: str, lookback_hours: int = 24) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN status='healthy' THEN 1 ELSE 0 END) as healthy "
                "FROM service_health_history WHERE service_id=? "
                "AND checked_at >= datetime('now', ?)",
                (service_id, f"-{lookback_hours} hours")
            ).fetchone()
            if row["total"] == 0:
                return 0.0
            return round(row["healthy"] / row["total"] * 100, 1)

    def stats(self) -> Dict:
        with self._conn() as conn:
            events = conn.execute("SELECT COUNT(*) as c FROM healing_events").fetchone()["c"]
            checks = conn.execute("SELECT COUNT(*) as c FROM service_health_history").fetchone()["c"]
            keys = conn.execute("SELECT COUNT(*) as c FROM api_key_status").fetchone()["c"]
            return {"healing_events": events, "health_checks": checks, "api_keys_tracked": keys}
