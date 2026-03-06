"""Feedback Loop Codex — SQLite storage for cycles and compounding metrics."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "feedback.db"


class FeedbackCodex:
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
                CREATE TABLE IF NOT EXISTS cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_number INTEGER NOT NULL,
                    status TEXT DEFAULT 'running',
                    phase TEXT DEFAULT 'discover',
                    discover_results TEXT,
                    create_results TEXT,
                    measure_results TEXT,
                    learn_results TEXT,
                    improve_results TEXT,
                    started_at TEXT DEFAULT (datetime('now')),
                    completed_at TEXT,
                    duration_seconds INTEGER
                );

                CREATE TABLE IF NOT EXISTS system_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id INTEGER,
                    source_system TEXT NOT NULL,
                    target_system TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    data_passed TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
                );

                CREATE TABLE IF NOT EXISTS compounding_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    previous_value REAL,
                    delta_pct REAL,
                    recorded_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (cycle_id) REFERENCES cycles(id)
                );

                CREATE INDEX IF NOT EXISTS idx_cycle_status ON cycles(status);
                CREATE INDEX IF NOT EXISTS idx_compound_metric ON compounding_metrics(metric_name);
            """)

    def start_cycle(self) -> int:
        with self._conn() as conn:
            last = conn.execute(
                "SELECT MAX(cycle_number) as n FROM cycles"
            ).fetchone()["n"]
            cycle_num = (last or 0) + 1
            cur = conn.execute(
                "INSERT INTO cycles (cycle_number) VALUES (?)", (cycle_num,)
            )
            return cur.lastrowid

    def update_phase(self, cycle_id: int, phase: str, results: Dict):
        field_map = {
            "discover": "discover_results",
            "create": "create_results",
            "measure": "measure_results",
            "learn": "learn_results",
            "improve": "improve_results",
        }
        field = field_map.get(phase)
        if field:
            with self._conn() as conn:
                conn.execute(
                    f"UPDATE cycles SET phase=?, {field}=? WHERE id=?",
                    (phase, json.dumps(results), cycle_id)
                )

    def complete_cycle(self, cycle_id: int, duration: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE cycles SET status='completed', completed_at=datetime('now'), "
                "duration_seconds=? WHERE id=?",
                (duration, cycle_id)
            )

    def log_interaction(self, cycle_id: int, source: str, target: str,
                         interaction_type: str, data: Dict = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO system_interactions (cycle_id, source_system, target_system, "
                "interaction_type, data_passed) VALUES (?, ?, ?, ?, ?)",
                (cycle_id, source, target, interaction_type,
                 json.dumps(data) if data else None)
            )

    def record_metric(self, cycle_id: int, metric: str, value: float):
        with self._conn() as conn:
            # Get previous value
            prev = conn.execute(
                "SELECT value FROM compounding_metrics WHERE metric_name=? "
                "ORDER BY recorded_at DESC LIMIT 1",
                (metric,)
            ).fetchone()

            prev_value = prev["value"] if prev else None
            delta = ((value - prev_value) / max(abs(prev_value), 0.01) * 100) if prev_value else None

            conn.execute(
                "INSERT INTO compounding_metrics (cycle_id, metric_name, value, "
                "previous_value, delta_pct) VALUES (?, ?, ?, ?, ?)",
                (cycle_id, metric, value, prev_value, delta)
            )

    def get_cycles(self, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, cycle_number, status, phase, started_at, completed_at, "
                "duration_seconds FROM cycles ORDER BY cycle_number DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_cycle(self, cycle_id: int) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM cycles WHERE id=?", (cycle_id,)).fetchone()
            if not row:
                return None
            cycle = dict(row)
            for field in ("discover_results", "create_results", "measure_results",
                          "learn_results", "improve_results"):
                try:
                    cycle[field] = json.loads(cycle.get(field, "null"))
                except (json.JSONDecodeError, TypeError):
                    pass
            return cycle

    def get_compounding_trend(self, metric: str, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM compounding_metrics WHERE metric_name=? "
                "ORDER BY recorded_at DESC LIMIT ?",
                (metric, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_improvement_rate(self) -> Dict:
        """Calculate average improvement per cycle."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT metric_name, AVG(delta_pct) as avg_delta, COUNT(*) as cycles "
                "FROM compounding_metrics WHERE delta_pct IS NOT NULL "
                "GROUP BY metric_name"
            ).fetchall()
            return {r["metric_name"]: {
                "avg_improvement_pct": round(r["avg_delta"], 2),
                "cycles_measured": r["cycles"],
            } for r in rows}

    def stats(self) -> Dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM cycles").fetchone()["c"]
            completed = conn.execute(
                "SELECT COUNT(*) as c FROM cycles WHERE status='completed'"
            ).fetchone()["c"]
            interactions = conn.execute(
                "SELECT COUNT(*) as c FROM system_interactions"
            ).fetchone()["c"]
            return {
                "total_cycles": total,
                "completed_cycles": completed,
                "system_interactions": interactions,
            }
