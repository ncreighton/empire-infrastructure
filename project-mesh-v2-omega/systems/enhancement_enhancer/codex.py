"""Enhancement Enhancer Codex — SQLite storage for quality, experiments, propagation."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "enhancer.db"


class EnhancerCodex:
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
                CREATE TABLE IF NOT EXISTS quality_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline TEXT NOT NULL,
                    site_slug TEXT,
                    quality_score REAL NOT NULL,
                    dimensions TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    pipeline TEXT NOT NULL,
                    variant_a TEXT NOT NULL,
                    variant_b TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    status TEXT DEFAULT 'running',
                    winner TEXT,
                    confidence REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS experiment_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    variant TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT,
                    observed_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS config_propagations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_experiment_id INTEGER,
                    config_key TEXT NOT NULL,
                    config_value TEXT NOT NULL,
                    applied_to TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT (datetime('now')),
                    applied_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_quality_pipeline ON quality_snapshots(pipeline);
                CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
            """)

    def log_quality(self, pipeline: str, score: float, site_slug: str = None,
                     dimensions: Dict = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO quality_snapshots (pipeline, site_slug, quality_score, dimensions) "
                "VALUES (?, ?, ?, ?)",
                (pipeline, site_slug, score, json.dumps(dimensions) if dimensions else None)
            )

    def create_experiment(self, name: str, pipeline: str, variant_a: str,
                           variant_b: str, metric: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO experiments (name, pipeline, variant_a, variant_b, metric) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, pipeline, variant_a, variant_b, metric)
            )
            return cur.lastrowid

    def add_observation(self, experiment_id: int, variant: str,
                         metric_value: float, metadata: Dict = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO experiment_observations (experiment_id, variant, metric_value, metadata) "
                "VALUES (?, ?, ?, ?)",
                (experiment_id, variant, metric_value,
                 json.dumps(metadata) if metadata else None)
            )

    def conclude_experiment(self, experiment_id: int, winner: str, confidence: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE experiments SET status='completed', winner=?, confidence=?, "
                "completed_at=datetime('now') WHERE id=?",
                (winner, confidence, experiment_id)
            )

    def create_propagation(self, experiment_id: int, config_key: str,
                            config_value: str, target_sites: List[str]) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO config_propagations (source_experiment_id, config_key, "
                "config_value, applied_to) VALUES (?, ?, ?, ?)",
                (experiment_id, config_key, config_value, json.dumps(target_sites))
            )
            return cur.lastrowid

    def mark_propagated(self, prop_id: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE config_propagations SET status='applied', applied_at=datetime('now') "
                "WHERE id=?", (prop_id,)
            )

    def get_quality_trend(self, pipeline: str, limit: int = 30) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM quality_snapshots WHERE pipeline=? "
                "ORDER BY created_at DESC LIMIT ?", (pipeline, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_experiments(self, status: str = None) -> List[Dict]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE status=? ORDER BY created_at DESC",
                    (status,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def get_experiment_results(self, experiment_id: int) -> Dict:
        with self._conn() as conn:
            exp = conn.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if not exp:
                return {}

            obs = conn.execute(
                "SELECT variant, AVG(metric_value) as avg_val, COUNT(*) as count "
                "FROM experiment_observations WHERE experiment_id=? "
                "GROUP BY variant", (experiment_id,)
            ).fetchall()

            return {
                "experiment": dict(exp),
                "results": [dict(o) for o in obs],
            }

    def stats(self) -> Dict:
        with self._conn() as conn:
            snapshots = conn.execute("SELECT COUNT(*) as c FROM quality_snapshots").fetchone()["c"]
            experiments = conn.execute("SELECT COUNT(*) as c FROM experiments").fetchone()["c"]
            running = conn.execute(
                "SELECT COUNT(*) as c FROM experiments WHERE status='running'"
            ).fetchone()["c"]
            props = conn.execute("SELECT COUNT(*) as c FROM config_propagations").fetchone()["c"]
            return {
                "quality_snapshots": snapshots,
                "experiments_total": experiments,
                "experiments_running": running,
                "propagations": props,
            }
