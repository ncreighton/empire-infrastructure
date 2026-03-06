"""Predictive Layer Codex — SQLite storage for anomalies, forecasts, decay."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "predictive.db"


class PredictiveCodex:
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
                CREATE TABLE IF NOT EXISTS anomaly_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT DEFAULT 'warning',
                    affected_sites TEXT,
                    description TEXT,
                    metrics TEXT,
                    detected_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_slug TEXT,
                    metric TEXT NOT NULL,
                    forecast_period TEXT NOT NULL,
                    predicted_value REAL,
                    confidence_low REAL,
                    confidence_high REAL,
                    actual_value REAL,
                    accuracy REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS decay_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_slug TEXT NOT NULL,
                    url TEXT,
                    title TEXT,
                    current_clicks INTEGER,
                    predicted_clicks_30d INTEGER,
                    decay_rate REAL,
                    days_until_zero INTEGER,
                    recommended_action TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_anomaly_type ON anomaly_events(anomaly_type);
                CREATE INDEX IF NOT EXISTS idx_forecast_site ON forecasts(site_slug);
                CREATE INDEX IF NOT EXISTS idx_decay_site ON decay_predictions(site_slug);
            """)

    def log_anomaly(self, anomaly_type: str, severity: str, affected_sites: List[str],
                     description: str, metrics: Dict = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO anomaly_events (anomaly_type, severity, affected_sites, "
                "description, metrics) VALUES (?, ?, ?, ?, ?)",
                (anomaly_type, severity, json.dumps(affected_sites),
                 description, json.dumps(metrics) if metrics else None)
            )
            return cur.lastrowid

    def log_forecast(self, site_slug: str, metric: str, period: str,
                      predicted: float, low: float = None, high: float = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO forecasts (site_slug, metric, forecast_period, "
                "predicted_value, confidence_low, confidence_high) VALUES (?, ?, ?, ?, ?, ?)",
                (site_slug, metric, period, predicted, low, high)
            )

    def log_decay(self, site_slug: str, url: str, title: str,
                   current_clicks: int, predicted_30d: int,
                   decay_rate: float, days_to_zero: int, action: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO decay_predictions (site_slug, url, title, "
                "current_clicks, predicted_clicks_30d, decay_rate, "
                "days_until_zero, recommended_action) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (site_slug, url, title, current_clicks, predicted_30d,
                 decay_rate, days_to_zero, action)
            )

    def update_forecast_actual(self, forecast_id: int, actual: float):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT predicted_value FROM forecasts WHERE id=?", (forecast_id,)
            ).fetchone()
            if row and row["predicted_value"]:
                accuracy = 1 - abs(actual - row["predicted_value"]) / max(abs(row["predicted_value"]), 1)
                conn.execute(
                    "UPDATE forecasts SET actual_value=?, accuracy=? WHERE id=?",
                    (actual, round(accuracy * 100, 1), forecast_id)
                )

    def get_anomalies(self, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM anomaly_events ORDER BY detected_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_decay_predictions(self, site_slug: str = None, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            if site_slug:
                rows = conn.execute(
                    "SELECT * FROM decay_predictions WHERE site_slug=? "
                    "ORDER BY decay_rate DESC LIMIT ?", (site_slug, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM decay_predictions ORDER BY decay_rate DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_forecast_accuracy(self) -> Dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT AVG(accuracy) as avg_acc, COUNT(*) as total "
                "FROM forecasts WHERE accuracy IS NOT NULL"
            ).fetchone()
            return {
                "average_accuracy": round(row["avg_acc"], 1) if row["avg_acc"] else None,
                "forecasts_with_actuals": row["total"],
            }

    def stats(self) -> Dict:
        with self._conn() as conn:
            anomalies = conn.execute("SELECT COUNT(*) as c FROM anomaly_events").fetchone()["c"]
            forecasts = conn.execute("SELECT COUNT(*) as c FROM forecasts").fetchone()["c"]
            decays = conn.execute("SELECT COUNT(*) as c FROM decay_predictions").fetchone()["c"]
            return {"anomalies": anomalies, "forecasts": forecasts, "decay_predictions": decays}
