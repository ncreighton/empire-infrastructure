"""Project Launcher Codex — SQLite storage for launch proposals and steps."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "launcher.db"


class LauncherCodex:
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
                CREATE TABLE IF NOT EXISTS launch_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    niche TEXT NOT NULL,
                    site_slug TEXT,
                    domain_suggestion TEXT,
                    status TEXT DEFAULT 'researching',
                    research_data TEXT,
                    roi_projection TEXT,
                    decision TEXT,
                    decision_reason TEXT,
                    similar_project TEXT,
                    brand_config TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    decided_at TEXT,
                    launched_at TEXT
                );

                CREATE TABLE IF NOT EXISTS launch_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_id INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    step_order INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending',
                    output_data TEXT,
                    error TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    completed_at TEXT,
                    FOREIGN KEY (proposal_id) REFERENCES launch_proposals(id)
                );

                CREATE INDEX IF NOT EXISTS idx_proposal_status ON launch_proposals(status);
            """)

    def create_proposal(self, niche: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO launch_proposals (niche) VALUES (?)", (niche,)
            )
            proposal_id = cur.lastrowid

            # Create launch steps
            steps = [
                "research", "roi_analysis", "decision",
                "genome_clone", "site_setup", "brand_generation",
                "manifest_creation", "initial_content_plan", "first_cascade",
            ]
            for i, step in enumerate(steps):
                conn.execute(
                    "INSERT INTO launch_steps (proposal_id, step_name, step_order) "
                    "VALUES (?, ?, ?)", (proposal_id, step, i)
                )

            return proposal_id

    def update_proposal(self, proposal_id: int, **kwargs):
        with self._conn() as conn:
            for key, value in kwargs.items():
                if key in ("status", "decision", "decision_reason", "site_slug",
                           "domain_suggestion", "similar_project"):
                    conn.execute(
                        f"UPDATE launch_proposals SET {key}=? WHERE id=?",
                        (value, proposal_id)
                    )
                elif key in ("research_data", "roi_projection", "brand_config"):
                    conn.execute(
                        f"UPDATE launch_proposals SET {key}=? WHERE id=?",
                        (json.dumps(value), proposal_id)
                    )

    def update_step(self, proposal_id: int, step_name: str, status: str,
                     output: Dict = None, error: str = None):
        with self._conn() as conn:
            conn.execute(
                "UPDATE launch_steps SET status=?, output_data=?, error=?, "
                "completed_at=datetime('now') "
                "WHERE proposal_id=? AND step_name=?",
                (status, json.dumps(output) if output else None,
                 error, proposal_id, step_name)
            )

    def get_proposal(self, proposal_id: int) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM launch_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            if not row:
                return None
            proposal = dict(row)
            for field in ("research_data", "roi_projection", "brand_config"):
                try:
                    proposal[field] = json.loads(proposal.get(field, "null"))
                except (json.JSONDecodeError, TypeError):
                    pass
            steps = conn.execute(
                "SELECT * FROM launch_steps WHERE proposal_id=? ORDER BY step_order",
                (proposal_id,)
            ).fetchall()
            proposal["steps"] = [dict(s) for s in steps]
            return proposal

    def get_proposals(self, status: str = None) -> List[Dict]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM launch_proposals WHERE status=? ORDER BY created_at DESC",
                    (status,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM launch_proposals ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> Dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM launch_proposals").fetchone()["c"]
            launched = conn.execute(
                "SELECT COUNT(*) as c FROM launch_proposals WHERE status='launched'"
            ).fetchone()["c"]
            declined = conn.execute(
                "SELECT COUNT(*) as c FROM launch_proposals WHERE decision='skip'"
            ).fetchone()["c"]
            return {"total_proposals": total, "launched": launched, "declined": declined}
