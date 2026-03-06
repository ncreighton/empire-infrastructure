"""Cascade Engine Codex — SQLite storage for cascades and step tracking."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "data" / "cascade.db"


class CascadeCodex:
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
                CREATE TABLE IF NOT EXISTS cascades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_slug TEXT NOT NULL,
                    title TEXT NOT NULL,
                    template TEXT DEFAULT 'full',
                    status TEXT DEFAULT 'pending',
                    steps_total INTEGER DEFAULT 0,
                    steps_completed INTEGER DEFAULT 0,
                    steps_failed INTEGER DEFAULT 0,
                    output_data TEXT,
                    error TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS cascade_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cascade_id INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    step_order INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending',
                    input_data TEXT,
                    output_data TEXT,
                    error TEXT,
                    duration_ms INTEGER,
                    created_at TEXT DEFAULT (datetime('now')),
                    started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY (cascade_id) REFERENCES cascades(id)
                );

                CREATE TABLE IF NOT EXISTS cascade_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    steps TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_cascade_status ON cascades(status);
                CREATE INDEX IF NOT EXISTS idx_step_cascade ON cascade_steps(cascade_id);

                -- Seed default templates
                INSERT OR IGNORE INTO cascade_templates (name, steps, description) VALUES
                ('full', '["article","image","wordpress","video","social","product","internal_link","email"]',
                 'Full 8-step cascade: article -> image -> publish -> video -> social -> product -> links -> email'),
                ('article_only', '["article","image","wordpress"]',
                 'Quick publish: article -> image -> WordPress'),
                ('video_first', '["video","article","wordpress"]',
                 'Video-first: video -> article -> publish');
            """)

    def create_cascade(self, site_slug: str, title: str, template: str = "full") -> int:
        with self._conn() as conn:
            # Get template steps
            tpl = conn.execute(
                "SELECT steps FROM cascade_templates WHERE name=?", (template,)
            ).fetchone()
            steps = json.loads(tpl["steps"]) if tpl else ["article", "image", "wordpress"]

            cur = conn.execute(
                "INSERT INTO cascades (site_slug, title, template, steps_total) VALUES (?, ?, ?, ?)",
                (site_slug, title, template, len(steps))
            )
            cascade_id = cur.lastrowid

            for i, step_name in enumerate(steps):
                conn.execute(
                    "INSERT INTO cascade_steps (cascade_id, step_name, step_order) VALUES (?, ?, ?)",
                    (cascade_id, step_name, i)
                )

            return cascade_id

    def get_cascade(self, cascade_id: int) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM cascades WHERE id=?", (cascade_id,)).fetchone()
            if not row:
                return None
            cascade = dict(row)
            steps = conn.execute(
                "SELECT * FROM cascade_steps WHERE cascade_id=? ORDER BY step_order",
                (cascade_id,)
            ).fetchall()
            cascade["steps"] = [dict(s) for s in steps]
            return cascade

    def update_cascade_status(self, cascade_id: int, status: str, error: str = None):
        with self._conn() as conn:
            if status == "running":
                conn.execute(
                    "UPDATE cascades SET status=?, started_at=datetime('now') WHERE id=?",
                    (status, cascade_id)
                )
            elif status in ("completed", "failed"):
                conn.execute(
                    "UPDATE cascades SET status=?, completed_at=datetime('now'), error=? WHERE id=?",
                    (status, error, cascade_id)
                )
            else:
                conn.execute(
                    "UPDATE cascades SET status=? WHERE id=?", (status, cascade_id)
                )

    def update_step(self, step_id: int, status: str, output_data: Dict = None,
                    error: str = None, duration_ms: int = None):
        with self._conn() as conn:
            conn.execute(
                "UPDATE cascade_steps SET status=?, output_data=?, error=?, "
                "duration_ms=?, completed_at=datetime('now') WHERE id=?",
                (status, json.dumps(output_data) if output_data else None,
                 error, duration_ms, step_id)
            )
            # Update parent counts
            step = conn.execute("SELECT cascade_id FROM cascade_steps WHERE id=?", (step_id,)).fetchone()
            if step:
                cid = step["cascade_id"]
                done = conn.execute(
                    "SELECT COUNT(*) as c FROM cascade_steps WHERE cascade_id=? AND status='completed'",
                    (cid,)
                ).fetchone()["c"]
                failed = conn.execute(
                    "SELECT COUNT(*) as c FROM cascade_steps WHERE cascade_id=? AND status='failed'",
                    (cid,)
                ).fetchone()["c"]
                conn.execute(
                    "UPDATE cascades SET steps_completed=?, steps_failed=? WHERE id=?",
                    (done, failed, cid)
                )

    def get_recent(self, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cascades ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_templates(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM cascade_templates").fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["steps"] = json.loads(d["steps"])
                except (json.JSONDecodeError, TypeError):
                    d["steps"] = []
                result.append(d)
            return result

    def stats(self) -> Dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM cascades").fetchone()["c"]
            completed = conn.execute(
                "SELECT COUNT(*) as c FROM cascades WHERE status='completed'"
            ).fetchone()["c"]
            running = conn.execute(
                "SELECT COUNT(*) as c FROM cascades WHERE status='running'"
            ).fetchone()["c"]
            return {"total": total, "completed": completed, "running": running,
                    "pending": total - completed - running}
