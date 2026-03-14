"""VibeCodex — SQLite persistence for VibeCoder missions.

4 tables in the existing openclaw.db:
  vibe_missions        — mission lifecycle + metadata
  vibe_mission_steps   — step execution log
  vibe_code_changes    — file-level diffs
  vibe_project_registry — known projects + deploy configs

Pattern: openclaw/forge/platform_codex.py
All logic is algorithmic — zero LLM cost.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from openclaw.vibecoder.models import (
    CodeChange,
    DeployTarget,
    Mission,
    MissionScope,
    MissionStatus,
    MissionStep,
    ProjectInfo,
    StepStatus,
    StepType,
    EngineType,
    VibeDashboard,
)


# ─── Schema ──────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vibe_missions (
    mission_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    scope TEXT NOT NULL DEFAULT 'unknown',
    status TEXT NOT NULL DEFAULT 'queued',
    priority INTEGER NOT NULL DEFAULT 5,
    auto_deploy INTEGER NOT NULL DEFAULT 0,
    branch_name TEXT DEFAULT '',
    commit_hash TEXT DEFAULT '',
    pr_url TEXT DEFAULT '',
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    duration_seconds REAL DEFAULT 0.0,
    errors_json TEXT DEFAULT '[]',
    warnings_json TEXT DEFAULT '[]',
    enrichments_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS vibe_mission_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    step_type TEXT NOT NULL,
    description TEXT DEFAULT '',
    target_file TEXT DEFAULT '',
    engine TEXT DEFAULT 'algorithmic',
    command TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT DEFAULT '',
    output TEXT DEFAULT '',
    tokens_used INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    retry_count INTEGER DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    FOREIGN KEY (mission_id) REFERENCES vibe_missions(mission_id)
);

CREATE TABLE IF NOT EXISTS vibe_code_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    change_type TEXT NOT NULL DEFAULT 'modified',
    diff TEXT DEFAULT '',
    lines_added INTEGER DEFAULT 0,
    lines_removed INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (mission_id) REFERENCES vibe_missions(mission_id)
);

CREATE TABLE IF NOT EXISTS vibe_project_registry (
    project_id TEXT PRIMARY KEY,
    root_path TEXT NOT NULL,
    language TEXT DEFAULT 'python',
    framework TEXT DEFAULT '',
    package_manager TEXT DEFAULT '',
    has_tests INTEGER DEFAULT 0,
    has_ci INTEGER DEFAULT 0,
    has_git INTEGER DEFAULT 0,
    has_docker INTEGER DEFAULT 0,
    deploy_target TEXT DEFAULT 'none',
    deploy_config_json TEXT DEFAULT '{}',
    total_files INTEGER DEFAULT 0,
    total_lines INTEGER DEFAULT 0,
    dependencies_json TEXT DEFAULT '[]',
    registered_at TEXT NOT NULL,
    last_scanned_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_vibe_missions_status ON vibe_missions(status);
CREATE INDEX IF NOT EXISTS idx_vibe_missions_project ON vibe_missions(project_id);
CREATE INDEX IF NOT EXISTS idx_vibe_steps_mission ON vibe_mission_steps(mission_id);
CREATE INDEX IF NOT EXISTS idx_vibe_changes_mission ON vibe_code_changes(mission_id);
"""


class VibeCodex:
    """SQLite persistence for VibeCoder missions and projects.

    Uses the same DB file as PlatformCodex (openclaw.db) but with
    separate vibe_* prefixed tables.
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(
                Path(__file__).resolve().parent.parent.parent.parent
                / "data" / "openclaw.db"
            )
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ─── Mission CRUD ─────────────────────────────────────────────────────

    def create_mission(self, mission: Mission) -> None:
        """Insert a new mission."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO vibe_missions
                   (mission_id, project_id, title, description, scope, status,
                    priority, auto_deploy, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mission.mission_id, mission.project_id, mission.title,
                    mission.description, mission.scope.value, mission.status.value,
                    mission.priority, int(mission.auto_deploy),
                    (mission.created_at or datetime.now()).isoformat(),
                ),
            )

    def update_mission_status(
        self,
        mission_id: str,
        status: MissionStatus,
        **kwargs: Any,
    ) -> None:
        """Update mission status and optional fields."""
        sets = ["status = ?"]
        params: list[Any] = [status.value]

        if "branch_name" in kwargs:
            sets.append("branch_name = ?")
            params.append(kwargs["branch_name"])
        if "commit_hash" in kwargs:
            sets.append("commit_hash = ?")
            params.append(kwargs["commit_hash"])
        if "pr_url" in kwargs:
            sets.append("pr_url = ?")
            params.append(kwargs["pr_url"])
        if "total_tokens" in kwargs:
            sets.append("total_tokens = ?")
            params.append(kwargs["total_tokens"])
        if "total_cost_usd" in kwargs:
            sets.append("total_cost_usd = ?")
            params.append(kwargs["total_cost_usd"])
        if "duration_seconds" in kwargs:
            sets.append("duration_seconds = ?")
            params.append(kwargs["duration_seconds"])
        if "errors" in kwargs:
            sets.append("errors_json = ?")
            params.append(json.dumps(kwargs["errors"]))
        if "warnings" in kwargs:
            sets.append("warnings_json = ?")
            params.append(json.dumps(kwargs["warnings"]))
        if status == MissionStatus.EXECUTING and "started_at" not in kwargs:
            sets.append("started_at = ?")
            params.append(datetime.now().isoformat())
        if status == MissionStatus.PAUSED:
            sets.append("started_at = COALESCE(started_at, ?)")
            params.append(datetime.now().isoformat())
        if status in (MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.CANCELLED):
            sets.append("completed_at = ?")
            params.append(datetime.now().isoformat())

        params.append(mission_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE vibe_missions SET {', '.join(sets)} WHERE mission_id = ?",
                params,
            )

    def get_mission(self, mission_id: str) -> dict[str, Any] | None:
        """Get a mission by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM vibe_missions WHERE mission_id = ?", (mission_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_missions(
        self,
        status: str | None = None,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List missions with optional filters."""
        query = "SELECT * FROM vibe_missions"
        conditions = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_queued_missions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get missions waiting for execution, ordered by priority."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM vibe_missions
                   WHERE status = 'queued'
                   ORDER BY priority ASC, created_at ASC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_mission(self, mission_id: str) -> bool:
        """Cancel/delete a mission (only if queued or paused)."""
        with self._connect() as conn:
            result = conn.execute(
                """UPDATE vibe_missions SET status = 'cancelled', completed_at = ?
                   WHERE mission_id = ? AND status IN ('queued', 'paused')""",
                (datetime.now().isoformat(), mission_id),
            )
            return result.rowcount > 0

    # ─── Step CRUD ────────────────────────────────────────────────────────

    def log_step(self, mission_id: str, step: MissionStep) -> None:
        """Log a mission step execution."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO vibe_mission_steps
                   (mission_id, step_number, step_type, description, target_file,
                    engine, command, status, error_message, output,
                    tokens_used, cost_usd, retry_count, started_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mission_id, step.step_number, step.step_type.value,
                    step.description, step.target_file, step.engine.value,
                    step.command, step.status.value, step.error_message,
                    step.output[:5000] if step.output else "",
                    step.tokens_used, step.cost_usd, step.retry_count,
                    step.started_at.isoformat() if step.started_at else None,
                    step.completed_at.isoformat() if step.completed_at else None,
                ),
            )

    def get_steps(self, mission_id: str) -> list[dict[str, Any]]:
        """Get all steps for a mission."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM vibe_mission_steps WHERE mission_id = ? ORDER BY step_number",
                (mission_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─── Code Changes ─────────────────────────────────────────────────────

    def log_change(self, mission_id: str, change: CodeChange) -> None:
        """Log a file change."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO vibe_code_changes
                   (mission_id, file_path, change_type, diff, lines_added,
                    lines_removed, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    mission_id, change.file_path, change.change_type,
                    change.diff[:10000] if change.diff else "",
                    change.lines_added, change.lines_removed,
                    datetime.now().isoformat(),
                ),
            )

    def get_changes(self, mission_id: str) -> list[dict[str, Any]]:
        """Get all code changes for a mission."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM vibe_code_changes WHERE mission_id = ? ORDER BY id",
                (mission_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─── Project Registry ─────────────────────────────────────────────────

    def register_project(self, info: ProjectInfo) -> None:
        """Register or update a project."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO vibe_project_registry
                   (project_id, root_path, language, framework, package_manager,
                    has_tests, has_ci, has_git, has_docker, deploy_target,
                    deploy_config_json, total_files, total_lines,
                    dependencies_json, registered_at, last_scanned_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    info.project_id, info.root_path, info.language,
                    info.framework, info.package_manager,
                    int(info.has_tests), int(info.has_ci),
                    int(info.has_git), int(info.has_docker),
                    info.deploy_target.value,
                    json.dumps(info.deploy_config),
                    info.total_files, info.total_lines,
                    json.dumps(info.dependencies),
                    datetime.now().isoformat(),
                    (info.scanned_at or datetime.now()).isoformat(),
                ),
            )

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Get a registered project by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM vibe_project_registry WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_all_projects(self) -> list[dict[str, Any]]:
        """Get all registered projects."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM vibe_project_registry ORDER BY project_id"
            ).fetchall()
            return [dict(r) for r in rows]

    def project_to_info(self, row: dict[str, Any]) -> ProjectInfo:
        """Convert a DB row to a ProjectInfo dataclass."""
        return ProjectInfo(
            project_id=row["project_id"],
            root_path=row["root_path"],
            language=row.get("language", "python"),
            framework=row.get("framework", ""),
            package_manager=row.get("package_manager", ""),
            has_tests=bool(row.get("has_tests", 0)),
            has_ci=bool(row.get("has_ci", 0)),
            has_git=bool(row.get("has_git", 0)),
            has_docker=bool(row.get("has_docker", 0)),
            deploy_target=DeployTarget(row.get("deploy_target", "none")),
            deploy_config=json.loads(row.get("deploy_config_json", "{}")),
            total_files=row.get("total_files", 0),
            total_lines=row.get("total_lines", 0),
            dependencies=json.loads(row.get("dependencies_json", "[]")),
        )

    # ─── Dashboard Stats ──────────────────────────────────────────────────

    def get_dashboard(self) -> VibeDashboard:
        """Get aggregate dashboard statistics."""
        with self._connect() as conn:
            # Totals
            total = conn.execute("SELECT COUNT(*) FROM vibe_missions").fetchone()[0]
            by_status = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) as cnt FROM vibe_missions GROUP BY status"
            ).fetchall():
                by_status[row["status"]] = row["cnt"]

            by_scope = {}
            for row in conn.execute(
                "SELECT scope, COUNT(*) as cnt FROM vibe_missions GROUP BY scope"
            ).fetchall():
                by_scope[row["scope"]] = row["cnt"]

            # Cost totals
            cost_row = conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) as tokens, "
                "COALESCE(SUM(total_cost_usd), 0) as cost, "
                "COALESCE(AVG(duration_seconds), 0) as avg_dur "
                "FROM vibe_missions WHERE status = 'completed'"
            ).fetchone()

            # Recent missions
            recent = conn.execute(
                """SELECT mission_id, project_id, title, scope, status,
                          total_cost_usd, duration_seconds, created_at
                   FROM vibe_missions ORDER BY created_at DESC LIMIT 10"""
            ).fetchall()

            projects = conn.execute(
                "SELECT COUNT(*) FROM vibe_project_registry"
            ).fetchone()[0]

            return VibeDashboard(
                total_missions=total,
                completed_missions=by_status.get("completed", 0),
                failed_missions=by_status.get("failed", 0),
                queued_missions=by_status.get("queued", 0),
                total_tokens_used=cost_row["tokens"],
                total_cost_usd=cost_row["cost"],
                avg_duration_seconds=cost_row["avg_dur"],
                missions_by_scope=by_scope,
                missions_by_status=by_status,
                recent_missions=[dict(r) for r in recent],
                registered_projects=projects,
            )
