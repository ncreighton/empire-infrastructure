"""PlatformCodex -- SQLite persistence for accounts, credentials, logs.

Part of the OpenClaw FORGE intelligence layer. Follows the PracticeCodex
pattern: SQLite-backed persistence for all signup activity, profile content,
credentials (Fernet-encrypted), and CAPTCHA encounters.

5 tables:
    accounts        — platform_id, name, status, username, profile_url, timestamps
    credentials     — platform_id, encrypted_data (Fernet), updated_at
    signup_log      — step-by-step execution log per platform
    profile_content — stored profile JSON with sentinel scores
    captcha_log     — CAPTCHA encounter history

All logic is algorithmic -- zero LLM cost.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from openclaw.models import (
    AccountStatus,
    Alert,
    AlertSeverity,
    CheckResult,
    CronJob,
    CronStatus,
    HealthCheck,
    HeartbeatTier,
    ProfileContent,
    SentinelScore,
    SignupStep,
    StepStatus,
    CaptchaType,
)
from openclaw.security.credential_store import CredentialStore


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    platform_id TEXT PRIMARY KEY,
    platform_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'not_started',
    username TEXT DEFAULT '',
    profile_url TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS credentials (
    platform_id TEXT PRIMARY KEY,
    encrypted_data TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (platform_id) REFERENCES accounts(platform_id)
);

CREATE TABLE IF NOT EXISTS signup_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    step_type TEXT NOT NULL,
    status TEXT NOT NULL,
    description TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    screenshot_path TEXT DEFAULT '',
    started_at TEXT,
    completed_at TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (platform_id) REFERENCES accounts(platform_id)
);

CREATE TABLE IF NOT EXISTS profile_content (
    platform_id TEXT PRIMARY KEY,
    content_json TEXT NOT NULL,
    sentinel_score REAL DEFAULT 0.0,
    grade TEXT DEFAULT 'F',
    updated_at TEXT NOT NULL,
    FOREIGN KEY (platform_id) REFERENCES accounts(platform_id)
);

CREATE TABLE IF NOT EXISTS captcha_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform_id TEXT NOT NULL,
    captcha_type TEXT NOT NULL,
    auto_solved INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL DEFAULT 0.0,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (platform_id) REFERENCES accounts(platform_id)
);

CREATE INDEX IF NOT EXISTS idx_signup_log_platform ON signup_log(platform_id);
CREATE INDEX IF NOT EXISTS idx_signup_log_timestamp ON signup_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_captcha_log_platform ON captcha_log(platform_id);
CREATE INDEX IF NOT EXISTS idx_captcha_log_timestamp ON captcha_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_accounts_status ON accounts(status);

-- Daemon tables (heartbeat, alerts, cron, action log)

CREATE TABLE IF NOT EXISTS heartbeat_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_name TEXT NOT NULL,
    tier TEXT NOT NULL,
    result TEXT NOT NULL,
    message TEXT DEFAULT '',
    details_json TEXT DEFAULT '{}',
    duration_ms REAL DEFAULT 0.0,
    checked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    severity TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT DEFAULT '{}',
    content_hash TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    delivered INTEGER DEFAULT 0,
    suppressed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cron_jobs (
    job_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schedule TEXT NOT NULL,
    action TEXT NOT NULL,
    params_json TEXT DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active',
    last_run TEXT,
    next_run TEXT,
    run_count INTEGER DEFAULT 0,
    fail_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cron_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    success INTEGER DEFAULT 0,
    result_json TEXT DEFAULT '{}',
    error TEXT DEFAULT '',
    FOREIGN KEY (job_id) REFERENCES cron_jobs(job_id)
);

CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT NOT NULL,
    target TEXT DEFAULT '',
    description TEXT DEFAULT '',
    result TEXT DEFAULT '',
    autonomous INTEGER DEFAULT 1,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_heartbeat_tier ON heartbeat_results(tier);
CREATE INDEX IF NOT EXISTS idx_heartbeat_time ON heartbeat_results(checked_at);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_hash ON alerts(content_hash);
CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_cron_next ON cron_jobs(next_run);
CREATE INDEX IF NOT EXISTS idx_cron_history_job ON cron_history(job_id);
CREATE INDEX IF NOT EXISTS idx_action_log_time ON action_log(timestamp);

-- Step model routing tables (intelligent Haiku/Sonnet routing)

CREATE TABLE IF NOT EXISTS step_model_promotions (
    platform_id TEXT NOT NULL,
    step_type TEXT NOT NULL,
    promoted_at TEXT NOT NULL,
    reason TEXT DEFAULT '',
    PRIMARY KEY (platform_id, step_type)
);

CREATE TABLE IF NOT EXISTS step_cost_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform_id TEXT NOT NULL,
    step_type TEXT NOT NULL,
    model_id TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    success INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_step_cost_time ON step_cost_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_step_cost_platform ON step_cost_log(platform_id);
"""


# =========================================================================== #
#  PlatformCodex                                                               #
# =========================================================================== #


class PlatformCodex:
    """SQLite persistence for OpenClaw accounts, credentials, and logs.

    Provides methods to track account status, store encrypted credentials,
    log signup steps, persist profile content with quality scores, and
    record CAPTCHA encounters.

    Usage::

        codex = PlatformCodex()

        # Track an account
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE, "myuser")

        # Store credentials (encrypted)
        codex.store_credentials("gumroad", {"email": "a@b.com", "password": "secret"})

        # Retrieve stats
        stats = codex.get_stats()
        print(stats["total_accounts"])   # 1
        print(stats["active_accounts"])  # 1
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or self._default_db_path()
        self.credential_store = CredentialStore()
        self._init_db()

    @staticmethod
    def _default_db_path() -> str:
        """Default database location: openclaw-agent/data/openclaw.db"""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "openclaw.db",
        )

    def _init_db(self) -> None:
        """Create all tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """Return a new database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ================================================================== #
    #  Account methods                                                     #
    # ================================================================== #

    def upsert_account(
        self,
        platform_id: str,
        platform_name: str,
        status: AccountStatus,
        username: str = "",
        profile_url: str = "",
    ) -> None:
        """Insert or update an account record.

        Args:
            platform_id: Unique platform identifier.
            platform_name: Human-readable platform name.
            status: Current account status.
            username: The username on the platform.
            profile_url: URL to the public profile.
        """
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO accounts (platform_id, platform_name, status, username, profile_url, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(platform_id) DO UPDATE SET
                    platform_name = excluded.platform_name,
                    status = excluded.status,
                    username = CASE WHEN excluded.username != '' THEN excluded.username ELSE accounts.username END,
                    profile_url = CASE WHEN excluded.profile_url != '' THEN excluded.profile_url ELSE accounts.profile_url END,
                    updated_at = excluded.updated_at
                """,
                (platform_id, platform_name, status.value, username, profile_url, now, now),
            )

    def get_account(self, platform_id: str) -> dict[str, Any] | None:
        """Get a single account record.

        Args:
            platform_id: The platform identifier.

        Returns:
            A dict with account fields, or ``None`` if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE platform_id = ?",
                (platform_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_accounts_by_status(self, status: AccountStatus) -> list[dict[str, Any]]:
        """Get all accounts with a given status.

        Args:
            status: The account status to filter by.

        Returns:
            A list of account dicts.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM accounts WHERE status = ? ORDER BY updated_at DESC",
                (status.value,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_accounts(self) -> list[dict[str, Any]]:
        """Get all account records.

        Returns:
            A list of account dicts sorted by updated_at descending.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM accounts ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def update_account_status(
        self, platform_id: str, status: AccountStatus
    ) -> None:
        """Update only the status field of an existing account.

        Args:
            platform_id: The platform identifier.
            status: The new account status.
        """
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE accounts SET status = ?, updated_at = ? WHERE platform_id = ?",
                (status.value, now, platform_id),
            )

    def delete_account(self, platform_id: str) -> None:
        """Delete an account and all related records.

        Args:
            platform_id: The platform identifier.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM captcha_log WHERE platform_id = ?", (platform_id,))
            conn.execute("DELETE FROM signup_log WHERE platform_id = ?", (platform_id,))
            conn.execute("DELETE FROM profile_content WHERE platform_id = ?", (platform_id,))
            conn.execute("DELETE FROM credentials WHERE platform_id = ?", (platform_id,))
            conn.execute("DELETE FROM accounts WHERE platform_id = ?", (platform_id,))

    # ================================================================== #
    #  Credential methods (encrypted)                                      #
    # ================================================================== #

    def store_credentials(self, platform_id: str, credentials: dict[str, Any]) -> None:
        """Store encrypted credentials for a platform.

        Args:
            platform_id: The platform identifier.
            credentials: A dict of credential data (e.g., email, password, tokens).
                Will be encrypted with Fernet before storage.
        """
        encrypted = self.credential_store.encrypt_dict(credentials)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO credentials (platform_id, encrypted_data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(platform_id) DO UPDATE SET
                    encrypted_data = excluded.encrypted_data,
                    updated_at = excluded.updated_at
                """,
                (platform_id, encrypted, now),
            )

    def get_credentials(self, platform_id: str) -> dict[str, Any] | None:
        """Retrieve and decrypt credentials for a platform.

        Args:
            platform_id: The platform identifier.

        Returns:
            A dict of decrypted credential data, or ``None`` if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT encrypted_data FROM credentials WHERE platform_id = ?",
                (platform_id,),
            ).fetchone()
            if not row:
                return None
            return self.credential_store.decrypt_dict(row["encrypted_data"])

    def delete_credentials(self, platform_id: str) -> None:
        """Delete stored credentials for a platform.

        Args:
            platform_id: The platform identifier.
        """
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM credentials WHERE platform_id = ?",
                (platform_id,),
            )

    # ================================================================== #
    #  Signup log                                                          #
    # ================================================================== #

    def log_step(self, platform_id: str, step: SignupStep) -> None:
        """Log a signup step execution.

        Args:
            platform_id: The platform identifier.
            step: The SignupStep dataclass with execution details.
        """
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signup_log
                    (platform_id, step_number, step_type, status, description,
                     error_message, screenshot_path, started_at, completed_at, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    platform_id,
                    step.step_number,
                    step.step_type.value,
                    step.status.value,
                    step.description,
                    step.error_message,
                    step.screenshot_path,
                    step.started_at.isoformat() if step.started_at else "",
                    step.completed_at.isoformat() if step.completed_at else "",
                    now,
                ),
            )

    def get_signup_log(self, platform_id: str) -> list[dict[str, Any]]:
        """Get the signup step log for a platform.

        Args:
            platform_id: The platform identifier.

        Returns:
            A list of step log dicts ordered by step_number.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM signup_log
                WHERE platform_id = ?
                ORDER BY step_number ASC, timestamp ASC
                """,
                (platform_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_failed_steps(self, platform_id: str) -> list[dict[str, Any]]:
        """Get only failed steps for a platform.

        Args:
            platform_id: The platform identifier.

        Returns:
            A list of failed step log dicts.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM signup_log
                WHERE platform_id = ? AND status = ?
                ORDER BY timestamp DESC
                """,
                (platform_id, StepStatus.FAILED.value),
            ).fetchall()
            return [dict(r) for r in rows]

    # ================================================================== #
    #  Profile content                                                     #
    # ================================================================== #

    def store_profile(
        self, content: ProfileContent, score: SentinelScore
    ) -> None:
        """Store profile content and its sentinel score.

        Args:
            content: The ProfileContent dataclass to persist.
            score: The SentinelScore from ProfileSentinel evaluation.
        """
        now = datetime.now().isoformat()

        # Serialize ProfileContent to JSON
        content_dict = {
            "platform_id": content.platform_id,
            "username": content.username,
            "display_name": content.display_name,
            "email": content.email,
            "bio": content.bio,
            "tagline": content.tagline,
            "description": content.description,
            "website_url": content.website_url,
            "avatar_path": content.avatar_path,
            "banner_path": content.banner_path,
            "social_links": content.social_links,
            "custom_fields": content.custom_fields,
            "seo_keywords": content.seo_keywords,
            "generated_at": content.generated_at.isoformat() if content.generated_at else "",
        }

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO profile_content (platform_id, content_json, sentinel_score, grade, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(platform_id) DO UPDATE SET
                    content_json = excluded.content_json,
                    sentinel_score = excluded.sentinel_score,
                    grade = excluded.grade,
                    updated_at = excluded.updated_at
                """,
                (
                    content.platform_id,
                    json.dumps(content_dict),
                    score.total_score,
                    score.grade.value,
                    now,
                ),
            )

    def get_profile(self, platform_id: str) -> dict[str, Any] | None:
        """Retrieve stored profile content for a platform.

        Args:
            platform_id: The platform identifier.

        Returns:
            A dict with ``content`` (parsed JSON), ``sentinel_score``,
            ``grade``, and ``updated_at``, or ``None`` if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profile_content WHERE platform_id = ?",
                (platform_id,),
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["content"] = json.loads(result.pop("content_json"))
            return result

    def get_all_profiles(self) -> list[dict[str, Any]]:
        """Retrieve all stored profiles with scores.

        Returns:
            A list of profile dicts sorted by sentinel_score descending.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM profile_content ORDER BY sentinel_score DESC"
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["content"] = json.loads(d.pop("content_json"))
                results.append(d)
            return results

    # ================================================================== #
    #  CAPTCHA log                                                         #
    # ================================================================== #

    def log_captcha(
        self,
        platform_id: str,
        captcha_type: str | CaptchaType,
        auto_solved: bool,
        duration: float,
    ) -> None:
        """Log a CAPTCHA encounter.

        Args:
            platform_id: The platform identifier.
            captcha_type: The type of CAPTCHA encountered.
            auto_solved: Whether it was solved automatically (True) or manually (False).
            duration: Time in seconds to solve the CAPTCHA.
        """
        now = datetime.now().isoformat()
        type_value = captcha_type.value if isinstance(captcha_type, CaptchaType) else captcha_type
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO captcha_log (platform_id, captcha_type, auto_solved, duration_seconds, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (platform_id, type_value, 1 if auto_solved else 0, duration, now),
            )

    def get_captcha_log(self, platform_id: str | None = None) -> list[dict[str, Any]]:
        """Get CAPTCHA encounter history.

        Args:
            platform_id: Optional platform filter. If ``None``, returns all.

        Returns:
            A list of CAPTCHA log dicts ordered by timestamp descending.
        """
        with self._connect() as conn:
            if platform_id:
                rows = conn.execute(
                    "SELECT * FROM captcha_log WHERE platform_id = ? ORDER BY timestamp DESC",
                    (platform_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM captcha_log ORDER BY timestamp DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    # ================================================================== #
    #  Stats                                                               #
    # ================================================================== #

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics across all tracked platforms.

        Returns:
            A dict with keys:
            - total_accounts: int
            - active_accounts: int
            - pending_signups: int
            - failed_signups: int
            - completed_profiles: int
            - avg_sentinel_score: float
            - total_steps_logged: int
            - total_captchas: int
            - captcha_auto_solve_rate: float (0.0-100.0)
            - accounts_by_status: dict[str, int]
        """
        with self._connect() as conn:
            # Total accounts
            total = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]

            # By status
            status_rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM accounts GROUP BY status"
            ).fetchall()
            accounts_by_status = {row["status"]: row["cnt"] for row in status_rows}

            active = accounts_by_status.get(AccountStatus.ACTIVE.value, 0)
            pending = (
                accounts_by_status.get(AccountStatus.SIGNUP_IN_PROGRESS.value, 0)
                + accounts_by_status.get(AccountStatus.EMAIL_VERIFICATION_PENDING.value, 0)
                + accounts_by_status.get(AccountStatus.PLANNED.value, 0)
            )
            failed = accounts_by_status.get(AccountStatus.SIGNUP_FAILED.value, 0)

            # Profile stats
            profile_stats = conn.execute(
                "SELECT COUNT(*) as cnt, AVG(sentinel_score) as avg_score FROM profile_content"
            ).fetchone()
            completed_profiles = profile_stats["cnt"] or 0
            avg_score = round(profile_stats["avg_score"] or 0.0, 1)

            # Step log stats
            total_steps = conn.execute("SELECT COUNT(*) FROM signup_log").fetchone()[0]

            # CAPTCHA stats
            captcha_stats = conn.execute(
                "SELECT COUNT(*) as total, SUM(auto_solved) as auto_solved FROM captcha_log"
            ).fetchone()
            total_captchas = captcha_stats["total"] or 0
            auto_solved = captcha_stats["auto_solved"] or 0
            captcha_rate = (
                round((auto_solved / total_captchas) * 100, 1)
                if total_captchas > 0
                else 0.0
            )

        return {
            "total_accounts": total,
            "active_accounts": active,
            "pending_signups": pending,
            "failed_signups": failed,
            "completed_profiles": completed_profiles,
            "avg_sentinel_score": avg_score,
            "total_steps_logged": total_steps,
            "total_captchas": total_captchas,
            "captcha_auto_solve_rate": captcha_rate,
            "accounts_by_status": accounts_by_status,
        }

    def get_recent_activity(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the most recent activity across all tables.

        Combines recent signup steps, CAPTCHA events, and account updates
        into a single timeline sorted by timestamp.

        Args:
            limit: Maximum number of activity items to return.

        Returns:
            A list of activity dicts with ``type``, ``platform_id``,
            ``summary``, and ``timestamp`` keys.
        """
        activities: list[dict[str, Any]] = []

        with self._connect() as conn:
            # Recent signup steps
            step_rows = conn.execute(
                """
                SELECT platform_id, step_number, step_type, status, description, timestamp
                FROM signup_log ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            for row in step_rows:
                activities.append({
                    "type": "signup_step",
                    "platform_id": row["platform_id"],
                    "summary": (
                        f"Step {row['step_number']} ({row['step_type']}): "
                        f"{row['status']} - {row['description']}"
                    ),
                    "timestamp": row["timestamp"],
                })

            # Recent CAPTCHA events
            captcha_rows = conn.execute(
                """
                SELECT platform_id, captcha_type, auto_solved, duration_seconds, timestamp
                FROM captcha_log ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            for row in captcha_rows:
                solved_str = "auto-solved" if row["auto_solved"] else "manual"
                activities.append({
                    "type": "captcha",
                    "platform_id": row["platform_id"],
                    "summary": (
                        f"CAPTCHA ({row['captcha_type']}): {solved_str} "
                        f"in {row['duration_seconds']:.1f}s"
                    ),
                    "timestamp": row["timestamp"],
                })

            # Recent account updates
            account_rows = conn.execute(
                """
                SELECT platform_id, platform_name, status, updated_at
                FROM accounts ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            for row in account_rows:
                activities.append({
                    "type": "account_update",
                    "platform_id": row["platform_id"],
                    "summary": f"{row['platform_name']}: status={row['status']}",
                    "timestamp": row["updated_at"],
                })

        # Sort all activities by timestamp descending and limit
        activities.sort(key=lambda a: a["timestamp"], reverse=True)
        return activities[:limit]

    # ================================================================== #
    #  Heartbeat results                                                   #
    # ================================================================== #

    def log_health_check(self, check: HealthCheck) -> None:
        """Persist a health check result."""
        checked_at = (check.checked_at or datetime.now()).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO heartbeat_results
                    (check_name, tier, result, message, details_json, duration_ms, checked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    check.name,
                    check.tier.value,
                    check.result.value,
                    check.message,
                    json.dumps(check.details),
                    check.duration_ms,
                    checked_at,
                ),
            )

    def get_recent_checks(
        self,
        tier: HeartbeatTier | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get recent health check results, optionally filtered by tier."""
        with self._connect() as conn:
            if tier:
                rows = conn.execute(
                    "SELECT * FROM heartbeat_results WHERE tier = ? ORDER BY checked_at DESC LIMIT ?",
                    (tier.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM heartbeat_results ORDER BY checked_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["details"] = json.loads(d.pop("details_json", "{}"))
                results.append(d)
            return results

    def get_latest_checks(self) -> dict[str, dict[str, Any]]:
        """Get the latest check result for each check_name."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT h.* FROM heartbeat_results h
                INNER JOIN (
                    SELECT check_name, MAX(checked_at) as max_time
                    FROM heartbeat_results GROUP BY check_name
                ) latest ON h.check_name = latest.check_name
                    AND h.checked_at = latest.max_time
                ORDER BY h.check_name
                """
            ).fetchall()
            results = {}
            for row in rows:
                d = dict(row)
                d["details"] = json.loads(d.pop("details_json", "{}"))
                results[d["check_name"]] = d
            return results

    # ================================================================== #
    #  Alerts                                                              #
    # ================================================================== #

    def insert_alert(self, alert: Alert) -> None:
        """Persist an alert record."""
        created_at = (alert.created_at or datetime.now()).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alerts
                    (alert_id, severity, source, title, message,
                     details_json, content_hash, created_at, delivered, suppressed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    alert.severity.value,
                    alert.source,
                    alert.title,
                    alert.message,
                    json.dumps(alert.details),
                    alert.content_hash,
                    created_at,
                    1 if alert.delivered else 0,
                    1 if alert.suppressed else 0,
                ),
            )

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get recent alerts with optional filtering."""
        query = "SELECT * FROM alerts WHERE 1=1"
        params: list[Any] = []
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["details"] = json.loads(d.pop("details_json", "{}"))
                results.append(d)
            return results

    def alert_hash_exists(self, content_hash: str, window_hours: int = 6) -> bool:
        """Check if an alert with the same content_hash exists within the dedup window."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(hours=window_hours)).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE content_hash = ? AND created_at > ?",
                (content_hash, cutoff),
            ).fetchone()
            return (row[0] or 0) > 0

    def get_alert_count_today(self, source: str) -> int:
        """Count alerts from a source in the current day."""
        today = datetime.now().strftime("%Y-%m-%d")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE source = ? AND created_at >= ? AND delivered = 1",
                (source, today),
            ).fetchone()
            return row[0] or 0

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged (delivered)."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE alerts SET delivered = 1 WHERE alert_id = ?",
                (alert_id,),
            )
            return cursor.rowcount > 0

    def get_suppressed_alerts(self) -> list[dict[str, Any]]:
        """Get alerts that were suppressed (for later flushing)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE suppressed = 1 AND delivered = 0 ORDER BY created_at ASC"
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["details"] = json.loads(d.pop("details_json", "{}"))
                results.append(d)
            return results

    def get_alert_stats(self) -> dict[str, Any]:
        """Get alert statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            delivered = conn.execute("SELECT COUNT(*) FROM alerts WHERE delivered = 1").fetchone()[0]
            suppressed = conn.execute("SELECT COUNT(*) FROM alerts WHERE suppressed = 1").fetchone()[0]
            by_severity = {}
            for row in conn.execute("SELECT severity, COUNT(*) as cnt FROM alerts GROUP BY severity").fetchall():
                by_severity[row["severity"]] = row["cnt"]
            return {
                "total_alerts": total,
                "delivered": delivered,
                "suppressed": suppressed,
                "by_severity": by_severity,
            }

    # ================================================================== #
    #  Cron jobs                                                           #
    # ================================================================== #

    def upsert_cron_job(self, job: CronJob) -> None:
        """Insert or update a cron job."""
        now = datetime.now().isoformat()
        created_at = (job.created_at or datetime.now()).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cron_jobs
                    (job_id, name, schedule, action, params_json, status,
                     last_run, next_run, run_count, fail_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    name = excluded.name,
                    schedule = excluded.schedule,
                    action = excluded.action,
                    params_json = excluded.params_json,
                    status = excluded.status,
                    last_run = excluded.last_run,
                    next_run = excluded.next_run,
                    run_count = excluded.run_count,
                    fail_count = excluded.fail_count
                """,
                (
                    job.job_id,
                    job.name,
                    job.schedule,
                    job.action,
                    json.dumps(job.params),
                    job.status.value,
                    job.last_run.isoformat() if job.last_run else None,
                    job.next_run.isoformat() if job.next_run else None,
                    job.run_count,
                    job.fail_count,
                    created_at,
                ),
            )

    def get_cron_job(self, job_id: str) -> CronJob | None:
        """Get a single cron job by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cron_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_cron_job(dict(row))

    def get_all_cron_jobs(self) -> list[CronJob]:
        """Get all cron jobs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cron_jobs ORDER BY next_run ASC"
            ).fetchall()
            return [self._row_to_cron_job(dict(r)) for r in rows]

    def get_due_cron_jobs(self) -> list[CronJob]:
        """Get cron jobs whose next_run <= now and status == active."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cron_jobs WHERE status = ? AND next_run <= ? ORDER BY next_run ASC",
                (CronStatus.ACTIVE.value, now),
            ).fetchall()
            return [self._row_to_cron_job(dict(r)) for r in rows]

    def update_cron_status(self, job_id: str, status: CronStatus) -> bool:
        """Update a cron job's status."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE cron_jobs SET status = ? WHERE job_id = ?",
                (status.value, job_id),
            )
            return cursor.rowcount > 0

    def update_cron_after_run(
        self, job_id: str, next_run: datetime, success: bool
    ) -> None:
        """Update a cron job after execution."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            if success:
                conn.execute(
                    "UPDATE cron_jobs SET last_run = ?, next_run = ?, run_count = run_count + 1 WHERE job_id = ?",
                    (now, next_run.isoformat(), job_id),
                )
            else:
                conn.execute(
                    "UPDATE cron_jobs SET last_run = ?, next_run = ?, run_count = run_count + 1, fail_count = fail_count + 1 WHERE job_id = ?",
                    (now, next_run.isoformat(), job_id),
                )

    def log_cron_run(
        self,
        job_id: str,
        started_at: datetime,
        completed_at: datetime | None = None,
        success: bool = False,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        """Log a cron job execution to history."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cron_history (job_id, started_at, completed_at, success, result_json, error)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    started_at.isoformat(),
                    completed_at.isoformat() if completed_at else None,
                    1 if success else 0,
                    json.dumps(result or {}),
                    error,
                ),
            )

    def get_cron_history(self, job_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get execution history for a cron job."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cron_history WHERE job_id = ? ORDER BY started_at DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["result"] = json.loads(d.pop("result_json", "{}"))
                results.append(d)
            return results

    @staticmethod
    def _row_to_cron_job(d: dict[str, Any]) -> CronJob:
        """Convert a database row dict to a CronJob dataclass."""
        return CronJob(
            job_id=d["job_id"],
            name=d["name"],
            schedule=d["schedule"],
            action=d["action"],
            params=json.loads(d.get("params_json", "{}")),
            status=CronStatus(d["status"]),
            last_run=datetime.fromisoformat(d["last_run"]) if d.get("last_run") else None,
            next_run=datetime.fromisoformat(d["next_run"]) if d.get("next_run") else None,
            run_count=d.get("run_count", 0),
            fail_count=d.get("fail_count", 0),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else None,
        )

    # ================================================================== #
    #  Action log                                                          #
    # ================================================================== #

    def log_action(
        self,
        action_type: str,
        target: str = "",
        description: str = "",
        result: str = "",
        autonomous: bool = True,
    ) -> None:
        """Log an action taken by the daemon."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO action_log (action_type, target, description, result, autonomous, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (action_type, target, description, result, 1 if autonomous else 0, now),
            )

    def get_action_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent action log entries."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM action_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ================================================================== #
    #  Step model routing (Haiku/Sonnet cost optimization)                 #
    # ================================================================== #

    def get_step_promotions(self, platform_id: str) -> dict[str, str]:
        """Get all promoted (platform, step_type) pairs.

        Returns dict of step_type -> promoted_at for the given platform.
        Only returns non-expired promotions (< 7 days old).
        """
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT step_type, promoted_at FROM step_model_promotions "
                "WHERE platform_id = ? AND promoted_at > ?",
                (platform_id, cutoff),
            ).fetchall()
            return {r["step_type"]: r["promoted_at"] for r in rows}

    def upsert_step_promotion(
        self, platform_id: str, step_type: str, reason: str = ""
    ) -> None:
        """Record that a step should be promoted to Sonnet for this platform."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO step_model_promotions (platform_id, step_type, promoted_at, reason)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(platform_id, step_type) DO UPDATE SET
                    promoted_at = excluded.promoted_at,
                    reason = excluded.reason
                """,
                (platform_id, step_type, now, reason),
            )

    def log_step_cost(
        self,
        platform_id: str,
        step_type: str,
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        success: bool = False,
    ) -> None:
        """Log a step execution for cost tracking."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO step_cost_log
                    (platform_id, step_type, model_id, input_tokens, output_tokens,
                     cost_usd, success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    platform_id, step_type, model_id,
                    input_tokens, output_tokens, cost_usd,
                    1 if success else 0, now,
                ),
            )

    def get_step_cost_report(self, days: int = 30) -> dict[str, Any]:
        """Get cost savings report for step model routing.

        Returns actual spend, counterfactual all-Sonnet spend, and savings.
        """
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Pricing per 1M tokens
        pricing = {
            "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        }
        sonnet_pricing = pricing["claude-sonnet-4-20250514"]

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT model_id, step_type, input_tokens, output_tokens, "
                "cost_usd, success FROM step_cost_log WHERE timestamp > ?",
                (cutoff,),
            ).fetchall()

        total_cost = 0.0
        counterfactual_cost = 0.0
        by_model: dict[str, dict[str, Any]] = {}
        by_step: dict[str, dict[str, Any]] = {}
        total_steps = 0
        successful_steps = 0

        for row in rows:
            r = dict(row)
            model = r["model_id"]
            step = r["step_type"]
            in_tok = r["input_tokens"]
            out_tok = r["output_tokens"]
            cost = r["cost_usd"]
            ok = r["success"]

            total_steps += 1
            if ok:
                successful_steps += 1
            total_cost += cost

            # Counterfactual: what if everything used Sonnet?
            cf_cost = (
                in_tok * sonnet_pricing["input"] / 1_000_000
                + out_tok * sonnet_pricing["output"] / 1_000_000
            )
            counterfactual_cost += cf_cost

            # By model
            if model not in by_model:
                by_model[model] = {"steps": 0, "cost": 0.0, "successes": 0}
            by_model[model]["steps"] += 1
            by_model[model]["cost"] += cost
            if ok:
                by_model[model]["successes"] += 1

            # By step type
            if step not in by_step:
                by_step[step] = {"steps": 0, "cost": 0.0, "haiku": 0, "sonnet": 0}
            by_step[step]["steps"] += 1
            by_step[step]["cost"] += cost
            if "haiku" in model:
                by_step[step]["haiku"] += 1
            else:
                by_step[step]["sonnet"] += 1

        savings = counterfactual_cost - total_cost
        savings_pct = (savings / counterfactual_cost * 100) if counterfactual_cost > 0 else 0

        return {
            "period_days": days,
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "total_cost_usd": round(total_cost, 6),
            "counterfactual_cost_usd": round(counterfactual_cost, 6),
            "savings_usd": round(savings, 6),
            "savings_pct": round(savings_pct, 1),
            "by_model": {
                m: {
                    "steps": v["steps"],
                    "cost_usd": round(v["cost"], 6),
                    "success_rate": round(v["successes"] / v["steps"] * 100, 1) if v["steps"] else 0,
                }
                for m, v in by_model.items()
            },
            "by_step_type": by_step,
        }

    def expire_old_promotions(self, days: int = 7) -> int:
        """Remove promotions older than N days. Returns count removed."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM step_model_promotions WHERE promoted_at < ?",
                (cutoff,),
            )
            return cursor.rowcount
