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
    ProfileContent,
    SentinelScore,
    SignupStep,
    StepStatus,
    StepType,
    CaptchaType,
    QualityGrade,
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
