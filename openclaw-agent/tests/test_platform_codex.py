"""Tests for openclaw/forge/platform_codex.py — SQLite persistence."""

import os
import pytest

from openclaw.forge.platform_codex import PlatformCodex
from openclaw.models import (
    AccountStatus,
    CaptchaType,
    ProfileContent,
    QualityGrade,
    SentinelScore,
    SignupStep,
    StepStatus,
    StepType,
)


@pytest.fixture
def codex(tmp_path):
    """Create a PlatformCodex backed by a temporary database."""
    db_path = str(tmp_path / "test_openclaw.db")
    return PlatformCodex(db_path=db_path)


class TestAccountOperations:
    def test_upsert_and_get_account(self, codex):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE, "myuser")
        account = codex.get_account("gumroad")
        assert account is not None
        assert account["platform_id"] == "gumroad"
        assert account["platform_name"] == "Gumroad"
        assert account["status"] == "active"
        assert account["username"] == "myuser"

    def test_get_nonexistent_account(self, codex):
        assert codex.get_account("nonexistent") is None

    def test_upsert_updates_existing(self, codex):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.PLANNED)
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE, "user1")
        account = codex.get_account("gumroad")
        assert account["status"] == "active"
        assert account["username"] == "user1"

    def test_update_account_status(self, codex):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.PLANNED)
        codex.update_account_status("gumroad", AccountStatus.SIGNUP_IN_PROGRESS)
        account = codex.get_account("gumroad")
        assert account["status"] == "signup_in_progress"

    def test_get_accounts_by_status(self, codex):
        codex.upsert_account("a", "A", AccountStatus.ACTIVE)
        codex.upsert_account("b", "B", AccountStatus.ACTIVE)
        codex.upsert_account("c", "C", AccountStatus.PLANNED)
        active = codex.get_accounts_by_status(AccountStatus.ACTIVE)
        assert len(active) == 2
        assert all(a["status"] == "active" for a in active)

    def test_get_all_accounts(self, codex):
        codex.upsert_account("a", "A", AccountStatus.ACTIVE)
        codex.upsert_account("b", "B", AccountStatus.PLANNED)
        all_accts = codex.get_all_accounts()
        assert len(all_accts) == 2

    def test_delete_account(self, codex):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        codex.delete_account("gumroad")
        assert codex.get_account("gumroad") is None


class TestCredentialOperations:
    def test_store_and_get_credentials(self, codex):
        # Must create account first (foreign key)
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        creds = {"email": "test@test.com", "password": "s3cret!", "token": "abc123"}
        codex.store_credentials("gumroad", creds)
        retrieved = codex.get_credentials("gumroad")
        assert retrieved is not None
        assert retrieved["email"] == "test@test.com"
        assert retrieved["password"] == "s3cret!"
        assert retrieved["token"] == "abc123"

    def test_get_nonexistent_credentials(self, codex):
        assert codex.get_credentials("nonexistent") is None

    def test_credentials_encrypted_in_db(self, codex, tmp_path):
        """Verify that the raw data in the DB is not plaintext."""
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        codex.store_credentials("gumroad", {"password": "mysecret"})
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "test_openclaw.db"))
        row = conn.execute(
            "SELECT encrypted_data FROM credentials WHERE platform_id = 'gumroad'"
        ).fetchone()
        conn.close()
        assert row is not None
        raw = row[0]
        assert "mysecret" not in raw  # Should be encrypted

    def test_delete_credentials(self, codex):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        codex.store_credentials("gumroad", {"password": "test"})
        codex.delete_credentials("gumroad")
        assert codex.get_credentials("gumroad") is None


class TestSignupLog:
    def test_log_step_and_retrieve(self, codex):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.SIGNUP_IN_PROGRESS)
        step = SignupStep(
            step_number=1,
            step_type=StepType.NAVIGATE,
            description="Navigate to signup page",
            status=StepStatus.COMPLETED,
        )
        codex.log_step("gumroad", step)
        log = codex.get_signup_log("gumroad")
        assert len(log) == 1
        assert log[0]["step_number"] == 1
        assert log[0]["step_type"] == "navigate"
        assert log[0]["status"] == "completed"

    def test_multiple_steps_ordered(self, codex):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.SIGNUP_IN_PROGRESS)
        for i in range(3):
            step = SignupStep(
                step_number=i + 1,
                step_type=StepType.FILL_FIELD,
                description=f"Step {i + 1}",
                status=StepStatus.COMPLETED,
            )
            codex.log_step("gumroad", step)
        log = codex.get_signup_log("gumroad")
        assert len(log) == 3
        assert [l["step_number"] for l in log] == [1, 2, 3]

    def test_get_failed_steps(self, codex):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.SIGNUP_IN_PROGRESS)
        codex.log_step("gumroad", SignupStep(
            step_number=1, step_type=StepType.FILL_FIELD,
            description="OK", status=StepStatus.COMPLETED,
        ))
        codex.log_step("gumroad", SignupStep(
            step_number=2, step_type=StepType.SOLVE_CAPTCHA,
            description="CAPTCHA", status=StepStatus.FAILED,
            error_message="Timeout",
        ))
        failed = codex.get_failed_steps("gumroad")
        assert len(failed) == 1
        assert failed[0]["step_number"] == 2


class TestProfileContent:
    def test_store_and_get_profile(self, codex):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        content = ProfileContent(
            platform_id="gumroad",
            username="testuser",
            bio="Test bio for gumroad",
            tagline="Test tagline",
        )
        score = SentinelScore(
            platform_id="gumroad",
            total_score=75.0,
            grade=QualityGrade.B,
        )
        codex.store_profile(content, score)
        profile = codex.get_profile("gumroad")
        assert profile is not None
        assert profile["content"]["username"] == "testuser"
        assert profile["sentinel_score"] == 75.0
        assert profile["grade"] == "B"

    def test_get_nonexistent_profile(self, codex):
        assert codex.get_profile("nonexistent") is None

    def test_get_all_profiles(self, codex):
        for pid, name, sc in [("a", "A", 80.0), ("b", "B", 60.0)]:
            codex.upsert_account(pid, name, AccountStatus.ACTIVE)
            codex.store_profile(
                ProfileContent(platform_id=pid, username=pid),
                SentinelScore(platform_id=pid, total_score=sc, grade=QualityGrade.B),
            )
        profiles = codex.get_all_profiles()
        assert len(profiles) == 2
        # Sorted by sentinel_score descending
        assert profiles[0]["sentinel_score"] >= profiles[1]["sentinel_score"]


class TestCaptchaLog:
    def test_log_captcha(self, codex):
        codex.upsert_account("etsy", "Etsy", AccountStatus.SIGNUP_IN_PROGRESS)
        codex.log_captcha("etsy", CaptchaType.RECAPTCHA_V3, auto_solved=True, duration=2.5)
        log = codex.get_captcha_log("etsy")
        assert len(log) == 1
        assert log[0]["captcha_type"] == "recaptcha_v3"
        assert log[0]["auto_solved"] == 1
        assert log[0]["duration_seconds"] == 2.5

    def test_get_all_captcha_logs(self, codex):
        codex.upsert_account("a", "A", AccountStatus.ACTIVE)
        codex.upsert_account("b", "B", AccountStatus.ACTIVE)
        codex.log_captcha("a", "hcaptcha", auto_solved=False, duration=10.0)
        codex.log_captcha("b", "recaptcha_v2", auto_solved=True, duration=3.0)
        all_logs = codex.get_captcha_log()
        assert len(all_logs) == 2


class TestStats:
    def test_get_stats_empty(self, codex):
        stats = codex.get_stats()
        assert stats["total_accounts"] == 0
        assert stats["active_accounts"] == 0

    def test_get_stats_populated(self, codex):
        codex.upsert_account("a", "A", AccountStatus.ACTIVE)
        codex.upsert_account("b", "B", AccountStatus.ACTIVE)
        codex.upsert_account("c", "C", AccountStatus.SIGNUP_FAILED)
        stats = codex.get_stats()
        assert stats["total_accounts"] == 3
        assert stats["active_accounts"] == 2
        assert stats["failed_signups"] == 1
        assert isinstance(stats["accounts_by_status"], dict)

    def test_get_recent_activity(self, codex):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        codex.log_step("gumroad", SignupStep(
            step_number=1, step_type=StepType.NAVIGATE,
            description="Nav", status=StepStatus.COMPLETED,
        ))
        activity = codex.get_recent_activity(limit=5)
        assert isinstance(activity, list)
        assert len(activity) >= 1
        for item in activity:
            assert "type" in item
            assert "platform_id" in item
            assert "summary" in item
            assert "timestamp" in item
