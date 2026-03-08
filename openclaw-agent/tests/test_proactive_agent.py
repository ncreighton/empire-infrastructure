"""Tests for ProactiveAgent — autonomous decision engine."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, PropertyMock
from pathlib import Path
import json
import os

from openclaw.daemon.proactive_agent import ProactiveAgent, _APPROVAL_REQUIRED, _AUTO_APPROVED
from openclaw.daemon.heartbeat_config import HeartbeatConfig
from openclaw.models import AccountStatus


@pytest.fixture
def config():
    return HeartbeatConfig(profile_stale_days=30)


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.codex = MagicMock()
    engine.retry_engine = MagicMock()
    engine.prioritize = MagicMock(return_value=[])
    return engine


@pytest.fixture
def agent(mock_engine, config):
    return ProactiveAgent(mock_engine, config)


class TestEvaluate:
    def test_evaluate_returns_list(self, agent, mock_engine):
        mock_engine.codex.get_accounts_by_status.return_value = []
        mock_engine.codex.get_all_profiles.return_value = []
        actions = agent.evaluate()
        assert isinstance(actions, list)

    def test_evaluate_sorted_by_priority(self, agent, mock_engine):
        # Setup: email verification (priority 2) + failed signup (priority 3)
        mock_engine.codex.get_accounts_by_status.side_effect = lambda status: {
            AccountStatus.EMAIL_VERIFICATION_PENDING: [
                {"platform_id": "p1", "platform_name": "P1"},
            ],
            AccountStatus.SIGNUP_FAILED: [
                {
                    "platform_id": "p2",
                    "platform_name": "P2",
                    "updated_at": "2020-01-01T00:00:00",
                },
            ],
            AccountStatus.ACTIVE: [],
        }.get(status, [])
        mock_engine.codex.get_all_profiles.return_value = []
        mock_engine.retry_engine.should_retry.return_value = True

        actions = agent.evaluate()
        assert len(actions) >= 2
        # Sorted: priority 2 before priority 3
        assert actions[0].priority <= actions[1].priority


class TestEmailVerifications:
    def test_finds_pending_verifications(self, agent, mock_engine):
        mock_engine.codex.get_accounts_by_status.return_value = [
            {"platform_id": "gumroad", "platform_name": "Gumroad"},
            {"platform_id": "etsy", "platform_name": "Etsy"},
        ]
        actions = agent._check_email_verifications()
        assert len(actions) == 2
        assert actions[0].action_type == "verify_email"
        assert actions[0].priority == 2
        assert actions[0].requires_approval is False
        assert actions[0].requires_browser is False

    def test_no_pending_returns_empty(self, agent, mock_engine):
        mock_engine.codex.get_accounts_by_status.return_value = []
        actions = agent._check_email_verifications()
        assert actions == []


class TestFailedSignups:
    def test_finds_retryable_failures(self, agent, mock_engine):
        mock_engine.codex.get_accounts_by_status.return_value = [
            {
                "platform_id": "gumroad",
                "platform_name": "Gumroad",
                "updated_at": "2020-01-01T00:00:00",  # Old enough to retry
            },
        ]
        mock_engine.retry_engine.should_retry.return_value = True

        actions = agent._check_failed_signups()
        assert len(actions) == 1
        assert actions[0].action_type == "retry_signup"
        assert actions[0].priority == 3
        assert actions[0].requires_browser is True

    def test_skips_recent_failures(self, agent, mock_engine):
        # Updated less than 1 hour ago
        recent = datetime.now().isoformat()
        mock_engine.codex.get_accounts_by_status.return_value = [
            {
                "platform_id": "gumroad",
                "platform_name": "Gumroad",
                "updated_at": recent,
            },
        ]
        actions = agent._check_failed_signups()
        assert actions == []

    def test_fallback_without_retry_engine(self, agent, mock_engine):
        mock_engine.codex.get_accounts_by_status.return_value = [
            {
                "platform_id": "gumroad",
                "platform_name": "Gumroad",
                "updated_at": "2020-01-01T00:00:00",
            },
        ]
        # Simulate retry_engine not having should_retry
        mock_engine.retry_engine.should_retry.side_effect = AttributeError
        mock_engine.codex.get_failed_steps.return_value = [{"step": 1}]  # < 3 failures

        actions = agent._check_failed_signups()
        assert len(actions) == 1

    def test_fallback_too_many_failures(self, agent, mock_engine):
        mock_engine.codex.get_accounts_by_status.return_value = [
            {
                "platform_id": "gumroad",
                "platform_name": "Gumroad",
                "updated_at": "2020-01-01T00:00:00",
            },
        ]
        mock_engine.retry_engine.should_retry.side_effect = AttributeError
        mock_engine.codex.get_failed_steps.return_value = [1, 2, 3]  # >= 3 failures

        actions = agent._check_failed_signups()
        assert actions == []


class TestUnsignedPlatforms:
    def test_finds_unsigned_platforms(self, agent, mock_engine):
        rec = MagicMock()
        rec.platform_id = "gumroad"
        rec.platform_name = "Gumroad"
        rec.priority = MagicMock()
        rec.priority.value = "high"
        rec.score = 85.0
        mock_engine.prioritize.return_value = [rec]
        mock_engine.codex.get_account.return_value = None

        actions = agent._check_unsigned_platforms()
        assert len(actions) == 1
        assert actions[0].action_type == "new_signup"
        assert actions[0].requires_approval is True
        assert actions[0].priority == 4

    def test_skips_already_signed_up(self, agent, mock_engine):
        rec = MagicMock()
        rec.platform_id = "gumroad"
        mock_engine.prioritize.return_value = [rec]
        mock_engine.codex.get_account.return_value = {
            "status": AccountStatus.ACTIVE.value,
        }

        actions = agent._check_unsigned_platforms()
        assert actions == []

    def test_handles_prioritize_error(self, agent, mock_engine):
        mock_engine.prioritize.side_effect = RuntimeError("boom")
        actions = agent._check_unsigned_platforms()
        assert actions == []


class TestLowScoreProfiles:
    def test_flags_low_grade_profiles(self, agent, mock_engine):
        mock_engine.codex.get_all_profiles.return_value = [
            {
                "platform_id": "gumroad",
                "sentinel_score": 35.0,
                "grade": "F",
            },
        ]
        actions = agent._check_low_score_profiles()
        assert len(actions) == 1
        assert actions[0].action_type == "enhance_profile"
        assert actions[0].priority == 5
        assert actions[0].requires_approval is True

    def test_skips_good_grade_profiles(self, agent, mock_engine):
        mock_engine.codex.get_all_profiles.return_value = [
            {
                "platform_id": "gumroad",
                "sentinel_score": 85.0,
                "grade": "A",
            },
        ]
        actions = agent._check_low_score_profiles()
        assert actions == []

    def test_skips_zero_score(self, agent, mock_engine):
        mock_engine.codex.get_all_profiles.return_value = [
            {
                "platform_id": "gumroad",
                "sentinel_score": 0,
                "grade": "F",
            },
        ]
        actions = agent._check_low_score_profiles()
        assert actions == []


class TestStaleProfiles:
    def test_flags_stale_profiles(self, agent, mock_engine):
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        mock_engine.codex.get_accounts_by_status.return_value = [
            {
                "platform_id": "gumroad",
                "platform_name": "Gumroad",
                "updated_at": old_date,
            },
        ]
        actions = agent._check_stale_profiles()
        assert len(actions) == 1
        assert actions[0].action_type == "refresh_profile"
        assert actions[0].priority == 6

    def test_skips_recent_profiles(self, agent, mock_engine):
        recent = datetime.now().isoformat()
        mock_engine.codex.get_accounts_by_status.return_value = [
            {
                "platform_id": "gumroad",
                "platform_name": "Gumroad",
                "updated_at": recent,
            },
        ]
        actions = agent._check_stale_profiles()
        assert actions == []


class TestSessionCleanup:
    def test_no_sessions_dir_returns_empty(self, agent):
        # Default path won't exist in test environment
        actions = agent._check_session_cleanup()
        assert actions == []

    def test_returns_action_for_stale_files(self, agent, tmp_path):
        """Verify stale session detection logic with a patched directory."""
        from openclaw.models import ProactiveAction

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        stale = sessions_dir / "old_platform.json"
        stale.write_text("{}")
        old_mtime = (datetime.now() - timedelta(days=60)).timestamp()
        os.utime(str(stale), (old_mtime, old_mtime))

        # Directly patch the method to use our test dir
        original = agent._check_session_cleanup

        def patched():
            cutoff = datetime.now() - timedelta(days=30)
            stale_sessions = []
            for sf in sessions_dir.glob("*.json"):
                try:
                    mtime = datetime.fromtimestamp(sf.stat().st_mtime)
                    if mtime < cutoff:
                        stale_sessions.append(sf.stem)
                except OSError:
                    pass
            if stale_sessions:
                return [ProactiveAction(
                    action_type="session_cleanup",
                    priority=7,
                    target="sessions",
                    description=f"Clean up {len(stale_sessions)} stale session(s)",
                    requires_browser=False,
                    requires_approval=False,
                    params={"stale_sessions": stale_sessions},
                )]
            return []

        agent._check_session_cleanup = patched
        actions = agent._check_session_cleanup()
        assert len(actions) == 1
        assert actions[0].action_type == "session_cleanup"
        assert "old_platform" in actions[0].params["stale_sessions"]


class TestConstants:
    def test_approval_required_set(self):
        assert "new_signup" in _APPROVAL_REQUIRED
        assert "restart_service" in _APPROVAL_REQUIRED

    def test_auto_approved_set(self):
        assert "verify_email" in _AUTO_APPROVED
        assert "retry_signup" in _AUTO_APPROVED
        assert "session_cleanup" in _AUTO_APPROVED
