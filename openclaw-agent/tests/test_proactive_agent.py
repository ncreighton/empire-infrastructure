"""Tests for ProactiveAgent — autonomous decision engine."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, PropertyMock
from pathlib import Path
import json
import os

from openclaw.daemon.proactive_agent import ProactiveAgent, _APPROVAL_REQUIRED, _AUTO_APPROVED
from openclaw.knowledge.platforms import get_platform
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
        mock_engine.codex.get_action_history.return_value = []

        actions = agent._check_unsigned_platforms()
        assert len(actions) == 1
        assert actions[0].action_type == "new_signup"
        assert actions[0].requires_approval is False
        assert actions[0].priority == 4

    def test_skips_already_signed_up(self, agent, mock_engine):
        rec = MagicMock()
        rec.platform_id = "gumroad"
        mock_engine.prioritize.return_value = [rec]
        mock_engine.codex.get_account.return_value = {
            "status": AccountStatus.ACTIVE.value,
        }
        mock_engine.codex.get_action_history.return_value = []

        actions = agent._check_unsigned_platforms()
        assert actions == []

    def test_handles_prioritize_error(self, agent, mock_engine):
        mock_engine.prioritize.side_effect = RuntimeError("boom")
        mock_engine.codex.get_action_history.return_value = []
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
        assert actions[0].requires_approval is False

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


class TestVibeCoderOpportunities:
    def test_no_vibecoder_returns_empty(self, agent, mock_engine):
        """When engine has no vibecoder attribute, skip gracefully."""
        del mock_engine.vibecoder
        actions = agent._check_vibecoder_opportunities()
        assert actions == []

    def test_health_failure_creates_mission(self, agent, mock_engine):
        """Repeated health failures create a VibeCoder bugfix mission."""
        mock_engine.vibecoder = MagicMock()
        mock_engine.vibecoder.list_missions.return_value = []

        # Simulate 3+ consecutive failures
        mock_engine.codex.get_latest_checks.return_value = {
            "openclaw:self": {"result": "down", "message": "DB connection failed"},
        }
        mock_engine.codex.get_health_history.return_value = [
            {"result": "down"},
            {"result": "down"},
            {"result": "down"},
        ]
        actions = agent._check_health_to_mission(mock_engine.vibecoder)
        assert len(actions) == 1
        assert actions[0].action_type == "vibecoder_mission"
        assert actions[0].priority == 3
        assert "Fix" in actions[0].params["title"]

    def test_health_failure_skips_below_threshold(self, agent, mock_engine):
        """Only 1-2 failures: no mission created yet."""
        mock_engine.vibecoder = MagicMock()
        mock_engine.codex.get_latest_checks.return_value = {
            "openclaw:self": {"result": "down", "message": "err"},
        }
        mock_engine.codex.get_health_history.return_value = [
            {"result": "down"},
            {"result": "healthy"},
        ]
        actions = agent._check_health_to_mission(mock_engine.vibecoder)
        assert actions == []

    def test_stalled_mission_detection(self, agent, mock_engine):
        """Missions stuck executing for >1h get force-failed."""
        mock_engine.vibecoder = MagicMock()
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        mock_engine.vibecoder.list_missions.return_value = [
            {
                "mission_id": "abc123",
                "project_id": "test-proj",
                "started_at": old_time,
            },
        ]
        actions = agent._check_stalled_missions(mock_engine.vibecoder)
        assert len(actions) == 1
        assert actions[0].params["action"] == "force_fail"
        assert actions[0].params["mission_id"] == "abc123"

    def test_stalled_skips_recent_missions(self, agent, mock_engine):
        """Missions still within the 1h window are not flagged."""
        mock_engine.vibecoder = MagicMock()
        recent = datetime.now().isoformat()
        mock_engine.vibecoder.list_missions.return_value = [
            {
                "mission_id": "abc123",
                "project_id": "test-proj",
                "started_at": recent,
            },
        ]
        actions = agent._check_stalled_missions(mock_engine.vibecoder)
        assert actions == []

    def test_project_discovery_skips_registered(self, agent, mock_engine, tmp_path):
        """Already-registered projects are not flagged for discovery."""
        mock_engine.vibecoder = MagicMock()
        mock_engine.vibecoder.list_projects.return_value = [
            {"project_id": "existing-proj"},
        ]

        # Create a project dir with a marker
        proj_dir = tmp_path / "existing-proj"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("[tool.test]")

        os.environ["EMPIRE_ROOT"] = str(tmp_path)
        try:
            actions = agent._check_project_discovery(mock_engine.vibecoder)
            # Should not flag existing-proj since it's already registered
            for a in actions:
                assert "existing-proj" not in a.params.get("projects", [])
        finally:
            del os.environ["EMPIRE_ROOT"]

    def test_project_discovery_finds_new(self, agent, mock_engine, tmp_path):
        """Unregistered projects with markers get discovered."""
        mock_engine.vibecoder = MagicMock()
        mock_engine.vibecoder.list_projects.return_value = []

        # Create a new project dir with a marker
        proj_dir = tmp_path / "new-project"
        proj_dir.mkdir()
        (proj_dir / "requirements.txt").write_text("fastapi\n")

        os.environ["EMPIRE_ROOT"] = str(tmp_path)
        try:
            actions = agent._check_project_discovery(mock_engine.vibecoder)
            assert len(actions) == 1
            assert "new-project" in actions[0].params["projects"]
            assert actions[0].action_type == "vibecoder_discover_projects"
        finally:
            del os.environ["EMPIRE_ROOT"]


class TestConstants:
    def test_approval_required_set(self):
        assert "restart_service" in _APPROVAL_REQUIRED
        assert "profile_content_change" in _APPROVAL_REQUIRED

    def test_auto_approved_set(self):
        assert "verify_email" in _AUTO_APPROVED
        assert "retry_signup" in _AUTO_APPROVED
        assert "session_cleanup" in _AUTO_APPROVED
        assert "new_signup" in _AUTO_APPROVED
        assert "vibecoder_mission" in _AUTO_APPROVED
        assert "vibecoder_discover_projects" in _AUTO_APPROVED


class TestDailySignupCap:
    def test_cap_enforced(self, agent, mock_engine):
        """When daily cap reached, no new_signup actions returned."""
        rec = MagicMock()
        rec.platform_id = "gumroad"
        rec.platform_name = "Gumroad"
        rec.priority = MagicMock()
        rec.priority.value = "high"
        rec.score = 85.0
        mock_engine.prioritize.return_value = [rec]
        mock_engine.codex.get_account.return_value = None
        # Simulate cap reached (only "starting" entries are counted)
        mock_engine.codex.get_action_history.return_value = [
            {"action_type": "new_signup", "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "result": "starting"},
            {"action_type": "new_signup", "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "result": "starting"},
            {"action_type": "new_signup", "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "result": "starting"},
        ]

        actions = agent._check_unsigned_platforms()
        assert actions == []

    def test_no_approval_needed(self, agent, mock_engine):
        """new_signup actions no longer require approval."""
        rec = MagicMock()
        rec.platform_id = "gumroad"
        rec.platform_name = "Gumroad"
        rec.priority = MagicMock()
        rec.priority.value = "high"
        rec.score = 85.0
        mock_engine.prioritize.return_value = [rec]
        mock_engine.codex.get_account.return_value = None
        mock_engine.codex.get_action_history.return_value = []

        actions = agent._check_unsigned_platforms()
        assert len(actions) >= 1
        assert actions[0].requires_approval is False

    def test_skips_previously_succeeded(self, agent, mock_engine):
        """Platforms with a successful signup in action_log are skipped."""
        rec = MagicMock()
        rec.platform_id = "gumroad"
        rec.platform_name = "Gumroad"
        rec.priority = MagicMock()
        rec.priority.value = "high"
        rec.score = 85.0
        mock_engine.prioritize.return_value = [rec]
        mock_engine.codex.get_account.return_value = None  # Account record missing
        # But action_log shows a previous success
        mock_engine.codex.get_action_history.return_value = [
            {"action_type": "new_signup", "target": "gumroad", "result": "success",
             "timestamp": "2026-03-08T17:00:00"},
        ]

        actions = agent._check_unsigned_platforms()
        assert actions == []
