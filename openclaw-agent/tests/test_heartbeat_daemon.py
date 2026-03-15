"""Tests for HeartbeatDaemon — main daemon loop, tiers, helpers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from openclaw.daemon.heartbeat_daemon import HeartbeatDaemon
from openclaw.daemon.heartbeat_config import HeartbeatConfig
from openclaw.forge.platform_codex import PlatformCodex
from openclaw.models import (
    Alert,
    AlertSeverity,
    CheckResult,
    HealthCheck,
    HeartbeatTier,
)


@pytest.fixture
def config():
    return HeartbeatConfig(
        wordpress_domains=[],
        service_ports={},
        pulse_interval=5,
        scan_interval=10,
        intel_interval=30,
    )


@pytest.fixture
def real_codex(tmp_path):
    """Real PlatformCodex for tests that need actual SQLite operations."""
    db_path = str(tmp_path / "test_daemon.db")
    return PlatformCodex(db_path=db_path)


@pytest.fixture
def mock_engine(real_codex):
    engine = MagicMock()
    engine.codex = real_codex
    engine.notifier = MagicMock()
    engine.notifier.notify = AsyncMock(return_value=[])
    engine.sentinel = MagicMock()
    engine.prioritize = MagicMock(return_value=[])
    return engine


@pytest.fixture
def daemon(mock_engine, config):
    return HeartbeatDaemon(mock_engine, config)


class TestInit:
    def test_daemon_initializes(self, daemon):
        assert daemon._running is False
        assert daemon._started_at is None
        assert daemon._tier_runs == {"PULSE": 0, "SCAN": 0, "INTEL": 0, "DAILY": 0}

    def test_daemon_has_components(self, daemon):
        assert daemon.alert_router is not None
        assert daemon.cron is not None
        assert daemon.proactive is not None
        assert daemon.healer is not None


class TestGetStatus:
    def test_status_when_not_running(self, daemon):
        status = daemon.get_status()
        assert status["running"] is False
        assert status["started_at"] is None
        assert status["uptime_seconds"] == 0.0

    def test_status_when_running(self, daemon):
        daemon._running = True
        daemon._started_at = datetime.now()
        status = daemon.get_status()
        assert status["running"] is True
        assert status["started_at"] is not None
        assert status["uptime_seconds"] >= 0


class TestPreflight:
    def test_preflight_succeeds_when_running(self, daemon):
        daemon._running = True
        assert daemon._preflight("PULSE") is True

    def test_preflight_fails_when_not_running(self, daemon):
        daemon._running = False
        assert daemon._preflight("PULSE") is False

    def test_preflight_fails_on_db_error(self, daemon):
        daemon._running = True
        with patch.object(daemon.codex, "get_stats", side_effect=RuntimeError("DB error")):
            assert daemon._preflight("SCAN") is False


class TestSelfHealthCheck:
    def test_healthy_when_db_ok(self, daemon):
        check = daemon._self_health_check()
        assert check.result == CheckResult.HEALTHY
        assert check.name == "openclaw:self"
        assert check.tier == HeartbeatTier.PULSE
        assert "accounts" in check.message

    def test_down_when_db_fails(self, daemon):
        with patch.object(daemon.codex, "get_stats", side_effect=RuntimeError("connection refused")):
            check = daemon._self_health_check()
            assert check.result == CheckResult.DOWN
            assert "connection refused" in check.message


class TestCheckToAlert:
    def test_down_becomes_critical(self):
        check = HealthCheck(
            name="wp:example.com",
            tier=HeartbeatTier.PULSE,
            result=CheckResult.DOWN,
            message="Site is down",
        )
        alert = HeartbeatDaemon._check_to_alert(check)
        assert alert.severity == AlertSeverity.CRITICAL
        assert "wp:example.com" in alert.title

    def test_degraded_becomes_warning(self):
        check = HealthCheck(
            name="service:api",
            tier=HeartbeatTier.PULSE,
            result=CheckResult.DEGRADED,
            message="Slow",
        )
        alert = HeartbeatDaemon._check_to_alert(check)
        assert alert.severity == AlertSeverity.WARNING

    def test_healthy_becomes_info(self):
        check = HealthCheck(
            name="test:x",
            tier=HeartbeatTier.PULSE,
            result=CheckResult.HEALTHY,
            message="OK",
        )
        alert = HeartbeatDaemon._check_to_alert(check)
        assert alert.severity == AlertSeverity.INFO

    def test_source_extracted_from_name(self):
        check = HealthCheck(
            name="service:dashboard",
            tier=HeartbeatTier.PULSE,
            result=CheckResult.DOWN,
            message="down",
        )
        alert = HeartbeatDaemon._check_to_alert(check)
        assert alert.source == "service_check"


class TestCompileDailyReport:
    def test_report_contains_key_info(self, daemon):
        report = daemon._compile_daily_report()
        assert "Daily Empire Health Report" in report
        assert "Accounts:" in report
        assert "total" in report
        assert "Alerts Today:" in report
        assert "Daemon Uptime:" in report


class TestRegisterDefaultCrons:
    def test_registers_default_jobs(self, daemon):
        daemon._register_default_crons()
        jobs = daemon.cron.get_all()
        assert len(jobs) == 12

    def test_default_cron_names(self, daemon):
        daemon._register_default_crons()
        jobs = daemon.cron.get_all()
        names = {j.name for j in jobs}
        assert "morning-briefing" in names
        assert "ops-check" in names
        assert "signup-retry" in names
        assert "daily-report" in names

    def test_idempotent_registration(self, daemon):
        daemon._register_default_crons()
        daemon._register_default_crons()
        jobs = daemon.cron.get_all()
        assert len(jobs) == 12  # No duplicates


class TestBuildActionRegistry:
    def test_registry_has_all_actions(self, daemon):
        daemon._build_action_registry()
        assert "morning_briefing" in daemon._action_registry
        assert "ops_check" in daemon._action_registry
        assert "signup_retry" in daemon._action_registry
        assert "session_cleanup" in daemon._action_registry
        assert "daily_report" in daemon._action_registry

    def test_registry_values_are_callable(self, daemon):
        daemon._build_action_registry()
        for name, func in daemon._action_registry.items():
            assert callable(func), f"{name} is not callable"


class TestStartStop:
    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, daemon):
        daemon._running = True
        await daemon.stop()
        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_start_already_running_returns_early(self, daemon):
        daemon._running = True
        # Should log warning and return immediately without entering loops
        await daemon.start()
        # Still running — no crash, no second loop started


class TestPIDFile:
    def test_write_pid(self, daemon, tmp_path):
        daemon._pid_file = tmp_path / "test_daemon.pid"
        daemon._write_pid()
        assert daemon._pid_file.exists()
        pid = int(daemon._pid_file.read_text().strip())
        assert pid > 0

    def test_cleanup_pid(self, daemon, tmp_path):
        daemon._pid_file = tmp_path / "test_daemon.pid"
        daemon._pid_file.write_text("12345")
        daemon._cleanup_pid()
        assert not daemon._pid_file.exists()

    def test_cleanup_pid_no_file(self, daemon, tmp_path):
        daemon._pid_file = tmp_path / "nonexistent.pid"
        # Should not raise
        daemon._cleanup_pid()


class TestRunPulse:
    @pytest.mark.asyncio
    async def test_pulse_runs_self_health(self, daemon):
        """Pulse with no WP domains or services should still run self-health."""
        await daemon._run_pulse()
        # Self-health check should be logged — verify via DB
        checks = daemon.codex.get_recent_checks(tier=HeartbeatTier.PULSE, limit=10)
        assert len(checks) >= 1

    @pytest.mark.asyncio
    async def test_pulse_with_domains(self, daemon, config):
        config.wordpress_domains = ["test.com"]
        with patch(
            "openclaw.daemon.heartbeat_daemon.wordpress_check.check_all_sites",
            new_callable=AsyncMock,
            return_value=[
                HealthCheck(
                    name="wp:test.com",
                    tier=HeartbeatTier.PULSE,
                    result=CheckResult.HEALTHY,
                    message="OK",
                )
            ],
        ):
            await daemon._run_pulse()

        checks = daemon.codex.get_recent_checks(tier=HeartbeatTier.PULSE, limit=10)
        assert len(checks) >= 2  # WP + self


class TestCronActions:
    @pytest.mark.asyncio
    async def test_morning_briefing_action(self, daemon):
        with patch.object(daemon.alert_router, "route", new_callable=AsyncMock, return_value=True):
            result = await daemon._cron_morning_briefing()
        assert result["delivered"] is True

    @pytest.mark.asyncio
    async def test_session_cleanup_action(self, daemon):
        with patch.object(daemon.healer, "clear_expired_sessions", new_callable=AsyncMock, return_value=3):
            result = await daemon._cron_session_cleanup()
        assert result["cleared"] == 3

    @pytest.mark.asyncio
    async def test_platform_scout_action(self, daemon):
        result = await daemon._cron_platform_scout()
        assert "recommendations" in result
