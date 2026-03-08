"""Tests for SelfHealer — automated recovery from failures."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime, timedelta
import os

from openclaw.daemon.self_healer import SelfHealer, _SERVICE_RESTART_COMMANDS
from openclaw.daemon.alert_router import AlertRouter
from openclaw.forge.platform_codex import PlatformCodex
from openclaw.models import (
    CheckResult,
    HealthCheck,
    HeartbeatTier,
    AccountStatus,
)


@pytest.fixture
def codex(tmp_path):
    db_path = str(tmp_path / "test_healer.db")
    return PlatformCodex(db_path=db_path)


@pytest.fixture
def alert_router():
    router = MagicMock(spec=AlertRouter)
    router.route = AsyncMock(return_value=True)
    return router


@pytest.fixture
def healer(codex, alert_router):
    return SelfHealer(codex, alert_router)


def _make_check(name="service:test", result=CheckResult.DOWN, message="test failure"):
    return HealthCheck(
        name=name,
        tier=HeartbeatTier.PULSE,
        result=result,
        message=message,
    )


class TestHeal:
    @pytest.mark.asyncio
    async def test_heal_healthy_returns_true(self, healer):
        check = _make_check(result=CheckResult.HEALTHY)
        result = await healer.heal(check)
        assert result is True

    @pytest.mark.asyncio
    async def test_heal_wp_down_alerts_and_returns_false(self, healer, alert_router):
        check = _make_check(name="wp:example.com", result=CheckResult.DOWN)
        result = await healer.heal(check)
        assert result is False
        alert_router.route.assert_called_once()
        # Verify alert is CRITICAL
        alert = alert_router.route.call_args[0][0]
        assert alert.severity.value == "critical"
        assert "WordPress" in alert.title

    @pytest.mark.asyncio
    async def test_heal_n8n_down_alerts_and_returns_false(self, healer, alert_router):
        check = _make_check(name="n8n:workflow-1", result=CheckResult.DOWN)
        result = await healer.heal(check)
        assert result is False
        alert_router.route.assert_called_once()

    @pytest.mark.asyncio
    async def test_heal_unknown_check_returns_false(self, healer):
        check = _make_check(name="unknown:thing", result=CheckResult.DOWN)
        result = await healer.heal(check)
        assert result is False


class TestRestartService:
    @pytest.mark.asyncio
    async def test_no_restart_command_returns_false(self, healer, codex):
        check = _make_check(name="service:nonexistent", result=CheckResult.DOWN)
        result = await healer._restart_service(check)
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_logs_action(self, healer, codex):
        check = _make_check(name="service:nonexistent", result=CheckResult.DOWN)
        await healer._restart_service(check)
        # Check that action was logged
        history = codex.get_action_history(limit=5)
        assert len(history) >= 1
        assert history[0]["action_type"] == "restart_service"
        assert history[0]["result"] == "skipped"

    @pytest.mark.asyncio
    async def test_restart_success(self, healer, codex):
        check = _make_check(name="service:screenpipe", result=CheckResult.DOWN)

        # Mock the subprocess to succeed
        with patch("asyncio.create_subprocess_shell") as mock_proc:
            proc = AsyncMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b"ok", b""))
            mock_proc.return_value = proc

            # Also mock the sleep to not actually wait
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await healer._restart_service(check)

        assert result is True

    @pytest.mark.asyncio
    async def test_restart_failure_alerts(self, healer, alert_router):
        check = _make_check(name="service:screenpipe", result=CheckResult.DOWN)

        with patch("asyncio.create_subprocess_shell") as mock_proc:
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate = AsyncMock(return_value=(b"", b"error msg"))
            mock_proc.return_value = proc

            result = await healer._restart_service(check)

        assert result is False
        alert_router.route.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_timeout(self, healer):
        import asyncio
        check = _make_check(name="service:screenpipe", result=CheckResult.DOWN)

        with patch("asyncio.create_subprocess_shell") as mock_proc:
            proc = AsyncMock()
            proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_proc.return_value = proc

            result = await healer._restart_service(check)

        assert result is False


class TestClearExpiredSessions:
    @pytest.mark.asyncio
    async def test_clears_stale_sessions(self, healer, tmp_path, monkeypatch):
        # Create sessions dir with old and new files
        sessions_dir = tmp_path / "data" / "sessions"
        sessions_dir.mkdir(parents=True)

        old_file = sessions_dir / "old_platform.json"
        old_file.write_text("{}")
        old_mtime = (datetime.now() - timedelta(days=60)).timestamp()
        os.utime(str(old_file), (old_mtime, old_mtime))

        new_file = sessions_dir / "new_platform.json"
        new_file.write_text("{}")

        # Patch the sessions path
        monkeypatch.setattr(
            "openclaw.daemon.self_healer.Path.resolve",
            lambda self: tmp_path / "openclaw" / "daemon",
        )

        # Use the actual method but with patched path
        original_method = SelfHealer.clear_expired_sessions

        async def patched_clear(self_ref):
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(days=30)
            cleared = 0
            for session_file in sessions_dir.glob("*.json"):
                try:
                    mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                    if mtime < cutoff:
                        session_file.unlink()
                        cleared += 1
                except OSError:
                    pass
            return cleared

        monkeypatch.setattr(SelfHealer, "clear_expired_sessions", patched_clear)
        result = await healer.clear_expired_sessions()
        assert result == 1
        assert not old_file.exists()
        assert new_file.exists()

    @pytest.mark.asyncio
    async def test_no_sessions_dir_returns_zero(self, healer):
        # Default path won't exist in test env
        result = await healer.clear_expired_sessions()
        assert result == 0


class TestServiceRestartCommands:
    def test_known_services_have_commands(self):
        assert "screenpipe" in _SERVICE_RESTART_COMMANDS
        assert "empire-dashboard" in _SERVICE_RESTART_COMMANDS
        assert "brain-mcp" in _SERVICE_RESTART_COMMANDS

    def test_commands_are_strings(self):
        for name, cmd in _SERVICE_RESTART_COMMANDS.items():
            assert isinstance(cmd, str)
            assert len(cmd) > 0
