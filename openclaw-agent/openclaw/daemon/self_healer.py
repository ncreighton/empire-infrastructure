"""SelfHealer — automated recovery from detected failures.

Auto-restarts services, retries failed signups, cleans up expired
resources. Logs all actions to the action_log table.
"""

from __future__ import annotations

import asyncio
import logging
import platform
from pathlib import Path
from typing import Any

from openclaw.daemon.alert_router import AlertRouter
from openclaw.forge.platform_codex import PlatformCodex
from openclaw.models import (
    AccountStatus,
    Alert,
    AlertSeverity,
    CheckResult,
    HealthCheck,
)

logger = logging.getLogger(__name__)

# Service restart commands — Windows (local dev)
_SERVICE_RESTART_COMMANDS_WINDOWS: dict[str, str] = {
    "screenpipe": 'powershell -Command "& {Stop-Process -Name screenpipe -Force -ErrorAction SilentlyContinue; Start-Sleep 2; Start-Process screenpipe}"',
    "empire-dashboard": 'powershell -File "D:/Claude Code Projects/scripts/restart-dashboard.ps1"',
    "grimoire-api": 'powershell -Command "& {cd grimoire-intelligence; Start-Process python -ArgumentList \'-m\',\'uvicorn\',\'api.app:app\',\'--port\',\'8080\'}"',
    "videoforge-api": 'powershell -Command "& {cd videoforge-engine; Start-Process python -ArgumentList \'-m\',\'uvicorn\',\'api.app:app\',\'--port\',\'8090\'}"',
    "brain-mcp": 'powershell -Command "& {cd EMPIRE-BRAIN; Start-Process python -ArgumentList \'brain-mcp-server.py\'}"',
}

# Service restart commands — Linux/Docker (VPS)
_SERVICE_RESTART_COMMANDS_LINUX: dict[str, str] = {
    "n8n": "docker restart empire-n8n",
    "empire-dashboard": "docker restart empire-dashboard",
    "article-audit": "docker restart empire-article-audit",
    "toolbox": "docker restart empire-toolbox",
}


def _get_restart_commands() -> dict[str, str]:
    """Return platform-appropriate restart commands.

    Returns empty dict inside Docker containers since Docker CLI
    isn't available — the daemon can only alert, not restart peers.
    """
    import os
    if os.path.exists("/.dockerenv"):
        return {}  # Can't run docker commands from inside a container
    if platform.system() == "Windows":
        return _SERVICE_RESTART_COMMANDS_WINDOWS
    return _SERVICE_RESTART_COMMANDS_LINUX


class SelfHealer:
    """Automated recovery from detected failures."""

    def __init__(self, codex: PlatformCodex, alert_router: AlertRouter):
        self.codex = codex
        self.alert_router = alert_router

    async def heal(self, check: HealthCheck) -> bool:
        """Attempt to heal a failed health check.

        Returns True if healed (or healing attempted), False if can't heal.
        """
        if check.result == CheckResult.HEALTHY:
            return True

        if check.name.startswith("service:") and check.result == CheckResult.DOWN:
            return await self._restart_service(check)

        if check.name.startswith("wp:") and check.result == CheckResult.DOWN:
            # Can't restart WP sites — just alert
            await self.alert_router.route(Alert(
                severity=AlertSeverity.CRITICAL,
                source="self_healer",
                title=f"WordPress site DOWN: {check.name}",
                message=check.message,
                details=check.details,
            ))
            return False

        if check.name.startswith("n8n:") and check.result == CheckResult.DOWN:
            await self.alert_router.route(Alert(
                severity=AlertSeverity.CRITICAL,
                source="self_healer",
                title=f"n8n DOWN: {check.name}",
                message=check.message,
            ))
            return False

        return False

    async def _restart_service(self, check: HealthCheck) -> bool:
        """Attempt to restart a local empire service."""
        service_name = check.name.replace("service:", "")
        restart_cmd = _get_restart_commands().get(service_name)

        if not restart_cmd:
            logger.warning(f"No restart command for service: {service_name}")
            self.codex.log_action(
                action_type="restart_service",
                target=service_name,
                description=f"Cannot restart — no command configured",
                result="skipped",
            )
            return False

        logger.info(f"Attempting to restart service: {service_name}")
        self.codex.log_action(
            action_type="restart_service",
            target=service_name,
            description=f"Auto-restarting due to: {check.message}",
            result="attempting",
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                restart_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30,
            )

            if proc.returncode == 0:
                # Wait and re-check
                await asyncio.sleep(10)
                self.codex.log_action(
                    action_type="restart_service",
                    target=service_name,
                    description=f"Restart command completed",
                    result="success",
                )
                logger.info(f"Service {service_name} restart command succeeded")
                return True
            else:
                error = stderr.decode()[:200] if stderr else "unknown"
                self.codex.log_action(
                    action_type="restart_service",
                    target=service_name,
                    description=f"Restart failed: {error}",
                    result="failed",
                )
                await self.alert_router.route(Alert(
                    severity=AlertSeverity.CRITICAL,
                    source="self_healer",
                    title=f"Service restart FAILED: {service_name}",
                    message=f"Restart command failed: {error}",
                ))
                return False

        except asyncio.TimeoutError:
            self.codex.log_action(
                action_type="restart_service",
                target=service_name,
                description="Restart command timed out after 30s",
                result="timeout",
            )
            return False
        except Exception as e:
            logger.error(f"Service restart error: {e}")
            return False

    async def retry_failed_signups(self, engine: Any) -> int:
        """Re-attempt signups that failed with transient errors.

        Args:
            engine: OpenClawEngine instance.

        Returns:
            Count of retried platforms.
        """
        from datetime import timedelta
        failed = self.codex.get_accounts_by_status(AccountStatus.SIGNUP_FAILED)
        cutoff = (
            __import__("datetime").datetime.now() - timedelta(hours=1)
        ).isoformat()

        retried = 0
        for account in failed:
            # Only retry if last attempt was > 1 hour ago
            if account.get("updated_at", "") > cutoff:
                continue

            platform_id = account["platform_id"]

            # Check retry engine
            can_proceed, _ = engine.rate_limiter.can_proceed(platform_id)
            if not can_proceed:
                continue

            self.codex.log_action(
                action_type="retry_signup",
                target=platform_id,
                description=f"Auto-retrying failed signup for {account['platform_name']}",
                result="attempting",
            )

            try:
                result = await engine.signup_async(platform_id)
                status = "success" if result.success else "failed"
                self.codex.log_action(
                    action_type="retry_signup",
                    target=platform_id,
                    description=f"Retry result: {result.status.value}",
                    result=status,
                )
                retried += 1
            except Exception as e:
                logger.error(f"Retry failed for {platform_id}: {e}")

        return retried

    async def clear_expired_sessions(self) -> int:
        """Remove session cookies older than 30 days.

        Returns:
            Count of sessions cleared.
        """
        from datetime import datetime, timedelta
        sessions_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sessions"
        if not sessions_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=30)
        cleared = 0
        for session_file in sessions_dir.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                if mtime < cutoff:
                    session_file.unlink()
                    cleared += 1
                    logger.debug(f"Cleared stale session: {session_file.stem}")
            except OSError:
                pass

        if cleared:
            self.codex.log_action(
                action_type="session_cleanup",
                target="sessions",
                description=f"Cleared {cleared} stale session(s)",
                result="success",
            )

        return cleared
