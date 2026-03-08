"""HeartbeatDaemon — autonomous daemon with 4 cascading heartbeat tiers.

Pattern: EMPIRE-BRAIN Evolution Engine
    - _timed() wrapper for interval enforcement
    - _preflight() validation before each tier
    - try/finally cleanup
    - Content hash dedup on alerts
    - Non-blocking webhook push

Tiers:
    PULSE  (5 min)  — WP sites + service ports + self-health
    SCAN   (30 min) — n8n + email + profiles + failed retries
    INTEL  (6 hr)   — GSC traffic + keyword gaps + recommendations
    DAILY  (24 hr)  — full report + security audit
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from openclaw.daemon.alert_router import AlertRouter
from openclaw.daemon.cron_scheduler import CronScheduler
from openclaw.daemon.heartbeat_config import HeartbeatConfig
from openclaw.daemon.proactive_agent import ProactiveAgent
from openclaw.daemon.self_healer import SelfHealer

from openclaw.daemon.checks import (
    wordpress_check,
    service_check,
    n8n_check,
    email_check,
    profile_check,
    seo_check,
    security_check,
)

from openclaw.models import (
    Alert,
    AlertSeverity,
    CheckResult,
    HealthCheck,
    HeartbeatTier,
)

logger = logging.getLogger(__name__)


class HeartbeatDaemon:
    """Autonomous daemon with 4 cascading heartbeat tiers.

    Usage::

        engine = OpenClawEngine()
        daemon = HeartbeatDaemon(engine)
        await daemon.start()  # Runs until daemon.stop()

    Or via CLI::

        python cli.py daemon start
    """

    def __init__(self, engine: Any, config: HeartbeatConfig | None = None):
        self.engine = engine
        self.config = config or HeartbeatConfig.load()
        self.codex = engine.codex

        self.alert_router = AlertRouter(self.config, self.codex, engine.notifier)
        self.cron = CronScheduler(self.codex)
        self.proactive = ProactiveAgent(engine, self.config)
        self.healer = SelfHealer(self.codex, self.alert_router)

        self._running = False
        self._started_at: datetime | None = None
        self._pid_file = Path(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ) / "data" / "daemon.pid"

        # Tier execution counters
        self._tier_runs: dict[str, int] = {
            "PULSE": 0, "SCAN": 0, "INTEL": 0, "DAILY": 0,
        }

        # Action registry for cron jobs
        self._action_registry: dict[str, Callable] = {}

    async def start(self):
        """Start the daemon with all tier loops + cron + proactive agent."""
        if self._running:
            logger.warning("HeartbeatDaemon is already running")
            return

        self._write_pid()
        self._running = True
        self._started_at = datetime.now()
        self._register_default_crons()
        self._build_action_registry()

        logger.info(
            "HeartbeatDaemon starting — "
            f"PULSE={self.config.pulse_interval}s, "
            f"SCAN={self.config.scan_interval}s, "
            f"INTEL={self.config.intel_interval}s, "
            f"DAILY=24h"
        )

        try:
            await asyncio.gather(
                self._tier_loop("PULSE", self.config.pulse_interval, self._run_pulse),
                self._tier_loop("SCAN", self.config.scan_interval, self._run_scan),
                self._tier_loop("INTEL", self.config.intel_interval, self._run_intel),
                self._tier_loop("DAILY", 86400, self._run_daily),
                self._cron_loop(),
                self._proactive_loop(),
            )
        finally:
            self._cleanup_pid()

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        self._cleanup_pid()
        logger.info("HeartbeatDaemon stopped")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status summary."""
        uptime = 0.0
        if self._started_at:
            uptime = (datetime.now() - self._started_at).total_seconds()

        return {
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "uptime_seconds": round(uptime, 0),
            "tier_runs": self._tier_runs,
            "cron_jobs": len(self.cron.get_all()),
            "pending_actions": len(self.proactive.evaluate()) if self._running else 0,
        }

    # ================================================================== #
    #  Tier loops                                                          #
    # ================================================================== #

    async def _tier_loop(self, name: str, interval: int, func: Callable):
        """Generic tier loop with interval enforcement."""
        while self._running:
            start = time.monotonic()
            try:
                if not self._preflight(name):
                    await asyncio.sleep(60)
                    continue
                await func()
                self._tier_runs[name] += 1
            except Exception as e:
                logger.error(f"[{name}] tier error: {e}", exc_info=True)
            elapsed = time.monotonic() - start
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _run_pulse(self):
        """PULSE tier (5 min): WP sites + service ports + self-health."""
        checks: list[HealthCheck] = []

        # WordPress site checks
        if self.config.wordpress_domains:
            wp_checks = await wordpress_check.check_all_sites(
                self.config.wordpress_domains
            )
            checks.extend(wp_checks)

        # Service port checks
        if self.config.service_ports:
            svc_checks = await service_check.check_all_services(
                self.config.service_ports
            )
            checks.extend(svc_checks)

        # Self-health
        checks.append(self._self_health_check())

        for check in checks:
            check.checked_at = datetime.now()
            self.codex.log_health_check(check)

            if check.result in (CheckResult.DOWN, CheckResult.DEGRADED):
                healed = await self.healer.heal(check)
                if not healed:
                    await self.alert_router.route(self._check_to_alert(check))

        logger.info(
            f"[PULSE] Completed: {len(checks)} checks, "
            f"{sum(1 for c in checks if c.result != CheckResult.HEALTHY)} issues"
        )

    async def _run_scan(self):
        """SCAN tier (30 min): n8n + email + profiles + flush queued alerts."""
        checks: list[HealthCheck] = []

        checks.extend(await n8n_check.check_workflows())
        checks.extend(await email_check.check_inbox())
        checks.extend(await profile_check.check_profiles(
            self.codex,
            self.engine.sentinel,
            self.config.profile_stale_days,
            self.config.score_drift_threshold,
        ))

        for check in checks:
            check.checked_at = datetime.now()
            self.codex.log_health_check(check)

            if check.result != CheckResult.HEALTHY:
                await self.alert_router.route(self._check_to_alert(check))

        # Flush any queued alerts from quiet hours
        flushed = await self.alert_router.flush_queued()

        logger.info(
            f"[SCAN] Completed: {len(checks)} checks, "
            f"{flushed} queued alerts flushed"
        )

    async def _run_intel(self):
        """INTEL tier (6 hr): GSC traffic + recommendations."""
        checks = await seo_check.check_traffic(self.config.gsc_drop_threshold)

        for check in checks:
            check.checked_at = datetime.now()
            self.codex.log_health_check(check)

            if check.result != CheckResult.HEALTHY:
                await self.alert_router.route(self._check_to_alert(check))

        # Re-run MarketOracle for fresh recommendations
        try:
            recs = self.engine.prioritize()
            if recs:
                top = recs[:3]
                logger.info(
                    f"[INTEL] Top platform recommendations: "
                    f"{[r.platform_id for r in top]}"
                )
                self.codex.log_action(
                    action_type="intel_recommendations",
                    target="market_oracle",
                    description=f"Top 3: {', '.join(r.platform_id for r in top)}",
                    result="success",
                )
        except Exception as e:
            logger.debug(f"[INTEL] Oracle failed: {e}")

        logger.info(f"[INTEL] Completed: {len(checks)} SEO checks")

    async def _run_daily(self):
        """DAILY tier (24 hr at 7 AM EST): full report + security."""
        # Wait for target hour
        await self._wait_for_daily_hour()

        # Security checks
        sec_checks = await security_check.check_plugin_security()
        for check in sec_checks:
            check.checked_at = datetime.now()
            self.codex.log_health_check(check)

        # Compile daily report
        report = self._compile_daily_report()
        await self.alert_router.route(Alert(
            severity=AlertSeverity.INFO,
            source="daily_report",
            title="Daily Empire Health Report",
            message=report,
        ))

        logger.info(f"[DAILY] Report sent, {len(sec_checks)} security checks completed")

    # ================================================================== #
    #  Cron + Proactive loops                                              #
    # ================================================================== #

    async def _cron_loop(self):
        """Execute due cron jobs every 60 seconds."""
        while self._running:
            try:
                due = self.cron.get_due_jobs()
                for job in due:
                    try:
                        await self.cron.execute_job(job, self._action_registry)
                    except Exception as e:
                        logger.error(f"Cron job '{job.name}' failed: {e}")
            except Exception as e:
                logger.error(f"Cron loop error: {e}")
            await asyncio.sleep(60)

    async def _proactive_loop(self):
        """Evaluate proactive actions every 15 minutes."""
        while self._running:
            try:
                actions = self.proactive.evaluate()
                for action in actions:
                    if action.requires_approval:
                        # Log for human review, don't execute
                        self.codex.log_action(
                            action.action_type,
                            action.target,
                            action.description,
                            "pending_approval",
                            autonomous=False,
                        )
                        continue

                    # Auto-approved actions: execute
                    try:
                        await self._execute_proactive_action(action)
                    except Exception as e:
                        logger.error(
                            f"Proactive action {action.action_type} failed: {e}"
                        )
            except Exception as e:
                logger.error(f"Proactive loop error: {e}")

            await asyncio.sleep(900)  # 15 min

    async def _execute_proactive_action(self, action):
        """Execute an auto-approved proactive action."""
        if action.action_type == "verify_email":
            if self.engine.email_verifier.is_configured:
                ok = await self.engine.email_verifier.auto_verify(
                    action.target, timeout_seconds=120,
                )
                result = "verified" if ok else "failed"
                self.codex.log_action(
                    "verify_email", action.target,
                    action.description, result,
                )

        elif action.action_type == "retry_signup":
            try:
                result = await self.engine.signup_async(action.target)
                status = "success" if result.success else "failed"
                self.codex.log_action(
                    "retry_signup", action.target,
                    action.description, status,
                )
            except Exception as e:
                self.codex.log_action(
                    "retry_signup", action.target,
                    action.description, f"error: {e}",
                )

        elif action.action_type == "session_cleanup":
            cleared = await self.healer.clear_expired_sessions()
            self.codex.log_action(
                "session_cleanup", "sessions",
                f"Cleared {cleared} stale sessions", "success",
            )

    # ================================================================== #
    #  Helpers                                                             #
    # ================================================================== #

    def _preflight(self, tier_name: str) -> bool:
        """Validate prerequisites before running a tier."""
        if not self._running:
            return False
        # Verify DB connectivity
        try:
            self.codex.get_stats()
            return True
        except Exception as e:
            logger.error(f"[{tier_name}] Preflight DB check failed: {e}")
            return False

    def _self_health_check(self) -> HealthCheck:
        """Check daemon's own health (DB, PID file)."""
        try:
            stats = self.codex.get_stats()
            return HealthCheck(
                name="openclaw:self",
                tier=HeartbeatTier.PULSE,
                result=CheckResult.HEALTHY,
                message=f"DB OK — {stats['total_accounts']} accounts tracked",
                details={"total_accounts": stats["total_accounts"]},
            )
        except Exception as e:
            return HealthCheck(
                name="openclaw:self",
                tier=HeartbeatTier.PULSE,
                result=CheckResult.DOWN,
                message=f"Self-health failed: {str(e)[:80]}",
            )

    @staticmethod
    def _check_to_alert(check: HealthCheck) -> Alert:
        """Convert a HealthCheck to an Alert."""
        if check.result == CheckResult.DOWN:
            severity = AlertSeverity.CRITICAL
        elif check.result == CheckResult.DEGRADED:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        return Alert(
            severity=severity,
            source=check.name.split(":")[0] + "_check",
            title=f"{check.name}: {check.result.value}",
            message=check.message,
            details=check.details,
        )

    def _compile_daily_report(self) -> str:
        """Compile a daily health summary."""
        stats = self.codex.get_stats()
        latest = self.codex.get_latest_checks()
        alert_stats = self.codex.get_alert_stats()

        healthy = sum(1 for c in latest.values() if c.get("result") == "healthy")
        degraded = sum(1 for c in latest.values() if c.get("result") == "degraded")
        down = sum(1 for c in latest.values() if c.get("result") == "down")

        lines = [
            "=== Daily Empire Health Report ===",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"Accounts: {stats['total_accounts']} total, {stats['active_accounts']} active, {stats['failed_signups']} failed",
            f"Avg Profile Score: {stats['avg_sentinel_score']}",
            f"CAPTCHA Auto-Solve Rate: {stats['captcha_auto_solve_rate']}%",
            "",
            f"Health Checks: {healthy} healthy, {degraded} degraded, {down} down",
            f"Alerts Today: {alert_stats.get('total_alerts', 0)} total, "
            f"{alert_stats.get('delivered', 0)} delivered, "
            f"{alert_stats.get('suppressed', 0)} suppressed",
            "",
            f"Daemon Uptime: {self._tier_runs}",
        ]
        return "\n".join(lines)

    async def _wait_for_daily_hour(self):
        """Wait until the configured daily hour."""
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(self.config.quiet_timezone)
            now = datetime.now(tz)
        except (ImportError, KeyError):
            now = datetime.now()

        if now.hour != self.config.daily_hour:
            # Calculate seconds until target hour
            target = now.replace(
                hour=self.config.daily_hour, minute=0, second=0, microsecond=0,
            )
            if target <= now:
                from datetime import timedelta
                target += timedelta(days=1)
            wait_seconds = (target - now).total_seconds()
            if wait_seconds > 0 and wait_seconds < 86400:
                logger.info(f"[DAILY] Waiting {wait_seconds:.0f}s until {self.config.daily_hour}:00")
                # Sleep in chunks so we can check _running
                while wait_seconds > 0 and self._running:
                    chunk = min(wait_seconds, 300)
                    await asyncio.sleep(chunk)
                    wait_seconds -= chunk

    def _register_default_crons(self):
        """Register the default cron jobs if not already in DB."""
        defaults = [
            ("morning-briefing", "daily 7am", "morning_briefing", {}),
            ("ops-check", "every 30m", "ops_check", {}),
            ("signup-retry", "every 6h", "signup_retry", {}),
            ("profile-refresh", "weekly mon", "profile_refresh", {}),
            ("stale-session-cleanup", "daily 3am", "session_cleanup", {}),
            ("email-inbox-sweep", "every 2h", "email_sweep", {}),
            ("platform-scout", "weekly wed", "platform_scout", {}),
            ("daily-report", "daily 8pm", "daily_report", {}),
        ]
        for name, schedule, action, params in defaults:
            self.cron.register(name, schedule, action, params)

    def _build_action_registry(self):
        """Build the action registry for cron job execution."""
        self._action_registry = {
            "morning_briefing": self._cron_morning_briefing,
            "ops_check": self._cron_ops_check,
            "signup_retry": self._cron_signup_retry,
            "profile_refresh": self._cron_profile_refresh,
            "session_cleanup": self._cron_session_cleanup,
            "email_sweep": self._cron_email_sweep,
            "platform_scout": self._cron_platform_scout,
            "daily_report": self._cron_daily_report,
        }

    # ─── Cron action handlers ───

    async def _cron_morning_briefing(self, **kwargs):
        report = self._compile_daily_report()
        await self.alert_router.route(Alert(
            severity=AlertSeverity.INFO,
            source="morning_briefing",
            title="Morning Briefing",
            message=report,
        ))
        return {"delivered": True}

    async def _cron_ops_check(self, **kwargs):
        await self._run_scan()
        return {"completed": True}

    async def _cron_signup_retry(self, **kwargs):
        count = await self.healer.retry_failed_signups(self.engine)
        return {"retried": count}

    async def _cron_profile_refresh(self, **kwargs):
        checks = await profile_check.check_profiles(
            self.codex, self.engine.sentinel,
            self.config.profile_stale_days,
            self.config.score_drift_threshold,
        )
        return {"checks": len(checks)}

    async def _cron_session_cleanup(self, **kwargs):
        cleared = await self.healer.clear_expired_sessions()
        return {"cleared": cleared}

    async def _cron_email_sweep(self, **kwargs):
        checks = await email_check.check_inbox()
        for check in checks:
            check.checked_at = datetime.now()
            self.codex.log_health_check(check)
        return {"checks": len(checks)}

    async def _cron_platform_scout(self, **kwargs):
        recs = self.engine.prioritize()
        return {"recommendations": len(recs)}

    async def _cron_daily_report(self, **kwargs):
        report = self._compile_daily_report()
        await self.alert_router.route(Alert(
            severity=AlertSeverity.INFO,
            source="daily_report",
            title="Daily Empire Report",
            message=report,
        ))
        return {"delivered": True}

    # ─── PID file management ───

    def _write_pid(self):
        """Write PID file."""
        try:
            self._pid_file.parent.mkdir(parents=True, exist_ok=True)
            self._pid_file.write_text(str(os.getpid()))
        except OSError as e:
            logger.debug(f"Cannot write PID file: {e}")

    def _cleanup_pid(self):
        """Remove PID file."""
        try:
            if self._pid_file.exists():
                self._pid_file.unlink()
        except OSError:
            pass

    @classmethod
    def is_running(cls) -> bool:
        """Check if a daemon process is already running."""
        pid_file = Path(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ) / "data" / "daemon.pid"
        if not pid_file.exists():
            return False
        try:
            pid = int(pid_file.read_text().strip())
            # Check if process is alive (cross-platform)
            import signal
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False
