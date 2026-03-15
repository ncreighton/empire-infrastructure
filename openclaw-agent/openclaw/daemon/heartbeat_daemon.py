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
from datetime import datetime, timedelta
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

    async def _notify_telegram(self, event_type: str, data: dict) -> None:
        """Push a notification to Telegram bot (best-effort)."""
        try:
            bot = getattr(self.engine, "telegram_bot", None)
            if bot:
                await bot.notify_if_not_muted(event_type, data)
        except Exception as e:
            logger.debug(f"Telegram notify failed (non-critical): {e}")

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

        # Start MissionDaemon alongside tier loops
        from openclaw.vibecoder.daemon.mission_daemon import MissionDaemon
        self._mission_daemon = MissionDaemon(self.engine.vibecoder)

        # Collect coroutines — Telegram bot added if token is configured
        coros = [
            self._tier_loop("PULSE", self.config.pulse_interval, self._run_pulse),
            self._tier_loop("SCAN", self.config.scan_interval, self._run_scan),
            self._tier_loop("INTEL", self.config.intel_interval, self._run_intel),
            self._tier_loop("DAILY", 86400, self._run_daily),
            self._cron_loop(),
            self._proactive_loop(),
            self._mission_daemon.start(),
        ]

        # Wire Telegram bot into daemon lifecycle
        telegram_bot = getattr(self.engine, "telegram_bot", None)
        if telegram_bot is not None:
            self._telegram_bot = telegram_bot
            coros.append(telegram_bot.run())
            logger.info("Telegram bot will start with daemon")

        try:
            await asyncio.gather(*coros)
        finally:
            self._cleanup_pid()

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        telegram_bot = getattr(self, "_telegram_bot", None)
        if telegram_bot is not None:
            await telegram_bot.stop()
        self._cleanup_pid()
        logger.info("HeartbeatDaemon stopped")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status summary."""
        uptime = 0.0
        if self._started_at:
            uptime = (datetime.now() - self._started_at).total_seconds()

        # API circuit breaker status (graceful fallback if not yet wired)
        from openclaw.browser.browser_manager import BrowserManager
        _api_available = (
            BrowserManager.api_available()
            if hasattr(BrowserManager, "api_available")
            else True
        )
        _api_circuit_error = (
            getattr(BrowserManager, "_api_circuit_error_msg", "")
            if not _api_available
            else ""
        )

        return {
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "uptime_seconds": round(uptime, 0),
            "tier_runs": self._tier_runs,
            "cron_jobs": len(self.cron.get_all()),
            "pending_actions": len(self.proactive.evaluate()) if self._running else 0,
            "api_available": _api_available,
            "api_circuit_error": _api_circuit_error,
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
                self.config.service_ports,
                service_hosts=self.config.service_hosts,
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

        issues = sum(1 for c in checks if c.result != CheckResult.HEALTHY)
        if issues > 0:
            await self._notify_telegram("health_check", {
                "tier": "SCAN",
                "checks": len(checks),
                "issues": issues,
            })

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

        issues = sum(1 for c in checks if c.result != CheckResult.HEALTHY)
        if issues > 0:
            await self._notify_telegram("health_check", {
                "tier": "INTEL",
                "checks": len(checks),
                "issues": issues,
            })

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

        issues = sum(1 for c in sec_checks if c.result != CheckResult.HEALTHY)
        await self._notify_telegram("health_check", {
            "tier": "DAILY",
            "checks": len(sec_checks),
            "issues": issues,
        })

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
                auto = [a for a in actions if not a.requires_approval]
                pending = [a for a in actions if a.requires_approval]

                logger.info(
                    f"[PROACTIVE] Evaluated: {len(actions)} total, "
                    f"{len(auto)} auto-approved, {len(pending)} pending approval"
                )

                # Deduplicate pending_approval logs: only log if not
                # already logged for this (action_type, target) in last 6h
                if pending:
                    recently_logged = set()
                    try:
                        history = self.codex.get_action_history(limit=200)
                        cutoff_6h = (
                            datetime.now() - timedelta(hours=6)
                        ).isoformat()
                        for h in history:
                            if (
                                h.get("result") == "pending_approval"
                                and h.get("timestamp", "") > cutoff_6h
                            ):
                                recently_logged.add(
                                    (h.get("action_type", ""), h.get("target", ""))
                                )
                    except Exception:
                        pass

                    for action in pending:
                        key = (action.action_type, action.target)
                        if key not in recently_logged:
                            self.codex.log_action(
                                action.action_type,
                                action.target,
                                action.description,
                                "pending_approval",
                                autonomous=False,
                            )
                            # Notify about pending approval
                            await self._notify_telegram("approval_needed", {
                                "action": action.action_type,
                                "target": action.target,
                                "description": action.description,
                            })

                # Limit to 1 real browser signup per cycle to keep daemon responsive.
                # Pre-flight rejections (disabled, 403, already-complete) don't count.
                browser_signup_running = False

                # Check API circuit breaker — skip browser actions when credits depleted.
                from openclaw.browser.browser_manager import BrowserManager
                _browser_actions = {
                    "new_signup", "retry_signup", "apply_profile",
                    "human_activity", "publish_content",
                }
                _api_ok = BrowserManager.api_available() if hasattr(BrowserManager, "api_available") else True
                _circuit_notified = False

                for action in auto:
                    # Skip browser-dependent actions when the API circuit is open.
                    if action.action_type in _browser_actions and not _api_ok:
                        logger.info(
                            f"[PROACTIVE] Skipping {action.action_type} "
                            f"-> {action.target} (API circuit open)"
                        )
                        if not _circuit_notified:
                            _circuit_error = getattr(
                                BrowserManager, "_api_circuit_error_msg", ""
                            ) if hasattr(BrowserManager, "_api_circuit_error_msg") else ""
                            await self._notify_telegram("error", {
                                "message": "API credits depleted — browser actions paused",
                                "detail": _circuit_error,
                            })
                            _circuit_notified = True
                        continue

                    if action.action_type in ("new_signup", "retry_signup"):
                        if browser_signup_running:
                            logger.info(
                                f"[PROACTIVE] Deferring {action.action_type} "
                                f"-> {action.target} (1 signup per cycle)"
                            )
                            continue

                    logger.info(
                        f"[PROACTIVE] Executing: {action.action_type} "
                        f"-> {action.target}"
                    )
                    await self._notify_telegram("proactive_action", {
                        "action": action.action_type,
                        "target": action.target,
                        "description": action.description,
                    })
                    try:
                        was_real = await self._execute_proactive_action(action)
                        if was_real and action.action_type in (
                            "new_signup", "retry_signup"
                        ):
                            browser_signup_running = True
                    except Exception as e:
                        logger.error(
                            f"Proactive action {action.action_type} failed: {e}",
                            exc_info=True,
                        )
            except Exception as e:
                logger.error(f"Proactive loop error: {e}", exc_info=True)

            await asyncio.sleep(900)  # 15 min

    async def _execute_proactive_action(self, action) -> bool:
        """Execute an auto-approved proactive action.

        Returns True if a real browser signup ran (not a pre-flight rejection).
        """
        # Secondary circuit breaker guard — _proactive_loop catches most cases,
        # but _execute_proactive_action can also be called directly (e.g. from tests
        # or future callers), so we enforce the check here as well.
        _browser_actions = {
            "new_signup", "retry_signup", "apply_profile",
            "human_activity", "publish_content",
        }
        if action.action_type in _browser_actions:
            from openclaw.browser.browser_manager import BrowserManager
            if hasattr(BrowserManager, "api_available") and not BrowserManager.api_available():
                logger.info(
                    f"[PROACTIVE] API circuit open, skipping {action.action_type} "
                    f"-> {action.target}"
                )
                return False

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
                await self._notify_telegram("email_verified" if ok else "error", {
                    "platform": action.target,
                    "result": "verified" if ok else "failed",
                })
            return False  # Not a browser signup

        elif action.action_type == "new_signup":
            # Autonomous signup — generate password, execute pipeline
            import secrets
            import string
            password = os.environ.get("OPENCLAW_DEFAULT_PASSWORD")
            if not password:
                # Generate a strong random password
                chars = string.ascii_letters + string.digits + "!@#$%"
                password = "".join(secrets.choice(chars) for _ in range(20))

            credentials = {
                "password": password,
                "email": os.environ.get("OPENCLAW_EMAIL", ""),
            }

            logger.info(
                f"[PROACTIVE] Autonomous signup: {action.target} "
                f"(difficulty={action.params.get('difficulty', '?')})"
            )

            try:
                result = await self.engine.signup_async(
                    action.target, credentials
                )

                # Only log "starting" if the attempt actually ran
                # (not rejected by pre-flight checks like disabled/403/already-complete)
                preflight_reject = (
                    not result.success
                    and result.errors
                    and any(
                        kw in result.errors[0]
                        for kw in ("disabled", "Already signed up", "HTTP 403",
                                   "HTTP 404", "HTTP 502", "HTTP 503",
                                   "unreachable", "Unknown platform")
                    )
                )
                if not preflight_reject:
                    self.codex.log_action(
                        "new_signup", action.target,
                        action.description, "starting",
                    )

                status = "success" if result.success else "failed"
                detail = f"status={result.status.value}"
                if result.errors:
                    detail += f", error={result.errors[0][:100]}"
                if not preflight_reject:
                    self.codex.log_action(
                        "new_signup", action.target,
                        f"{action.description} -> {detail}", status,
                    )
                else:
                    logger.debug(
                        f"[PROACTIVE] Pre-flight reject {action.target}: "
                        f"{result.errors[0][:100]}"
                    )

                if result.success:
                    logger.info(
                        f"[PROACTIVE] Signup SUCCESS: {action.target} "
                        f"(profile_url={result.profile_url})"
                    )
                    # Alert on success
                    await self.alert_router.route(Alert(
                        severity=AlertSeverity.INFO,
                        source="proactive_agent",
                        title=f"Signup completed: {action.target}",
                        message=f"Successfully signed up for {action.target}. "
                                f"Status: {result.status.value}",
                    ))
                    await self._notify_telegram("signup_completed", {
                        "platform": action.target,
                        "profile_url": result.profile_url or "",
                    })
                elif not preflight_reject:
                    logger.warning(
                        f"[PROACTIVE] Signup FAILED: {action.target} "
                        f"({result.errors})"
                    )
                    await self._notify_telegram("signup_failed", {
                        "platform": action.target,
                        "error": result.errors[0][:100] if result.errors else "unknown",
                    })
                else:
                    logger.warning(
                        f"[PROACTIVE] Signup FAILED: {action.target} "
                        f"({result.errors})"
                    )

                return not preflight_reject

            except Exception as e:
                logger.error(f"[PROACTIVE] Signup error for {action.target}: {e}")
                self.codex.log_action(
                    "new_signup", action.target,
                    action.description, f"error: {str(e)[:200]}",
                )
                return True  # Error during real attempt = still counts

        elif action.action_type == "retry_signup":
            try:
                # Build credentials (same as new_signup)
                import secrets
                import string
                password = os.environ.get("OPENCLAW_DEFAULT_PASSWORD")
                if not password:
                    chars = string.ascii_letters + string.digits + "!@#$%"
                    password = "".join(secrets.choice(chars) for _ in range(20))
                credentials = {
                    "password": password,
                    "email": os.environ.get("OPENCLAW_EMAIL", ""),
                }
                result = await self.engine.signup_async(
                    action.target, credentials
                )

                # Detect pre-flight rejections
                preflight_reject = (
                    not result.success
                    and result.errors
                    and any(
                        kw in result.errors[0]
                        for kw in ("disabled", "Already signed up", "HTTP 403",
                                   "HTTP 404", "HTTP 502", "HTTP 503",
                                   "unreachable", "Unknown platform")
                    )
                )

                if not preflight_reject:
                    status = "success" if result.success else "failed"
                    detail = f"status={result.status.value}"
                    if result.errors:
                        detail += f", error={result.errors[0][:100]}"
                    self.codex.log_action(
                        "retry_signup", action.target,
                        f"{action.description} -> {detail}", status,
                    )
                    if result.success:
                        await self._notify_telegram("signup_completed", {
                            "platform": action.target,
                            "profile_url": result.profile_url or "",
                        })
                    else:
                        await self._notify_telegram("signup_failed", {
                            "platform": action.target,
                            "error": result.errors[0][:100] if result.errors else "unknown",
                        })
                else:
                    logger.debug(
                        f"[PROACTIVE] Pre-flight reject {action.target}: "
                        f"{result.errors[0][:100]}"
                    )
                return not preflight_reject
            except Exception as e:
                self.codex.log_action(
                    "retry_signup", action.target,
                    action.description, f"error: {e}",
                )
                return True  # Error during real attempt = still counts

        elif action.action_type == "enhance_profile":
            try:
                # Generate fresh profile content via ProfileSmith
                content = self.engine.generate_profile(action.target)
                # Score the NEW content and auto-enhance if below threshold
                score, content = self.engine.sentinel.score_and_enhance(
                    content, threshold=75.0
                )
                new_grade = score.grade.value
                new_score = score.total_score
                # Persist enhanced content + score
                self.codex.store_profile(content, score)
                self.codex.log_action(
                    "enhance_profile", action.target,
                    f"{action.description} -> grade {new_grade}, score {new_score:.0f}",
                    "success",
                )
                logger.info(
                    f"[PROACTIVE] Enhanced profile: {action.target} "
                    f"-> grade {new_grade}, score {new_score:.0f}"
                )
                await self._notify_telegram("profile_enhanced", {
                    "platform": action.target,
                    "grade": new_grade,
                    "score": f"{new_score:.0f}",
                })
            except Exception as e:
                logger.error(f"[PROACTIVE] Enhance failed for {action.target}: {e}")
                self.codex.log_action(
                    "enhance_profile", action.target,
                    action.description, f"error: {str(e)[:200]}",
                )

        elif action.action_type == "apply_profile":
            try:
                from openclaw.automation.profile_applier import ProfileApplier
                applier = ProfileApplier(codex=self.codex)
                result = await applier.apply_profile(action.target)
                status = "success" if result.success else "failed"
                self.codex.log_action(
                    "apply_profile", action.target,
                    (
                        f"Applied {len(result.fields_applied)} fields "
                        f"{result.fields_applied}, "
                        f"failed {len(result.fields_failed)} {result.fields_failed}"
                    ),
                    status,
                )
                if result.success:
                    logger.info(
                        f"[PROACTIVE] Profile applied: {action.target} "
                        f"(fields={result.fields_applied})"
                    )
                else:
                    logger.warning(
                        f"[PROACTIVE] Profile apply incomplete: {action.target} "
                        f"(applied={result.fields_applied}, "
                        f"failed={result.fields_failed}, "
                        f"errors={result.errors})"
                    )
                await self._notify_telegram(
                    "apply_profile_completed" if result.success else "apply_profile_failed",
                    {
                        "platform": action.target,
                        "fields_applied": ", ".join(result.fields_applied) if result.fields_applied else "none",
                        "fields_failed": ", ".join(result.fields_failed) if result.fields_failed else "none",
                    },
                )
            except Exception as e:
                logger.error(
                    f"[PROACTIVE] apply_profile failed for {action.target}: {e}"
                )
                self.codex.log_action(
                    "apply_profile", action.target,
                    action.description, f"error: {str(e)[:200]}",
                )

        elif action.action_type == "human_activity":
            try:
                from openclaw.daemon.human_activity import HumanActivityEngine
                activity_engine = HumanActivityEngine(codex=self.codex)
                session = await activity_engine.run_session(action.target)
                status = "success" if session.activities_failed == 0 else "partial"
                if session.activities_completed == 0:
                    status = "failed" if session.activities_failed > 0 else "skipped"
                self.codex.log_action(
                    "human_activity", action.target,
                    (
                        f"Session: {session.activities_completed} done, "
                        f"{session.activities_failed} failed, "
                        f"{session.duration_seconds:.0f}s"
                    ),
                    status,
                )
                logger.info(
                    f"[PROACTIVE] Human activity: {action.target} — "
                    f"{session.activities_completed} ok, "
                    f"{session.activities_failed} failed, "
                    f"{session.duration_seconds:.0f}s"
                )
                await self._notify_telegram("human_activity_completed", {
                    "platform": action.target,
                    "activities": f"{session.activities_completed} done, {session.activities_failed} failed",
                    "duration": f"{session.duration_seconds:.0f}s",
                })
            except Exception as e:
                logger.error(
                    f"[PROACTIVE] human_activity failed for {action.target}: {e}",
                    exc_info=True,
                )
                self.codex.log_action(
                    "human_activity", action.target,
                    action.description, f"error: {str(e)[:200]}",
                )

        elif action.action_type == "session_cleanup":
            cleared = await self.healer.clear_expired_sessions()
            self.codex.log_action(
                "session_cleanup", "sessions",
                f"Cleared {cleared} stale sessions", "success",
            )

        elif action.action_type == "vibecoder_mission":
            await self._handle_vibecoder_mission(action)

        elif action.action_type == "vibecoder_discover_projects":
            await self._handle_vibecoder_discover(action)

        elif action.action_type == "publish_content":
            # Publishing always requires approval — this branch runs only when
            # the action has been explicitly approved by the operator.
            try:
                from openclaw.automation.content_publisher import (
                    ContentPublisher,
                    PublishableContent,
                )

                publisher = ContentPublisher(codex=self.codex)
                content_data = action.params.get("content", {})
                content = PublishableContent(
                    title=content_data.get("title", "Untitled"),
                    description=content_data.get("description", ""),
                    price=float(content_data.get("price", 0.0)),
                    currency=content_data.get("currency", "USD"),
                    category=content_data.get("category", ""),
                    tags=content_data.get("tags", []),
                    file_path=content_data.get("file_path", ""),
                    cover_image_path=content_data.get("cover_image_path", ""),
                    preview_text=content_data.get("preview_text", ""),
                    extra_fields=content_data.get("extra_fields", {}),
                )
                result = await publisher.publish(action.target, content)
                status = "success" if result.success else "failed"
                listing_detail = (
                    result.published_url
                    or ("pending review" if result.needs_review else "no url")
                )
                self.codex.log_action(
                    "publish_content",
                    action.target,
                    f"Published: {content.title} -> {listing_detail}",
                    status,
                )
                if result.success:
                    logger.info(
                        f"[PROACTIVE] Content published: {action.target} "
                        f"'{content.title}' -> {listing_detail}"
                    )
                    await self.alert_router.route(Alert(
                        severity=AlertSeverity.INFO,
                        source="proactive_agent",
                        title=f"Content published: {action.target}",
                        message=(
                            f"Published '{content.title}' on {action.target}. "
                            f"URL: {listing_detail}"
                        ),
                    ))
                else:
                    logger.warning(
                        f"[PROACTIVE] Content publish failed: {action.target} "
                        f"— {result.errors}"
                    )
                await self._notify_telegram(
                    "publish_completed" if result.success else "publish_failed",
                    {
                        "platform": action.target,
                        "title": content.title,
                        "url": result.published_url or "pending review",
                    },
                )
            except Exception as e:
                logger.error(
                    f"[PROACTIVE] publish_content failed for {action.target}: {e}",
                    exc_info=True,
                )
                self.codex.log_action(
                    "publish_content",
                    action.target,
                    action.description,
                    f"error: {str(e)[:200]}",
                )

        return False  # Non-signup actions don't count

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
            # Product Factory — autonomous skill/workflow generation
            ("product-factory-daily", "daily 4am", "factory_daily_run",
             {"skills": 2, "workflows": 3}),
            ("product-factory-performance", "daily 2pm", "factory_performance", {}),
            ("product-factory-weekly-bundle", "weekly sun", "factory_weekly_bundle", {}),
            # Model Router — expire stale step promotions
            ("step-promotion-expiry", "daily 5am", "expire_step_promotions", {"days": 7}),
            # VibeCoder — auto-discover projects weekly
            ("vibecoder-discover", "weekly thu", "vibecoder_discover_all", {}),
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
            # Product Factory actions
            "factory_daily_run": self._cron_factory_daily_run,
            "factory_performance": self._cron_factory_performance,
            "factory_weekly_bundle": self._cron_factory_weekly_bundle,
            # Model Router
            "expire_step_promotions": self._cron_expire_step_promotions,
            # VibeCoder
            "vibecoder_discover_all": self._cron_vibecoder_discover_all,
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

    # ─── Product Factory cron actions ───

    async def _run_factory_subprocess(self, script: str, args: list[str]) -> dict:
        """Run a Product Factory script as a subprocess.

        The factory scripts live in the Supercharger workspace (read-only mount)
        but write output to /app/data/factory/ (writable Docker volume).
        """
        import asyncio as _asyncio

        # Locate factory script — try Supercharger mount first, then fallback
        factory_paths = [
            "/app/supercharger/workspace/skills/product-factory",
            "/app/data/factory",
        ]
        script_path = None
        for base in factory_paths:
            candidate = f"{base}/{script}"
            if Path(candidate).exists():
                script_path = candidate
                break

        if not script_path:
            msg = f"Factory script not found: {script} (searched {factory_paths})"
            logger.error(f"[FACTORY] {msg}")
            return {"error": msg}

        cmd = ["python3", script_path] + args
        logger.info(f"[FACTORY] Running: {' '.join(cmd)}")

        try:
            proc = await _asyncio.create_subprocess_exec(
                *cmd,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            stdout, stderr = await _asyncio.wait_for(
                proc.communicate(), timeout=600,  # 10 min max
            )
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()

            if stdout_text:
                for line in stdout_text.split("\n")[-20:]:
                    logger.info(f"[FACTORY] {line}")

            if proc.returncode != 0:
                logger.error(f"[FACTORY] Exit code {proc.returncode}: {stderr_text[-500:]}")
                return {
                    "success": False,
                    "exit_code": proc.returncode,
                    "stderr": stderr_text[-500:],
                    "stdout": stdout_text[-500:],
                }

            return {
                "success": True,
                "exit_code": 0,
                "stdout": stdout_text[-1000:],
            }

        except asyncio.TimeoutError:
            logger.error("[FACTORY] Subprocess timed out after 600s")
            try:
                proc.kill()
            except Exception:
                pass
            return {"error": "timeout", "success": False}
        except Exception as e:
            logger.error(f"[FACTORY] Subprocess error: {e}", exc_info=True)
            return {"error": str(e)[:200], "success": False}

    async def _cron_factory_daily_run(self, skills: int = 2, workflows: int = 3, **kwargs):
        """Daily factory run: generate skills + workflows, publish to marketplaces."""
        logger.info(f"[FACTORY] Daily run starting: {skills} skills, {workflows} workflows")

        result = await self._run_factory_subprocess(
            "product_factory.py",
            ["run", "--skills", str(skills), "--workflows", str(workflows)],
        )

        # Send alert with results
        if result.get("success"):
            await self.alert_router.route(Alert(
                severity=AlertSeverity.INFO,
                source="product_factory",
                title="Product Factory Daily Run Complete",
                message=f"Generated {skills} skills + {workflows} workflows.\n"
                        f"Output: {result.get('stdout', '')[-300:]}",
            ))
        else:
            await self.alert_router.route(Alert(
                severity=AlertSeverity.WARNING,
                source="product_factory",
                title="Product Factory Daily Run Failed",
                message=f"Error: {result.get('error') or result.get('stderr', 'unknown')[:300]}",
            ))

        self.codex.log_action(
            "factory_daily_run", "product_factory",
            f"skills={skills}, workflows={workflows}",
            "success" if result.get("success") else "failed",
        )
        return result

    async def _cron_factory_performance(self, **kwargs):
        """Daily performance check: update metrics, extract learnings."""
        logger.info("[FACTORY] Performance check starting")

        result = await self._run_factory_subprocess(
            "product_factory.py", ["performance"],
        )

        self.codex.log_action(
            "factory_performance", "product_factory",
            "Performance metrics update",
            "success" if result.get("success") else "failed",
        )
        return result

    async def _cron_factory_weekly_bundle(self, **kwargs):
        """Weekly bundle: package accumulated products into Gumroad bundles."""
        logger.info("[FACTORY] Weekly bundle starting")

        result = await self._run_factory_subprocess(
            "factory_daemon.py", ["bundle"],
        )

        if result.get("success"):
            await self.alert_router.route(Alert(
                severity=AlertSeverity.INFO,
                source="product_factory",
                title="Product Factory Weekly Bundle Complete",
                message=f"Bundle packaged.\n{result.get('stdout', '')[-300:]}",
            ))

        self.codex.log_action(
            "factory_weekly_bundle", "product_factory",
            "Weekly Gumroad bundle creation",
            "success" if result.get("success") else "failed",
        )
        return result

    async def _cron_expire_step_promotions(self, **kwargs):
        """Expire stale Haiku→Sonnet step promotions (platforms change UIs)."""
        days = kwargs.get("days", 7)
        step_router = getattr(self.engine, "step_router", None)
        if step_router:
            expired = step_router.expire_promotions(days=days)
            logger.info(f"[MODEL-ROUTER] Expired {expired} stale step promotions (>{days}d)")
            return {"expired": expired}
        # Fallback: expire directly via codex
        self.codex.expire_old_promotions(days=days)
        logger.info(f"[MODEL-ROUTER] Expired stale step promotions via codex (>{days}d)")
        return {"expired": -1}

    async def _cron_vibecoder_discover_all(self, **kwargs):
        """Weekly: scan empire root and register unregistered projects."""
        import os as _os
        from pathlib import Path as _Path

        vibecoder = getattr(self.engine, "vibecoder", None)
        if not vibecoder:
            return {"error": "VibeCoder not available"}

        empire_root = _Path(_os.environ.get("EMPIRE_ROOT", "D:/Claude Code Projects"))
        if not empire_root.is_dir():
            return {"error": f"Empire root not found: {empire_root}"}

        registered = {p["project_id"] for p in vibecoder.list_projects()}
        _MARKERS = ("CLAUDE.md", "pyproject.toml", "package.json", "requirements.txt")
        _SKIP = {
            "node_modules", "__pycache__", ".git", "assets", "reports",
            "configs", "credentials", "docs", "prompts", "n8n", "launchers",
        }
        discovered = 0
        for child in empire_root.iterdir():
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            if child.name in _SKIP or child.name in registered:
                continue
            has_marker = any((child / m).exists() for m in _MARKERS)
            if has_marker:
                try:
                    vibecoder.register_project(child.name, str(child))
                    discovered += 1
                except Exception as e:
                    logger.debug(f"[VIBECODER] Failed to register {child.name}: {e}")

        logger.info(f"[VIBECODER] Discovered {discovered} new project(s)")
        self.codex.log_action(
            "vibecoder_discover_all", "vibecoder",
            f"Weekly discovery: {discovered} new project(s)",
            "success",
        )
        return {"discovered": discovered}

    # ─── VibeCoder integration ───

    async def _handle_vibecoder_mission(self, action) -> None:
        """Create or manage a VibeCoder mission from ProactiveAgent."""
        params = action.params or {}
        vibecoder = getattr(self.engine, "vibecoder", None)
        if not vibecoder:
            logger.warning("[PROACTIVE] VibeCoder not available on engine")
            return

        # Force-fail a stalled mission
        if params.get("action") == "force_fail":
            mission_id = params.get("mission_id", "")
            try:
                from openclaw.vibecoder.models import MissionStatus
                vibecoder.codex.update_mission_status(
                    mission_id, MissionStatus.FAILED,
                    errors=["Force-failed by ProactiveAgent: execution stalled"],
                )
                self.codex.log_action(
                    "vibecoder_mission", action.target,
                    f"Force-failed stalled mission {mission_id}",
                    "success",
                )
                logger.info(
                    f"[PROACTIVE] Force-failed stalled mission: {mission_id}"
                )
            except Exception as e:
                logger.error(f"[PROACTIVE] Force-fail failed: {e}")
            return

        # Create a new mission from health issue
        project_id = params.get("project_id", action.target)
        title = params.get("title", f"Auto-fix for {action.target}")
        description = params.get(
            "mission_description", action.description,
        )
        priority = params.get("priority", 5)

        try:
            mission = vibecoder.submit_mission(
                project_id=project_id,
                title=title,
                description=description,
                priority=priority,
                auto_deploy=False,  # Never auto-deploy from proactive agent
            )
            self.codex.log_action(
                "vibecoder_mission", project_id,
                f"Queued mission {mission.mission_id}: {title}",
                "success",
            )
            logger.info(
                f"[PROACTIVE] VibeCoder mission queued: {mission.mission_id} "
                f"({mission.scope.value}) — {title}"
            )

            # Notify via alert
            await self.alert_router.route(Alert(
                severity=AlertSeverity.INFO,
                source="vibecoder",
                title=f"Auto-mission created: {title}",
                message=(
                    f"Mission {mission.mission_id} queued for {project_id}. "
                    f"Scope: {mission.scope.value}. "
                    f"The MissionDaemon will execute it automatically."
                ),
            ))

        except Exception as e:
            logger.error(
                f"[PROACTIVE] Failed to create VibeCoder mission: {e}",
                exc_info=True,
            )
            self.codex.log_action(
                "vibecoder_mission", project_id,
                f"Failed to queue mission: {title}",
                f"error: {str(e)[:200]}",
            )

    async def _handle_vibecoder_discover(self, action) -> None:
        """Auto-discover and register empire projects in VibeCoder."""
        params = action.params or {}
        projects = params.get("projects", [])
        vibecoder = getattr(self.engine, "vibecoder", None)
        if not vibecoder:
            return

        empire_root = os.environ.get("EMPIRE_ROOT", "D:/Claude Code Projects")
        registered = 0
        errors = 0

        for project_id in projects:
            root_path = os.path.join(empire_root, project_id)
            if not os.path.isdir(root_path):
                continue
            try:
                info = vibecoder.register_project(project_id, root_path)
                registered += 1
                logger.info(
                    f"[PROACTIVE] Registered project: {project_id} "
                    f"({info.language}/{info.framework}, {info.total_files} files, "
                    f"deploy={info.deploy_target.value})"
                )
            except Exception as e:
                errors += 1
                logger.warning(
                    f"[PROACTIVE] Failed to register {project_id}: {e}"
                )

        self.codex.log_action(
            "vibecoder_discover_projects", "vibecoder",
            f"Discovered {registered} project(s), {errors} error(s)",
            "success" if errors == 0 else "partial",
        )

        if registered > 0:
            await self.alert_router.route(Alert(
                severity=AlertSeverity.INFO,
                source="vibecoder",
                title=f"Auto-discovered {registered} project(s)",
                message=(
                    f"Registered {registered} new project(s) in VibeCoder: "
                    f"{', '.join(projects[:5])}"
                ),
            ))

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
