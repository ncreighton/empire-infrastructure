"""OpenClawEngine — master orchestrator wiring FORGE, AMPLIFY, browser, and agents.

Pattern: videoforge-engine/videoforge/videoforge_engine.py
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from openclaw.agents.executor_agent import ExecutorAgent
from openclaw.agents.monitor_agent import MonitorAgent
from openclaw.agents.planner_agent import PlannerAgent
from openclaw.agents.verification_agent import VerificationAgent
from openclaw.amplify.amplify_pipeline import AmplifyPipeline
from openclaw.browser.browser_manager import BrowserManager
from openclaw.browser.gologin_manager import GoLoginBrowserManager
from openclaw.browser.captcha_handler import CaptchaHandler
from openclaw.browser.proxy_manager import ProxyManager
from openclaw.browser.step_router import StepRouter
from openclaw.forge.market_oracle import MarketOracle
from openclaw.forge.platform_codex import PlatformCodex
from openclaw.forge.platform_scout import PlatformScout
from openclaw.forge.profile_sentinel import ProfileSentinel
from openclaw.forge.profile_smith import ProfileSmith
from openclaw.knowledge.platforms import get_all_platform_ids, get_platform
from openclaw.models import (
    AccountStatus,
    DashboardStats,
    OpenClawResult,
    OracleRecommendation,
    ProfileContent,
    ScoutResult,
    SentinelScore,
)
from openclaw.automation.email_verifier import EmailVerifier
from openclaw.automation.rate_limiter import RateLimiter
from openclaw.automation.retry_engine import RetryEngine
from openclaw.automation.profile_sync import ProfileSync
from openclaw.automation.webhook_notifier import WebhookNotifier
from openclaw.vibecoder import VibeCoderEngine

logger = logging.getLogger(__name__)


class OpenClawEngine:
    """Master orchestrator for the OpenClaw platform signup pipeline."""

    def __init__(self, db_path: str | None = None, headless: bool = True):
        # FORGE modules
        self.scout = PlatformScout()
        self.sentinel = ProfileSentinel()
        self.oracle = MarketOracle()
        self.smith = ProfileSmith()
        self.codex = PlatformCodex(db_path=db_path)

        # AMPLIFY
        self.amplify = AmplifyPipeline()

        # Browser + agents
        self.captcha = CaptchaHandler()
        self.proxy_manager = ProxyManager()
        self.step_router = StepRouter(self.codex)
        self.headless = headless

        # Automation
        self.email_verifier = EmailVerifier()
        self.rate_limiter = RateLimiter()
        self.retry_engine = RetryEngine()
        self.notifier = WebhookNotifier()
        self.profile_sync = ProfileSync(codex=self.codex, sentinel=self.sentinel)

        # VibeCoder — autonomous coding agent (wired to notifier for lifecycle events)
        self.vibecoder = VibeCoderEngine(
            db_path=db_path, notifier=self.notifier,
        )

        # Telegram bot — command center + notification sink
        from openclaw.comms.telegram_bot import OpenClawTelegramBot
        self.telegram_bot = OpenClawTelegramBot(self)
        self.notifier.telegram_bot = self.telegram_bot

    def signup(self, platform_id: str, credentials: dict[str, str] | None = None) -> OpenClawResult:
        """Full signup pipeline: Scout → Smith → Planner → AMPLIFY → Executor → Verify → Codex."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an async context — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run, self.signup_async(platform_id, credentials)
                ).result()
        else:
            return asyncio.run(self.signup_async(platform_id, credentials))

    async def signup_async(
        self,
        platform_id: str,
        credentials: dict[str, str] | None = None,
    ) -> OpenClawResult:
        """Async version of signup pipeline."""
        platform = get_platform(platform_id)
        if not platform:
            return OpenClawResult(
                platform_id=platform_id,
                platform_name="Unknown",
                success=False,
                errors=[f"Unknown platform: {platform_id}"],
            )

        # Skip disabled platforms
        if not getattr(platform, "enabled", True):
            return OpenClawResult(
                platform_id=platform_id,
                platform_name=platform.name,
                success=False,
                errors=[f"Platform {platform_id} is disabled"],
            )

        # Guard: refuse to re-signup for already-completed platforms
        existing = self.codex.get_account(platform_id)
        if existing and existing.get("status") in (
            AccountStatus.ACTIVE.value,
            AccountStatus.PROFILE_COMPLETE.value,
        ):
            logger.info(
                f"[{platform_id}] Already signed up (status={existing['status']}), skipping"
            )
            return OpenClawResult(
                platform_id=platform_id,
                platform_name=platform.name,
                success=True,
                status=AccountStatus(existing["status"]),
                profile_url=existing.get("profile_url", ""),
                username=existing.get("username", ""),
            )

        # Pre-flight: verify signup URL is reachable before burning API tokens
        try:
            import httpx
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=10.0,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as client:
                resp = await client.head(platform.signup_url)
                if resp.status_code in (403, 404, 502, 503):
                    logger.warning(
                        f"[{platform_id}] Signup URL returned {resp.status_code}: "
                        f"{platform.signup_url}"
                    )
                    return OpenClawResult(
                        platform_id=platform_id,
                        platform_name=platform.name,
                        success=False,
                        errors=[
                            f"Signup URL returned HTTP {resp.status_code} — "
                            f"skipping to avoid wasting API tokens"
                        ],
                    )
        except Exception as e:
            logger.warning(f"[{platform_id}] Pre-flight URL check failed: {e}")
            return OpenClawResult(
                platform_id=platform_id,
                platform_name=platform.name,
                success=False,
                errors=[f"Signup URL unreachable: {e}"],
            )

        # Rate limit check
        can_proceed, reason = self.rate_limiter.can_proceed(platform_id)
        if not can_proceed:
            wait_time = self.rate_limiter.wait_time(platform_id)
            logger.info(f"[{platform_id}] Rate limited: {reason}. Waiting {wait_time:.0f}s...")
            await asyncio.sleep(wait_time)

        self.rate_limiter.record_attempt(platform_id)

        result = OpenClawResult(
            platform_id=platform_id,
            platform_name=platform.name,
            started_at=datetime.now(),
        )

        # Notify signup started
        try:
            await self.notifier.notify_signup_started(platform_id, platform.name)
        except Exception as e:
            logger.debug(f"[{platform_id}] Webhook notify failed (non-critical): {e}")

        try:
            # Step 1: Scout — analyze platform readiness
            logger.info(f"[{platform_id}] Step 1: Scouting platform...")
            scout_result = self.scout.analyze(platform_id)
            result.scout_result = scout_result

            # Step 2: Smith — generate profile content
            logger.info(f"[{platform_id}] Step 2: Generating profile content...")
            profile_content = self.smith.generate_profile(platform_id)
            result.profile_content = profile_content

            # Step 3: Planner — create signup plan
            logger.info(f"[{platform_id}] Step 3: Planning signup steps...")
            planner = PlannerAgent()
            plan = planner.plan_signup(platform_id, profile_content)

            # Step 4: AMPLIFY — optimize the plan
            logger.info(f"[{platform_id}] Step 4: Running AMPLIFY pipeline...")
            amplify_result = self.amplify.amplify(plan)
            result.amplify_result = amplify_result

            if not amplify_result.ready:
                result.warnings.append(
                    f"AMPLIFY score {amplify_result.quality_score:.0f}/100 — "
                    "proceeding with caution"
                )

            # Step 5: Sentinel — pre-execution quality check
            logger.info(f"[{platform_id}] Step 5: Pre-execution quality check...")
            sentinel_score = self.sentinel.score(profile_content)
            result.sentinel_score = sentinel_score

            if sentinel_score.total_score < 50:
                # Auto-enhance if quality is low
                sentinel_score, profile_content = self.sentinel.score_and_enhance(
                    profile_content, threshold=60
                )
                result.sentinel_score = sentinel_score
                result.profile_content = profile_content
                plan.profile_content = profile_content

            # Step 6: Execute — run browser automation (wrapped in RetryEngine)
            logger.info(f"[{platform_id}] Step 6: Executing signup...")
            self.codex.upsert_account(
                platform_id, platform.name, AccountStatus.SIGNUP_IN_PROGRESS
            )

            # Use GoLogin Orbita when configured and NOT in Docker
            # GoLogin SDK spawns local Orbita browser which needs GUI — won't work in Docker
            import os
            use_gologin = (
                os.environ.get("GOLOGIN_API_TOKEN")
                and os.environ.get("GOLOGIN_PROFILE_ID")
                and not os.path.exists("/.dockerenv")
                and os.environ.get("OPENCLAW_BROWSER_MODE", "").lower() != "playwright"
            )
            if use_gologin:
                logger.info(f"[{platform_id}] Using GoLogin anti-detect browser")
                browser = GoLoginBrowserManager(headless=self.headless)
            else:
                logger.info(f"[{platform_id}] Using Playwright stealth browser")
                browser = BrowserManager(
                    headless=self.headless,
                    proxy_manager=self.proxy_manager,
                )
            monitor = MonitorAgent()
            executor = ExecutorAgent(
                browser_manager=browser,
                captcha_handler=self.captcha,
                monitor=monitor,
                step_router=self.step_router,
            )

            async def _execute():
                return await executor.execute_plan(plan, credentials)

            plan = await self.retry_engine.execute_with_retry(
                _execute, platform_id,
            )

            result.steps_completed = plan.completed_steps
            result.steps_total = plan.total_steps

            # Log monitor detections
            monitor_summary = monitor.get_summary()
            if monitor_summary["total_errors"] > 0:
                result.warnings.append(
                    f"Monitor detected {monitor_summary['total_errors']} error(s) during execution"
                )
            if monitor_summary["total_captchas"] > 0:
                result.warnings.append(
                    f"Monitor detected {monitor_summary['total_captchas']} CAPTCHA(s)"
                )

            # Step 7: Verify — check signup success
            logger.info(f"[{platform_id}] Step 7: Verifying signup...")
            verifier = VerificationAgent(sentinel=self.sentinel)
            verification = await verifier.verify_signup(plan)

            result.status = verification["status"]
            result.success = verification["verified"]
            result.profile_url = verification.get("profile_url", "")
            result.username = profile_content.username

            if verification.get("issues"):
                result.warnings.extend(verification["issues"])

            # Step 7b: Auto-verify email if needed
            if result.status == AccountStatus.EMAIL_VERIFICATION_PENDING:
                if self.email_verifier.is_configured:
                    logger.info(f"[{platform_id}] Step 7b: Auto-verifying email...")
                    email_ok = await self.email_verifier.auto_verify(
                        platform_id, timeout_seconds=180
                    )
                    if email_ok:
                        result.status = AccountStatus.ACTIVE
                        result.warnings.append("Email auto-verified successfully")
                    else:
                        result.warnings.append(
                            "Email auto-verification failed — check inbox manually"
                        )

            # Step 8: Persist to Codex
            logger.info(f"[{platform_id}] Step 8: Saving to Codex...")
            self.codex.upsert_account(
                platform_id=platform_id,
                platform_name=platform.name,
                status=result.status,
                username=result.username,
                profile_url=result.profile_url,
            )

            if credentials:
                self.codex.store_credentials(platform_id, credentials)

            self.codex.store_profile(profile_content, sentinel_score)

            # Log each step
            for step in plan.steps:
                self.codex.log_step(platform_id, step)

        except Exception as e:
            logger.error(f"[{platform_id}] Pipeline failed: {e}")
            result.success = False
            result.status = AccountStatus.SIGNUP_FAILED
            result.errors.append(str(e))

            self.codex.upsert_account(
                platform_id, platform.name, AccountStatus.SIGNUP_FAILED
            )
            self.rate_limiter.record_failure(platform_id)

        result.completed_at = datetime.now()
        if result.started_at and result.completed_at:
            result.duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()

        # Record rate limiter outcome
        if result.success:
            self.rate_limiter.record_success(platform_id)

        # Notify completion
        try:
            if result.success:
                await self.notifier.notify_signup_completed(
                    platform_id, platform.name, result.profile_url,
                    result.sentinel_score.total_score if result.sentinel_score else 0,
                    result.duration_seconds,
                )
            else:
                await self.notifier.notify_signup_failed(
                    platform_id, platform.name,
                    result.errors[0] if result.errors else "Unknown error",
                    result.steps_completed,
                )
        except Exception as e:
            logger.debug(f"[{platform_id}] Webhook notify failed (non-critical): {e}")

        logger.info(
            f"[{platform_id}] Pipeline complete: "
            f"success={result.success}, status={result.status.value}, "
            f"duration={result.duration_seconds:.1f}s"
        )
        return result

    async def signup_with_retry(
        self,
        platform_id: str,
        credentials: dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> OpenClawResult:
        """Signup with full retry logic — wraps signup_async with RetryEngine."""
        async def _attempt():
            return await self.signup_async(platform_id, credentials)

        # Temporarily override max_retries on the retry policy
        original_max = self.retry_engine.policy.max_retries
        self.retry_engine.policy.max_retries = max_retries
        try:
            return await self.retry_engine.execute_with_retry(
                _attempt, platform_id,
            )
        finally:
            self.retry_engine.policy.max_retries = original_max

    async def signup_batch(
        self,
        platform_ids: list[str],
        credentials: dict[str, str] | None = None,
        delay_seconds: int = 30,
    ) -> list[OpenClawResult]:
        """Sign up on multiple platforms sequentially with delays."""
        results = []
        for i, pid in enumerate(platform_ids):
            logger.info(f"Batch signup {i+1}/{len(platform_ids)}: {pid}")
            result = await self.signup_async(pid, credentials)
            results.append(result)

            if i < len(platform_ids) - 1:
                logger.info(f"Waiting {delay_seconds}s before next signup...")
                await asyncio.sleep(delay_seconds)

        return results

    def generate_profile(self, platform_id: str) -> ProfileContent:
        """Dry-run: generate profile content without browser execution."""
        return self.smith.generate_profile(platform_id)

    def score_profile(self, platform_id: str) -> SentinelScore | None:
        """Score an existing profile from the Codex."""
        stored = self.codex.get_profile(platform_id)
        if not stored:
            return None
        # get_profile() wraps the actual fields inside a "content" key
        profile_data = stored.get("content", stored)
        content = ProfileContent(
            platform_id=platform_id,
            username=profile_data.get("username", ""),
            display_name=profile_data.get("display_name", ""),
            email=profile_data.get("email", ""),
            bio=profile_data.get("bio", ""),
            tagline=profile_data.get("tagline", ""),
            description=profile_data.get("description", ""),
            website_url=profile_data.get("website_url", ""),
            avatar_path=profile_data.get("avatar_path", ""),
            banner_path=profile_data.get("banner_path", ""),
            social_links=profile_data.get("social_links", {}),
            seo_keywords=profile_data.get("seo_keywords", []),
        )
        return self.sentinel.score(content)

    def analyze_platform(self, platform_id: str) -> ScoutResult:
        """Analyze a platform for signup readiness."""
        return self.scout.analyze(platform_id)

    def prioritize(self) -> list[OracleRecommendation]:
        """Get prioritized list of platforms to sign up for."""
        completed = set()
        disabled = set()
        for account in self.codex.get_all_accounts():
            if account.get("status") in (
                AccountStatus.ACTIVE.value,
                AccountStatus.PROFILE_COMPLETE.value,
            ):
                completed.add(account["platform_id"])
        # Exclude disabled platforms
        for pid in get_all_platform_ids():
            platform = get_platform(pid)
            if platform and not getattr(platform, "enabled", True):
                disabled.add(pid)
        recs = self.oracle.prioritize_platforms(completed=completed)
        return [r for r in recs if r.platform_id not in disabled]

    def get_dashboard(self) -> DashboardStats:
        """Get aggregate dashboard statistics."""
        stats = self.codex.get_stats()
        recent = self.codex.get_recent_activity(limit=10)

        return DashboardStats(
            total_platforms=len(get_all_platform_ids()),
            active_accounts=stats.get("active", 0),
            pending_signups=stats.get("pending", 0),
            failed_signups=stats.get("failed", 0),
            avg_profile_score=stats.get("avg_score", 0),
            platforms_by_category=stats.get("by_category", {}),
            platforms_by_status=stats.get("by_status", {}),
            recent_activity=recent,
        )

    def sync_preview(
        self,
        changes: dict[str, str],
        platform_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Preview what a profile sync would change (no execution)."""
        plan = self.profile_sync.plan_sync(changes, platform_ids)
        return self.profile_sync.preview_sync(plan)

    async def sync(
        self,
        changes: dict[str, str],
        platform_ids: list[str] | None = None,
        browser: bool = False,
    ) -> dict[str, Any]:
        """Sync profile content across platforms.

        Args:
            changes: Field name → new value. Valid fields: bio, tagline, description, website_url.
            platform_ids: Target specific platforms. None = all active.
            browser: If True, use browser automation. Otherwise local-only (Codex update).
        """
        plan = self.profile_sync.plan_sync(changes, platform_ids)

        if browser:
            plan = await self.profile_sync.execute_sync(plan)
        else:
            self.profile_sync.update_local(plan)

        # Notify webhook
        try:
            fields_updated = [c.field_name for c in plan.changes]
            await self.notifier.notify_sync_completed(
                total_platforms=len(plan.target_platforms),
                succeeded=plan.succeeded,
                failed=plan.failed,
                fields_updated=fields_updated,
            )
        except Exception as e:
            logger.debug(f"Sync webhook notify failed (non-critical): {e}")

        return {
            "total_platforms": len(plan.target_platforms),
            "succeeded": plan.succeeded,
            "failed": plan.failed,
            "duration_seconds": plan.duration_seconds,
            "results": [
                {
                    "platform_id": r.platform_id,
                    "platform_name": r.platform_name,
                    "success": r.success,
                    "changes_applied": r.changes_applied,
                    "errors": r.errors,
                    "new_score": r.new_score,
                }
                for r in plan.results
            ],
        }

    def get_sync_status(self) -> dict[str, Any]:
        """Get profile consistency status across all active platforms."""
        return self.profile_sync.get_sync_status()

    def get_platform_status(self, platform_id: str) -> dict[str, Any]:
        """Get detailed status for a single platform."""
        account = self.codex.get_account(platform_id)
        platform = get_platform(platform_id)
        profile = self.codex.get_profile(platform_id)
        log = self.codex.get_signup_log(platform_id)

        return {
            "platform": {
                "id": platform_id,
                "name": platform.name if platform else "Unknown",
                "category": platform.category.value if platform else "",
                "complexity": platform.complexity.value if platform else "",
            },
            "account": account,
            "profile": profile,
            "signup_log": log,
        }
