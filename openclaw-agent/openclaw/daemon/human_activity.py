"""HumanActivityEngine — orchestrates organic activity sessions on platforms.

Makes accounts look alive by browsing feeds, viewing profiles, liking content,
and searching — all with human-like timing and session restore.

Design principles:
- Session restore skips re-login (uses saved cookies)
- Random activity selection weighted toward passive browsing
- Cooldown enforcement via action_log query
- Graceful per-activity failure (continue session on error)
- Sonnet model for all activities (varied layouts need strong vision)
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

from openclaw.browser.stealth import randomize_delay
from openclaw.knowledge.activity_playbooks import Activity, ActivityPlaybook, get_playbook_for_platform
from openclaw.knowledge.platforms import get_platform

if TYPE_CHECKING:
    from openclaw.forge.platform_codex import PlatformCodex
    from openclaw.browser.browser_manager import BrowserManager

logger = logging.getLogger(__name__)

# Sonnet for all activity agent runs — varied page layouts require strong vision
_ACTIVITY_MODEL = "claude-sonnet-4-20250514"

# Human-like inter-activity delay range (seconds)
_INTER_ACTIVITY_DELAY: tuple[int, int] = (10, 60)

# Default action_type logged to codex for activity sessions
_ACTION_TYPE = "human_activity"


@dataclass
class ActivitySession:
    """Record of a completed activity session on a single platform."""

    platform_id: str
    activities_completed: int
    activities_failed: int
    duration_seconds: float
    started_at: datetime
    completed_at: datetime | None = None
    errors: list[str] = field(default_factory=list)


class HumanActivityEngine:
    """Orchestrates organic human-like activity sessions across platforms.

    Usage::

        engine = HumanActivityEngine(codex=codex)
        session = await engine.run_session("gumroad")
        # Or run several platforms sequentially:
        sessions = await engine.run_batch(["gumroad", "producthunt", "huggingface"])
    """

    def __init__(
        self,
        codex: PlatformCodex | None = None,
        browser_manager: BrowserManager | None = None,
    ):
        self.codex = codex
        self._browser_manager = browser_manager

    # ================================================================== #
    #  Public API                                                          #
    # ================================================================== #

    async def run_session(self, platform_id: str) -> ActivitySession:
        """Run an organic activity session on a platform.

        Steps:
        1. Look up playbook for platform category
        2. Pick a weighted-random subset of activities
        3. Launch browser with session restore (saved cookies)
        4. Ensure logged in; skip platform if login fails
        5. Execute each activity with human-like inter-activity delays
        6. Save updated session cookies
        7. Log session result to codex action_log

        Returns an ActivitySession record regardless of success/failure.
        """
        started_at = datetime.now()
        session = ActivitySession(
            platform_id=platform_id,
            activities_completed=0,
            activities_failed=0,
            duration_seconds=0.0,
            started_at=started_at,
        )

        platform = get_platform(platform_id)
        if not platform:
            msg = f"Unknown platform: {platform_id}"
            logger.warning(f"[HumanActivity] {msg}")
            session.errors.append(msg)
            session.completed_at = datetime.now()
            session.duration_seconds = (session.completed_at - started_at).total_seconds()
            return session

        playbook = get_playbook_for_platform(platform_id)
        if not playbook:
            msg = f"No activity playbook for category '{platform.category.value}'"
            logger.warning(f"[HumanActivity] {platform_id}: {msg}")
            session.errors.append(msg)
            session.completed_at = datetime.now()
            session.duration_seconds = (session.completed_at - started_at).total_seconds()
            return session

        activities = self._select_activities(playbook)
        logger.info(
            f"[HumanActivity] Starting session on {platform_id} "
            f"({len(activities)} activities, category={platform.category.value})"
        )

        browser = self._get_browser_manager()
        try:
            await browser.launch(platform_id)

            # Attempt to ensure we're logged in
            logged_in = await self._ensure_logged_in(platform_id, browser)
            if not logged_in:
                msg = "Could not verify login; skipping activity session"
                logger.warning(f"[HumanActivity] {platform_id}: {msg}")
                session.errors.append(msg)
            else:
                for i, activity in enumerate(activities):
                    # Random pause between activities (human-like)
                    if i > 0:
                        delay = random.uniform(*_INTER_ACTIVITY_DELAY)
                        logger.debug(
                            f"[HumanActivity] {platform_id}: "
                            f"inter-activity pause {delay:.0f}s"
                        )
                        await asyncio.sleep(delay)

                    success = await self._execute_activity(activity, platform_id, browser)
                    if success:
                        session.activities_completed += 1
                    else:
                        session.activities_failed += 1

            # Persist updated cookies
            try:
                await browser.save_session(platform_id)
            except Exception as e:
                logger.debug(f"[HumanActivity] {platform_id}: save_session failed: {e}")

        except Exception as e:
            msg = f"Session error: {str(e)[:200]}"
            logger.error(f"[HumanActivity] {platform_id}: {msg}", exc_info=True)
            session.errors.append(msg)
        finally:
            try:
                await browser.close()
            except Exception:
                pass
            session.completed_at = datetime.now()
            session.duration_seconds = (
                session.completed_at - started_at
            ).total_seconds()

        self._log_session(session)

        logger.info(
            f"[HumanActivity] Session done: {platform_id} — "
            f"{session.activities_completed} ok, "
            f"{session.activities_failed} failed, "
            f"{session.duration_seconds:.0f}s"
        )
        return session

    async def run_batch(self, platform_ids: list[str]) -> list[ActivitySession]:
        """Run activity sessions on multiple platforms sequentially.

        Runs one platform at a time to avoid detection from simultaneous browser
        sessions on the same machine.
        """
        sessions: list[ActivitySession] = []
        for platform_id in platform_ids:
            try:
                session = await self.run_session(platform_id)
                sessions.append(session)
            except Exception as e:
                logger.error(
                    f"[HumanActivity] run_batch: unexpected error for {platform_id}: {e}",
                    exc_info=True,
                )
        return sessions

    def get_eligible_platforms(self, cooldown_hours: int | None = None) -> list[str]:
        """Return platform IDs eligible for an activity session.

        Eligibility criteria:
        - Account status is ACTIVE or PROFILE_COMPLETE (account exists and works)
        - Platform is enabled
        - Last human_activity session was more than cooldown_hours ago (or never)
        - Playbook exists for the platform's category

        Args:
            cooldown_hours: Override default cooldown. If None, uses each
                playbook's own cooldown_hours.
        """
        from openclaw.models import AccountStatus

        if not self.codex:
            return []

        # Collect active/complete accounts
        eligible_statuses = [AccountStatus.ACTIVE, AccountStatus.PROFILE_COMPLETE]
        candidates: list[str] = []
        for status in eligible_statuses:
            try:
                accounts = self.codex.get_accounts_by_status(status)
                candidates.extend(a["platform_id"] for a in accounts)
            except Exception:
                pass

        # Build a dict of last activity time per platform from action_log
        last_activity: dict[str, str] = {}
        try:
            history = self.codex.get_action_history(limit=500)
            for entry in history:
                if entry.get("action_type") != _ACTION_TYPE:
                    continue
                target = entry.get("target", "")
                if target and target not in last_activity:
                    last_activity[target] = entry.get("timestamp", "")
        except Exception:
            pass

        eligible: list[str] = []
        now = datetime.now()

        for platform_id in candidates:
            platform = get_platform(platform_id)
            if not platform or not getattr(platform, "enabled", True):
                continue

            playbook = get_playbook_for_platform(platform_id)
            if not playbook:
                continue

            effective_cooldown = (
                cooldown_hours if cooldown_hours is not None else playbook.cooldown_hours
            )
            cutoff = (now - timedelta(hours=effective_cooldown)).isoformat()

            last = last_activity.get(platform_id, "")
            if last and last > cutoff:
                # Still within cooldown window
                continue

            eligible.append(platform_id)

        return eligible

    # ================================================================== #
    #  Internal helpers                                                    #
    # ================================================================== #

    async def _execute_activity(
        self,
        activity: Activity,
        platform_id: str,
        browser: BrowserManager,
    ) -> bool:
        """Execute a single activity using browser-use agent.

        Returns True on success, False on failure. Never raises.
        """
        min_dur, max_dur = activity.duration_seconds
        timeout_secs = max_dur + 30  # Give agent a bit of extra time

        logger.debug(
            f"[HumanActivity] {platform_id}: executing '{activity.activity_type}'"
        )

        task = (
            f"{activity.description}\n\n"
            f"Important: Complete this naturally and don't rush. "
            f"You have {max_dur} seconds for this activity. "
            f"If you cannot find the relevant UI element, just scroll the page instead."
        )

        try:
            result = await browser.run_agent(
                task=task,
                platform_id=platform_id,
                max_steps=15,
                model=_ACTIVITY_MODEL,
            )
            success = result.get("success", False)
            if not success:
                err = result.get("error", result.get("final_text", "unknown"))
                logger.debug(
                    f"[HumanActivity] {platform_id}: activity '{activity.activity_type}' "
                    f"failed — {str(err)[:120]}"
                )
            return success
        except Exception as e:
            logger.warning(
                f"[HumanActivity] {platform_id}: activity '{activity.activity_type}' "
                f"exception — {str(e)[:120]}"
            )
            return False

    async def _ensure_logged_in(
        self,
        platform_id: str,
        browser: BrowserManager,
    ) -> bool:
        """Check whether the browser session is already authenticated.

        We attempt a lightweight check: navigate to the platform's login URL and
        see if the agent detects that we're already logged in (via saved session
        cookies restored in browser.launch). If we cannot confirm login we return
        True anyway to let the activities proceed — many browsing activities don't
        require authentication and the agent will handle any login prompts.
        """
        platform = get_platform(platform_id)
        if not platform:
            return False

        login_url = platform.login_url or platform.signup_url
        if not login_url:
            # No login URL defined — proceed anyway
            return True

        try:
            result = await browser.run_agent(
                task=(
                    f"Navigate to {login_url}. "
                    "Check if you are already logged in. "
                    "If you see a dashboard, feed, or user menu — you are logged in. "
                    "If you see a login form — you are NOT logged in. "
                    "Do NOT fill in any forms. Just check and report 'logged_in' or 'not_logged_in'."
                ),
                platform_id=platform_id,
                max_steps=5,
                model=_ACTIVITY_MODEL,
            )
            final_text = (result.get("final_text") or "").lower()
            # If agent couldn't tell, default to True (let activities proceed)
            if "not_logged_in" in final_text or "login form" in final_text:
                logger.debug(f"[HumanActivity] {platform_id}: not logged in")
                return False
            return True
        except Exception as e:
            logger.debug(
                f"[HumanActivity] {platform_id}: login check failed ({e}), proceeding anyway"
            )
            return True

    def _select_activities(self, playbook: ActivityPlaybook) -> list[Activity]:
        """Select a weighted-random subset of activities for this session.

        Count is sampled from playbook.activities_per_session range.
        Activities are sampled without replacement using weights.
        """
        min_count, max_count = playbook.activities_per_session
        count = random.randint(min_count, max_count)
        count = min(count, len(playbook.activities))

        weights = [a.weight for a in playbook.activities]
        selected: list[Activity] = []
        remaining = list(zip(weights, playbook.activities))

        for _ in range(count):
            if not remaining:
                break
            total = sum(w for w, _ in remaining)
            r = random.uniform(0, total)
            cumulative = 0.0
            chosen_idx = 0
            for i, (w, _) in enumerate(remaining):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = i
                    break
            _, chosen_activity = remaining.pop(chosen_idx)
            selected.append(chosen_activity)

        return selected

    def _get_browser_manager(self) -> BrowserManager:
        """Return the configured browser manager or create a new headless one."""
        if self._browser_manager:
            return self._browser_manager
        from openclaw.browser.browser_manager import BrowserManager
        return BrowserManager(headless=True)

    def _log_session(self, session: ActivitySession) -> None:
        """Persist session summary to codex action_log."""
        if not self.codex:
            return
        status = "success" if session.activities_failed == 0 else "partial"
        if session.activities_completed == 0 and session.activities_failed == 0:
            status = "skipped"
        elif session.activities_completed == 0:
            status = "failed"

        description = (
            f"Human activity session: "
            f"{session.activities_completed} completed, "
            f"{session.activities_failed} failed, "
            f"{session.duration_seconds:.0f}s"
        )
        if session.errors:
            description += f" — errors: {'; '.join(session.errors[:2])}"

        try:
            self.codex.log_action(
                action_type=_ACTION_TYPE,
                target=session.platform_id,
                description=description,
                result=status,
            )
        except Exception as e:
            logger.warning(f"[HumanActivity] Failed to log session to codex: {e}")
