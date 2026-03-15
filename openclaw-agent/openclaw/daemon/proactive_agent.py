"""ProactiveAgent — autonomous decision engine.

Evaluates current state from Codex + health checks and returns
prioritized action recommendations. Purely algorithmic (no LLM).

Autonomous signups are auto-approved with daily caps and platform
prioritization (no-CAPTCHA, no-email platforms first).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from openclaw.daemon.heartbeat_config import HeartbeatConfig
from openclaw.knowledge.platforms import get_platform
from openclaw.models import (
    AccountStatus,
    ProactiveAction,
)

logger = logging.getLogger(__name__)

# Actions that require human confirmation
_APPROVAL_REQUIRED = {"profile_content_change", "restart_service"}

# Actions the daemon executes immediately
_AUTO_APPROVED = {
    "verify_email", "retry_signup", "session_cleanup",
    "health_check", "report", "new_signup",
}


class ProactiveAgent:
    """Autonomous decision engine — determines the next best action.

    Decision tree (highest priority first):
        1. CRITICAL alerts -> self-heal (restart service, retry signup)
        2. Pending email verifications -> auto-verify
        3. Failed signups with transient errors -> retry
        4. High-priority unsigned platforms -> sign up (daily cap enforced)
        5. Low-score profiles -> re-optimize
        6. Stale profiles -> refresh
        7. Nothing to do -> log idle state
    """

    def __init__(self, engine: Any, config: HeartbeatConfig):
        # Avoid circular import — engine is OpenClawEngine
        self.engine = engine
        self.config = config
        self.codex = engine.codex

    def evaluate(self) -> list[ProactiveAction]:
        """Evaluate current state and return prioritized actions."""
        actions: list[ProactiveAction] = []

        actions.extend(self._check_email_verifications())
        actions.extend(self._check_failed_signups())
        actions.extend(self._check_unsigned_platforms())
        actions.extend(self._check_low_score_profiles())
        actions.extend(self._check_stale_profiles())
        actions.extend(self._check_session_cleanup())
        actions.extend(self._check_vibecoder_opportunities())

        # Sort by priority (1=highest)
        actions.sort(key=lambda a: a.priority)
        return actions

    def _check_email_verifications(self) -> list[ProactiveAction]:
        """Find accounts with pending email verification."""
        pending = self.codex.get_accounts_by_status(
            AccountStatus.EMAIL_VERIFICATION_PENDING
        )
        actions = []
        for account in pending:
            actions.append(ProactiveAction(
                action_type="verify_email",
                priority=2,
                target=account["platform_id"],
                description=f"Auto-verify email for {account['platform_name']}",
                requires_browser=False,
                requires_approval=False,
            ))
        return actions

    def _check_failed_signups(self) -> list[ProactiveAction]:
        """Find failed signups eligible for retry.

        Skips platforms that:
        - Are disabled
        - Were updated less than 1 hour ago
        - Have CAPTCHA (per knowledge base) — these need manual solving
        - Last failed due to CAPTCHA (per action_log) — runtime detection
        """
        failed = self.codex.get_accounts_by_status(AccountStatus.SIGNUP_FAILED)
        cutoff = (datetime.now() - timedelta(hours=1)).isoformat()

        # Build set of platforms whose last failure was CAPTCHA-related
        captcha_blocked = self._get_captcha_blocked_platforms()

        actions = []
        for account in failed:
            pid = account["platform_id"]

            # Skip disabled platforms
            platform = get_platform(pid)
            if platform and not platform.enabled:
                continue

            # Skip platforms with known CAPTCHA (per knowledge base)
            if platform:
                ct = getattr(platform, "captcha_type", None)
                captcha_none = (
                    ct is None
                    or (hasattr(ct, "value") and ct.value == "none")
                    or str(ct) == "CaptchaType.NONE"
                )
                if not captcha_none:
                    logger.debug(
                        f"Skipping retry {pid}: has {ct} (knowledge base)"
                    )
                    continue

            # Skip platforms whose last attempt failed due to CAPTCHA
            if pid in captcha_blocked:
                logger.debug(
                    f"Skipping retry {pid}: last failure was CAPTCHA-related"
                )
                continue

            # Only retry if last attempt was > 1 hour ago
            updated_at = account.get("updated_at", "")
            if updated_at and updated_at > cutoff:
                continue

            # Check if retry engine allows it
            try:
                if self.engine.retry_engine.should_retry(pid):
                    actions.append(ProactiveAction(
                        action_type="retry_signup",
                        priority=3,
                        target=pid,
                        description=f"Retry failed signup for {account['platform_name']}",
                        requires_browser=True,
                        requires_approval=False,
                    ))
            except (AttributeError, TypeError):
                # RetryEngine might not have should_retry — check history instead
                fail_steps = self.codex.get_failed_steps(pid)
                if len(fail_steps) < 3:  # Less than 3 failures = still retryable
                    actions.append(ProactiveAction(
                        action_type="retry_signup",
                        priority=3,
                        target=pid,
                        description=f"Retry failed signup for {account['platform_name']}",
                        requires_browser=True,
                        requires_approval=False,
                    ))

        return actions

    def _get_captcha_blocked_platforms(self) -> set[str]:
        """Return platform IDs whose last signup/retry failed due to CAPTCHA.

        Scans recent action_log for CAPTCHA-related failure keywords in the
        description or result fields.
        """
        _CAPTCHA_KEYWORDS = (
            "captcha", "recaptcha", "hcaptcha", "turnstile",
            "CAPTCHA", "reCAPTCHA", "hCaptcha", "Turnstile",
        )
        blocked: set[str] = set()
        try:
            history = self.codex.get_action_history(limit=200)
            # Track the most recent result per platform for signup/retry actions
            seen: set[str] = set()
            for h in history:
                if h.get("action_type") not in ("new_signup", "retry_signup"):
                    continue
                target = h.get("target", "")
                if target in seen:
                    continue
                seen.add(target)
                # Check description + result for CAPTCHA keywords
                desc = h.get("description", "")
                result = h.get("result", "")
                text = f"{desc} {result}"
                if any(kw in text for kw in _CAPTCHA_KEYWORDS):
                    blocked.add(target)
        except Exception:
            pass
        return blocked

    def _check_unsigned_platforms(self) -> list[ProactiveAction]:
        """Find high-priority platforms not yet signed up.

        Prioritizes platforms that are easiest to automate:
        1. No CAPTCHA + no email verification (highest priority)
        2. No CAPTCHA + email required
        3. Has CAPTCHA (lowest priority for autonomous)

        Enforces daily signup cap from config.
        """
        # Check daily cap
        signups_today = self._count_todays_signups()
        remaining = self.config.max_signups_per_day - signups_today
        if remaining <= 0:
            logger.debug(
                f"Daily signup cap reached ({signups_today}/{self.config.max_signups_per_day})"
            )
            return []

        try:
            recs = self.engine.prioritize()
        except Exception:
            return []

        # Build set of platforms with previous successful signups from action_log
        # (safety net in case account records are lost during container rebuilds)
        previously_succeeded = set()
        try:
            history = self.codex.get_action_history(limit=500)
            for h in history:
                if (
                    h.get("action_type") == "new_signup"
                    and h.get("result") == "success"
                ):
                    previously_succeeded.add(h.get("target", ""))
        except Exception:
            pass

        # Score each recommendation by automation difficulty
        candidates = []
        for rec in recs:
            account = self.codex.get_account(rec.platform_id)
            if account and account.get("status") not in (
                AccountStatus.NOT_STARTED.value,
                None,
            ):
                continue

            # Extra guard: skip if action_log shows a previous success
            if rec.platform_id in previously_succeeded:
                logger.debug(
                    f"Skipping {rec.platform_id}: previous successful signup in action_log"
                )
                continue

            platform = get_platform(rec.platform_id)
            if not platform:
                continue

            # Skip disabled platforms (broken URLs, non-existent domains, etc.)
            if not getattr(platform, "enabled", True):
                continue

            # Compute automation score (lower = easier)
            has_captcha = getattr(platform, "captcha_type", None)
            captcha_none = (
                has_captcha is None
                or (hasattr(has_captcha, "value") and has_captcha.value == "none")
                or str(has_captcha) == "CaptchaType.NONE"
            )
            needs_email = getattr(platform, "requires_email_verification", True)

            if captcha_none and not needs_email:
                difficulty = 0  # Easiest — no captcha, no email
            elif captcha_none and needs_email:
                difficulty = 1  # Medium — no captcha, but needs email
            else:
                difficulty = 2  # Hardest — has captcha

            candidates.append((difficulty, rec))

        # Sort by difficulty (easiest first), then by oracle score
        candidates.sort(key=lambda x: (x[0], -x[1].score))

        actions = []
        for difficulty, rec in candidates[:remaining]:
            difficulty_label = ["easy", "medium", "hard"][difficulty]
            actions.append(ProactiveAction(
                action_type="new_signup",
                priority=4,
                target=rec.platform_id,
                description=(
                    f"Sign up for {rec.platform_name} "
                    f"(score={rec.score:.0f}, difficulty={difficulty_label})"
                ),
                requires_browser=True,
                requires_approval=False,
                params={
                    "priority": rec.priority.value,
                    "score": rec.score,
                    "difficulty": difficulty_label,
                },
            ))

        return actions

    def _count_todays_signups(self) -> int:
        """Count how many signup attempts were started today.

        Only counts 'starting' entries — each signup attempt creates exactly
        one 'starting' entry when it begins. Counting 'failed' or 'success'
        too would double-count since those are the completion of the same attempt.
        """
        try:
            history = self.codex.get_action_history(limit=200)
            today = datetime.now().strftime("%Y-%m-%d")
            return sum(
                1 for h in history
                if h.get("action_type") == "new_signup"
                and h.get("timestamp", "").startswith(today)
                and h.get("result") == "starting"
            )
        except Exception:
            return 0

    def _check_low_score_profiles(self) -> list[ProactiveAction]:
        """Find profiles below quality threshold.

        Skips platforms that have no profile fields (bio_max_length=0 and
        description_max_length=0) since enhancing them is pointless.
        """
        from openclaw.knowledge.platforms import get_platform

        profiles = self.codex.get_all_profiles()
        actions = []
        for profile in profiles:
            score = profile.get("sentinel_score", 0)
            grade = profile.get("grade", "F")
            if grade in ("D", "F") and score > 0:
                # Skip platforms that don't support profile content
                platform = get_platform(profile["platform_id"])
                if platform and platform.bio_max_length == 0 and platform.description_max_length == 0:
                    continue
                actions.append(ProactiveAction(
                    action_type="enhance_profile",
                    priority=5,
                    target=profile["platform_id"],
                    description=f"Enhance low-quality profile (grade {grade}, score {score:.0f})",
                    requires_browser=False,
                    requires_approval=False,
                    params={"current_score": score, "grade": grade},
                ))
        return actions

    def _check_stale_profiles(self) -> list[ProactiveAction]:
        """Find profiles not updated in N days."""
        cutoff = (
            datetime.now() - timedelta(days=self.config.profile_stale_days)
        ).isoformat()

        active = self.codex.get_accounts_by_status(AccountStatus.ACTIVE)
        actions = []
        for account in active:
            updated_at = account.get("updated_at", "")
            if updated_at and updated_at < cutoff:
                actions.append(ProactiveAction(
                    action_type="refresh_profile",
                    priority=6,
                    target=account["platform_id"],
                    description=f"Refresh stale profile for {account['platform_name']}",
                    requires_browser=True,
                    requires_approval=True,
                ))

        return actions

    def _check_session_cleanup(self) -> list[ProactiveAction]:
        """Check for stale session cookies."""
        from pathlib import Path
        sessions_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sessions"
        if not sessions_dir.exists():
            return []

        cutoff = datetime.now() - timedelta(days=30)
        stale_sessions = []
        for session_file in sessions_dir.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                if mtime < cutoff:
                    stale_sessions.append(session_file.stem)
            except OSError:
                pass

        if stale_sessions:
            return [ProactiveAction(
                action_type="session_cleanup",
                priority=7,
                target="sessions",
                description=f"Clean up {len(stale_sessions)} stale session cookie(s)",
                requires_browser=False,
                requires_approval=False,
                params={"stale_sessions": stale_sessions},
            )]

        return []
