"""ProactiveAgent — autonomous decision engine.

Evaluates current state from Codex + health checks and returns
prioritized action recommendations. Purely algorithmic (no LLM).

Actions requiring approval are logged for human review.
Auto-approved actions execute immediately.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from openclaw.daemon.heartbeat_config import HeartbeatConfig
from openclaw.models import (
    AccountStatus,
    ProactiveAction,
)

logger = logging.getLogger(__name__)

# Actions that require human confirmation
_APPROVAL_REQUIRED = {"new_signup", "profile_content_change", "restart_service"}

# Actions the daemon executes immediately
_AUTO_APPROVED = {"verify_email", "retry_signup", "session_cleanup", "health_check", "report"}


class ProactiveAgent:
    """Autonomous decision engine — determines the next best action.

    Decision tree (highest priority first):
        1. CRITICAL alerts → self-heal (restart service, retry signup)
        2. Pending email verifications → auto-verify
        3. Failed signups with transient errors → retry
        4. High-priority unsigned platforms → sign up
        5. Low-score profiles → re-optimize
        6. Stale profiles → refresh
        7. Nothing to do → log idle state
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
        """Find failed signups eligible for retry."""
        failed = self.codex.get_accounts_by_status(AccountStatus.SIGNUP_FAILED)
        cutoff = (datetime.now() - timedelta(hours=1)).isoformat()

        actions = []
        for account in failed:
            # Only retry if last attempt was > 1 hour ago
            updated_at = account.get("updated_at", "")
            if updated_at and updated_at > cutoff:
                continue

            # Check if retry engine allows it
            try:
                if self.engine.retry_engine.should_retry(account["platform_id"]):
                    actions.append(ProactiveAction(
                        action_type="retry_signup",
                        priority=3,
                        target=account["platform_id"],
                        description=f"Retry failed signup for {account['platform_name']}",
                        requires_browser=True,
                        requires_approval=False,
                    ))
            except (AttributeError, TypeError):
                # RetryEngine might not have should_retry — check history instead
                fail_steps = self.codex.get_failed_steps(account["platform_id"])
                if len(fail_steps) < 3:  # Less than 3 failures = still retryable
                    actions.append(ProactiveAction(
                        action_type="retry_signup",
                        priority=3,
                        target=account["platform_id"],
                        description=f"Retry failed signup for {account['platform_name']}",
                        requires_browser=True,
                        requires_approval=False,
                    ))

        return actions

    def _check_unsigned_platforms(self) -> list[ProactiveAction]:
        """Find high-priority platforms not yet signed up."""
        try:
            recs = self.engine.prioritize()
        except Exception:
            return []

        actions = []
        for rec in recs[:5]:  # Top 5 recommendations
            account = self.codex.get_account(rec.platform_id)
            if account and account.get("status") not in (
                AccountStatus.NOT_STARTED.value,
                None,
            ):
                continue

            actions.append(ProactiveAction(
                action_type="new_signup",
                priority=4,
                target=rec.platform_id,
                description=f"Sign up for {rec.platform_name} (priority: {rec.priority.value})",
                requires_browser=True,
                requires_approval=True,  # New signups require approval
                params={"priority": rec.priority.value, "score": rec.score},
            ))

        return actions

    def _check_low_score_profiles(self) -> list[ProactiveAction]:
        """Find profiles below quality threshold."""
        profiles = self.codex.get_all_profiles()
        actions = []
        for profile in profiles:
            score = profile.get("sentinel_score", 0)
            grade = profile.get("grade", "F")
            if grade in ("D", "F") and score > 0:
                actions.append(ProactiveAction(
                    action_type="enhance_profile",
                    priority=5,
                    target=profile["platform_id"],
                    description=f"Enhance low-quality profile (grade {grade}, score {score:.0f})",
                    requires_browser=False,
                    requires_approval=True,
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
