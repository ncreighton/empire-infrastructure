"""VerificationAgent — post-signup verification and profile scoring."""

from __future__ import annotations

import logging
from datetime import datetime

from openclaw.browser.browser_manager import BrowserManager
from openclaw.forge.profile_sentinel import ProfileSentinel
from openclaw.knowledge.platforms import get_platform
from openclaw.models import (
    AccountStatus,
    SignupPlan,
)

logger = logging.getLogger(__name__)


class VerificationAgent:
    """Verify signup success and score the resulting profile."""

    def __init__(
        self,
        browser_manager: BrowserManager | None = None,
        sentinel: ProfileSentinel | None = None,
    ):
        self.browser = browser_manager or BrowserManager()
        self.sentinel = sentinel or ProfileSentinel()

    async def verify_signup(
        self,
        plan: SignupPlan,
    ) -> dict:
        """Verify that signup completed successfully.

        Returns:
            Dict with:
                - verified: bool
                - status: AccountStatus
                - profile_url: str
                - sentinel_score: SentinelScore
                - issues: list[str]
        """
        platform = get_platform(plan.platform_id)
        if not platform:
            return {
                "verified": False,
                "status": AccountStatus.SIGNUP_FAILED,
                "profile_url": "",
                "sentinel_score": None,
                "issues": [f"Unknown platform: {plan.platform_id}"],
            }

        issues = []
        verified = False
        status = AccountStatus.SIGNUP_FAILED
        profile_url = ""

        # Check step completion
        completed = sum(
            1 for s in plan.steps if s.status.value in ("completed", "skipped")
        )
        total = plan.total_steps
        completion_rate = completed / total if total > 0 else 0

        # Critical steps must be actually completed (not skipped)
        critical_types = {"navigate", "fill_field", "fill_textarea", "submit_form"}
        critical_steps = [
            s for s in plan.steps if s.step_type.value in critical_types
        ]
        critical_failed = [
            s for s in critical_steps if s.status.value not in ("completed",)
        ]

        if critical_failed:
            # Any critical step failure means signup didn't work
            failed_descs = [s.description for s in critical_failed]
            if len(critical_failed) >= len(critical_steps) / 2:
                status = AccountStatus.SIGNUP_FAILED
                issues.append(f"Critical steps failed: {', '.join(failed_descs[:3])}")
            else:
                status = AccountStatus.PROFILE_INCOMPLETE
                issues.append(f"Incomplete: {len(critical_failed)} critical steps failed")
        elif completion_rate >= 0.8:
            # Check for email verification pending
            email_steps = [
                s for s in plan.steps
                if s.step_type.value == "verify_email"
            ]
            if email_steps and email_steps[0].status.value == "needs_human":
                status = AccountStatus.EMAIL_VERIFICATION_PENDING
                issues.append("Email verification still pending")
            else:
                verified = True
                status = AccountStatus.PROFILE_COMPLETE
        elif completion_rate >= 0.5:
            status = AccountStatus.PROFILE_INCOMPLETE
            failed_steps = [
                s.description for s in plan.steps
                if s.status.value == "failed"
            ]
            issues.append(f"Incomplete: {len(failed_steps)} steps failed")
        else:
            status = AccountStatus.SIGNUP_FAILED
            issues.append(f"Only {completed}/{total} steps completed")

        # Build profile URL if template exists
        if platform.profile_url_template and plan.profile_content:
            profile_url = platform.profile_url_template.replace(
                "{username}", plan.profile_content.username
            )

        # Score the profile content
        sentinel_score = None
        if plan.profile_content:
            sentinel_score = self.sentinel.score(plan.profile_content)
            if sentinel_score.total_score < 60:
                issues.append(
                    f"Profile quality below threshold: "
                    f"{sentinel_score.total_score:.0f}/100 ({sentinel_score.grade.value})"
                )

        return {
            "verified": verified,
            "status": status,
            "profile_url": profile_url,
            "sentinel_score": sentinel_score,
            "issues": issues,
            "completion_rate": completion_rate,
            "verified_at": datetime.now().isoformat(),
        }

    async def verify_profile_live(
        self,
        platform_id: str,
        profile_url: str,
    ) -> dict:
        """Visit the live profile page and verify it's accessible.

        This uses the browser to check the profile is publicly visible.
        """
        platform = get_platform(platform_id)
        if not platform or not profile_url:
            return {"live": False, "reason": "No profile URL"}

        try:
            await self.browser.launch()
            result = await self.browser.run_agent(
                task=(
                    f"Navigate to {profile_url} and verify the profile page loads. "
                    f"Check if the username/display name is visible. "
                    f"Report whether the profile is publicly accessible."
                ),
                platform_id=platform_id,
                max_steps=5,
            )

            screenshot = await self.browser.take_screenshot(
                name=f"{platform_id}_profile_verify"
            )

            return {
                "live": result.get("success", False),
                "screenshot": screenshot,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Live verification failed for {platform_id}: {e}")
            return {"live": False, "reason": str(e)}
        finally:
            await self.browser.close()

    def quick_verify(self, plan: SignupPlan) -> tuple[bool, AccountStatus]:
        """Quick verification without browser — just check step statuses."""
        completed = sum(
            1 for s in plan.steps if s.status.value in ("completed", "skipped")
        )
        total = plan.total_steps
        rate = completed / total if total > 0 else 0

        # Critical steps must be actually completed (not just skipped)
        critical_types = {"navigate", "fill_field", "fill_textarea", "submit_form"}
        critical_failed = any(
            s.step_type.value in critical_types and s.status.value != "completed"
            for s in plan.steps
        )

        if critical_failed:
            if rate >= 0.5:
                return False, AccountStatus.PROFILE_INCOMPLETE
            return False, AccountStatus.SIGNUP_FAILED

        if rate >= 0.8:
            has_pending_email = any(
                s.step_type.value == "verify_email" and s.status.value == "needs_human"
                for s in plan.steps
            )
            if has_pending_email:
                return True, AccountStatus.EMAIL_VERIFICATION_PENDING
            return True, AccountStatus.PROFILE_COMPLETE

        if rate >= 0.5:
            return False, AccountStatus.PROFILE_INCOMPLETE

        return False, AccountStatus.SIGNUP_FAILED
