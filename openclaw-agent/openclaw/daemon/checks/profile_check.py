"""Profile health checks — stale profiles, score drift, low grades.

SCAN tier: runs every 30 minutes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from openclaw.forge.platform_codex import PlatformCodex
from openclaw.forge.profile_sentinel import ProfileSentinel
from openclaw.models import (
    AccountStatus,
    CheckResult,
    HealthCheck,
    HeartbeatTier,
    ProfileContent,
)

logger = logging.getLogger(__name__)


async def check_profiles(
    codex: PlatformCodex,
    sentinel: ProfileSentinel,
    stale_days: int = 30,
    drift_threshold: float = 10.0,
) -> list[HealthCheck]:
    """Check all active profiles for staleness, score drift, and low grades.

    Args:
        codex: PlatformCodex for data access.
        sentinel: ProfileSentinel for re-scoring.
        stale_days: Days before a profile is considered stale.
        drift_threshold: Score drop that triggers a warning.

    Returns:
        List of HealthCheck results for flagged profiles.
    """
    checks: list[HealthCheck] = []
    cutoff = (datetime.now() - timedelta(days=stale_days)).isoformat()

    # Get all active accounts
    active_accounts = codex.get_accounts_by_status(AccountStatus.ACTIVE)
    if not active_accounts:
        return [HealthCheck(
            name="profiles:overall",
            tier=HeartbeatTier.SCAN,
            result=CheckResult.HEALTHY,
            message="No active accounts to check",
        )]

    stale_count = 0
    low_grade_count = 0
    drift_count = 0

    for account in active_accounts:
        platform_id = account["platform_id"]

        # Check staleness
        updated_at = account.get("updated_at", "")
        if updated_at and updated_at < cutoff:
            stale_count += 1
            checks.append(HealthCheck(
                name=f"profile:{platform_id}:stale",
                tier=HeartbeatTier.SCAN,
                result=CheckResult.DEGRADED,
                message=f"Profile not updated since {updated_at[:10]} ({stale_days}+ days)",
                details={"platform_id": platform_id, "last_updated": updated_at},
            ))

        # Check stored profile quality
        profile = codex.get_profile(platform_id)
        if profile:
            stored_score = profile.get("sentinel_score", 0)
            grade = profile.get("grade", "F")

            # Flag low grades
            if grade in ("D", "F"):
                low_grade_count += 1
                checks.append(HealthCheck(
                    name=f"profile:{platform_id}:low_grade",
                    tier=HeartbeatTier.SCAN,
                    result=CheckResult.DEGRADED,
                    message=f"Profile grade {grade} (score: {stored_score:.0f})",
                    details={"platform_id": platform_id, "score": stored_score, "grade": grade},
                ))

            # Re-score and check for drift
            content = profile.get("content", {})
            if content:
                try:
                    pc = ProfileContent(
                        platform_id=platform_id,
                        username=content.get("username", ""),
                        bio=content.get("bio", ""),
                        tagline=content.get("tagline", ""),
                        description=content.get("description", ""),
                        website_url=content.get("website_url", ""),
                        avatar_path=content.get("avatar_path", ""),
                    )
                    current_score = sentinel.score(pc)
                    if stored_score - current_score.total_score > drift_threshold:
                        drift_count += 1
                        checks.append(HealthCheck(
                            name=f"profile:{platform_id}:drift",
                            tier=HeartbeatTier.SCAN,
                            result=CheckResult.DEGRADED,
                            message=f"Score drift: {stored_score:.0f} → {current_score.total_score:.0f}",
                            details={
                                "platform_id": platform_id,
                                "stored_score": stored_score,
                                "current_score": current_score.total_score,
                                "drift": round(stored_score - current_score.total_score, 1),
                            },
                        ))
                except Exception as e:
                    logger.debug(f"Failed to re-score {platform_id}: {e}")

    # Summary check
    total_issues = stale_count + low_grade_count + drift_count
    if total_issues == 0:
        checks.append(HealthCheck(
            name="profiles:overall",
            tier=HeartbeatTier.SCAN,
            result=CheckResult.HEALTHY,
            message=f"All {len(active_accounts)} active profiles healthy",
        ))
    else:
        checks.append(HealthCheck(
            name="profiles:overall",
            tier=HeartbeatTier.SCAN,
            result=CheckResult.DEGRADED,
            message=f"{total_issues} issue(s): {stale_count} stale, {low_grade_count} low grade, {drift_count} score drift",
            details={"stale": stale_count, "low_grade": low_grade_count, "drift": drift_count},
        ))

    return checks
