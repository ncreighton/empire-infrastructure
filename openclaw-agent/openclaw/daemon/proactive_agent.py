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
_APPROVAL_REQUIRED = {"profile_content_change", "restart_service", "publish_content"}

# Actions the daemon executes immediately (publish_content is also listed here but
# is always created with requires_approval=True, so it routes to the approval queue)
_AUTO_APPROVED = {
    "verify_email", "retry_signup", "session_cleanup",
    "health_check", "report", "new_signup", "vibecoder_mission",
    "vibecoder_discover_projects", "apply_profile", "human_activity",
    "publish_content",
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
        7. Human activity sessions -> keep accounts alive
        8. Publishing opportunities -> suggest content publishing (approval required)
        9. Nothing to do -> log idle state
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
        actions.extend(self._check_unapplied_profiles())
        actions.extend(self._check_stale_profiles())
        actions.extend(self._check_activity_sessions())
        actions.extend(self._check_session_cleanup())
        actions.extend(self._check_vibecoder_opportunities())
        actions.extend(self._check_publishing_opportunities())

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

    def _check_unapplied_profiles(self) -> list[ProactiveAction]:
        """Find profiles stored in the Codex that haven't been applied to the live platform.

        Criteria for triggering an apply_profile action:
        - Account status is PROFILE_INCOMPLETE or ACTIVE
        - Stored profile has grade B or higher (score >= 75)
        - No successful apply_profile action in the last 7 days

        Priority is 5 (same as enhance_profile) since applying content is only
        meaningful once a quality profile exists.
        """
        seven_days_ago = (
            datetime.now() - timedelta(days=7)
        ).isoformat()

        # Build set of platforms that had a successful apply in the last 7 days
        recently_applied: set[str] = set()
        try:
            history = self.codex.get_action_history(limit=500)
            for h in history:
                if (
                    h.get("action_type") == "apply_profile"
                    and h.get("result") == "success"
                    and h.get("timestamp", "") >= seven_days_ago
                ):
                    recently_applied.add(h.get("target", ""))
        except Exception:
            pass

        # Find profiles with score >= 75 (grade B or higher)
        profiles = self.codex.get_all_profiles()
        actions = []
        for profile in profiles:
            pid = profile.get("platform_id", "")
            if not pid:
                continue

            score = profile.get("sentinel_score", 0)
            grade = profile.get("grade", "F")

            # Only apply high-quality profiles (grade B = score >= 75)
            if score < 75 or grade in ("C", "D", "F"):
                continue

            # Skip if applied recently
            if pid in recently_applied:
                continue

            # Only attempt for accounts that are active or have incomplete profiles
            account = self.codex.get_account(pid)
            if not account:
                continue
            status = account.get("status", "")
            if status not in (
                AccountStatus.PROFILE_INCOMPLETE.value,
                AccountStatus.ACTIVE.value,
            ):
                continue

            # Skip platforms that have no editable profile fields
            from openclaw.knowledge.platforms import get_platform
            platform = get_platform(pid)
            if platform and platform.bio_max_length == 0 and platform.description_max_length == 0:
                continue

            actions.append(ProactiveAction(
                action_type="apply_profile",
                priority=5,
                target=pid,
                description=(
                    f"Apply stored profile to {account.get('platform_name', pid)} "
                    f"(grade {grade}, score {score:.0f})"
                ),
                requires_browser=True,
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

    def _check_activity_sessions(self) -> list[ProactiveAction]:
        """Find active accounts overdue for an organic activity session.

        Recommends human_activity actions for platforms that:
        - Have an ACTIVE or PROFILE_COMPLETE account
        - Are enabled
        - Have an activity playbook (category is supported)
        - Haven't had a successful human_activity session in the playbook's
          cooldown period (default 24 hours)

        Capped at 3 platforms per evaluation cycle to avoid flooding the
        proactive loop with browser-heavy tasks.
        """
        from openclaw.knowledge.activity_playbooks import get_playbook_for_platform
        from openclaw.models import AccountStatus

        _MAX_PER_CYCLE = 3
        _ELIGIBLE_STATUSES = {
            AccountStatus.ACTIVE.value,
            AccountStatus.PROFILE_COMPLETE.value,
        }

        # Build set of platforms and their last activity timestamp
        last_activity: dict[str, str] = {}
        try:
            history = self.codex.get_action_history(limit=500)
            for entry in history:
                if entry.get("action_type") != "human_activity":
                    continue
                target = entry.get("target", "")
                if target and target not in last_activity:
                    last_activity[target] = entry.get("timestamp", "")
        except Exception:
            pass

        # Collect eligible account platform IDs
        candidates: list[dict] = []
        for status_enum in (AccountStatus.ACTIVE, AccountStatus.PROFILE_COMPLETE):
            try:
                accounts = self.codex.get_accounts_by_status(status_enum)
                candidates.extend(accounts)
            except Exception:
                pass

        now = datetime.now()
        actions: list[ProactiveAction] = []

        for account in candidates:
            if len(actions) >= _MAX_PER_CYCLE:
                break

            pid = account.get("platform_id", "")
            if not pid:
                continue

            # Skip disabled platforms
            platform = get_platform(pid)
            if not platform or not getattr(platform, "enabled", True):
                continue

            # Skip platforms without a playbook
            playbook = get_playbook_for_platform(pid)
            if not playbook:
                continue

            # Enforce cooldown using playbook's cooldown_hours
            cutoff = (now - timedelta(hours=playbook.cooldown_hours)).isoformat()
            last = last_activity.get(pid, "")
            if last and last > cutoff:
                continue  # Still within cooldown window

            actions.append(ProactiveAction(
                action_type="human_activity",
                priority=6,
                target=pid,
                description=(
                    f"Run organic activity session on {account.get('platform_name', pid)} "
                    f"(cooldown={playbook.cooldown_hours}h, "
                    f"category={platform.category.value})"
                ),
                requires_browser=True,
                requires_approval=False,
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

    def _check_vibecoder_opportunities(self) -> list[ProactiveAction]:
        """Detect issues that VibeCoder can auto-fix via coding missions.

        Checks:
        1. Health checks with repeated failures → create bugfix missions
        2. Unregistered projects in the empire → auto-discover and register
        3. Failed VibeCoder missions → suggest retry with more context

        All missions are queued (not executed immediately) so the MissionDaemon
        picks them up in the next poll cycle.
        """
        actions: list[ProactiveAction] = []

        # Guard: skip if vibecoder isn't available
        vibecoder = getattr(self.engine, "vibecoder", None)
        if not vibecoder:
            return actions

        # 1. Auto-discover unregistered projects
        actions.extend(self._check_project_discovery(vibecoder))

        # 2. Health-issue → VibeCoder mission bridge
        actions.extend(self._check_health_to_mission(vibecoder))

        # 3. Retry stalled/failed missions
        actions.extend(self._check_stalled_missions(vibecoder))

        return actions

    def _check_project_discovery(self, vibecoder) -> list[ProactiveAction]:
        """Find empire project directories not yet registered in VibeCoder."""
        from pathlib import Path

        empire_root = Path(os.environ.get("EMPIRE_ROOT", "D:/Claude Code Projects"))
        if not empire_root.is_dir():
            return []

        # Get already-registered projects
        try:
            registered = {p["project_id"] for p in vibecoder.list_projects()}
        except Exception:
            registered = set()

        # Scan for project directories (must have CLAUDE.md or pyproject.toml or package.json)
        _PROJECT_MARKERS = (
            "CLAUDE.md", "pyproject.toml", "package.json", "Cargo.toml",
            "go.mod", "requirements.txt",
        )
        unregistered = []
        try:
            for child in empire_root.iterdir():
                if not child.is_dir():
                    continue
                if child.name.startswith(".") or child.name.startswith("_"):
                    continue
                # Skip common non-project dirs
                if child.name in (
                    "node_modules", "__pycache__", ".git", "assets",
                    "reports", "configs", "credentials", "docs",
                    "prompts", "n8n", "launchers",
                ):
                    continue
                # Check for project markers
                has_marker = any(
                    (child / marker).exists() for marker in _PROJECT_MARKERS
                )
                if has_marker and child.name not in registered:
                    unregistered.append(child.name)
        except OSError:
            return []

        if not unregistered:
            return []

        # Limit to 5 per cycle to avoid overwhelming
        batch = unregistered[:5]
        return [ProactiveAction(
            action_type="vibecoder_discover_projects",
            priority=8,
            target="vibecoder",
            description=f"Auto-discover {len(batch)} unregistered project(s): {', '.join(batch[:3])}{'...' if len(batch) > 3 else ''}",
            requires_browser=False,
            requires_approval=False,
            params={"projects": batch},
        )]

    def _check_health_to_mission(self, vibecoder) -> list[ProactiveAction]:
        """Convert repeated health failures into VibeCoder bugfix missions.

        If a service check has failed 3+ times in a row, the ProactiveAgent
        creates a VibeCoder mission to investigate and fix the issue.
        """
        actions: list[ProactiveAction] = []

        try:
            # Get recent health checks
            latest = self.codex.get_latest_checks()
        except Exception:
            return actions

        # Look for repeatedly failing services
        for name, check_data in latest.items():
            if check_data.get("result") != "down":
                continue

            # Count consecutive failures in history
            try:
                history = self.codex.get_health_history(name, limit=5)
                consecutive_failures = 0
                for h in history:
                    if h.get("result") in ("down", "degraded"):
                        consecutive_failures += 1
                    else:
                        break
            except Exception:
                consecutive_failures = 1

            if consecutive_failures < 3:
                continue

            # Check we haven't already created a mission for this
            try:
                recent_missions = vibecoder.list_missions(
                    status="queued", limit=20,
                )
                already_queued = any(
                    m.get("title", "").startswith(f"[auto] Fix {name}")
                    for m in recent_missions
                )
                if already_queued:
                    continue
            except Exception:
                pass

            # Derive project_id from health check name (e.g., "openclaw:self" → "openclaw-agent")
            project_id = name.split(":")[0].replace("_", "-")
            message = check_data.get("message", "Unknown error")

            actions.append(ProactiveAction(
                action_type="vibecoder_mission",
                priority=3,
                target=project_id,
                description=(
                    f"Auto-fix: {name} has failed {consecutive_failures}x "
                    f"consecutively — {message[:100]}"
                ),
                requires_browser=False,
                requires_approval=False,
                params={
                    "project_id": project_id,
                    "title": f"[auto] Fix {name} health check failure",
                    "mission_description": (
                        f"The health check '{name}' has failed {consecutive_failures} "
                        f"times consecutively. Last error: {message[:300]}. "
                        f"Investigate the root cause and fix it."
                    ),
                    "scope": "bugfix",
                    "priority": 2,
                },
            ))

        return actions

    def _check_stalled_missions(self, vibecoder) -> list[ProactiveAction]:
        """Find VibeCoder missions stuck in executing state for too long."""
        actions: list[ProactiveAction] = []

        try:
            executing = vibecoder.list_missions(status="executing", limit=10)
        except Exception:
            return actions

        stall_threshold = timedelta(hours=1)
        now = datetime.now()

        for mission in executing:
            started = mission.get("started_at")
            if not started:
                continue
            try:
                started_dt = datetime.fromisoformat(started)
            except (ValueError, TypeError):
                continue

            if now - started_dt > stall_threshold:
                actions.append(ProactiveAction(
                    action_type="vibecoder_mission",
                    priority=4,
                    target=mission.get("project_id", "unknown"),
                    description=(
                        f"Mission {mission['mission_id']} stuck in executing "
                        f"for {(now - started_dt).total_seconds() / 3600:.1f}h — "
                        f"force-failing for retry"
                    ),
                    requires_browser=False,
                    requires_approval=False,
                    params={
                        "action": "force_fail",
                        "mission_id": mission["mission_id"],
                    },
                ))

        return actions

    # ── Publishing opportunities ──────────────────────────────────────────────

    def _check_publishing_opportunities(self) -> list[ProactiveAction]:
        """Identify marketplace accounts that have no published content yet.

        Looks for:
        1. Platforms in publishable categories (ai_marketplace, digital_product,
           workflow_marketplace, prompt_marketplace, 3d_models) with ACTIVE accounts.
        2. Platforms where no 'publish_content' action has been logged (i.e. nothing
           has ever been published there).
        3. Availability of product files in the products directory or venture-agent
           output directory.

        All actions are created with requires_approval=True — publishing is higher-risk
        and should never run autonomously without human sign-off.

        Priority: 7 — runs only after signups, profile quality, and cleanup are handled.
        """
        from openclaw.knowledge.publishing_playbooks import get_publishing_playbook

        # Categories worth publishing on
        _PUBLISHABLE_CATEGORIES = {
            "ai_marketplace",
            "digital_product",
            "workflow_marketplace",
            "prompt_marketplace",
            "3d_models",
        }

        # ── 1. Check if there are any products available ─────────────────────
        products_dir = self._find_products_directory()
        available_products = self._scan_available_products(products_dir)
        if not available_products:
            logger.debug(
                "[ProactiveAgent] No publishable products found — skipping publish check"
            )
            return []

        # ── 2. Build set of platforms already published on ───────────────────
        already_published: set[str] = set()
        try:
            history = self.codex.get_action_history(limit=500)
            for h in history:
                if h.get("action_type") == "publish_content" and h.get("result") == "success":
                    already_published.add(h.get("target", ""))
        except Exception:
            pass

        # ── 3. Find active marketplace accounts with no published content ────
        actions: list[ProactiveAction] = []

        try:
            active_accounts = self.codex.get_accounts_by_status(AccountStatus.ACTIVE)
        except Exception:
            return []

        for account in active_accounts:
            pid = account.get("platform_id", "")
            if not pid or pid in already_published:
                continue

            platform = get_platform(pid)
            if not platform:
                continue

            category_value = platform.category.value
            if category_value not in _PUBLISHABLE_CATEGORIES:
                continue

            playbook = get_publishing_playbook(category_value)
            if not playbook:
                continue

            # Pick the best matching product for this platform
            product = self._match_product_to_platform(
                available_products, category_value
            )
            if not product:
                continue

            actions.append(ProactiveAction(
                action_type="publish_content",
                priority=7,
                target=pid,
                description=(
                    f"Publish '{product['title']}' on {account.get('platform_name', pid)} "
                    f"(category={category_value}, no content published yet)"
                ),
                requires_browser=True,
                requires_approval=True,  # Always — publishing is irreversible
                params={
                    "content": {
                        "title": product["title"],
                        "description": product["description"],
                        "price": product.get("price", 0.0),
                        "category": product.get("category", ""),
                        "tags": product.get("tags", []),
                        "file_path": product.get("file_path", ""),
                        "cover_image_path": product.get("cover_image_path", ""),
                        "preview_text": product.get("preview_text", ""),
                    },
                    "platform_category": category_value,
                    "product_source": product.get("source", ""),
                },
            ))

        # Limit to 3 suggestions per cycle — don't overwhelm the approval queue
        return actions[:3]

    def _find_products_directory(self) -> list[str]:
        """Return candidate directories that may contain publishable products.

        Checks, in order:
        1. OPENCLAW_PRODUCTS_DIR env var
        2. ../venture-agent/data/products/ (sister project)
        3. ./data/products/ (local agent data dir)
        """
        from pathlib import Path

        candidates: list[Path] = []

        env_dir = os.environ.get("OPENCLAW_PRODUCTS_DIR", "")
        if env_dir:
            candidates.append(Path(env_dir))

        # Relative to this file: openclaw-agent/openclaw/daemon/ -> up 3 levels
        agent_root = Path(__file__).resolve().parent.parent.parent
        candidates.append(agent_root / "data" / "products")

        # Look for venture-agent alongside openclaw-agent
        empire_root = agent_root.parent
        candidates.append(empire_root / "venture-agent" / "data" / "products")

        return [str(p) for p in candidates if p.is_dir()]

    def _scan_available_products(
        self, product_dirs: list[str]
    ) -> list[dict]:
        """Scan product directories for publishable product manifests.

        Looks for:
        - JSON manifest files (product.json or any *.json with title/description keys)
        - ZIP files with a companion *.json metadata sidecar
        - STL files with a companion *.json sidecar

        Returns a list of product dicts with keys:
            title, description, price, category, tags, file_path,
            cover_image_path, preview_text, source
        """
        import json
        from pathlib import Path

        products: list[dict] = []

        for dir_str in product_dirs:
            dir_path = Path(dir_str)
            try:
                for json_file in dir_path.rglob("*.json"):
                    try:
                        data = json.loads(json_file.read_text(encoding="utf-8"))
                    except Exception:
                        continue

                    # Must have at minimum a title and description
                    if not isinstance(data, dict):
                        continue
                    title = data.get("title", "").strip()
                    description = data.get("description", "").strip()
                    if not title or not description:
                        continue

                    product: dict = {
                        "title": title,
                        "description": description,
                        "price": float(data.get("price", 0.0)),
                        "category": data.get("category", ""),
                        "tags": data.get("tags", []),
                        "file_path": data.get("file_path", ""),
                        "cover_image_path": data.get("cover_image_path", ""),
                        "preview_text": data.get("preview_text", ""),
                        "content_type": data.get("content_type", "product"),
                        "source": str(json_file),
                    }

                    # Resolve relative file_path relative to the json file's dir
                    if product["file_path"] and not Path(product["file_path"]).is_absolute():
                        resolved = json_file.parent / product["file_path"]
                        if resolved.exists():
                            product["file_path"] = str(resolved)

                    products.append(product)
            except (OSError, PermissionError):
                continue

        return products

    def _match_product_to_platform(
        self, products: list[dict], category: str
    ) -> dict | None:
        """Pick the most suitable product for a given platform category.

        Preference order:
        1. Product whose content_type exactly matches the category's content_type
        2. Any product with a file_path that exists on disk
        3. Any product with a non-empty title/description

        Returns None if no product is suitable.
        """
        from openclaw.knowledge.publishing_playbooks import get_publishing_playbook
        from pathlib import Path

        playbook = get_publishing_playbook(category)
        target_type = playbook.content_type if playbook else ""

        # Score each product: higher = better match
        scored: list[tuple[int, dict]] = []
        for p in products:
            score = 0
            if target_type and p.get("content_type", "") == target_type:
                score += 10
            if p.get("file_path") and Path(p["file_path"]).exists():
                score += 5
            if p.get("cover_image_path") and Path(p.get("cover_image_path", "")).exists():
                score += 2
            scored.append((score, p))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
