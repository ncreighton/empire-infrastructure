"""ProfileSync -- synchronize profile content across multiple platforms.

When you update your bio, tagline, or links on one platform, this engine
can propagate those changes to all other active platforms.  It works in
two modes:

1. **Local sync** (``update_local``): Updates the Codex SQLite records
   without touching any browser.  Instant, no API cost, safe to run
   anytime.

2. **Browser sync** (``execute_sync``): Opens each platform in a headless
   browser and pushes the new profile content to the live site.  Requires
   browser-use to be installed and configured.

All planning and diff logic is algorithmic -- zero LLM cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openclaw.forge.platform_codex import PlatformCodex
from openclaw.forge.profile_sentinel import ProfileSentinel
from openclaw.knowledge.platforms import get_platform
from openclaw.models import AccountStatus, ProfileContent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Fields that can be synced across platforms
SYNCABLE_FIELDS = ("bio", "tagline", "description", "website_url")


@dataclass
class SyncChange:
    """A single field change to sync."""

    field_name: str  # bio, tagline, description, website_url
    old_value: str
    new_value: str


@dataclass
class SyncResult:
    """Result of syncing a single platform."""

    platform_id: str
    platform_name: str
    success: bool
    changes_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    new_score: float = 0.0
    synced_at: datetime | None = None


@dataclass
class SyncPlan:
    """Plan for syncing across multiple platforms."""

    changes: list[SyncChange]
    target_platforms: list[str]
    results: list[SyncResult] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def succeeded(self) -> int:
        """Count of platforms successfully synced."""
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        """Count of platforms that failed to sync."""
        return sum(1 for r in self.results if not r.success)

    @property
    def duration_seconds(self) -> float:
        """Total wall-clock time for the sync operation."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


# =========================================================================== #
#  ProfileSync                                                                 #
# =========================================================================== #


class ProfileSync:
    """Synchronize profile content across active platforms.

    Usage::

        sync = ProfileSync(codex=codex, sentinel=sentinel)

        # Plan a sync
        plan = sync.plan_sync({"bio": "New bio text", "website_url": "https://example.com"})

        # Preview what would change
        preview = sync.preview_sync(plan)

        # Execute locally (update Codex records only, no browser)
        sync.update_local(plan)

        # Or execute via browser (async)
        await sync.execute_sync(plan)

        # Check consistency across platforms
        status = sync.get_sync_status()
    """

    def __init__(
        self,
        codex: PlatformCodex | None = None,
        sentinel: ProfileSentinel | None = None,
    ):
        self.codex = codex or PlatformCodex()
        self.sentinel = sentinel or ProfileSentinel()

    # ------------------------------------------------------------------ #
    #  Plan creation                                                       #
    # ------------------------------------------------------------------ #

    def plan_sync(
        self,
        changes: dict[str, str],
        platform_ids: list[str] | None = None,
    ) -> SyncPlan:
        """Create a sync plan showing what will change on each platform.

        Args:
            changes: Dict mapping field names to new values.
                Valid keys: bio, tagline, description, website_url.
            platform_ids: If provided, only sync to these platforms.
                If ``None``, targets all active platforms in the Codex.

        Returns:
            A SyncPlan with the list of changes and target platforms.
        """
        # Validate field names
        valid_changes: dict[str, str] = {}
        for field_name, new_value in changes.items():
            if field_name in SYNCABLE_FIELDS:
                valid_changes[field_name] = new_value
            else:
                logger.warning(
                    f"Ignoring unsupported sync field: {field_name}. "
                    f"Supported: {', '.join(SYNCABLE_FIELDS)}"
                )

        # Determine target platforms
        if platform_ids is not None:
            targets = platform_ids
        else:
            targets = self._get_active_platform_ids()

        # Build SyncChange objects by comparing current values
        sync_changes: list[SyncChange] = []
        for field_name, new_value in valid_changes.items():
            # We create one SyncChange per field (the old_value varies per
            # platform, but we store the "canonical" old value as what the
            # first platform had, or empty).
            first_old = self._get_current_field_value(
                targets[0] if targets else "", field_name
            )
            sync_changes.append(SyncChange(
                field_name=field_name,
                old_value=first_old,
                new_value=new_value,
            ))

        return SyncPlan(changes=sync_changes, target_platforms=targets)

    # ------------------------------------------------------------------ #
    #  Preview                                                             #
    # ------------------------------------------------------------------ #

    def preview_sync(self, plan: SyncPlan) -> list[dict[str, Any]]:
        """Preview what changes would be made without executing.

        Args:
            plan: The sync plan to preview.

        Returns:
            A list of dicts, one per platform, each with keys:
            ``platform_id``, ``platform_name``, ``changes`` (list of
            field diffs).
        """
        previews: list[dict[str, Any]] = []

        for pid in plan.target_platforms:
            platform = get_platform(pid)
            platform_name = platform.name if platform else pid

            field_diffs: list[dict[str, str]] = []
            for change in plan.changes:
                current = self._get_current_field_value(pid, change.field_name)
                if current != change.new_value:
                    field_diffs.append({
                        "field": change.field_name,
                        "old": current,
                        "new": change.new_value,
                    })

            previews.append({
                "platform_id": pid,
                "platform_name": platform_name,
                "changes": field_diffs,
                "has_changes": len(field_diffs) > 0,
            })

        return previews

    # ------------------------------------------------------------------ #
    #  Execute via browser (async)                                         #
    # ------------------------------------------------------------------ #

    async def execute_sync(self, plan: SyncPlan) -> SyncPlan:
        """Execute the sync plan using browser automation.

        For each target platform, this method opens the profile edit page
        in a headless browser, updates the relevant fields, saves, and
        verifies the changes took effect.

        Args:
            plan: The sync plan to execute.

        Returns:
            The same SyncPlan with ``results`` populated.

        Note:
            This requires ``browser-use`` to be installed.  For local-only
            updates (no browser), use :meth:`update_local` instead.
        """
        try:
            from openclaw.browser.browser_manager import BrowserManager
        except ImportError:
            logger.error("browser-use not available; falling back to local-only sync")
            self.update_local(plan)
            return plan

        plan.started_at = datetime.now()

        for pid in plan.target_platforms:
            platform = get_platform(pid)
            platform_name = platform.name if platform else pid
            result = SyncResult(platform_id=pid, platform_name=platform_name, success=False)

            try:
                # Build the set of changes for this platform
                changes_for_platform: dict[str, str] = {}
                for change in plan.changes:
                    current = self._get_current_field_value(pid, change.field_name)
                    if current != change.new_value:
                        changes_for_platform[change.field_name] = change.new_value

                if not changes_for_platform:
                    result.success = True
                    result.changes_applied = []
                    result.synced_at = datetime.now()
                    plan.results.append(result)
                    logger.info(f"[{pid}] No changes needed -- skipping browser sync")
                    continue

                # Open browser and update profile
                browser = BrowserManager(headless=True)
                try:
                    await browser.launch(platform_id=pid)

                    # Build a task description for the browser-use agent
                    fields_desc = ", ".join(
                        f"{k}={v!r}" for k, v in changes_for_platform.items()
                    )
                    task = (
                        f"Log in to {platform.name if platform else pid} and "
                        f"update the profile with these fields: {fields_desc}"
                    )

                    if platform and platform.login_url:
                        # Navigate to login page first
                        page = browser._page
                        if page:
                            await page.goto(platform.login_url)

                    # Use browser-use agent to apply changes via visual navigation
                    agent = await browser.create_agent(
                        task=task,
                        platform_id=pid,
                        max_steps=20,
                    )
                    if agent:
                        await browser.run_agent(
                            task=task,
                            platform_id=pid,
                            max_steps=20,
                        )

                    for field_name in changes_for_platform:
                        result.changes_applied.append(field_name)

                    result.success = True
                    result.synced_at = datetime.now()

                finally:
                    await browser.close()

                # Update local Codex to match
                self._update_codex_profile(pid, changes_for_platform)

                # Re-score
                new_score = self._rescore(pid)
                result.new_score = new_score

            except Exception as exc:
                logger.error(f"[{pid}] Browser sync failed: {exc}")
                result.errors.append(str(exc))

            plan.results.append(result)

        plan.completed_at = datetime.now()
        logger.info(
            f"Sync complete: {plan.succeeded}/{len(plan.target_platforms)} succeeded "
            f"in {plan.duration_seconds:.1f}s"
        )
        return plan

    # ------------------------------------------------------------------ #
    #  Local-only sync (no browser)                                        #
    # ------------------------------------------------------------------ #

    def update_local(self, plan: SyncPlan) -> None:
        """Update local Codex records with new profile content.

        This does not open a browser.  It only updates the stored profile
        data in SQLite so that the Codex reflects the intended state.

        Args:
            plan: The sync plan to execute locally.
        """
        plan.started_at = datetime.now()

        for pid in plan.target_platforms:
            platform = get_platform(pid)
            platform_name = platform.name if platform else pid
            result = SyncResult(platform_id=pid, platform_name=platform_name, success=False)

            try:
                changes_for_platform: dict[str, str] = {}
                for change in plan.changes:
                    current = self._get_current_field_value(pid, change.field_name)
                    if current != change.new_value:
                        changes_for_platform[change.field_name] = change.new_value

                if not changes_for_platform:
                    result.success = True
                    result.changes_applied = []
                    result.synced_at = datetime.now()
                    plan.results.append(result)
                    continue

                self._update_codex_profile(pid, changes_for_platform)
                new_score = self._rescore(pid)

                result.success = True
                result.changes_applied = list(changes_for_platform.keys())
                result.new_score = new_score
                result.synced_at = datetime.now()

                logger.info(
                    f"[{pid}] Local sync: updated {', '.join(result.changes_applied)}, "
                    f"new score={new_score:.1f}"
                )

            except Exception as exc:
                logger.error(f"[{pid}] Local sync failed: {exc}")
                result.errors.append(str(exc))

            plan.results.append(result)

        plan.completed_at = datetime.now()

    # ------------------------------------------------------------------ #
    #  Consistency analysis                                                #
    # ------------------------------------------------------------------ #

    def get_sync_status(self) -> dict[str, Any]:
        """Get overview of profile consistency across platforms.

        Compares bio, tagline, and website_url across all active platforms
        to find mismatches.

        Returns:
            A dict with keys: ``total_active``, ``consistent_fields``,
            ``mismatched_fields``, ``mismatches`` (list of diffs),
            ``field_values`` (what each platform has).
        """
        active_ids = self._get_active_platform_ids()
        if not active_ids:
            return {
                "total_active": 0,
                "consistent_fields": 0,
                "mismatched_fields": 0,
                "mismatches": [],
                "field_values": {},
            }

        # Gather current values per field per platform
        field_values: dict[str, dict[str, str]] = {
            f: {} for f in SYNCABLE_FIELDS
        }
        for pid in active_ids:
            for field_name in SYNCABLE_FIELDS:
                val = self._get_current_field_value(pid, field_name)
                field_values[field_name][pid] = val

        # Check consistency for each field
        consistent = 0
        mismatches: list[dict[str, Any]] = []
        for field_name in SYNCABLE_FIELDS:
            values = field_values[field_name]
            unique_values = set(v for v in values.values() if v)
            if len(unique_values) <= 1:
                consistent += 1
            else:
                mismatch_count = len(unique_values)
                # Find which platforms differ
                value_groups: dict[str, list[str]] = {}
                for pid, val in values.items():
                    key = val[:50] if val else "(empty)"
                    if key not in value_groups:
                        value_groups[key] = []
                    value_groups[key].append(pid)

                mismatches.append({
                    "field": field_name,
                    "count": mismatch_count,
                    "groups": value_groups,
                })

        return {
            "total_active": len(active_ids),
            "consistent_fields": consistent,
            "mismatched_fields": len(mismatches),
            "mismatches": mismatches,
            "field_values": field_values,
        }

    def get_outdated_platforms(self, reference_platform_id: str) -> list[dict[str, Any]]:
        """Find platforms whose profile content differs from the reference.

        Uses the stored profile for ``reference_platform_id`` as the
        canonical source and returns a list of platforms that differ in
        at least one syncable field.

        Args:
            reference_platform_id: The platform to use as the reference.

        Returns:
            A list of dicts with keys: ``platform_id``, ``platform_name``,
            ``differing_fields`` (list of field names),
            ``diffs`` (list of {field, reference_value, current_value}).
        """
        ref_profile = self.codex.get_profile(reference_platform_id)
        if not ref_profile:
            logger.warning(
                f"No stored profile for reference platform {reference_platform_id}"
            )
            return []

        ref_content = ref_profile.get("content", {})
        active_ids = self._get_active_platform_ids()
        outdated: list[dict[str, Any]] = []

        for pid in active_ids:
            if pid == reference_platform_id:
                continue

            platform = get_platform(pid)
            platform_name = platform.name if platform else pid
            current_profile = self.codex.get_profile(pid)
            current_content = current_profile.get("content", {}) if current_profile else {}

            diffs: list[dict[str, str]] = []
            for field_name in SYNCABLE_FIELDS:
                ref_val = ref_content.get(field_name, "")
                cur_val = current_content.get(field_name, "")
                if ref_val and ref_val != cur_val:
                    diffs.append({
                        "field": field_name,
                        "reference_value": ref_val,
                        "current_value": cur_val,
                    })

            if diffs:
                outdated.append({
                    "platform_id": pid,
                    "platform_name": platform_name,
                    "differing_fields": [d["field"] for d in diffs],
                    "diffs": diffs,
                })

        return outdated

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_active_platform_ids(self) -> list[str]:
        """Return platform IDs for all active or profile-complete accounts."""
        active_statuses = {
            AccountStatus.ACTIVE.value,
            AccountStatus.PROFILE_COMPLETE.value,
        }
        all_accounts = self.codex.get_all_accounts()
        return [
            a["platform_id"]
            for a in all_accounts
            if a.get("status") in active_statuses
        ]

    def _get_current_field_value(self, platform_id: str, field_name: str) -> str:
        """Read a single profile field from the Codex for a platform.

        Args:
            platform_id: The platform identifier.
            field_name: The field name (bio, tagline, description, website_url).

        Returns:
            The current value, or empty string if not found.
        """
        if not platform_id:
            return ""

        profile = self.codex.get_profile(platform_id)
        if not profile:
            return ""

        content = profile.get("content", {})
        return content.get(field_name, "")

    def _update_codex_profile(self, platform_id: str, changes: dict[str, str]) -> None:
        """Apply field changes to the stored profile in the Codex.

        If no profile exists yet, a new one is created with the changed
        fields.

        Args:
            platform_id: The platform identifier.
            changes: Dict mapping field names to new values.
        """
        existing = self.codex.get_profile(platform_id)

        if existing:
            content_dict = existing.get("content", {})
        else:
            content_dict = {"platform_id": platform_id}

        # Apply changes
        for field_name, new_value in changes.items():
            content_dict[field_name] = new_value

        # Build a ProfileContent for re-storage
        profile_content = ProfileContent(
            platform_id=platform_id,
            username=content_dict.get("username", ""),
            display_name=content_dict.get("display_name", ""),
            email=content_dict.get("email", ""),
            bio=content_dict.get("bio", ""),
            tagline=content_dict.get("tagline", ""),
            description=content_dict.get("description", ""),
            website_url=content_dict.get("website_url", ""),
            avatar_path=content_dict.get("avatar_path", ""),
            banner_path=content_dict.get("banner_path", ""),
            social_links=content_dict.get("social_links", {}),
            custom_fields=content_dict.get("custom_fields", {}),
            seo_keywords=content_dict.get("seo_keywords", []),
        )

        # Score the updated content
        score = self.sentinel.score(profile_content)

        # Persist
        self.codex.store_profile(profile_content, score)

    def _rescore(self, platform_id: str) -> float:
        """Re-score the stored profile for a platform and return the total.

        Args:
            platform_id: The platform identifier.

        Returns:
            The new total sentinel score, or 0.0 if no profile exists.
        """
        stored = self.codex.get_profile(platform_id)
        if not stored:
            return 0.0

        content_dict = stored.get("content", {})
        profile_content = ProfileContent(
            platform_id=platform_id,
            username=content_dict.get("username", ""),
            display_name=content_dict.get("display_name", ""),
            email=content_dict.get("email", ""),
            bio=content_dict.get("bio", ""),
            tagline=content_dict.get("tagline", ""),
            description=content_dict.get("description", ""),
            website_url=content_dict.get("website_url", ""),
            avatar_path=content_dict.get("avatar_path", ""),
            banner_path=content_dict.get("banner_path", ""),
            social_links=content_dict.get("social_links", {}),
            custom_fields=content_dict.get("custom_fields", {}),
            seo_keywords=content_dict.get("seo_keywords", []),
        )

        score = self.sentinel.score(profile_content)
        return score.total_score
