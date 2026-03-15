"""ProfileApplier — applies stored ProfileContent to live platform profiles.

Uses BrowserManager + browser-use Agent to navigate to each platform's
profile/settings page and fill in bio, tagline, description, website,
social links, avatar, and banner from the PlatformCodex.

All navigation is done via natural language instructions to the vision agent
(no CSS selectors) because profile settings pages vary enormously across
platforms. Claude Sonnet is used throughout since profile pages frequently
have complex tabbed UIs, nested menus, and save/confirm flows.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from openclaw.browser.stealth import randomize_delay, add_human_delays
from openclaw.knowledge.platforms import get_platform
from openclaw.models import AccountStatus, ProfileContent

if TYPE_CHECKING:
    from openclaw.browser.browser_manager import BrowserManager
    from openclaw.forge.platform_codex import PlatformCodex

logger = logging.getLogger(__name__)

# Sonnet for all profile-apply tasks — profile settings pages require careful
# visual reasoning (tabbed UIs, nested menus, inline saves, avatar crop dialogs)
_PROFILE_MODEL = "claude-sonnet-4-20250514"

# Fields that count as "applied" for success/failure tracking
_FIELD_NAMES = [
    "bio",
    "tagline",
    "description",
    "website",
    "social_links",
    "avatar",
    "banner",
]


@dataclass
class ProfileApplyResult:
    """Result of applying a profile to a live platform."""

    platform_id: str
    success: bool = False
    fields_applied: list[str] = field(default_factory=list)
    fields_failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class ProfileApplier:
    """Apply stored ProfileContent from the Codex to live platform profiles.

    Usage::

        applier = ProfileApplier()
        result = await applier.apply_profile("gumroad")
        print(result.fields_applied)  # ["bio", "tagline", "website"]
        print(result.success)         # True

    The applier:
    1. Retrieves the stored ProfileContent from the Codex
    2. Launches a browser with the existing session (so it's already logged in)
    3. Navigates to the platform's profile/settings page
    4. Uses the vision agent to fill each available field
    5. Saves the profile
    6. Logs the result to the action_log
    """

    def __init__(
        self,
        browser_manager: BrowserManager | None = None,
        codex: PlatformCodex | None = None,
    ):
        self._browser = browser_manager
        self.codex = codex
        self.delays = add_human_delays()

    def _get_browser(self) -> BrowserManager:
        """Return or lazy-create the BrowserManager."""
        if self._browser is None:
            from openclaw.browser.browser_manager import BrowserManager
            self._browser = BrowserManager()
        return self._browser

    def _get_codex(self) -> PlatformCodex:
        """Return or lazy-create the PlatformCodex."""
        if self.codex is None:
            from openclaw.forge.platform_codex import PlatformCodex
            self.codex = PlatformCodex()
        return self.codex

    # ================================================================== #
    #  Public API                                                          #
    # ================================================================== #

    async def apply_profile(self, platform_id: str) -> ProfileApplyResult:
        """Apply stored profile content to the live platform profile.

        Retrieves credentials and content from the Codex, launches a
        browser with the existing session, navigates to the profile
        settings page, fills all available fields, and saves.

        Args:
            platform_id: The platform identifier (e.g. "gumroad").

        Returns:
            A ProfileApplyResult with per-field success/failure details.
        """
        result = ProfileApplyResult(platform_id=platform_id)
        start = time.monotonic()
        browser = self._get_browser()
        codex = self._get_codex()

        try:
            # 1. Load stored profile from codex
            profile_row = codex.get_profile(platform_id)
            if not profile_row:
                msg = f"No stored profile found for {platform_id}"
                logger.warning(f"[ProfileApplier] {msg}")
                result.errors.append(msg)
                return result

            content_dict: dict[str, Any] = profile_row["content"]
            content = _dict_to_profile_content(platform_id, content_dict)

            # 2. Get platform config for navigation URLs
            platform = get_platform(platform_id)
            if not platform:
                msg = f"Unknown platform: {platform_id}"
                logger.warning(f"[ProfileApplier] {msg}")
                result.errors.append(msg)
                return result

            # 3. Launch browser and restore session
            await browser.launch(platform_id)

            # 4. Navigate to profile settings
            nav_ok = await self._navigate_to_profile_settings(
                platform_id, platform.login_url, browser
            )
            if not nav_ok:
                msg = f"Could not navigate to profile settings for {platform_id}"
                logger.warning(f"[ProfileApplier] {msg}")
                result.errors.append(msg)
                # Still attempt fields — the agent may recover
                # (e.g., already on settings page from restored session)

            # 5. Apply each profile field
            await asyncio.sleep(randomize_delay(self.delays["page_load_wait"]))

            if content.bio and platform.bio_max_length > 0:
                ok = await self._apply_bio(
                    platform_id, content.bio, platform, browser
                )
                _record_field(result, "bio", ok)

            if content.tagline and platform.tagline_max_length > 0:
                ok = await self._apply_tagline(
                    platform_id, content.tagline, platform, browser
                )
                _record_field(result, "tagline", ok)

            if content.description and platform.description_max_length > 0:
                ok = await self._apply_description(
                    platform_id, content.description, platform, browser
                )
                _record_field(result, "description", ok)

            if content.website_url and platform.allows_links:
                ok = await self._apply_website(
                    platform_id, content.website_url, platform, browser
                )
                _record_field(result, "website", ok)

            if content.social_links and platform.allows_links:
                ok = await self._apply_social_links(
                    platform_id, content.social_links, platform, browser
                )
                _record_field(result, "social_links", ok)

            if content.avatar_path and platform.allows_avatar:
                ok = await self._apply_avatar(
                    platform_id, content.avatar_path, platform, browser
                )
                _record_field(result, "avatar", ok)

            if content.banner_path and platform.allows_banner:
                ok = await self._apply_banner(
                    platform_id, content.banner_path, platform, browser
                )
                _record_field(result, "banner", ok)

            # 6. Save the profile
            if result.fields_applied:
                save_ok = await self._save_profile(platform_id, browser)
                if not save_ok:
                    result.errors.append("Profile save may not have completed")
                    logger.warning(
                        f"[ProfileApplier] Save step returned failure for {platform_id}"
                    )

            # 7. Take a screenshot as evidence
            screenshot = await browser.take_screenshot(
                name=f"{platform_id}_profile_applied"
            )
            if screenshot:
                result.screenshots.append(screenshot)

            # 8. Determine overall success
            result.success = bool(result.fields_applied) and not any(
                f in result.fields_failed
                for f in ("bio", "tagline")  # Core fields — if both fail, call it failed
            )

            # 9. Save session so next run benefits from staying logged in
            await browser.save_session(platform_id)

        except Exception as e:
            msg = f"ProfileApplier exception for {platform_id}: {e}"
            logger.error(f"[ProfileApplier] {msg}", exc_info=True)
            result.errors.append(str(e))
        finally:
            result.duration_seconds = time.monotonic() - start
            await browser.close()

        # 10. Log to action_log regardless of outcome
        status = "success" if result.success else "failed"
        fields_summary = (
            f"applied={result.fields_applied}, failed={result.fields_failed}"
        )
        codex.log_action(
            "apply_profile",
            platform_id,
            f"Applied {len(result.fields_applied)} fields to {platform_id}: {fields_summary}",
            status,
        )

        logger.info(
            f"[ProfileApplier] {platform_id}: success={result.success}, "
            f"fields_applied={result.fields_applied}, "
            f"fields_failed={result.fields_failed}, "
            f"duration={result.duration_seconds:.1f}s"
        )
        return result

    async def apply_batch(
        self, platform_ids: list[str]
    ) -> dict[str, ProfileApplyResult]:
        """Apply profiles to multiple platforms sequentially.

        Runs one platform at a time to avoid browser conflicts and
        respect platform rate limits.

        Args:
            platform_ids: List of platform identifiers to process.

        Returns:
            Dict mapping platform_id to its ProfileApplyResult.
        """
        results: dict[str, ProfileApplyResult] = {}
        for platform_id in platform_ids:
            logger.info(f"[ProfileApplier] Batch: processing {platform_id}")
            results[platform_id] = await self.apply_profile(platform_id)
            # Human-like pause between platforms
            await asyncio.sleep(randomize_delay((3.0, 8.0)))
        return results

    # ================================================================== #
    #  Navigation                                                          #
    # ================================================================== #

    async def _navigate_to_profile_settings(
        self,
        platform_id: str,
        login_url: str,
        browser: BrowserManager,
    ) -> bool:
        """Navigate to the platform's profile/account settings page.

        Tries multiple strategies:
        1. Use the platform's login_url (which often leads to the dashboard)
        2. Ask the vision agent to find and navigate to profile settings

        Args:
            platform_id: Platform identifier for logging.
            login_url: Platform login/dashboard URL from PlatformConfig.
            browser: Active BrowserManager instance.

        Returns:
            True if navigation succeeded (page loaded), False otherwise.
        """
        # First: navigate to the platform dashboard / home (restores session context)
        if login_url:
            nav_result = await browser.run_agent(
                task=(
                    f"Navigate to {login_url} and wait for the page to fully load. "
                    f"If a login form is shown, the session may have expired — "
                    f"report 'SESSION_EXPIRED'. Otherwise report 'LOADED'."
                ),
                platform_id=platform_id,
                max_steps=5,
                model=_PROFILE_MODEL,
            )
            final = (nav_result.get("final_text") or "").lower()
            if "session_expired" in final:
                logger.warning(
                    f"[ProfileApplier] Session expired for {platform_id}"
                )
                # Try to proceed anyway — maybe the settings page is accessible
        else:
            logger.debug(
                f"[ProfileApplier] No login_url configured for {platform_id}, "
                f"relying on restored session cookies"
            )

        # Second: find and navigate to the profile/edit page
        settings_result = await browser.run_agent(
            task=(
                "Navigate to the profile editing or account settings page. "
                "Look for: a user avatar or profile picture in the header/nav, "
                "a 'Profile', 'Edit Profile', 'Account Settings', or 'Settings' "
                "link in the navigation or user menu. "
                "Click through menus as needed until you reach a page where you "
                "can edit your profile bio, description, or personal information. "
                "Once on the settings page, report 'ON_SETTINGS_PAGE'."
            ),
            platform_id=platform_id,
            max_steps=10,
            model=_PROFILE_MODEL,
        )
        final = (settings_result.get("final_text") or "").lower()
        return "on_settings_page" in final or settings_result.get("success", False)

    # ================================================================== #
    #  Field applicators                                                   #
    # ================================================================== #

    async def _apply_bio(
        self,
        platform_id: str,
        bio: str,
        platform: Any,
        browser: BrowserManager,
    ) -> bool:
        """Fill in the bio/about field.

        Args:
            platform_id: Platform identifier.
            bio: Bio text to apply (already within platform.bio_max_length).
            platform: PlatformConfig for the platform.
            browser: Active BrowserManager.

        Returns:
            True if the field was filled successfully.
        """
        # Respect the character limit
        max_len = platform.bio_max_length or len(bio)
        bio_text = bio[:max_len]

        result = await browser.run_agent(
            task=(
                f"Find the bio or 'About' text field on this profile settings page "
                f"and replace its contents with the following text exactly:\n\n"
                f"{bio_text}\n\n"
                f"The field might be labeled 'Bio', 'About', 'Short bio', 'About me', "
                f"'Introduce yourself', or similar. Clear any existing text first, "
                f"then type the new text. Do NOT submit/save the form yet."
            ),
            platform_id=platform_id,
            max_steps=8,
            model=_PROFILE_MODEL,
        )
        return result.get("success", False)

    async def _apply_tagline(
        self,
        platform_id: str,
        tagline: str,
        platform: Any,
        browser: BrowserManager,
    ) -> bool:
        """Fill in the tagline/headline field.

        Args:
            platform_id: Platform identifier.
            tagline: Tagline text to apply.
            platform: PlatformConfig for the platform.
            browser: Active BrowserManager.

        Returns:
            True if the field was filled successfully.
        """
        max_len = platform.tagline_max_length or len(tagline)
        tagline_text = tagline[:max_len]

        result = await browser.run_agent(
            task=(
                f"Find the tagline, headline, or subtitle field on this profile "
                f"settings page and replace its contents with:\n\n"
                f"{tagline_text}\n\n"
                f"The field might be labeled 'Tagline', 'Headline', 'Title', "
                f"'Professional headline', 'Subtitle', or 'One-liner'. "
                f"Clear any existing text first, then type the new text. "
                f"Do NOT submit/save the form yet."
            ),
            platform_id=platform_id,
            max_steps=8,
            model=_PROFILE_MODEL,
        )
        return result.get("success", False)

    async def _apply_description(
        self,
        platform_id: str,
        description: str,
        platform: Any,
        browser: BrowserManager,
    ) -> bool:
        """Fill in the longer description field.

        Args:
            platform_id: Platform identifier.
            description: Description text to apply.
            platform: PlatformConfig for the platform.
            browser: Active BrowserManager.

        Returns:
            True if the field was filled successfully.
        """
        max_len = platform.description_max_length or len(description)
        desc_text = description[:max_len]

        result = await browser.run_agent(
            task=(
                f"Find the description or 'About me' long-form text area on this "
                f"profile settings page and replace its contents with:\n\n"
                f"{desc_text}\n\n"
                f"The field might be labeled 'Description', 'Full bio', 'About me', "
                f"'Profile description', 'Introduction', or be a large text area "
                f"below the bio/tagline. Clear any existing text first. "
                f"Do NOT submit/save the form yet."
            ),
            platform_id=platform_id,
            max_steps=10,
            model=_PROFILE_MODEL,
        )
        return result.get("success", False)

    async def _apply_website(
        self,
        platform_id: str,
        website_url: str,
        platform: Any,
        browser: BrowserManager,
    ) -> bool:
        """Fill in the website URL field.

        Args:
            platform_id: Platform identifier.
            website_url: Website URL to apply.
            platform: PlatformConfig for the platform.
            browser: Active BrowserManager.

        Returns:
            True if the field was filled successfully.
        """
        result = await browser.run_agent(
            task=(
                f"Find the website or personal URL field on this profile settings "
                f"page and replace its contents with: {website_url}\n\n"
                f"The field might be labeled 'Website', 'Personal website', "
                f"'Portfolio', 'URL', 'Link', or 'Homepage'. "
                f"Clear any existing text first, then type the URL. "
                f"Do NOT submit/save the form yet."
            ),
            platform_id=platform_id,
            max_steps=7,
            model=_PROFILE_MODEL,
        )
        return result.get("success", False)

    async def _apply_social_links(
        self,
        platform_id: str,
        social_links: dict[str, str],
        platform: Any,
        browser: BrowserManager,
    ) -> bool:
        """Fill in social media link fields.

        Attempts to fill each network's link field (Twitter, GitHub, LinkedIn,
        etc.) using the natural language vision agent.

        Args:
            platform_id: Platform identifier.
            social_links: Dict mapping network name to URL (e.g. {"twitter": "..."}).
            platform: PlatformConfig for the platform.
            browser: Active BrowserManager.

        Returns:
            True if at least one social link was applied.
        """
        if not social_links:
            return False

        applied_any = False
        for network, url in social_links.items():
            if not url:
                continue
            result = await browser.run_agent(
                task=(
                    f"Find the {network} link or social media field for {network} "
                    f"on this profile settings page and enter: {url}\n\n"
                    f"Look for a field labeled '{network}', '{network.title()}', "
                    f"'{network.upper()}', or a social media links section. "
                    f"Clear any existing text first. "
                    f"Do NOT submit/save the form yet."
                ),
                platform_id=platform_id,
                max_steps=7,
                model=_PROFILE_MODEL,
            )
            if result.get("success", False):
                applied_any = True
            await asyncio.sleep(randomize_delay(self.delays["form_field_pause"]))

        return applied_any

    async def _apply_avatar(
        self,
        platform_id: str,
        avatar_path: str,
        platform: Any,
        browser: BrowserManager,
    ) -> bool:
        """Upload the avatar image.

        Args:
            platform_id: Platform identifier.
            avatar_path: Absolute file path to the avatar image.
            platform: PlatformConfig for the platform.
            browser: Active BrowserManager.

        Returns:
            True if avatar was uploaded successfully.
        """
        import os
        if not os.path.isfile(avatar_path):
            logger.warning(
                f"[ProfileApplier] Avatar file not found: {avatar_path}"
            )
            return False

        result = await browser.run_agent(
            task=(
                f"Find the profile picture, avatar, or photo upload button on this "
                f"profile settings page and upload the image at: {avatar_path}\n\n"
                f"Look for: a circular profile picture, a camera icon, an 'Upload photo' "
                f"button, or a 'Change avatar' link. Click it to open the file dialog, "
                f"then upload the file. If a crop/edit dialog appears, accept the "
                f"default crop and confirm. Do NOT submit/save the full form yet."
            ),
            platform_id=platform_id,
            max_steps=12,
            model=_PROFILE_MODEL,
        )
        return result.get("success", False)

    async def _apply_banner(
        self,
        platform_id: str,
        banner_path: str,
        platform: Any,
        browser: BrowserManager,
    ) -> bool:
        """Upload the banner/cover image.

        Args:
            platform_id: Platform identifier.
            banner_path: Absolute file path to the banner image.
            platform: PlatformConfig for the platform.
            browser: Active BrowserManager.

        Returns:
            True if banner was uploaded successfully.
        """
        import os
        if not os.path.isfile(banner_path):
            logger.warning(
                f"[ProfileApplier] Banner file not found: {banner_path}"
            )
            return False

        result = await browser.run_agent(
            task=(
                f"Find the banner, cover photo, or header image upload button on this "
                f"profile settings page and upload the image at: {banner_path}\n\n"
                f"Look for: a wide banner area at the top of the profile preview, "
                f"a 'Upload cover', 'Change banner', or 'Cover photo' button. "
                f"Click it to open the file dialog, then upload the file. "
                f"If a crop/edit dialog appears, accept the default and confirm. "
                f"Do NOT submit/save the full form yet."
            ),
            platform_id=platform_id,
            max_steps=12,
            model=_PROFILE_MODEL,
        )
        return result.get("success", False)

    async def _save_profile(
        self, platform_id: str, browser: BrowserManager
    ) -> bool:
        """Find and click the save/update button for the profile settings.

        Args:
            platform_id: Platform identifier.
            browser: Active BrowserManager.

        Returns:
            True if save succeeded (or appeared to succeed).
        """
        # Brief pause before saving (human-like)
        await asyncio.sleep(randomize_delay(self.delays["submit_pause"]))

        result = await browser.run_agent(
            task=(
                "Find and click the Save, Update, or Submit button on this profile "
                "settings page to save all the changes made. "
                "Look for: 'Save changes', 'Update profile', 'Save profile', "
                "'Save', 'Submit', or similar buttons. "
                "After clicking, wait 3 seconds and confirm a success message, "
                "green notification, or that the page stayed on the settings page "
                "(not redirected to login). "
                "Report 'SAVED' if the save appeared to succeed, 'SAVE_FAILED' otherwise."
            ),
            platform_id=platform_id,
            max_steps=8,
            model=_PROFILE_MODEL,
        )
        final = (result.get("final_text") or "").lower()
        return "saved" in final or result.get("success", False)


# =========================================================================== #
#  Helpers                                                                      #
# =========================================================================== #


def _dict_to_profile_content(
    platform_id: str, content_dict: dict[str, Any]
) -> ProfileContent:
    """Convert a stored content dict back to a ProfileContent dataclass.

    The codex stores profile content as a JSON dict. This reconstructs the
    dataclass, tolerating missing fields with safe defaults.

    Args:
        platform_id: Platform identifier (used to set the platform_id field).
        content_dict: Dict from ``codex.get_profile()["content"]``.

    Returns:
        A ProfileContent dataclass with all available fields populated.
    """
    return ProfileContent(
        platform_id=content_dict.get("platform_id", platform_id),
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


def _record_field(result: ProfileApplyResult, field_name: str, ok: bool) -> None:
    """Record a field application outcome in the result object.

    Args:
        result: The ProfileApplyResult to mutate.
        field_name: Name of the field (e.g. "bio", "tagline").
        ok: Whether the field was applied successfully.
    """
    if ok:
        result.fields_applied.append(field_name)
        logger.debug(f"[ProfileApplier] Field '{field_name}' applied OK")
    else:
        result.fields_failed.append(field_name)
        logger.debug(f"[ProfileApplier] Field '{field_name}' failed")
