"""ContentPublisher — uploads digital products and posts content on marketplace platforms.

Uses browser-use Agent (same pattern as ExecutorAgent) to navigate each
platform's upload/create flow, fill all required fields, upload files,
set pricing, and click publish.

Zero AI cost for decision logic — only the browser agent calls LLM for
visual navigation.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from openclaw.forge.platform_codex import PlatformCodex

logger = logging.getLogger(__name__)

# Sonnet for the publish agent — form filling + file upload + submission is
# moderately complex visual navigation. Matches _DEFAULT_MODEL in ExecutorAgent.
_PUBLISH_MODEL = "claude-sonnet-4-20250514"


# ─── Data models ─────────────────────────────────────────────────────────────


@dataclass
class PublishableContent:
    """Content package ready for publishing on a marketplace platform."""

    title: str
    description: str
    price: float = 0.0
    """0.0 = free listing."""
    currency: str = "USD"
    category: str = ""
    tags: list[str] = field(default_factory=list)
    file_path: str = ""
    """Absolute path to the main product file (ZIP, JSON, STL, PDF, etc.)."""
    cover_image_path: str = ""
    """Absolute path to the cover/thumbnail image."""
    preview_text: str = ""
    """Short preview excerpt shown before purchase."""
    extra_fields: dict[str, str] = field(default_factory=dict)
    """Platform-specific fields not covered by the standard schema."""


@dataclass
class PublishResult:
    """Result of a content publishing attempt."""

    platform_id: str
    success: bool = False
    published_url: str = ""
    """URL of the live listing after publish."""
    listing_id: str = ""
    """Platform-specific listing or product ID."""
    errors: list[str] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    needs_review: bool = False
    """True when the platform queued the listing for manual review."""


# ─── ContentPublisher ─────────────────────────────────────────────────────────


class ContentPublisher:
    """Publishes digital products on marketplace platforms via browser automation.

    Uses the same BrowserManager + run_agent() pattern as ExecutorAgent.
    One browser instance per publish() call — launched, used, then closed.

    Usage::

        publisher = ContentPublisher(codex=engine.codex)
        content = PublishableContent(
            title="AI Automation Bundle",
            description="...",
            price=9.99,
            file_path="/tmp/bundle.zip",
            cover_image_path="/tmp/cover.png",
        )
        result = await publisher.publish("gumroad", content)
    """

    def __init__(
        self,
        browser_manager: Any = None,
        codex: Any = None,
    ):
        self._browser_manager = browser_manager
        self.codex = codex

    # ── Public API ───────────────────────────────────────────────────────────

    async def publish(
        self, platform_id: str, content: PublishableContent
    ) -> PublishResult:
        """Publish content on a single platform.

        Steps:
        1. Look up the platform config + publishing playbook.
        2. Launch browser with session restore (skip login if cookies valid).
        3. Navigate to the upload/create page.
        4. Fill all content fields via browser agent.
        5. Upload product file + cover image.
        6. Set pricing.
        7. Click publish/submit.
        8. Extract the published URL from the result page.
        9. Log to codex.

        Args:
            platform_id: Registered platform ID (e.g. 'gumroad').
            content: Fully-populated content package.

        Returns:
            PublishResult with success flag and published URL.
        """
        from openclaw.knowledge.publishing_playbooks import get_playbook_for_platform

        start_time = time.monotonic()
        result = PublishResult(platform_id=platform_id)

        # ── 1. Resolve playbook ──────────────────────────────────────────────
        playbook = get_playbook_for_platform(platform_id)
        if not playbook:
            result.errors.append(
                f"No publishing playbook found for platform '{platform_id}'"
            )
            logger.warning(
                f"[ContentPublisher] No playbook for {platform_id} — skipping"
            )
            return result

        result.needs_review = playbook.requires_review

        # ── 2. Acquire browser ───────────────────────────────────────────────
        browser = self._get_browser(platform_id)

        try:
            await browser.launch(platform_id)
            logger.info(
                f"[ContentPublisher] Browser launched for {platform_id} "
                f"(content_type={playbook.content_type})"
            )

            # ── 3. Navigate to upload page ───────────────────────────────────
            nav_ok = await self._navigate_to_upload_page(
                platform_id, playbook, browser
            )
            if not nav_ok:
                result.errors.append(
                    "Could not navigate to the upload/create page"
                )
                return result

            # ── 4. Fill fields ───────────────────────────────────────────────
            filled = await self._fill_content_fields(
                content, playbook, platform_id, browser
            )
            logger.info(
                f"[ContentPublisher] Fields filled for {platform_id}: {filled}"
            )

            # ── 5. Upload files ──────────────────────────────────────────────
            if content.file_path or content.cover_image_path:
                upload_ok = await self._upload_files(
                    content, playbook, platform_id, browser
                )
                if not upload_ok:
                    result.errors.append("File upload failed or was skipped")
                    # Non-fatal: continue to try submitting

            # ── 6. Set pricing ───────────────────────────────────────────────
            if playbook.has_pricing:
                await self._set_pricing(content, playbook, platform_id, browser)

            # ── 7. Submit ────────────────────────────────────────────────────
            success, published_url = await self._submit_publish(
                platform_id, playbook, browser
            )
            result.success = success
            result.published_url = published_url

            if success:
                logger.info(
                    f"[ContentPublisher] Published on {platform_id}: "
                    f"{published_url or '(url pending review)'}"
                )
                # ── 8. Save session (stay logged in) ─────────────────────────
                await browser.save_session(platform_id)
            else:
                result.errors.append("Publish/submit step did not confirm success")
                logger.warning(
                    f"[ContentPublisher] Publish may have failed on {platform_id}"
                )

        except Exception as exc:
            err_msg = str(exc)[:300]
            result.errors.append(err_msg)
            logger.error(
                f"[ContentPublisher] Exception publishing on {platform_id}: {exc}",
                exc_info=True,
            )
        finally:
            result.duration_seconds = time.monotonic() - start_time
            await browser.close()

        # ── 9. Log to codex ──────────────────────────────────────────────────
        if self.codex:
            status = "success" if result.success else "failed"
            detail = result.published_url or (
                "pending review" if result.needs_review else result.errors[0][:100] if result.errors else "unknown"
            )
            try:
                self.codex.log_action(
                    "publish_content",
                    platform_id,
                    f"Published '{content.title}' -> {detail}",
                    status,
                )
            except Exception as log_exc:
                logger.debug(f"Could not log to codex: {log_exc}")

        return result

    async def publish_batch(
        self,
        content: PublishableContent,
        platform_ids: list[str],
    ) -> dict[str, PublishResult]:
        """Publish the same content across multiple platforms sequentially.

        Args:
            content: Content to publish everywhere.
            platform_ids: List of platform IDs to target.

        Returns:
            Dict mapping platform_id -> PublishResult.
        """
        results: dict[str, PublishResult] = {}
        for pid in platform_ids:
            logger.info(f"[ContentPublisher] Batch: publishing on {pid}")
            results[pid] = await self.publish(pid, content)
        return results

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_browser(self, platform_id: str) -> Any:
        """Return configured BrowserManager — re-use injected instance or create new."""
        if self._browser_manager is not None:
            return self._browser_manager

        from openclaw.browser.browser_manager import BrowserManager

        headless = os.environ.get("OPENCLAW_HEADLESS", "true").lower() != "false"
        return BrowserManager(headless=headless)

    async def _navigate_to_upload_page(
        self, platform_id: str, playbook: Any, browser: Any
    ) -> bool:
        """Navigate to the platform's content upload or creation page.

        First checks whether the current page already looks like the create
        page (e.g. if session was restored and the agent is already there).
        Falls back to the hint-driven agent approach.
        """
        task = (
            f"You are on the {platform_id} platform. "
            f"{playbook.upload_page_hint} "
            f"Navigate to the page where you can create a new {playbook.content_type} "
            f"listing or upload a product. "
            "Once you can see the creation/upload form, stop navigating."
        )
        result = await browser.run_agent(
            task=task,
            platform_id=platform_id,
            max_steps=10,
            model=_PUBLISH_MODEL,
        )
        return result.get("success", False)

    async def _fill_content_fields(
        self,
        content: PublishableContent,
        playbook: Any,
        platform_id: str,
        browser: Any,
    ) -> list[str]:
        """Fill all content fields on the upload form.

        Returns the list of field names that were successfully addressed.

        The method builds a single consolidated task to minimise agent round-trips.
        """
        # Build the field-filling instructions from the playbook + content values
        instructions: list[str] = []

        _content_map = {
            "title": content.title,
            "description": content.description,
            "price": str(content.price) if content.price else "",
            "category": content.category,
            "tags": ", ".join(content.tags) if content.tags else "",
            "preview_text": content.preview_text,
            "prompt_text": content.extra_fields.get("prompt_text", ""),
        }
        # Merge extra_fields so platform-specific fields are also filled
        _content_map.update(content.extra_fields)

        addressed_fields: list[str] = []

        for pf in playbook.fields:
            # Skip file upload fields — handled by _upload_files
            if pf.field_type == "file_upload":
                continue

            value = _content_map.get(pf.name, "")
            if not value:
                if pf.required:
                    logger.debug(
                        f"[ContentPublisher] Required field '{pf.name}' has no value"
                    )
                continue

            addressed_fields.append(pf.name)

            if pf.field_type == "tag_input":
                instructions.append(
                    f"Fill the tags/keywords field with: {value}. "
                    f"Add each tag individually if needed. Hint: {pf.hint}"
                )
            elif pf.field_type == "select":
                instructions.append(
                    f"For the '{pf.name}' dropdown/select, choose the option "
                    f"closest to: {value}. Hint: {pf.hint}"
                )
            elif pf.field_type in ("textarea", "text"):
                truncated = value
                if pf.max_length > 0 and len(value) > pf.max_length:
                    truncated = value[: pf.max_length]
                instructions.append(
                    f"Fill the '{pf.name}' field with the following text "
                    f"(clear any existing content first): {truncated}"
                )
            elif pf.field_type == "number":
                instructions.append(
                    f"Set the '{pf.name}' numeric field to: {value}"
                )
            elif pf.field_type == "checkbox":
                instructions.append(
                    f"Check the '{pf.name}' checkbox if it is unchecked."
                )

        if not instructions:
            logger.info(
                f"[ContentPublisher] No text fields to fill on {platform_id}"
            )
            return []

        combined_task = (
            f"Fill in the {platform_id} product creation form with the following "
            f"information. Do each step in order:\n"
            + "\n".join(f"- {instr}" for instr in instructions)
        )

        result = await browser.run_agent(
            task=combined_task,
            platform_id=platform_id,
            max_steps=len(instructions) * 3 + 5,
            model=_PUBLISH_MODEL,
        )
        if not result.get("success", False):
            logger.warning(
                f"[ContentPublisher] Field-filling agent reported failure on {platform_id}"
            )

        return addressed_fields

    async def _upload_files(
        self,
        content: PublishableContent,
        playbook: Any,
        platform_id: str,
        browser: Any,
    ) -> bool:
        """Upload product file and/or cover image.

        Returns True if at least the product file was uploaded successfully
        (or if there is no product file to upload).
        """
        uploads: list[tuple[str, str]] = []  # (field_hint, absolute_path)

        if content.file_path and Path(content.file_path).exists():
            uploads.append(("product/main file upload input", content.file_path))
        elif content.file_path:
            logger.warning(
                f"[ContentPublisher] Product file not found: {content.file_path}"
            )

        if content.cover_image_path and Path(content.cover_image_path).exists():
            uploads.append(("cover image / thumbnail upload input", content.cover_image_path))
        elif content.cover_image_path:
            logger.warning(
                f"[ContentPublisher] Cover image not found: {content.cover_image_path}"
            )

        if not uploads:
            return True  # Nothing to upload

        overall_ok = True
        for field_hint, file_path in uploads:
            task = (
                f"Find the {field_hint} on the {platform_id} form. "
                f"Upload the file located at: {file_path}. "
                f"Wait for the upload to complete (progress bar should disappear "
                f"or a success confirmation should appear) before continuing."
            )
            result = await browser.run_agent(
                task=task,
                platform_id=platform_id,
                max_steps=8,
                model=_PUBLISH_MODEL,
            )
            upload_ok = result.get("success", False)
            if not upload_ok:
                logger.warning(
                    f"[ContentPublisher] Upload failed for {file_path} on {platform_id}"
                )
                overall_ok = False

        return overall_ok

    async def _set_pricing(
        self,
        content: PublishableContent,
        playbook: Any,
        platform_id: str,
        browser: Any,
    ) -> bool:
        """Set the product price.

        Many platforms have a dedicated price input separate from the general
        description fields. This step handles it explicitly.
        """
        price_str = str(content.price) if content.price > 0 else "0"
        task = (
            f"Find the price input field on the {platform_id} form and set it to "
            f"{price_str} {content.currency}. "
            "If the platform has a 'free' toggle, enable it when the price is 0. "
            "If there is a 'pay what you want' or minimum price option, configure it "
            "with the minimum set to the provided price."
        )
        result = await browser.run_agent(
            task=task,
            platform_id=platform_id,
            max_steps=5,
            model=_PUBLISH_MODEL,
        )
        return result.get("success", False)

    async def _submit_publish(
        self,
        platform_id: str,
        playbook: Any,
        browser: Any,
    ) -> tuple[bool, str]:
        """Click the publish/submit button and extract the published URL.

        Returns:
            (success: bool, published_url: str)
        """
        task = (
            f"All fields on the {platform_id} form are filled. "
            f"{playbook.submit_button_hint} "
            "After clicking the button, wait for the page to respond. "
            "If a confirmation dialog appears, confirm it. "
            "Once the listing is live or submitted for review, report the "
            "URL of the published listing page. "
            "If the URL does not change, look for a 'View listing', 'View product', "
            "'See your listing', or 'Share' link and report that URL. "
            "Start your final response with 'PUBLISHED_URL: <url>' if you can "
            "find the listing URL, or 'SUBMITTED_FOR_REVIEW' if the platform "
            "requires manual review."
        )
        result = await browser.run_agent(
            task=task,
            platform_id=platform_id,
            max_steps=15,
            model=_PUBLISH_MODEL,
        )

        success = result.get("success", False)
        final_text: str = result.get("final_text", "") or ""

        # Extract the listing URL from the agent's final response
        published_url = ""
        if "PUBLISHED_URL:" in final_text:
            try:
                url_part = final_text.split("PUBLISHED_URL:")[1].strip().split()[0]
                if url_part.startswith("http"):
                    published_url = url_part
            except (IndexError, ValueError):
                pass

        if not published_url:
            # Try to get the current page URL as a fallback
            try:
                current_url = await browser.get_page_url()
                if current_url and current_url.startswith("http"):
                    published_url = current_url
            except Exception:
                pass

        return success, published_url
