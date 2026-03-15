"""WebhookNotifier -- send notifications on signup events to empire dashboard and external hooks.

Fires HTTP POST requests to configured webhook endpoints whenever
significant events occur in the OpenClaw pipeline: signup started,
completed, failed, CAPTCHA encountered, batch finished, sync completed,
etc.

Configuration is loaded from environment variables:
    OPENCLAW_WEBHOOK_URL     -- Primary webhook endpoint
    OPENCLAW_DASHBOARD_URL   -- Empire dashboard alerts API (port 8000)

Additional webhooks can be registered at runtime via ``add_webhook()``.

All notification delivery is best-effort: failures are logged but never
block the pipeline.  An in-memory event history is kept for debugging
and is accessible via ``get_event_history()``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """All event types that can trigger webhook notifications."""

    APPLY_PROFILE_COMPLETED = "apply_profile_completed"
    APPLY_PROFILE_FAILED = "apply_profile_failed"
    APPLY_PROFILE_STARTED = "apply_profile_started"
    APPROVAL_NEEDED = "approval_needed"
    BATCH_COMPLETED = "batch_completed"
    BATCH_STARTED = "batch_started"
    CAPTCHA_NEEDED = "captcha_needed"
    CAPTCHA_SOLVED = "captcha_solved"
    EMAIL_VERIFICATION_NEEDED = "email_verification_needed"
    EMAIL_VERIFIED = "email_verified"
    ERROR = "error"
    HUMAN_ACTIVITY_COMPLETED = "human_activity_completed"
    HUMAN_ACTIVITY_STARTED = "human_activity_started"
    MISSION_COMPLETED = "mission_completed"
    MISSION_DEPLOYED = "mission_deployed"
    MISSION_FAILED = "mission_failed"
    MISSION_QUEUED = "mission_queued"
    MISSION_STARTED = "mission_started"
    PROFILE_ENHANCED = "profile_enhanced"
    PROFILE_SCORED = "profile_scored"
    PROJECT_DISCOVERED = "project_discovered"
    PUBLISH_COMPLETED = "publish_completed"
    PUBLISH_FAILED = "publish_failed"
    PUBLISH_STARTED = "publish_started"
    SIGNUP_COMPLETED = "signup_completed"
    SIGNUP_FAILED = "signup_failed"
    SIGNUP_STARTED = "signup_started"
    SYNC_COMPLETED = "sync_completed"


# ---------------------------------------------------------------------------
# Webhook config
# ---------------------------------------------------------------------------


@dataclass
class WebhookConfig:
    """Configuration for a single webhook endpoint."""

    url: str
    events: list[EventType] = field(default_factory=lambda: list(EventType))
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    name: str = ""

    def accepts(self, event_type: EventType) -> bool:
        """Check whether this webhook should receive the given event type."""
        return self.enabled and event_type in self.events


# ---------------------------------------------------------------------------
# Delivery result
# ---------------------------------------------------------------------------


@dataclass
class DeliveryResult:
    """Outcome of delivering a single notification to a webhook."""

    webhook_name: str
    url: str
    status_code: int = 0
    success: bool = False
    error: str = ""
    duration_ms: float = 0.0


# =========================================================================== #
#  WebhookNotifier                                                             #
# =========================================================================== #


class WebhookNotifier:
    """Send event notifications to configured webhook endpoints.

    Usage::

        notifier = WebhookNotifier()
        notifier.add_webhook("https://hooks.example.com/openclaw", name="slack")

        await notifier.notify_signup_started("gumroad", "Gumroad")
        await notifier.notify_signup_completed(
            "gumroad", "Gumroad",
            profile_url="https://gumroad.com/openclaw",
            score=85.0,
            duration=42.3,
        )

        # Review history
        history = notifier.get_event_history(limit=10)
        stats = notifier.get_stats()
    """

    # Maximum events to keep in memory
    _MAX_HISTORY = 500

    # HTTP timeout for webhook delivery (seconds)
    _TIMEOUT = 10.0

    def __init__(self) -> None:
        self.webhooks: list[WebhookConfig] = []
        self.event_history: list[dict[str, Any]] = []
        self.telegram_bot: Any = None  # Set by OpenClawEngine after bot init
        self._load_from_env()

    # ------------------------------------------------------------------ #
    #  Configuration                                                       #
    # ------------------------------------------------------------------ #

    def _load_from_env(self) -> None:
        """Load webhook URLs from environment variables.

        OPENCLAW_WEBHOOK_URL   -- Primary webhook (receives all events)
        OPENCLAW_DASHBOARD_URL -- Empire dashboard alerts API
        """
        primary_url = os.environ.get("OPENCLAW_WEBHOOK_URL", "")
        if primary_url:
            self.webhooks.append(WebhookConfig(
                url=primary_url,
                name="primary",
                events=list(EventType),
            ))
            logger.info(f"Loaded primary webhook: {primary_url}")

        dashboard_url = os.environ.get("OPENCLAW_DASHBOARD_URL", "")
        if dashboard_url:
            # Ensure the URL points to the alerts endpoint
            if not dashboard_url.endswith("/"):
                dashboard_url += "/"
            alerts_url = dashboard_url.rstrip("/") + "/api/alerts"
            self.webhooks.append(WebhookConfig(
                url=alerts_url,
                name="empire-dashboard",
                events=list(EventType),
                headers={"Content-Type": "application/json"},
            ))
            logger.info(f"Loaded dashboard webhook: {alerts_url}")

    def add_webhook(
        self,
        url: str,
        events: list[EventType] | None = None,
        name: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        """Register a new webhook endpoint.

        Args:
            url: The URL to POST event payloads to.
            events: List of event types to send.  Defaults to all events.
            name: Human-readable label for this webhook.
            headers: Extra HTTP headers to include in requests.
        """
        config = WebhookConfig(
            url=url,
            events=events if events is not None else list(EventType),
            name=name or url[:40],
            headers=headers or {},
            enabled=True,
        )
        self.webhooks.append(config)
        logger.info(f"Added webhook '{config.name}': {url}")

    def remove_webhook(self, name: str) -> bool:
        """Remove a webhook by name.

        Args:
            name: The name of the webhook to remove.

        Returns:
            True if a webhook was removed, False if not found.
        """
        before = len(self.webhooks)
        self.webhooks = [w for w in self.webhooks if w.name != name]
        removed = len(self.webhooks) < before
        if removed:
            logger.info(f"Removed webhook '{name}'")
        return removed

    # ------------------------------------------------------------------ #
    #  Core notification                                                   #
    # ------------------------------------------------------------------ #

    async def notify(
        self,
        event_type: EventType,
        data: dict[str, Any],
    ) -> list[DeliveryResult]:
        """Send a notification to all matching webhooks.

        Args:
            event_type: The type of event.
            data: Event-specific payload data.

        Returns:
            A list of DeliveryResult objects, one per webhook that was
            contacted.
        """
        payload = {
            "event_type": event_type.value,
            "timestamp": datetime.now().isoformat(),
            "source": "openclaw-agent",
            "data": data,
        }

        # Record in history
        self._record_event(event_type, data)

        # Push to Telegram bot if wired
        if self.telegram_bot is not None:
            try:
                await self.telegram_bot.notify_if_not_muted(
                    event_type.value, data,
                )
            except Exception as e:
                logger.debug(f"Telegram notification failed (non-critical): {e}")

        # Find matching webhooks
        targets = [w for w in self.webhooks if w.accepts(event_type)]
        if not targets:
            logger.debug(
                f"No webhooks configured for event {event_type.value}"
            )
            return []

        results: list[DeliveryResult] = []
        async with httpx.AsyncClient(timeout=self._TIMEOUT) as client:
            for webhook in targets:
                result = await self._deliver(client, webhook, payload)
                results.append(result)

        return results

    async def _deliver(
        self,
        client: httpx.AsyncClient,
        webhook: WebhookConfig,
        payload: dict[str, Any],
    ) -> DeliveryResult:
        """Deliver a payload to a single webhook endpoint.

        Delivery is best-effort: exceptions are caught and logged, never
        re-raised.

        Args:
            client: The httpx async client to use.
            webhook: The target webhook configuration.
            payload: The JSON payload to send.

        Returns:
            A DeliveryResult with status and timing information.
        """
        result = DeliveryResult(
            webhook_name=webhook.name,
            url=webhook.url,
        )

        headers = {"Content-Type": "application/json"}
        headers.update(webhook.headers)

        start = time.monotonic()
        try:
            response = await client.post(
                webhook.url,
                json=payload,
                headers=headers,
            )
            result.status_code = response.status_code
            result.success = 200 <= response.status_code < 300

            if not result.success:
                result.error = f"HTTP {response.status_code}"
                logger.warning(
                    f"Webhook '{webhook.name}' returned {response.status_code}: "
                    f"{response.text[:200]}"
                )
            else:
                logger.debug(
                    f"Webhook '{webhook.name}' delivered: "
                    f"{payload.get('event_type', 'unknown')}"
                )

        except httpx.TimeoutException:
            result.error = "timeout"
            logger.warning(f"Webhook '{webhook.name}' timed out ({self._TIMEOUT}s)")

        except httpx.ConnectError as exc:
            result.error = f"connection_error: {exc}"
            logger.warning(f"Webhook '{webhook.name}' connection failed: {exc}")

        except Exception as exc:
            result.error = str(exc)
            logger.error(f"Webhook '{webhook.name}' unexpected error: {exc}")

        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    # ------------------------------------------------------------------ #
    #  Convenience methods — typed event helpers                            #
    # ------------------------------------------------------------------ #

    async def notify_signup_started(
        self, platform_id: str, platform_name: str
    ) -> list[DeliveryResult]:
        """Notify that a signup has started on a platform."""
        return await self.notify(EventType.SIGNUP_STARTED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
        })

    async def notify_signup_completed(
        self,
        platform_id: str,
        platform_name: str,
        profile_url: str = "",
        score: float = 0.0,
        duration: float = 0.0,
    ) -> list[DeliveryResult]:
        """Notify that a signup was completed successfully."""
        return await self.notify(EventType.SIGNUP_COMPLETED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "profile_url": profile_url,
            "sentinel_score": score,
            "duration_seconds": round(duration, 1),
        })

    async def notify_signup_failed(
        self,
        platform_id: str,
        platform_name: str,
        error: str = "",
        step_completed: int = 0,
    ) -> list[DeliveryResult]:
        """Notify that a signup attempt failed."""
        return await self.notify(EventType.SIGNUP_FAILED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "error": error,
            "steps_completed": step_completed,
        })

    async def notify_captcha_needed(
        self,
        platform_id: str,
        captcha_type: str = "",
        task_id: str = "",
    ) -> list[DeliveryResult]:
        """Notify that a CAPTCHA was encountered and needs solving."""
        return await self.notify(EventType.CAPTCHA_NEEDED, {
            "platform_id": platform_id,
            "captcha_type": captcha_type,
            "task_id": task_id,
        })

    async def notify_captcha_solved(
        self,
        platform_id: str,
        captcha_type: str = "",
        auto_solved: bool = False,
        duration: float = 0.0,
    ) -> list[DeliveryResult]:
        """Notify that a CAPTCHA was solved."""
        return await self.notify(EventType.CAPTCHA_SOLVED, {
            "platform_id": platform_id,
            "captcha_type": captcha_type,
            "auto_solved": auto_solved,
            "duration_seconds": round(duration, 1),
        })

    async def notify_email_verification_needed(
        self,
        platform_id: str,
        email: str = "",
    ) -> list[DeliveryResult]:
        """Notify that email verification is required."""
        # Mask the email for privacy
        masked = ""
        if email and "@" in email:
            local, domain = email.split("@", 1)
            masked = local[:2] + "***@" + domain
        return await self.notify(EventType.EMAIL_VERIFICATION_NEEDED, {
            "platform_id": platform_id,
            "email_masked": masked,
        })

    async def notify_email_verified(
        self, platform_id: str
    ) -> list[DeliveryResult]:
        """Notify that email verification was completed."""
        return await self.notify(EventType.EMAIL_VERIFIED, {
            "platform_id": platform_id,
        })

    async def notify_profile_scored(
        self,
        platform_id: str,
        platform_name: str,
        score: float = 0.0,
        grade: str = "",
    ) -> list[DeliveryResult]:
        """Notify that a profile was scored by the Sentinel."""
        return await self.notify(EventType.PROFILE_SCORED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "sentinel_score": score,
            "grade": grade,
        })

    async def notify_batch_started(
        self,
        platform_ids: list[str],
    ) -> list[DeliveryResult]:
        """Notify that a batch signup has started."""
        return await self.notify(EventType.BATCH_STARTED, {
            "platform_ids": platform_ids,
            "total_platforms": len(platform_ids),
        })

    async def notify_batch_completed(
        self,
        total: int,
        succeeded: int,
        failed: int,
        duration: float = 0.0,
    ) -> list[DeliveryResult]:
        """Notify that a batch signup run has completed."""
        return await self.notify(EventType.BATCH_COMPLETED, {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "duration_seconds": round(duration, 1),
            "success_rate": round(succeeded / total * 100, 1) if total > 0 else 0.0,
        })

    async def notify_sync_completed(
        self,
        total_platforms: int,
        succeeded: int,
        failed: int,
        fields_updated: list[str] | None = None,
    ) -> list[DeliveryResult]:
        """Notify that a profile sync has completed."""
        return await self.notify(EventType.SYNC_COMPLETED, {
            "total_platforms": total_platforms,
            "succeeded": succeeded,
            "failed": failed,
            "fields_updated": fields_updated or [],
        })

    async def notify_error(
        self,
        error: str,
        context: dict[str, Any] | None = None,
    ) -> list[DeliveryResult]:
        """Notify about a general error."""
        data: dict[str, Any] = {"error": error}
        if context:
            data["context"] = context
        return await self.notify(EventType.ERROR, data)

    async def notify_apply_profile_started(
        self, platform_id: str, platform_name: str
    ) -> list[DeliveryResult]:
        """Notify that profile application has started on a platform."""
        return await self.notify(EventType.APPLY_PROFILE_STARTED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
        })

    async def notify_apply_profile_completed(
        self,
        platform_id: str,
        platform_name: str,
        fields_applied: list[str],
        fields_failed: list[str],
    ) -> list[DeliveryResult]:
        """Notify that profile application completed on a platform."""
        return await self.notify(EventType.APPLY_PROFILE_COMPLETED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "fields_applied": fields_applied,
            "fields_failed": fields_failed,
        })

    async def notify_apply_profile_failed(
        self,
        platform_id: str,
        platform_name: str,
        error: str,
    ) -> list[DeliveryResult]:
        """Notify that profile application failed on a platform."""
        return await self.notify(EventType.APPLY_PROFILE_FAILED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "error": error,
        })

    async def notify_human_activity_started(
        self, platform_id: str, platform_name: str
    ) -> list[DeliveryResult]:
        """Notify that human activity simulation has started on a platform."""
        return await self.notify(EventType.HUMAN_ACTIVITY_STARTED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
        })

    async def notify_human_activity_completed(
        self,
        platform_id: str,
        platform_name: str,
        activities_completed: int,
        activities_failed: int,
        duration_seconds: float,
    ) -> list[DeliveryResult]:
        """Notify that human activity simulation completed on a platform."""
        return await self.notify(EventType.HUMAN_ACTIVITY_COMPLETED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "activities_completed": activities_completed,
            "activities_failed": activities_failed,
            "duration_seconds": round(duration_seconds, 1),
        })

    async def notify_publish_started(
        self, platform_id: str, platform_name: str, title: str
    ) -> list[DeliveryResult]:
        """Notify that content publishing has started on a platform."""
        return await self.notify(EventType.PUBLISH_STARTED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "title": title,
        })

    async def notify_publish_completed(
        self,
        platform_id: str,
        platform_name: str,
        title: str,
        published_url: str,
        needs_review: bool = False,
    ) -> list[DeliveryResult]:
        """Notify that content was published successfully on a platform."""
        return await self.notify(EventType.PUBLISH_COMPLETED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "title": title,
            "published_url": published_url,
            "needs_review": needs_review,
        })

    async def notify_publish_failed(
        self,
        platform_id: str,
        platform_name: str,
        title: str,
        error: str,
    ) -> list[DeliveryResult]:
        """Notify that content publishing failed on a platform."""
        return await self.notify(EventType.PUBLISH_FAILED, {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "title": title,
            "error": error,
        })

    async def notify_profile_enhanced(
        self,
        platform_id: str,
        grade: str,
        score: float,
    ) -> list[DeliveryResult]:
        """Notify that a profile was enhanced and graded."""
        return await self.notify(EventType.PROFILE_ENHANCED, {
            "platform_id": platform_id,
            "grade": grade,
            "score": round(score, 0),
        })

    async def notify_approval_needed(
        self,
        action_type: str,
        target: str,
        description: str,
    ) -> list[DeliveryResult]:
        """Notify that a human approval is required before proceeding."""
        return await self.notify(EventType.APPROVAL_NEEDED, {
            "action_type": action_type,
            "target": target,
            "description": description,
        })

    # ------------------------------------------------------------------ #
    #  Event history                                                       #
    # ------------------------------------------------------------------ #

    def _record_event(self, event_type: EventType, data: dict[str, Any]) -> None:
        """Record an event in the in-memory history ring buffer."""
        self.event_history.append({
            "event_type": event_type.value,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        })

        # Trim to max size
        if len(self.event_history) > self._MAX_HISTORY:
            self.event_history = self.event_history[-self._MAX_HISTORY:]

    def get_event_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent event history, newest first.

        Args:
            limit: Maximum number of events to return.

        Returns:
            A list of event dicts with ``event_type``, ``timestamp``,
            and ``data`` keys.
        """
        return list(reversed(self.event_history[-limit:]))

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate notification statistics.

        Returns:
            A dict with keys:
            - total_events: Total events recorded
            - webhooks_configured: Number of active webhook endpoints
            - events_by_type: Dict mapping event type to count
            - recent_errors: Count of ERROR events in the last 50
        """
        events_by_type: dict[str, int] = {}
        for event in self.event_history:
            et = event.get("event_type", "unknown")
            events_by_type[et] = events_by_type.get(et, 0) + 1

        recent = self.event_history[-50:] if self.event_history else []
        recent_errors = sum(
            1 for e in recent
            if e.get("event_type") == EventType.ERROR.value
        )

        return {
            "total_events": len(self.event_history),
            "webhooks_configured": len(self.webhooks),
            "webhooks_enabled": sum(1 for w in self.webhooks if w.enabled),
            "events_by_type": events_by_type,
            "recent_errors": recent_errors,
        }
