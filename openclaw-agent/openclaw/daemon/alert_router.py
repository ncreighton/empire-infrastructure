"""AlertRouter — severity routing, dedup, quiet hours, rate limiting.

Wires into WebhookNotifier for delivery. Adds content-hash dedup,
quiet-hours suppression, and per-source daily rate limits.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any

from openclaw.daemon.heartbeat_config import HeartbeatConfig
from openclaw.forge.platform_codex import PlatformCodex
from openclaw.automation.webhook_notifier import EventType, WebhookNotifier
from openclaw.models import Alert, AlertSeverity

logger = logging.getLogger(__name__)


class AlertRouter:
    """Route alerts through severity/dedup/quiet-hours/rate-limit pipeline."""

    def __init__(
        self,
        config: HeartbeatConfig,
        codex: PlatformCodex,
        notifier: WebhookNotifier,
    ):
        self.config = config
        self.codex = codex
        self.notifier = notifier

    async def route(self, alert: Alert) -> bool:
        """Route an alert through the full pipeline.

        Returns True if delivered, False if suppressed.
        """
        # Fill defaults
        if not alert.alert_id:
            alert.alert_id = str(uuid.uuid4())[:12]
        if not alert.created_at:
            alert.created_at = datetime.now()
        if not alert.content_hash:
            alert.content_hash = self._compute_hash(alert)

        # 1. Dedup: check content_hash within dedup window
        if self._is_duplicate(alert):
            alert.suppressed = True
            self.codex.insert_alert(alert)
            logger.debug(f"Alert suppressed (dedup): {alert.title}")
            return False

        # 2. Quiet hours: suppress WARNING/INFO during quiet hours
        if self._in_quiet_hours() and alert.severity != AlertSeverity.CRITICAL:
            alert.suppressed = True
            self.codex.insert_alert(alert)
            logger.debug(f"Alert suppressed (quiet hours): {alert.title}")
            return False

        # 3. Rate limit: max N alerts per day per source
        if self._rate_limited(alert) and alert.severity != AlertSeverity.CRITICAL:
            alert.suppressed = True
            self.codex.insert_alert(alert)
            logger.debug(f"Alert suppressed (rate limit): {alert.title}")
            return False

        # 4. Deliver via WebhookNotifier
        try:
            await self.notifier.notify(EventType.ERROR, {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "source": alert.source,
                "title": alert.title,
                "message": alert.message,
                "details": alert.details,
            })
        except Exception as e:
            logger.warning(f"Alert delivery failed (non-critical): {e}")

        alert.delivered = True
        self.codex.insert_alert(alert)
        logger.info(f"Alert delivered [{alert.severity.value}]: {alert.title}")
        return True

    async def flush_queued(self) -> int:
        """Deliver suppressed alerts that are now outside quiet hours.

        Called by the SCAN tier to deliver queued alerts after quiet hours end.

        Returns:
            Count of alerts flushed.
        """
        if self._in_quiet_hours():
            return 0

        suppressed = self.codex.get_suppressed_alerts()
        flushed = 0
        for row in suppressed:
            if row.get("severity") == AlertSeverity.CRITICAL.value:
                continue  # Already delivered

            try:
                await self.notifier.notify(EventType.ERROR, {
                    "alert_id": row["alert_id"],
                    "severity": row["severity"],
                    "source": row["source"],
                    "title": row["title"],
                    "message": row["message"],
                    "details": row.get("details", {}),
                    "queued": True,
                })
                self.codex.acknowledge_alert(row["alert_id"])
                flushed += 1
            except Exception as e:
                logger.debug(f"Flush delivery failed: {e}")

        if flushed:
            logger.info(f"Flushed {flushed} queued alerts")
        return flushed

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if same content_hash exists within dedup window."""
        return self.codex.alert_hash_exists(
            alert.content_hash,
            window_hours=self.config.dedup_window_hours,
        )

    def _in_quiet_hours(self) -> bool:
        """Check current time against quiet hours config."""
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(self.config.quiet_timezone)
            now = datetime.now(tz)
        except (ImportError, KeyError):
            now = datetime.now()

        hour = now.hour
        start = self.config.quiet_start_hour
        end = self.config.quiet_end_hour

        if start > end:
            # Crosses midnight (e.g., 23:00 - 07:00)
            return hour >= start or hour < end
        else:
            return start <= hour < end

    def _rate_limited(self, alert: Alert) -> bool:
        """Check if source has exceeded daily alert limit."""
        count = self.codex.get_alert_count_today(alert.source)
        return count >= self.config.max_alerts_per_day

    @staticmethod
    def _compute_hash(alert: Alert) -> str:
        """Compute a content hash for deduplication."""
        content = f"{alert.source}:{alert.severity.value}:{alert.title}:{alert.message}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
