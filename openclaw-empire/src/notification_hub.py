"""
Notification Hub — OpenClaw Empire Edition

Unified notification delivery system for Nick Creighton's 16-site WordPress
publishing empire.  Routes alerts to the right channels based on severity,
category, and user preferences.

Channels:
    WhatsApp   — via OpenClaw gateway REST API
    Telegram   — via Bot API (sendMessage)
    Discord    — via webhook (rich embeds)
    Email      — via stdlib smtplib (HTML with severity-colored headers)
    Android    — via Termux:API through OpenClaw node invocation

All data persisted to: data/notifications/

Usage:
    from src.notification_hub import get_hub

    hub = get_hub()
    hub.send("Deploy complete", "All 16 sites updated.", severity="success")

    # Convenience
    hub.critical("Site down", "witchcraftforbeginners.com unreachable")

    # Pre-built template
    hub.notify_site_down("witchcraft", "witchcraftforbeginners.com", "HTTP 503")
"""

from __future__ import annotations

import argparse
import asyncio
import email.mime.multipart
import email.mime.text
import json
import logging
import os
import smtplib
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("notification_hub")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

NOTIFICATION_DATA_DIR = Path(r"D:\Claude Code Projects\openclaw-empire\data\notifications")
HISTORY_FILE = NOTIFICATION_DATA_DIR / "history.json"
CHANNELS_FILE = NOTIFICATION_DATA_DIR / "channels.json"
RULES_FILE = NOTIFICATION_DATA_DIR / "rules.json"
DIGEST_STATE_FILE = NOTIFICATION_DATA_DIR / "digest_state.json"

# Ensure data directory exists on import
NOTIFICATION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Maximum history entries to keep on disk
MAX_HISTORY_ENTRIES = 2000

# ---------------------------------------------------------------------------
# Environment variable defaults
# ---------------------------------------------------------------------------

OPENCLAW_GATEWAY_URL = os.environ.get("OPENCLAW_GATEWAY_URL", "http://localhost:18789")
OPENCLAW_GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
NOTIFICATION_EMAIL_TO = os.environ.get("NOTIFICATION_EMAIL_TO", "")

# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc


def _now_utc() -> datetime:
    """Return the current time in UTC, timezone-aware."""
    return datetime.now(UTC)


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return _now_utc().isoformat()


def _now_eastern() -> datetime:
    """Return the current time in US/Eastern, timezone-aware."""
    return _now_utc().astimezone(EASTERN)


def _parse_iso(s: str | None) -> datetime | None:
    """Parse an ISO-8601 string back to a timezone-aware datetime."""
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


# ---------------------------------------------------------------------------
# JSON persistence helpers (atomic writes)
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        if os.name == "nt":
            os.replace(str(tmp), str(path))
        else:
            tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ===================================================================
# SEVERITY & CATEGORY ENUMS
# ===================================================================

class Severity(str, Enum):
    """Notification severity levels, ordered from lowest to highest."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    CRITICAL = "critical"


SEVERITY_RANK: dict[str, int] = {
    "info": 0,
    "success": 0,
    "warning": 1,
    "critical": 2,
}


class Category(str, Enum):
    """Notification categories matching empire subsystems."""
    REVENUE = "revenue"
    CONTENT = "content"
    SEO = "seo"
    HEALTH = "health"
    SECURITY = "security"
    SCHEDULER = "scheduler"
    GENERAL = "general"


class Channel(str, Enum):
    """Supported delivery channels."""
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    ANDROID = "android"


# Severity-based emoji prefixes for plain-text channels
SEVERITY_EMOJI: dict[str, str] = {
    "info": "[i]",
    "success": "[OK]",
    "warning": "[!]",
    "critical": "[!!!]",
}

# Discord embed colors by severity (decimal)
DISCORD_COLORS: dict[str, int] = {
    "info": 0x3498DB,      # blue
    "success": 0x2ECC71,   # green
    "warning": 0xF39C12,   # yellow/orange
    "critical": 0xE74C3C,  # red
}

# Email header background colors
EMAIL_COLORS: dict[str, str] = {
    "info": "#3498DB",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "critical": "#E74C3C",
}


# ===================================================================
# DATA CLASSES
# ===================================================================

@dataclass
class Notification:
    """A single notification record with delivery tracking."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    message: str = ""
    severity: str = "info"
    category: str = "general"
    source: str = ""
    site_id: str | None = None
    data: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    delivered_to: list[str] = field(default_factory=list)
    delivery_status: dict[str, str] = field(default_factory=dict)
    read: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Notification:
        data = dict(data)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class ChannelConfig:
    """Configuration for a single delivery channel."""

    channel: str = "whatsapp"
    enabled: bool = False
    config: dict = field(default_factory=dict)
    min_severity: str = "info"
    categories: list[str] = field(default_factory=lambda: ["all"])
    quiet_hours: tuple[int, int] | None = None  # e.g., (23, 7) = 11PM-7AM

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuple to list for JSON serialization
        if d["quiet_hours"] is not None:
            d["quiet_hours"] = list(d["quiet_hours"])
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ChannelConfig:
        data = dict(data)
        # Convert list back to tuple for quiet_hours
        if data.get("quiet_hours") is not None:
            data["quiet_hours"] = tuple(data["quiet_hours"])
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class NotificationRule:
    """Routing rule that maps category+severity to channels."""

    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: str = "general"
    min_severity: str = "info"
    channels: list[str] = field(default_factory=list)
    template: str | None = None
    throttle_minutes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NotificationRule:
        data = dict(data)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ===================================================================
# DEFAULT RULES
# ===================================================================

DEFAULT_RULES: list[dict] = [
    {
        "rule_id": "rule-critical-all",
        "category": "all",
        "min_severity": "critical",
        "channels": ["whatsapp", "telegram", "discord", "email", "android"],
        "throttle_minutes": 0,
    },
    {
        "rule_id": "rule-revenue-alerts",
        "category": "revenue",
        "min_severity": "warning",
        "channels": ["whatsapp", "telegram"],
        "throttle_minutes": 5,
    },
    {
        "rule_id": "rule-content-published",
        "category": "content",
        "min_severity": "success",
        "channels": ["discord"],
        "throttle_minutes": 0,
    },
    {
        "rule_id": "rule-seo-issues",
        "category": "seo",
        "min_severity": "warning",
        "channels": ["email"],
        "throttle_minutes": 1440,  # daily digest = 24h throttle
    },
    {
        "rule_id": "rule-health-alerts",
        "category": "health",
        "min_severity": "warning",
        "channels": ["whatsapp", "android"],
        "throttle_minutes": 10,
    },
    {
        "rule_id": "rule-scheduler-failures",
        "category": "scheduler",
        "min_severity": "warning",
        "channels": ["telegram"],
        "throttle_minutes": 15,
    },
    {
        "rule_id": "rule-security-events",
        "category": "security",
        "min_severity": "warning",
        "channels": ["whatsapp", "telegram", "android"],
        "throttle_minutes": 0,
    },
    {
        "rule_id": "rule-general-info",
        "category": "general",
        "min_severity": "info",
        "channels": ["discord"],
        "throttle_minutes": 0,
    },
]

# Default channel configurations
DEFAULT_CHANNELS: dict[str, dict] = {
    "whatsapp": {
        "channel": "whatsapp",
        "enabled": True,
        "config": {
            "gateway_url": OPENCLAW_GATEWAY_URL,
            "gateway_token": OPENCLAW_GATEWAY_TOKEN,
        },
        "min_severity": "warning",
        "categories": ["all"],
        "quiet_hours": [23, 7],
    },
    "telegram": {
        "channel": "telegram",
        "enabled": bool(TELEGRAM_BOT_TOKEN),
        "config": {
            "bot_token": TELEGRAM_BOT_TOKEN,
            "chat_id": TELEGRAM_CHAT_ID,
        },
        "min_severity": "info",
        "categories": ["all"],
        "quiet_hours": None,
    },
    "discord": {
        "channel": "discord",
        "enabled": bool(DISCORD_WEBHOOK_URL),
        "config": {
            "webhook_url": DISCORD_WEBHOOK_URL,
        },
        "min_severity": "info",
        "categories": ["all"],
        "quiet_hours": None,
    },
    "email": {
        "channel": "email",
        "enabled": bool(SMTP_HOST and SMTP_USER),
        "config": {
            "smtp_host": SMTP_HOST,
            "smtp_port": SMTP_PORT,
            "smtp_user": SMTP_USER,
            "smtp_password": SMTP_PASSWORD,
            "to_address": NOTIFICATION_EMAIL_TO,
        },
        "min_severity": "warning",
        "categories": ["all"],
        "quiet_hours": None,
    },
    "android": {
        "channel": "android",
        "enabled": True,
        "config": {
            "gateway_url": OPENCLAW_GATEWAY_URL,
            "gateway_token": OPENCLAW_GATEWAY_TOKEN,
        },
        "min_severity": "warning",
        "categories": ["health", "security"],
        "quiet_hours": [23, 7],
    },
}


# ===================================================================
# NOTIFICATION HUB
# ===================================================================

class NotificationHub:
    """
    Central notification routing hub for the OpenClaw empire.

    Routes notifications to the appropriate channels based on rules,
    severity, category, quiet hours, and throttle settings.  Provides
    template methods for common empire events, message formatting for
    each channel, and a persistent history + digest system.
    """

    def __init__(self) -> None:
        self._channels: dict[str, ChannelConfig] = {}
        self._rules: list[NotificationRule] = []
        self._history: list[dict] = []
        self._throttle_tracker: dict[str, float] = {}  # key -> last_sent_timestamp
        self._digest_state: dict = {}

        # Load persisted state
        self._load_channels()
        self._load_rules()
        self._load_history()
        self._load_digest_state()

        logger.info(
            "NotificationHub initialized — %d channels, %d rules, %d history entries.",
            len(self._channels),
            len(self._rules),
            len(self._history),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_channels(self) -> None:
        """Load channel configs from disk, merging with defaults."""
        raw = _load_json(CHANNELS_FILE, {})
        self._channels = {}

        # Start with defaults
        for name, default_data in DEFAULT_CHANNELS.items():
            if name in raw:
                try:
                    self._channels[name] = ChannelConfig.from_dict(raw[name])
                except (TypeError, KeyError) as exc:
                    logger.warning("Malformed channel config for %s: %s", name, exc)
                    self._channels[name] = ChannelConfig.from_dict(default_data)
            else:
                self._channels[name] = ChannelConfig.from_dict(default_data)

        # Load any extra channels from disk that are not defaults
        for name, data in raw.items():
            if name not in self._channels:
                try:
                    self._channels[name] = ChannelConfig.from_dict(data)
                except (TypeError, KeyError) as exc:
                    logger.warning("Skipping malformed channel %s: %s", name, exc)

    def _save_channels(self) -> None:
        """Persist channel configs to disk."""
        data = {name: cfg.to_dict() for name, cfg in self._channels.items()}
        _save_json(CHANNELS_FILE, data)

    def _load_rules(self) -> None:
        """Load notification rules from disk, falling back to defaults."""
        raw = _load_json(RULES_FILE, [])
        if isinstance(raw, list) and raw:
            self._rules = []
            for item in raw:
                try:
                    self._rules.append(NotificationRule.from_dict(item))
                except (TypeError, KeyError) as exc:
                    logger.warning("Skipping malformed rule: %s", exc)
        else:
            # Initialize with default rules
            self._rules = [NotificationRule.from_dict(r) for r in DEFAULT_RULES]
            self._save_rules()

    def _save_rules(self) -> None:
        """Persist notification rules to disk."""
        _save_json(RULES_FILE, [r.to_dict() for r in self._rules])

    def _load_history(self) -> None:
        """Load notification history from disk."""
        raw = _load_json(HISTORY_FILE, [])
        if isinstance(raw, list):
            self._history = raw[-MAX_HISTORY_ENTRIES:]
        else:
            self._history = []

    def _save_history(self) -> None:
        """Persist notification history to disk."""
        self._history = self._history[-MAX_HISTORY_ENTRIES:]
        _save_json(HISTORY_FILE, self._history)

    def _load_digest_state(self) -> None:
        """Load digest deduplication state."""
        self._digest_state = _load_json(DIGEST_STATE_FILE, {
            "last_daily_digest": None,
            "last_weekly_summary": None,
            "digest_notification_ids": [],
        })

    def _save_digest_state(self) -> None:
        """Persist digest state."""
        _save_json(DIGEST_STATE_FILE, self._digest_state)

    def _append_history(self, notification: Notification) -> None:
        """Append a notification to the history buffer and persist."""
        self._history.append(notification.to_dict())
        if len(self._history) > MAX_HISTORY_ENTRIES:
            self._history = self._history[-MAX_HISTORY_ENTRIES:]
        self._save_history()

    # ------------------------------------------------------------------
    # Quiet hours check
    # ------------------------------------------------------------------

    def _is_quiet_hours(self, quiet_hours: tuple[int, int] | None) -> bool:
        """Check if the current time in US/Eastern falls within quiet hours.

        quiet_hours is a tuple (start_hour, end_hour) where the range wraps
        at midnight.  E.g., (23, 7) means 11PM through 7AM is quiet.
        """
        if quiet_hours is None:
            return False
        start_h, end_h = quiet_hours
        now_h = _now_eastern().hour
        if start_h > end_h:
            # Wraps midnight: e.g., 23-7 means 23,0,1,2,3,4,5,6
            return now_h >= start_h or now_h < end_h
        else:
            # Same-day range: e.g., 1-6 means 1,2,3,4,5
            return start_h <= now_h < end_h

    # ------------------------------------------------------------------
    # Throttle check
    # ------------------------------------------------------------------

    def _throttle_key(self, category: str, source: str) -> str:
        """Build the key used for throttle tracking."""
        return f"{category}:{source}"

    def _is_throttled(self, category: str, source: str, throttle_minutes: int) -> bool:
        """Check if a notification for this category+source is within the throttle window."""
        if throttle_minutes <= 0:
            return False
        key = self._throttle_key(category, source)
        last_sent = self._throttle_tracker.get(key, 0.0)
        elapsed = time.time() - last_sent
        return elapsed < (throttle_minutes * 60)

    def _record_throttle(self, category: str, source: str) -> None:
        """Record the current time as the last send time for throttle tracking."""
        key = self._throttle_key(category, source)
        self._throttle_tracker[key] = time.time()

    # ------------------------------------------------------------------
    # Channel routing
    # ------------------------------------------------------------------

    def _resolve_channels(self, notification: Notification) -> list[str]:
        """Determine which channels a notification should be delivered to.

        Evaluates all rules that match the notification's category and
        severity, then filters by channel-level config (enabled, min_severity,
        categories, quiet hours).  Critical notifications bypass quiet hours.
        """
        severity_rank = SEVERITY_RANK.get(notification.severity, 0)
        target_channels: set[str] = set()

        for rule in self._rules:
            # Category match: "all" matches everything, or exact match
            cat_match = (
                rule.category == "all"
                or rule.category == notification.category
            )
            if not cat_match:
                continue

            # Severity meets rule minimum
            rule_rank = SEVERITY_RANK.get(rule.min_severity, 0)
            if severity_rank < rule_rank:
                continue

            # Check throttle for this rule
            if rule.throttle_minutes > 0:
                if self._is_throttled(
                    notification.category,
                    notification.source or "",
                    rule.throttle_minutes,
                ):
                    # Critical notifications bypass throttle
                    if notification.severity != "critical":
                        continue

            target_channels.update(rule.channels)

        # Filter by channel-level config
        final_channels: list[str] = []
        for ch_name in target_channels:
            cfg = self._channels.get(ch_name)
            if cfg is None or not cfg.enabled:
                continue

            # Channel-level severity gate
            ch_rank = SEVERITY_RANK.get(cfg.min_severity, 0)
            if severity_rank < ch_rank:
                continue

            # Channel-level category filter
            if "all" not in cfg.categories:
                if notification.category not in cfg.categories:
                    continue

            # Quiet hours (critical bypasses)
            if notification.severity != "critical":
                if self._is_quiet_hours(cfg.quiet_hours):
                    continue

            final_channels.append(ch_name)

        return sorted(set(final_channels))

    # ==================================================================
    # SENDING
    # ==================================================================

    def send(
        self,
        title: str,
        message: str,
        severity: str = "info",
        category: str = "general",
        source: str | None = None,
        site_id: str | None = None,
        data: dict | None = None,
    ) -> Notification:
        """Send a notification, routing it to appropriate channels.

        This is the primary entry point.  Creates a Notification object,
        resolves target channels via rules, delivers to each, and records
        the result in history.

        Args:
            title:    Short headline text
            message:  Detailed notification body
            severity: "info", "success", "warning", or "critical"
            category: "revenue", "content", "seo", "health", "security",
                      "scheduler", or "general"
            source:   Name of the calling module (for throttle grouping)
            site_id:  Relevant site ID, if applicable
            data:     Arbitrary metadata dict

        Returns:
            The completed Notification with delivery status populated.
        """
        notification = Notification(
            title=title,
            message=message,
            severity=severity,
            category=category,
            source=source or "",
            site_id=site_id,
            data=data or {},
        )

        channels = self._resolve_channels(notification)
        logger.info(
            "Sending notification %r [%s/%s] to %d channel(s): %s",
            title, severity, category, len(channels),
            ", ".join(channels) if channels else "(none)",
        )

        for ch_name in channels:
            success = self._deliver_to_channel(notification, ch_name)
            status = "sent" if success else "failed"
            notification.delivery_status[ch_name] = status
            if success:
                notification.delivered_to.append(ch_name)

        # Record throttle for matched rules
        if notification.delivered_to:
            self._record_throttle(category, source or "")

        # Persist to history
        self._append_history(notification)
        return notification

    def send_to_channel(self, notification: Notification, channel: str) -> bool:
        """Send a notification directly to a specific channel, bypassing rules.

        Returns True if delivery succeeded.
        """
        success = self._deliver_to_channel(notification, channel)
        notification.delivery_status[channel] = "sent" if success else "failed"
        if success and channel not in notification.delivered_to:
            notification.delivered_to.append(channel)
        self._append_history(notification)
        return success

    # ------------------------------------------------------------------
    # Convenience severity methods
    # ------------------------------------------------------------------

    def info(self, title: str, message: str, **kwargs: Any) -> Notification:
        """Send an info-level notification."""
        return self.send(title, message, severity="info", **kwargs)

    def warning(self, title: str, message: str, **kwargs: Any) -> Notification:
        """Send a warning-level notification."""
        return self.send(title, message, severity="warning", **kwargs)

    def critical(self, title: str, message: str, **kwargs: Any) -> Notification:
        """Send a critical-level notification."""
        return self.send(title, message, severity="critical", **kwargs)

    def success(self, title: str, message: str, **kwargs: Any) -> Notification:
        """Send a success-level notification."""
        return self.send(title, message, severity="success", **kwargs)

    # ==================================================================
    # ASYNC WRAPPERS
    # ==================================================================

    async def asend(
        self,
        title: str,
        message: str,
        severity: str = "info",
        category: str = "general",
        source: str | None = None,
        site_id: str | None = None,
        data: dict | None = None,
    ) -> Notification:
        """Async version of send().  Delivers to all channels concurrently."""
        notification = Notification(
            title=title,
            message=message,
            severity=severity,
            category=category,
            source=source or "",
            site_id=site_id,
            data=data or {},
        )

        channels = self._resolve_channels(notification)
        logger.info(
            "Async-sending notification %r [%s/%s] to %d channel(s): %s",
            title, severity, category, len(channels),
            ", ".join(channels) if channels else "(none)",
        )

        if channels:
            tasks = [
                self._adeliver_to_channel(notification, ch) for ch in channels
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for ch_name, result in zip(channels, results):
                if isinstance(result, Exception):
                    logger.error("Async delivery to %s failed: %s", ch_name, result)
                    notification.delivery_status[ch_name] = "failed"
                elif result:
                    notification.delivery_status[ch_name] = "sent"
                    notification.delivered_to.append(ch_name)
                else:
                    notification.delivery_status[ch_name] = "failed"

        if notification.delivered_to:
            self._record_throttle(category, source or "")

        self._append_history(notification)
        return notification

    async def _adeliver_to_channel(self, notification: Notification, channel: str) -> bool:
        """Async wrapper around _deliver_to_channel using a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._deliver_to_channel, notification, channel
        )

    async def ainfo(self, title: str, message: str, **kwargs: Any) -> Notification:
        """Async info-level send."""
        return await self.asend(title, message, severity="info", **kwargs)

    async def awarning(self, title: str, message: str, **kwargs: Any) -> Notification:
        """Async warning-level send."""
        return await self.asend(title, message, severity="warning", **kwargs)

    async def acritical(self, title: str, message: str, **kwargs: Any) -> Notification:
        """Async critical-level send."""
        return await self.asend(title, message, severity="critical", **kwargs)

    async def asuccess(self, title: str, message: str, **kwargs: Any) -> Notification:
        """Async success-level send."""
        return await self.asend(title, message, severity="success", **kwargs)

    # ==================================================================
    # CHANNEL DELIVERY
    # ==================================================================

    def _deliver_to_channel(self, notification: Notification, channel: str) -> bool:
        """Dispatch delivery to the correct channel handler.

        Returns True on success, False on failure.  Exceptions are caught
        and logged rather than propagated.
        """
        try:
            if channel == "whatsapp":
                return self._send_whatsapp(notification)
            elif channel == "telegram":
                return self._send_telegram(notification)
            elif channel == "discord":
                return self._send_discord(notification)
            elif channel == "email":
                return self._send_email(notification)
            elif channel == "android":
                return self._send_android(notification)
            else:
                logger.warning("Unknown channel: %s", channel)
                return False
        except Exception as exc:
            logger.error("Delivery to %s failed: %s", channel, exc)
            return False

    # ------------------------------------------------------------------
    # WhatsApp (via OpenClaw Gateway)
    # ------------------------------------------------------------------

    def _send_whatsapp(self, notification: Notification) -> bool:
        """Send notification via WhatsApp through the OpenClaw gateway.

        POST http://localhost:18789/api/messages/send
        Body: {"to": "owner", "message": "<formatted text>"}
        """
        import urllib.request
        import urllib.error

        cfg = self._channels.get("whatsapp")
        if cfg is None:
            return False

        gateway_url = cfg.config.get("gateway_url", OPENCLAW_GATEWAY_URL)
        gateway_token = cfg.config.get("gateway_token", OPENCLAW_GATEWAY_TOKEN)
        url = f"{gateway_url.rstrip('/')}/api/messages/send"

        text = self._format_whatsapp(notification)
        payload = json.dumps({"to": "owner", "message": text}).encode("utf-8")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if gateway_token:
            headers["Authorization"] = f"Bearer {gateway_token}"

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.debug("WhatsApp delivery: HTTP %d", resp.status)
                return resp.status < 400
        except urllib.error.HTTPError as exc:
            logger.error("WhatsApp delivery failed: HTTP %d — %s", exc.code, exc.reason)
            return False
        except Exception as exc:
            logger.error("WhatsApp delivery error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Telegram (Bot API)
    # ------------------------------------------------------------------

    def _send_telegram(self, notification: Notification) -> bool:
        """Send notification via Telegram Bot API.

        POST https://api.telegram.org/bot{token}/sendMessage
        Body: {"chat_id": ..., "text": ..., "parse_mode": "Markdown"}
        """
        import urllib.request
        import urllib.error

        cfg = self._channels.get("telegram")
        if cfg is None:
            return False

        bot_token = cfg.config.get("bot_token", TELEGRAM_BOT_TOKEN)
        chat_id = cfg.config.get("chat_id", TELEGRAM_CHAT_ID)

        if not bot_token or not chat_id:
            logger.warning("Telegram not configured (missing bot_token or chat_id).")
            return False

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        text = self._format_telegram(notification)

        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if body.get("ok"):
                    logger.debug("Telegram delivery: success (message_id=%s)", body.get("result", {}).get("message_id"))
                    return True
                logger.error("Telegram API returned ok=false: %s", body)
                return False
        except urllib.error.HTTPError as exc:
            logger.error("Telegram delivery failed: HTTP %d", exc.code)
            return False
        except Exception as exc:
            logger.error("Telegram delivery error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Discord (Webhook)
    # ------------------------------------------------------------------

    def _send_discord(self, notification: Notification) -> bool:
        """Send notification via Discord webhook with rich embed.

        POST {webhook_url}
        Body: {"content": ..., "embeds": [...]}
        """
        import urllib.request
        import urllib.error

        cfg = self._channels.get("discord")
        if cfg is None:
            return False

        webhook_url = cfg.config.get("webhook_url", DISCORD_WEBHOOK_URL)
        if not webhook_url:
            logger.warning("Discord not configured (missing webhook_url).")
            return False

        embed = self._format_discord(notification)

        payload = json.dumps({
            "embeds": [embed],
        }).encode("utf-8")

        req = urllib.request.Request(
            webhook_url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.debug("Discord delivery: HTTP %d", resp.status)
                return resp.status < 300
        except urllib.error.HTTPError as exc:
            # Discord returns 204 No Content on success, which is fine.
            # 429 means rate-limited.
            if exc.code == 204:
                return True
            logger.error("Discord delivery failed: HTTP %d", exc.code)
            return False
        except Exception as exc:
            logger.error("Discord delivery error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Email (SMTP)
    # ------------------------------------------------------------------

    def _send_email(self, notification: Notification) -> bool:
        """Send notification via SMTP with an HTML email body.

        Uses stdlib smtplib with STARTTLS.
        """
        cfg = self._channels.get("email")
        if cfg is None:
            return False

        smtp_host = cfg.config.get("smtp_host", SMTP_HOST)
        smtp_port = int(cfg.config.get("smtp_port", SMTP_PORT))
        smtp_user = cfg.config.get("smtp_user", SMTP_USER)
        smtp_password = cfg.config.get("smtp_password", SMTP_PASSWORD)
        to_address = cfg.config.get("to_address", NOTIFICATION_EMAIL_TO)

        if not all([smtp_host, smtp_user, smtp_password, to_address]):
            logger.warning("Email not configured (missing SMTP credentials or recipient).")
            return False

        subject, html_body = self._format_email(notification)

        msg = email.mime.multipart.MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_address
        msg["X-Priority"] = "1" if notification.severity == "critical" else "3"

        # Plain-text fallback
        plain_text = self._format_whatsapp(notification)
        msg.attach(email.mime.text.MIMEText(plain_text, "plain", "utf-8"))
        msg.attach(email.mime.text.MIMEText(html_body, "html", "utf-8"))

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, [to_address], msg.as_string())
            logger.debug("Email delivery: success to %s", to_address)
            return True
        except smtplib.SMTPException as exc:
            logger.error("Email delivery failed: %s", exc)
            return False
        except Exception as exc:
            logger.error("Email delivery error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Android (via Termux:API through OpenClaw node)
    # ------------------------------------------------------------------

    def _send_android(self, notification: Notification) -> bool:
        """Send notification to Android phone via Termux:API.

        POST http://localhost:18789/api/nodes/invoke
        Body: {"node": "android", "command": "termux-notification", "args": {...}}

        For critical notifications, also sends vibrate and TTS commands.
        """
        import urllib.request
        import urllib.error

        cfg = self._channels.get("android")
        if cfg is None:
            return False

        gateway_url = cfg.config.get("gateway_url", OPENCLAW_GATEWAY_URL)
        gateway_token = cfg.config.get("gateway_token", OPENCLAW_GATEWAY_TOKEN)
        invoke_url = f"{gateway_url.rstrip('/')}/api/nodes/invoke"

        formatted = self._format_android(notification)

        # Build the termux-notification command arguments
        notification_cmd = (
            f'termux-notification '
            f'--title "{formatted["title"]}" '
            f'--content "{formatted["content"]}" '
            f'--priority {formatted["priority"]}'
        )

        payload = json.dumps({
            "node": "android",
            "command": notification_cmd,
        }).encode("utf-8")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if gateway_token:
            headers["Authorization"] = f"Bearer {gateway_token}"

        req = urllib.request.Request(invoke_url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.debug("Android notification: HTTP %d", resp.status)
                success = resp.status < 400
        except Exception as exc:
            logger.error("Android notification failed: %s", exc)
            return False

        # Critical notifications get extra attention: vibrate + TTS
        if notification.severity == "critical" and success:
            self._android_critical_extras(invoke_url, headers, notification)

        return success

    def _android_critical_extras(
        self, invoke_url: str, headers: dict[str, str], notification: Notification
    ) -> None:
        """Send vibrate and TTS commands for critical notifications."""
        import urllib.request

        # Vibrate for 1 second
        vibrate_payload = json.dumps({
            "node": "android",
            "command": "termux-vibrate -d 1000",
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                invoke_url, data=vibrate_payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            logger.warning("Android vibrate failed: %s", exc)

        # Text-to-speech announcement
        tts_text = f"Critical alert: {notification.title}. {notification.message[:100]}"
        tts_payload = json.dumps({
            "node": "android",
            "command": f'termux-tts-speak "{tts_text}"',
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                invoke_url, data=tts_payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            logger.warning("Android TTS failed: %s", exc)

    # ==================================================================
    # MESSAGE FORMATTING
    # ==================================================================

    def _format_whatsapp(self, notification: Notification) -> str:
        """Format notification as plain text for WhatsApp.

        Uses severity prefix indicators and clean layout.
        """
        prefix = SEVERITY_EMOJI.get(notification.severity, "[i]")
        lines: list[str] = []
        lines.append(f"{prefix} *{notification.title}*")
        lines.append("")
        lines.append(notification.message)

        if notification.site_id:
            lines.append("")
            lines.append(f"Site: {notification.site_id}")

        if notification.category and notification.category != "general":
            lines.append(f"Category: {notification.category}")

        if notification.source:
            lines.append(f"Source: {notification.source}")

        created = _parse_iso(notification.created_at)
        if created:
            eastern = created.astimezone(EASTERN)
            lines.append(f"Time: {eastern.strftime('%I:%M %p ET  %m/%d/%Y')}")

        return "\n".join(lines)

    def _format_telegram(self, notification: Notification) -> str:
        """Format notification as Telegram Markdown.

        Uses bold/italic for emphasis, severity indicators.
        """
        prefix = SEVERITY_EMOJI.get(notification.severity, "[i]")
        lines: list[str] = []
        lines.append(f"{prefix} *{notification.title}*")
        lines.append("")
        lines.append(notification.message)

        details: list[str] = []
        if notification.site_id:
            details.append(f"_Site:_ `{notification.site_id}`")
        if notification.category and notification.category != "general":
            details.append(f"_Category:_ `{notification.category}`")
        if notification.source:
            details.append(f"_Source:_ `{notification.source}`")

        created = _parse_iso(notification.created_at)
        if created:
            eastern = created.astimezone(EASTERN)
            details.append(f"_Time:_ `{eastern.strftime('%I:%M %p ET')}`")

        if details:
            lines.append("")
            lines.extend(details)

        return "\n".join(lines)

    def _format_discord(self, notification: Notification) -> dict:
        """Format notification as a Discord embed dict.

        Returns a dict suitable for inclusion in the "embeds" array of a
        Discord webhook payload.
        """
        color = DISCORD_COLORS.get(notification.severity, 0x3498DB)

        embed: dict[str, Any] = {
            "title": notification.title,
            "description": notification.message,
            "color": color,
            "timestamp": notification.created_at,
        }

        fields: list[dict] = []
        if notification.severity:
            fields.append({
                "name": "Severity",
                "value": notification.severity.upper(),
                "inline": True,
            })
        if notification.category and notification.category != "general":
            fields.append({
                "name": "Category",
                "value": notification.category,
                "inline": True,
            })
        if notification.site_id:
            fields.append({
                "name": "Site",
                "value": notification.site_id,
                "inline": True,
            })
        if notification.source:
            fields.append({
                "name": "Source",
                "value": notification.source,
                "inline": True,
            })

        # Add up to 3 data fields if present
        if notification.data:
            data_items = list(notification.data.items())[:3]
            for key, value in data_items:
                fields.append({
                    "name": key.replace("_", " ").title(),
                    "value": str(value)[:100],
                    "inline": True,
                })

        if fields:
            embed["fields"] = fields

        embed["footer"] = {
            "text": "OpenClaw Empire Notification Hub",
        }

        return embed

    def _format_email(self, notification: Notification) -> tuple[str, str]:
        """Format notification as an HTML email.

        Returns (subject, html_body).
        """
        severity_upper = notification.severity.upper()
        subject = f"[{severity_upper}] {notification.title}"

        bg_color = EMAIL_COLORS.get(notification.severity, "#3498DB")

        created = _parse_iso(notification.created_at)
        time_str = ""
        if created:
            eastern = created.astimezone(EASTERN)
            time_str = eastern.strftime("%I:%M %p ET on %B %d, %Y")

        # Build metadata rows
        meta_rows = ""
        if notification.site_id:
            meta_rows += f"<tr><td style='padding:4px 8px;font-weight:bold;'>Site</td><td style='padding:4px 8px;'>{notification.site_id}</td></tr>"
        if notification.category and notification.category != "general":
            meta_rows += f"<tr><td style='padding:4px 8px;font-weight:bold;'>Category</td><td style='padding:4px 8px;'>{notification.category}</td></tr>"
        if notification.source:
            meta_rows += f"<tr><td style='padding:4px 8px;font-weight:bold;'>Source</td><td style='padding:4px 8px;'>{notification.source}</td></tr>"
        if time_str:
            meta_rows += f"<tr><td style='padding:4px 8px;font-weight:bold;'>Time</td><td style='padding:4px 8px;'>{time_str}</td></tr>"

        # Build data rows
        data_rows = ""
        if notification.data:
            for key, value in list(notification.data.items())[:6]:
                label = key.replace("_", " ").title()
                data_rows += f"<tr><td style='padding:4px 8px;font-weight:bold;'>{label}</td><td style='padding:4px 8px;'>{value}</td></tr>"

        meta_section = ""
        if meta_rows:
            meta_section = f"""
            <table style="border-collapse:collapse;margin-top:16px;font-size:14px;">
                {meta_rows}
            </table>
            """

        data_section = ""
        if data_rows:
            data_section = f"""
            <h3 style="margin-top:20px;color:#333;">Details</h3>
            <table style="border-collapse:collapse;font-size:14px;background:#f9f9f9;border-radius:4px;">
                {data_rows}
            </table>
            """

        html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,Helvetica,sans-serif;margin:0;padding:0;background:#f4f4f4;">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;margin:20px auto;background:#ffffff;border-radius:8px;overflow:hidden;">
    <tr>
        <td style="background:{bg_color};padding:20px 24px;">
            <h1 style="margin:0;color:#ffffff;font-size:20px;">[{severity_upper}] {notification.title}</h1>
        </td>
    </tr>
    <tr>
        <td style="padding:24px;">
            <p style="font-size:16px;line-height:1.5;color:#333;margin:0;">{notification.message}</p>
            {meta_section}
            {data_section}
        </td>
    </tr>
    <tr>
        <td style="padding:16px 24px;background:#f8f8f8;font-size:12px;color:#999;text-align:center;">
            OpenClaw Empire Notification Hub &mdash; Automated alert
        </td>
    </tr>
</table>
</body>
</html>"""

        return subject, html_body

    def _format_android(self, notification: Notification) -> dict:
        """Format notification for Termux:API on Android.

        Returns dict with "title", "content", and "priority" fields.
        """
        prefix = SEVERITY_EMOJI.get(notification.severity, "[i]")

        content_parts: list[str] = [notification.message]
        if notification.site_id:
            content_parts.append(f"Site: {notification.site_id}")

        # Map severity to Android notification priority
        priority_map = {
            "info": "low",
            "success": "default",
            "warning": "high",
            "critical": "max",
        }

        return {
            "title": f"{prefix} {notification.title}",
            "content": " | ".join(content_parts),
            "priority": priority_map.get(notification.severity, "default"),
        }

    # ==================================================================
    # TEMPLATE NOTIFICATIONS (pre-formatted for common events)
    # ==================================================================

    def notify_revenue_alert(
        self,
        alert_type: str,
        amount: float,
        site_id: str | None = None,
        details: str | None = None,
    ) -> Notification:
        """Send a revenue-related alert.

        alert_type examples: "drop", "spike", "milestone", "zero_revenue"
        """
        severity = "warning"
        if alert_type in ("spike", "milestone"):
            severity = "info"
        elif alert_type in ("zero_revenue", "stream_zero"):
            severity = "critical"

        title = f"Revenue Alert: {alert_type.replace('_', ' ').title()}"
        message = f"Amount: ${amount:,.2f}"
        if details:
            message += f"\n{details}"

        return self.send(
            title=title,
            message=message,
            severity=severity,
            category="revenue",
            source="revenue_tracker",
            site_id=site_id,
            data={"alert_type": alert_type, "amount": amount},
        )

    def notify_site_down(self, site_id: str, domain: str, error: str) -> Notification:
        """Send a critical alert when a site is unreachable."""
        return self.send(
            title=f"SITE DOWN: {domain}",
            message=f"Site {domain} ({site_id}) is unreachable.\nError: {error}",
            severity="critical",
            category="health",
            source="health_monitor",
            site_id=site_id,
            data={"domain": domain, "error": error},
        )

    def notify_site_recovered(
        self, site_id: str, domain: str, downtime_minutes: int
    ) -> Notification:
        """Send a success notification when a site recovers from downtime."""
        return self.send(
            title=f"Site Recovered: {domain}",
            message=f"Site {domain} ({site_id}) is back online after {downtime_minutes} minutes of downtime.",
            severity="success",
            category="health",
            source="health_monitor",
            site_id=site_id,
            data={"domain": domain, "downtime_minutes": downtime_minutes},
        )

    def notify_content_published(
        self, site_id: str, title: str, url: str, wp_post_id: int
    ) -> Notification:
        """Send a success notification when content is published."""
        return self.send(
            title=f"Published: {title}",
            message=f"New article published on {site_id}.\nURL: {url}\nPost ID: {wp_post_id}",
            severity="success",
            category="content",
            source="content_generator",
            site_id=site_id,
            data={"url": url, "wp_post_id": wp_post_id, "article_title": title},
        )

    def notify_seo_issues(
        self, site_id: str, critical_count: int, warning_count: int
    ) -> Notification:
        """Send notification about SEO issues found during an audit."""
        severity = "critical" if critical_count > 0 else "warning"
        total = critical_count + warning_count
        return self.send(
            title=f"SEO Issues: {site_id} ({total} total)",
            message=(
                f"SEO audit found {critical_count} critical and {warning_count} "
                f"warning issues on {site_id}."
            ),
            severity=severity,
            category="seo",
            source="seo_auditor",
            site_id=site_id,
            data={"critical_count": critical_count, "warning_count": warning_count},
        )

    def notify_task_completed(self, task_name: str, result_summary: str) -> Notification:
        """Send info notification when a scheduled task completes."""
        return self.send(
            title=f"Task Completed: {task_name}",
            message=result_summary,
            severity="info",
            category="scheduler",
            source="task_scheduler",
            data={"task_name": task_name},
        )

    def notify_task_failed(self, task_name: str, error: str) -> Notification:
        """Send warning notification when a scheduled task fails."""
        return self.send(
            title=f"Task Failed: {task_name}",
            message=f"Scheduled task '{task_name}' failed.\nError: {error}",
            severity="warning",
            category="scheduler",
            source="task_scheduler",
            data={"task_name": task_name, "error": error},
        )

    def notify_security_event(self, event_type: str, details: str) -> Notification:
        """Send critical notification for security events."""
        return self.send(
            title=f"Security: {event_type.replace('_', ' ').title()}",
            message=details,
            severity="critical",
            category="security",
            source="security_monitor",
            data={"event_type": event_type},
        )

    def notify_milestone(
        self, milestone_type: str, value: Any, details: str | None = None
    ) -> Notification:
        """Send success notification for milestones achieved."""
        message = f"Milestone reached: {milestone_type} = {value}"
        if details:
            message += f"\n{details}"
        return self.send(
            title=f"Milestone: {milestone_type.replace('_', ' ').title()}",
            message=message,
            severity="success",
            category="general",
            source="milestone_tracker",
            data={"milestone_type": milestone_type, "value": value},
        )

    def notify_scheduler_failure(self, job_name: str, error: str) -> Notification:
        """Send warning notification for scheduler infrastructure failures."""
        return self.send(
            title=f"Scheduler Failure: {job_name}",
            message=f"The scheduler encountered an error running '{job_name}'.\nError: {error}",
            severity="warning",
            category="scheduler",
            source="task_scheduler",
            data={"job_name": job_name, "error": error},
        )

    # ==================================================================
    # CHANNEL MANAGEMENT
    # ==================================================================

    def configure_channel(
        self,
        channel: str,
        enabled: bool | None = None,
        config: dict | None = None,
        min_severity: str | None = None,
        categories: list[str] | None = None,
        quiet_hours: tuple[int, int] | None = ...,  # type: ignore[assignment]
    ) -> ChannelConfig:
        """Configure or update a delivery channel.

        Pass only the fields you want to change; others are preserved.
        To clear quiet_hours, pass quiet_hours=None explicitly.
        The sentinel default (...) means "do not change".

        Returns the updated ChannelConfig.
        """
        cfg = self._channels.get(channel)
        if cfg is None:
            cfg = ChannelConfig(channel=channel)
            self._channels[channel] = cfg

        if enabled is not None:
            cfg.enabled = enabled
        if config is not None:
            cfg.config.update(config)
        if min_severity is not None:
            cfg.min_severity = min_severity
        if categories is not None:
            cfg.categories = categories
        if quiet_hours is not ...:  # type: ignore[comparison-overlap]
            cfg.quiet_hours = quiet_hours  # type: ignore[assignment]

        self._save_channels()
        logger.info("Channel %r configured: enabled=%s, min_severity=%s",
                     channel, cfg.enabled, cfg.min_severity)
        return cfg

    def get_channel_config(self, channel: str) -> ChannelConfig | None:
        """Get the configuration for a specific channel."""
        return self._channels.get(channel)

    def list_channels(self) -> dict[str, ChannelConfig]:
        """Return all channel configurations."""
        return dict(self._channels)

    def test_channel(self, channel: str) -> bool:
        """Send a test notification to a specific channel.

        Returns True if the test was delivered successfully.
        """
        test_notification = Notification(
            title="Test Notification",
            message=(
                "This is a test notification from the OpenClaw Empire "
                "Notification Hub. If you see this, the channel is working."
            ),
            severity="info",
            category="general",
            source="notification_hub_test",
        )

        logger.info("Sending test notification to channel: %s", channel)
        return self._deliver_to_channel(test_notification, channel)

    # ==================================================================
    # RULES MANAGEMENT
    # ==================================================================

    def add_rule(
        self,
        category: str,
        min_severity: str,
        channels: list[str],
        throttle_minutes: int = 0,
        template: str | None = None,
    ) -> NotificationRule:
        """Add a new notification routing rule.

        Returns the created NotificationRule.
        """
        rule = NotificationRule(
            category=category,
            min_severity=min_severity,
            channels=channels,
            throttle_minutes=throttle_minutes,
            template=template,
        )
        self._rules.append(rule)
        self._save_rules()
        logger.info(
            "Added rule %s: %s/%s -> %s (throttle=%dm)",
            rule.rule_id[:8], category, min_severity,
            ",".join(channels), throttle_minutes,
        )
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by its ID. Returns True if the rule was found and removed."""
        for i, rule in enumerate(self._rules):
            if rule.rule_id == rule_id:
                self._rules.pop(i)
                self._save_rules()
                logger.info("Removed rule: %s", rule_id[:8])
                return True
        return False

    def get_rules(self) -> list[NotificationRule]:
        """Return all notification rules."""
        return list(self._rules)

    def reset_rules_to_defaults(self) -> list[NotificationRule]:
        """Reset all rules to the empire defaults."""
        self._rules = [NotificationRule.from_dict(r) for r in DEFAULT_RULES]
        self._save_rules()
        logger.info("Rules reset to defaults (%d rules).", len(self._rules))
        return self._rules

    # ==================================================================
    # HISTORY
    # ==================================================================

    def get_history(
        self,
        limit: int = 50,
        category: str | None = None,
        severity: str | None = None,
    ) -> list[Notification]:
        """Retrieve notification history, newest first.

        Args:
            limit:    Maximum entries to return
            category: Filter by category
            severity: Filter by severity

        Returns:
            List of Notification objects.
        """
        entries = list(self._history)

        if category is not None:
            entries = [e for e in entries if e.get("category") == category]
        if severity is not None:
            entries = [e for e in entries if e.get("severity") == severity]

        entries = list(reversed(entries))[:limit]
        return [Notification.from_dict(e) for e in entries]

    def get_unread(self, limit: int = 20) -> list[Notification]:
        """Retrieve unread notifications, newest first."""
        entries = [e for e in self._history if not e.get("read", False)]
        entries = list(reversed(entries))[:limit]
        return [Notification.from_dict(e) for e in entries]

    def mark_read(self, notification_id: str) -> bool:
        """Mark a notification as read by its ID.

        Returns True if the notification was found and marked.
        """
        for entry in self._history:
            if entry.get("id") == notification_id:
                entry["read"] = True
                self._save_history()
                return True
        return False

    def mark_all_read(self) -> int:
        """Mark all notifications as read. Returns count of newly marked items."""
        count = 0
        for entry in self._history:
            if not entry.get("read", False):
                entry["read"] = True
                count += 1
        if count > 0:
            self._save_history()
        return count

    def get_delivery_stats(self, days: int = 7) -> dict:
        """Get delivery statistics for the last N days.

        Returns dict with total counts, per-channel stats, per-severity
        breakdown, and success/failure rates.
        """
        cutoff = (_now_utc() - timedelta(days=days)).isoformat()
        recent = [
            e for e in self._history
            if e.get("created_at", "") >= cutoff
        ]

        total = len(recent)
        by_severity: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_channel: dict[str, dict[str, int]] = {}
        delivered_count = 0
        failed_count = 0

        for entry in recent:
            sev = entry.get("severity", "info")
            by_severity[sev] = by_severity.get(sev, 0) + 1

            cat = entry.get("category", "general")
            by_category[cat] = by_category.get(cat, 0) + 1

            status = entry.get("delivery_status", {})
            for ch_name, ch_status in status.items():
                if ch_name not in by_channel:
                    by_channel[ch_name] = {"sent": 0, "failed": 0, "skipped": 0}
                by_channel[ch_name][ch_status] = by_channel[ch_name].get(ch_status, 0) + 1

                if ch_status == "sent":
                    delivered_count += 1
                elif ch_status == "failed":
                    failed_count += 1

        total_attempts = delivered_count + failed_count
        success_rate = (
            round(delivered_count / total_attempts * 100, 1)
            if total_attempts > 0
            else 100.0
        )

        return {
            "period_days": days,
            "total_notifications": total,
            "total_deliveries": delivered_count,
            "total_failures": failed_count,
            "success_rate": success_rate,
            "by_severity": by_severity,
            "by_category": by_category,
            "by_channel": by_channel,
        }

    # ==================================================================
    # DIGEST MODE
    # ==================================================================

    def send_daily_digest(self) -> Notification:
        """Compile and send a daily digest of the last 24 hours.

        Collects all info/warning notifications since the last digest
        and sends a single summary instead of individual alerts.  Critical
        notifications are always sent immediately and are excluded from
        the digest but referenced in the summary.
        """
        cutoff = (_now_utc() - timedelta(hours=24)).isoformat()
        last_digest = self._digest_state.get("last_daily_digest")

        # Use whichever is more recent: 24h ago or last digest
        effective_cutoff = cutoff
        if last_digest and last_digest > cutoff:
            effective_cutoff = last_digest

        recent = [
            e for e in self._history
            if e.get("created_at", "") >= effective_cutoff
        ]

        if not recent:
            logger.info("Daily digest: no notifications to summarize.")
            return self.send(
                title="Daily Digest",
                message="No notifications in the last 24 hours. All quiet.",
                severity="info",
                category="general",
                source="notification_hub",
            )

        # Categorize entries
        by_severity: dict[str, list[dict]] = {}
        by_category: dict[str, list[dict]] = {}
        for entry in recent:
            sev = entry.get("severity", "info")
            by_severity.setdefault(sev, []).append(entry)
            cat = entry.get("category", "general")
            by_category.setdefault(cat, []).append(entry)

        critical_count = len(by_severity.get("critical", []))
        warning_count = len(by_severity.get("warning", []))
        info_count = len(by_severity.get("info", []))
        success_count = len(by_severity.get("success", []))

        # Build digest message
        lines: list[str] = []
        lines.append(f"DAILY DIGEST — {_now_eastern().strftime('%B %d, %Y')}")
        lines.append(f"{'=' * 40}")
        lines.append(f"Total notifications: {len(recent)}")
        lines.append(f"  Critical: {critical_count}")
        lines.append(f"  Warning:  {warning_count}")
        lines.append(f"  Info:     {info_count}")
        lines.append(f"  Success:  {success_count}")
        lines.append("")

        # Highlight critical items
        if critical_count > 0:
            lines.append("CRITICAL ALERTS:")
            for entry in by_severity["critical"]:
                lines.append(f"  [!!!] {entry.get('title', 'Untitled')}")
                if entry.get("site_id"):
                    lines.append(f"        Site: {entry['site_id']}")
            lines.append("")

        # Summarize by category
        if by_category:
            lines.append("BY CATEGORY:")
            for cat, entries in sorted(by_category.items()):
                lines.append(f"  {cat}: {len(entries)} notification(s)")
            lines.append("")

        # Recent warnings (up to 5)
        warnings = by_severity.get("warning", [])
        if warnings:
            lines.append("RECENT WARNINGS:")
            for entry in warnings[-5:]:
                lines.append(f"  [!] {entry.get('title', 'Untitled')}")
            if len(warnings) > 5:
                lines.append(f"  ... and {len(warnings) - 5} more")

        # Update digest state
        self._digest_state["last_daily_digest"] = _now_iso()
        self._digest_state["digest_notification_ids"] = [
            e.get("id", "") for e in recent
        ]
        self._save_digest_state()

        severity = "warning" if critical_count > 0 else "info"
        return self.send(
            title=f"Daily Digest: {len(recent)} notifications",
            message="\n".join(lines),
            severity=severity,
            category="general",
            source="notification_hub",
        )

    def send_weekly_summary(self) -> Notification:
        """Compile and send a weekly summary of all notifications.

        Provides high-level statistics and highlights from the past 7 days.
        """
        stats = self.get_delivery_stats(days=7)
        cutoff = (_now_utc() - timedelta(days=7)).isoformat()

        recent = [
            e for e in self._history
            if e.get("created_at", "") >= cutoff
        ]

        lines: list[str] = []
        lines.append(f"WEEKLY NOTIFICATION SUMMARY")
        lines.append(f"{'=' * 40}")
        lines.append(f"Total notifications: {stats['total_notifications']}")
        lines.append(f"Deliveries: {stats['total_deliveries']} successful, {stats['total_failures']} failed")
        lines.append(f"Success rate: {stats['success_rate']}%")
        lines.append("")

        # Severity breakdown
        lines.append("BY SEVERITY:")
        for sev in ["critical", "warning", "info", "success"]:
            count = stats["by_severity"].get(sev, 0)
            if count > 0:
                lines.append(f"  {sev}: {count}")
        lines.append("")

        # Category breakdown
        if stats["by_category"]:
            lines.append("BY CATEGORY:")
            for cat, count in sorted(stats["by_category"].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {cat}: {count}")
            lines.append("")

        # Channel performance
        if stats["by_channel"]:
            lines.append("CHANNEL PERFORMANCE:")
            for ch_name, ch_stats in stats["by_channel"].items():
                sent = ch_stats.get("sent", 0)
                failed = ch_stats.get("failed", 0)
                total_ch = sent + failed
                rate = round(sent / total_ch * 100, 1) if total_ch > 0 else 100.0
                lines.append(f"  {ch_name}: {sent}/{total_ch} delivered ({rate}%)")

        # Update state
        self._digest_state["last_weekly_summary"] = _now_iso()
        self._save_digest_state()

        return self.send(
            title=f"Weekly Summary: {stats['total_notifications']} notifications",
            message="\n".join(lines),
            severity="info",
            category="general",
            source="notification_hub",
        )

    # ==================================================================
    # CLEAR / MAINTENANCE
    # ==================================================================

    def clear_history(self) -> int:
        """Clear all notification history. Returns count of cleared entries."""
        count = len(self._history)
        self._history = []
        self._save_history()
        logger.info("Cleared %d history entries.", count)
        return count

    def clear_throttles(self) -> None:
        """Clear all throttle trackers, allowing immediate resend."""
        self._throttle_tracker.clear()
        logger.info("Throttle state cleared.")


# ===================================================================
# SINGLETON
# ===================================================================

_hub_instance: NotificationHub | None = None


def get_hub() -> NotificationHub:
    """Get the global NotificationHub singleton.

    Creates the instance on first call, loading persisted state from disk.
    """
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = NotificationHub()
    return _hub_instance


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def notify(
    title: str,
    message: str,
    severity: str = "info",
    category: str = "general",
    **kwargs: Any,
) -> Notification:
    """Convenience: send a notification via the singleton hub."""
    return get_hub().send(title, message, severity=severity, category=category, **kwargs)


def notify_critical(title: str, message: str, **kwargs: Any) -> Notification:
    """Convenience: send a critical notification."""
    return get_hub().critical(title, message, **kwargs)


def notify_site_down(site_id: str, domain: str, error: str) -> Notification:
    """Convenience: send a site-down alert."""
    return get_hub().notify_site_down(site_id, domain, error)


def notify_content_published(
    site_id: str, title: str, url: str, wp_post_id: int
) -> Notification:
    """Convenience: send a content-published notification."""
    return get_hub().notify_content_published(site_id, title, url, wp_post_id)


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def _format_table(headers: list[str], rows: list[list[str]], max_col_width: int = 50) -> str:
    """Format a simple ASCII table for CLI output."""
    if not rows:
        return "(no results)"

    truncated_rows = []
    for row in rows:
        truncated_rows.append([
            val[:max_col_width - 3] + "..." if len(val) > max_col_width else val
            for val in row
        ])

    col_widths = [len(h) for h in headers]
    for row in truncated_rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    lines = [fmt.format(*headers)]
    lines.append("  ".join("-" * w for w in col_widths))
    for row in truncated_rows:
        while len(row) < len(headers):
            row.append("")
        lines.append(fmt.format(*row))

    return "\n".join(lines)


def _cmd_send(args: argparse.Namespace) -> None:
    """Send a notification from the CLI."""
    hub = get_hub()
    notification = hub.send(
        title=args.title,
        message=args.message,
        severity=args.severity,
        category=args.category or "general",
        source="cli",
    )
    print(f"Notification sent: {notification.id[:8]}")
    print(f"  Title:    {notification.title}")
    print(f"  Severity: {notification.severity}")
    if notification.delivered_to:
        print(f"  Delivered to: {', '.join(notification.delivered_to)}")
    else:
        print(f"  Delivered to: (none)")
    for ch, status in notification.delivery_status.items():
        print(f"    {ch}: {status}")


def _cmd_test(args: argparse.Namespace) -> None:
    """Test a specific channel."""
    hub = get_hub()
    channel = args.channel
    print(f"Testing channel: {channel}...")
    success = hub.test_channel(channel)
    if success:
        print(f"  Channel '{channel}' is working.")
    else:
        print(f"  Channel '{channel}' FAILED. Check configuration and logs.")


def _cmd_history(args: argparse.Namespace) -> None:
    """Show notification history."""
    hub = get_hub()
    history = hub.get_history(
        limit=args.limit,
        category=args.category,
        severity=args.severity,
    )

    if not history:
        print("No notifications found.")
        return

    headers = ["ID", "Time", "Severity", "Category", "Title", "Channels", "Read"]
    rows = []
    for n in history:
        created = _parse_iso(n.created_at)
        time_str = ""
        if created:
            eastern = created.astimezone(EASTERN)
            time_str = eastern.strftime("%m/%d %I:%M %p")

        channels_str = ", ".join(n.delivered_to) if n.delivered_to else "(none)"
        rows.append([
            n.id[:8] + "...",
            time_str,
            n.severity,
            n.category,
            n.title[:40],
            channels_str,
            "Y" if n.read else "N",
        ])

    print(f"\n  Notification History  --  {len(history)} entries\n")
    print(_format_table(headers, rows))
    print()


def _cmd_unread(args: argparse.Namespace) -> None:
    """Show unread notifications."""
    hub = get_hub()
    unread = hub.get_unread(limit=args.limit)

    if not unread:
        print("No unread notifications.")
        return

    headers = ["ID", "Time", "Severity", "Title"]
    rows = []
    for n in unread:
        created = _parse_iso(n.created_at)
        time_str = ""
        if created:
            eastern = created.astimezone(EASTERN)
            time_str = eastern.strftime("%m/%d %I:%M %p")

        rows.append([
            n.id[:8] + "...",
            time_str,
            n.severity,
            n.title[:50],
        ])

    print(f"\n  Unread Notifications  --  {len(unread)} entries\n")
    print(_format_table(headers, rows))
    print()


def _cmd_channels(args: argparse.Namespace) -> None:
    """List channel configurations."""
    hub = get_hub()
    channels = hub.list_channels()

    if not channels:
        print("No channels configured.")
        return

    headers = ["Channel", "Enabled", "Min Severity", "Categories", "Quiet Hours"]
    rows = []
    for name, cfg in sorted(channels.items()):
        qh = ""
        if cfg.quiet_hours:
            qh = f"{cfg.quiet_hours[0]:02d}:00-{cfg.quiet_hours[1]:02d}:00 ET"

        cats = ", ".join(cfg.categories) if cfg.categories else "all"
        rows.append([
            name,
            "Yes" if cfg.enabled else "No",
            cfg.min_severity,
            cats,
            qh or "(none)",
        ])

    print(f"\n  Notification Channels  --  {len(channels)} configured\n")
    print(_format_table(headers, rows))
    print()


def _cmd_digest(args: argparse.Namespace) -> None:
    """Send daily digest now."""
    hub = get_hub()
    print("Generating daily digest...")
    notification = hub.send_daily_digest()
    print(f"Digest sent: {notification.id[:8]}")
    if notification.delivered_to:
        print(f"  Delivered to: {', '.join(notification.delivered_to)}")
    print(f"\n{notification.message}")


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show delivery statistics."""
    hub = get_hub()
    stats = hub.get_delivery_stats(days=args.days)

    print(f"\n  Delivery Statistics (last {stats['period_days']} days)\n")
    print(f"  {'=' * 45}")
    print(f"  Total notifications:  {stats['total_notifications']}")
    print(f"  Total deliveries:     {stats['total_deliveries']}")
    print(f"  Total failures:       {stats['total_failures']}")
    print(f"  Success rate:         {stats['success_rate']}%")

    if stats["by_severity"]:
        print(f"\n  By Severity:")
        for sev, count in sorted(stats["by_severity"].items()):
            print(f"    {sev:<12} {count}")

    if stats["by_category"]:
        print(f"\n  By Category:")
        for cat, count in sorted(stats["by_category"].items(), key=lambda x: x[1], reverse=True):
            print(f"    {cat:<15} {count}")

    if stats["by_channel"]:
        print(f"\n  By Channel:")
        for ch_name, ch_stats in sorted(stats["by_channel"].items()):
            sent = ch_stats.get("sent", 0)
            failed = ch_stats.get("failed", 0)
            total_ch = sent + failed
            rate = round(sent / total_ch * 100, 1) if total_ch > 0 else 100.0
            print(f"    {ch_name:<12} {sent} sent / {failed} failed ({rate}%)")

    print()


def _cmd_rules(args: argparse.Namespace) -> None:
    """List notification rules."""
    hub = get_hub()
    rules = hub.get_rules()

    if not rules:
        print("No notification rules configured.")
        return

    headers = ["Rule ID", "Category", "Min Severity", "Channels", "Throttle"]
    rows = []
    for rule in rules:
        throttle_str = f"{rule.throttle_minutes}m" if rule.throttle_minutes > 0 else "none"
        rows.append([
            rule.rule_id[:12] + "..." if len(rule.rule_id) > 12 else rule.rule_id,
            rule.category,
            rule.min_severity,
            ", ".join(rule.channels),
            throttle_str,
        ])

    print(f"\n  Notification Rules  --  {len(rules)} rules\n")
    print(_format_table(headers, rows))
    print()


def _cmd_weekly(args: argparse.Namespace) -> None:
    """Send weekly summary now."""
    hub = get_hub()
    print("Generating weekly summary...")
    notification = hub.send_weekly_summary()
    print(f"Summary sent: {notification.id[:8]}")
    if notification.delivered_to:
        print(f"  Delivered to: {', '.join(notification.delivered_to)}")
    print(f"\n{notification.message}")


def _cmd_mark_read(args: argparse.Namespace) -> None:
    """Mark notification(s) as read."""
    hub = get_hub()
    if args.all:
        count = hub.mark_all_read()
        print(f"Marked {count} notifications as read.")
    elif args.notification_id:
        found = hub.mark_read(args.notification_id)
        if found:
            print(f"Marked {args.notification_id[:8]}... as read.")
        else:
            # Try partial match
            for entry in hub._history:
                if entry.get("id", "").startswith(args.notification_id):
                    hub.mark_read(entry["id"])
                    print(f"Marked {entry['id'][:8]}... as read.")
                    return
            print(f"Notification not found: {args.notification_id}")
    else:
        print("Specify --all or a notification ID.")


def main() -> None:
    """CLI entry point for the notification hub."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="notification_hub",
        description="OpenClaw Empire Notification Hub — Unified Alert System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # send
    sp_send = subparsers.add_parser("send", help="Send a notification")
    sp_send.add_argument("--title", required=True, help="Notification title")
    sp_send.add_argument("--message", required=True, help="Notification message body")
    sp_send.add_argument(
        "--severity", choices=["info", "success", "warning", "critical"],
        default="info", help="Severity level (default: info)",
    )
    sp_send.add_argument(
        "--category", default="general",
        help="Category (revenue/content/seo/health/security/scheduler/general)",
    )
    sp_send.set_defaults(func=_cmd_send)

    # test
    sp_test = subparsers.add_parser("test", help="Send test notification to a channel")
    sp_test.add_argument(
        "--channel", required=True,
        choices=["whatsapp", "telegram", "discord", "email", "android"],
        help="Channel to test",
    )
    sp_test.set_defaults(func=_cmd_test)

    # history
    sp_history = subparsers.add_parser("history", help="Show notification history")
    sp_history.add_argument("--limit", type=int, default=20, help="Max entries (default: 20)")
    sp_history.add_argument("--category", default=None, help="Filter by category")
    sp_history.add_argument("--severity", default=None, help="Filter by severity")
    sp_history.set_defaults(func=_cmd_history)

    # unread
    sp_unread = subparsers.add_parser("unread", help="Show unread notifications")
    sp_unread.add_argument("--limit", type=int, default=20, help="Max entries (default: 20)")
    sp_unread.set_defaults(func=_cmd_unread)

    # channels
    sp_channels = subparsers.add_parser("channels", help="List channel configurations")
    sp_channels.set_defaults(func=_cmd_channels)

    # digest
    sp_digest = subparsers.add_parser("digest", help="Send daily digest now")
    sp_digest.set_defaults(func=_cmd_digest)

    # weekly
    sp_weekly = subparsers.add_parser("weekly", help="Send weekly summary now")
    sp_weekly.set_defaults(func=_cmd_weekly)

    # stats
    sp_stats = subparsers.add_parser("stats", help="Show delivery statistics")
    sp_stats.add_argument("--days", type=int, default=7, help="Lookback period in days (default: 7)")
    sp_stats.set_defaults(func=_cmd_stats)

    # rules
    sp_rules = subparsers.add_parser("rules", help="List notification rules")
    sp_rules.set_defaults(func=_cmd_rules)

    # mark-read
    sp_mark = subparsers.add_parser("mark-read", help="Mark notifications as read")
    sp_mark.add_argument("--all", action="store_true", help="Mark all as read")
    sp_mark.add_argument("notification_id", nargs="?", default=None, help="Notification ID (or prefix)")
    sp_mark.set_defaults(func=_cmd_mark_read)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


# ===================================================================
# MODULE ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    main()
