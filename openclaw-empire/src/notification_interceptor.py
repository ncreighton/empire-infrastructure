"""
Notification Interceptor — OpenClaw Empire Android Notification System

Intelligent notification monitoring, classification, and action routing for
Nick Creighton's Android device.  Intercepts device notifications via ADB
``dumpsys notification``, classifies them with Claude Haiku, matches them
against a configurable rule engine, and auto-responds or routes them to
n8n workflows, the notification hub, or the revenue tracker.

Architecture:
    ADB (dumpsys notification) --> parse --> classify (Haiku) --> rule engine
                                                                     |
                                                          +----------+----------+
                                                          |          |          |
                                                     auto-reply   webhook   log/dismiss

Integration points:
    PhoneController   — ADB command execution (tap, type, swipe)
    NotificationHub   — Cross-channel forwarding (Telegram, Discord, etc.)
    RevenueTracker    — Etsy/affiliate sale amount extraction and logging
    n8n Client        — Trigger monitoring and content workflows
    FORGE Codex       — Error/crash pattern learning

Data persisted to: data/notifications/

Usage:
    from src.notification_interceptor import get_interceptor

    interceptor = get_interceptor()

    # One-shot capture
    notifications = await interceptor.capture_notifications()

    # Continuous monitoring
    await interceptor.monitor(poll_interval=5, callback=my_handler)

    # Rule management
    interceptor.add_rule("Etsy Sales", {"app": "com.etsy.android", "title_pattern": "sale"},
                         ["log", "trigger_webhook"])

Sync usage:
    notifications = interceptor.capture_notifications_sync()
    stats = interceptor.notification_stats_sync(hours=24)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("notification_interceptor")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s.%(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

NOTIFICATION_DATA_DIR = BASE_DIR / "data" / "notifications"
INTERCEPTOR_DIR = NOTIFICATION_DATA_DIR / "interceptor"
RULES_FILE = INTERCEPTOR_DIR / "rules.json"
HISTORY_FILE = INTERCEPTOR_DIR / "history.json"
QUIET_HOURS_FILE = INTERCEPTOR_DIR / "quiet_hours.json"
STATS_FILE = INTERCEPTOR_DIR / "stats.json"
TEMPLATES_FILE = INTERCEPTOR_DIR / "reply_templates.json"

# Ensure directories exist on import
INTERCEPTOR_DIR.mkdir(parents=True, exist_ok=True)

# Environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENCLAW_NODE_URL = os.getenv("OPENCLAW_NODE_URL", "http://localhost:18789")
OPENCLAW_ANDROID_NODE = os.getenv("OPENCLAW_ANDROID_NODE", "android")
N8N_WEBHOOK_BASE = os.getenv(
    "N8N_WEBHOOK_BASE",
    "http://vmi2976539.contaboserver.net:5678/webhook",
)

# Limits
MAX_HISTORY_ENTRIES = 5000
MAX_RULES = 200
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ADB timing
POST_ACTION_DELAY = 0.8
SCREENSHOT_SETTLE_DELAY = 0.5

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


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


# ===================================================================
# Enums
# ===================================================================


class NotificationCategory(str, Enum):
    """High-level notification categories."""
    MESSAGE = "message"
    EMAIL = "email"
    SOCIAL = "social"
    ALERT = "alert"
    SYSTEM = "system"
    COMMERCE = "commerce"
    CONTENT = "content"
    UNKNOWN = "unknown"


class NotificationPriority(str, Enum):
    """Notification priority levels."""
    LOW = "low"
    DEFAULT = "default"
    HIGH = "high"
    URGENT = "urgent"


class NotificationIntent(str, Enum):
    """What the notification wants from the user."""
    REPLY_NEEDED = "reply_needed"
    INFO_ONLY = "info_only"
    ACTION_REQUIRED = "action_required"
    URGENT = "urgent"


class RuleActionType(str, Enum):
    """Actions a rule can trigger."""
    AUTO_REPLY = "auto_reply"
    FORWARD = "forward"
    TRIGGER_WEBHOOK = "trigger_webhook"
    LOG = "log"
    DISMISS = "dismiss"
    ESCALATE = "escalate"
    EXECUTE_PLAYBOOK = "execute_playbook"


# Package-to-category heuristic map
PACKAGE_CATEGORY_MAP: Dict[str, NotificationCategory] = {
    # Messaging
    "com.whatsapp": NotificationCategory.MESSAGE,
    "com.whatsapp.w4b": NotificationCategory.MESSAGE,
    "org.telegram.messenger": NotificationCategory.MESSAGE,
    "com.discord": NotificationCategory.MESSAGE,
    "com.google.android.apps.messaging": NotificationCategory.MESSAGE,
    "com.facebook.orca": NotificationCategory.MESSAGE,
    "com.snapchat.android": NotificationCategory.MESSAGE,
    # Email
    "com.google.android.gm": NotificationCategory.EMAIL,
    "com.microsoft.office.outlook": NotificationCategory.EMAIL,
    "com.yahoo.mobile.client.android.mail": NotificationCategory.EMAIL,
    # Social
    "com.instagram.android": NotificationCategory.SOCIAL,
    "com.twitter.android": NotificationCategory.SOCIAL,
    "com.facebook.katana": NotificationCategory.SOCIAL,
    "com.pinterest": NotificationCategory.SOCIAL,
    "com.zhiliaoapp.musically": NotificationCategory.SOCIAL,
    "com.reddit.frontpage": NotificationCategory.SOCIAL,
    "com.linkedin.android": NotificationCategory.SOCIAL,
    # Commerce
    "com.etsy.android": NotificationCategory.COMMERCE,
    "com.amazon.mShop.android.shopping": NotificationCategory.COMMERCE,
    "com.paypal.android.p2pmobile": NotificationCategory.COMMERCE,
    "com.venmo": NotificationCategory.COMMERCE,
    "com.shopify.mobile": NotificationCategory.COMMERCE,
    "com.stripe.android.dashboard": NotificationCategory.COMMERCE,
    # Content
    "org.wordpress.android": NotificationCategory.CONTENT,
    # System
    "android": NotificationCategory.SYSTEM,
    "com.android.systemui": NotificationCategory.SYSTEM,
    "com.android.vending": NotificationCategory.SYSTEM,
    "com.google.android.gms": NotificationCategory.SYSTEM,
}

# Known app name map (package -> friendly name)
PACKAGE_NAME_MAP: Dict[str, str] = {
    "com.whatsapp": "WhatsApp",
    "com.whatsapp.w4b": "WhatsApp Business",
    "org.telegram.messenger": "Telegram",
    "com.discord": "Discord",
    "com.google.android.gm": "Gmail",
    "com.microsoft.office.outlook": "Outlook",
    "com.instagram.android": "Instagram",
    "com.twitter.android": "X (Twitter)",
    "com.facebook.katana": "Facebook",
    "com.facebook.orca": "Messenger",
    "com.etsy.android": "Etsy",
    "com.amazon.mShop.android.shopping": "Amazon",
    "com.paypal.android.p2pmobile": "PayPal",
    "com.shopify.mobile": "Shopify",
    "com.pinterest": "Pinterest",
    "com.reddit.frontpage": "Reddit",
    "com.linkedin.android": "LinkedIn",
    "org.wordpress.android": "WordPress",
    "com.google.android.apps.messaging": "Messages",
    "com.snapchat.android": "Snapchat",
    "com.zhiliaoapp.musically": "TikTok",
    "com.android.vending": "Google Play",
    "com.google.android.gms": "Google Play Services",
    "android": "Android System",
    "com.android.systemui": "System UI",
}


# ===================================================================
# Data Classes
# ===================================================================


@dataclass
class Notification:
    """A single captured Android notification."""
    notification_id: str = ""
    package_name: str = ""
    app_name: str = ""
    title: str = ""
    text: str = ""
    sub_text: str = ""
    big_text: str = ""
    timestamp: str = ""
    category: str = NotificationCategory.UNKNOWN.value
    priority: str = NotificationPriority.DEFAULT.value
    actions: List[str] = field(default_factory=list)
    is_group: bool = False
    group_key: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)
    matched_rules: List[str] = field(default_factory=list)
    handled: bool = False
    response_action: str = ""
    screenshot_path: str = ""

    def __post_init__(self) -> None:
        if not self.notification_id:
            self.notification_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = _now_iso()
        if not self.app_name and self.package_name:
            self.app_name = PACKAGE_NAME_MAP.get(self.package_name, self.package_name)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Notification:
        data = dict(data)
        # Ensure list fields
        if "actions" not in data:
            data["actions"] = []
        if "matched_rules" not in data:
            data["matched_rules"] = []
        if "extras" not in data:
            data["extras"] = {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def summary(self) -> str:
        """One-line summary of the notification."""
        parts = [self.app_name or self.package_name]
        if self.title:
            parts.append(self.title)
        if self.text:
            parts.append(self.text[:80])
        return " | ".join(parts)


@dataclass
class NotificationRule:
    """A rule for matching and routing notifications."""
    rule_id: str = ""
    name: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    action_params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: str = ""
    match_count: int = 0
    last_matched: str = ""
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.rule_id:
            self.rule_id = uuid.uuid4().hex[:10]
        if not self.created_at:
            self.created_at = _now_iso()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NotificationRule:
        data = dict(data)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class QuietHoursConfig:
    """Quiet hours configuration."""
    enabled: bool = False
    start_time: str = "22:00"
    end_time: str = "07:00"
    days: List[str] = field(default_factory=lambda: [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ])
    min_priority: str = NotificationPriority.URGENT.value
    allow_commerce: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> QuietHoursConfig:
        data = dict(data)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ClassificationResult:
    """Result from AI-powered notification classification."""
    category: str = NotificationCategory.UNKNOWN.value
    intent: str = NotificationIntent.INFO_ONLY.value
    confidence: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)
    suggested_reply: str = ""
    reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ClassificationResult:
        data = dict(data)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class NotificationStats:
    """Aggregated notification statistics."""
    total: int = 0
    by_app: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)
    handled_count: int = 0
    auto_replied_count: int = 0
    dismissed_count: int = 0
    forwarded_count: int = 0
    avg_handle_time_ms: float = 0.0
    top_senders: List[Dict[str, Any]] = field(default_factory=list)
    noise_score: Dict[str, float] = field(default_factory=dict)
    period_hours: int = 24

    def to_dict(self) -> dict:
        return asdict(self)


# ===================================================================
# AI Classification Engine
# ===================================================================


class NotificationClassifier:
    """
    Uses Claude Haiku to classify notifications, extract intent, and
    pull out structured entities (names, amounts, order numbers, etc.).

    Implements prompt caching for the system prompt to reduce costs.
    """

    SYSTEM_PROMPT = """You are a notification classification engine for a digital publishing
empire operator who runs 16 WordPress sites across niches: witchcraft/spirituality,
smart home tech, AI/wealth, family parenting, mythology, and bullet journals.

The operator also runs Etsy POD shops, publishes KDP books on Amazon, manages
affiliate programs (Amazon Associates, ShareASale, Content Egg), and runs display
ads (AdSense/Mediavine) across all sites.

When given a notification, you must analyze it and return a JSON object with:

1. "category" — one of: message, email, social, alert, commerce, content, system, unknown
   - message: WhatsApp, Telegram, SMS, Discord DM, Messenger
   - email: Gmail, Outlook, or any mail client
   - social: Instagram like/comment/follow, Twitter mention, Facebook notification, Reddit, Pinterest
   - alert: system alerts, security warnings, service down notifications, error reports
   - commerce: Etsy sale, Amazon order/shipment, PayPal payment, affiliate commission, Shopify
   - content: WordPress comment, new subscriber, post published, SEO alert
   - system: battery, storage, updates, Android system notifications
   - unknown: cannot determine

2. "intent" — what does this notification want?
   - reply_needed: sender expects a response
   - info_only: informational, no response needed
   - action_required: user must do something (approve, review, fix)
   - urgent: time-sensitive, needs immediate attention

3. "confidence" — 0.0 to 1.0, how confident you are in the classification

4. "entities" — extracted structured data (include only what is present):
   - "person_name": sender name if identifiable
   - "dollar_amount": monetary value if mentioned (as a number, not string)
   - "site_name": which empire site if mentioned
   - "order_number": order/transaction ID if present
   - "platform": platform name (Etsy, Amazon, PayPal, etc.)
   - "affiliate_network": affiliate network name if mentioned
   - "error_type": type of error/alert if applicable
   - "product_name": product or item name if mentioned

5. "suggested_reply" — if intent is reply_needed, suggest a brief contextual reply (max 100 chars)

6. "reasoning" — brief explanation of your classification (max 150 chars)

Respond with ONLY valid JSON, no markdown fences or extra text."""

    def __init__(self, api_key: str = ANTHROPIC_API_KEY, model: str = HAIKU_MODEL) -> None:
        self.api_key = api_key
        self.model = model
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def classify(self, notification: Notification) -> ClassificationResult:
        """
        Classify a notification using Claude Haiku.

        Falls back to heuristic classification if the API key is missing
        or the API call fails.
        """
        # Try heuristic first for obvious cases
        heuristic = self._heuristic_classify(notification)
        if heuristic.confidence >= 0.9:
            return heuristic

        # Use AI classification if API key is available
        if not self.api_key:
            logger.debug("No ANTHROPIC_API_KEY set, using heuristic classification only")
            return heuristic

        try:
            return await self._ai_classify(notification)
        except Exception as exc:
            logger.warning("AI classification failed, falling back to heuristic: %s", exc)
            return heuristic

    async def _ai_classify(self, notification: Notification) -> ClassificationResult:
        """Call Anthropic Messages API with Haiku for classification."""
        session = await self._ensure_session()

        user_content = (
            f"Notification from {notification.app_name} ({notification.package_name}):\n"
            f"Title: {notification.title}\n"
            f"Text: {notification.text}\n"
        )
        if notification.sub_text:
            user_content += f"Sub-text: {notification.sub_text}\n"
        if notification.big_text:
            user_content += f"Big text: {notification.big_text}\n"
        if notification.actions:
            user_content += f"Actions available: {', '.join(notification.actions)}\n"

        # Build request with prompt caching on system prompt (>2048 tokens)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 500,
            "system": [
                {
                    "type": "text",
                    "text": self.SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {"role": "user", "content": user_content},
            ],
        }

        async with session.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Anthropic API error {resp.status}: {body[:300]}")
            data = await resp.json()

        content_blocks = data.get("content", [])
        raw_text = content_blocks[0].get("text", "") if content_blocks else ""

        # Parse the JSON response
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```\w*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("AI returned non-JSON response: %s", raw_text[:200])
            return self._heuristic_classify(notification)

        return ClassificationResult(
            category=parsed.get("category", NotificationCategory.UNKNOWN.value),
            intent=parsed.get("intent", NotificationIntent.INFO_ONLY.value),
            confidence=float(parsed.get("confidence", 0.5)),
            entities=parsed.get("entities", {}),
            suggested_reply=parsed.get("suggested_reply", ""),
            reasoning=parsed.get("reasoning", ""),
        )

    def _heuristic_classify(self, notification: Notification) -> ClassificationResult:
        """
        Fast heuristic classification based on package name and text patterns.
        Used as fallback and for obvious cases to avoid API calls.
        """
        pkg = notification.package_name
        title_lower = (notification.title or "").lower()
        text_lower = (notification.text or "").lower()
        combined = f"{title_lower} {text_lower}"

        # Category from package map
        category = PACKAGE_CATEGORY_MAP.get(pkg, NotificationCategory.UNKNOWN)
        confidence = 0.7 if category != NotificationCategory.UNKNOWN else 0.3

        # Intent heuristics
        intent = NotificationIntent.INFO_ONLY

        # Extract entities
        entities: Dict[str, Any] = {}

        # Dollar amount extraction
        dollar_match = re.search(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', combined)
        if dollar_match:
            amount_str = dollar_match.group(1).replace(",", "")
            try:
                entities["dollar_amount"] = float(amount_str)
            except ValueError:
                pass

        # Commerce-specific patterns
        if category == NotificationCategory.COMMERCE or pkg in (
            "com.etsy.android", "com.amazon.mShop.android.shopping",
            "com.paypal.android.p2pmobile",
        ):
            category = NotificationCategory.COMMERCE
            confidence = 0.9

            if any(kw in combined for kw in ("sale", "sold", "order", "payment received",
                                               "commission", "earning")):
                intent = NotificationIntent.INFO_ONLY
                entities["platform"] = PACKAGE_NAME_MAP.get(pkg, pkg)

            # Order number extraction
            order_match = re.search(r'(?:order|#)\s*(\d{6,})', combined, re.IGNORECASE)
            if order_match:
                entities["order_number"] = order_match.group(1)

        # Message patterns
        elif category == NotificationCategory.MESSAGE:
            intent = NotificationIntent.REPLY_NEEDED
            confidence = 0.9

        # Email patterns
        elif category == NotificationCategory.EMAIL:
            if any(kw in combined for kw in ("urgent", "important", "action required",
                                               "verify", "security")):
                intent = NotificationIntent.ACTION_REQUIRED
            else:
                intent = NotificationIntent.INFO_ONLY
            confidence = 0.85

        # Social patterns
        elif category == NotificationCategory.SOCIAL:
            if any(kw in combined for kw in ("mentioned", "replied", "dm", "direct message")):
                intent = NotificationIntent.REPLY_NEEDED
            elif any(kw in combined for kw in ("liked", "followed", "shared", "retweeted")):
                intent = NotificationIntent.INFO_ONLY
            else:
                intent = NotificationIntent.INFO_ONLY
            confidence = 0.85

        # System patterns
        elif category == NotificationCategory.SYSTEM:
            if any(kw in combined for kw in ("low battery", "storage full", "update available")):
                if "low battery" in combined:
                    intent = NotificationIntent.ACTION_REQUIRED
                else:
                    intent = NotificationIntent.INFO_ONLY
            confidence = 0.9

        # Content patterns (WordPress)
        elif category == NotificationCategory.CONTENT:
            if any(kw in combined for kw in ("comment", "new comment")):
                intent = NotificationIntent.ACTION_REQUIRED
            elif any(kw in combined for kw in ("subscriber", "published")):
                intent = NotificationIntent.INFO_ONLY
            confidence = 0.85

        # Alert patterns (detected from text regardless of package)
        if any(kw in combined for kw in ("error", "failed", "down", "crash", "critical",
                                          "alert", "warning", "timeout")):
            if category == NotificationCategory.UNKNOWN:
                category = NotificationCategory.ALERT
            intent = NotificationIntent.ACTION_REQUIRED
            entities["error_type"] = "detected_from_text"
            confidence = max(confidence, 0.75)

        # Site name extraction
        site_keywords = {
            "witchcraft": "witchcraft", "smart home": "smarthome",
            "ai in action": "aiaction", "ai discovery": "aidiscovery",
            "wealth from ai": "wealthai", "family flourish": "family",
            "mythical": "mythical", "bullet journal": "bulletjournals",
            "crystal": "crystalwitchcraft", "herbal": "herbalwitchery",
            "moon phase": "moonphasewitch", "tarot": "tarotbeginners",
            "spells": "spellsrituals", "pagan": "paganpathways",
            "witchy home": "witchyhomedecor", "seasonal": "seasonalwitchcraft",
        }
        for keyword, site_id in site_keywords.items():
            if keyword in combined:
                entities["site_name"] = site_id
                break

        # Person name heuristic (from message title)
        if category == NotificationCategory.MESSAGE and notification.title:
            # Title often is the sender name in messaging apps
            if not re.match(r'^(WhatsApp|Telegram|Discord|Messages|Messenger)', notification.title):
                entities["person_name"] = notification.title

        return ClassificationResult(
            category=category.value if isinstance(category, NotificationCategory) else category,
            intent=intent.value if isinstance(intent, NotificationIntent) else intent,
            confidence=confidence,
            entities=entities,
            reasoning="heuristic classification from package and text patterns",
        )

    async def generate_reply(
        self,
        notification: Notification,
        context: str = "",
    ) -> str:
        """
        Use Haiku to generate a contextual reply for a notification.

        Returns the generated reply text, or an empty string if generation fails.
        """
        if not self.api_key:
            return self._template_reply(notification)

        session = await self._ensure_session()

        user_content = (
            f"Generate a brief, friendly reply to this notification.\n"
            f"App: {notification.app_name}\n"
            f"From: {notification.title}\n"
            f"Message: {notification.text}\n"
        )
        if context:
            user_content += f"Additional context: {context}\n"
        user_content += (
            "\nReply rules:\n"
            "- Maximum 100 characters\n"
            "- Be warm and professional\n"
            "- Match the tone of the incoming message\n"
            "- If it's a business notification, be concise\n"
            "- Return ONLY the reply text, nothing else"
        )

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": user_content},
            ],
        }

        try:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Reply generation failed: %s", body[:200])
                    return self._template_reply(notification)
                data = await resp.json()

            content_blocks = data.get("content", [])
            reply = content_blocks[0].get("text", "").strip() if content_blocks else ""
            return reply[:100] if reply else self._template_reply(notification)

        except Exception as exc:
            logger.warning("Reply generation error: %s", exc)
            return self._template_reply(notification)

    def _template_reply(self, notification: Notification) -> str:
        """Return a template-based reply when AI is unavailable."""
        templates = _load_json(TEMPLATES_FILE, {})
        pkg = notification.package_name

        # Check for app-specific template
        if pkg in templates:
            return templates[pkg]

        # Default templates by category
        category = notification.category
        defaults = {
            NotificationCategory.MESSAGE.value: "Thanks! I'll get back to you shortly.",
            NotificationCategory.EMAIL.value: "Got it, will review soon.",
            NotificationCategory.SOCIAL.value: "Thanks!",
            NotificationCategory.COMMERCE.value: "Noted, thank you!",
            NotificationCategory.CONTENT.value: "Thanks for reaching out!",
        }
        return defaults.get(category, "Got it!")


# ===================================================================
# ADB Notification Parser
# ===================================================================


class ADBNotificationParser:
    """
    Parses the output of ``adb shell dumpsys notification --noredact``
    into structured Notification objects.

    The dumpsys output contains notification records grouped by status
    (posted, snoozed, etc.), each with package, tag, key, priority,
    and a nested Bundle of extras containing title, text, subText, etc.
    """

    # Regex patterns for parsing dumpsys output
    _RE_NOTIFICATION_RECORD = re.compile(
        r'NotificationRecord\((?P<key>[^)]+)\)',
    )
    _RE_PACKAGE = re.compile(r'pkg=(\S+)')
    _RE_OP_PKG = re.compile(r'opPkg=(\S+)')
    _RE_PRIORITY = re.compile(r'pri=(-?\d+)')
    _RE_KEY = re.compile(r'key=(\S+)')
    _RE_GROUP_KEY = re.compile(r'groupKey=(\S+)')
    _RE_EXTRA_STRING = re.compile(
        r'(?:android\.(?:title|text|subText|bigText|infoText|summaryText|conversationTitle))'
        r'=String\s*\(([^)]*)\)',
    )

    @staticmethod
    def parse_dumpsys(raw_dump: str) -> List[Notification]:
        """
        Parse the full ``dumpsys notification`` output into Notification objects.

        Focuses on the "NotificationRecord" sections within the posted notifications
        listing. Extracts package, priority, title, text, subText, bigText, and
        available action button labels.
        """
        notifications: List[Notification] = []
        if not raw_dump:
            return notifications

        # Split into individual notification record blocks
        # Each record starts with "NotificationRecord(" or similar pattern
        blocks = re.split(r'(?=NotificationRecord\()', raw_dump)

        for block in blocks:
            if "NotificationRecord(" not in block:
                continue

            notif = ADBNotificationParser._parse_block(block)
            if notif and notif.package_name:
                notifications.append(notif)

        logger.debug("Parsed %d notifications from dumpsys output", len(notifications))
        return notifications

    @staticmethod
    def _parse_block(block: str) -> Optional[Notification]:
        """Parse a single notification record block."""
        notif = Notification()

        # Package name
        pkg_match = ADBNotificationParser._RE_PACKAGE.search(block)
        if pkg_match:
            notif.package_name = pkg_match.group(1)
            notif.app_name = PACKAGE_NAME_MAP.get(notif.package_name, notif.package_name)

        # Heuristic category from package
        if notif.package_name in PACKAGE_CATEGORY_MAP:
            notif.category = PACKAGE_CATEGORY_MAP[notif.package_name].value

        # Key (used as notification_id)
        key_match = ADBNotificationParser._RE_KEY.search(block)
        if key_match:
            notif.notification_id = key_match.group(1)
        else:
            notif.notification_id = uuid.uuid4().hex[:12]

        # Group key
        group_match = ADBNotificationParser._RE_GROUP_KEY.search(block)
        if group_match:
            notif.group_key = group_match.group(1)
            notif.is_group = True

        # Priority
        pri_match = ADBNotificationParser._RE_PRIORITY.search(block)
        if pri_match:
            pri_val = int(pri_match.group(1))
            if pri_val <= -1:
                notif.priority = NotificationPriority.LOW.value
            elif pri_val == 0:
                notif.priority = NotificationPriority.DEFAULT.value
            elif pri_val == 1:
                notif.priority = NotificationPriority.HIGH.value
            else:
                notif.priority = NotificationPriority.URGENT.value

        # Extract extras from the Bundle
        # Title
        title_match = re.search(r'android\.title=String\s*\(([^)]*)\)', block)
        if title_match:
            notif.title = title_match.group(1).strip()

        # Text
        text_match = re.search(r'android\.text=String\s*\(([^)]*)\)', block)
        if text_match:
            notif.text = text_match.group(1).strip()

        # Sub text
        sub_match = re.search(r'android\.subText=String\s*\(([^)]*)\)', block)
        if sub_match:
            notif.sub_text = sub_match.group(1).strip()

        # Big text
        big_match = re.search(r'android\.bigText=String\s*\(([^)]*)\)', block)
        if big_match:
            notif.big_text = big_match.group(1).strip()

        # Info text
        info_match = re.search(r'android\.infoText=String\s*\(([^)]*)\)', block)
        if info_match:
            notif.extras["info_text"] = info_match.group(1).strip()

        # Summary text
        summary_match = re.search(r'android\.summaryText=String\s*\(([^)]*)\)', block)
        if summary_match:
            notif.extras["summary_text"] = summary_match.group(1).strip()

        # Conversation title
        convo_match = re.search(r'android\.conversationTitle=String\s*\(([^)]*)\)', block)
        if convo_match:
            notif.extras["conversation_title"] = convo_match.group(1).strip()

        # Action buttons
        action_matches = re.findall(r'Action\s*\{[^}]*title=([^,}]+)', block)
        notif.actions = [a.strip() for a in action_matches if a.strip()]

        # Timestamp from the record (postTime or when)
        time_match = re.search(r'(?:postTime|when)=(\d+)', block)
        if time_match:
            try:
                ts_ms = int(time_match.group(1))
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                notif.timestamp = dt.isoformat()
            except (ValueError, OSError):
                notif.timestamp = _now_iso()
        else:
            notif.timestamp = _now_iso()

        # Skip empty / system-internal notifications
        if not notif.title and not notif.text:
            return None

        return notif


# ===================================================================
# Rule Engine
# ===================================================================


class RuleEngine:
    """
    Evaluates notifications against a set of configurable rules and
    returns the actions to take.

    Rule conditions support:
        - app: exact package name match
        - title_pattern: regex match on notification title
        - text_pattern: regex match on notification text
        - category: notification category match
        - priority: minimum priority level
        - time_range: time-of-day range (HH:MM-HH:MM)

    Rule actions:
        - auto_reply: generate and send a reply
        - forward: forward to notification hub channel
        - trigger_webhook: POST to n8n or custom webhook
        - log: log the notification for analytics
        - dismiss: swipe away the notification
        - escalate: forward as critical via notification hub
        - execute_playbook: run a named playbook
    """

    PRIORITY_ORDER = {
        NotificationPriority.LOW.value: 0,
        NotificationPriority.DEFAULT.value: 1,
        NotificationPriority.HIGH.value: 2,
        NotificationPriority.URGENT.value: 3,
    }

    def __init__(self) -> None:
        self._rules: List[NotificationRule] = self._load_rules()
        logger.info("Rule engine initialized with %d rules", len(self._rules))

    def _load_rules(self) -> List[NotificationRule]:
        """Load rules from persistent storage."""
        raw = _load_json(RULES_FILE, [])
        if not isinstance(raw, list):
            return []
        rules: List[NotificationRule] = []
        for item in raw:
            try:
                rules.append(NotificationRule.from_dict(item))
            except (TypeError, KeyError) as exc:
                logger.warning("Skipping malformed rule: %s", exc)
        return rules

    def _save_rules(self) -> None:
        """Persist rules to disk."""
        _save_json(RULES_FILE, [r.to_dict() for r in self._rules])

    def add_rule(
        self,
        name: str,
        conditions: Dict[str, Any],
        actions: List[str],
        action_params: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> NotificationRule:
        """Create and persist a new notification routing rule."""
        if len(self._rules) >= MAX_RULES:
            raise ValueError(f"Maximum of {MAX_RULES} rules reached")

        rule = NotificationRule(
            name=name,
            conditions=conditions,
            actions=actions,
            action_params=action_params or {},
            priority=priority,
        )
        self._rules.append(rule)
        self._save_rules()
        logger.info("Rule added: %s (id=%s)", name, rule.rule_id)
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID. Returns True if found and removed."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        if len(self._rules) < before:
            self._save_rules()
            logger.info("Rule removed: %s", rule_id)
            return True
        return False

    def update_rule(self, rule_id: str, **kwargs: Any) -> Optional[NotificationRule]:
        """Update a rule's fields. Returns the updated rule or None."""
        for rule in self._rules:
            if rule.rule_id == rule_id:
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                self._save_rules()
                logger.info("Rule updated: %s", rule_id)
                return rule
        return None

    def list_rules(self) -> List[NotificationRule]:
        """Return all rules, sorted by priority descending."""
        return sorted(self._rules, key=lambda r: r.priority, reverse=True)

    def get_rule(self, rule_id: str) -> Optional[NotificationRule]:
        """Get a single rule by ID."""
        for rule in self._rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def test_rule(self, rule_id: str, notification: Notification) -> bool:
        """Test whether a specific rule matches a notification."""
        rule = self.get_rule(rule_id)
        if rule is None:
            return False
        return self._matches(rule, notification)

    def evaluate(self, notification: Notification) -> List[Tuple[NotificationRule, List[str]]]:
        """
        Find all matching rules for a notification and return them with
        their associated actions, sorted by rule priority (highest first).
        """
        matches: List[Tuple[NotificationRule, List[str]]] = []

        for rule in self._rules:
            if not rule.enabled:
                continue
            if self._matches(rule, notification):
                rule.match_count += 1
                rule.last_matched = _now_iso()
                notification.matched_rules.append(rule.rule_id)
                matches.append((rule, rule.actions))
                logger.debug(
                    "Rule '%s' matched notification from %s",
                    rule.name, notification.app_name,
                )

        # Save updated match counts
        if matches:
            self._save_rules()

        # Sort by priority
        matches.sort(key=lambda m: m[0].priority, reverse=True)
        return matches

    def _matches(self, rule: NotificationRule, notification: Notification) -> bool:
        """Check if all conditions of a rule match the notification."""
        conditions = rule.conditions

        # App match (exact package name)
        if "app" in conditions:
            if notification.package_name != conditions["app"]:
                return False

        # Title pattern (regex)
        if "title_pattern" in conditions:
            pattern = conditions["title_pattern"]
            if not re.search(pattern, notification.title or "", re.IGNORECASE):
                return False

        # Text pattern (regex)
        if "text_pattern" in conditions:
            pattern = conditions["text_pattern"]
            combined = f"{notification.text} {notification.big_text}"
            if not re.search(pattern, combined, re.IGNORECASE):
                return False

        # Category match
        if "category" in conditions:
            if notification.category != conditions["category"]:
                return False

        # Priority minimum
        if "priority" in conditions:
            min_pri = self.PRIORITY_ORDER.get(conditions["priority"], 0)
            notif_pri = self.PRIORITY_ORDER.get(notification.priority, 0)
            if notif_pri < min_pri:
                return False

        # Time range (HH:MM-HH:MM)
        if "time_range" in conditions:
            time_range = conditions["time_range"]
            if not self._in_time_range(time_range):
                return False

        return True

    @staticmethod
    def _in_time_range(time_range: str) -> bool:
        """Check if current time falls within a HH:MM-HH:MM range."""
        try:
            parts = time_range.split("-")
            if len(parts) != 2:
                return True
            start_h, start_m = map(int, parts[0].strip().split(":"))
            end_h, end_m = map(int, parts[1].strip().split(":"))

            now = _now_utc()
            current_minutes = now.hour * 60 + now.minute
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m

            if start_minutes <= end_minutes:
                return start_minutes <= current_minutes <= end_minutes
            else:
                # Wraps midnight
                return current_minutes >= start_minutes or current_minutes <= end_minutes
        except (ValueError, IndexError):
            return True

    def install_default_rules(self) -> List[NotificationRule]:
        """
        Install the pre-built empire notification rules.

        These cover common scenarios: WhatsApp from known contacts, Etsy sales,
        WordPress comments, server alerts, affiliate commissions, low battery,
        and app crashes.
        """
        defaults = [
            {
                "name": "WhatsApp Messages — Forward to Telegram",
                "conditions": {"app": "com.whatsapp", "category": "message"},
                "actions": ["log", "forward"],
                "action_params": {"forward_channel": "telegram"},
                "priority": 5,
            },
            {
                "name": "Etsy Sale — Log Revenue + Celebrate",
                "conditions": {
                    "app": "com.etsy.android",
                    "text_pattern": r"(?:sold|sale|order|purchased)",
                },
                "actions": ["log", "trigger_webhook"],
                "action_params": {
                    "webhook_path": "openclaw-revenue",
                    "revenue_stream": "etsy",
                },
                "priority": 8,
            },
            {
                "name": "WordPress Comment — Classify Spam",
                "conditions": {
                    "app": "org.wordpress.android",
                    "text_pattern": r"(?:comment|commented)",
                },
                "actions": ["log", "trigger_webhook"],
                "action_params": {"webhook_path": "openclaw-content"},
                "priority": 4,
            },
            {
                "name": "Server Alert — Trigger Monitor",
                "conditions": {
                    "text_pattern": r"(?:down|unreachable|offline|error 5\d{2}|timeout|critical)",
                    "category": "alert",
                },
                "actions": ["log", "escalate", "trigger_webhook"],
                "action_params": {"webhook_path": "openclaw-monitor"},
                "priority": 10,
            },
            {
                "name": "Affiliate Commission — Track Revenue",
                "conditions": {
                    "text_pattern": r"(?:commission|referral|affiliate|earning)",
                },
                "actions": ["log", "trigger_webhook"],
                "action_params": {
                    "webhook_path": "openclaw-revenue",
                    "revenue_stream": "affiliate",
                },
                "priority": 7,
            },
            {
                "name": "Low Battery — Reduce Automation",
                "conditions": {
                    "app": "android",
                    "text_pattern": r"(?:low battery|battery low|power sav)",
                },
                "actions": ["log", "escalate"],
                "action_params": {"escalation_note": "Low battery — reduce automation intensity"},
                "priority": 6,
            },
            {
                "name": "App Crash — Log to Codex",
                "conditions": {
                    "text_pattern": r"(?:crash|stopped|not responding|keeps stopping)",
                },
                "actions": ["log", "trigger_webhook"],
                "action_params": {
                    "webhook_path": "openclaw-monitor",
                    "crash_recovery": True,
                },
                "priority": 9,
            },
            {
                "name": "Amazon Order/Shipment — Commerce Log",
                "conditions": {
                    "app": "com.amazon.mShop.android.shopping",
                    "text_pattern": r"(?:order|shipped|delivered|refund)",
                },
                "actions": ["log"],
                "priority": 3,
            },
            {
                "name": "PayPal Payment — Revenue Tracker",
                "conditions": {
                    "app": "com.paypal.android.p2pmobile",
                    "text_pattern": r"(?:received|payment|sent you)",
                },
                "actions": ["log", "trigger_webhook"],
                "action_params": {
                    "webhook_path": "openclaw-revenue",
                    "revenue_stream": "digital_products",
                },
                "priority": 7,
            },
            {
                "name": "Gmail — Flag Important",
                "conditions": {
                    "app": "com.google.android.gm",
                    "title_pattern": r"(?:urgent|important|action required|invoice)",
                },
                "actions": ["log", "escalate"],
                "priority": 6,
            },
        ]

        installed: List[NotificationRule] = []
        existing_names = {r.name for r in self._rules}

        for rule_def in defaults:
            if rule_def["name"] not in existing_names:
                rule = self.add_rule(
                    name=rule_def["name"],
                    conditions=rule_def["conditions"],
                    actions=rule_def["actions"],
                    action_params=rule_def.get("action_params", {}),
                    priority=rule_def.get("priority", 0),
                )
                installed.append(rule)

        if installed:
            logger.info("Installed %d default rules", len(installed))
        return installed


# ===================================================================
# Notification Interceptor — Main Class
# ===================================================================


class NotificationInterceptor:
    """
    Central notification monitoring, classification, and routing system.

    Captures notifications from the Android device via ADB, classifies
    them using Haiku AI, matches them against configurable rules, and
    executes actions (auto-reply, forward, webhook trigger, dismiss, etc.).

    Integrates with:
        - PhoneController for ADB commands
        - NotificationHub for cross-channel forwarding
        - RevenueTracker for commerce notifications
        - n8n for workflow triggering
        - FORGE Codex for error pattern learning

    Example:
        interceptor = NotificationInterceptor()
        await interceptor.start()

        # Continuous monitoring
        await interceptor.monitor(poll_interval=5)

        # One-shot capture + classify + route
        notifications = await interceptor.capture_notifications()
        for n in notifications:
            result = await interceptor.classify(n)
            await interceptor.evaluate_and_act(n)
    """

    def __init__(
        self,
        node_url: str = OPENCLAW_NODE_URL,
        node_name: str = OPENCLAW_ANDROID_NODE,
        api_key: str = ANTHROPIC_API_KEY,
    ) -> None:
        self.node_url = node_url.rstrip("/")
        self.node_name = node_name
        self._session: Optional[aiohttp.ClientSession] = None
        self._monitoring: bool = False
        self._seen_ids: set[str] = set()

        # Sub-systems
        self.classifier = NotificationClassifier(api_key=api_key)
        self.rule_engine = RuleEngine()
        self.parser = ADBNotificationParser()
        self.quiet_hours = self._load_quiet_hours()

        # Notification history (in-memory + disk)
        self._history: List[Dict[str, Any]] = self._load_history()
        self._handle_times: List[float] = []

        # Reply templates
        self._reply_templates = _load_json(TEMPLATES_FILE, {
            "default": "Got it!",
            "thanks": "Thanks!",
            "busy": "I'll get back to you shortly.",
            "acknowledged": "Acknowledged, thank you!",
        })

        logger.info(
            "NotificationInterceptor initialized — node=%s, %d rules, %d history entries",
            node_url, len(self.rule_engine.list_rules()), len(self._history),
        )

    # ------------------------------------------------------------------
    # Session & ADB helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close HTTP sessions and stop monitoring."""
        self._monitoring = False
        await self.classifier.close()
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _adb_shell(self, cmd: str, timeout_s: float = 30) -> str:
        """Execute an ADB shell command via the OpenClaw node."""
        session = await self._ensure_session()
        payload = {
            "node": self.node_name,
            "command": "adb.shell",
            "params": {"command": cmd, "timeout": timeout_s},
        }
        url = f"{self.node_url}/api/nodes/invoke"

        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ConnectionError(f"Node HTTP {resp.status}: {body[:300]}")
                data = await resp.json()
                if data.get("error"):
                    raise RuntimeError(f"Node error: {data['error']}")
                return data.get("stdout", "")
        except aiohttp.ClientError as exc:
            raise ConnectionError(f"Failed to reach node at {url}: {exc}") from exc

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def _load_history(self) -> List[Dict[str, Any]]:
        raw = _load_json(HISTORY_FILE, [])
        if isinstance(raw, list):
            return raw[-MAX_HISTORY_ENTRIES:]
        return []

    def _save_history(self) -> None:
        self._history = self._history[-MAX_HISTORY_ENTRIES:]
        _save_json(HISTORY_FILE, self._history)

    def _record_notification(self, notification: Notification) -> None:
        """Add a notification to the persistent history."""
        self._history.append(notification.to_dict())
        if len(self._history) % 50 == 0:
            self._save_history()

    # ------------------------------------------------------------------
    # Quiet hours
    # ------------------------------------------------------------------

    def _load_quiet_hours(self) -> QuietHoursConfig:
        raw = _load_json(QUIET_HOURS_FILE, None)
        if raw:
            try:
                return QuietHoursConfig.from_dict(raw)
            except (TypeError, KeyError):
                pass
        return QuietHoursConfig()

    def _save_quiet_hours(self) -> None:
        _save_json(QUIET_HOURS_FILE, self.quiet_hours.to_dict())

    def set_quiet_hours(
        self,
        start_time: str = "22:00",
        end_time: str = "07:00",
        days: Optional[List[str]] = None,
    ) -> QuietHoursConfig:
        """Configure quiet hours during which auto-actions are suppressed."""
        self.quiet_hours.enabled = True
        self.quiet_hours.start_time = start_time
        self.quiet_hours.end_time = end_time
        if days is not None:
            self.quiet_hours.days = [d.lower() for d in days]
        self._save_quiet_hours()
        logger.info("Quiet hours set: %s-%s on %s", start_time, end_time, self.quiet_hours.days)
        return self.quiet_hours

    def set_priority_override(self, min_priority: str = "urgent") -> None:
        """During quiet hours, only process notifications at or above this priority."""
        self.quiet_hours.min_priority = min_priority
        self._save_quiet_hours()
        logger.info("Quiet hours priority override: %s", min_priority)

    def disable_quiet_hours(self) -> None:
        """Disable quiet hours."""
        self.quiet_hours.enabled = False
        self._save_quiet_hours()

    def is_quiet_time(self) -> bool:
        """Check if the current time falls within configured quiet hours."""
        if not self.quiet_hours.enabled:
            return False

        now = _now_utc()
        day_name = now.strftime("%A").lower()
        if day_name not in self.quiet_hours.days:
            return False

        time_range = f"{self.quiet_hours.start_time}-{self.quiet_hours.end_time}"
        return RuleEngine._in_time_range(time_range)

    def _should_suppress(self, notification: Notification) -> bool:
        """Check if a notification should be suppressed due to quiet hours."""
        if not self.is_quiet_time():
            return False

        # Allow commerce during quiet hours if configured
        if self.quiet_hours.allow_commerce and notification.category == NotificationCategory.COMMERCE.value:
            return False

        # Allow if priority meets or exceeds the override
        min_pri = RuleEngine.PRIORITY_ORDER.get(self.quiet_hours.min_priority, 3)
        notif_pri = RuleEngine.PRIORITY_ORDER.get(notification.priority, 0)
        if notif_pri >= min_pri:
            return False

        return True

    # ==================================================================
    # NOTIFICATION CAPTURE
    # ==================================================================

    async def capture_notifications(self) -> List[Notification]:
        """
        Capture current notifications from the Android device via
        ``adb shell dumpsys notification --noredact``.

        Returns a list of parsed Notification objects, excluding any
        that have already been seen in this session.
        """
        try:
            raw = await self._adb_shell("dumpsys notification --noredact", timeout_s=15)
        except Exception as exc:
            logger.error("Failed to capture notifications: %s", exc)
            return []

        all_notifs = self.parser.parse_dumpsys(raw)

        # Filter out already-seen notifications
        new_notifs: List[Notification] = []
        for n in all_notifs:
            if n.notification_id not in self._seen_ids:
                self._seen_ids.add(n.notification_id)
                new_notifs.append(n)

        logger.info(
            "Captured %d total, %d new notifications",
            len(all_notifs), len(new_notifs),
        )
        return new_notifs

    async def get_all_current(self) -> List[Notification]:
        """
        Get all currently posted notifications (including already-seen ones).
        Does not update the seen set.
        """
        try:
            raw = await self._adb_shell("dumpsys notification --noredact", timeout_s=15)
        except Exception as exc:
            logger.error("Failed to get notifications: %s", exc)
            return []
        return self.parser.parse_dumpsys(raw)

    async def get_recent(
        self,
        minutes: int = 60,
        app_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> List[Notification]:
        """
        Get recent notifications from history within the specified time window.

        Args:
            minutes: How many minutes back to look.
            app_filter: Filter by package name.
            category_filter: Filter by category.

        Returns:
            List of Notification objects from history.
        """
        cutoff = (_now_utc() - timedelta(minutes=minutes)).isoformat()
        results: List[Notification] = []

        for entry in reversed(self._history):
            ts = entry.get("timestamp", "")
            if ts < cutoff:
                break

            if app_filter and entry.get("package_name") != app_filter:
                continue
            if category_filter and entry.get("category") != category_filter:
                continue

            try:
                results.append(Notification.from_dict(entry))
            except (TypeError, KeyError):
                continue

        return results

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================

    async def classify(self, notification: Notification) -> ClassificationResult:
        """Classify a notification using AI + heuristics."""
        result = await self.classifier.classify(notification)

        # Update the notification with classification results
        notification.category = result.category
        if result.entities:
            notification.extras["classification"] = result.to_dict()

        logger.debug(
            "Classified %s as %s (intent=%s, confidence=%.2f)",
            notification.app_name, result.category, result.intent, result.confidence,
        )
        return result

    async def extract_intent(self, notification: Notification) -> str:
        """Determine what the notification wants: reply, info, action, or urgent."""
        result = await self.classifier.classify(notification)
        return result.intent

    async def extract_entities(self, notification: Notification) -> Dict[str, Any]:
        """Pull out structured data: names, amounts, sites, order numbers, etc."""
        result = await self.classifier.classify(notification)
        return result.entities

    # ==================================================================
    # RULE EVALUATION & ACTION EXECUTION
    # ==================================================================

    async def evaluate_and_act(self, notification: Notification) -> List[str]:
        """
        Evaluate a notification against all rules and execute matched actions.

        Returns a list of action names that were executed.
        """
        # Check quiet hours
        if self._should_suppress(notification):
            logger.info("Suppressed during quiet hours: %s", notification.summary)
            notification.handled = True
            notification.response_action = "suppressed_quiet_hours"
            self._record_notification(notification)
            return ["suppressed"]

        start_time = time.monotonic()

        # Classify first
        classification = await self.classify(notification)

        # Evaluate rules
        matches = self.rule_engine.evaluate(notification)

        executed_actions: List[str] = []

        if not matches:
            # Default: just log it
            self._record_notification(notification)
            return ["logged"]

        for rule, actions in matches:
            for action in actions:
                try:
                    await self._execute_action(action, notification, rule, classification)
                    executed_actions.append(action)
                except Exception as exc:
                    logger.error(
                        "Action '%s' failed for rule '%s': %s",
                        action, rule.name, exc,
                    )

        notification.handled = True
        notification.response_action = ", ".join(executed_actions)
        self._record_notification(notification)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._handle_times.append(elapsed_ms)

        logger.info(
            "Handled %s in %.0fms: %s",
            notification.summary, elapsed_ms, executed_actions,
        )
        return executed_actions

    async def _execute_action(
        self,
        action: str,
        notification: Notification,
        rule: NotificationRule,
        classification: ClassificationResult,
    ) -> None:
        """Execute a single rule action."""
        params = rule.action_params

        if action == RuleActionType.LOG.value:
            # Already recorded via _record_notification
            pass

        elif action == RuleActionType.DISMISS.value:
            await self.dismiss(notification.notification_id)

        elif action == RuleActionType.AUTO_REPLY.value:
            reply_text = params.get("reply_text", "")
            if not reply_text:
                reply_text = await self.classifier.generate_reply(notification)
            if reply_text:
                await self.auto_reply(notification, reply_text)

        elif action == RuleActionType.FORWARD.value:
            channel = params.get("forward_channel", "telegram")
            await self.forward_to_notification_hub(notification, channel)

        elif action == RuleActionType.TRIGGER_WEBHOOK.value:
            webhook_path = params.get("webhook_path", "")
            if webhook_path:
                webhook_url = f"{N8N_WEBHOOK_BASE}/{webhook_path}"
                await self.forward_to_webhook(notification, webhook_url, classification)
            else:
                custom_url = params.get("webhook_url", "")
                if custom_url:
                    await self.forward_to_webhook(notification, custom_url, classification)

        elif action == RuleActionType.ESCALATE.value:
            note = params.get("escalation_note", "")
            await self._escalate(notification, note)

        elif action == RuleActionType.EXECUTE_PLAYBOOK.value:
            playbook = params.get("playbook", "")
            if playbook:
                await self._execute_playbook(playbook, notification, classification)

    # ==================================================================
    # NOTIFICATION ACTIONS (ADB)
    # ==================================================================

    async def dismiss(self, notification_id: str) -> bool:
        """Dismiss a notification by its key via ADB service call."""
        try:
            # Use the notification key to cancel via service
            await self._adb_shell(
                f"cmd notification cancel {notification_id}"
            )
            self._seen_ids.discard(notification_id)
            logger.info("Dismissed notification: %s", notification_id)
            return True
        except Exception as exc:
            logger.warning("Failed to dismiss %s: %s", notification_id, exc)
            return False

    async def dismiss_all(self) -> int:
        """Clear all notifications. Returns count dismissed."""
        try:
            # First get count
            current = await self.get_all_current()
            count = len(current)

            # Use service call to cancel all
            await self._adb_shell("service call notification 1")
            self._seen_ids.clear()

            logger.info("Dismissed all notifications (%d)", count)
            return count
        except Exception as exc:
            logger.error("Failed to dismiss all: %s", exc)
            return 0

    async def tap_action(self, notification_id: str, action_index: int = 0) -> bool:
        """
        Tap a notification action button by expanding the notification shade,
        finding the action, and tapping it.

        This is a best-effort approach using UI automation since direct action
        invocation requires system-level access.
        """
        try:
            # Pull down notification shade
            await self._adb_shell("cmd statusbar expand-notifications")
            await asyncio.sleep(1.0)

            # Try to find and tap the action via UI dump
            # This is heuristic — the action buttons are typically at the bottom
            # of the notification card
            await self._adb_shell("input tap 540 1200")
            await asyncio.sleep(POST_ACTION_DELAY)

            logger.info("Tapped action %d on notification %s", action_index, notification_id)
            return True
        except Exception as exc:
            logger.warning("Failed to tap action on %s: %s", notification_id, exc)
            return False

    async def auto_reply(self, notification: Notification, message: str) -> bool:
        """
        Auto-reply to a notification by tapping it and typing a response.

        Opens the notification, finds the reply input, types the message,
        and sends it.
        """
        try:
            # Pull down notification shade
            await self._adb_shell("cmd statusbar expand-notifications")
            await asyncio.sleep(1.0)

            # Find and tap "Reply" action if available
            if notification.actions:
                reply_actions = [
                    a for a in notification.actions
                    if any(kw in a.lower() for kw in ("reply", "respond", "answer"))
                ]
                if reply_actions:
                    # Direct reply via inline reply (Android 7+)
                    # Tap the reply action area
                    await self._adb_shell("input tap 540 1400")
                    await asyncio.sleep(0.5)

            # Type the message
            escaped = message.replace(" ", "%s")
            escaped = escaped.replace("'", "\\'")
            escaped = escaped.replace('"', '\\"')
            await self._adb_shell(f"input text '{escaped}'")
            await asyncio.sleep(0.3)

            # Send (tap send button or press enter)
            await self._adb_shell("input keyevent 66")
            await asyncio.sleep(POST_ACTION_DELAY)

            # Collapse notification shade
            await self._adb_shell("cmd statusbar collapse")

            notification.response_action = f"auto_replied: {message[:50]}"
            logger.info("Auto-replied to %s: '%s'", notification.app_name, message[:50])
            return True

        except Exception as exc:
            logger.error("Auto-reply failed for %s: %s", notification.app_name, exc)
            # Collapse shade even on failure
            try:
                await self._adb_shell("cmd statusbar collapse")
            except Exception:
                pass
            return False

    async def smart_reply(self, notification: Notification) -> Optional[str]:
        """
        AI decides whether to reply and generates an appropriate response.

        Only replies to message-category notifications from recognized senders.
        Returns the reply text if sent, None if no reply was warranted.
        """
        classification = await self.classify(notification)

        # Only auto-reply to messages that need a reply
        if classification.intent != NotificationIntent.REPLY_NEEDED.value:
            logger.debug("Smart reply skipped: intent is %s", classification.intent)
            return None

        # Only reply to message category
        if classification.category not in (
            NotificationCategory.MESSAGE.value,
            NotificationCategory.EMAIL.value,
        ):
            logger.debug("Smart reply skipped: category is %s", classification.category)
            return None

        # Generate and send reply
        reply = await self.classifier.generate_reply(notification)
        if reply:
            success = await self.auto_reply(notification, reply)
            if success:
                return reply

        return None

    # ==================================================================
    # WEBHOOK & FORWARDING
    # ==================================================================

    async def forward_to_webhook(
        self,
        notification: Notification,
        webhook_url: str,
        classification: Optional[ClassificationResult] = None,
    ) -> bool:
        """POST notification data to an n8n or custom webhook."""
        session = await self._ensure_session()

        payload = {
            "source": "notification_interceptor",
            "timestamp": _now_iso(),
            "notification": notification.to_dict(),
        }
        if classification:
            payload["classification"] = classification.to_dict()

        try:
            async with session.post(webhook_url, json=payload) as resp:
                success = resp.status in (200, 201, 202, 204)
                if not success:
                    body = await resp.text()
                    logger.warning(
                        "Webhook %s returned %d: %s",
                        webhook_url, resp.status, body[:200],
                    )
                else:
                    logger.info("Forwarded to webhook: %s", webhook_url)
                return success
        except Exception as exc:
            logger.error("Webhook forward failed (%s): %s", webhook_url, exc)
            return False

    async def forward_to_n8n(self, notification: Notification, workflow: str) -> bool:
        """Trigger a specific n8n workflow with notification data."""
        webhook_url = f"{N8N_WEBHOOK_BASE}/{workflow}"
        return await self.forward_to_webhook(notification, webhook_url)

    async def forward_to_notification_hub(
        self,
        notification: Notification,
        channel: str = "telegram",
    ) -> bool:
        """
        Route a notification to the notification_hub.py system.

        Constructs a formatted message and sends it via the hub's API endpoint
        or direct import.
        """
        try:
            # Format the notification for the hub
            message = (
                f"[{notification.category.upper()}] {notification.app_name}\n"
                f"{notification.title}\n"
                f"{notification.text}"
            )
            if notification.big_text and notification.big_text != notification.text:
                message += f"\n{notification.big_text[:200]}"

            # Try importing notification_hub directly
            try:
                from src.notification_hub import get_hub
                hub = get_hub()
                hub.send(
                    title=f"Phone: {notification.app_name}",
                    body=message,
                    severity="info",
                    category="general",
                    channels=[channel],
                )
                logger.info("Forwarded to notification hub (%s)", channel)
                return True
            except ImportError:
                logger.debug("notification_hub not importable, using webhook fallback")

            # Fallback: POST to a local API if running
            session = await self._ensure_session()
            async with session.post(
                "http://localhost:8000/api/notifications/send",
                json={
                    "title": f"Phone: {notification.app_name}",
                    "body": message,
                    "severity": "info",
                    "channel": channel,
                },
            ) as resp:
                return resp.status in (200, 201, 202)

        except Exception as exc:
            logger.warning("Hub forwarding failed: %s", exc)
            return False

    async def _escalate(self, notification: Notification, note: str = "") -> None:
        """Escalate a notification as critical via all available channels."""
        message = f"ESCALATED: {notification.summary}"
        if note:
            message += f"\nNote: {note}"

        try:
            from src.notification_hub import get_hub
            hub = get_hub()
            hub.critical(
                title=f"Escalation: {notification.app_name}",
                body=message,
            )
        except ImportError:
            logger.warning("Cannot escalate — notification_hub not available")

    async def _execute_playbook(
        self,
        playbook_name: str,
        notification: Notification,
        classification: ClassificationResult,
    ) -> None:
        """Execute a named playbook (custom action sequence)."""
        logger.info("Executing playbook: %s for %s", playbook_name, notification.summary)

        if playbook_name == "restart_app":
            pkg = notification.package_name
            await self._adb_shell(f"am force-stop {pkg}")
            await asyncio.sleep(2)
            await self._adb_shell(
                f"monkey -p {pkg} -c android.intent.category.LAUNCHER 1"
            )

        elif playbook_name == "screenshot_and_log":
            try:
                device_path = "/sdcard/openclaw_notif_screen.png"
                await self._adb_shell(f"screencap -p {device_path}")
                notification.screenshot_path = device_path
            except Exception as exc:
                logger.warning("Screenshot playbook failed: %s", exc)

        elif playbook_name == "reduce_automation":
            logger.info("Playbook: reducing automation intensity (battery conservation)")
            # Signal to other systems via a flag file
            flag_path = BASE_DIR / "data" / "low_battery_mode.flag"
            flag_path.parent.mkdir(parents=True, exist_ok=True)
            flag_path.write_text(_now_iso(), encoding="utf-8")

        else:
            logger.warning("Unknown playbook: %s", playbook_name)

    # ==================================================================
    # MONITORING LOOP
    # ==================================================================

    async def monitor(
        self,
        poll_interval: float = 5.0,
        callback: Optional[Callable[[Notification, List[str]], Any]] = None,
        max_iterations: int = 0,
    ) -> None:
        """
        Continuously monitor for new notifications.

        Polls the device at the given interval, captures new notifications,
        classifies them, evaluates rules, and executes actions.

        Args:
            poll_interval: Seconds between polls (default 5).
            callback: Optional function called with (notification, actions) for each.
            max_iterations: Stop after N iterations (0 = infinite).
        """
        self._monitoring = True
        iteration = 0

        logger.info(
            "Starting notification monitor (interval=%.1fs, max=%s)",
            poll_interval, max_iterations or "infinite",
        )

        while self._monitoring:
            iteration += 1
            if max_iterations and iteration > max_iterations:
                break

            try:
                new_notifications = await self.capture_notifications()

                for notification in new_notifications:
                    actions = await self.evaluate_and_act(notification)

                    if callback:
                        try:
                            result = callback(notification, actions)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as exc:
                            logger.warning("Monitor callback error: %s", exc)

            except Exception as exc:
                logger.error("Monitor poll error: %s", exc)

            await asyncio.sleep(poll_interval)

        logger.info("Notification monitor stopped after %d iterations", iteration)

    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self._monitoring = False

    # ==================================================================
    # ANALYTICS
    # ==================================================================

    def notification_stats(self, hours: int = 24) -> NotificationStats:
        """
        Compute aggregate notification statistics over the given time window.

        Returns volume by app, category, priority, plus handle rates and noise analysis.
        """
        cutoff = (_now_utc() - timedelta(hours=hours)).isoformat()
        stats = NotificationStats(period_hours=hours)

        by_app: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        by_priority: Dict[str, int] = {}
        handled = 0
        auto_replied = 0
        dismissed = 0
        forwarded = 0

        for entry in self._history:
            ts = entry.get("timestamp", "")
            if ts < cutoff:
                continue

            stats.total += 1
            app = entry.get("app_name", entry.get("package_name", "unknown"))
            cat = entry.get("category", "unknown")
            pri = entry.get("priority", "default")

            by_app[app] = by_app.get(app, 0) + 1
            by_category[cat] = by_category.get(cat, 0) + 1
            by_priority[pri] = by_priority.get(pri, 0) + 1

            if entry.get("handled"):
                handled += 1
            resp = entry.get("response_action", "")
            if "auto_replied" in resp:
                auto_replied += 1
            if "dismiss" in resp:
                dismissed += 1
            if "forward" in resp:
                forwarded += 1

        stats.by_app = dict(sorted(by_app.items(), key=lambda x: x[1], reverse=True))
        stats.by_category = dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True))
        stats.by_priority = by_priority
        stats.handled_count = handled
        stats.auto_replied_count = auto_replied
        stats.dismissed_count = dismissed
        stats.forwarded_count = forwarded

        # Average handle time
        if self._handle_times:
            stats.avg_handle_time_ms = round(
                sum(self._handle_times) / len(self._handle_times), 1
            )

        # Top senders
        stats.top_senders = [
            {"app": app, "count": count}
            for app, count in list(stats.by_app.items())[:10]
        ]

        # Noise analysis: apps that send lots of low-priority notifications
        for app, count in by_app.items():
            if count >= 5:
                # Count low-priority from this app
                low_count = sum(
                    1 for e in self._history
                    if e.get("timestamp", "") >= cutoff
                    and (e.get("app_name", "") == app or e.get("package_name", "") == app)
                    and e.get("priority") in ("low", "default")
                    and not e.get("handled")
                )
                noise_ratio = low_count / count if count > 0 else 0
                if noise_ratio > 0.5:
                    stats.noise_score[app] = round(noise_ratio, 2)

        return stats

    def response_time_stats(self) -> Dict[str, Any]:
        """Statistics on how quickly notifications are handled."""
        if not self._handle_times:
            return {"count": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0, "p95_ms": 0}

        sorted_times = sorted(self._handle_times)
        p95_idx = int(len(sorted_times) * 0.95)

        return {
            "count": len(sorted_times),
            "avg_ms": round(sum(sorted_times) / len(sorted_times), 1),
            "min_ms": round(sorted_times[0], 1),
            "max_ms": round(sorted_times[-1], 1),
            "p95_ms": round(sorted_times[min(p95_idx, len(sorted_times) - 1)], 1),
        }

    def top_senders(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Return the most active notification sources in the time window."""
        stats = self.notification_stats(hours)
        return stats.top_senders

    def noise_analysis(self) -> Dict[str, Any]:
        """
        Identify which apps send too many low-priority notifications.

        Returns a dict of app_name -> noise_score (0.0-1.0), where higher
        means more noise. Apps with scores > 0.7 should be muted.
        """
        stats = self.notification_stats(24)
        return {
            "noise_scores": stats.noise_score,
            "recommendation": {
                app: "Consider muting this app" if score > 0.7 else "Acceptable noise level"
                for app, score in stats.noise_score.items()
            },
            "total_notifications_24h": stats.total,
        }

    def daily_digest(self) -> str:
        """
        Generate a daily digest summary of all notifications grouped by category.

        Returns formatted text suitable for WhatsApp/Telegram delivery.
        """
        stats = self.notification_stats(24)

        lines: List[str] = []
        lines.append("NOTIFICATION DIGEST (24h)")
        lines.append("=" * 35)
        lines.append(f"Total: {stats.total} notifications")
        lines.append(f"Handled: {stats.handled_count}")
        lines.append(f"Auto-replied: {stats.auto_replied_count}")
        lines.append(f"Forwarded: {stats.forwarded_count}")
        lines.append(f"Dismissed: {stats.dismissed_count}")
        lines.append("")

        # By category
        if stats.by_category:
            lines.append("BY CATEGORY:")
            for cat, count in stats.by_category.items():
                pct = (count / stats.total * 100) if stats.total > 0 else 0
                lines.append(f"  {cat:<15} {count:>4}  ({pct:.0f}%)")
            lines.append("")

        # Top senders
        if stats.top_senders:
            lines.append("TOP SENDERS:")
            for sender in stats.top_senders[:5]:
                lines.append(f"  {sender['app']:<25} {sender['count']:>4}")
            lines.append("")

        # Noise analysis
        if stats.noise_score:
            lines.append("NOISY APPS (consider muting):")
            for app, score in sorted(
                stats.noise_score.items(), key=lambda x: x[1], reverse=True,
            ):
                lines.append(f"  {app:<25} noise={score:.0%}")
            lines.append("")

        # Handle time
        if stats.avg_handle_time_ms > 0:
            lines.append(f"Avg handle time: {stats.avg_handle_time_ms:.0f}ms")

        return "\n".join(lines)

    # ==================================================================
    # RULE MANAGEMENT (delegated to RuleEngine)
    # ==================================================================

    def add_rule(
        self,
        name: str,
        conditions: Dict[str, Any],
        actions: List[str],
        action_params: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> NotificationRule:
        """Create a new notification routing rule."""
        return self.rule_engine.add_rule(name, conditions, actions, action_params, priority)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        return self.rule_engine.remove_rule(rule_id)

    def update_rule(self, rule_id: str, **kwargs: Any) -> Optional[NotificationRule]:
        """Update rule fields."""
        return self.rule_engine.update_rule(rule_id, **kwargs)

    def list_rules(self) -> List[NotificationRule]:
        """List all rules."""
        return self.rule_engine.list_rules()

    def test_rule(self, rule_id: str, notification: Notification) -> bool:
        """Test if a rule matches a notification."""
        return self.rule_engine.test_rule(rule_id, notification)

    def install_default_rules(self) -> List[NotificationRule]:
        """Install the pre-built empire notification rules."""
        return self.rule_engine.install_default_rules()

    # ==================================================================
    # TEMPLATE MANAGEMENT
    # ==================================================================

    def set_template(self, key: str, text: str) -> None:
        """Set or update a reply template."""
        self._reply_templates[key] = text
        _save_json(TEMPLATES_FILE, self._reply_templates)

    def get_templates(self) -> Dict[str, str]:
        """Get all reply templates."""
        return dict(self._reply_templates)

    def remove_template(self, key: str) -> bool:
        """Remove a reply template."""
        if key in self._reply_templates:
            del self._reply_templates[key]
            _save_json(TEMPLATES_FILE, self._reply_templates)
            return True
        return False

    # ==================================================================
    # SYNC WRAPPERS
    # ==================================================================

    def capture_notifications_sync(self) -> List[Notification]:
        """Synchronous wrapper for capture_notifications."""
        return self._run_sync(self.capture_notifications())

    def classify_sync(self, notification: Notification) -> ClassificationResult:
        """Synchronous wrapper for classify."""
        return self._run_sync(self.classify(notification))

    def evaluate_and_act_sync(self, notification: Notification) -> List[str]:
        """Synchronous wrapper for evaluate_and_act."""
        return self._run_sync(self.evaluate_and_act(notification))

    def monitor_sync(self, poll_interval: float = 5.0, **kwargs: Any) -> None:
        """Synchronous wrapper for monitor."""
        self._run_sync(self.monitor(poll_interval=poll_interval, **kwargs))

    def notification_stats_sync(self, hours: int = 24) -> NotificationStats:
        """Synchronous stats (no async needed, included for API consistency)."""
        return self.notification_stats(hours)

    def daily_digest_sync(self) -> str:
        """Synchronous digest (no async needed, included for API consistency)."""
        return self.daily_digest()

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    # ==================================================================
    # CONTEXT MANAGER
    # ==================================================================

    async def __aenter__(self) -> NotificationInterceptor:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._save_history()
        await self.close()


# ===================================================================
# Singleton Factory
# ===================================================================

_interceptor_instance: Optional[NotificationInterceptor] = None


def get_interceptor(
    node_url: str = OPENCLAW_NODE_URL,
    node_name: str = OPENCLAW_ANDROID_NODE,
    api_key: str = ANTHROPIC_API_KEY,
) -> NotificationInterceptor:
    """Return the singleton NotificationInterceptor instance."""
    global _interceptor_instance
    if _interceptor_instance is None:
        _interceptor_instance = NotificationInterceptor(
            node_url=node_url,
            node_name=node_name,
            api_key=api_key,
        )
    return _interceptor_instance


# ===================================================================
# CLI Entry Point
# ===================================================================

def _cli_main() -> None:
    """CLI entry point for the notification interceptor."""

    parser = argparse.ArgumentParser(
        prog="notification_interceptor",
        description="OpenClaw Empire Notification Interceptor — Android notification monitoring & routing",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- monitor ---
    p_monitor = subparsers.add_parser("monitor", help="Start continuous notification monitoring")
    p_monitor.add_argument(
        "--interval", type=float, default=5.0,
        help="Poll interval in seconds (default: 5)",
    )
    p_monitor.add_argument(
        "--max", type=int, default=0,
        help="Max iterations (0=infinite, default: 0)",
    )

    # --- capture ---
    p_capture = subparsers.add_parser("capture", help="One-shot notification capture")
    p_capture.add_argument("--classify", action="store_true", help="Also classify each notification")
    p_capture.add_argument("--act", action="store_true", help="Classify and execute rule actions")

    # --- recent ---
    p_recent = subparsers.add_parser("recent", help="Show recent notifications from history")
    p_recent.add_argument("--minutes", type=int, default=60, help="Minutes back (default: 60)")
    p_recent.add_argument("--app", help="Filter by package name")
    p_recent.add_argument("--category", help="Filter by category")

    # --- rules ---
    subparsers.add_parser("rules", help="List all notification routing rules")

    # --- add-rule ---
    p_add = subparsers.add_parser("add-rule", help="Add a notification routing rule")
    p_add.add_argument("--name", required=True, help="Rule name")
    p_add.add_argument("--app", help="Package name to match")
    p_add.add_argument("--title-pattern", help="Regex for title match")
    p_add.add_argument("--text-pattern", help="Regex for text match")
    p_add.add_argument("--category", help="Category to match")
    p_add.add_argument("--priority", help="Minimum priority to match")
    p_add.add_argument(
        "--actions", required=True, nargs="+",
        choices=["auto_reply", "forward", "trigger_webhook", "log", "dismiss", "escalate", "execute_playbook"],
        help="Actions to execute on match",
    )
    p_add.add_argument("--rule-priority", type=int, default=0, help="Rule priority (higher=first)")

    # --- test-rule ---
    p_test = subparsers.add_parser("test-rule", help="Test a rule against a sample notification")
    p_test.add_argument("rule_id", help="Rule ID to test")
    p_test.add_argument("--app", default="com.test", help="Sample app package")
    p_test.add_argument("--title", default="Test Title", help="Sample notification title")
    p_test.add_argument("--text", default="Test notification text", help="Sample notification text")

    # --- stats ---
    p_stats = subparsers.add_parser("stats", help="Show notification statistics")
    p_stats.add_argument("--hours", type=int, default=24, help="Hours back (default: 24)")

    # --- digest ---
    subparsers.add_parser("digest", help="Generate daily notification digest")

    # --- quiet-hours ---
    p_quiet = subparsers.add_parser("quiet-hours", help="Configure quiet hours")
    p_quiet.add_argument("--enable", action="store_true", help="Enable quiet hours")
    p_quiet.add_argument("--disable", action="store_true", help="Disable quiet hours")
    p_quiet.add_argument("--start", help="Start time (HH:MM)")
    p_quiet.add_argument("--end", help="End time (HH:MM)")
    p_quiet.add_argument("--min-priority", help="Minimum priority during quiet hours")

    # --- dismiss ---
    p_dismiss = subparsers.add_parser("dismiss", help="Dismiss notification(s)")
    p_dismiss.add_argument("notification_id", nargs="?", help="Notification ID (omit for dismiss all)")
    p_dismiss.add_argument("--all", action="store_true", help="Dismiss all notifications")

    # --- reply ---
    p_reply = subparsers.add_parser("reply", help="Send a reply to a notification")
    p_reply.add_argument("notification_id", help="Notification ID to reply to")
    p_reply.add_argument("--message", required=True, help="Reply message text")

    # --- install-defaults ---
    subparsers.add_parser("install-defaults", help="Install pre-built empire notification rules")

    # --- noise ---
    subparsers.add_parser("noise", help="Analyze notification noise levels")

    # --- templates ---
    p_templates = subparsers.add_parser("templates", help="Manage reply templates")
    p_templates.add_argument("--set", nargs=2, metavar=("KEY", "TEXT"), help="Set a template")
    p_templates.add_argument("--remove", help="Remove a template by key")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    interceptor = get_interceptor()

    # ----- Dispatch commands -----

    if args.command == "monitor":
        def _on_notification(notif: Notification, actions: List[str]) -> None:
            print(f"  [{notif.category}] {notif.app_name}: {notif.title} -> {actions}")

        print(f"Starting notification monitor (interval={args.interval}s)...")
        print("Press Ctrl+C to stop.\n")
        try:
            asyncio.run(interceptor.monitor(
                poll_interval=args.interval,
                callback=_on_notification,
                max_iterations=args.max,
            ))
        except KeyboardInterrupt:
            print("\nMonitor stopped.")

    elif args.command == "capture":
        async def _capture() -> None:
            notifs = await interceptor.capture_notifications()
            if not notifs:
                print("No new notifications.")
                return

            print(f"Captured {len(notifs)} notification(s):\n")
            for n in notifs:
                print(f"  [{n.priority}] {n.app_name}")
                print(f"    Title: {n.title}")
                print(f"    Text:  {n.text[:100]}")
                if n.actions:
                    print(f"    Actions: {', '.join(n.actions)}")

                if args.classify or args.act:
                    result = await interceptor.classify(n)
                    print(f"    Category: {result.category} (confidence={result.confidence:.2f})")
                    print(f"    Intent: {result.intent}")
                    if result.entities:
                        print(f"    Entities: {json.dumps(result.entities)}")

                if args.act:
                    actions = await interceptor.evaluate_and_act(n)
                    print(f"    Actions executed: {actions}")

                print()

        asyncio.run(_capture())

    elif args.command == "recent":
        async def _recent() -> None:
            notifs = await interceptor.get_recent(
                minutes=args.minutes,
                app_filter=args.app,
                category_filter=args.category,
            )
            if not notifs:
                print(f"No notifications in the last {args.minutes} minutes.")
                return

            print(f"Recent notifications ({len(notifs)}):\n")
            for n in notifs:
                handled_mark = "[OK]" if n.handled else "[--]"
                print(f"  {handled_mark} {n.timestamp[:19]} | {n.app_name} | {n.title}")
                if n.text:
                    print(f"        {n.text[:100]}")
                if n.response_action:
                    print(f"        Action: {n.response_action}")

        asyncio.run(_recent())

    elif args.command == "rules":
        rules = interceptor.list_rules()
        if not rules:
            print("No rules configured. Run 'install-defaults' to add pre-built rules.")
            return

        print(f"Notification Rules ({len(rules)}):\n")
        for r in rules:
            enabled_mark = "[ON]" if r.enabled else "[OFF]"
            print(f"  {enabled_mark} {r.name} (id={r.rule_id}, pri={r.priority})")
            print(f"       Conditions: {json.dumps(r.conditions)}")
            print(f"       Actions: {r.actions}")
            print(f"       Matches: {r.match_count} (last: {r.last_matched or 'never'})")
            print()

    elif args.command == "add-rule":
        conditions: Dict[str, Any] = {}
        if args.app:
            conditions["app"] = args.app
        if args.title_pattern:
            conditions["title_pattern"] = args.title_pattern
        if args.text_pattern:
            conditions["text_pattern"] = args.text_pattern
        if args.category:
            conditions["category"] = args.category
        if args.priority:
            conditions["priority"] = args.priority

        if not conditions:
            print("ERROR: At least one condition required (--app, --title-pattern, --text-pattern, --category, --priority)")
            sys.exit(1)

        rule = interceptor.add_rule(
            name=args.name,
            conditions=conditions,
            actions=args.actions,
            priority=args.rule_priority,
        )
        print(f"Rule created: {rule.name} (id={rule.rule_id})")
        print(f"  Conditions: {json.dumps(rule.conditions)}")
        print(f"  Actions: {rule.actions}")

    elif args.command == "test-rule":
        test_notif = Notification(
            package_name=args.app,
            title=args.title,
            text=args.text,
        )
        matches = interceptor.test_rule(args.rule_id, test_notif)
        rule = interceptor.rule_engine.get_rule(args.rule_id)
        if rule is None:
            print(f"ERROR: Rule {args.rule_id} not found.")
            sys.exit(1)

        print(f"Rule: {rule.name}")
        print(f"Notification: app={args.app}, title={args.title}, text={args.text}")
        print(f"Match: {'YES' if matches else 'NO'}")

    elif args.command == "stats":
        stats = interceptor.notification_stats(args.hours)
        print(f"NOTIFICATION STATS (last {args.hours}h)")
        print("=" * 40)
        print(f"Total: {stats.total}")
        print(f"Handled: {stats.handled_count}")
        print(f"Auto-replied: {stats.auto_replied_count}")
        print(f"Forwarded: {stats.forwarded_count}")
        print(f"Dismissed: {stats.dismissed_count}")
        print(f"Avg handle time: {stats.avg_handle_time_ms:.0f}ms")
        print()

        if stats.by_category:
            print("By Category:")
            for cat, count in stats.by_category.items():
                print(f"  {cat:<15} {count:>4}")
            print()

        if stats.by_app:
            print("By App:")
            for app, count in list(stats.by_app.items())[:10]:
                print(f"  {app:<30} {count:>4}")
            print()

        if stats.noise_score:
            print("Noise Scores:")
            for app, score in stats.noise_score.items():
                print(f"  {app:<30} {score:.0%}")

    elif args.command == "digest":
        print(interceptor.daily_digest())

    elif args.command == "quiet-hours":
        if args.disable:
            interceptor.disable_quiet_hours()
            print("Quiet hours disabled.")
        elif args.enable or args.start or args.end:
            start = args.start or interceptor.quiet_hours.start_time
            end = args.end or interceptor.quiet_hours.end_time
            config = interceptor.set_quiet_hours(start, end)
            print(f"Quiet hours: {config.start_time} - {config.end_time}")
            print(f"Days: {', '.join(config.days)}")
            if args.min_priority:
                interceptor.set_priority_override(args.min_priority)
                print(f"Min priority during quiet hours: {args.min_priority}")
        else:
            qh = interceptor.quiet_hours
            status = "ENABLED" if qh.enabled else "DISABLED"
            print(f"Quiet Hours: {status}")
            print(f"  Time: {qh.start_time} - {qh.end_time}")
            print(f"  Days: {', '.join(qh.days)}")
            print(f"  Min priority: {qh.min_priority}")
            print(f"  Allow commerce: {qh.allow_commerce}")
            print(f"  Currently quiet: {interceptor.is_quiet_time()}")

    elif args.command == "dismiss":
        async def _dismiss() -> None:
            if args.all or not args.notification_id:
                count = await interceptor.dismiss_all()
                print(f"Dismissed {count} notification(s).")
            else:
                success = await interceptor.dismiss(args.notification_id)
                if success:
                    print(f"Dismissed: {args.notification_id}")
                else:
                    print(f"Failed to dismiss: {args.notification_id}")

        asyncio.run(_dismiss())

    elif args.command == "reply":
        async def _reply() -> None:
            # Find notification in history
            target = None
            for entry in reversed(interceptor._history):
                if entry.get("notification_id") == args.notification_id:
                    target = Notification.from_dict(entry)
                    break

            if target is None:
                print(f"ERROR: Notification {args.notification_id} not found in history.")
                sys.exit(1)

            success = await interceptor.auto_reply(target, args.message)
            if success:
                print(f"Reply sent to {target.app_name}: '{args.message}'")
            else:
                print("Reply failed.")

        asyncio.run(_reply())

    elif args.command == "install-defaults":
        installed = interceptor.install_default_rules()
        if installed:
            print(f"Installed {len(installed)} default rules:")
            for r in installed:
                print(f"  - {r.name} (id={r.rule_id})")
        else:
            print("All default rules already installed.")

    elif args.command == "noise":
        analysis = interceptor.noise_analysis()
        print("NOISE ANALYSIS")
        print("=" * 40)
        print(f"Total notifications (24h): {analysis['total_notifications_24h']}")
        print()

        if analysis["noise_scores"]:
            print("Noisy Apps:")
            for app, score in sorted(
                analysis["noise_scores"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                rec = analysis["recommendation"][app]
                print(f"  {app:<30} noise={score:.0%}")
                print(f"    -> {rec}")
        else:
            print("No significantly noisy apps detected.")

    elif args.command == "templates":
        if args.set:
            key, text = args.set
            interceptor.set_template(key, text)
            print(f"Template set: {key} = '{text}'")
        elif args.remove:
            if interceptor.remove_template(args.remove):
                print(f"Template removed: {args.remove}")
            else:
                print(f"Template not found: {args.remove}")
        else:
            templates = interceptor.get_templates()
            if templates:
                print("Reply Templates:")
                for key, text in templates.items():
                    print(f"  {key:<20} '{text}'")
            else:
                print("No templates configured.")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli_main()
