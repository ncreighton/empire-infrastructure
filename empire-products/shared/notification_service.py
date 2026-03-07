"""
Empire Products -- Notification Service
==========================================
Multi-channel notification delivery for scheduled content,
system alerts, and user engagement campaigns.

Channels:
  - WhatsApp: via Cloud API (primary)
  - Telegram: via Bot API (primary)
  - Email: via SMTP or SendGrid API (secondary)
  - Webhook: for n8n/Zapier integrations

Env vars:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
  SENDGRID_API_KEY
  NOTIFICATION_FROM_EMAIL
  NOTIFICATION_EMAIL_PROVIDER   ("smtp" or "sendgrid", default: "smtp")
  NOTIFICATION_WEBHOOK_URL
  WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_ACCESS_TOKEN
  TELEGRAM_BOT_TOKEN

Usage:
    from shared.notification_service import get_notification_service

    svc = get_notification_service()

    # Send to a specific channel
    await svc.send(Notification(
        user_id=42,
        channel=NotificationChannel.EMAIL,
        subject="Your Weekly Dream Report",
        body="Here is your report...",
        category="dream_report",
    ))

    # Send to user's preferred channel
    await svc.send_to_user(user_id=42, body="Good morning!", category="devotion")

    # Bulk send
    await svc.send_bulk([notif1, notif2, notif3])

    # Daily digest
    await svc.send_daily_digest(user_id=42)

    # Welcome sequence
    await svc.send_welcome_sequence(user_id=42, product_slug="daily_devotion")
"""

import asyncio
import logging
import os
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from html import escape as html_escape
from typing import Optional

import httpx

logger = logging.getLogger("empire.notifications")


# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════

class NotificationChannel(Enum):
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    EMAIL = "email"
    WEBHOOK = "webhook"


class NotificationPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class Notification:
    """A notification to be delivered."""
    user_id: int
    channel: NotificationChannel
    subject: str = ""              # for email
    body: str = ""                 # plain text body
    html_body: str = ""            # for email HTML (auto-generated if empty)
    template: str = ""             # template name (e.g. "daily_devotion_delivery")
    template_data: dict = field(default_factory=dict)
    priority: str = "normal"       # low, normal, high
    category: str = ""             # devotion, dream_report, severing, system, marketing
    document_url: str = ""
    document_filename: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class DeliveryResult:
    """Result of a single notification delivery attempt."""
    notification: Notification
    success: bool
    channel: NotificationChannel
    error: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════
# Templates
# ═══════════════════════════════════════════════════════════════════════

# Built-in notification templates.  Each template has a plain-text body
# and an HTML body variant.  Placeholders use {key} format.

_BUILTIN_TEMPLATES: dict[str, dict[str, str]] = {
    # ── Daily Devotion delivery ──────────────────────────────────
    "daily_devotion_delivery": {
        "subject": "Your Daily Devotion for {date}",
        "body": (
            "Good morning, {name}!\n"
            "\n"
            "Here is your daily devotion for {date}.\n"
            "\n"
            "Moon Phase: {moon_phase}\n"
            "Day Ruler: {day_ruler}\n"
            "\n"
            "{suggestion_type}:\n"
            "{suggestion}\n"
            "\n"
            "Quick Practice:\n"
            "{quick_practice}\n"
            "\n"
            "Journal Prompt:\n"
            "{journal_prompt}\n"
            "\n"
            "Affirmation:\n"
            "{affirmation}\n"
            "\n"
            "Blessed be."
        ),
        "html_body": (
            "<h2>Good morning, {name}!</h2>"
            "<p>Here is your daily devotion for <strong>{date}</strong>.</p>"
            "<table style='border-collapse:collapse;width:100%;max-width:600px;'>"
            "<tr><td style='padding:4px 8px;font-weight:bold;'>Moon Phase</td>"
            "<td style='padding:4px 8px;'>{moon_phase}</td></tr>"
            "<tr><td style='padding:4px 8px;font-weight:bold;'>Day Ruler</td>"
            "<td style='padding:4px 8px;'>{day_ruler}</td></tr>"
            "</table>"
            "<h3>{suggestion_type}</h3>"
            "<p>{suggestion}</p>"
            "<h3>Quick Practice</h3>"
            "<p>{quick_practice}</p>"
            "<h3>Journal Prompt</h3>"
            "<p><em>{journal_prompt}</em></p>"
            "<blockquote style='border-left:3px solid #7c3aed;padding-left:12px;"
            "color:#7c3aed;font-style:italic;'>{affirmation}</blockquote>"
            "<p style='color:#999;font-size:12px;'>Blessed be.</p>"
        ),
    },

    # ── Weekly Dream Report ──────────────────────────────────────
    "dream_weekly_report": {
        "subject": "Your Weekly Dream Report - {week_label}",
        "body": (
            "Hello {name},\n"
            "\n"
            "Here is your dream analysis for the past week ({week_label}).\n"
            "\n"
            "Dreams logged: {dream_count}\n"
            "\n"
            "Top Symbols:\n"
            "{top_symbols}\n"
            "\n"
            "Emotional Landscape:\n"
            "{top_emotions}\n"
            "\n"
            "Recurring Themes:\n"
            "{recurring_themes}\n"
            "\n"
            "Patterns Detected:\n"
            "{patterns}\n"
            "\n"
            "Keep logging your dreams with /dream to deepen your insights.\n"
        ),
        "html_body": (
            "<h2>Weekly Dream Report</h2>"
            "<p>Hello {name}, here is your dream analysis for <strong>{week_label}</strong>.</p>"
            "<p>Dreams logged: <strong>{dream_count}</strong></p>"
            "<h3>Top Symbols</h3><p>{top_symbols_html}</p>"
            "<h3>Emotional Landscape</h3><p>{top_emotions_html}</p>"
            "<h3>Recurring Themes</h3><p>{recurring_themes_html}</p>"
            "<h3>Patterns Detected</h3><p>{patterns_html}</p>"
            "<p style='color:#999;font-size:12px;'>Keep logging your dreams with "
            "<code>/dream</code> to deepen your insights.</p>"
        ),
    },

    # ── Severing day content ─────────────────────────────────────
    "severing_day_N": {
        "subject": "The Severing - Day {day_number}: {day_name}",
        "body": (
            "Day {day_number} of 7: {day_name}\n"
            "\n"
            "{ritual_content}\n"
            "\n"
            "Take your time with today's practice. There is no rush.\n"
            "\n"
            "Progress: [{progress_bar}] Day {day_number}/7\n"
        ),
        "html_body": (
            "<h2>The Severing &mdash; Day {day_number}: {day_name}</h2>"
            "<div style='background:#1a1a2e;color:#e0e0e0;padding:16px;"
            "border-radius:8px;'>{ritual_content_html}</div>"
            "<p style='margin-top:12px;color:#999;'>Take your time with today's "
            "practice. There is no rush.</p>"
            "<p>Progress: <code>[{progress_bar}]</code> Day {day_number}/7</p>"
        ),
    },

    # ── Subscription welcome ─────────────────────────────────────
    "subscription_welcome": {
        "subject": "Welcome to {product_name}!",
        "body": (
            "Welcome to {product_name}!\n"
            "\n"
            "Hi {name}, thank you for subscribing. Here is what you can expect:\n"
            "\n"
            "{product_description}\n"
            "\n"
            "Getting started:\n"
            "{getting_started}\n"
            "\n"
            "Commands you can use:\n"
            "{commands}\n"
            "\n"
            "If you have any questions, just reply to this message.\n"
            "\n"
            "Blessed be."
        ),
        "html_body": (
            "<h2>Welcome to {product_name}!</h2>"
            "<p>Hi {name}, thank you for subscribing.</p>"
            "<p>{product_description}</p>"
            "<h3>Getting Started</h3>"
            "<p>{getting_started_html}</p>"
            "<h3>Commands</h3>"
            "<p>{commands_html}</p>"
            "<p style='color:#999;font-size:12px;'>If you have any questions, "
            "just reply to this message. Blessed be.</p>"
        ),
    },

    # ── Subscription expiring ────────────────────────────────────
    "subscription_expiring": {
        "subject": "Your {product_name} subscription is expiring soon",
        "body": (
            "Hi {name},\n"
            "\n"
            "Your {product_name} subscription is set to expire on {expiry_date}.\n"
            "\n"
            "During your time with us, you have:\n"
            "{usage_summary}\n"
            "\n"
            "To keep your access, no action is needed if auto-renewal is on.\n"
            "If you cancelled, you can resubscribe anytime with /buy {product_slug}.\n"
            "\n"
            "We hope to continue this journey with you.\n"
        ),
        "html_body": (
            "<h2>Your subscription is expiring soon</h2>"
            "<p>Hi {name}, your <strong>{product_name}</strong> subscription "
            "is set to expire on <strong>{expiry_date}</strong>.</p>"
            "<p>During your time with us, you have:</p>"
            "<ul>{usage_summary_html}</ul>"
            "<p>To keep your access, no action is needed if auto-renewal is on. "
            "If you cancelled, you can resubscribe anytime with "
            "<code>/buy {product_slug}</code>.</p>"
        ),
    },

    # ── System alert ─────────────────────────────────────────────
    "system_alert": {
        "subject": "[Empire] System Alert: {alert_title}",
        "body": (
            "SYSTEM ALERT: {alert_title}\n"
            "\n"
            "Severity: {severity}\n"
            "Component: {component}\n"
            "Time: {timestamp}\n"
            "\n"
            "{alert_message}\n"
            "\n"
            "Action required: {action_required}\n"
        ),
        "html_body": (
            "<h2 style='color:#dc2626;'>System Alert: {alert_title}</h2>"
            "<table style='border-collapse:collapse;'>"
            "<tr><td style='padding:4px 8px;font-weight:bold;'>Severity</td>"
            "<td style='padding:4px 8px;'>{severity}</td></tr>"
            "<tr><td style='padding:4px 8px;font-weight:bold;'>Component</td>"
            "<td style='padding:4px 8px;'>{component}</td></tr>"
            "<tr><td style='padding:4px 8px;font-weight:bold;'>Time</td>"
            "<td style='padding:4px 8px;'>{timestamp}</td></tr>"
            "</table>"
            "<p>{alert_message}</p>"
            "<p><strong>Action required:</strong> {action_required}</p>"
        ),
    },

    # ── Weekly digest ────────────────────────────────────────────
    "weekly_digest": {
        "subject": "Your Weekly Empire Digest - {week_label}",
        "body": (
            "Hello {name},\n"
            "\n"
            "Here is your activity summary for the week of {week_label}.\n"
            "\n"
            "Active Products:\n"
            "{active_products}\n"
            "\n"
            "This Week's Activity:\n"
            "{activity_summary}\n"
            "\n"
            "Highlights:\n"
            "{highlights}\n"
            "\n"
            "Upcoming:\n"
            "{upcoming}\n"
            "\n"
            "Thank you for being part of the Empire.\n"
        ),
        "html_body": (
            "<h2>Your Weekly Empire Digest</h2>"
            "<p>Hello {name}, here is your activity summary for the week of "
            "<strong>{week_label}</strong>.</p>"
            "<h3>Active Products</h3>{active_products_html}"
            "<h3>This Week's Activity</h3>{activity_summary_html}"
            "<h3>Highlights</h3>{highlights_html}"
            "<h3>Upcoming</h3>{upcoming_html}"
            "<p style='color:#999;font-size:12px;'>Thank you for being part "
            "of the Empire.</p>"
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Email wrapper (plain HTML layout)
# ═══════════════════════════════════════════════════════════════════════

_EMAIL_WRAPPER = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
         Helvetica, Arial, sans-serif; line-height: 1.6; color: #333;
         max-width: 640px; margin: 0 auto; padding: 20px; }}
  h2 {{ color: #7c3aed; }}
  h3 {{ color: #4c1d95; margin-top: 24px; }}
  blockquote {{ border-left: 3px solid #7c3aed; padding-left: 12px;
               color: #7c3aed; font-style: italic; margin: 16px 0; }}
  code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px;
         font-size: 0.9em; }}
  table {{ margin: 12px 0; }}
  td {{ vertical-align: top; }}
  .footer {{ margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e7eb;
            color: #999; font-size: 12px; }}
</style>
</head>
<body>
{content}
<div class="footer">
  <p>Sent by Empire Products &mdash; <a href="https://openclaw.ai">openclaw.ai</a></p>
  <p>To change your notification preferences, reply with /settings.</p>
</div>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════
# NotificationService
# ═══════════════════════════════════════════════════════════════════════

class NotificationService:
    """Delivers notifications across channels with retry and templating."""

    MAX_RETRIES = 2
    RETRY_DELAY_SECONDS = 2.0

    def __init__(self):
        self._templates: dict[str, dict[str, str]] = {}
        self._load_templates()

    # ─── Public API ────────────────────────────────────────────

    async def send(self, notification: Notification) -> DeliveryResult:
        """Send a single notification. Routes to the appropriate channel."""
        # Render template if specified
        if notification.template:
            rendered = self._render_template(
                notification.template, notification.template_data
            )
            if rendered:
                if not notification.body:
                    notification.body = rendered.get("body", "")
                if not notification.html_body:
                    notification.html_body = rendered.get("html_body", "")
                if not notification.subject:
                    notification.subject = rendered.get("subject", "")

        # Dispatch to channel handler
        success = False
        error = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                if notification.channel == NotificationChannel.WHATSAPP:
                    success = await self._send_whatsapp(notification)
                elif notification.channel == NotificationChannel.TELEGRAM:
                    success = await self._send_telegram(notification)
                elif notification.channel == NotificationChannel.EMAIL:
                    success = await self._send_email(notification)
                elif notification.channel == NotificationChannel.WEBHOOK:
                    success = await self._send_webhook(notification)
                else:
                    error = f"Unknown channel: {notification.channel}"
                    break

                if success:
                    break
            except Exception as e:
                error = f"Attempt {attempt} failed: {e}"
                logger.warning(
                    f"Notification send attempt {attempt}/{self.MAX_RETRIES} "
                    f"failed for user {notification.user_id} via "
                    f"{notification.channel.value}: {e}"
                )
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.RETRY_DELAY_SECONDS * attempt)

        if success:
            logger.info(
                f"Notification sent to user {notification.user_id} via "
                f"{notification.channel.value} (category={notification.category})"
            )
            self._log_notification(notification, success=True)
        else:
            logger.error(
                f"Notification failed for user {notification.user_id} via "
                f"{notification.channel.value}: {error}"
            )
            self._log_notification(notification, success=False, error=error)

        return DeliveryResult(
            notification=notification,
            success=success,
            channel=notification.channel,
            error=error if not success else "",
        )

    async def send_bulk(
        self, notifications: list[Notification], concurrency: int = 5
    ) -> dict:
        """Send multiple notifications efficiently with bounded concurrency.

        Returns a summary dict with counts and individual results.
        """
        if not notifications:
            return {"total": 0, "sent": 0, "failed": 0, "results": []}

        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded_send(notif: Notification) -> DeliveryResult:
            async with semaphore:
                return await self.send(notif)

        results = await asyncio.gather(
            *[_bounded_send(n) for n in notifications],
            return_exceptions=True,
        )

        sent = 0
        failed = 0
        delivery_results = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                delivery_results.append(DeliveryResult(
                    notification=notifications[i],
                    success=False,
                    channel=notifications[i].channel,
                    error=str(result),
                ))
            elif isinstance(result, DeliveryResult):
                delivery_results.append(result)
                if result.success:
                    sent += 1
                else:
                    failed += 1
            else:
                failed += 1

        logger.info(
            f"Bulk notification complete: {sent} sent, {failed} failed "
            f"out of {len(notifications)}"
        )

        return {
            "total": len(notifications),
            "sent": sent,
            "failed": failed,
            "results": delivery_results,
        }

    async def send_to_user(
        self,
        user_id: int,
        body: str,
        subject: str = "",
        category: str = "",
        document_url: str = "",
        template: str = "",
        template_data: dict = None,
        priority: str = "normal",
    ) -> dict:
        """Send to a user via their registered platform channel.

        Resolves the user's platform from the database and sends
        through that channel. If the user has an email registered and
        the platform channel fails, falls back to email.
        """
        from shared.user_manager import get_user_manager
        um = get_user_manager()
        user = um.get_user_by_id(user_id)

        if not user:
            logger.error(f"send_to_user: user {user_id} not found")
            return {"success": False, "error": "User not found"}

        # Determine primary channel from user platform
        platform = user.get("platform", "")
        channel_map = {
            "whatsapp": NotificationChannel.WHATSAPP,
            "telegram": NotificationChannel.TELEGRAM,
        }
        primary_channel = channel_map.get(platform)

        if not primary_channel:
            # Try email as fallback
            email = user.get("email") or _get_user_email(um, user_id)
            if email:
                primary_channel = NotificationChannel.EMAIL
            else:
                return {"success": False, "error": f"No deliverable channel for user {user_id}"}

        notification = Notification(
            user_id=user_id,
            channel=primary_channel,
            subject=subject,
            body=body,
            template=template,
            template_data=template_data or {},
            priority=priority,
            category=category,
            document_url=document_url,
        )

        result = await self.send(notification)

        # If primary channel failed and user has email, try email fallback
        if not result.success and primary_channel != NotificationChannel.EMAIL:
            email = user.get("email") or _get_user_email(um, user_id)
            if email:
                logger.info(
                    f"Primary channel {primary_channel.value} failed for user "
                    f"{user_id}, falling back to email"
                )
                fallback = Notification(
                    user_id=user_id,
                    channel=NotificationChannel.EMAIL,
                    subject=subject or f"Notification from Empire Products",
                    body=body,
                    template=template,
                    template_data=template_data or {},
                    priority=priority,
                    category=category,
                    document_url=document_url,
                )
                result = await self.send(fallback)

        return {
            "success": result.success,
            "channel": result.channel.value,
            "error": result.error,
        }

    # ─── Digest & Sequence Methods ────────────────────────────

    async def send_daily_digest(self, user_id: int) -> dict:
        """Send a daily digest email summarizing the user's activity.

        Queries the past 24 hours of usage, subscriptions, and
        product-specific data, then renders the weekly_digest template
        (adapted for daily use).
        """
        from shared.user_manager import get_user_manager
        um = get_user_manager()
        user = um.get_user_by_id(user_id)

        if not user:
            return {"success": False, "error": "User not found"}

        name = user.get("display_name") or "there"
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        date_label = now.strftime("%B %d, %Y")

        # Gather activity data
        usage_rows = um._execute(
            """SELECT product_slug, action, COUNT(*) as cnt
               FROM empire_usage_log
               WHERE user_id = %s AND created_at >= %s
               GROUP BY product_slug, action
               ORDER BY cnt DESC""",
            (user_id, yesterday),
        )

        # Active subscriptions
        subscriptions = um.get_user_subscriptions(user_id)

        # Build template data
        active_products_lines = []
        for sub in subscriptions:
            slug = sub.get("product_slug", "unknown")
            status = sub.get("status", "unknown")
            active_products_lines.append(f"  - {slug} ({status})")
        active_products = "\n".join(active_products_lines) if active_products_lines else "  No active subscriptions"

        activity_lines = []
        for row in usage_rows:
            slug = row.get("product_slug", "")
            action = row.get("action", "")
            cnt = row.get("cnt", 0)
            activity_lines.append(f"  - {slug}: {action} x{cnt}")
        activity_summary = "\n".join(activity_lines) if activity_lines else "  No activity recorded"

        # Dream count (if applicable)
        dream_count_rows = um._execute(
            """SELECT COUNT(*) as cnt FROM empire_dream_log
               WHERE user_id = %s AND dream_date >= %s""",
            (user_id, yesterday.date()),
        )
        dream_count = dream_count_rows[0]["cnt"] if dream_count_rows else 0

        highlights_parts = []
        if dream_count > 0:
            highlights_parts.append(f"  - Logged {dream_count} dream(s)")
        total_actions = sum(r.get("cnt", 0) for r in usage_rows)
        if total_actions > 0:
            highlights_parts.append(f"  - {total_actions} total interactions")
        highlights = "\n".join(highlights_parts) if highlights_parts else "  Keep engaging to build your highlights!"

        template_data = {
            "name": name,
            "week_label": date_label,
            "active_products": active_products,
            "active_products_html": _text_list_to_html(active_products_lines),
            "activity_summary": activity_summary,
            "activity_summary_html": _text_list_to_html(activity_lines),
            "highlights": highlights,
            "highlights_html": _text_list_to_html(highlights_parts),
            "upcoming": "  Check /help for available commands and features.",
            "upcoming_html": "<p>Check <code>/help</code> for available commands and features.</p>",
        }

        return await self.send_to_user(
            user_id=user_id,
            body="",  # template will fill this
            subject=f"Your Daily Empire Digest - {date_label}",
            category="digest",
            template="weekly_digest",
            template_data=template_data,
        )

    async def send_welcome_sequence(
        self, user_id: int, product_slug: str
    ) -> dict:
        """Send welcome/onboarding notification for a new subscriber.

        Looks up the product configuration and renders the
        subscription_welcome template with product-specific content.
        """
        from shared.user_manager import get_user_manager
        from shared.subscription_engine import get_subscription_engine

        um = get_user_manager()
        engine = get_subscription_engine()

        user = um.get_user_by_id(user_id)
        if not user:
            return {"success": False, "error": "User not found"}

        product = engine.get_product_config(product_slug)
        if not product:
            return {"success": False, "error": f"Unknown product: {product_slug}"}

        name = user.get("display_name") or "there"
        product_name = product.get("name", product_slug)
        description = product.get("description", "")

        # Build getting-started instructions per product type
        getting_started_map = {
            "daily_devotion": (
                "1. Your daily practice will be delivered each morning\n"
                "2. Use /devotion to request today's practice anytime\n"
                "3. Set your preferred time with /settings time HH:MM"
            ),
            "dream_oracle": (
                "1. Log a dream by sending /dream followed by your dream text\n"
                "2. View patterns with /dream patterns\n"
                "3. Weekly reports are sent every Sunday"
            ),
            "the_severing": (
                "1. Your 7-day ritual sequence begins tomorrow\n"
                "2. Each day's content will be delivered automatically\n"
                "3. Take your time -- there is no rush"
            ),
            "spell_forge": (
                "1. Request a spell with /spell followed by your intention\n"
                "2. Each spell is uniquely crafted for you\n"
                "3. Use /spell types to see available categories"
            ),
            "the_familiar": (
                "1. Just start chatting -- I am your magical companion\n"
                "2. Ask questions about witchcraft, tarot, or astrology\n"
                "3. I learn your preferences over time"
            ),
            "altar_eye": (
                "1. Send a photo of your altar for analysis\n"
                "2. I will identify items and suggest improvements\n"
                "3. Get personalized recommendations"
            ),
            "intuition_engine": (
                "1. Draw cards with /tarot\n"
                "2. Build your personal meaning dictionary over time\n"
                "3. Use /tarot meanings to review your interpretations"
            ),
        }

        getting_started = getting_started_map.get(
            product_slug,
            "Use /help to see all available commands for this product.",
        )

        # Commands relevant to this product
        commands_map = {
            "daily_devotion": "/devotion, /settings",
            "dream_oracle": "/dream, /dream patterns, /dream report",
            "the_severing": "/severing, /severing progress",
            "spell_forge": "/spell, /spell types",
            "the_familiar": "Just chat naturally!",
            "altar_eye": "Send a photo of your altar",
            "intuition_engine": "/tarot, /tarot meanings",
        }
        commands = commands_map.get(product_slug, "/help")

        template_data = {
            "name": name,
            "product_name": product_name,
            "product_slug": product_slug,
            "product_description": description,
            "getting_started": getting_started,
            "getting_started_html": getting_started.replace("\n", "<br>"),
            "commands": commands,
            "commands_html": f"<code>{html_escape(commands)}</code>",
        }

        return await self.send_to_user(
            user_id=user_id,
            body="",
            subject=f"Welcome to {product_name}!",
            category="welcome",
            template="subscription_welcome",
            template_data=template_data,
            priority="high",
        )

    # ─── Channel implementations ──────────────────────────────

    async def _send_whatsapp(self, notification: Notification) -> bool:
        """Send notification via WhatsApp Cloud API.

        Mirrors the approach in api/webhooks.py _send_whatsapp_message
        but works from a Notification object rather than OutgoingMessage.
        """
        phone_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        access_token = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")

        if not phone_id or not access_token:
            raise RuntimeError("WhatsApp credentials not configured")

        # Resolve user's WhatsApp phone number
        platform_user_id = self._resolve_platform_user_id(notification.user_id, "whatsapp")
        if not platform_user_id:
            raise RuntimeError(f"No WhatsApp ID found for user {notification.user_id}")

        url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        text = notification.body
        if not text:
            raise RuntimeError("WhatsApp notification has no body text")

        # Handle documents
        if notification.document_url:
            payload = {
                "messaging_product": "whatsapp",
                "to": platform_user_id,
                "type": "document",
                "document": {
                    "link": notification.document_url,
                    "filename": notification.document_filename or "document.pdf",
                    "caption": text[:1024],
                },
            }
        else:
            # Split long messages (WhatsApp 4096 char limit)
            if len(text) > 4096:
                return await self._send_whatsapp_chunked(
                    url, headers, platform_user_id, text
                )

            payload = {
                "messaging_product": "whatsapp",
                "to": platform_user_id,
                "type": "text",
                "text": {"preview_url": True, "body": text},
            }

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"WhatsApp API error {resp.status_code}: {resp.text[:300]}"
                )

        return True

    async def _send_whatsapp_chunked(
        self, url: str, headers: dict, to: str, text: str
    ) -> bool:
        """Send a long WhatsApp message in chunks."""
        chunks = _split_message(text, 4096)
        async with httpx.AsyncClient() as client:
            for i, chunk in enumerate(chunks):
                payload = {
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "text",
                    "text": {"preview_url": i == 0, "body": chunk},
                }
                resp = await client.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"WhatsApp chunk {i+1} failed: {resp.status_code} "
                        f"{resp.text[:200]}"
                    )
        return True

    async def _send_telegram(self, notification: Notification) -> bool:
        """Send notification via Telegram Bot API.

        Mirrors the approach in api/webhooks.py _send_telegram_message
        but works from a Notification object.
        """
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not bot_token:
            raise RuntimeError("Telegram bot token not configured")

        chat_id = self._resolve_platform_user_id(notification.user_id, "telegram")
        if not chat_id:
            raise RuntimeError(f"No Telegram ID found for user {notification.user_id}")

        base_url = f"https://api.telegram.org/bot{bot_token}"
        text = notification.body

        if not text:
            raise RuntimeError("Telegram notification has no body text")

        async with httpx.AsyncClient() as client:
            # Handle documents
            if notification.document_url:
                doc_payload = {
                    "chat_id": chat_id,
                    "document": notification.document_url,
                    "caption": text[:1024],
                    "parse_mode": "HTML",
                }
                if notification.document_filename:
                    doc_payload["filename"] = notification.document_filename

                resp = await client.post(
                    f"{base_url}/sendDocument", json=doc_payload, timeout=30
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Telegram document error {resp.status_code}: "
                        f"{resp.text[:300]}"
                    )
                return True

            # Split long messages (Telegram 4096 char limit)
            if len(text) > 4096:
                chunks = _split_message(text, 4096)
                for i, chunk in enumerate(chunks):
                    payload = {
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                    }
                    resp = await client.post(
                        f"{base_url}/sendMessage", json=payload, timeout=30
                    )
                    if resp.status_code != 200:
                        raise RuntimeError(
                            f"Telegram chunk {i+1} failed: {resp.status_code} "
                            f"{resp.text[:200]}"
                        )
                return True

            # Normal message
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
            }
            resp = await client.post(
                f"{base_url}/sendMessage", json=payload, timeout=30
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Telegram API error {resp.status_code}: {resp.text[:300]}"
                )

        return True

    async def _send_email(self, notification: Notification) -> bool:
        """Send email via SMTP or SendGrid depending on configuration."""
        provider = os.environ.get("NOTIFICATION_EMAIL_PROVIDER", "smtp").lower()
        from_email = os.environ.get(
            "NOTIFICATION_FROM_EMAIL", "noreply@openclaw.ai"
        )

        # Resolve recipient email
        to_email = self._resolve_user_email(notification.user_id)
        if not to_email:
            raise RuntimeError(
                f"No email address found for user {notification.user_id}"
            )

        subject = notification.subject or "Notification from Empire Products"
        body_text = notification.body
        body_html = notification.html_body

        # Auto-generate HTML from plain text if no HTML provided
        if not body_html and body_text:
            body_html = _plain_text_to_html(body_text)

        # Wrap HTML in the email template
        if body_html:
            body_html = _EMAIL_WRAPPER.format(content=body_html)

        if provider == "sendgrid":
            return await self._send_email_sendgrid(
                from_email, to_email, subject, body_text, body_html
            )
        else:
            return await self._send_email_smtp(
                from_email, to_email, subject, body_text, body_html
            )

    async def _send_email_smtp(
        self,
        from_email: str,
        to_email: str,
        subject: str,
        body_text: str,
        body_html: str,
    ) -> bool:
        """Send email via SMTP (smtplib). Runs in a thread to avoid blocking."""
        host = os.environ.get("SMTP_HOST", "localhost")
        port = int(os.environ.get("SMTP_PORT", "587"))
        user = os.environ.get("SMTP_USER", "")
        password = os.environ.get("SMTP_PASSWORD", "")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        if body_text:
            msg.attach(MIMEText(body_text, "plain", "utf-8"))
        if body_html:
            msg.attach(MIMEText(body_html, "html", "utf-8"))

        def _smtp_send():
            with smtplib.SMTP(host, port, timeout=30) as server:
                if port in (587, 465):
                    server.starttls()
                if user and password:
                    server.login(user, password)
                server.send_message(msg)

        # Run blocking SMTP in a thread pool
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _smtp_send)

        logger.info(f"Email sent via SMTP to {to_email}: {subject}")
        return True

    async def _send_email_sendgrid(
        self,
        from_email: str,
        to_email: str,
        subject: str,
        body_text: str,
        body_html: str,
    ) -> bool:
        """Send email via SendGrid API v3 (httpx POST)."""
        api_key = os.environ.get("SENDGRID_API_KEY", "")
        if not api_key:
            raise RuntimeError("SENDGRID_API_KEY not configured")

        content = []
        if body_text:
            content.append({"type": "text/plain", "value": body_text})
        if body_html:
            content.append({"type": "text/html", "value": body_html})

        if not content:
            raise RuntimeError("Email has no content (text or HTML)")

        payload = {
            "personalizations": [
                {"to": [{"email": to_email}], "subject": subject}
            ],
            "from": {"email": from_email},
            "content": content,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            # SendGrid returns 202 on success
            if resp.status_code not in (200, 201, 202):
                raise RuntimeError(
                    f"SendGrid API error {resp.status_code}: {resp.text[:300]}"
                )

        logger.info(f"Email sent via SendGrid to {to_email}: {subject}")
        return True

    async def _send_webhook(self, notification: Notification) -> bool:
        """Send webhook notification to configured URL (n8n/Zapier)."""
        webhook_url = os.environ.get("NOTIFICATION_WEBHOOK_URL", "")
        if not webhook_url:
            raise RuntimeError("NOTIFICATION_WEBHOOK_URL not configured")

        payload = {
            "event": "notification",
            "user_id": notification.user_id,
            "channel": notification.channel.value,
            "category": notification.category,
            "priority": notification.priority,
            "subject": notification.subject,
            "body": notification.body,
            "document_url": notification.document_url,
            "document_filename": notification.document_filename,
            "metadata": notification.metadata,
            "template": notification.template,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Webhook error {resp.status_code}: {resp.text[:300]}"
                )

        logger.info(
            f"Webhook notification sent for user {notification.user_id} "
            f"(category={notification.category})"
        )
        return True

    # ─── Templating ───────────────────────────────────────────

    def _render_template(
        self, template_name: str, data: dict
    ) -> Optional[dict[str, str]]:
        """Render a notification template with the provided data.

        Returns a dict with 'subject', 'body', and 'html_body' keys,
        or None if the template is not found.

        Missing placeholders are replaced with empty strings rather
        than raising an error, so partial data is handled gracefully.
        """
        template = self._templates.get(template_name)
        if not template:
            logger.warning(f"Template not found: {template_name}")
            return None

        result = {}
        for key in ("subject", "body", "html_body"):
            raw = template.get(key, "")
            try:
                result[key] = _safe_format(raw, data)
            except Exception as e:
                logger.warning(
                    f"Template render error for {template_name}.{key}: {e}"
                )
                result[key] = raw

        return result

    def _load_templates(self):
        """Load built-in notification templates."""
        self._templates = dict(_BUILTIN_TEMPLATES)
        logger.debug(f"Loaded {len(self._templates)} notification templates")

    def register_template(
        self, name: str, subject: str, body: str, html_body: str = ""
    ):
        """Register a custom notification template at runtime."""
        self._templates[name] = {
            "subject": subject,
            "body": body,
            "html_body": html_body,
        }
        logger.info(f"Registered custom template: {name}")

    # ─── User resolution helpers ──────────────────────────────

    def _resolve_platform_user_id(
        self, user_id: int, platform: str
    ) -> Optional[str]:
        """Look up a user's platform-specific ID (phone number or chat_id)."""
        try:
            from shared.user_manager import get_user_manager
            um = get_user_manager()
            user = um.get_user_by_id(user_id)
            if user and user.get("platform") == platform:
                return user.get("platform_user_id")
            # If user is on a different platform, check for secondary registrations
            # (future: multi-platform identity linking)
            return None
        except Exception as e:
            logger.error(f"Failed to resolve platform user ID: {e}")
            return None

    def _resolve_user_email(self, user_id: int) -> Optional[str]:
        """Look up a user's email address for email delivery."""
        try:
            from shared.user_manager import get_user_manager
            um = get_user_manager()
            user = um.get_user_by_id(user_id)
            if not user:
                return None
            # Check email field directly
            email = user.get("email")
            if email:
                return email
            # Check preferences for stored email
            return _get_user_email(um, user_id)
        except Exception as e:
            logger.error(f"Failed to resolve user email: {e}")
            return None

    # ─── Notification logging ─────────────────────────────────

    def _log_notification(
        self, notification: Notification, success: bool, error: str = ""
    ):
        """Log notification delivery to the database for analytics.

        This is best-effort -- failures here should not affect delivery.
        """
        try:
            from shared.user_manager import get_user_manager
            um = get_user_manager()
            um._execute(
                """INSERT INTO empire_notification_log
                   (user_id, channel, category, template, priority, success, error_message)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    notification.user_id,
                    notification.channel.value,
                    notification.category,
                    notification.template,
                    notification.priority,
                    success,
                    error[:500] if error else "",
                ),
                fetch=False,
            )
        except Exception as e:
            # Best-effort logging -- do not let this break notification delivery
            logger.debug(f"Failed to log notification: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════

def _get_user_email(um, user_id: int) -> Optional[str]:
    """Retrieve email from user preferences JSONB."""
    email = um.get_preference(user_id, "email")
    if email and isinstance(email, str) and "@" in email:
        return email
    return None


def _split_message(text: str, max_length: int = 4096) -> list[str]:
    """Split a long message into chunks, preserving line boundaries.

    Same algorithm as api/webhooks.py _split_message for consistency.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Find a good split point (prefer newline, then space)
        split_at = text.rfind("\n", 0, max_length)
        if split_at < max_length // 2:
            split_at = text.rfind(" ", 0, max_length)
        if split_at < max_length // 2:
            split_at = max_length

        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()

    return chunks


def _plain_text_to_html(text: str) -> str:
    """Convert plain text to basic HTML, preserving newlines and indentation."""
    escaped = html_escape(text)
    # Convert double newlines to paragraph breaks
    paragraphs = escaped.split("\n\n")
    html_parts = []
    for para in paragraphs:
        # Convert single newlines to <br> within a paragraph
        lines = para.split("\n")
        html_parts.append("<p>" + "<br>\n".join(lines) + "</p>")
    return "\n".join(html_parts)


def _text_list_to_html(items: list[str]) -> str:
    """Convert a list of text items to an HTML unordered list."""
    if not items:
        return "<p>None</p>"
    li_items = []
    for item in items:
        # Strip leading "  - " formatting from plain text
        cleaned = item.strip()
        if cleaned.startswith("- "):
            cleaned = cleaned[2:]
        li_items.append(f"<li>{html_escape(cleaned)}</li>")
    return "<ul>" + "".join(li_items) + "</ul>"


def _safe_format(template: str, data: dict) -> str:
    """Format a template string, replacing missing keys with empty strings.

    Unlike str.format(), this does not raise KeyError for missing keys.
    """

    class DefaultDict(dict):
        def __missing__(self, key):
            return ""

    try:
        return template.format_map(DefaultDict(data))
    except (ValueError, IndexError):
        # Fallback: return the raw template if formatting fails entirely
        return template


# ═══════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════

_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Get or create the NotificationService singleton."""
    global _service
    if _service is None:
        _service = NotificationService()
    return _service
