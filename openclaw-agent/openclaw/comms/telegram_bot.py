"""OpenClaw Telegram Bot — real-time notifications + full command center.

Serves two purposes:
    1. **Notification sink** — receives ALL webhook events and pushes them to Telegram
    2. **Command center** — execute any daemon/VibeCoder/empire command via chat

Uses python-telegram-bot v21+ async Application.  Runs as a coroutine
alongside HeartbeatDaemon in asyncio.gather().

Environment:
    TELEGRAM_COMMANDER_TOKEN  — Bot token
    TELEGRAM_ADMIN_IDS        — Comma-separated admin user IDs (default: 8246744420)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Callable

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TELEGRAM_TOKEN: str = os.getenv("OPENCLAW_TELEGRAM_TOKEN", "") or os.getenv("TELEGRAM_COMMANDER_TOKEN", "")
ADMIN_IDS: list[int] = [
    int(x.strip())
    for x in os.getenv("TELEGRAM_ADMIN_IDS", "8246744420").split(",")
    if x.strip()
]


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------


def admin_only(func: Callable) -> Callable:
    """Restrict handler to admin users only."""

    @functools.wraps(func)
    async def wrapper(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args: Any, **kwargs: Any
    ) -> Any:
        user = update.effective_user
        if not user or user.id not in ADMIN_IDS:
            uid = user.id if user else "unknown"
            logger.warning("Unauthorized Telegram access from user %s", uid)
            if update.message:
                await update.message.reply_text("Access denied.")
            elif update.callback_query:
                await update.callback_query.answer("Access denied.", show_alert=True)
            return
        return await func(update, context, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MD_SPECIAL = re.compile(r"([_*\[\]()~`>#+=|{}.!\\-])")


def escape_md(text: str) -> str:
    """Escape MarkdownV2 special characters."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def truncate(text: str, max_len: int = 80) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def _get_msg(update: Update) -> Any:
    """Extract reply target from update (works for both messages and callbacks)."""
    if update.message:
        return update.message
    if update.callback_query:
        return update.callback_query.message
    return None


def _ts_within_minutes(timestamp: Any, minutes: int) -> bool:
    """Return True if *timestamp* falls within the last *minutes* minutes.

    Accepts ISO-8601 strings, datetime objects, or Unix float timestamps.
    Returns False on any parse failure.
    """
    if timestamp is None:
        return False
    try:
        if isinstance(timestamp, datetime):
            ts_dt = timestamp
        elif isinstance(timestamp, (int, float)):
            ts_dt = datetime.fromtimestamp(float(timestamp))
        else:
            raw = str(timestamp).replace("Z", "+00:00")
            try:
                ts_dt = datetime.fromisoformat(raw)
            except ValueError:
                # Try truncated ISO format e.g. "2026-03-14 12:34:56"
                ts_dt = datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S")
        # Make naive comparison safe: strip tzinfo if present
        if ts_dt.tzinfo is not None:
            from datetime import timezone
            now = datetime.now(tz=timezone.utc)
        else:
            now = datetime.now()
        delta = now - ts_dt
        return delta.total_seconds() <= minutes * 60
    except Exception:
        return False


async def safe_send(
    target: Any,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
    edit: bool = False,
) -> None:
    """Send with MarkdownV2, fallback to plain text."""
    try:
        if edit:
            await target.edit_message_text(
                text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup,
            )
        else:
            await target.reply_text(
                text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup,
            )
    except Exception:
        plain = re.sub(r"[\\*_`\[\]()~>#+=|{}.!-]", "", text)
        try:
            if edit:
                await target.edit_message_text(plain, reply_markup=reply_markup)
            else:
                await target.reply_text(plain, reply_markup=reply_markup)
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", e)


# ---------------------------------------------------------------------------
# Event → emoji mapping
# ---------------------------------------------------------------------------

EVENT_EMOJI: dict[str, str] = {
    "signup_started": "\U0001f680",              # rocket
    "signup_completed": "\u2705",                # check
    "signup_failed": "\u274c",                   # cross
    "captcha_needed": "\U0001f916",              # robot
    "captcha_solved": "\U0001f513",              # unlocked
    "email_verification_needed": "\U0001f4e7",   # email
    "email_verified": "\U0001f4ec",              # mailbox
    "profile_scored": "\U0001f4ca",              # chart
    "batch_started": "\U0001f3c1",               # flag
    "batch_completed": "\U0001f3c6",             # trophy
    "sync_completed": "\U0001f504",              # arrows
    "error": "\U0001f6a8",                       # alert
    "mission_queued": "\U0001f4cb",              # clipboard
    "mission_started": "\u2699\ufe0f",           # gear
    "mission_completed": "\u2705",               # check
    "mission_failed": "\U0001f4a5",              # boom
    "mission_deployed": "\U0001f680",            # rocket
    "project_discovered": "\U0001f50d",          # magnifier
    "health_check": "\U0001f3e5",                # hospital
    "alert": "\U0001f514",                       # bell
    "proactive_action": "\U0001f9e0",            # brain
    "cron_executed": "\u23f0",                   # alarm clock
    "daemon_started": "\u25b6\ufe0f",            # play
    "daemon_stopped": "\u23f9\ufe0f",            # stop
    "apply_profile_started": "\U0001f3a8",       # artist palette
    "apply_profile_completed": "\u2705",         # check
    "apply_profile_failed": "\u274c",            # cross
    "human_activity_started": "\U0001f6b6",      # walking
    "human_activity_completed": "\u2705",        # check
    "publish_started": "\U0001f4e4",             # outbox
    "publish_completed": "\U0001f389",           # party popper
    "publish_failed": "\u274c",                  # cross
    "profile_enhanced": "\u2b06\ufe0f",          # up arrow
    "approval_needed": "\U0001f514",             # bell
}


def _event_icon(event_type: str) -> str:
    return EVENT_EMOJI.get(event_type, "\U0001f4ac")  # speech bubble default


# ---------------------------------------------------------------------------
# Keyboards
# ---------------------------------------------------------------------------


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 Status", callback_data="cmd_status"),
            InlineKeyboardButton("🏥 Health", callback_data="cmd_health"),
        ],
        [
            InlineKeyboardButton("👤 Accounts", callback_data="cmd_accounts"),
            InlineKeyboardButton("📋 Profiles", callback_data="cmd_profiles"),
        ],
        [
            InlineKeyboardButton("🚶 Activity", callback_data="cmd_activity"),
            InlineKeyboardButton("🔔 Pending", callback_data="cmd_pending"),
        ],
        [
            InlineKeyboardButton("⚙️ Missions", callback_data="cmd_missions"),
            InlineKeyboardButton("🚨 Alerts", callback_data="cmd_alerts"),
        ],
        [
            InlineKeyboardButton("💰 Costs", callback_data="cmd_costs"),
            InlineKeyboardButton("⏰ Crons", callback_data="cmd_crons"),
        ],
        [
            InlineKeyboardButton("📡 Live", callback_data="cmd_live"),
            InlineKeyboardButton("📄 Report", callback_data="cmd_report"),
        ],
    ])


# ---------------------------------------------------------------------------
# OpenClawTelegramBot
# ---------------------------------------------------------------------------


class OpenClawTelegramBot:
    """Telegram bot for OpenClaw notifications + command center.

    Usage::

        bot = OpenClawTelegramBot(engine)
        # As coroutine in asyncio.gather:
        await bot.run()

        # Push notification from anywhere:
        await bot.notify("mission_completed", {"title": "Fix bug", "project": "myproj"})
    """

    def __init__(self, engine: Any):
        self.engine = engine
        self._app: Application | None = None
        self._running = False
        self._started_at: datetime | None = None

    async def run(self) -> None:
        """Start the bot (runs until stopped)."""
        if not TELEGRAM_TOKEN:
            logger.warning("TELEGRAM_COMMANDER_TOKEN not set — Telegram bot disabled")
            return

        self._app = (
            Application.builder()
            .token(TELEGRAM_TOKEN)
            .build()
        )

        self._register_handlers(self._app)
        self._app.post_init = self._post_init

        self._running = True
        self._started_at = datetime.now()
        logger.info("OpenClaw Telegram bot starting (polling)")

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
        )

        # Keep running until stopped
        try:
            while self._running:
                await asyncio.sleep(1)
        finally:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("OpenClaw Telegram bot stopped")

    async def stop(self) -> None:
        self._running = False

    async def notify(self, event_type: str, data: dict[str, Any]) -> None:
        """Push a notification to all admin users."""
        if not self._app or not self._running:
            return

        icon = _event_icon(event_type)
        label = event_type.replace("_", " ").title()

        lines = [f"{icon} *{escape_md(label)}*"]

        # Build details from data
        for key, val in data.items():
            if key in ("timestamp", "source"):
                continue
            display_key = key.replace("_", " ").title()
            lines.append(f"  {escape_md(display_key)}: {escape_md(truncate(str(val), 120))}")

        text = "\n".join(lines)

        for uid in ADMIN_IDS:
            try:
                await self._app.bot.send_message(
                    uid, text, parse_mode=ParseMode.MARKDOWN_V2,
                )
            except Exception:
                plain = re.sub(r"[\\*_`\[\]()~>#+=|{}.!-]", "", text)
                try:
                    await self._app.bot.send_message(uid, plain)
                except Exception as e:
                    logger.error("Failed to notify admin %s: %s", uid, e)

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    @staticmethod
    async def _post_init(app: Application) -> None:
        """Register commands with Telegram after init."""
        commands = [
            BotCommand("start", "Main menu"),
            BotCommand("help", "Show all commands"),
            BotCommand("status", "Daemon status"),
            BotCommand("health", "Empire health checks"),
            BotCommand("alerts", "Recent alerts"),
            BotCommand("missions", "VibeCoder missions"),
            BotCommand("projects", "Registered projects"),
            BotCommand("costs", "Model routing costs"),
            BotCommand("dashboard", "Dashboard stats"),
            BotCommand("crons", "Cron job list"),
            BotCommand("vibe", "Submit VibeCoder mission"),
            BotCommand("mute", "Mute notifications (minutes)"),
            BotCommand("unmute", "Resume notifications"),
            BotCommand("accounts", "All accounts with status breakdown"),
            BotCommand("profiles", "Profile quality scores"),
            BotCommand("activity", "Recent human activity sessions"),
            BotCommand("pending", "Actions pending approval"),
            BotCommand("approve", "Approve a pending action"),
            BotCommand("deny", "Deny a pending action"),
            BotCommand("live", "What is happening right now"),
            BotCommand("report", "Comprehensive daily report"),
            BotCommand("fleet", "GoLogin browser identity fleet"),
        ]
        await app.bot.set_my_commands(commands)

    def _register_handlers(self, app: Application) -> None:
        """Register all command + callback handlers."""
        commands = [
            ("start", self._cmd_start),
            ("help", self._cmd_help),
            ("status", self._cmd_status),
            ("health", self._cmd_health),
            ("alerts", self._cmd_alerts),
            ("missions", self._cmd_missions),
            ("projects", self._cmd_projects),
            ("costs", self._cmd_costs),
            ("dashboard", self._cmd_dashboard),
            ("crons", self._cmd_crons),
            ("vibe", self._cmd_vibe),
            ("mute", self._cmd_mute),
            ("unmute", self._cmd_unmute),
            ("accounts", self._cmd_accounts),
            ("profiles", self._cmd_profiles),
            ("activity", self._cmd_activity),
            ("pending", self._cmd_pending),
            ("approve", self._cmd_approve),
            ("deny", self._cmd_deny),
            ("live", self._cmd_live),
            ("report", self._cmd_report),
            ("fleet", self._cmd_fleet),
        ]
        for name, handler in commands:
            app.add_handler(CommandHandler(name, handler))

        # Callback query handler for inline buttons
        app.add_handler(CallbackQueryHandler(self._callback_handler, pattern=r"^cmd_"))
        app.add_handler(CallbackQueryHandler(self._mission_callback, pattern=r"^mission_"))
        app.add_handler(CallbackQueryHandler(self._approval_callback, pattern=r"^approve_"))
        app.add_handler(CallbackQueryHandler(self._approval_callback, pattern=r"^deny_"))

    # -------------------------------------------------------------------
    # Command handlers
    # -------------------------------------------------------------------

    @admin_only
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        text = "*OpenClaw Command Center*\nSelect an action:"
        await safe_send(msg, text, reply_markup=main_menu())

    @admin_only
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        lines = [
            "*OpenClaw Commands*\n",
            "*Monitoring*",
            "/status \\- Daemon status \\+ uptime",
            "/health \\- Empire health checks",
            "/alerts \\- Recent alerts",
            "/live \\- What is happening right now",
            "/report \\- Comprehensive daily report",
            "",
            "*Accounts \\& Profiles*",
            "/accounts \\- All accounts with status breakdown",
            "/profiles \\- Profile quality scores",
            "/activity \\- Recent human activity sessions",
            "/pending \\- Actions pending approval",
            "/approve \\<platform\\> \\- Approve a pending action",
            "/deny \\<platform\\> \\- Deny a pending action",
            "",
            "*VibeCoder*",
            "/missions \\- VibeCoder mission list",
            "/projects \\- Registered projects",
            "/vibe \\<title\\> \\| \\<description\\> \\- Submit mission",
            "",
            "*Infrastructure*",
            "/costs \\- Model routing cost report",
            "/dashboard \\- Aggregate stats",
            "/crons \\- Cron job schedule",
            "",
            "*Notifications*",
            "/mute \\<minutes\\> \\- Mute notifications",
            "/unmute \\- Resume notifications",
        ]
        await safe_send(msg, "\n".join(lines))

    @admin_only
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        await msg.chat.send_action("typing")
        daemon = getattr(self.engine, "_daemon", None)
        if daemon:
            status = daemon.get_status()
            running = status.get("running", False)
            icon = "\u25b6\ufe0f" if running else "\u23f9\ufe0f"
            uptime = status.get("uptime_seconds", 0)
            hours = int(uptime // 3600)
            mins = int((uptime % 3600) // 60)
            tier_runs = status.get("tier_runs", {})

            lines = [
                f"{icon} *Daemon {'Running' if running else 'Stopped'}*",
                f"Uptime: {hours}h {mins}m",
                "",
                "*Tier Runs:*",
            ]
            for tier, count in tier_runs.items():
                lines.append(f"  {escape_md(tier)}: {count}")
        else:
            lines = ["\u23f9\ufe0f *Daemon not initialized*"]

        await safe_send(msg, "\n".join(lines))

    @admin_only
    async def _cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        await msg.chat.send_action("typing")
        try:
            checks = self.engine.codex.get_recent_checks(limit=20)
            if not checks:
                await msg.reply_text("No health checks recorded yet.")
                return

            lines = ["*Recent Health Checks*\n"]
            for c in checks[:15]:
                result = c.get("result", "unknown")
                name = c.get("check_name", "?")
                icon = {
                    "healthy": "\U0001f7e2",
                    "degraded": "\U0001f7e1",
                    "down": "\U0001f534",
                }.get(result, "\u26aa")
                detail = truncate(c.get("message", ""), 60)
                lines.append(f"{icon} {escape_md(name)}: {escape_md(detail)}")

            await safe_send(msg, "\n".join(lines))
        except Exception as e:
            await msg.reply_text(f"Health check error: {e}")

    @admin_only
    async def _cmd_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        await msg.chat.send_action("typing")
        try:
            alerts = self.engine.codex.get_alerts(limit=10)
            if not alerts:
                await msg.reply_text("No alerts.")
                return

            lines = ["*Recent Alerts*\n"]
            for a in alerts[:10]:
                sev = a.get("severity", "info")
                icon = {
                    "critical": "\U0001f534",
                    "warning": "\U0001f7e1",
                    "info": "\U0001f535",
                }.get(sev, "\u26aa")
                title = truncate(a.get("title", "?"), 60)
                ack = "\u2705" if a.get("acknowledged") else ""
                lines.append(f"{icon} {escape_md(title)} {ack}")

            await safe_send(msg, "\n".join(lines))
        except Exception as e:
            await msg.reply_text(f"Alerts error: {e}")

    @admin_only
    async def _cmd_missions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        await msg.chat.send_action("typing")
        vibe = getattr(self.engine, "vibecoder", None)
        if not vibe:
            await msg.reply_text("VibeCoder not initialized.")
            return

        try:
            missions = vibe.list_missions(limit=10)
            if not missions:
                await msg.reply_text("No missions.")
                return

            lines = ["*VibeCoder Missions*\n"]
            for m in missions:
                status_icon = {
                    "queued": "\U0001f4cb",
                    "executing": "\u2699\ufe0f",
                    "reviewing": "\U0001f50d",
                    "completed": "\u2705",
                    "failed": "\u274c",
                    "deployed": "\U0001f680",
                    "paused": "\u23f8\ufe0f",
                }.get(m.status, "\u2753")
                title = truncate(m.title, 40)
                lines.append(
                    f"{status_icon} `{escape_md(m.mission_id[:8])}` {escape_md(title)}"
                )

            keyboard = None
            if missions and missions[0].status in ("failed",):
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton(
                        "Retry Latest", callback_data=f"mission_retry_{missions[0].mission_id}",
                    )]
                ])

            await safe_send(msg, "\n".join(lines), reply_markup=keyboard)
        except Exception as e:
            await msg.reply_text(f"Missions error: {e}")

    @admin_only
    async def _cmd_projects(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        await msg.chat.send_action("typing")
        vibe = getattr(self.engine, "vibecoder", None)
        if not vibe:
            await msg.reply_text("VibeCoder not initialized.")
            return

        try:
            projects = vibe.list_projects()
            if not projects:
                await msg.reply_text("No registered projects.")
                return

            lines = ["*Registered Projects*\n"]
            for p in projects:
                lines.append(f"\U0001f4c1 `{escape_md(p.project_id)}` \\- {escape_md(p.root_path)}")

            await safe_send(msg, "\n".join(lines))
        except Exception as e:
            await msg.reply_text(f"Projects error: {e}")

    @admin_only
    async def _cmd_costs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        await msg.chat.send_action("typing")
        vibe = getattr(self.engine, "vibecoder", None)
        if not vibe:
            await msg.reply_text("VibeCoder not initialized.")
            return

        try:
            report = vibe.model_router.spend_report()
            total_cost = report.get("total_cost", 0)
            total_calls = report.get("total_calls", 0)
            lines = [
                "*Model Router Costs*\n",
                f"Total Spend: ${escape_md(f'{total_cost:.4f}')}",
                f"Calls: {total_calls}",
            ]
            by_model = report.get("by_model", {})
            for model, data in by_model.items():
                cost = data.get("cost", 0)
                calls = data.get("calls", 0)
                cost_str = escape_md(f"{cost:.4f}")
                lines.append(f"  {escape_md(model)}: ${cost_str} \\({calls} calls\\)")

            await safe_send(msg, "\n".join(lines))
        except Exception as e:
            await msg.reply_text(f"Costs error: {e}")

    @admin_only
    async def _cmd_dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        await msg.chat.send_action("typing")
        try:
            stats = self.engine.get_dashboard()
            lines = [
                "*Dashboard*\n",
                f"Accounts: {stats.total_accounts}",
                f"Active: {stats.active_accounts}",
                f"Pending: {stats.pending_signups}",
                f"Categories: {stats.categories_covered}",
            ]
            await safe_send(msg, "\n".join(lines))
        except Exception as e:
            await msg.reply_text(f"Dashboard error: {e}")

    @admin_only
    async def _cmd_crons(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        await msg.chat.send_action("typing")
        daemon = getattr(self.engine, "_daemon", None)
        if not daemon:
            await msg.reply_text("Daemon not initialized.")
            return

        try:
            jobs = daemon.cron.get_all()
            if not jobs:
                await msg.reply_text("No cron jobs.")
                return

            lines = ["*Cron Jobs*\n"]
            for j in jobs:
                icon = "\u25b6\ufe0f" if j.enabled else "\u23f8\ufe0f"
                lines.append(f"{icon} {escape_md(j.name)} \\- {escape_md(j.schedule)}")

            await safe_send(msg, "\n".join(lines))
        except Exception as e:
            await msg.reply_text(f"Cron error: {e}")

    @admin_only
    async def _cmd_vibe(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Submit a VibeCoder mission: /vibe <title> | <description>"""
        msg = _get_msg(update)
        if not msg:
            return
        if not context.args:
            await msg.reply_text(
                "Usage: /vibe <title> | <description>\n"
                "Example: /vibe Fix login bug | The login form crashes on empty email"
            )
            return

        raw = " ".join(context.args)
        if "|" in raw:
            title, description = raw.split("|", 1)
            title = title.strip()
            description = description.strip()
        else:
            title = raw.strip()
            description = title

        vibe = getattr(self.engine, "vibecoder", None)
        if not vibe:
            await msg.reply_text("VibeCoder not initialized.")
            return

        await msg.chat.send_action("typing")
        try:
            mission = vibe.submit_mission(
                project_id="openclaw-agent",
                title=title,
                description=description,
            )
            await msg.reply_text(
                f"\U0001f4cb Mission queued: {mission.mission_id[:8]}\n"
                f"Title: {title}\n"
                f"Status: {mission.status}"
            )
        except Exception as e:
            await msg.reply_text(f"Mission submit failed: {e}")

    @admin_only
    async def _cmd_mute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Mute notifications for N minutes."""
        msg = _get_msg(update)
        if not msg:
            return
        minutes = 30
        if context.args:
            try:
                minutes = int(context.args[0])
            except ValueError:
                pass

        self._muted_until = time.time() + (minutes * 60)
        await msg.reply_text(f"Notifications muted for {minutes} minutes.")

    @admin_only
    async def _cmd_unmute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = _get_msg(update)
        if not msg:
            return
        self._muted_until = 0
        await msg.reply_text("Notifications resumed.")

    @admin_only
    async def _cmd_accounts(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show all accounts with status breakdown."""
        msg = update.message or (update.callback_query and update.callback_query.message)
        if msg:
            await msg.chat.send_action("typing")
        try:
            stats = self.engine.codex.get_stats()
            accounts = self.engine.codex.get_all_accounts()

            lines = ["*Accounts Overview*\n"]

            # Summary row
            total = stats.get("total_accounts", 0)
            active = stats.get("active_accounts", 0)
            pending = stats.get("pending_accounts", 0)
            failed = stats.get("failed_accounts", 0)
            lines.append(
                f"Total: {total} \\| "
                f"\u2705 {active} active \\| "
                f"\u23f3 {pending} pending \\| "
                f"\u274c {failed} failed\n"
            )

            if not accounts:
                lines.append("_No accounts recorded yet\\._")
            else:
                STATUS_ICON = {
                    "active": "\u2705",
                    "pending": "\u23f3",
                    "failed": "\u274c",
                    "incomplete": "\U0001f4dd",
                }
                shown = accounts[:20]
                for acc in shown:
                    platform = truncate(acc.get("platform_name") or acc.get("platform_id", "?"), 30)
                    status = acc.get("status", "unknown")
                    icon = STATUS_ICON.get(status, "\u26aa")
                    updated = acc.get("updated_at", "")
                    if updated:
                        try:
                            updated = str(updated)[:10]
                        except Exception:
                            updated = ""
                    lines.append(f"{icon} {escape_md(platform)} \\- {escape_md(updated)}")

                if len(accounts) > 20:
                    lines.append(f"\n_\\.\\.\\. and {len(accounts) - 20} more_")

            target = update.message if update.message else update.callback_query
            await safe_send(target, "\n".join(lines))
        except Exception as e:
            target = update.message if update.message else update.callback_query
            if target:
                await target.reply_text(f"Accounts error: {e}")

    @admin_only
    async def _cmd_profiles(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show profile quality scores."""
        msg = update.message or (update.callback_query and update.callback_query.message)
        if msg:
            await msg.chat.send_action("typing")
        try:
            profiles = self.engine.codex.get_all_profiles()

            lines = ["*Profile Quality Scores*\n"]

            if not profiles:
                lines.append("_No profiles recorded yet\\._")
            else:
                GRADE_ICON = {
                    "A": "\U0001f7e2",
                    "B": "\U0001f7e1",
                    "C": "\U0001f7e0",
                    "D": "\U0001f534",
                    "F": "\U0001f534",
                }
                shown = profiles[:20]
                for prof in shown:
                    platform = truncate(
                        prof.get("platform_name") or prof.get("platform_id", "?"), 30
                    )
                    grade = prof.get("grade", "?")
                    score = prof.get("sentinel_score", 0)
                    icon = GRADE_ICON.get(grade, "\u26aa")
                    updated = prof.get("updated_at", "")
                    if updated:
                        try:
                            updated = str(updated)[:10]
                        except Exception:
                            updated = ""
                    lines.append(
                        f"{icon} {escape_md(platform)} \\- "
                        f"Grade {escape_md(str(grade))} "
                        f"\\({escape_md(str(score))}\\) "
                        f"{escape_md(updated)}"
                    )

                if len(profiles) > 20:
                    lines.append(f"\n_\\.\\.\\. and {len(profiles) - 20} more_")

            target = update.message if update.message else update.callback_query
            await safe_send(target, "\n".join(lines))
        except Exception as e:
            target = update.message if update.message else update.callback_query
            if target:
                await target.reply_text(f"Profiles error: {e}")

    @admin_only
    async def _cmd_activity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show recent human_activity sessions."""
        msg = update.message or (update.callback_query and update.callback_query.message)
        if msg:
            await msg.chat.send_action("typing")
        try:
            history = self.engine.codex.get_action_history(limit=50)
            sessions = [
                h for h in history if h.get("action_type") == "human_activity"
            ][:10]

            lines = ["*Recent Activity Sessions*\n"]

            if not sessions:
                lines.append("_No human activity sessions recorded yet\\._")
            else:
                for s in sessions:
                    platform = truncate(s.get("target", "?"), 25)
                    result = s.get("result", "?")
                    result_icon = "\u2705" if result == "success" else (
                        "\u274c" if result == "failed" else "\u23f3"
                    )
                    ts = truncate(str(s.get("timestamp", "")), 16)
                    desc = truncate(s.get("description", ""), 40)
                    lines.append(
                        f"{result_icon} {escape_md(platform)} \\- "
                        f"{escape_md(desc)} \\({escape_md(ts)}\\)"
                    )

            target = update.message if update.message else update.callback_query
            await safe_send(target, "\n".join(lines))
        except Exception as e:
            target = update.message if update.message else update.callback_query
            if target:
                await target.reply_text(f"Activity error: {e}")

    @admin_only
    async def _cmd_pending(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show pending approval actions with inline approve/deny buttons."""
        msg = update.message or (update.callback_query and update.callback_query.message)
        if msg:
            await msg.chat.send_action("typing")
        try:
            history = self.engine.codex.get_action_history(limit=100)
            pending = [h for h in history if h.get("result") == "pending_approval"]

            lines = ["*Pending Approvals*\n"]

            if not pending:
                lines.append("_No actions pending approval\\._")
                target = update.message if update.message else update.callback_query
                await safe_send(target, "\n".join(lines))
                return

            # Build one message per pending action with inline buttons
            for action in pending:
                action_type = action.get("action_type", "?")
                target_id = action.get("target", "?")
                desc = truncate(action.get("description", ""), 60)
                text = (
                    f"\U0001f514 *Pending Approval*\n"
                    f"Action: {escape_md(action_type)}\n"
                    f"Target: {escape_md(target_id)}\n"
                    f"Desc: {escape_md(desc)}"
                )
                keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton(
                            "Approve",
                            callback_data=f"approve_{action_type}_{target_id}",
                        ),
                        InlineKeyboardButton(
                            "Deny",
                            callback_data=f"deny_{action_type}_{target_id}",
                        ),
                    ]
                ])
                target = update.message if update.message else update.callback_query
                await safe_send(target, text, reply_markup=keyboard)

        except Exception as e:
            target = update.message if update.message else update.callback_query
            if target:
                await target.reply_text(f"Pending error: {e}")

    @admin_only
    async def _cmd_approve(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Approve a pending action: /approve <platform>"""
        msg = _get_msg(update)
        if not msg:
            return
        if not context.args:
            await msg.reply_text("Usage: /approve <platform>")
            return
        platform = context.args[0].strip()
        try:
            history = self.engine.codex.get_action_history(limit=100)
            pending = [
                h for h in history
                if h.get("result") == "pending_approval" and h.get("target") == platform
            ]
            if not pending:
                await msg.reply_text(f"No pending action found for platform: {platform}")
                return
            action = pending[0]
            self.engine.codex.log_action(
                action_type=action.get("action_type", "approve"),
                target=platform,
                description=f"Approved via Telegram: {action.get('description', '')}",
                result="approved",
                autonomous=False,
            )
            await msg.reply_text(f"\u2705 Approved action for {platform}")
        except Exception as e:
            await msg.reply_text(f"Approve error: {e}")

    @admin_only
    async def _cmd_deny(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Deny a pending action: /deny <platform>"""
        msg = _get_msg(update)
        if not msg:
            return
        if not context.args:
            await msg.reply_text("Usage: /deny <platform>")
            return
        platform = context.args[0].strip()
        try:
            history = self.engine.codex.get_action_history(limit=100)
            pending = [
                h for h in history
                if h.get("result") == "pending_approval" and h.get("target") == platform
            ]
            if not pending:
                await msg.reply_text(f"No pending action found for platform: {platform}")
                return
            action = pending[0]
            self.engine.codex.log_action(
                action_type=action.get("action_type", "deny"),
                target=platform,
                description=f"Denied via Telegram: {action.get('description', '')}",
                result="denied",
                autonomous=False,
            )
            await msg.reply_text(f"\u274c Denied action for {platform}")
        except Exception as e:
            await msg.reply_text(f"Deny error: {e}")

    @admin_only
    async def _cmd_live(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show what is happening right now."""
        msg = update.message or (update.callback_query and update.callback_query.message)
        if msg:
            await msg.chat.send_action("typing")
        try:
            lines = ["*\U0001f4e1 Live Status*\n"]

            # Daemon uptime
            daemon = getattr(self.engine, "_daemon", None)
            if daemon:
                status = daemon.get_status()
                running = status.get("running", False)
                icon = "\u25b6\ufe0f" if running else "\u23f9\ufe0f"
                uptime = status.get("uptime_seconds", 0)
                hours = int(uptime // 3600)
                mins = int((uptime % 3600) // 60)
                lines.append(f"{icon} Daemon {'Running' if running else 'Stopped'} \\- {hours}h {mins}m uptime")
            else:
                lines.append("\u23f9\ufe0f Daemon not initialized")

            lines.append("")

            # Recent proactive actions (last 15 min)
            try:
                now = datetime.now().timestamp()
                history = self.engine.codex.get_action_history(limit=50)
                recent = [
                    h for h in history
                    if h.get("autonomous") and _ts_within_minutes(h.get("timestamp"), 15)
                ]
                lines.append(f"*Recent Actions \\(15m\\):* {len(recent)}")
                for h in recent[:5]:
                    action_type = truncate(h.get("action_type", "?"), 25)
                    target = truncate(h.get("target", ""), 20)
                    result = h.get("result", "?")
                    r_icon = "\u2705" if result == "success" else (
                        "\u274c" if result in ("failed", "error") else "\u23f3"
                    )
                    lines.append(f"  {r_icon} {escape_md(action_type)} \u2192 {escape_md(target)}")
            except Exception:
                lines.append("_Could not load recent actions_")

            lines.append("")

            # Next cron jobs
            if daemon:
                try:
                    jobs = daemon.cron.get_all()
                    enabled_jobs = [j for j in jobs if j.enabled][:3]
                    if enabled_jobs:
                        lines.append("*Next Cron Jobs:*")
                        for j in enabled_jobs:
                            lines.append(f"  \u23f0 {escape_md(j.name)} \\- {escape_md(j.schedule)}")
                except Exception:
                    pass

            lines.append("")

            # Pending approvals count
            try:
                history = self.engine.codex.get_action_history(limit=100)
                pending_count = sum(1 for h in history if h.get("result") == "pending_approval")
                if pending_count:
                    lines.append(f"\U0001f514 *Pending Approvals:* {pending_count} \\(use /pending\\)")
            except Exception:
                pass

            # Recent errors
            try:
                history = self.engine.codex.get_action_history(limit=50)
                errors = [
                    h for h in history
                    if h.get("result") in ("failed", "error")
                ][:5]
                if errors:
                    lines.append("")
                    lines.append("*Recent Errors:*")
                    for e in errors:
                        action_type = truncate(e.get("action_type", "?"), 25)
                        target = truncate(e.get("target", ""), 20)
                        lines.append(f"  \u274c {escape_md(action_type)} \u2192 {escape_md(target)}")
            except Exception:
                pass

            target = update.message if update.message else update.callback_query
            await safe_send(target, "\n".join(lines))
        except Exception as e:
            target = update.message if update.message else update.callback_query
            if target:
                await target.reply_text(f"Live error: {e}")

    @admin_only
    async def _cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comprehensive daily report."""
        msg = update.message or (update.callback_query and update.callback_query.message)
        if msg:
            await msg.chat.send_action("typing")
        try:
            lines = ["*\U0001f4c4 Daily Report*\n"]

            # Account status breakdown
            try:
                stats = self.engine.codex.get_stats()
                accounts = self.engine.codex.get_all_accounts()
                status_counts: dict[str, int] = {}
                for acc in accounts:
                    s = acc.get("status", "unknown")
                    status_counts[s] = status_counts.get(s, 0) + 1
                lines.append("*Accounts by Status:*")
                for s, count in status_counts.items():
                    lines.append(f"  {escape_md(s)}: {count}")
                lines.append("")
            except Exception:
                lines.append("_Account stats unavailable_\n")

            # Profile quality distribution
            try:
                profiles = self.engine.codex.get_all_profiles()
                grade_counts: dict[str, int] = {}
                for p in profiles:
                    g = p.get("grade", "?")
                    grade_counts[g] = grade_counts.get(g, 0) + 1
                if grade_counts:
                    lines.append("*Profile Grade Distribution:*")
                    for grade in ("A", "B", "C", "D", "F"):
                        if grade in grade_counts:
                            lines.append(f"  Grade {grade}: {grade_counts[grade]}")
                    lines.append("")
            except Exception:
                lines.append("_Profile stats unavailable_\n")

            # Action stats for last 24h
            try:
                history = self.engine.codex.get_action_history(limit=500)
                last24 = [h for h in history if _ts_within_minutes(h.get("timestamp"), 1440)]

                activity_sessions = [
                    h for h in last24 if h.get("action_type") == "human_activity"
                ]
                lines.append(f"*Activity Sessions \\(24h\\):* {len(activity_sessions)}")

                signups = [h for h in last24 if h.get("action_type") == "signup"]
                s_ok = sum(1 for h in signups if h.get("result") == "success")
                s_fail = sum(1 for h in signups if h.get("result") in ("failed", "error"))
                lines.append(
                    f"*Signups \\(24h\\):* {len(signups)} attempted \\| "
                    f"\u2705 {s_ok} \\| \u274c {s_fail}"
                )

                apply_profile = [
                    h for h in last24 if h.get("action_type") == "apply_profile"
                ]
                lines.append(f"*Profiles Applied \\(24h\\):* {len(apply_profile)}")

                publish = [h for h in last24 if h.get("action_type") == "publish"]
                lines.append(f"*Content Published \\(24h\\):* {len(publish)}")
                lines.append("")
            except Exception:
                lines.append("_Action stats unavailable_\n")

            # VibeCoder missions last 24h
            try:
                vibe = getattr(self.engine, "vibecoder", None)
                if vibe:
                    missions = vibe.list_missions(limit=100)
                    vibe_24h = [
                        m for m in missions
                        if _ts_within_minutes(getattr(m, "created_at", None), 1440)
                    ]
                    lines.append(f"*VibeCoder Missions \\(24h\\):* {len(vibe_24h)}")
                    if vibe_24h:
                        s_done = sum(1 for m in vibe_24h if m.status == "completed")
                        s_fail = sum(1 for m in vibe_24h if m.status == "failed")
                        lines.append(f"  \u2705 {s_done} completed \\| \u274c {s_fail} failed")
                    lines.append("")
            except Exception:
                pass

            # Model costs last 24h
            try:
                vibe = getattr(self.engine, "vibecoder", None)
                if vibe:
                    report = vibe.model_router.spend_report()
                    total_cost = report.get("total_cost", 0)
                    total_calls = report.get("total_calls", 0)
                    lines.append(
                        f"*Model Costs \\(all time\\):* "
                        f"${escape_md(f'{total_cost:.4f}')} over {total_calls} calls"
                    )
            except Exception:
                pass

            target = update.message if update.message else update.callback_query
            await safe_send(target, "\n".join(lines))
        except Exception as e:
            target = update.message if update.message else update.callback_query
            if target:
                await target.reply_text(f"Report error: {e}")

    @admin_only
    async def _cmd_fleet(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """GoLogin browser identity fleet status."""
        try:
            from openclaw.browser.identity_manager import IdentityManager

            im = IdentityManager()
            stats = im.stats()

            lines = [
                "*\U0001f30d GoLogin Identity Fleet*\n",
                f"*Dedicated Profiles:* {stats['dedicated_profiles']}",
                f"*Pool Profiles:* {stats['pool_profiles']}",
                f"*Total:* {stats['total_profiles']}\n",
                "*Dedicated Assignments:*",
            ]
            for pid in stats["dedicated_platforms"]:
                a = im.resolve(pid)
                if a:
                    lines.append(
                        f"  \u2022 {escape_md(pid)}: "
                        f"`{escape_md(a.profile_id[:12])}\\.\\.\\."
                        f"` \\({escape_md(a.profile_name)}\\)"
                    )

            lines.append("\n*Pool Coverage:*")
            lines.append(
                f"  {46 - stats['dedicated_profiles']} platforms share "
                f"{stats['pool_profiles']} pooled profiles"
            )

            target = update.message if update.message else update.callback_query
            await safe_send(target, "\n".join(lines))
        except Exception as e:
            target = update.message if update.message else update.callback_query
            if target:
                await target.reply_text(f"Fleet error: {e}")

    # -------------------------------------------------------------------
    # Callback handlers
    # -------------------------------------------------------------------

    @admin_only
    async def _callback_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        await query.answer()

        cmd = query.data.replace("cmd_", "")
        handler_map = {
            "status": self._cmd_status,
            "health": self._cmd_health,
            "alerts": self._cmd_alerts,
            "missions": self._cmd_missions,
            "projects": self._cmd_projects,
            "costs": self._cmd_costs,
            "dashboard": self._cmd_dashboard,
            "crons": self._cmd_crons,
            "accounts": self._cmd_accounts,
            "profiles": self._cmd_profiles,
            "activity": self._cmd_activity,
            "pending": self._cmd_pending,
            "live": self._cmd_live,
            "report": self._cmd_report,
        }

        handler = handler_map.get(cmd)
        if handler:
            # Re-create a fake message context for the handler
            await handler(update, context)

    @admin_only
    async def _mission_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        await query.answer()

        data = query.data
        vibe = getattr(self.engine, "vibecoder", None)
        if not vibe:
            await query.edit_message_text("VibeCoder not available.")
            return

        if data.startswith("mission_retry_"):
            mission_id = data.replace("mission_retry_", "")
            try:
                vibe.retry_mission(mission_id)
                await query.edit_message_text(f"Mission {mission_id[:8]} queued for retry.")
            except Exception as e:
                await query.edit_message_text(f"Retry failed: {e}")

    @admin_only
    async def _approval_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle inline approve_/deny_ callback buttons from /pending."""
        query = update.callback_query
        await query.answer()

        data = query.data  # e.g. "approve_human_activity_gumroad" or "deny_signup_etsy"
        if data.startswith("approve_"):
            decision = "approved"
            rest = data[len("approve_"):]
        elif data.startswith("deny_"):
            decision = "denied"
            rest = data[len("deny_"):]
        else:
            await query.edit_message_text("Unknown action.")
            return

        # rest = "<action_type>_<target>" — split on first underscore that precedes platform name
        # We stored it as approve_{action_type}_{target}, so split at first "_" after prefix
        parts = rest.split("_", 1)
        if len(parts) < 2:
            action_type = rest
            target = rest
        else:
            action_type, target = parts[0], parts[1]

        try:
            self.engine.codex.log_action(
                action_type=action_type,
                target=target,
                description=f"{decision.capitalize()} via Telegram inline button",
                result=decision,
                autonomous=False,
            )
            icon = "\u2705" if decision == "approved" else "\u274c"
            await query.edit_message_text(
                f"{icon} Action *{escape_md(action_type)}* for *{escape_md(target)}* has been *{escape_md(decision)}*\\.",
                parse_mode="MarkdownV2",
            )
        except Exception as e:
            try:
                await query.edit_message_text(f"Error processing {decision}: {e}")
            except Exception:
                pass

    # -------------------------------------------------------------------
    # Mute check
    # -------------------------------------------------------------------

    _muted_until: float = 0

    async def _is_muted(self) -> bool:
        return time.time() < self._muted_until

    async def notify_if_not_muted(self, event_type: str, data: dict[str, Any]) -> None:
        """Notify unless muted."""
        if await self._is_muted():
            return
        await self.notify(event_type, data)
