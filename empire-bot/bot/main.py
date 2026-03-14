"""Empire Brain Commander Bot — main entry point."""

import datetime
import logging
import sys

from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
)

from bot.config import TELEGRAM_TOKEN

# Handlers
from bot.handlers.start import cmd_start, cmd_help, cmd_status, menu_callback
from bot.handlers.brain import (
    cmd_query, cmd_learn, cmd_patterns, cmd_opportunities,
    cmd_briefing, cmd_forecast, brain_callback,
)
from bot.handlers.evolve import (
    cmd_evolve, cmd_ideas, cmd_enhancements, cmd_discoveries,
    evolve_callback,
)
from bot.handlers.sites import cmd_sites, cmd_site, cmd_posts, sites_callback
from bot.handlers.health import cmd_health, cmd_docker, health_callback
from bot.handlers.infra import cmd_n8n, cmd_logs, infra_callback
from bot.handlers.content import cmd_articles, cmd_images, cmd_videos, content_callback
from bot.handlers.stats import cmd_stats, cmd_credits, stats_callback

# Scheduled jobs
from bot.jobs.morning_briefing import morning_briefing
from bot.jobs.health_watchdog import health_watchdog
from bot.jobs.evolution_digest import evolution_digest
from bot.jobs.credit_monitor import credit_monitor
from bot.jobs.weekly_report import weekly_report

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_app() -> Application:
    """Build and configure the Telegram bot application."""
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_COMMANDER_TOKEN not set")
        sys.exit(1)

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # ── Command handlers ─────────────────────────────────────────
    commands = [
        # Core
        ("start", cmd_start),
        ("help", cmd_help),
        ("status", cmd_status),
        ("menu", cmd_start),
        # Brain
        ("query", cmd_query),
        ("learn", cmd_learn),
        ("patterns", cmd_patterns),
        ("opportunities", cmd_opportunities),
        ("briefing", cmd_briefing),
        ("forecast", cmd_forecast),
        # Evolution
        ("evolve", cmd_evolve),
        ("ideas", cmd_ideas),
        ("enhancements", cmd_enhancements),
        ("discoveries", cmd_discoveries),
        # Sites
        ("sites", cmd_sites),
        ("site", cmd_site),
        ("posts", cmd_posts),
        # Health
        ("health", cmd_health),
        ("docker", cmd_docker),
        # Infrastructure
        ("n8n", cmd_n8n),
        ("logs", cmd_logs),
        # Content
        ("articles", cmd_articles),
        ("images", cmd_images),
        ("videos", cmd_videos),
        # Stats
        ("stats", cmd_stats),
        ("credits", cmd_credits),
    ]
    for name, handler in commands:
        app.add_handler(CommandHandler(name, handler))

    # ── Callback query handlers (ordered by specificity) ─────────
    # Evolution actions (approve/reject) — most specific patterns first
    app.add_handler(CallbackQueryHandler(
        evolve_callback, pattern=r"^(ideas|enhancements|discoveries)_(approve|reject|list)_"))
    app.add_handler(CallbackQueryHandler(
        evolve_callback, pattern=r"^evolve_"))

    # Site callbacks
    app.add_handler(CallbackQueryHandler(sites_callback, pattern=r"^(site_|siteact_|menu_sites)"))

    # Health callbacks
    app.add_handler(CallbackQueryHandler(health_callback, pattern=r"^(health_|docker_)"))

    # Brain callbacks
    app.add_handler(CallbackQueryHandler(brain_callback, pattern=r"^brain_"))

    # Infra callbacks
    app.add_handler(CallbackQueryHandler(infra_callback, pattern=r"^infra_"))

    # Content callbacks
    app.add_handler(CallbackQueryHandler(content_callback, pattern=r"^content_"))

    # Stats callbacks
    app.add_handler(CallbackQueryHandler(stats_callback, pattern=r"^stats_"))

    # Main menu navigation (catch-all for menu_ prefixes)
    app.add_handler(CallbackQueryHandler(menu_callback, pattern=r"^menu_"))

    # ── Scheduled jobs ───────────────────────────────────────────
    jq = app.job_queue

    # Morning briefing: daily at 6:00 AM
    jq.run_daily(
        morning_briefing,
        time=datetime.time(hour=6, minute=0, second=0),
        name="morning_briefing",
    )

    # Health watchdog: every 5 minutes
    jq.run_repeating(
        health_watchdog,
        interval=datetime.timedelta(minutes=5),
        first=datetime.timedelta(seconds=30),
        name="health_watchdog",
    )

    # Evolution digest: every 6 hours
    jq.run_repeating(
        evolution_digest,
        interval=datetime.timedelta(hours=6),
        first=datetime.timedelta(minutes=5),
        name="evolution_digest",
    )

    # Credit monitor: daily at 8:00 PM
    jq.run_daily(
        credit_monitor,
        time=datetime.time(hour=20, minute=0, second=0),
        name="credit_monitor",
    )

    # Weekly report: Sunday at 8:00 AM (day_of_week=6 is Sunday for python-telegram-bot)
    jq.run_daily(
        weekly_report,
        time=datetime.time(hour=8, minute=0, second=0),
        days=(6,),
        name="weekly_report",
    )

    # Global error handler
    async def error_handler(update, context):
        logger.error("Unhandled exception: %s", context.error, exc_info=context.error)
        if update and update.effective_message:
            try:
                await update.effective_message.reply_text(f"Error: {context.error}")
            except Exception:
                pass

    app.add_error_handler(error_handler)

    logger.info("Bot configured: %d commands, 5 scheduled jobs", len(commands))
    return app


async def post_init(app: Application):
    """Set bot commands after initialization."""
    from telegram import BotCommand
    await app.bot.set_my_commands([
        BotCommand("start", "Main menu"),
        BotCommand("help", "Command reference"),
        BotCommand("status", "Quick status"),
        BotCommand("query", "Search the Brain"),
        BotCommand("learn", "Teach the Brain"),
        BotCommand("evolve", "Run evolution cycle"),
        BotCommand("ideas", "Pending ideas"),
        BotCommand("enhancements", "Pending enhancements"),
        BotCommand("sites", "All sites"),
        BotCommand("health", "Service health"),
        BotCommand("docker", "Docker management"),
        BotCommand("stats", "Brain statistics"),
        BotCommand("credits", "Credit usage"),
        BotCommand("n8n", "Workflow status"),
    ])
    logger.info("Bot commands registered with Telegram")


def main():
    """Entry point."""
    app = build_app()
    app.post_init = post_init

    logger.info("Starting Empire Brain Commander Bot...")
    app.run_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
