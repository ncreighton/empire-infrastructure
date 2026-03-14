"""Start, help, and main menu handlers."""

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.utils.auth import admin_only
from bot.utils.formatters import escape_md
from bot.utils.keyboards import (
    main_menu, brain_menu, evolve_menu, health_menu,
    infra_menu, content_menu, stats_menu,
)

logger = logging.getLogger(__name__)

HELP_TEXT = """
*Empire Brain Commander*

*Core Commands:*
/start \\- Main menu
/help \\- This help message
/status \\- Quick status overview

*Brain:*
/query \\<text\\> \\- Search the Brain
/learn \\<text\\> \\- Teach the Brain
/patterns \\- View detected patterns
/opportunities \\- View open opportunities
/briefing \\- Today's briefing
/forecast \\- Weekly oracle forecast

*Evolution:*
/evolve \\[quick\\|discover\\|full\\] \\- Run evolution cycle
/ideas \\- Browse pending ideas
/enhancements \\- Browse enhancements
/discoveries \\- Browse discoveries

*Sites:*
/sites \\- All sites overview
/site \\<name\\> \\- Site details
/posts \\<site\\> \\- Recent posts

*Health:*
/health \\- Service health
/docker \\<ps\\|restart\\|logs\\> \\- Docker management

*Infrastructure:*
/n8n \\- Workflow status
/logs \\<service\\> \\- View service logs

*Stats:*
/stats \\- Brain statistics
/credits \\- Credit usage report
"""


@admin_only
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command — show main menu."""
    name = update.effective_user.first_name if update.effective_user else "Commander"
    try:
        await update.message.reply_text(
            f"Welcome back, *{escape_md(name)}*\\! 🧠\n\n"
            "Empire Brain Commander at your service\\.\n"
            "Choose a category:",
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=main_menu(),
        )
    except Exception as e:
        logger.error("cmd_start send failed: %s", e)
        await update.message.reply_text(
            f"Welcome back, {name}! 🧠\n\nEmpire Brain Commander at your service.\nChoose a category:",
            reply_markup=main_menu(),
        )


@admin_only
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN_V2)


@admin_only
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick status overview from Brain."""
    from bot.services.brain_client import BrainClient
    brain = BrainClient()
    try:
        data = await brain.health()
        stats = data.get("brain_stats", {})
        lines = [
            "*Empire Status* 🧠\n",
            f"Brain: {escape_md(data.get('status', 'unknown'))}",
            f"Version: {escape_md(data.get('version', '?'))}",
            f"Projects: {escape_md(str(stats.get('projects', '?')))}",
            f"Skills: {escape_md(str(stats.get('skills', '?')))}",
            f"Patterns: {escape_md(str(stats.get('patterns', '?')))}",
            f"Learnings: {escape_md(str(stats.get('learnings', '?')))}",
        ]
        await update.message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=main_menu(),
        )
    except Exception as e:
        await update.message.reply_text(f"Brain unreachable: {e}")


@admin_only
async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle main menu navigation callbacks."""
    query = update.callback_query
    await query.answer()

    data = query.data
    menus = {
        "menu_home": ("*Empire Brain Commander* 🧠\n\nChoose a category:", main_menu()),
        "menu_brain": ("*Brain Tools* 🧠\n\nQuery, learn, explore:", brain_menu()),
        "menu_evolve": ("*Evolution Engine* 🧬\n\nCycles, ideas, enhancements:", evolve_menu()),
        "menu_health": ("*Health Monitor* 💚\n\nServices, Docker, diagnostics:", health_menu()),
        "menu_infra": ("*Infrastructure* 🔧\n\nWorkflows, logs, deploys:", infra_menu()),
        "menu_content": ("*Content Pipeline* 📝\n\nArticles, images:", content_menu()),
        "menu_stats": ("*Statistics* 📊\n\nMetrics, credits, adoption:", stats_menu()),
        "menu_sites": None,  # handled separately in sites handler
    }

    if data in menus and menus[data]:
        text, keyboard = menus[data]
        await query.edit_message_text(
            text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard,
        )
