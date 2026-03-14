"""Content pipeline handlers: articles, images, video topics."""

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.services.wp_client import WPClient, get_site_ids
from bot.utils.auth import admin_only
from bot.utils.formatters import escape_md, truncate, format_timestamp
from bot.utils.keyboards import content_menu

logger = logging.getLogger(__name__)
brain = BrainClient()


async def _safe_send(target, text: str, reply_markup=None):
    try:
        if hasattr(target, "edit_message_text"):
            await target.edit_message_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup)
        elif hasattr(target, "message") and target.message:
            await target.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup)
    except Exception:
        import re
        plain = re.sub(r"[\\*_`\[\]()~>#+=|{}.!-]", "", text)
        try:
            if hasattr(target, "edit_message_text"):
                await target.edit_message_text(plain, reply_markup=reply_markup)
            elif hasattr(target, "message") and target.message:
                await target.message.reply_text(plain, reply_markup=reply_markup)
        except Exception as e:
            logger.error("Send failed: %s", e)


@admin_only
async def cmd_articles(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /articles <site> — list recent articles."""
    if not context.args:
        sites = ", ".join(sorted(get_site_ids())[:8])
        await update.message.reply_text(f"Usage: /articles <site_id>\n\nSites: {sites}...")
        return

    site_id = context.args[0].lower()
    await update.message.chat.send_action("typing")

    try:
        wp = WPClient(site_id)
        posts = await wp.posts(per_page=10)

        lines = [f"*Articles \\- {escape_md(site_id)}*\n"]
        for p in posts:
            title = p.get("title", {}).get("rendered", "Untitled")
            date = format_timestamp(p.get("date", ""))
            pid = p.get("id", "?")
            lines.append(f"• \\[{escape_md(str(pid))}\\] {escape_md(truncate(title, 60))} _\\({escape_md(date)}\\)_")

        if not posts:
            lines.append("_No articles found_")

        await _safe_send(update, "\n".join(lines), reply_markup=content_menu())
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@admin_only
async def cmd_images(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /images <site> <title> — trigger image generation."""
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /images <site_id> <article title>")
        return

    site_id = context.args[0].lower()
    title = " ".join(context.args[1:])

    await update.message.reply_text(
        f"Image generation for *{escape_md(site_id)}*:\n"
        f"Title: _{escape_md(title)}_\n\n"
        f"Run on Windows:\n"
        f"`python article_images_pipeline.py --site {escape_md(site_id)} --title \"{escape_md(title)}\" --enhanced`",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


@admin_only
async def cmd_videos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /videos — list video topic ideas."""
    await update.message.chat.send_action("typing")
    try:
        data = await brain.witchcraft_topics(count=5)
        topics = data.get("topics", data) if isinstance(data, dict) else data

        lines = ["*Video Topic Ideas* 🎬\n"]
        if isinstance(topics, list):
            for t in topics[:10]:
                if isinstance(t, dict):
                    title = t.get("topic", t.get("title", str(t)))
                else:
                    title = str(t)
                lines.append(f"• {escape_md(truncate(title, 80))}")
        else:
            lines.append(escape_md(str(topics)))

        await _safe_send(update, "\n".join(lines), reply_markup=content_menu())
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def content_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle content menu callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "content_articles":
        sites = ", ".join(sorted(get_site_ids())[:10])
        await query.edit_message_text(
            f"Send /articles <site_id> to list posts.\n\nSites: {sites}...",
            reply_markup=content_menu(),
        )

    elif data == "content_images":
        await query.edit_message_text(
            "Send /images <site_id> <title> to generate images.\n\n"
            "Example: /images witchcraftforbeginners Full Moon Ritual Guide",
            reply_markup=content_menu(),
        )
