"""Statistics handlers: Brain stats, credits, adoption, CLAUDE.md sizes."""

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.utils.auth import admin_only
from bot.utils.formatters import escape_md, format_number, truncate
from bot.utils.keyboards import stats_menu

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
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats — Brain statistics overview."""
    await update.message.chat.send_action("typing")
    try:
        data = await brain.stats()
        lines = ["*Brain Statistics* 📊\n"]

        if isinstance(data, dict):
            for key, val in data.items():
                if key == "timestamp":
                    continue
                display_key = key.replace("_", " ").title()
                if isinstance(val, (int, float)):
                    lines.append(f"{escape_md(display_key)}: *{escape_md(format_number(val))}*")
                elif isinstance(val, dict):
                    lines.append(f"\n*{escape_md(display_key)}:*")
                    for k2, v2 in val.items():
                        lines.append(f"  {escape_md(k2)}: {escape_md(str(v2))}")
                else:
                    lines.append(f"{escape_md(display_key)}: {escape_md(str(val))}")

        await _safe_send(update, "\n".join(lines), reply_markup=stats_menu())
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@admin_only
async def cmd_credits(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /credits — credit usage report."""
    await update.message.chat.send_action("typing")
    try:
        data = await brain.credit_status()
        lines = ["*Credit Status* 💰\n"]

        savings = data.get("potential_savings", {})
        if savings:
            lines.append("*Potential Savings:*")
            for k, v in savings.items():
                lines.append(f"  {escape_md(k)}: {escape_md(str(v))}")

        advisory = data.get("advisory", "")
        if advisory:
            lines.append(f"\n*Advisory:* {escape_md(truncate(str(advisory), 200))}")

        await _safe_send(update, "\n".join(lines), reply_markup=stats_menu())
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def stats_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle stats menu callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "stats_overview":
        try:
            result = await brain.stats()
            lines = ["*Brain Stats* 📊\n"]
            if isinstance(result, dict):
                for k, v in result.items():
                    if k == "timestamp":
                        continue
                    if isinstance(v, (int, float)):
                        lines.append(f"{escape_md(k.replace('_', ' ').title())}: *{escape_md(format_number(v))}*")
            await _safe_send(query, "\n".join(lines), reply_markup=stats_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=stats_menu())

    elif data == "stats_credits":
        try:
            result = await brain.credit_report()
            text = f"*Credit Report* 💰\n\n{escape_md(truncate(str(result), 3500))}"
            await _safe_send(query, text, reply_markup=stats_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=stats_menu())

    elif data == "stats_adoption":
        try:
            result = await brain.adoption_metrics()
            text = f"*Adoption Metrics* 📈\n\n{escape_md(truncate(str(result), 3500))}"
            await _safe_send(query, text, reply_markup=stats_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=stats_menu())

    elif data == "stats_claudemd":
        try:
            result = await brain.claude_md_sizes()
            files = result.get("files", [])
            total_tokens = result.get("total_tokens", 0)
            lines = [
                f"*CLAUDE\\.md Sizes* 📏\n",
                f"Total tokens: *{escape_md(format_number(total_tokens))}*\n",
            ]
            for f in files[:15]:
                name = f.get("file", f.get("project", "?"))
                tokens = f.get("tokens", 0)
                lines.append(f"• {escape_md(str(name))}: {escape_md(format_number(tokens))}")
            await _safe_send(query, "\n".join(lines), reply_markup=stats_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=stats_menu())
