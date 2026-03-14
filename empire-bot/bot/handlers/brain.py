"""Brain query, learn, patterns, opportunities, briefing, forecast handlers."""

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.utils.auth import admin_only
from bot.utils.formatters import escape_md, truncate, status_icon, format_timestamp
from bot.utils.keyboards import brain_menu, main_menu

logger = logging.getLogger(__name__)
brain = BrainClient()


async def _safe_send(update_or_query, text: str, reply_markup=None):
    """Send with markdown, fallback to plain text."""
    target = update_or_query.message if hasattr(update_or_query, "message") and update_or_query.message else None
    edit = hasattr(update_or_query, "edit_message_text")

    try:
        if edit:
            await update_or_query.edit_message_text(
                text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup,
            )
        elif target:
            await target.reply_text(
                text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup,
            )
    except Exception:
        import re
        plain = re.sub(r"[\\*_`\[\]()~>#+=|{}.!-]", "", text)
        try:
            if edit:
                await update_or_query.edit_message_text(plain, reply_markup=reply_markup)
            elif target:
                await target.reply_text(plain, reply_markup=reply_markup)
        except Exception as e2:
            logger.error("Failed to send message: %s", e2)


@admin_only
async def cmd_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /query <text> — search the Brain."""
    if not context.args:
        await update.message.reply_text("Usage: /query <search text>")
        return

    query_text = " ".join(context.args)
    await update.message.chat.send_action("typing")

    try:
        data = await brain.query(query_text)
        results = data.get("results", [])
        if not results:
            await update.message.reply_text(f"No results for: {query_text}")
            return

        lines = [f"*Brain Query:* {escape_md(query_text)}\n"]
        for i, r in enumerate(results[:10], 1):
            title = r.get("name", r.get("title", r.get("content", "?")))
            rtype = r.get("type", "")
            lines.append(f"{i}\\. {escape_md(truncate(str(title), 80))} _\\({escape_md(rtype)}\\)_")

        await _safe_send(update, "\n".join(lines), reply_markup=brain_menu())
    except Exception as e:
        await update.message.reply_text(f"Query failed: {e}")


@admin_only
async def cmd_learn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /learn <text> — teach the Brain."""
    if not context.args:
        await update.message.reply_text("Usage: /learn <lesson text>")
        return

    content = " ".join(context.args)
    try:
        data = await brain.learn(content, source="telegram")
        lid = data.get("learning_id", "?")
        await update.message.reply_text(f"Learned! (ID: {lid})")
    except Exception as e:
        await update.message.reply_text(f"Learn failed: {e}")


@admin_only
async def cmd_patterns(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /patterns — list detected patterns."""
    await update.message.chat.send_action("typing")
    try:
        data = await brain.patterns()
        patterns = data.get("patterns", [])
        count = data.get("count", len(patterns))

        lines = [f"*Detected Patterns* \\({escape_md(str(count))}\\)\n"]
        for p in patterns[:15]:
            name = p.get("name", p.get("pattern_name", "?"))
            ptype = p.get("type", p.get("pattern_type", ""))
            conf = p.get("confidence", 0)
            lines.append(f"• {escape_md(name)} _\\({escape_md(ptype)}\\)_ \\- {escape_md(f'{conf:.0%}')}")

        await _safe_send(update, "\n".join(lines), reply_markup=brain_menu())
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@admin_only
async def cmd_opportunities(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /opportunities — list open opportunities."""
    await update.message.chat.send_action("typing")
    try:
        data = await brain.opportunities()
        opps = data.get("opportunities", [])
        count = data.get("count", len(opps))

        lines = [f"*Open Opportunities* \\({escape_md(str(count))}\\)\n"]
        for o in opps[:15]:
            title = o.get("title", o.get("description", "?"))
            otype = o.get("type", o.get("opportunity_type", ""))
            status = o.get("status", "open")
            lines.append(f"{status_icon(status)} {escape_md(truncate(str(title), 80))} _\\({escape_md(otype)}\\)_")

        await _safe_send(update, "\n".join(lines), reply_markup=brain_menu())
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@admin_only
async def cmd_briefing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /briefing — today's morning briefing."""
    await update.message.chat.send_action("typing")
    try:
        data = await brain.briefing()
        # Format briefing data
        lines = ["*Morning Briefing* ☀️\n"]

        if isinstance(data, dict):
            for key, val in data.items():
                if key in ("timestamp", "version"):
                    continue
                display_key = key.replace("_", " ").title()
                if isinstance(val, dict):
                    lines.append(f"\n*{escape_md(display_key)}:*")
                    for k2, v2 in val.items():
                        lines.append(f"  {escape_md(k2)}: {escape_md(str(v2))}")
                elif isinstance(val, list):
                    lines.append(f"\n*{escape_md(display_key)}:* {escape_md(str(len(val)))} items")
                    for item in val[:5]:
                        if isinstance(item, dict):
                            summary = item.get("title", item.get("name", str(item)))
                            lines.append(f"  • {escape_md(truncate(str(summary), 80))}")
                        else:
                            lines.append(f"  • {escape_md(truncate(str(item), 80))}")
                else:
                    lines.append(f"{escape_md(display_key)}: {escape_md(str(val))}")
        else:
            lines.append(escape_md(str(data)))

        await _safe_send(update, "\n".join(lines), reply_markup=brain_menu())
    except Exception as e:
        await update.message.reply_text(f"Briefing failed: {e}")


@admin_only
async def cmd_forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /forecast — weekly oracle forecast."""
    await update.message.chat.send_action("typing")
    try:
        data = await brain.forecast()
        lines = ["*Weekly Forecast* 🔮\n"]

        if isinstance(data, dict):
            for key, val in data.items():
                if key in ("timestamp",):
                    continue
                display_key = key.replace("_", " ").title()
                if isinstance(val, list):
                    lines.append(f"\n*{escape_md(display_key)}:*")
                    for item in val[:10]:
                        lines.append(f"  • {escape_md(truncate(str(item), 100))}")
                else:
                    lines.append(f"{escape_md(display_key)}: {escape_md(str(val))}")
        else:
            lines.append(escape_md(str(data)))

        await _safe_send(update, "\n".join(lines), reply_markup=brain_menu())
    except Exception as e:
        await update.message.reply_text(f"Forecast failed: {e}")


async def brain_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle brain menu callback buttons."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "brain_query":
        await query.edit_message_text(
            "Send a query with:\n/query <your search text>",
            reply_markup=brain_menu(),
        )
    elif data == "brain_learn":
        await query.edit_message_text(
            "Teach the Brain with:\n/learn <lesson text>",
            reply_markup=brain_menu(),
        )
    elif data == "brain_patterns":
        try:
            result = await brain.patterns()
            patterns = result.get("patterns", [])
            lines = [f"*Patterns* \\({escape_md(str(len(patterns)))}\\)\n"]
            for p in patterns[:10]:
                name = p.get("name", p.get("pattern_name", "?"))
                lines.append(f"• {escape_md(str(name))}")
            await _safe_send(query, "\n".join(lines), reply_markup=brain_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=brain_menu())
    elif data == "brain_opps":
        try:
            result = await brain.opportunities()
            opps = result.get("opportunities", [])
            lines = [f"*Opportunities* \\({escape_md(str(len(opps)))}\\)\n"]
            for o in opps[:10]:
                title = o.get("title", o.get("description", "?"))
                lines.append(f"💡 {escape_md(truncate(str(title), 80))}")
            await _safe_send(query, "\n".join(lines), reply_markup=brain_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=brain_menu())
    elif data == "brain_briefing":
        try:
            result = await brain.briefing()
            text = f"*Briefing*\n\n{escape_md(truncate(str(result), 3500))}"
            await _safe_send(query, text, reply_markup=brain_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=brain_menu())
    elif data == "brain_forecast":
        try:
            result = await brain.forecast()
            text = f"*Forecast* 🔮\n\n{escape_md(truncate(str(result), 3500))}"
            await _safe_send(query, text, reply_markup=brain_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=brain_menu())
