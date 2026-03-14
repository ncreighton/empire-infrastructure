"""Infrastructure handlers: n8n workflows, logs."""

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.n8n_client import N8nClient
from bot.services import docker_client
from bot.utils.auth import admin_only
from bot.utils.formatters import escape_md, truncate, status_icon
from bot.utils.keyboards import infra_menu

logger = logging.getLogger(__name__)
n8n = N8nClient()


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
async def cmd_n8n(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /n8n — list workflows and their status."""
    await update.message.chat.send_action("typing")
    try:
        data = await n8n.workflows()
        workflows = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(workflows, dict):
            workflows = workflows.get("data", [])

        lines = ["*n8n Workflows* ⚙️\n"]
        for wf in workflows[:20] if isinstance(workflows, list) else []:
            name = wf.get("name", "?")
            active = wf.get("active", False)
            icon = "🟢" if active else "⚫"
            wid = wf.get("id", "?")
            lines.append(f"{icon} {escape_md(truncate(name, 40))} \\[{escape_md(str(wid))}\\]")

        if not workflows:
            lines.append("_No workflows found_")

        await _safe_send(update, "\n".join(lines), reply_markup=infra_menu())
    except Exception as e:
        await update.message.reply_text(f"n8n error: {e}")


@admin_only
async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /logs <service> [lines] — view service logs."""
    if not context.args:
        await update.message.reply_text("Usage: /logs <service> [lines=30]")
        return

    service = context.args[0]
    lines_count = int(context.args[1]) if len(context.args) > 1 else 30

    await update.message.chat.send_action("typing")
    code, output = await docker_client.logs(service, lines_count)

    text = f"*Logs: {escape_md(service)}*\n\n```\n{output[:3500]}\n```"
    try:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception:
        await update.message.reply_text(f"Logs ({service}):\n{output[:3500]}")


async def infra_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle infra menu callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "infra_n8n":
        try:
            result = await n8n.workflows()
            workflows = result.get("data", result) if isinstance(result, dict) else result
            if isinstance(workflows, dict):
                workflows = workflows.get("data", [])

            lines = ["*n8n Workflows* ⚙️\n"]
            for wf in (workflows[:20] if isinstance(workflows, list) else []):
                name = wf.get("name", "?")
                active = wf.get("active", False)
                icon = "🟢" if active else "⚫"
                lines.append(f"{icon} {escape_md(truncate(name, 40))}")

            await _safe_send(query, "\n".join(lines), reply_markup=infra_menu())
        except Exception as e:
            await query.edit_message_text(f"n8n error: {e}", reply_markup=infra_menu())

    elif data == "infra_logs":
        await query.edit_message_text(
            "Send /logs <service> to view logs\\.\n\n"
            "Services: n8n, dashboard, article\\-audit, toolbox, openclaw\\-agent, bookforge, crypto\\-claw, empire\\-bot",
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=infra_menu(),
        )
