"""Health monitoring handlers: services, Docker, Brain health."""

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.services.dashboard_client import DashboardClient
from bot.services import docker_client
from bot.utils.auth import admin_only
from bot.utils.formatters import escape_md, status_icon, truncate
from bot.utils.keyboards import health_menu, docker_service_actions, main_menu

logger = logging.getLogger(__name__)
brain = BrainClient()
dashboard = DashboardClient()


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
async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /health — overview of all service health."""
    await update.message.chat.send_action("typing")
    lines = ["Service Health\n"]

    # Brain (short timeout — it's cross-network)
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{brain.base_url}/health")
            r.raise_for_status()
            bh = r.json()
        lines.append(f"🟢 Brain MCP: {bh.get('status', 'ok')}")
    except Exception:
        lines.append("🔴 Brain MCP: unreachable")

    # Dashboard services
    try:
        svc_data = await dashboard.services()
        services_dict = svc_data.get("services", {}) if isinstance(svc_data, dict) else {}
        if isinstance(services_dict, dict):
            for name, info in services_dict.items():
                if isinstance(info, dict):
                    status = info.get("status", "unknown")
                else:
                    status = str(info)
                icon = "🟢" if status in ("healthy", "up", "running", "ok") else "🔴" if status in ("down", "error") else "⚪"
                lines.append(f"{icon} {name}: {status}")
        elif isinstance(svc_data, list):
            for svc in svc_data:
                name = svc.get("name", svc.get("service", "?"))
                status = svc.get("status", "unknown")
                icon = "🟢" if status in ("healthy", "up", "running", "ok") else "🔴"
                lines.append(f"{icon} {name}: {status}")
        # Also show overall
        overall = svc_data.get("overall", "") if isinstance(svc_data, dict) else ""
        if overall:
            lines.insert(1, f"Overall: {overall}")
    except Exception as e:
        lines.append(f"🔴 Dashboard: {e}")

    # Send as plain text (no markdown issues)
    try:
        await update.message.reply_text("\n".join(lines), reply_markup=health_menu())
    except Exception as e:
        logger.error("cmd_health send failed: %s", e)
        await update.message.reply_text(f"Health check error: {e}")


@admin_only
async def cmd_docker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /docker <ps|restart|logs> [service]."""
    if not context.args:
        await update.message.reply_text("Usage: /docker <ps|restart|logs> [service]")
        return

    action = context.args[0].lower()
    service = context.args[1] if len(context.args) > 1 else None

    await update.message.chat.send_action("typing")

    if action == "ps":
        code, output = await docker_client.ps()
        await update.message.reply_text(f"```\n{output[:3500]}\n```", parse_mode=ParseMode.MARKDOWN_V2)

    elif action == "restart":
        if not service:
            await update.message.reply_text("Usage: /docker restart <service>")
            return
        code, output = await docker_client.restart(service)
        emoji = "✅" if code == 0 else "❌"
        await update.message.reply_text(f"{emoji} Restart {service}: {output[:500]}")

    elif action == "logs":
        if not service:
            await update.message.reply_text("Usage: /docker logs <service>")
            return
        lines = int(context.args[2]) if len(context.args) > 2 else 30
        code, output = await docker_client.logs(service, lines)
        await update.message.reply_text(f"```\n{output[:3500]}\n```", parse_mode=ParseMode.MARKDOWN_V2)

    else:
        await update.message.reply_text("Unknown action. Use: ps, restart, logs")


async def health_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle health menu callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "health_all":
        lines = ["*Service Health* 💚\n"]
        try:
            bh = await brain.health()
            lines.append(f"{status_icon(bh.get('status', 'unknown'))} Brain: {escape_md(bh.get('status', '?'))}")
        except Exception:
            lines.append(f"{status_icon('down')} Brain: unreachable")

        try:
            svc_data = await dashboard.services()
            services = svc_data if isinstance(svc_data, list) else svc_data.get("services", [])
            for svc in services:
                name = svc.get("name", svc.get("service", "?"))
                status = svc.get("status", "unknown")
                lines.append(f"{status_icon(status)} {escape_md(name)}: {escape_md(status)}")
        except Exception:
            lines.append(f"{status_icon('down')} Dashboard: unreachable")

        await _safe_send(query, "\n".join(lines), reply_markup=health_menu())

    elif data == "health_docker":
        code, output = await docker_client.ps()
        text = f"*Docker Containers* 🐳\n\n```\n{output[:3000]}\n```"
        try:
            await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=health_menu())
        except Exception:
            await query.edit_message_text(f"Docker PS:\n{output[:3500]}", reply_markup=health_menu())

    elif data == "health_brain":
        try:
            bh = await brain.brain_health()
            lines = ["*Brain Health* 🧠\n"]
            if isinstance(bh, dict):
                for k, v in bh.items():
                    if k == "timestamp":
                        continue
                    lines.append(f"{escape_md(k)}: {escape_md(str(v))}")
            await _safe_send(query, "\n".join(lines), reply_markup=health_menu())
        except Exception as e:
            await query.edit_message_text(f"Brain health error: {e}", reply_markup=health_menu())

    elif data == "health_dashboard":
        try:
            dh = await dashboard.health()
            lines = ["*Dashboard Health*\n"]
            if isinstance(dh, dict):
                for k, v in dh.items():
                    lines.append(f"{escape_md(k)}: {escape_md(str(v))}")
            await _safe_send(query, "\n".join(lines), reply_markup=health_menu())
        except Exception as e:
            await query.edit_message_text(f"Dashboard error: {e}", reply_markup=health_menu())

    # Docker service actions
    elif data.startswith("docker_restart_"):
        service = data[len("docker_restart_"):]
        code, output = await docker_client.restart(service)
        emoji = "✅" if code == 0 else "❌"
        await query.edit_message_text(f"{emoji} Restart {service}: {output[:500]}", reply_markup=health_menu())

    elif data.startswith("docker_logs_"):
        service = data[len("docker_logs_"):]
        code, output = await docker_client.logs(service, 20)
        text = f"Logs ({service}):\n```\n{output[:3000]}\n```"
        try:
            await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=health_menu())
        except Exception:
            await query.edit_message_text(f"Logs ({service}):\n{output[:3500]}", reply_markup=health_menu())
