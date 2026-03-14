"""Every 5 minutes: check services, alert on down/recovery."""

import logging
from typing import Any

from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.services.dashboard_client import DashboardClient
from bot.config import ADMIN_IDS
from bot.utils.formatters import status_icon

logger = logging.getLogger(__name__)
brain = BrainClient()
dashboard = DashboardClient()

# Track previous states for change detection
_prev_states: dict[str, str] = {}


async def health_watchdog(context: ContextTypes.DEFAULT_TYPE):
    """Check all services, alert only on state changes."""
    global _prev_states
    current_states: dict[str, str] = {}

    # Check Brain MCP
    try:
        bh = await brain.health()
        current_states["Brain MCP"] = bh.get("status", "unknown")
    except Exception:
        current_states["Brain MCP"] = "down"

    # Check Dashboard services
    try:
        svc_data = await dashboard.services()
        services = svc_data if isinstance(svc_data, list) else svc_data.get("services", [])
        for svc in services:
            name = svc.get("name", svc.get("service", "?"))
            current_states[name] = svc.get("status", "unknown")
    except Exception:
        current_states["Dashboard"] = "down"

    # Detect changes
    alerts: list[str] = []
    for name, status in current_states.items():
        prev = _prev_states.get(name)
        if prev is None:
            # First run, don't alert
            continue

        is_down = status.lower() in ("down", "unhealthy", "error", "stopped")
        was_down = prev.lower() in ("down", "unhealthy", "error", "stopped")

        if is_down and not was_down:
            alerts.append(f"🔴 {name} went DOWN ({prev} → {status})")
        elif not is_down and was_down:
            alerts.append(f"🟢 {name} RECOVERED ({prev} → {status})")

    _prev_states = current_states

    # Send alerts if any
    if alerts:
        text = "⚠️ *Service Alert*\n\n" + "\n".join(alerts)
        for uid in ADMIN_IDS:
            try:
                await context.bot.send_message(uid, text)
            except Exception as e:
                logger.error("Alert send failed to %s: %s", uid, e)
