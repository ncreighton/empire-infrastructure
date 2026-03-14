"""Daily 6AM morning briefing push."""

import logging

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.config import ADMIN_IDS
from bot.utils.formatters import escape_md, truncate, format_number

logger = logging.getLogger(__name__)
brain = BrainClient()


async def morning_briefing(context: ContextTypes.DEFAULT_TYPE):
    """Push daily briefing to all admin users."""
    logger.info("Running morning briefing job")

    try:
        data = await brain.briefing()
    except Exception as e:
        logger.error("Morning briefing failed: %s", e)
        for uid in ADMIN_IDS:
            try:
                await context.bot.send_message(uid, f"Morning briefing failed: {e}")
            except Exception:
                pass
        return

    # Build briefing message
    lines = ["☀️ *Morning Briefing*\n"]

    if isinstance(data, dict):
        # Stats section
        stats = data.get("stats", data.get("brain_stats", {}))
        if isinstance(stats, dict):
            lines.append("*Empire Stats:*")
            for k in ("projects", "skills", "patterns", "learnings", "functions"):
                if k in stats:
                    lines.append(f"  {escape_md(k.title())}: {escape_md(format_number(stats[k]))}")
            lines.append("")

        # Opportunities
        opps = data.get("opportunities", data.get("top_opportunities", []))
        if isinstance(opps, list) and opps:
            lines.append(f"*Opportunities* \\({escape_md(str(len(opps)))}\\):")
            for o in opps[:5]:
                if isinstance(o, dict):
                    title = o.get("title", o.get("description", str(o)))
                    lines.append(f"  💡 {escape_md(truncate(str(title), 80))}")

        # Health
        health = data.get("health", data.get("service_health", {}))
        if isinstance(health, dict):
            lines.append("\n*Health:*")
            for k, v in health.items():
                lines.append(f"  {escape_md(k)}: {escape_md(str(v))}")

        # Activity
        activity = data.get("activity", data.get("recent_activity", []))
        if isinstance(activity, list) and activity:
            lines.append(f"\n*Recent Activity* \\({escape_md(str(len(activity)))}\\):")
            for a in activity[:5]:
                if isinstance(a, dict):
                    desc = a.get("description", a.get("event", str(a)))
                    lines.append(f"  • {escape_md(truncate(str(desc), 80))}")
    else:
        lines.append(escape_md(truncate(str(data), 3000)))

    text = "\n".join(lines)

    for uid in ADMIN_IDS:
        try:
            await context.bot.send_message(uid, text, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            # Fallback to plain text
            import re
            plain = re.sub(r"[\\*_`\[\]()~>#+=|{}.!-]", "", text)
            try:
                await context.bot.send_message(uid, plain)
            except Exception as e2:
                logger.error("Failed to send briefing to %s: %s", uid, e2)
