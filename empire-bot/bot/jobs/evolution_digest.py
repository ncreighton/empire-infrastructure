"""Every 6 hours: summarize new evolution items."""

import logging

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.config import ADMIN_IDS
from bot.utils.formatters import escape_md, truncate

logger = logging.getLogger(__name__)
brain = BrainClient()


async def evolution_digest(context: ContextTypes.DEFAULT_TYPE):
    """Send digest of pending evolution items."""
    logger.info("Running evolution digest job")

    try:
        status = await brain.evolution_status()
    except Exception as e:
        logger.error("Evolution digest failed: %s", e)
        return

    pending = status.get("pending", {})
    ideas_count = pending.get("ideas", 0)
    enhancements_count = pending.get("enhancements", 0)
    discoveries_count = pending.get("discoveries", 0)

    total = ideas_count + enhancements_count + discoveries_count
    if total == 0:
        return  # Nothing to report

    lines = [
        "🧬 *Evolution Digest*\n",
        f"Pending items: *{escape_md(str(total))}*\n",
    ]

    # Ideas summary
    if ideas_count > 0:
        lines.append(f"💡 *Ideas:* {escape_md(str(ideas_count))}")
        try:
            data = await brain.ideas(status="proposed", limit=3)
            for idea in data.get("ideas", [])[:3]:
                title = idea.get("title", idea.get("description", "?"))
                lines.append(f"  • {escape_md(truncate(str(title), 80))}")
        except Exception:
            pass

    # Enhancements summary
    if enhancements_count > 0:
        lines.append(f"\n🔧 *Enhancements:* {escape_md(str(enhancements_count))}")
        try:
            data = await brain.enhancements(status="pending", limit=3)
            for enh in data.get("enhancements", [])[:3]:
                title = enh.get("title", enh.get("description", "?"))
                proj = enh.get("project", "")
                suffix = f" \\({escape_md(proj)}\\)" if proj else ""
                lines.append(f"  • {escape_md(truncate(str(title), 70))}{suffix}")
        except Exception:
            pass

    # Discoveries summary
    if discoveries_count > 0:
        lines.append(f"\n🔬 *Discoveries:* {escape_md(str(discoveries_count))}")
        try:
            data = await brain.discoveries(status="discovered", limit=3)
            for disc in data.get("discoveries", [])[:3]:
                title = disc.get("title", disc.get("description", "?"))
                lines.append(f"  • {escape_md(truncate(str(title), 80))}")
        except Exception:
            pass

    lines.append("\nUse /ideas, /enhancements, /discoveries to review\\.")

    text = "\n".join(lines)
    for uid in ADMIN_IDS:
        try:
            await context.bot.send_message(uid, text, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            import re
            plain = re.sub(r"[\\*_`\[\]()~>#+=|{}.!-]", "", text)
            try:
                await context.bot.send_message(uid, plain)
            except Exception as e:
                logger.error("Digest send failed to %s: %s", uid, e)
