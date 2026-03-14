"""Sunday 8AM comprehensive weekly digest."""

import logging

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.config import ADMIN_IDS
from bot.utils.formatters import escape_md, truncate, format_number

logger = logging.getLogger(__name__)
brain = BrainClient()


async def weekly_report(context: ContextTypes.DEFAULT_TYPE):
    """Push comprehensive weekly report."""
    logger.info("Running weekly report job")

    lines = ["📊 *Weekly Empire Report*\n"]

    # Brain stats
    try:
        stats = await brain.stats()
        if isinstance(stats, dict):
            lines.append("*Brain Overview:*")
            for k in ("projects", "skills", "patterns", "learnings", "functions", "events"):
                if k in stats:
                    lines.append(f"  {escape_md(k.title())}: {escape_md(format_number(stats[k]))}")
            lines.append("")
    except Exception as e:
        lines.append(f"Brain stats error: {escape_md(str(e))}\n")

    # Evolution status
    try:
        evo = await brain.evolution_status()
        totals = evo.get("totals", {})
        pending = evo.get("pending", {})
        adoption = evo.get("adoption", {})

        lines.append("*Evolution Engine:*")
        lines.append(f"  Total cycles: {escape_md(str(totals.get('evolutions', 0)))}")
        lines.append(f"  Ideas: {escape_md(str(totals.get('ideas', 0)))} \\({escape_md(str(pending.get('ideas', 0)))} pending\\)")
        lines.append(f"  Enhancements: {escape_md(str(totals.get('enhancements', 0)))} \\({escape_md(str(pending.get('enhancements', 0)))} pending\\)")
        lines.append(f"  Discoveries: {escape_md(str(totals.get('discoveries', 0)))} \\({escape_md(str(pending.get('discoveries', 0)))} pending\\)")
        if isinstance(adoption, dict) and adoption:
            rate = adoption.get("adoption_rate", adoption.get("rate", "?"))
            lines.append(f"  Adoption rate: {escape_md(str(rate))}")
        lines.append("")
    except Exception as e:
        lines.append(f"Evolution error: {escape_md(str(e))}\n")

    # Credit status
    try:
        credits = await brain.credit_status()
        if isinstance(credits, dict):
            advisory = credits.get("advisory", "")
            if advisory:
                lines.append(f"*Credits:* {escape_md(truncate(str(advisory), 200))}")
                lines.append("")
    except Exception:
        pass

    # Opportunities count
    try:
        opps = await brain.opportunities()
        count = opps.get("count", len(opps.get("opportunities", [])))
        lines.append(f"*Open Opportunities:* {escape_md(str(count))}")
    except Exception:
        pass

    lines.append("\nHave a productive week\\! 🚀")

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
                logger.error("Weekly report send failed to %s: %s", uid, e)
