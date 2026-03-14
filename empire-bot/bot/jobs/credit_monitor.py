"""Daily 8PM credit usage alert."""

import logging

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.config import ADMIN_IDS
from bot.utils.formatters import escape_md, truncate

logger = logging.getLogger(__name__)
brain = BrainClient()


async def credit_monitor(context: ContextTypes.DEFAULT_TYPE):
    """Push daily credit usage summary."""
    logger.info("Running credit monitor job")

    try:
        data = await brain.credit_status()
    except Exception as e:
        logger.error("Credit monitor failed: %s", e)
        return

    lines = ["💰 *Daily Credit Report*\n"]

    savings = data.get("potential_savings", {})
    if isinstance(savings, dict) and savings:
        lines.append("*Potential Savings:*")
        for k, v in savings.items():
            lines.append(f"  {escape_md(k)}: {escape_md(str(v))}")

    patterns = data.get("session_patterns", {})
    if isinstance(patterns, dict) and patterns:
        lines.append("\n*Session Patterns:*")
        for k, v in list(patterns.items())[:5]:
            lines.append(f"  {escape_md(k)}: {escape_md(str(v))}")

    advisory = data.get("advisory", "")
    if advisory:
        lines.append(f"\n*Advisory:*\n{escape_md(truncate(str(advisory), 300))}")

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
                logger.error("Credit alert send failed to %s: %s", uid, e)
