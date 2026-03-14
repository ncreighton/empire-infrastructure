"""Telegram message formatting utilities."""

import re
from datetime import datetime


def escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r"\_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(special)}])", r"\\\1", str(text))


def mono(text: str) -> str:
    """Wrap text in monospace backticks."""
    return f"`{escape_md(text)}`"


def bold(text: str) -> str:
    """Wrap text in bold markers."""
    return f"*{escape_md(text)}*"


def header(text: str) -> str:
    """Format a section header."""
    return f"*{escape_md(text)}*\n{'—' * 20}"


def status_icon(status: str) -> str:
    """Return emoji for a given status string."""
    icons = {
        "healthy": "🟢", "up": "🟢", "running": "🟢", "ok": "🟢",
        "active": "🟢", "open": "🟡", "pending": "🟡", "proposed": "🟡",
        "discovered": "🔵", "approved": "✅", "applied": "✅",
        "completed": "✅", "integrated": "✅",
        "rejected": "❌", "dismissed": "❌", "down": "🔴",
        "unhealthy": "🔴", "error": "🔴", "stopped": "⚫",
        "in_progress": "🔄", "evaluated": "🔍", "recommended": "💡",
    }
    return icons.get(status.lower(), "⚪")


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def format_number(n) -> str:
    """Format a number with commas."""
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return str(n)


def format_timestamp(ts: str) -> str:
    """Format an ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %H:%M")
    except (ValueError, AttributeError):
        return str(ts)[:16] if ts else "N/A"


def format_dict_list(items: list[dict], fields: list[str], max_items: int = 10) -> str:
    """Format a list of dicts as a compact numbered list."""
    lines = []
    for i, item in enumerate(items[:max_items], 1):
        parts = []
        for f in fields:
            val = item.get(f, "")
            if val:
                parts.append(str(val))
        lines.append(f"{i}\\. {escape_md(' | '.join(parts))}")
    if len(items) > max_items:
        lines.append(f"_\\.\\.\\.and {len(items) - max_items} more_")
    return "\n".join(lines)


def format_health_table(services: list[dict]) -> str:
    """Format services as a health status list."""
    lines = []
    for svc in services:
        name = svc.get("name", svc.get("service", "unknown"))
        status = svc.get("status", "unknown")
        icon = status_icon(status)
        lines.append(f"{icon} {escape_md(name)}: {escape_md(status)}")
    return "\n".join(lines)


def format_stats_block(stats: dict) -> str:
    """Format a stats dict as key-value pairs."""
    lines = []
    for key, val in stats.items():
        display_key = key.replace("_", " ").title()
        lines.append(f"  {escape_md(display_key)}: {bold(str(val))}")
    return "\n".join(lines)
