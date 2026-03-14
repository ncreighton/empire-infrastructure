"""Evolution engine handlers: cycles, ideas, enhancements, discoveries."""

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.utils.auth import admin_only
from bot.utils.formatters import escape_md, truncate, status_icon
from bot.utils.keyboards import evolve_menu, item_actions, pagination
from bot.config import PAGE_SIZE

logger = logging.getLogger(__name__)
brain = BrainClient()


async def _safe_send(target, text: str, reply_markup=None):
    """Send with markdown, fallback to plain."""
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
async def cmd_evolve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /evolve [quick|discover|full] — trigger evolution cycle."""
    cycle = context.args[0] if context.args else "quick"
    if cycle not in ("quick", "discover", "full"):
        await update.message.reply_text("Usage: /evolve [quick|discover|full]")
        return

    await update.message.chat.send_action("typing")
    try:
        data = await brain.evolve(cycle)
        result = data.get("result", data)
        lines = [f"*Evolution Cycle: {escape_md(cycle)}* 🧬\n"]

        if isinstance(result, dict):
            for key, val in result.items():
                lines.append(f"{escape_md(key)}: {escape_md(str(val))}")
        else:
            lines.append(escape_md(str(result)))

        await _safe_send(update, "\n".join(lines), reply_markup=evolve_menu())
    except Exception as e:
        await update.message.reply_text(f"Evolve failed: {e}")


@admin_only
async def cmd_ideas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ideas — list pending ideas."""
    await update.message.chat.send_action("typing")
    try:
        data = await brain.ideas(status="proposed")
        items = data.get("ideas", [])
        await _send_item_list(update, items, "ideas", "Ideas 💡")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@admin_only
async def cmd_enhancements(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /enhancements — list pending enhancements."""
    await update.message.chat.send_action("typing")
    try:
        data = await brain.enhancements(status="pending")
        items = data.get("enhancements", [])
        await _send_item_list(update, items, "enhancements", "Enhancements 🔧")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@admin_only
async def cmd_discoveries(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /discoveries — list discoveries."""
    await update.message.chat.send_action("typing")
    try:
        data = await brain.discoveries(status="discovered")
        items = data.get("discoveries", [])
        await _send_item_list(update, items, "discoveries", "Discoveries 🔬")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def _send_item_list(update, items, item_type, title, page=0):
    """Send paginated list of evolution items."""
    total = len(items)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    start = page * PAGE_SIZE
    page_items = items[start:start + PAGE_SIZE]

    lines = [f"*{escape_md(title)}* \\({escape_md(str(total))} total, page {page + 1}/{total_pages}\\)\n"]
    for item in page_items:
        iid = item.get("id", "?")
        title_text = item.get("title", item.get("description", item.get("content", "?")))
        status = item.get("status", "?")
        lines.append(
            f"{status_icon(status)} \\[{escape_md(str(iid))}\\] {escape_md(truncate(str(title_text), 80))}"
        )

    if not page_items:
        lines.append("_No items found_")

    kb = pagination(item_type, page, total_pages)
    await _safe_send(update, "\n".join(lines), reply_markup=kb)


async def evolve_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle evolution menu and item action callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    # Cycle triggers
    if data in ("evolve_quick", "evolve_discover", "evolve_full"):
        cycle = data.split("_")[1]
        try:
            result = await brain.evolve(cycle)
            text = f"*{escape_md(cycle.title())} Cycle Started* 🧬\n\n{escape_md(truncate(str(result), 3000))}"
            await _safe_send(query, text, reply_markup=evolve_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=evolve_menu())
        return

    # Status
    if data == "evolve_status":
        try:
            result = await brain.evolution_status()
            pending = result.get("pending", {})
            totals = result.get("totals", {})
            lines = [
                "*Evolution Status* 🧬\n",
                f"*Pending:*",
                f"  Ideas: {escape_md(str(pending.get('ideas', 0)))}",
                f"  Enhancements: {escape_md(str(pending.get('enhancements', 0)))}",
                f"  Discoveries: {escape_md(str(pending.get('discoveries', 0)))}",
                f"\n*Totals:*",
                f"  Evolutions: {escape_md(str(totals.get('evolutions', 0)))}",
                f"  Ideas: {escape_md(str(totals.get('ideas', 0)))}",
                f"  Enhancements: {escape_md(str(totals.get('enhancements', 0)))}",
                f"  Discoveries: {escape_md(str(totals.get('discoveries', 0)))}",
            ]
            await _safe_send(query, "\n".join(lines), reply_markup=evolve_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=evolve_menu())
        return

    # Adoption metrics
    if data == "evolve_adoption":
        try:
            result = await brain.adoption_metrics()
            text = f"*Adoption Metrics* 📈\n\n{escape_md(truncate(str(result), 3500))}"
            await _safe_send(query, text, reply_markup=evolve_menu())
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=evolve_menu())
        return

    # List pages
    for item_type in ("ideas", "enhancements", "discoveries"):
        if data.startswith(f"{item_type}_list_"):
            page = int(data.split("_")[-1])
            try:
                if item_type == "ideas":
                    result = await brain.ideas(status="proposed")
                    items = result.get("ideas", [])
                elif item_type == "enhancements":
                    result = await brain.enhancements(status="pending")
                    items = result.get("enhancements", [])
                else:
                    result = await brain.discoveries(status="discovered")
                    items = result.get("discoveries", [])

                total = len(items)
                total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
                start = page * PAGE_SIZE
                page_items = items[start:start + PAGE_SIZE]

                label = item_type.title()
                lines = [f"*{escape_md(label)}* \\(page {page + 1}/{total_pages}\\)\n"]
                for item in page_items:
                    iid = item.get("id", "?")
                    title = item.get("title", item.get("description", "?"))
                    lines.append(f"• \\[{escape_md(str(iid))}\\] {escape_md(truncate(str(title), 80))}")

                kb = pagination(item_type, page, total_pages)
                await _safe_send(query, "\n".join(lines), reply_markup=kb)
            except Exception as e:
                await query.edit_message_text(f"Error: {e}", reply_markup=evolve_menu())
            return

    # Browse buttons from evolve menu
    if data == "evolve_ideas":
        try:
            result = await brain.ideas(status="proposed")
            items = result.get("ideas", [])
            total = len(items)
            total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
            page_items = items[:PAGE_SIZE]
            lines = [f"*Ideas* 💡 \\({escape_md(str(total))} pending\\)\n"]
            for item in page_items:
                iid = item.get("id", "?")
                title = item.get("title", item.get("description", "?"))
                lines.append(f"• \\[{escape_md(str(iid))}\\] {escape_md(truncate(str(title), 80))}")
            if not page_items:
                lines.append("_No pending ideas_")
            kb = pagination("ideas", 0, total_pages)
            await _safe_send(query, "\n".join(lines), reply_markup=kb)
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=evolve_menu())
        return

    if data == "evolve_enhancements":
        try:
            result = await brain.enhancements(status="pending")
            items = result.get("enhancements", [])
            total = len(items)
            total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
            page_items = items[:PAGE_SIZE]
            lines = [f"*Enhancements* 🔧 \\({escape_md(str(total))} pending\\)\n"]
            for item in page_items:
                iid = item.get("id", "?")
                title = item.get("title", item.get("description", "?"))
                proj = item.get("project", "")
                suffix = f" _\\({escape_md(proj)}\\)_" if proj else ""
                lines.append(f"• \\[{escape_md(str(iid))}\\] {escape_md(truncate(str(title), 70))}{suffix}")
            if not page_items:
                lines.append("_No pending enhancements_")
            kb = pagination("enhancements", 0, total_pages)
            await _safe_send(query, "\n".join(lines), reply_markup=kb)
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=evolve_menu())
        return

    if data == "evolve_discoveries":
        try:
            result = await brain.discoveries(status="discovered")
            items = result.get("discoveries", [])
            total = len(items)
            total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
            page_items = items[:PAGE_SIZE]
            lines = [f"*Discoveries* 🔬 \\({escape_md(str(total))} found\\)\n"]
            for item in page_items:
                iid = item.get("id", "?")
                title = item.get("title", item.get("description", "?"))
                lines.append(f"• \\[{escape_md(str(iid))}\\] {escape_md(truncate(str(title), 80))}")
            if not page_items:
                lines.append("_No new discoveries_")
            kb = pagination("discoveries", 0, total_pages)
            await _safe_send(query, "\n".join(lines), reply_markup=kb)
        except Exception as e:
            await query.edit_message_text(f"Error: {e}", reply_markup=evolve_menu())
        return

    # Approve/Reject actions
    for item_type, update_fn, status_map in [
        ("ideas", brain.idea_update, {"approve": "approved", "reject": "rejected"}),
        ("enhancements", brain.enhancement_update, {"approve": "approved", "reject": "rejected"}),
        ("discoveries", brain.discovery_update, {"approve": "integrated", "reject": "dismissed"}),
    ]:
        for action, new_status in status_map.items():
            prefix = f"{item_type}_{action}_"
            if data.startswith(prefix):
                item_id = int(data[len(prefix):])
                try:
                    await update_fn(item_id, new_status)
                    emoji = "✅" if "approv" in action or "integrat" in action else "❌"
                    await query.edit_message_text(
                        f"{emoji} {item_type.title()[:-1]} #{item_id} → {new_status}",
                        reply_markup=evolve_menu(),
                    )
                except Exception as e:
                    await query.edit_message_text(f"Error: {e}", reply_markup=evolve_menu())
                return
