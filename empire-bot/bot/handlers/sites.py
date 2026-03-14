"""Site management handlers: list sites, site details, posts, cache."""

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.services.brain_client import BrainClient
from bot.services.wp_client import WPClient, get_site_ids, get_site_config
from bot.utils.auth import admin_only
from bot.utils.formatters import escape_md, truncate, format_timestamp
from bot.utils.keyboards import sites_grid, site_actions, main_menu

logger = logging.getLogger(__name__)
brain = BrainClient()


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
async def cmd_sites(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /sites — list all sites."""
    site_ids = get_site_ids()
    sites = []
    for sid in sorted(site_ids):
        cfg = get_site_config(sid)
        name = cfg.get("name", sid) if cfg else sid
        sites.append({"id": sid, "name": name})

    await update.message.reply_text(
        f"*Empire Sites* \\({escape_md(str(len(sites)))}\\)\n\nTap a site for details:",
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=sites_grid(sites),
    )


@admin_only
async def cmd_site(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /site <name> — site details."""
    if not context.args:
        await update.message.reply_text("Usage: /site <site_id>")
        return

    site_id = context.args[0].lower()
    cfg = get_site_config(site_id)
    if not cfg:
        await update.message.reply_text(f"Unknown site: {site_id}")
        return

    await update.message.chat.send_action("typing")

    name = cfg.get("name", site_id)
    domain = cfg.get("domain", "?")

    lines = [
        f"*{escape_md(name)}*\n",
        f"Domain: {escape_md(domain)}",
    ]

    # Try to get post count
    try:
        wp = WPClient(site_id)
        count = await wp.post_count()
        lines.append(f"Published Posts: {escape_md(str(count))}")
    except Exception:
        pass

    # Try to get brain context
    try:
        ctx = await brain.site_context(site_id)
        site_data = ctx.get("site", {})
        if site_data.get("category"):
            lines.append(f"Category: {escape_md(site_data['category'])}")
        skills = ctx.get("skills", [])
        if skills:
            lines.append(f"Skills: {escape_md(str(len(skills)))}")
    except Exception:
        pass

    await _safe_send(update, "\n".join(lines), reply_markup=site_actions(site_id))


@admin_only
async def cmd_posts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /posts <site> — recent posts."""
    if not context.args:
        await update.message.reply_text("Usage: /posts <site_id>")
        return

    site_id = context.args[0].lower()
    await update.message.chat.send_action("typing")

    try:
        wp = WPClient(site_id)
        posts = await wp.posts(per_page=10)

        cfg = get_site_config(site_id)
        name = cfg.get("name", site_id) if cfg else site_id

        lines = [f"*Recent Posts \\- {escape_md(name)}*\n"]
        for p in posts:
            title = p.get("title", {}).get("rendered", "Untitled")
            date = format_timestamp(p.get("date", ""))
            pid = p.get("id", "?")
            lines.append(f"• \\[{escape_md(str(pid))}\\] {escape_md(truncate(title, 60))} _\\({escape_md(date)}\\)_")

        if not posts:
            lines.append("_No posts found_")

        await _safe_send(update, "\n".join(lines), reply_markup=site_actions(site_id))
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def sites_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle site-related callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    # Menu sites button
    if data == "menu_sites":
        site_ids = get_site_ids()
        sites = [{"id": sid, "name": (get_site_config(sid) or {}).get("name", sid)} for sid in sorted(site_ids)]
        await _safe_send(
            query,
            f"*Empire Sites* \\({escape_md(str(len(sites)))}\\)\n\nTap a site:",
            reply_markup=sites_grid(sites),
        )
        return

    # Site selection: site_<id>
    if data.startswith("site_"):
        site_id = data[5:]
        cfg = get_site_config(site_id)
        if not cfg:
            await query.edit_message_text(f"Unknown site: {site_id}")
            return

        name = cfg.get("name", site_id)
        domain = cfg.get("domain", "?")
        lines = [f"*{escape_md(name)}*\n", f"Domain: {escape_md(domain)}"]

        try:
            wp = WPClient(site_id)
            count = await wp.post_count()
            lines.append(f"Posts: {escape_md(str(count))}")
        except Exception:
            pass

        await _safe_send(query, "\n".join(lines), reply_markup=site_actions(site_id))
        return

    # Site actions: siteact_<action>_<site_id>
    if data.startswith("siteact_"):
        parts = data.split("_", 2)
        if len(parts) < 3:
            return
        action = parts[1]
        site_id = parts[2]

        if action == "posts":
            try:
                wp = WPClient(site_id)
                posts = await wp.posts(per_page=10)
                lines = [f"*Recent Posts*\n"]
                for p in posts:
                    title = p.get("title", {}).get("rendered", "Untitled")
                    pid = p.get("id", "?")
                    lines.append(f"• \\[{escape_md(str(pid))}\\] {escape_md(truncate(title, 60))}")
                if not posts:
                    lines.append("_No posts_")
                await _safe_send(query, "\n".join(lines), reply_markup=site_actions(site_id))
            except Exception as e:
                await query.edit_message_text(f"Error: {e}", reply_markup=site_actions(site_id))

        elif action == "cache":
            try:
                wp = WPClient(site_id)
                cleared = await wp.clear_cache()
                msg = "Cache cleared ✅" if cleared else "No cache plugin responded"
                await query.edit_message_text(msg, reply_markup=site_actions(site_id))
            except Exception as e:
                await query.edit_message_text(f"Error: {e}", reply_markup=site_actions(site_id))

        elif action == "stats":
            try:
                wp = WPClient(site_id)
                health = await wp.site_health()
                lines = [f"*Site Health*\n"]
                lines.append(f"Status: {'🟢' if health.get('ok') else '🔴'} {escape_md(str(health.get('status_code', '?')))}")
                if health.get("response_time_ms"):
                    lines.append(f"Response: {escape_md(str(health['response_time_ms']))}ms")
                if health.get("error"):
                    lines.append(f"Error: {escape_md(health['error'])}")
                await _safe_send(query, "\n".join(lines), reply_markup=site_actions(site_id))
            except Exception as e:
                await query.edit_message_text(f"Error: {e}", reply_markup=site_actions(site_id))

        elif action == "brain":
            try:
                ctx = await brain.site_context(site_id)
                lines = [f"*Brain Context*\n"]
                site_data = ctx.get("site", {})
                for k in ("category", "health_score", "function_count"):
                    if site_data.get(k):
                        lines.append(f"{escape_md(k)}: {escape_md(str(site_data[k]))}")
                skills = ctx.get("skills", [])
                lines.append(f"Skills: {escape_md(str(len(skills)))}")
                learnings = ctx.get("learnings", [])
                lines.append(f"Learnings: {escape_md(str(len(learnings)))}")
                await _safe_send(query, "\n".join(lines), reply_markup=site_actions(site_id))
            except Exception as e:
                await query.edit_message_text(f"Error: {e}", reply_markup=site_actions(site_id))
