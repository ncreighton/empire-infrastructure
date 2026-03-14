"""Reusable inline keyboard builders for Telegram."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def main_menu() -> InlineKeyboardMarkup:
    """Main menu with 8 category buttons."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🧠 Brain", callback_data="menu_brain"),
            InlineKeyboardButton("🧬 Evolution", callback_data="menu_evolve"),
        ],
        [
            InlineKeyboardButton("🌐 Sites", callback_data="menu_sites"),
            InlineKeyboardButton("💚 Health", callback_data="menu_health"),
        ],
        [
            InlineKeyboardButton("🔧 Infra", callback_data="menu_infra"),
            InlineKeyboardButton("📝 Content", callback_data="menu_content"),
        ],
        [
            InlineKeyboardButton("📊 Stats", callback_data="menu_stats"),
            InlineKeyboardButton("🏠 Home", callback_data="menu_home"),
        ],
    ])


def brain_menu() -> InlineKeyboardMarkup:
    """Brain sub-menu."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔍 Query", callback_data="brain_query"),
            InlineKeyboardButton("📚 Learn", callback_data="brain_learn"),
        ],
        [
            InlineKeyboardButton("🔗 Patterns", callback_data="brain_patterns"),
            InlineKeyboardButton("💡 Opportunities", callback_data="brain_opps"),
        ],
        [
            InlineKeyboardButton("📋 Briefing", callback_data="brain_briefing"),
            InlineKeyboardButton("🔮 Forecast", callback_data="brain_forecast"),
        ],
        [InlineKeyboardButton("« Back", callback_data="menu_home")],
    ])


def evolve_menu() -> InlineKeyboardMarkup:
    """Evolution sub-menu."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⚡ Quick Cycle", callback_data="evolve_quick"),
            InlineKeyboardButton("🔍 Discover", callback_data="evolve_discover"),
        ],
        [
            InlineKeyboardButton("🌀 Full Cycle", callback_data="evolve_full"),
            InlineKeyboardButton("📊 Status", callback_data="evolve_status"),
        ],
        [
            InlineKeyboardButton("💡 Ideas", callback_data="evolve_ideas"),
            InlineKeyboardButton("🔧 Enhancements", callback_data="evolve_enhancements"),
        ],
        [
            InlineKeyboardButton("🔬 Discoveries", callback_data="evolve_discoveries"),
            InlineKeyboardButton("📈 Adoption", callback_data="evolve_adoption"),
        ],
        [InlineKeyboardButton("« Back", callback_data="menu_home")],
    ])


def health_menu() -> InlineKeyboardMarkup:
    """Health sub-menu."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("💚 All Services", callback_data="health_all"),
            InlineKeyboardButton("🐳 Docker PS", callback_data="health_docker"),
        ],
        [
            InlineKeyboardButton("🧠 Brain Health", callback_data="health_brain"),
            InlineKeyboardButton("📋 Dashboard", callback_data="health_dashboard"),
        ],
        [InlineKeyboardButton("« Back", callback_data="menu_home")],
    ])


def infra_menu() -> InlineKeyboardMarkup:
    """Infrastructure sub-menu."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⚙️ n8n Workflows", callback_data="infra_n8n"),
            InlineKeyboardButton("📜 Logs", callback_data="infra_logs"),
        ],
        [InlineKeyboardButton("« Back", callback_data="menu_home")],
    ])


def content_menu() -> InlineKeyboardMarkup:
    """Content sub-menu."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📄 Articles", callback_data="content_articles"),
            InlineKeyboardButton("🖼️ Images", callback_data="content_images"),
        ],
        [InlineKeyboardButton("« Back", callback_data="menu_home")],
    ])


def stats_menu() -> InlineKeyboardMarkup:
    """Stats sub-menu."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 Overview", callback_data="stats_overview"),
            InlineKeyboardButton("💰 Credits", callback_data="stats_credits"),
        ],
        [
            InlineKeyboardButton("📈 Adoption", callback_data="stats_adoption"),
            InlineKeyboardButton("📏 CLAUDE.md Sizes", callback_data="stats_claudemd"),
        ],
        [InlineKeyboardButton("« Back", callback_data="menu_home")],
    ])


def sites_grid(sites: list[dict]) -> InlineKeyboardMarkup:
    """Build a grid of site buttons from site list."""
    buttons = []
    row = []
    for site in sites:
        sid = site.get("id", site.get("slug", ""))
        name = site.get("name", sid)
        # Truncate long names
        label = name[:15] if len(name) > 15 else name
        row.append(InlineKeyboardButton(label, callback_data=f"site_{sid}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("« Back", callback_data="menu_home")])
    return InlineKeyboardMarkup(buttons)


def item_actions(item_type: str, item_id: int, page: int = 0) -> InlineKeyboardMarkup:
    """Approve/Reject/Skip buttons for an evolution item."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"{item_type}_approve_{item_id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"{item_type}_reject_{item_id}"),
        ],
        [
            InlineKeyboardButton("⏭️ Skip", callback_data=f"{item_type}_list_{page}"),
            InlineKeyboardButton("« Back", callback_data="menu_evolve"),
        ],
    ])


def pagination(prefix: str, page: int, total_pages: int) -> InlineKeyboardMarkup:
    """Pagination buttons."""
    buttons = []
    row = []
    if page > 0:
        row.append(InlineKeyboardButton("« Prev", callback_data=f"{prefix}_list_{page - 1}"))
    if page < total_pages - 1:
        row.append(InlineKeyboardButton("Next »", callback_data=f"{prefix}_list_{page + 1}"))
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("« Back", callback_data="menu_evolve")])
    return InlineKeyboardMarkup(buttons)


def docker_service_actions(service: str) -> InlineKeyboardMarkup:
    """Actions for a specific Docker service."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔄 Restart", callback_data=f"docker_restart_{service}"),
            InlineKeyboardButton("📜 Logs", callback_data=f"docker_logs_{service}"),
        ],
        [InlineKeyboardButton("« Back", callback_data="health_docker")],
    ])


def site_actions(site_id: str) -> InlineKeyboardMarkup:
    """Actions for a specific site."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📄 Recent Posts", callback_data=f"siteact_posts_{site_id}"),
            InlineKeyboardButton("🗑️ Clear Cache", callback_data=f"siteact_cache_{site_id}"),
        ],
        [
            InlineKeyboardButton("📊 Stats", callback_data=f"siteact_stats_{site_id}"),
            InlineKeyboardButton("🧠 Brain Context", callback_data=f"siteact_brain_{site_id}"),
        ],
        [InlineKeyboardButton("« Back", callback_data="menu_sites")],
    ])
