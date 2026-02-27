"""Content calendar — weekly schedule and seasonal timing."""

from datetime import datetime, timedelta


# Weekly content schedule
WEEKLY_SCHEDULE = {
    0: {"type": "article", "label": "Monday: How-To Article"},
    1: {"type": "post", "label": "Tuesday: Quick Tip"},
    2: {"type": "review", "label": "Wednesday: Product Review"},
    3: {"type": "post", "label": "Thursday: Discussion Post"},
    4: {"type": "article", "label": "Friday: Guide Article"},
    5: {"type": "listing", "label": "Saturday: Model Listing"},
    6: {"type": "post", "label": "Sunday: Community Showcase"},
}

# Seasonal themes by month
SEASONAL_THEMES = {
    1: {
        "theme": "New Year Organization",
        "topics": ["desk organizers", "cable management", "label holders"],
        "note": "Resolution-driven functional prints",
    },
    2: {
        "theme": "Valentine's Day Gifts",
        "topics": ["heart designs", "gift boxes", "lithophanes"],
        "note": "Gift-focused content peaks mid-February",
    },
    3: {
        "theme": "Spring Cleaning & Garden",
        "topics": ["storage solutions", "plant pots", "seed starters"],
        "note": "Transition to outdoor/garden content",
    },
    4: {
        "theme": "Spring Projects",
        "topics": ["garden tools", "outdoor accessories", "Easter prints"],
        "note": "Hobby season ramp-up",
    },
    5: {
        "theme": "Maker Month",
        "topics": ["functional upgrades", "printer mods", "workspace tools"],
        "note": "Community building, skill sharing",
    },
    6: {
        "theme": "Summer Fun",
        "topics": ["outdoor games", "water accessories", "travel gadgets"],
        "note": "Portable and outdoor prints",
    },
    7: {
        "theme": "Mid-Year Printer Guide",
        "topics": ["printer roundups", "filament tests", "upgrade guides"],
        "note": "Prime Day deals content",
    },
    8: {
        "theme": "Back to School",
        "topics": ["desk accessories", "pen holders", "organizers"],
        "note": "Student and teacher focused",
    },
    9: {
        "theme": "Fall Prep",
        "topics": ["coasters", "candle holders", "home decor"],
        "note": "Cozy home prints",
    },
    10: {
        "theme": "Halloween Prints",
        "topics": ["decorations", "costumes", "spooky designs"],
        "note": "HIGH ENGAGEMENT — Halloween content peaks",
    },
    11: {
        "theme": "Gift Guide Season",
        "topics": ["gift ideas", "stocking stuffers", "personalized prints"],
        "note": "Start gift content early November",
    },
    12: {
        "theme": "Holiday Gifts & Year-End",
        "topics": ["ornaments", "gift boxes", "print farm tips"],
        "note": "Peak gift-giving content",
    },
}


class ContentCalendar:
    """Manages content scheduling and seasonal timing."""

    def weekly_plan(self) -> list[dict]:
        """Get the content plan for the current week."""
        today = datetime.now()
        # Start from Monday of current week
        monday = today - timedelta(days=today.weekday())

        plan = []
        for day_offset in range(7):
            date = monday + timedelta(days=day_offset)
            schedule = WEEKLY_SCHEDULE[day_offset]
            seasonal = SEASONAL_THEMES.get(date.month, {})

            plan.append({
                "date": date.strftime("%Y-%m-%d"),
                "day": date.strftime("%A"),
                "content_type": schedule["type"],
                "label": schedule["label"],
                "seasonal_theme": seasonal.get("theme", ""),
                "seasonal_topics": seasonal.get("topics", []),
                "is_today": date.date() == today.date(),
            })

        return plan

    def get_seasonal_context(self) -> str:
        """Get seasonal context string for content generation."""
        month = datetime.now().month
        seasonal = SEASONAL_THEMES.get(month, {})

        if not seasonal:
            return ""

        parts = [f"Current seasonal theme: {seasonal['theme']}"]
        if seasonal.get("topics"):
            parts.append(f"Hot topics: {', '.join(seasonal['topics'])}")
        if seasonal.get("note"):
            parts.append(f"Note: {seasonal['note']}")

        # Look ahead
        next_month = (month % 12) + 1
        next_seasonal = SEASONAL_THEMES.get(next_month, {})
        if next_seasonal:
            parts.append(f"Coming next: {next_seasonal['theme']}")

        return " | ".join(parts)

    def todays_content_type(self) -> str:
        """Get the content type scheduled for today."""
        weekday = datetime.now().weekday()
        return WEEKLY_SCHEDULE[weekday]["type"]

    def format_calendar(self) -> str:
        """Format the weekly calendar for display."""
        plan = self.weekly_plan()
        lines = ["Weekly Content Calendar", "=" * 50]

        for day in plan:
            marker = " >>> TODAY" if day["is_today"] else ""
            line = f"  {day['date']} {day['label']}{marker}"
            if day["seasonal_theme"]:
                line += f"\n    Theme: {day['seasonal_theme']}"
            lines.append(line)

        month = datetime.now().month
        seasonal = SEASONAL_THEMES.get(month, {})
        if seasonal:
            lines.extend([
                "",
                f"Monthly Theme: {seasonal['theme']}",
                f"Focus: {seasonal.get('note', '')}",
            ])

        return "\n".join(lines)
