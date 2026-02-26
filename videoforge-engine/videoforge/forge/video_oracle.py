"""VideoOracle — Posting time optimization, seasonal angles, content calendar."""

from datetime import datetime, timedelta
from ..models import OracleRecommendation
from ..knowledge.niche_profiles import NICHE_PROFILES
from ..knowledge.trending_formats import get_trending_formats

# Best posting times by platform (UTC)
_BEST_TIMES = {
    "youtube_shorts": {
        "Monday": ["14:00", "18:00"],
        "Tuesday": ["14:00", "18:00"],
        "Wednesday": ["14:00", "18:00", "21:00"],
        "Thursday": ["14:00", "18:00"],
        "Friday": ["12:00", "15:00", "20:00"],
        "Saturday": ["10:00", "14:00", "18:00"],
        "Sunday": ["10:00", "14:00", "17:00"],
    },
    "tiktok": {
        "Monday": ["15:00", "19:00"],
        "Tuesday": ["15:00", "19:00", "22:00"],
        "Wednesday": ["15:00", "19:00"],
        "Thursday": ["15:00", "19:00", "22:00"],
        "Friday": ["13:00", "17:00", "21:00"],
        "Saturday": ["11:00", "15:00", "19:00"],
        "Sunday": ["11:00", "15:00", "18:00"],
    },
    "youtube": {
        "Monday": ["16:00"],
        "Tuesday": ["16:00"],
        "Wednesday": ["16:00"],
        "Thursday": ["16:00"],
        "Friday": ["15:00"],
        "Saturday": ["12:00"],
        "Sunday": ["12:00"],
    },
    "instagram_reels": {
        "Monday": ["11:00", "17:00"],
        "Tuesday": ["11:00", "17:00"],
        "Wednesday": ["11:00", "17:00"],
        "Thursday": ["11:00", "17:00", "20:00"],
        "Friday": ["11:00", "14:00"],
        "Saturday": ["10:00", "14:00"],
        "Sunday": ["10:00", "14:00"],
    },
    "facebook_reels": {
        "Monday": ["13:00", "17:00"],
        "Tuesday": ["13:00", "17:00"],
        "Wednesday": ["13:00", "17:00"],
        "Thursday": ["13:00", "17:00"],
        "Friday": ["13:00", "16:00"],
        "Saturday": ["12:00", "15:00"],
        "Sunday": ["12:00", "15:00"],
    },
}

# Seasonal content angles by month
_SEASONAL_ANGLES = {
    1: {"season": "winter", "angles": ["New Year resolutions", "winter content", "fresh starts", "goal setting"]},
    2: {"season": "winter", "angles": ["Valentine's Day", "Imbolc", "self-love", "winter tech deals"]},
    3: {"season": "spring", "angles": ["Spring equinox", "Ostara", "spring cleaning", "new beginnings"]},
    4: {"season": "spring", "angles": ["Easter", "spring refresh", "outdoor tech", "garden planning"]},
    5: {"season": "spring", "angles": ["Beltane", "Mother's Day", "outdoor season", "summer prep"]},
    6: {"season": "summer", "angles": ["Litha", "summer solstice", "outdoor activities", "travel tech"]},
    7: {"season": "summer", "angles": ["mid-year review", "summer projects", "vacation content"]},
    8: {"season": "summer", "angles": ["Lughnasadh", "back to school", "harvest prep", "fall preview"]},
    9: {"season": "fall", "angles": ["Mabon", "fall equinox", "autumn aesthetics", "cozy season"]},
    10: {"season": "fall", "angles": ["Samhain", "Halloween", "spooky content", "October vibes"]},
    11: {"season": "fall", "angles": ["Black Friday", "gift guides", "holiday prep", "gratitude"]},
    12: {"season": "winter", "angles": ["Yule", "holiday content", "year review", "gift guides", "winter solstice"]},
}

# Posting frequency recommendations
_FREQUENCY = {
    "youtube_shorts": {"ideal": "1-2/day", "minimum": "3/week", "maximum": "3/day"},
    "tiktok": {"ideal": "1-3/day", "minimum": "4/week", "maximum": "5/day"},
    "youtube": {"ideal": "1-2/week", "minimum": "1/week", "maximum": "3/week"},
    "instagram_reels": {"ideal": "1/day", "minimum": "3/week", "maximum": "2/day"},
    "facebook_reels": {"ideal": "1/day", "minimum": "3/week", "maximum": "2/day"},
}


class VideoOracle:
    """Recommends optimal posting times, seasonal angles, and content calendars."""

    def recommend(self, niche: str, platform: str = "youtube_shorts",
                  dt: datetime = None) -> OracleRecommendation:
        """Full recommendation for a niche + platform combo."""
        if dt is None:
            dt = datetime.utcnow()

        day_name = dt.strftime("%A")
        month = dt.month

        # Best posting time
        platform_times = _BEST_TIMES.get(platform, _BEST_TIMES["youtube_shorts"])
        day_times = platform_times.get(day_name, ["14:00"])
        best_time = day_times[0]

        # Seasonal angle
        seasonal = _SEASONAL_ANGLES.get(month, {"season": "unknown", "angles": []})
        angles = seasonal["angles"]
        seasonal_angle = angles[0] if angles else "evergreen content"

        # Trending formats
        profile = NICHE_PROFILES.get(niche, {})
        category = profile.get("category", "tech")
        trending = get_trending_formats(niche=category, platform=platform)
        trending_keys = [t["key"] for t in trending[:5]]

        # Content calendar (7 days)
        calendar = self._generate_calendar(niche, platform, dt)

        # Frequency
        freq = _FREQUENCY.get(platform, _FREQUENCY["youtube_shorts"])
        freq_rec = freq.get("ideal", "1/day")

        return OracleRecommendation(
            best_post_time=best_time,
            best_day=day_name,
            seasonal_angle=seasonal_angle,
            trending_formats=trending_keys,
            content_calendar=calendar,
            frequency_recommendation=freq_rec,
            competition_level=self._estimate_competition(category),
        )

    def get_best_times(self, platform: str, day: str = None) -> list:
        """Get best posting times for a platform, optionally for a specific day."""
        times = _BEST_TIMES.get(platform, _BEST_TIMES["youtube_shorts"])
        if day:
            return times.get(day, ["14:00"])
        # All times flattened
        all_times = set()
        for day_times in times.values():
            all_times.update(day_times)
        return sorted(all_times)

    def get_seasonal_angle(self, month: int = None) -> dict:
        """Get seasonal angle for a month."""
        if month is None:
            month = datetime.utcnow().month
        return _SEASONAL_ANGLES.get(month, {"season": "unknown", "angles": []})

    def _generate_calendar(self, niche: str, platform: str, start: datetime) -> list:
        """Generate a 7-day content calendar."""
        profile = NICHE_PROFILES.get(niche, {})
        pillars = profile.get("content_pillars", ["content"])
        best_formats = profile.get("best_formats", ["educational", "listicle"])
        category = profile.get("category", "tech")

        calendar = []
        for i in range(7):
            day = start + timedelta(days=i)
            day_name = day.strftime("%A")
            month = day.month

            times = _BEST_TIMES.get(platform, {}).get(day_name, ["14:00"])
            seasonal = _SEASONAL_ANGLES.get(month, {})

            pillar = pillars[i % len(pillars)]
            fmt = best_formats[i % len(best_formats)]

            calendar.append({
                "date": day.strftime("%Y-%m-%d"),
                "day": day_name,
                "post_times": times,
                "content_pillar": pillar,
                "suggested_format": fmt,
                "seasonal_context": seasonal.get("season", ""),
            })

        return calendar

    def _estimate_competition(self, category: str) -> str:
        """Rough competition estimate by niche category."""
        high_competition = {"tech", "business", "fitness", "lifestyle"}
        medium_competition = {"ai_news", "review", "journal"}
        if category in high_competition:
            return "high"
        if category in medium_competition:
            return "medium"
        return "low"
