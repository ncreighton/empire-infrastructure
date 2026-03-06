"""Seasonal Calendar — Detects seasonal urgency for keywords."""

from datetime import datetime
from typing import Dict, Optional

# Seasonal keyword patterns by month (1-12)
SEASONAL_KEYWORDS = {
    # Holiday / Calendar
    "christmas": {"peak": 12, "ramp": 10, "multiplier": 3.0},
    "halloween": {"peak": 10, "ramp": 8, "multiplier": 2.5},
    "valentines": {"peak": 2, "ramp": 1, "multiplier": 2.0},
    "mothers day": {"peak": 5, "ramp": 4, "multiplier": 2.0},
    "fathers day": {"peak": 6, "ramp": 5, "multiplier": 1.8},
    "black friday": {"peak": 11, "ramp": 10, "multiplier": 3.0},
    "new year": {"peak": 1, "ramp": 12, "multiplier": 2.0},
    "easter": {"peak": 4, "ramp": 3, "multiplier": 1.5},
    "back to school": {"peak": 8, "ramp": 7, "multiplier": 2.0},
    "cyber monday": {"peak": 11, "ramp": 10, "multiplier": 2.5},

    # Niche-specific seasonal
    "spring garden": {"peak": 3, "ramp": 2, "multiplier": 2.0},
    "summer": {"peak": 6, "ramp": 5, "multiplier": 1.5},
    "winter solstice": {"peak": 12, "ramp": 11, "multiplier": 2.0},
    "samhain": {"peak": 10, "ramp": 9, "multiplier": 2.5},
    "beltane": {"peak": 5, "ramp": 4, "multiplier": 2.0},
    "imbolc": {"peak": 2, "ramp": 1, "multiplier": 1.8},
    "litha": {"peak": 6, "ramp": 5, "multiplier": 1.8},
    "mabon": {"peak": 9, "ramp": 8, "multiplier": 1.8},
    "yule": {"peak": 12, "ramp": 11, "multiplier": 2.0},
    "ostara": {"peak": 3, "ramp": 2, "multiplier": 1.8},

    # Smart home / tech
    "prime day": {"peak": 7, "ramp": 6, "multiplier": 2.5},
    "gift guide": {"peak": 12, "ramp": 10, "multiplier": 2.5},
    "best deals": {"peak": 11, "ramp": 10, "multiplier": 2.0},

    # Bullet journal
    "planner": {"peak": 1, "ramp": 11, "multiplier": 2.0},
    "goal setting": {"peak": 1, "ramp": 12, "multiplier": 2.0},
    "habit tracker": {"peak": 1, "ramp": 12, "multiplier": 1.8},
}


class SeasonalDetector:
    """Detects seasonal urgency for keywords."""

    def get_seasonal_boost(self, keyword: str) -> float:
        """
        Returns a 0-70 seasonal urgency boost based on keyword and current month.
        Higher if we're in the ramp-up period for a seasonal keyword.
        """
        now = datetime.now()
        current_month = now.month
        keyword_lower = keyword.lower()

        best_boost = 0.0

        for pattern, config in SEASONAL_KEYWORDS.items():
            if pattern in keyword_lower:
                peak = config["peak"]
                ramp = config["ramp"]
                multiplier = config["multiplier"]

                # Calculate months until peak
                months_until = (peak - current_month) % 12

                if months_until == 0:
                    # We're in peak month
                    boost = 70 * multiplier / 3.0
                elif months_until <= 1:
                    # One month before peak
                    boost = 60 * multiplier / 3.0
                elif months_until <= 2:
                    # Two months before peak (ramp-up zone)
                    boost = 45 * multiplier / 3.0
                elif months_until <= 3:
                    # Three months before — good time to start content
                    boost = 30 * multiplier / 3.0
                else:
                    boost = 0

                best_boost = max(best_boost, boost)

        return round(min(70, best_boost), 1)

    def get_upcoming_seasons(self, months_ahead: int = 3) -> list:
        """Get seasonal keywords that will peak in the next N months."""
        now = datetime.now()
        current_month = now.month
        upcoming = []

        for pattern, config in SEASONAL_KEYWORDS.items():
            months_until = (config["peak"] - current_month) % 12
            if 0 < months_until <= months_ahead:
                upcoming.append({
                    "keyword_pattern": pattern,
                    "peak_month": config["peak"],
                    "months_until_peak": months_until,
                    "multiplier": config["multiplier"],
                })

        upcoming.sort(key=lambda x: x["months_until_peak"])
        return upcoming
