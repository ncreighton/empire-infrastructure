"""Opportunity Scorer — 5-dimension composite scoring for content opportunities."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Scoring weights (sum to 1.0)
WEIGHTS = {
    "traffic_potential": 0.30,
    "monetization_fit": 0.25,
    "effort": 0.15,
    "cross_site_synergy": 0.15,
    "seasonal_urgency": 0.15,
}

# Amazon affiliate niches with high commission potential
HIGH_MONETIZATION_NICHES = {
    "smarthomewizards", "smarthomegearreviews", "pulsegearreviews",
    "wearablegearreviews", "theconnectedhaven",
}

# Niches with digital product potential
DIGITAL_PRODUCT_NICHES = {
    "witchcraftforbeginners", "bulletjournals", "manifestandalign",
    "mythicalarchives",
}

SITES_CONFIG = Path(__file__).parent.parent.parent.parent / "config" / "sites.json"


def _load_sites() -> Dict:
    if SITES_CONFIG.exists():
        try:
            data = json.loads(SITES_CONFIG.read_text("utf-8"))
            return data.get("sites", data)
        except Exception:
            return {}
    return {}


class OpportunityScorer:
    """Scores opportunities across 5 dimensions."""

    def __init__(self):
        self.sites = _load_sites()

    def score(self, keyword_data: Dict, site_slug: str,
              cross_site_count: int = 0, seasonal_boost: float = 0) -> Dict:
        """
        Score an opportunity on 5 dimensions (each 0-100), return composite.

        keyword_data should contain: position, impressions, clicks, ctr, keyword
        """
        position = keyword_data.get("position", 100)
        impressions = keyword_data.get("impressions", 0)
        clicks = keyword_data.get("clicks", 0)

        dimensions = {}

        # 1. Traffic Potential (0-100)
        # Striking distance (5-20) with high impressions = high potential
        if 5 <= position <= 10:
            pos_score = 90
        elif 10 < position <= 15:
            pos_score = 70
        elif 15 < position <= 20:
            pos_score = 50
        elif position < 5:
            pos_score = 30  # Already ranking well
        else:
            pos_score = 20  # Too far

        imp_score = min(100, impressions / 50)  # 5000 impressions = max
        dimensions["traffic_potential"] = round((pos_score * 0.6 + imp_score * 0.4), 1)

        # 2. Monetization Fit (0-100)
        mon_score = 50  # Default
        if site_slug in HIGH_MONETIZATION_NICHES:
            mon_score = 80
            keyword_lower = keyword_data.get("keyword", "").lower()
            if any(w in keyword_lower for w in ["best", "review", "vs", "compare", "buy"]):
                mon_score = 95
        elif site_slug in DIGITAL_PRODUCT_NICHES:
            mon_score = 70
        dimensions["monetization_fit"] = mon_score

        # 3. Effort (0-100, higher = easier/less effort)
        has_existing = bool(keyword_data.get("existing_url"))
        if has_existing and position <= 20:
            effort_score = 85  # Just needs optimization
        elif has_existing:
            effort_score = 60  # Needs significant rewrite
        else:
            effort_score = 40  # New content needed
        dimensions["effort"] = effort_score

        # 4. Cross-Site Synergy (0-100)
        if cross_site_count >= 3:
            synergy = 90
        elif cross_site_count >= 2:
            synergy = 70
        elif cross_site_count >= 1:
            synergy = 40
        else:
            synergy = 10
        dimensions["cross_site_synergy"] = synergy

        # 5. Seasonal Urgency (0-100)
        base_seasonal = 30  # Evergreen default
        dimensions["seasonal_urgency"] = min(100, base_seasonal + seasonal_boost)

        # Composite score
        composite = sum(
            dimensions[dim] * weight
            for dim, weight in WEIGHTS.items()
        )

        # Store metadata
        dimensions["current_position"] = position
        dimensions["impressions"] = impressions
        dimensions["clicks"] = clicks
        dimensions["existing_url"] = keyword_data.get("existing_url")

        return {
            "composite_score": round(composite, 1),
            "dimensions": dimensions,
            "grade": self._grade(composite),
        }

    def _grade(self, score: float) -> str:
        if score >= 80:
            return "A"
        if score >= 65:
            return "B"
        if score >= 50:
            return "C"
        if score >= 35:
            return "D"
        return "F"
