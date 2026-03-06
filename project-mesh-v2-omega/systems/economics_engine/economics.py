"""Empire Economics Engine — Tracks costs, revenue, ROI across the empire."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Known API costs per unit
API_COSTS = {
    "anthropic_haiku": {"per_1m_input": 0.80, "per_1m_output": 4.00},
    "anthropic_sonnet": {"per_1m_input": 3.00, "per_1m_output": 15.00},
    "anthropic_opus": {"per_1m_input": 15.00, "per_1m_output": 75.00},
    "elevenlabs_tts": {"per_1k_chars": 0.03},
    "fal_image": {"per_image": 0.04},
    "creatomate_render": {"per_render": 0.50},
    "openrouter_llm": {"per_1m_input": 1.50, "per_1m_output": 7.50},
    "pexels": {"per_request": 0.00},  # Free
    "wordpress_hosting": {"per_month_per_site": 5.00},
}

# Estimated per-article costs by pipeline step
STEP_COST_ESTIMATES = {
    "article_generation": 0.15,   # ~5K tokens Sonnet
    "image_generation": 0.20,     # 5 images via FAL
    "video_creation": 0.50,       # Full VideoForge pipeline
    "social_captions": 0.02,      # Haiku for captions
    "wordpress_publish": 0.00,    # Free (API call)
    "email_newsletter": 0.00,     # Free tier
}


class EconomicsEngine:
    """Tracks and calculates empire-wide economics."""

    def __init__(self):
        from .codex import EconomicsCodex
        self.codex = EconomicsCodex()

    def log_cascade_cost(self, site_slug: str, title: str,
                          steps_completed: List[str]) -> Dict:
        """Log costs for a completed cascade based on steps used."""
        total_cost = 0
        breakdown = {}

        for step in steps_completed:
            cost = STEP_COST_ESTIMATES.get(step, 0)
            if cost > 0:
                breakdown[step] = cost
                total_cost += cost
                self.codex.log_cost(site_slug, "content_pipeline", step, cost,
                                     f"Cascade: {title}")

        return {"total_cost": round(total_cost, 2), "breakdown": breakdown}

    def log_revenue(self, site_slug: str, source: str, amount: float,
                    description: str = None):
        """Log a revenue event."""
        self.codex.log_revenue(site_slug, source, amount, description)

        try:
            from core.event_bus import publish
            publish("economics.revenue", {
                "site": site_slug,
                "source": source,
                "amount": amount,
            }, "economics_engine")
        except Exception:
            pass

    def calculate_article_roi(self, site_slug: str, title: str,
                               cost: float, revenue: float) -> Dict:
        """Calculate and store article-level ROI."""
        self.codex.update_article_economics(site_slug, title, cost, revenue)
        roi = ((revenue - cost) / cost * 100) if cost > 0 else 0
        return {
            "title": title,
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "roi": round(roi, 1),
            "profit": round(revenue - cost, 2),
        }

    def recalculate_site(self, site_slug: str):
        """Recalculate site economics from all article data."""
        self.codex.update_site_economics(site_slug)

    def get_empire_pnl(self) -> Dict:
        """Full empire P&L."""
        summary = self.codex.get_empire_summary()

        try:
            from core.event_bus import publish
            publish("economics.pnl_calculated", {
                "total_revenue": summary["total_revenue"],
                "total_cost": summary["total_cost"],
                "roi": summary["total_roi"],
            }, "economics_engine")
        except Exception:
            pass

        return summary

    def get_site_pnl(self, site_slug: str) -> Optional[Dict]:
        return self.codex.get_site_summary(site_slug)

    def get_allocation_recommendations(self) -> List[Dict]:
        """Recommend investment allocation based on ROI data."""
        summary = self.codex.get_empire_summary()
        sites = summary.get("sites", [])

        if not sites:
            return [{"recommendation": "No data yet — run cascades to generate cost data"}]

        # Sort by ROI
        positive_roi = [s for s in sites if s.get("roi", 0) > 0]
        negative_roi = [s for s in sites if s.get("roi", 0) <= 0]

        recommendations = []
        for site in sorted(positive_roi, key=lambda x: x.get("roi", 0), reverse=True)[:5]:
            recommendations.append({
                "site": site["site_slug"],
                "action": "increase_investment",
                "roi": site["roi"],
                "reason": f"ROI of {site['roi']:.0f}% — increase content production",
            })

        for site in negative_roi[:3]:
            recommendations.append({
                "site": site["site_slug"],
                "action": "optimize_or_pause",
                "roi": site["roi"],
                "reason": f"Negative ROI ({site['roi']:.0f}%) — optimize content or pause",
            })

        return recommendations

    def get_top_articles(self, site_slug: str = None) -> List[Dict]:
        return self.codex.get_top_articles(site_slug)

    def get_cost_reference(self) -> Dict:
        """Return the API cost reference table."""
        return {"api_costs": API_COSTS, "step_estimates": STEP_COST_ESTIMATES}

    def get_stats(self) -> Dict:
        return self.codex.stats()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Empire Economics Engine")
    parser.add_argument("--empire", action="store_true", help="Empire P&L")
    parser.add_argument("--site", help="Site P&L")
    parser.add_argument("--allocation", action="store_true", help="Investment recommendations")
    parser.add_argument("--costs", action="store_true", help="Cost reference")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    engine = EconomicsEngine()

    if args.empire:
        result = engine.get_empire_pnl()
    elif args.site:
        result = engine.get_site_pnl(args.site) or {"error": "No data for site"}
    elif args.allocation:
        result = engine.get_allocation_recommendations()
    elif args.costs:
        result = engine.get_cost_reference()
    else:
        result = engine.get_stats()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
