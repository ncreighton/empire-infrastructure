"""Opportunity Finder — Discovers and scores content opportunities across the empire."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SITES_CONFIG = Path(__file__).parent.parent.parent.parent / "config" / "sites.json"


class OpportunityFinder:
    """Discovers content opportunities by analyzing keyword data across all sites."""

    def __init__(self):
        from .codex import OpportunityCodex
        from .scoring import OpportunityScorer
        from .data_sources import DataSourceAggregator
        from .seasonal import SeasonalDetector

        self.codex = OpportunityCodex()
        self.scorer = OpportunityScorer()
        self.data = DataSourceAggregator()
        self.seasonal = SeasonalDetector()
        self.sites = self._load_sites()

    def _load_sites(self) -> Dict:
        if SITES_CONFIG.exists():
            try:
                data = json.loads(SITES_CONFIG.read_text("utf-8"))
                return data.get("sites", data)
            except Exception:
                return {}
        return {}

    def scan_site(self, site_slug: str) -> Dict:
        """Find opportunities for a single site."""
        log.info(f"Scanning opportunities for {site_slug}...")

        # Get striking-distance keywords
        keywords = self.data.get_striking_distance_keywords(site_slug)
        if not keywords:
            log.info(f"No striking-distance keywords found for {site_slug}")
            return {"site": site_slug, "opportunities": [], "total": 0}

        # Get content inventory for gap detection
        inventory = self.data.get_content_inventory(site_slug)
        existing_urls = {a.get("url", "") for a in inventory}

        opportunities = []
        for kw in keywords:
            keyword = kw.get("keyword", "")

            # Determine opportunity type
            has_content = kw.get("url") in existing_urls if kw.get("url") else False
            opp_type = "optimize" if has_content else "content_gap"

            # Check cross-site presence
            cross_site_count = 0
            try:
                cross = self.data.find_keyword_across_sites(keyword)
                cross_site_count = len({r["site_slug"] for r in cross}) - 1
            except Exception:
                pass

            # Seasonal boost
            seasonal_boost = self.seasonal.get_seasonal_boost(keyword)

            # Score
            scored = self.scorer.score(
                keyword_data={
                    "keyword": keyword,
                    "position": kw.get("position", 100),
                    "impressions": kw.get("impressions", 0),
                    "clicks": kw.get("clicks", 0),
                    "ctr": kw.get("ctr", 0),
                    "existing_url": kw.get("url"),
                },
                site_slug=site_slug,
                cross_site_count=cross_site_count,
                seasonal_boost=seasonal_boost,
            )

            # Store in codex
            opp_id = self.codex.upsert_opportunity(
                site_slug=site_slug,
                keyword=keyword,
                opp_type=opp_type,
                score=scored["composite_score"],
                dimensions=scored["dimensions"],
                details={
                    "grade": scored["grade"],
                    "cross_site_count": cross_site_count,
                    "seasonal_boost": seasonal_boost,
                },
            )

            opportunities.append({
                "id": opp_id,
                "keyword": keyword,
                "type": opp_type,
                "score": scored["composite_score"],
                "grade": scored["grade"],
                "position": kw.get("position"),
                "impressions": kw.get("impressions"),
            })

        # Sort by score
        opportunities.sort(key=lambda x: x["score"], reverse=True)

        # Log snapshot
        top_score = opportunities[0]["score"] if opportunities else 0
        self.codex.log_snapshot(
            site_slug, len(opportunities), top_score,
            f"Found {len(opportunities)} opportunities"
        )

        # Publish event
        try:
            from core.event_bus import publish
            publish("opportunity.scan_complete", {
                "site": site_slug,
                "found": len(opportunities),
                "top_score": top_score,
            }, "opportunity_finder")
        except Exception:
            pass

        return {
            "site": site_slug,
            "opportunities": opportunities[:20],
            "total": len(opportunities),
        }

    def run_daily_scan(self) -> Dict:
        """Scan all sites for opportunities."""
        log.info("Running daily opportunity scan across all sites...")

        if not self.data.available:
            # Fallback: scan sites from config
            site_slugs = list(self.sites.keys())
        else:
            site_slugs = self.data.get_all_site_slugs()
            # Also include sites from config that might not have data yet
            for slug in self.sites:
                if slug not in site_slugs:
                    site_slugs.append(slug)

        results = []
        total_found = 0

        for slug in site_slugs:
            try:
                result = self.scan_site(slug)
                results.append(result)
                total_found += result.get("total", 0)
            except Exception as e:
                log.error(f"Scan failed for {slug}: {e}")
                results.append({"site": slug, "error": str(e)})

        # Publish summary event
        try:
            from core.event_bus import publish
            publish("opportunity.daily_scan_complete", {
                "sites_scanned": len(results),
                "total_opportunities": total_found,
            }, "opportunity_finder")
        except Exception:
            pass

        return {
            "sites_scanned": len(results),
            "total_opportunities": total_found,
            "results": results,
        }

    def get_queue(self, site_slug: str = None, limit: int = 20) -> List[Dict]:
        """Get the prioritized opportunity queue."""
        return self.codex.get_queue(site_slug, limit)

    def get_cross_site(self) -> List[Dict]:
        """Get cross-site keyword opportunities."""
        return self.codex.get_cross_site_keywords()

    def get_upcoming_seasonal(self, months: int = 3) -> List[Dict]:
        """Get upcoming seasonal opportunities."""
        return self.seasonal.get_upcoming_seasons(months)

    def get_stats(self) -> Dict:
        """Opportunity system statistics."""
        return self.codex.stats()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Opportunity Finder")
    parser.add_argument("--scan", action="store_true", help="Run daily scan")
    parser.add_argument("--site", help="Scan specific site")
    parser.add_argument("--queue", action="store_true", help="Show opportunity queue")
    parser.add_argument("--cross-site", action="store_true", help="Show cross-site opportunities")
    parser.add_argument("--seasonal", action="store_true", help="Show upcoming seasonal")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    finder = OpportunityFinder()

    if args.scan and args.site:
        result = finder.scan_site(args.site)
    elif args.scan:
        result = finder.run_daily_scan()
    elif args.queue:
        result = finder.get_queue(args.site)
    elif args.cross_site:
        result = finder.get_cross_site()
    elif args.seasonal:
        result = finder.get_upcoming_seasonal()
    else:
        result = finder.get_stats()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
