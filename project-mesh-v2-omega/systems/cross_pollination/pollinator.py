"""Cross-Pollination Engine — Detects keyword overlaps and suggests cross-site linking."""

import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SITES_CONFIG = Path(__file__).parent.parent.parent.parent / "config" / "sites.json"

# Niche clusters — sites that share potential audiences
NICHE_CLUSTERS = {
    "smart_home": ["smarthomewizards", "smarthomegearreviews", "theconnectedhaven"],
    "ai_tech": ["wealthfromai", "aidiscoverydigest", "aiinactionhub", "clearainews"],
    "wearable_tech": ["pulsegearreviews", "wearablegearreviews"],
    "spiritual": ["witchcraftforbeginners", "manifestandalign", "mythicalarchives"],
    "lifestyle": ["bulletjournals", "familyflourish", "celebrationseason"],
}


def _get_supabase():
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if url and key:
            return create_client(url, key)
    except ImportError:
        pass
    return None


class CrossPollinationEngine:
    """Detects keyword overlaps across sites and suggests cross-linking."""

    def __init__(self):
        from .codex import PollinationCodex
        self.codex = PollinationCodex()
        self.supabase = _get_supabase()
        self.sites = self._load_sites()

    def _load_sites(self) -> Dict:
        if SITES_CONFIG.exists():
            try:
                data = json.loads(SITES_CONFIG.read_text("utf-8"))
                return data.get("sites", data)
            except Exception:
                return {}
        return {}

    def detect_overlaps(self) -> Dict:
        """Detect keyword overlaps between all site pairs."""
        if not self.supabase:
            return {"status": "unavailable", "reason": "Supabase not configured"}

        site_slugs = list(self.sites.keys())
        site_keywords = {}

        # Fetch keywords for each site
        for slug in site_slugs:
            try:
                resp = self.supabase.table("keyword_rankings") \
                    .select("keyword") \
                    .eq("site_slug", slug) \
                    .limit(500) \
                    .execute()
                site_keywords[slug] = {r["keyword"].lower() for r in (resp.data or [])}
            except Exception as e:
                log.debug(f"Failed to get keywords for {slug}: {e}")
                site_keywords[slug] = set()

        # Compare all pairs
        overlaps = []
        for site_a, site_b in combinations(site_slugs, 2):
            kw_a = site_keywords.get(site_a, set())
            kw_b = site_keywords.get(site_b, set())

            if not kw_a or not kw_b:
                continue

            shared = kw_a & kw_b
            if shared:
                overlap_score = len(shared) / min(len(kw_a), len(kw_b)) * 100
                self.codex.upsert_overlap(
                    site_a, site_b, len(shared), overlap_score,
                    list(shared)[:10]
                )
                overlaps.append({
                    "sites": [site_a, site_b],
                    "shared_keywords": len(shared),
                    "overlap_score": round(overlap_score, 1),
                    "sample": list(shared)[:5],
                })

        # Save clusters
        for cluster_name, cluster_sites in NICHE_CLUSTERS.items():
            active_sites = [s for s in cluster_sites if s in site_keywords]
            if len(active_sites) >= 2:
                all_kw = set()
                for s in active_sites:
                    all_kw.update(site_keywords.get(s, set()))
                self.codex.save_cluster(
                    cluster_name, active_sites, len(all_kw),
                    f"Cluster with {len(active_sites)} sites, {len(all_kw)} total keywords"
                )

        try:
            from core.event_bus import publish
            publish("pollination.overlap_scan", {
                "pairs_analyzed": len(overlaps),
                "significant": sum(1 for o in overlaps if o["overlap_score"] > 5),
            }, "cross_pollination")
        except Exception:
            pass

        return {
            "pairs_analyzed": len(overlaps),
            "overlaps": sorted(overlaps, key=lambda x: x["overlap_score"], reverse=True)[:20],
        }

    def suggest_links(self, source_site: str, target_site: str) -> List[Dict]:
        """Suggest cross-site links based on shared keywords."""
        if not self.supabase:
            return []

        suggestions = []
        try:
            # Get shared keywords
            source_kw = self.supabase.table("keyword_rankings") \
                .select("keyword,url,clicks") \
                .eq("site_slug", source_site) \
                .order("clicks", desc=True) \
                .limit(200) \
                .execute()

            target_kw = self.supabase.table("keyword_rankings") \
                .select("keyword,url,clicks") \
                .eq("site_slug", target_site) \
                .order("clicks", desc=True) \
                .limit(200) \
                .execute()

            source_data = {r["keyword"].lower(): r for r in (source_kw.data or [])}
            target_data = {r["keyword"].lower(): r for r in (target_kw.data or [])}

            shared = set(source_data.keys()) & set(target_data.keys())

            for keyword in list(shared)[:20]:
                src = source_data[keyword]
                tgt = target_data[keyword]
                if src.get("url") and tgt.get("url"):
                    promo_id = self.codex.suggest_promotion(
                        source_site, src["url"],
                        target_site, tgt["url"],
                        keyword.title(),
                        keyword,
                    )
                    suggestions.append({
                        "id": promo_id,
                        "keyword": keyword,
                        "source_url": src["url"],
                        "target_url": tgt["url"],
                        "anchor_text": keyword.title(),
                    })

        except Exception as e:
            log.error(f"Suggest links error: {e}")

        return suggestions

    def inject_link(self, promo_id: int) -> Dict:
        """Inject a suggested cross-link via WordPress REST API."""
        # Get the promotion details
        suggestions = self.codex.get_suggestions()
        promo = next((s for s in suggestions if s["id"] == promo_id), None)

        if not promo:
            return {"status": "error", "reason": "Promotion not found"}

        # WordPress injection would go here using REST API
        # For now, mark as injected
        self.codex.mark_injected(promo_id)

        try:
            from core.event_bus import publish
            publish("pollination.link_injected", {
                "source": promo["source_site"],
                "target": promo["target_site"],
                "keyword": promo["keyword"],
            }, "cross_pollination")
        except Exception:
            pass

        return {"status": "injected", "promo_id": promo_id}

    def get_overlaps(self) -> List[Dict]:
        return self.codex.get_overlaps()

    def get_clusters(self) -> List[Dict]:
        return self.codex.get_clusters()

    def get_stats(self) -> Dict:
        return self.codex.stats()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-Pollination Engine")
    parser.add_argument("--detect", action="store_true", help="Detect keyword overlaps")
    parser.add_argument("--suggest", nargs=2, help="Suggest links: source target")
    parser.add_argument("--overlaps", action="store_true", help="Show overlaps")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    engine = CrossPollinationEngine()

    if args.detect:
        result = engine.detect_overlaps()
    elif args.suggest:
        result = engine.suggest_links(args.suggest[0], args.suggest[1])
    elif args.overlaps:
        result = engine.get_overlaps()
    else:
        result = engine.get_stats()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
