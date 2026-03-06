"""Intelligence Amplifier — Grades articles, detects patterns, builds playbooks."""

import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SITES_CONFIG = Path(__file__).parent.parent.parent.parent / "config" / "sites.json"

# Niche mapping from site slug
NICHE_MAP = {
    "witchcraftforbeginners": "witchcraft",
    "smarthomewizards": "smart_home",
    "smarthomegearreviews": "smart_home",
    "mythicalarchives": "mythology",
    "bulletjournals": "journaling",
    "wealthfromai": "ai_finance",
    "aidiscoverydigest": "ai_news",
    "aiinactionhub": "ai_tools",
    "clearainews": "ai_news",
    "pulsegearreviews": "fitness_tech",
    "wearablegearreviews": "wearable_tech",
    "theconnectedhaven": "smart_home",
    "manifestandalign": "manifestation",
    "familyflourish": "family",
    "celebrationseason": "celebrations",
}


def _classify_headline(title: str) -> str:
    """Classify a headline into a formula type."""
    title_lower = title.lower()
    if re.match(r'^\d+\s', title_lower) or "top" in title_lower:
        return "listicle"
    if title_lower.startswith("how to") or title_lower.startswith("how do"):
        return "how_to"
    if "vs" in title_lower or "versus" in title_lower:
        return "comparison"
    if "?" in title:
        return "question"
    if "best" in title_lower:
        return "best_of"
    if "guide" in title_lower or "complete" in title_lower:
        return "guide"
    if "review" in title_lower:
        return "review"
    return "statement"


def _grade_article(clicks: int, impressions: int, ctr: float = 0) -> str:
    """Grade article performance A-F."""
    if clicks >= 100:
        return "A"
    if clicks >= 50:
        return "B"
    if clicks >= 20:
        return "C"
    if clicks >= 5:
        return "D"
    return "F"


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


class IntelligenceAmplifier:
    """Analyzes content performance across the empire and builds playbooks."""

    def __init__(self):
        from .codex import IntelligenceCodex
        self.codex = IntelligenceCodex()
        self.supabase = _get_supabase()

    def analyze_site(self, site_slug: str) -> Dict:
        """Analyze all articles for a site, grade them, detect patterns."""
        niche = NICHE_MAP.get(site_slug, site_slug)
        articles = self._fetch_articles(site_slug)

        graded = []
        headline_types = Counter()
        grade_dist = Counter()

        for article in articles:
            title = article.get("title", "")
            clicks = article.get("clicks", 0)
            impressions = article.get("impressions", 0)
            ctr = article.get("ctr", 0)

            headline_type = _classify_headline(title)
            grade = _grade_article(clicks, impressions, ctr)

            headline_types[headline_type] += 1
            grade_dist[grade] += 1

            data = {
                "clicks": clicks,
                "impressions": impressions,
                "sessions": article.get("sessions", 0),
                "position": article.get("position"),
                "ctr": ctr,
                "headline_type": headline_type,
                "word_count": article.get("word_count"),
                "grade": grade,
                "url": article.get("url"),
                "post_id": article.get("post_id"),
            }

            self.codex.upsert_article(site_slug, title, data)
            graded.append({"title": title, "grade": grade, "clicks": clicks})

        # Detect winning patterns
        self._detect_patterns(site_slug, niche)

        # Build/update playbook
        self._build_playbook(site_slug, niche)

        result = {
            "site": site_slug,
            "niche": niche,
            "articles_analyzed": len(graded),
            "grade_distribution": dict(grade_dist),
            "headline_types": dict(headline_types),
            "top_articles": sorted(graded, key=lambda x: x["clicks"], reverse=True)[:10],
        }

        try:
            from core.event_bus import publish
            publish("intelligence.analysis_complete", {
                "site": site_slug,
                "articles": len(graded),
                "a_count": grade_dist.get("A", 0),
            }, "intelligence_amplifier")
        except Exception:
            pass

        return result

    def _fetch_articles(self, site_slug: str) -> List[Dict]:
        """Fetch article data from Supabase."""
        if not self.supabase:
            return []

        articles = []
        try:
            # Get top pages with click data
            resp = self.supabase.table("top_pages") \
                .select("page_url,clicks,impressions,ctr,position") \
                .eq("site_slug", site_slug) \
                .order("clicks", desc=True) \
                .limit(200) \
                .execute()

            for row in (resp.data or []):
                articles.append({
                    "url": row.get("page_url"),
                    "title": row.get("page_url", "").split("/")[-2].replace("-", " ").title()
                    if row.get("page_url") else "",
                    "clicks": row.get("clicks", 0),
                    "impressions": row.get("impressions", 0),
                    "ctr": row.get("ctr", 0),
                    "position": row.get("position"),
                })
        except Exception as e:
            log.error(f"Failed to fetch articles for {site_slug}: {e}")

        return articles

    def _detect_patterns(self, site_slug: str, niche: str):
        """Detect winning content patterns from graded articles."""
        articles = self.codex.get_articles(site_slug)
        if len(articles) < 5:
            return

        # Pattern: best headline types by grade
        type_performance = {}
        for a in articles:
            ht = a.get("headline_type", "unknown")
            if ht not in type_performance:
                type_performance[ht] = {"a_b": 0, "total": 0}
            type_performance[ht]["total"] += 1
            if a.get("grade") in ("A", "B"):
                type_performance[ht]["a_b"] += 1

        for ht, perf in type_performance.items():
            if perf["total"] >= 3:
                win_rate = perf["a_b"] / perf["total"]
                self.codex.save_pattern(
                    niche, "headline_type", ht, win_rate,
                    perf["total"], site_slug
                )

    def _build_playbook(self, site_slug: str, niche: str):
        """Build/update a niche playbook from performance data."""
        articles = self.codex.get_articles(site_slug)
        if len(articles) < 5:
            return

        a_articles = [a for a in articles if a.get("grade") in ("A", "B")]

        # Optimal word count from A/B articles
        word_counts = [a.get("word_count") for a in a_articles if a.get("word_count")]
        wc_min = min(word_counts) if word_counts else 1000
        wc_max = max(word_counts) if word_counts else 3000

        # Best headline formulas
        ht_counts = Counter(a.get("headline_type") for a in a_articles)
        best_formulas = [ht for ht, _ in ht_counts.most_common(3)]

        # Grade distribution
        grade_counts = Counter(a.get("grade") for a in articles)
        total = len(articles)
        if grade_counts.get("A", 0) / max(total, 1) > 0.3:
            avg_grade = "A"
        elif (grade_counts.get("A", 0) + grade_counts.get("B", 0)) / max(total, 1) > 0.4:
            avg_grade = "B"
        else:
            avg_grade = "C"

        playbook = {
            "word_count_min": wc_min,
            "word_count_max": wc_max,
            "headline_formulas": best_formulas,
            "publish_days": [],  # Would need date analysis
            "ctas": [],
            "avg_grade": avg_grade,
            "top_topics": [a.get("title", "") for a in a_articles[:5]],
        }

        self.codex.save_playbook(niche, playbook)

    def get_playbook(self, niche: str) -> Optional[Dict]:
        return self.codex.get_playbook(niche)

    def get_decaying(self, site_slug: str = None) -> List[Dict]:
        return self.codex.get_decaying(site_slug)

    def get_stats(self) -> Dict:
        return self.codex.stats()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Intelligence Amplifier")
    parser.add_argument("--analyze", help="Analyze a site")
    parser.add_argument("--playbook", help="Get playbook for a niche")
    parser.add_argument("--decaying", help="Show decaying articles (site or 'all')")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    amp = IntelligenceAmplifier()

    if args.analyze:
        result = amp.analyze_site(args.analyze)
    elif args.playbook:
        result = amp.get_playbook(args.playbook) or {"error": "No playbook found"}
    elif args.decaying:
        site = None if args.decaying == "all" else args.decaying
        result = amp.get_decaying(site)
    else:
        result = amp.get_stats()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
