"""
Content Gap Analyzer — Map GSC queries to posts, find uncovered keyword clusters,
suggest new articles, identify thin content expansion opportunities.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

from systems.site_evolution.utils import load_site_config, get_site_domain, SITE_CATEGORIES

log = logging.getLogger(__name__)


def _get_posts(site_slug: str, limit: int = 100) -> List[Dict]:
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        return _wp_request(
            site_slug, "GET",
            f"wp/v2/posts?per_page={limit}&status=publish"
            "&_fields=id,title,content,link,date,modified"
        ) or []
    except Exception as e:
        log.warning("Could not fetch posts for %s: %s", site_slug, e)
        return []


def _extract_title(post: Dict) -> str:
    t = post.get("title", {})
    return t.get("rendered", "") if isinstance(t, dict) else str(t)


def _word_count(post: Dict) -> int:
    content = post.get("content", {})
    if isinstance(content, dict):
        content = content.get("rendered", "")
    clean = re.sub(r'<[^>]+>', '', content)
    return len(clean.split())


# Seasonal calendars per site category
SEASONAL_CALENDAR = {
    "spiritual": {
        1: ["New Year rituals", "intention setting", "winter solstice follow-up"],
        2: ["Imbolc", "self-love spells", "candle magic"],
        3: ["spring equinox", "Ostara", "new beginnings"],
        4: ["Beltane prep", "garden magic", "earth day"],
        5: ["Beltane", "flower magic", "abundance spells"],
        6: ["Litha", "summer solstice", "sun magic"],
        7: ["mid-summer rituals", "water magic", "crystal charging"],
        8: ["Lughnasadh", "Lammas", "harvest gratitude"],
        9: ["Mabon", "autumn equinox", "balance rituals"],
        10: ["Samhain", "ancestor work", "divination", "Halloween"],
        11: ["shadow work", "gratitude magic", "protection"],
        12: ["Yule", "winter solstice", "year review rituals"],
    },
    "ai_tech": {
        1: ["AI predictions", "tech trends", "CES coverage"],
        2: ["AI tools roundup", "productivity AI"],
        3: ["spring cleaning digital", "AI workflow optimization"],
        4: ["AI conference season", "new model releases"],
        5: ["summer project ideas", "AI automation"],
        6: ["mid-year AI review", "emerging tech"],
        7: ["AI for business", "cost optimization"],
        8: ["back to school AI tools", "student AI guides"],
        9: ["fall tech launches", "AI comparison guides"],
        10: ["AI safety", "ethics in AI"],
        11: ["Black Friday AI deals", "best AI tools"],
        12: ["year in review AI", "2027 predictions"],
    },
    "review": {
        1: ["New Year deals", "CES new products"],
        2: ["Valentine's gifts", "winter essentials"],
        3: ["spring cleaning gear", "outdoor prep"],
        4: ["spring deals", "new releases"],
        5: ["summer gear prep", "outdoor tech"],
        6: ["mid-year best-of lists", "summer essentials"],
        7: ["Amazon Prime Day deals", "summer sales"],
        8: ["back to school gear", "fall prep"],
        9: ["fall launches", "new model releases"],
        10: ["early holiday deals", "gift guide prep"],
        11: ["Black Friday", "Cyber Monday", "gift guides"],
        12: ["holiday gift guides", "year-end best-of"],
    },
    "lifestyle": {
        1: ["New Year planning", "habit tracking", "goal setting"],
        2: ["self-care", "winter wellness"],
        3: ["spring refresh", "decluttering"],
        4: ["Easter", "spring activities"],
        5: ["Mother's Day", "outdoor activities"],
        6: ["summer planning", "Father's Day"],
        7: ["summer activities", "travel tips"],
        8: ["back to school", "fall prep"],
        9: ["fall activities", "routine reset"],
        10: ["Halloween", "autumn wellness"],
        11: ["Thanksgiving", "gratitude practices"],
        12: ["holiday planning", "year reflection"],
    },
}


class ContentGapAnalyzer:
    """Analyze keyword coverage gaps and suggest content opportunities."""

    def analyze_keyword_coverage(self, site_slug: str) -> Dict:
        """Map GSC queries to existing posts, find uncovered clusters.

        Returns: {covered_queries, uncovered_queries, coverage_pct, suggestions}
        """
        posts = _get_posts(site_slug)
        post_titles = {_extract_title(p).lower() for p in posts}

        # Get GSC queries
        queries = []
        try:
            from systems.site_evolution.seo.search_analytics import SearchAnalytics
            sa = SearchAnalytics()
            top = sa.gsc_get_top_queries(site_slug, days=90, limit=100)
            queries = top if isinstance(top, list) else []
        except Exception as e:
            log.debug("GSC queries not available for %s: %s", site_slug, e)

        if not queries:
            return {
                "site_slug": site_slug,
                "covered_queries": [],
                "uncovered_queries": [],
                "coverage_pct": 0,
                "note": "No GSC data available",
            }

        covered = []
        uncovered = []

        for q in queries:
            query_text = q.get("query", q.get("keys", [""])[0] if isinstance(q.get("keys"), list) else "").lower()
            if not query_text:
                continue

            # Check if any post title covers this query
            is_covered = any(
                query_text in title or title in query_text
                for title in post_titles
            )

            entry = {
                "query": query_text,
                "clicks": q.get("clicks", 0),
                "impressions": q.get("impressions", 0),
                "position": q.get("position", 0),
            }

            if is_covered:
                covered.append(entry)
            else:
                uncovered.append(entry)

        # Sort uncovered by impressions (highest opportunity)
        uncovered.sort(key=lambda x: x.get("impressions", 0), reverse=True)

        coverage_pct = len(covered) / max(len(covered) + len(uncovered), 1) * 100

        return {
            "site_slug": site_slug,
            "total_queries": len(covered) + len(uncovered),
            "covered_queries": covered[:20],
            "uncovered_queries": uncovered[:20],
            "coverage_pct": round(coverage_pct, 1),
        }

    def suggest_new_articles(self, site_slug: str, max_suggestions: int = 10) -> List[Dict]:
        """Suggest new article topics from rising/uncovered keywords."""
        coverage = self.analyze_keyword_coverage(site_slug)
        uncovered = coverage.get("uncovered_queries", [])

        suggestions = []
        for q in uncovered[:max_suggestions]:
            query = q["query"]
            # Generate article title from query
            title = query.title()
            if not any(title.startswith(w) for w in ("How", "What", "Why", "Best", "Top")):
                title = f"Complete Guide to {title}"

            suggestions.append({
                "title": title,
                "keyword": q["query"],
                "impressions": q.get("impressions", 0),
                "current_position": q.get("position", 0),
                "opportunity": "high" if q.get("impressions", 0) > 100 else "medium",
            })

        return suggestions

    def analyze_thin_content(self, site_slug: str) -> List[Dict]:
        """Find posts < 800 words with good impressions = expansion opportunities."""
        posts = _get_posts(site_slug)

        # Get GSC data for pages
        page_performance = {}
        try:
            from systems.site_evolution.seo.search_analytics import SearchAnalytics
            sa = SearchAnalytics()
            perf = sa.gsc_get_performance(site_slug, days=28, dimensions=["page"])
            for row in perf.get("rows", []):
                url = row.get("keys", [""])[0] if isinstance(row.get("keys"), list) else ""
                page_performance[url] = {
                    "clicks": row.get("clicks", 0),
                    "impressions": row.get("impressions", 0),
                }
        except Exception:
            pass

        thin = []
        for post in posts:
            wc = _word_count(post)
            if wc >= 800:
                continue

            url = post.get("link", "")
            perf = page_performance.get(url, {})
            impressions = perf.get("impressions", 0)

            thin.append({
                "id": post.get("id"),
                "title": _extract_title(post),
                "url": url,
                "word_count": wc,
                "impressions": impressions,
                "priority": "high" if impressions > 50 and wc < 500 else "medium" if impressions > 10 else "low",
            })

        # Sort by impressions descending (highest-traffic thin content = most urgent)
        thin.sort(key=lambda x: x.get("impressions", 0), reverse=True)
        return thin

    def get_content_calendar(self, site_slug: str) -> Dict:
        """Seasonal content calendar based on site category."""
        from datetime import date
        current_month = date.today().month

        # Determine site category
        site_category = "lifestyle"  # default
        for cat, sites in SITE_CATEGORIES.items():
            if site_slug in sites:
                site_category = cat
                break

        calendar = SEASONAL_CALENDAR.get(site_category, SEASONAL_CALENDAR["lifestyle"])

        return {
            "site_slug": site_slug,
            "category": site_category,
            "current_month": current_month,
            "this_month_topics": calendar.get(current_month, []),
            "next_month_topics": calendar.get(current_month % 12 + 1, []),
            "full_calendar": {str(m): topics for m, topics in calendar.items()},
        }
