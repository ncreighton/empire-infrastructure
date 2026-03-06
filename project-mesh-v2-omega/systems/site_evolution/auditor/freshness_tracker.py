"""
Freshness Tracker — Audit content staleness, prioritize updates,
generate "Last Updated" snippets, seasonal content calendar.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from systems.site_evolution.utils import load_site_config, SITE_CATEGORIES

log = logging.getLogger(__name__)


def _get_posts(site_slug: str, limit: int = 100) -> List[Dict]:
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        return _wp_request(
            site_slug, "GET",
            f"wp/v2/posts?per_page={limit}&status=publish"
            "&_fields=id,title,link,date,modified,categories"
        ) or []
    except Exception as e:
        log.warning("Could not fetch posts for %s: %s", site_slug, e)
        return []


def _extract_title(post: Dict) -> str:
    t = post.get("title", {})
    return t.get("rendered", "") if isinstance(t, dict) else str(t)


def _parse_date(date_str: str) -> datetime:
    """Parse WordPress date string to datetime."""
    try:
        # WordPress returns ISO 8601 format
        clean = date_str.replace("Z", "+00:00") if "Z" in date_str else date_str
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


class FreshnessTracker:
    """Track content freshness and prioritize updates."""

    def audit_freshness(self, site_slug: str) -> Dict:
        """Flag stale (>6mo) and critical (>12mo) posts.

        Returns: {total_posts, fresh, stale, critical, score, issues}
        """
        posts = _get_posts(site_slug)
        now = datetime.now(timezone.utc)
        score = 50
        issues = []

        fresh = []      # Updated within 6 months
        stale = []      # 6-12 months since update
        critical = []   # >12 months since update

        for post in posts:
            modified = post.get("modified", post.get("date", ""))
            if not modified:
                continue

            mod_date = _parse_date(modified)
            days_since = (now - mod_date).days
            title = _extract_title(post)

            entry = {
                "id": post.get("id"),
                "title": title,
                "url": post.get("link", ""),
                "modified": modified,
                "days_since_update": days_since,
            }

            if days_since <= 180:
                fresh.append(entry)
            elif days_since <= 365:
                stale.append(entry)
            else:
                critical.append(entry)

        total = len(posts)
        if total == 0:
            return {"score": 0, "issues": [{"type": "critical", "msg": "No posts found"}]}

        fresh_ratio = len(fresh) / total

        # Scoring
        if fresh_ratio >= 0.8:
            score += 30
        elif fresh_ratio >= 0.5:
            score += 15
        else:
            issues.append({"type": "warning", "msg": f"Only {fresh_ratio:.0%} of content updated in last 6 months"})

        if not critical:
            score += 20
        elif len(critical) <= 3:
            score += 10
            issues.append({"type": "info", "msg": f"{len(critical)} posts older than 12 months"})
        else:
            issues.append({"type": "warning", "msg": f"{len(critical)} posts critically outdated (>12 months)"})

        # Sort stale/critical by days (most urgent first)
        stale.sort(key=lambda x: x["days_since_update"], reverse=True)
        critical.sort(key=lambda x: x["days_since_update"], reverse=True)

        return {
            "site_slug": site_slug,
            "total_posts": total,
            "fresh": len(fresh),
            "stale": len(stale),
            "critical": len(critical),
            "fresh_pct": round(fresh_ratio * 100, 1),
            "stale_posts": stale[:10],
            "critical_posts": critical[:10],
            "score": min(100, score),
            "issues": issues,
        }

    def get_stale_content(self, site_slug: str, days: int = 180) -> List[Dict]:
        """Get content not updated in `days` days, sorted by staleness."""
        posts = _get_posts(site_slug)
        now = datetime.now(timezone.utc)
        stale = []

        for post in posts:
            modified = post.get("modified", post.get("date", ""))
            if not modified:
                continue

            mod_date = _parse_date(modified)
            days_since = (now - mod_date).days

            if days_since >= days:
                stale.append({
                    "id": post.get("id"),
                    "title": _extract_title(post),
                    "url": post.get("link", ""),
                    "modified": modified,
                    "days_since_update": days_since,
                })

        stale.sort(key=lambda x: x["days_since_update"], reverse=True)
        return stale

    def get_update_priority(self, site_slug: str) -> List[Dict]:
        """Cross-ref stale content with GSC clicks.

        Stale + high traffic = URGENT update needed.
        """
        stale = self.get_stale_content(site_slug, days=180)

        # Get GSC page data
        page_clicks = {}
        try:
            from systems.site_evolution.seo.search_analytics import SearchAnalytics
            sa = SearchAnalytics()
            perf = sa.gsc_get_performance(site_slug, days=28, dimensions=["page"])
            for row in perf.get("rows", []):
                url = row.get("keys", [""])[0] if isinstance(row.get("keys"), list) else ""
                page_clicks[url] = row.get("clicks", 0)
        except Exception:
            pass

        # Annotate stale content with traffic data
        for item in stale:
            url = item.get("url", "")
            clicks = page_clicks.get(url, 0)
            item["monthly_clicks"] = clicks
            item["priority"] = "URGENT" if clicks > 20 and item["days_since_update"] > 365 else \
                               "HIGH" if clicks > 10 else \
                               "MEDIUM" if item["days_since_update"] > 365 else "LOW"

        # Sort by priority then clicks
        priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        stale.sort(key=lambda x: (priority_order.get(x.get("priority", "LOW"), 4),
                                   -x.get("monthly_clicks", 0)))
        return stale

    def generate_update_date_snippet(self, site_slug: str) -> str:
        """PHP snippet to display "Last Updated" date on posts."""
        return f"""<?php
/**
 * Last Updated Date — {site_slug}
 * Shows "Last Updated" date above content on single posts.
 */
function evo_show_last_updated($content) {{
    if (!is_single()) return $content;

    $modified = get_the_modified_date('F j, Y');
    $published = get_the_date('F j, Y');

    // Only show if modified date differs from published date
    if ($modified === $published) return $content;

    $badge = '<div class="evo-last-updated" style="display:inline-flex;align-items:center;gap:6px;'
           . 'padding:6px 14px;background:var(--color-bg-alt,#f8fafc);border-radius:6px;'
           . 'font-size:13px;color:var(--color-text-muted,#64748b);margin-bottom:20px;">'
           . '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
           . '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
           . '<span>Last updated: <strong>' . esc_html($modified) . '</strong></span>'
           . '</div>';

    return $badge . $content;
}}
add_filter('the_content', 'evo_show_last_updated', 3);
"""

    def get_seasonal_calendar(self, site_slug: str) -> Dict:
        """Category-aware seasonal content calendar."""
        # Delegate to ContentGapAnalyzer's calendar
        try:
            from systems.site_evolution.seo.content_gap import ContentGapAnalyzer
            return ContentGapAnalyzer().get_content_calendar(site_slug)
        except ImportError:
            return {"site_slug": site_slug, "note": "Content gap module not available"}
