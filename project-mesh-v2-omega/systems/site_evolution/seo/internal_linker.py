"""
Internal Linker — Build keyword→URL maps, find orphan pages, suggest contextual links.
Generates PHP auto-linker snippets for Code Snippets deployment.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional
from difflib import SequenceMatcher

from systems.site_evolution.utils import load_site_config, get_site_domain

log = logging.getLogger(__name__)


def _get_all_posts(site_slug: str, limit: int = 100) -> List[Dict]:
    """Fetch all published posts for link mapping."""
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        return _wp_request(
            site_slug, "GET",
            f"wp/v2/posts?per_page={limit}&status=publish"
            "&_fields=id,title,content,excerpt,link,categories,tags"
        ) or []
    except Exception as e:
        log.warning("Could not fetch posts for %s: %s", site_slug, e)
        return []


def _get_all_pages(site_slug: str) -> List[Dict]:
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        return _wp_request(
            site_slug, "GET",
            "wp/v2/pages?per_page=50&status=publish&_fields=id,title,link,content"
        ) or []
    except Exception as e:
        log.debug("Could not fetch pages for %s: %s", site_slug, e)
        return []


def _extract_title(post: Dict) -> str:
    t = post.get("title", {})
    return t.get("rendered", "") if isinstance(t, dict) else str(t)


def _extract_links(html: str) -> List[str]:
    """Extract all href URLs from HTML content."""
    return re.findall(r'<a\s[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE)


def _clean_text(html: str) -> str:
    return re.sub(r'<[^>]+>', '', html).strip()


class InternalLinker:
    """Build internal link maps and generate auto-linking snippets."""

    def build_link_map(self, site_slug: str) -> Dict[str, str]:
        """Build keyword→URL map from all post titles and content.

        Returns: {keyword_phrase: url}
        """
        posts = _get_all_posts(site_slug)
        link_map = {}

        for post in posts:
            title = _extract_title(post)
            url = post.get("link", "")
            if not title or not url:
                continue

            # Map full title
            clean_title = _clean_text(title).lower().strip()
            if len(clean_title) > 3:
                link_map[clean_title] = url

            # Map significant title fragments (3+ words)
            words = clean_title.split()
            if len(words) >= 4:
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i:i+3])
                    if len(phrase) > 10 and phrase not in link_map:
                        link_map[phrase] = url

        log.info("Built link map for %s: %d entries", site_slug, len(link_map))
        return link_map

    def find_orphan_pages(self, site_slug: str) -> List[Dict]:
        """Find pages/posts with zero inbound internal links."""
        domain = get_site_domain(site_slug)
        posts = _get_all_posts(site_slug)
        pages = _get_all_pages(site_slug)
        all_content = posts + pages

        # Build set of all URLs
        all_urls = {}
        for item in all_content:
            url = item.get("link", "")
            if url:
                all_urls[url] = _extract_title(item)

        # Count inbound links for each URL
        inbound_counts = defaultdict(int)
        for item in all_content:
            content = item.get("content", {})
            if isinstance(content, dict):
                content = content.get("rendered", "")
            links = _extract_links(content)
            for link in links:
                if domain and domain in link:
                    # Normalize URL
                    normalized = link.rstrip("/")
                    for target_url in all_urls:
                        if normalized == target_url.rstrip("/"):
                            inbound_counts[target_url] += 1

        orphans = []
        for url, title in all_urls.items():
            if inbound_counts.get(url, 0) == 0:
                # Skip homepage
                path = url.replace(f"https://{domain}", "").strip("/")
                if path and path not in ("", "home"):
                    orphans.append({
                        "url": url,
                        "title": title,
                        "inbound_links": 0,
                    })

        log.info("Found %d orphan pages for %s", len(orphans), site_slug)
        return orphans

    def suggest_links(self, site_slug: str, post_id: int,
                      max_links: int = 5) -> List[Dict]:
        """Suggest internal links for a specific post based on content keywords."""
        from systems.site_evolution.deployer.wp_deployer import _wp_request

        try:
            post = _wp_request(
                site_slug, "GET",
                f"wp/v2/posts/{post_id}?_fields=id,title,content,link"
            )
        except Exception:
            return []

        if not post:
            return []

        content = post.get("content", {})
        if isinstance(content, dict):
            content = content.get("rendered", "")
        clean_content = _clean_text(content).lower()
        post_url = post.get("link", "")

        # Get all other posts
        all_posts = _get_all_posts(site_slug)
        suggestions = []

        for other in all_posts:
            other_url = other.get("link", "")
            if other_url == post_url or not other_url:
                continue

            other_title = _extract_title(other).lower().strip()
            if not other_title:
                continue

            # Check if title keywords appear in content
            title_words = set(other_title.split()) - {"the", "a", "an", "is", "of", "to", "and", "in", "for", "on", "with", "how", "what", "why"}
            if len(title_words) < 2:
                continue

            match_count = sum(1 for w in title_words if w in clean_content)
            relevance = match_count / max(len(title_words), 1)

            if relevance >= 0.5:
                # Check it's not already linked
                existing_links = _extract_links(content if isinstance(content, str) else "")
                already_linked = any(other_url.rstrip("/") in link for link in existing_links)

                if not already_linked:
                    suggestions.append({
                        "target_url": other_url,
                        "target_title": _extract_title(other),
                        "relevance": round(relevance, 2),
                        "anchor_text": _extract_title(other),
                    })

        # Sort by relevance, limit
        suggestions.sort(key=lambda s: s["relevance"], reverse=True)
        return suggestions[:max_links]

    def generate_link_injection_snippet(self, site_slug: str) -> str:
        """Generate PHP auto-linker that injects internal links into content."""
        link_map = self.build_link_map(site_slug)
        if not link_map:
            return "<!-- No link map available -->"

        # Take top 30 most important links
        entries = list(link_map.items())[:30]
        php_array = ",\n    ".join(
            f"'{self._escape_php(kw)}' => '{self._escape_php(url)}'"
            for kw, url in entries
        )

        return f"""<?php
/**
 * Auto Internal Linker — {site_slug}
 * Automatically links keyword phrases to relevant internal pages.
 * Max 3 auto-links per post to avoid over-optimization.
 */
function evo_auto_internal_links($content) {{
    if (!is_single()) return $content;

    $link_map = array(
    {php_array}
    );

    $link_count = 0;
    $max_links = 3;

    foreach ($link_map as $keyword => $url) {{
        if ($link_count >= $max_links) break;
        // Only link first occurrence, case-insensitive
        $pattern = '/\\b(' . preg_quote($keyword, '/') . ')\\b(?![^<]*<\\/a>)/i';
        $replacement = '<a href="' . esc_url($url) . '" class="evo-auto-link">$1</a>';
        $new_content = preg_replace($pattern, $replacement, $content, 1, $count);
        if ($count > 0) {{
            $content = $new_content;
            $link_count++;
        }}
    }}

    return $content;
}}
add_filter('the_content', 'evo_auto_internal_links', 20);
"""

    def get_link_equity_report(self, site_slug: str) -> Dict:
        """Hub/spoke analysis — find hub pages and leaf pages."""
        domain = get_site_domain(site_slug)
        posts = _get_all_posts(site_slug)
        pages = _get_all_pages(site_slug)
        all_content = posts + pages

        outbound = defaultdict(int)
        inbound = defaultdict(int)

        for item in all_content:
            url = item.get("link", "")
            content = item.get("content", {})
            if isinstance(content, dict):
                content = content.get("rendered", "")
            links = _extract_links(content)
            internal_links = [l for l in links if domain and domain in l]
            outbound[url] = len(internal_links)
            for link in internal_links:
                inbound[link.rstrip("/")] += 1

        hubs = sorted(
            [(url, outbound[url]) for url in outbound if outbound[url] >= 3],
            key=lambda x: x[1], reverse=True
        )[:10]

        leaves = [
            {"url": url, "inbound": inbound.get(url.rstrip("/"), 0), "outbound": outbound.get(url, 0)}
            for url in outbound
            if outbound.get(url, 0) <= 1 and inbound.get(url.rstrip("/"), 0) <= 1
        ]

        return {
            "site_slug": site_slug,
            "total_pages": len(all_content),
            "hub_pages": [{"url": u, "outbound_links": c} for u, c in hubs],
            "leaf_pages": leaves[:20],
            "avg_internal_links": sum(outbound.values()) / max(len(outbound), 1),
        }

    def audit_internal_links(self, site_slug: str) -> Dict:
        """Score internal linking 0-100 for auditor integration."""
        score = 0
        issues = []

        posts = _get_all_posts(site_slug)
        domain = get_site_domain(site_slug)

        if not posts:
            return {"score": 0, "issues": [{"type": "critical", "msg": "No posts to audit"}]}

        total_internal = 0
        posts_with_links = 0

        for post in posts:
            content = post.get("content", {})
            if isinstance(content, dict):
                content = content.get("rendered", "")
            links = _extract_links(content)
            internal = [l for l in links if domain and domain in l]
            total_internal += len(internal)
            if internal:
                posts_with_links += 1

        avg_links = total_internal / max(len(posts), 1)
        link_coverage = posts_with_links / max(len(posts), 1)

        # Scoring
        if avg_links >= 3:
            score += 35
        elif avg_links >= 1.5:
            score += 25
        elif avg_links >= 0.5:
            score += 10
        else:
            issues.append({"type": "critical", "msg": f"Very low internal link density: {avg_links:.1f} avg per post"})

        if link_coverage >= 0.8:
            score += 25
        elif link_coverage >= 0.5:
            score += 15
        else:
            issues.append({"type": "warning", "msg": f"Only {link_coverage:.0%} of posts have internal links"})

        # Orphan pages check
        orphans = self.find_orphan_pages(site_slug)
        orphan_ratio = len(orphans) / max(len(posts), 1)
        if orphan_ratio <= 0.1:
            score += 25
        elif orphan_ratio <= 0.3:
            score += 15
        else:
            issues.append({"type": "warning", "msg": f"{len(orphans)} orphan pages with no inbound links"})

        # Hub pages bonus
        report = self.get_link_equity_report(site_slug)
        if report.get("hub_pages"):
            score += 15
        else:
            issues.append({"type": "info", "msg": "No hub pages with 3+ outbound internal links"})

        return {"score": min(100, score), "issues": issues}

    @staticmethod
    def _escape_php(s: str) -> str:
        return s.replace("'", "\\'").replace("\\", "\\\\")
