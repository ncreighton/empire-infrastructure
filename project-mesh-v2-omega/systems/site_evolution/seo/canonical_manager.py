"""
Canonical Manager — Detect duplicate/cannibalized content, audit canonicals,
detect redirect chains, generate canonical enforcement snippets.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from systems.site_evolution.utils import load_site_config, get_site_domain

log = logging.getLogger(__name__)


def _get_posts(site_slug: str, limit: int = 100) -> List[Dict]:
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        return _wp_request(
            site_slug, "GET",
            f"wp/v2/posts?per_page={limit}&status=publish"
            "&_fields=id,title,link,slug,categories"
        ) or []
    except Exception as e:
        log.warning("Could not fetch posts for %s: %s", site_slug, e)
        return []


def _extract_title(post: Dict) -> str:
    t = post.get("title", {})
    return t.get("rendered", "") if isinstance(t, dict) else str(t)


class CanonicalManager:
    """Manage canonical URLs and detect content duplication issues."""

    def detect_duplicates(self, site_slug: str, threshold: float = 0.7) -> List[Dict]:
        """Find posts with similar titles (>threshold similarity) that may cannibalize.

        Returns list of {post_a, post_b, similarity, recommendation}
        """
        posts = _get_posts(site_slug)
        duplicates = []

        for i, post_a in enumerate(posts):
            title_a = _extract_title(post_a).lower().strip()
            if not title_a:
                continue

            for post_b in posts[i + 1:]:
                title_b = _extract_title(post_b).lower().strip()
                if not title_b:
                    continue

                similarity = SequenceMatcher(None, title_a, title_b).ratio()
                if similarity >= threshold:
                    duplicates.append({
                        "post_a": {
                            "id": post_a.get("id"),
                            "title": _extract_title(post_a),
                            "url": post_a.get("link", ""),
                        },
                        "post_b": {
                            "id": post_b.get("id"),
                            "title": _extract_title(post_b),
                            "url": post_b.get("link", ""),
                        },
                        "similarity": round(similarity, 2),
                        "recommendation": "merge" if similarity > 0.85 else "differentiate",
                    })

        duplicates.sort(key=lambda d: d["similarity"], reverse=True)
        return duplicates

    def audit_canonicals(self, site_slug: str) -> Dict:
        """Verify <link rel="canonical"> on all pages.

        Returns: {total_checked, valid, missing, mismatched, score}
        """
        import requests
        domain = get_site_domain(site_slug)
        posts = _get_posts(site_slug, limit=30)

        valid = 0
        missing = 0
        mismatched = []
        errors = 0

        for post in posts:
            url = post.get("link", "")
            if not url:
                continue

            try:
                resp = requests.get(url, timeout=10,
                                    headers={"User-Agent": "EvoCanonicalAuditor/1.0"})
                html = resp.text[:5000]

                # Find canonical link
                match = re.search(r'<link\s+rel=["\']canonical["\']\s+href=["\']([^"\']+)["\']', html)
                if match:
                    canonical = match.group(1)
                    # Check if canonical matches the actual URL
                    if canonical.rstrip("/") == url.rstrip("/"):
                        valid += 1
                    else:
                        mismatched.append({
                            "url": url,
                            "canonical": canonical,
                            "title": _extract_title(post),
                        })
                else:
                    missing += 1

            except requests.RequestException:
                errors += 1

        total = valid + missing + len(mismatched) + errors
        score = (valid / max(total, 1)) * 100

        return {
            "site_slug": site_slug,
            "total_checked": total,
            "valid": valid,
            "missing": missing,
            "mismatched": mismatched,
            "errors": errors,
            "score": round(score),
        }

    def detect_redirect_chains(self, site_slug: str) -> List[Dict]:
        """Find 301→301→200 redirect chains that slow crawling."""
        import requests
        posts = _get_posts(site_slug, limit=50)
        chains = []

        for post in posts:
            url = post.get("link", "")
            if not url:
                continue

            try:
                resp = requests.head(url, timeout=8, allow_redirects=True,
                                     headers={"User-Agent": "EvoRedirectChecker/1.0"})
                if len(resp.history) >= 2:
                    chain = [{"url": r.url, "status": r.status_code} for r in resp.history]
                    chain.append({"url": resp.url, "status": resp.status_code})
                    chains.append({
                        "original_url": url,
                        "chain": chain,
                        "hops": len(resp.history),
                        "final_url": resp.url,
                    })
            except requests.RequestException:
                pass

        return chains

    def generate_canonical_snippet(self, site_slug: str) -> str:
        """PHP snippet to enforce proper canonical URLs."""
        return f"""<?php
/**
 * Canonical Enforcement — {site_slug}
 * Ensures every page has a proper canonical URL.
 * Works alongside RankMath/Yoast if present.
 */
function evo_enforce_canonical() {{
    if (is_admin()) return;

    // Only add if no canonical already exists (don't conflict with SEO plugins)
    if (has_action('wp_head', 'rel_canonical')) return;

    $canonical = '';
    if (is_singular()) {{
        $canonical = get_permalink();
    }} elseif (is_home() || is_front_page()) {{
        $canonical = home_url('/');
    }} elseif (is_category() || is_tag() || is_tax()) {{
        $canonical = get_term_link(get_queried_object());
    }}

    if ($canonical && !is_wp_error($canonical)) {{
        echo '<link rel="canonical" href="' . esc_url($canonical) . '" />' . "\\n";
    }}
}}
add_action('wp_head', 'evo_enforce_canonical', 1);
"""

    def generate_redirect_rules(self, redirects: List[Dict]) -> str:
        """PHP redirect snippet from list of {from_url, to_url} dicts."""
        if not redirects:
            return "<!-- No redirects needed -->"

        entries = "\n    ".join(
            f"'{r['from_url']}' => '{r['to_url']}',"
            for r in redirects[:50]
        )

        return f"""<?php
/**
 * Canonical Redirects — Clean up redirect chains.
 */
function evo_canonical_redirects() {{
    if (is_admin()) return;

    $redirects = array(
    {entries}
    );

    $request = rtrim($_SERVER['REQUEST_URI'], '/');
    if (isset($redirects[$request])) {{
        wp_redirect($redirects[$request], 301);
        exit;
    }}
}}
add_action('template_redirect', 'evo_canonical_redirects', 1);
"""
