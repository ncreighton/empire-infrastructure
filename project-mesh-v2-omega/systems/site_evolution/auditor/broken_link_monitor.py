"""
Broken Link Monitor — Crawl internal/external links, detect 404s, redirect chains.
Generates PHP redirect snippets for Code Snippets deployment.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from systems.site_evolution.utils import load_site_config, get_site_domain

log = logging.getLogger(__name__)


def _get_posts(site_slug: str, limit: int = 50) -> List[Dict]:
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        return _wp_request(
            site_slug, "GET",
            f"wp/v2/posts?per_page={limit}&status=publish"
            "&_fields=id,title,content,link"
        ) or []
    except Exception as e:
        log.warning("Could not fetch posts for %s: %s", site_slug, e)
        return []


def _extract_links(html: str) -> List[Tuple[str, str]]:
    """Extract (url, anchor_text) pairs from HTML."""
    pattern = re.compile(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    results = []
    for match in pattern.finditer(html):
        url = match.group(1).strip()
        anchor = re.sub(r'<[^>]+>', '', match.group(2)).strip()
        if url and not url.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
            results.append((url, anchor))
    return results


def _check_url(url: str, timeout: int = 8) -> Dict:
    """HEAD-check a single URL. Returns status info."""
    try:
        import requests
        resp = requests.head(url, timeout=timeout, allow_redirects=True,
                             headers={"User-Agent": "EvoLinkChecker/1.0"})

        # Track redirect chain
        redirects = []
        if resp.history:
            for r in resp.history:
                redirects.append({
                    "url": r.url,
                    "status": r.status_code,
                })

        return {
            "url": url,
            "status_code": resp.status_code,
            "final_url": resp.url,
            "redirects": redirects,
            "redirect_count": len(redirects),
            "is_broken": resp.status_code >= 400,
            "is_redirect": len(redirects) > 0,
        }
    except Exception as e:
        return {
            "url": url,
            "status_code": 0,
            "final_url": url,
            "redirects": [],
            "redirect_count": 0,
            "is_broken": True,
            "error": str(e),
        }


class BrokenLinkMonitor:
    """Crawl and check all links across a WordPress site."""

    def crawl_links(self, site_slug: str, limit: int = 50) -> Dict:
        """Extract and check all links from published posts.

        Returns: {total_links, internal, external, broken, redirects, results}
        """
        domain = get_site_domain(site_slug)
        posts = _get_posts(site_slug, limit)

        all_links = set()
        link_sources = {}  # url -> list of source post IDs

        for post in posts:
            content = post.get("content", {})
            if isinstance(content, dict):
                content = content.get("rendered", "")
            links = _extract_links(content)
            for url, anchor in links:
                # Normalize relative URLs
                if url.startswith("/"):
                    url = f"https://{domain}{url}"
                all_links.add(url)
                link_sources.setdefault(url, []).append(post.get("id"))

        log.info("Crawling %d unique links for %s", len(all_links), site_slug)

        # Check URLs in parallel (max 8 workers to be respectful)
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_check_url, url): url for url in all_links}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    result["source_posts"] = link_sources.get(result["url"], [])
                    results.append(result)
                except Exception as e:
                    log.debug("Link check error: %s", e)

        internal = [r for r in results if domain and domain in r["url"]]
        external = [r for r in results if domain and domain not in r["url"]]
        broken = [r for r in results if r["is_broken"]]
        redirects = [r for r in results if r["redirect_count"] > 0]

        return {
            "site_slug": site_slug,
            "total_links": len(results),
            "internal_count": len(internal),
            "external_count": len(external),
            "broken_count": len(broken),
            "redirect_count": len(redirects),
            "broken": broken,
            "redirects": redirects,
        }

    def detect_broken_internal(self, site_slug: str) -> List[Dict]:
        """Find internal 404s."""
        domain = get_site_domain(site_slug)
        crawl = self.crawl_links(site_slug)
        return [
            r for r in crawl.get("broken", [])
            if domain and domain in r["url"]
        ]

    def detect_broken_external(self, site_slug: str) -> List[Dict]:
        """Find external 404s/timeouts."""
        domain = get_site_domain(site_slug)
        crawl = self.crawl_links(site_slug)
        return [
            r for r in crawl.get("broken", [])
            if domain and domain not in r["url"]
        ]

    def detect_redirect_chains(self, site_slug: str) -> List[Dict]:
        """Find redirect chains (2+ redirects)."""
        crawl = self.crawl_links(site_slug)
        return [
            r for r in crawl.get("redirects", [])
            if r["redirect_count"] >= 2
        ]

    def generate_redirect_snippet(self, redirects: List[Dict]) -> str:
        """Generate PHP redirect snippet from a list of {from, to} redirects.

        Args:
            redirects: [{from_url: str, to_url: str}]
        """
        if not redirects:
            return "<!-- No redirects needed -->"

        entries = "\n    ".join(
            f"'{r['from_url']}' => '{r['to_url']}',"
            for r in redirects[:50]
        )

        return f"""<?php
/**
 * Auto Redirects — Generated by Site Evolution Engine.
 * 301 redirects for broken/moved URLs.
 */
function evo_auto_redirects() {{
    if (is_admin()) return;

    $redirects = array(
    {entries}
    );

    $request_uri = rtrim($_SERVER['REQUEST_URI'], '/');
    foreach ($redirects as $from => $to) {{
        if (rtrim($from, '/') === $request_uri) {{
            wp_redirect($to, 301);
            exit;
        }}
    }}
}}
add_action('template_redirect', 'evo_auto_redirects', 1);
"""

    def get_link_health_score(self, site_slug: str) -> Dict:
        """Score 0-100 for auditor integration."""
        score = 50  # Base
        issues = []

        try:
            crawl = self.crawl_links(site_slug, limit=30)
        except Exception as e:
            return {"score": 30, "issues": [{"type": "warning", "msg": f"Link crawl failed: {e}"}]}

        total = crawl.get("total_links", 0)
        broken = crawl.get("broken_count", 0)
        redirects = crawl.get("redirect_count", 0)

        if total == 0:
            return {"score": 50, "issues": [{"type": "info", "msg": "No links found to check"}]}

        # Broken link ratio
        broken_ratio = broken / max(total, 1)
        if broken_ratio == 0:
            score += 30
        elif broken_ratio < 0.02:
            score += 20
        elif broken_ratio < 0.05:
            score += 10
        else:
            issues.append({"type": "critical", "msg": f"{broken} broken links found ({broken_ratio:.0%})"})

        # Redirect chain check
        chains = [r for r in crawl.get("redirects", []) if r["redirect_count"] >= 2]
        if not chains:
            score += 20
        elif len(chains) <= 3:
            score += 10
            issues.append({"type": "info", "msg": f"{len(chains)} redirect chains detected"})
        else:
            issues.append({"type": "warning", "msg": f"{len(chains)} redirect chains (slow)"})

        return {"score": min(100, score), "issues": issues}
