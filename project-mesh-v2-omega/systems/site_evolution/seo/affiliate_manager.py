"""
Affiliate Link Manager — Audit Amazon/affiliate links, generate disclosure snippets,
auto-add nofollow/sponsored, GA4 click tracking.
"""

import logging
import re
from typing import Dict, List

from systems.site_evolution.utils import load_site_config, get_site_domain, get_site_brand_name

log = logging.getLogger(__name__)

# Affiliate link patterns
AFFILIATE_PATTERNS = [
    r'amazon\.\w+/.*(?:tag=|ref=)',
    r'amzn\.to/',
    r'shareasale\.com/',
    r'awin1\.com/',
    r'commission-junction\.com/',
    r'clickbank\.net/',
    r'partner\.com/',
    r'affiliates?\.',
    r'ref=\w+',
    r'tracking_id=',
]


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


def _extract_title(post: Dict) -> str:
    t = post.get("title", {})
    return t.get("rendered", "") if isinstance(t, dict) else str(t)


class AffiliateLinkManager:
    """Audit and optimize affiliate links across sites."""

    def audit_affiliate_links(self, site_slug: str) -> Dict:
        """Find affiliate links and check rel="nofollow sponsored" compliance.

        Returns: {total_posts, posts_with_affiliate, links_found, compliant, non_compliant, score}
        """
        posts = _get_posts(site_slug)
        all_links = []
        posts_with_affiliate = 0

        for post in posts:
            content = post.get("content", {})
            if isinstance(content, dict):
                content = content.get("rendered", "")

            # Find all <a> tags
            a_tags = re.findall(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', content, re.IGNORECASE | re.DOTALL)
            post_has_affiliate = False

            for url, anchor_html in a_tags:
                is_affiliate = any(re.search(p, url, re.IGNORECASE) for p in AFFILIATE_PATTERNS)
                if not is_affiliate:
                    continue

                post_has_affiliate = True
                # Check for rel attributes
                full_tag = re.search(
                    r'<a\s[^>]*href=["\']' + re.escape(url) + r'["\'][^>]*>',
                    content, re.IGNORECASE
                )
                tag_str = full_tag.group(0) if full_tag else ""
                rel_match = re.search(r'rel=["\']([^"\']*)["\']', tag_str, re.IGNORECASE)
                rel_value = rel_match.group(1) if rel_match else ""

                has_nofollow = "nofollow" in rel_value
                has_sponsored = "sponsored" in rel_value

                all_links.append({
                    "post_id": post.get("id"),
                    "post_title": _extract_title(post),
                    "url": url[:200],
                    "has_nofollow": has_nofollow,
                    "has_sponsored": has_sponsored,
                    "compliant": has_nofollow and has_sponsored,
                    "rel": rel_value,
                })

            if post_has_affiliate:
                posts_with_affiliate += 1

        compliant = [l for l in all_links if l["compliant"]]
        non_compliant = [l for l in all_links if not l["compliant"]]

        # Score
        if not all_links:
            score = 100  # No affiliate links = no compliance needed
        else:
            compliance_rate = len(compliant) / len(all_links)
            score = int(compliance_rate * 80) + 20  # Base 20 for having tracking

        return {
            "site_slug": site_slug,
            "total_posts": len(posts),
            "posts_with_affiliate": posts_with_affiliate,
            "links_found": len(all_links),
            "compliant": len(compliant),
            "non_compliant": len(non_compliant),
            "non_compliant_details": non_compliant[:20],
            "score": min(100, score),
        }

    def generate_disclosure_snippet(self, site_slug: str) -> str:
        """PHP snippet to auto-insert affiliate disclosure on posts with affiliate links."""
        brand = get_site_brand_name(site_slug)
        return f"""<?php
/**
 * Affiliate Disclosure — {site_slug}
 * Auto-inserts disclosure at top of posts containing affiliate links.
 */
function evo_affiliate_disclosure($content) {{
    if (!is_single()) return $content;

    // Check if post contains affiliate links
    $affiliate_patterns = array('amazon.', 'amzn.to', 'shareasale.com', 'awin1.com', 'tag=');
    $has_affiliate = false;
    foreach ($affiliate_patterns as $pattern) {{
        if (stripos($content, $pattern) !== false) {{
            $has_affiliate = true;
            break;
        }}
    }}

    if (!$has_affiliate) return $content;

    $disclosure = '<div class="evo-disclosure" style="background:#FEF3C7;border:1px solid #F59E0B;border-radius:8px;padding:12px 16px;margin-bottom:24px;font-size:14px;line-height:1.5;color:#92400E;">'
        . '<strong>Disclosure:</strong> {brand} may earn a commission from qualifying purchases through affiliate links in this article. '
        . 'This helps support our work at no additional cost to you. <a href="/affiliate-disclosure" style="color:#92400E;">Learn more</a>.'
        . '</div>';

    return $disclosure . $content;
}}
add_filter('the_content', 'evo_affiliate_disclosure', 5);
"""

    def generate_nofollow_snippet(self) -> str:
        """PHP snippet to auto-add rel='nofollow sponsored' to affiliate links."""
        return """<?php
/**
 * Affiliate Nofollow — Auto-add nofollow sponsored to external affiliate links.
 */
function evo_affiliate_nofollow($content) {
    if (!is_single() && !is_page()) return $content;

    $affiliate_domains = array('amazon.', 'amzn.to', 'shareasale.com', 'awin1.com',
                                'commission-junction.com', 'clickbank.net');

    $content = preg_replace_callback(
        r'/<a\s([^>]*href=["\']([^"\']+)["\'][^>]*)>/i',
        function($matches) use ($affiliate_domains) {
            $tag = $matches[1];
            $url = $matches[2];

            // Check if URL is an affiliate link
            $is_affiliate = false;
            foreach ($affiliate_domains as $domain) {
                if (stripos($url, $domain) !== false) {
                    $is_affiliate = true;
                    break;
                }
            }

            if (!$is_affiliate) return $matches[0];

            // Add rel="nofollow sponsored" if not already present
            if (preg_match('/rel=["\']([^"\']*)["\']/', $tag, $rel_match)) {
                $rel = $rel_match[1];
                if (strpos($rel, 'nofollow') === false) $rel .= ' nofollow';
                if (strpos($rel, 'sponsored') === false) $rel .= ' sponsored';
                $tag = preg_replace('/rel=["\'][^"\']*["\']/', 'rel="' . trim($rel) . '"', $tag);
            } else {
                $tag .= ' rel="nofollow sponsored"';
            }

            return '<a ' . $tag . '>';
        },
        $content
    );

    return $content;
}
add_filter('the_content', 'evo_affiliate_nofollow', 15);
"""

    def generate_click_tracking_snippet(self, site_slug: str) -> str:
        """JS snippet for GA4 event tracking on affiliate link clicks."""
        return """<script>
/**
 * Affiliate Click Tracking — GA4 event on affiliate link clicks.
 */
(function(){
  var affiliateDomains = ['amazon.', 'amzn.to', 'shareasale.com', 'awin1.com'];

  document.addEventListener('click', function(e) {
    var link = e.target.closest('a[href]');
    if (!link) return;

    var href = link.getAttribute('href') || '';
    var isAffiliate = affiliateDomains.some(function(d) { return href.indexOf(d) !== -1; });

    if (isAffiliate && typeof gtag === 'function') {
      gtag('event', 'affiliate_click', {
        'event_category': 'affiliate',
        'event_label': href.substring(0, 200),
        'link_text': (link.textContent || '').substring(0, 100),
        'page_url': window.location.pathname,
      });
    }
  });
})();
</script>"""

    def get_affiliate_health_score(self, site_slug: str) -> Dict:
        """Score 0-100 for auditor integration."""
        audit = self.audit_affiliate_links(site_slug)
        return {
            "score": audit.get("score", 100),
            "issues": [] if audit.get("score", 100) >= 80 else [
                {"type": "warning", "msg": f"{audit.get('non_compliant', 0)} affiliate links missing nofollow/sponsored"}
            ],
        }

    def generate_all_affiliate_snippets(self, site_slug: str) -> Dict[str, str]:
        """Generate all affiliate compliance snippets."""
        return {
            "disclosure": self.generate_disclosure_snippet(site_slug),
            "nofollow": self.generate_nofollow_snippet(),
            "click_tracking": self.generate_click_tracking_snippet(site_slug),
        }
