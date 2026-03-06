"""
AdSense Optimizer — Audit ad placement, generate optimal ad snippets,
viewability optimization, density compliance.
"""

import logging
import re
from typing import Dict, List

from systems.site_evolution.utils import load_site_config, get_site_domain

log = logging.getLogger(__name__)


class AdSenseOptimizer:
    """Optimize AdSense ad placement and compliance."""

    def audit_ad_placement(self, site_slug: str) -> Dict:
        """Check current ad configuration.

        Returns: {has_adsense, ad_count, above_fold, density_ok, score, issues}
        """
        domain = get_site_domain(site_slug)
        score = 30  # Base
        issues = []
        has_adsense = False
        ad_count = 0

        if not domain:
            return {"score": 0, "issues": [{"type": "warning", "msg": "No domain configured"}]}

        try:
            import requests
            resp = requests.get(f"https://{domain}", timeout=10,
                                headers={"User-Agent": "EvoAdAuditor/1.0"})
            html = resp.text

            # Check for AdSense
            if "adsbygoogle" in html or "pagead2.googlesyndication.com" in html:
                has_adsense = True
                score += 20

            # Count ad units
            ad_count = html.count("adsbygoogle") + html.count("data-ad-slot")

            # Check density (Google recommends max 30% of viewport)
            content_length = len(re.sub(r'<[^>]+>', '', html))
            if ad_count > 0:
                if ad_count <= 3:
                    score += 20
                elif ad_count <= 5:
                    score += 10
                    issues.append({"type": "info", "msg": f"{ad_count} ad units — consider reducing"})
                else:
                    issues.append({"type": "warning", "msg": f"High ad density: {ad_count} ad units"})

            # Check for ads.txt
            try:
                ads_resp = requests.get(f"https://{domain}/ads.txt", timeout=5)
                if ads_resp.status_code == 200 and "google.com" in ads_resp.text:
                    score += 15
                else:
                    issues.append({"type": "warning", "msg": "ads.txt not found or incomplete"})
            except requests.RequestException:
                issues.append({"type": "info", "msg": "Could not verify ads.txt"})

            if not has_adsense:
                issues.append({"type": "info", "msg": "No AdSense detected"})

        except Exception as e:
            issues.append({"type": "warning", "msg": f"Ad audit failed: {e}"})

        return {
            "site_slug": site_slug,
            "has_adsense": has_adsense,
            "ad_count": ad_count,
            "score": min(100, score),
            "issues": issues,
        }

    def generate_optimal_ad_snippet(self, site_slug: str) -> str:
        """PHP snippet for optimal in-content ad placement.

        Places ads after 2nd paragraph, mid-content, and before conclusion.
        """
        return f"""<?php
/**
 * Optimal Ad Placement — {site_slug}
 * Inserts ads after 2nd paragraph, mid-content, and before last paragraph.
 * Only on single posts, not pages.
 */
function evo_insert_content_ads($content) {{
    if (!is_single()) return $content;

    // Don't insert on short content
    if (str_word_count(strip_tags($content)) < 300) return $content;

    $ad_code = '<div class="evo-ad-unit" style="margin:24px 0;text-align:center;min-height:250px;">'
             . '<!-- Ad placement managed by theme/plugin -->'
             . '<ins class="adsbygoogle" style="display:block" data-ad-format="auto" data-full-width-responsive="true"></ins>'
             . '<script>(adsbygoogle = window.adsbygoogle || []).push({{}});</script>'
             . '</div>';

    // Split content by paragraphs
    $paragraphs = explode('</p>', $content);
    $total = count($paragraphs);

    if ($total < 4) return $content;

    // Insert after 2nd paragraph
    $insert_positions = array(2);

    // Insert at mid-content
    $mid = intval($total / 2);
    if ($mid > 3 && $mid !== 2) {{
        $insert_positions[] = $mid;
    }}

    // Insert before last paragraph (if enough content)
    if ($total > 8) {{
        $insert_positions[] = $total - 2;
    }}

    // Rebuild content with ad insertions
    $new_content = '';
    $ad_count = 0;
    $max_ads = 3;

    for ($i = 0; $i < $total; $i++) {{
        $new_content .= $paragraphs[$i] . ($i < $total - 1 ? '</p>' : '');
        if (in_array($i + 1, $insert_positions) && $ad_count < $max_ads) {{
            $new_content .= $ad_code;
            $ad_count++;
        }}
    }}

    return $new_content;
}}
add_filter('the_content', 'evo_insert_content_ads', 50);
"""

    def generate_viewability_snippet(self) -> str:
        """JS snippet to lazy-load ads only when in viewport."""
        return """<script>
/**
 * Ad Viewability — Only load ads when they enter the viewport.
 */
(function(){
  if (typeof IntersectionObserver === 'undefined') return;

  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        var ad = entry.target;
        if (ad.dataset.evoAdLoaded) return;
        ad.dataset.evoAdLoaded = 'true';
        // Trigger adsbygoogle push
        try { (adsbygoogle = window.adsbygoogle || []).push({}); } catch(e) {}
        observer.unobserve(ad);
      }
    });
  }, { rootMargin: '200px' });

  document.querySelectorAll('.evo-ad-unit ins.adsbygoogle').forEach(function(ad) {
    observer.observe(ad);
  });
})();
</script>"""

    def generate_density_check_snippet(self) -> str:
        """PHP snippet to prevent exceeding Google ad density limits."""
        return """<?php
/**
 * Ad Density Guard — Prevents too many ads on short content.
 */
function evo_ad_density_guard($content) {
    if (!is_single()) return $content;

    $word_count = str_word_count(strip_tags($content));
    $ad_count = substr_count($content, 'evo-ad-unit');

    // Google guideline: roughly 1 ad per 300 words max
    $max_ads = max(1, intval($word_count / 300));

    if ($ad_count > $max_ads) {
        // Remove excess ads from the end
        $excess = $ad_count - $max_ads;
        for ($i = 0; $i < $excess; $i++) {
            $pos = strrpos($content, '<div class="evo-ad-unit"');
            if ($pos !== false) {
                $end_pos = strpos($content, '</div>', $pos);
                if ($end_pos !== false) {
                    $content = substr($content, 0, $pos) . substr($content, $end_pos + 6);
                }
            }
        }
    }

    return $content;
}
add_filter('the_content', 'evo_ad_density_guard', 99);
"""
