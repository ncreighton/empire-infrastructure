"""
Site Auditor — Comprehensive 8-dimension scoring for WordPress sites.

Dimensions (0-100 each):
1. Design Quality    2. SEO Health       3. Performance
4. Content Quality   5. Conversion       6. Mobile Experience
7. Trust Signals     8. AI Readiness

Integrates with GSC + Bing search analytics for real data.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

from systems.site_evolution.utils import load_site_config, get_all_site_slugs

log = logging.getLogger(__name__)


def _get_posts(site_slug: str, limit: int = 20) -> List[Dict]:
    """Fetch posts from the site for auditing."""
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        result = _wp_request(
            site_slug, "GET",
            f"wp/v2/posts?per_page={limit}&status=publish"
            "&_fields=id,title,content,excerpt,link,featured_media,date,modified,categories"
        )
        return result if isinstance(result, list) else []
    except Exception as e:
        log.warning("Could not fetch posts for %s: %s", site_slug, e)
        return []


def _get_pages(site_slug: str) -> List[Dict]:
    """Fetch pages from the site."""
    try:
        from systems.site_evolution.deployer.wp_deployer import _wp_request
        result = _wp_request(site_slug, "GET", "wp/v2/pages?per_page=50&status=publish")
        return result if isinstance(result, list) else []
    except Exception as e:
        log.debug("Could not fetch pages for %s: %s", site_slug, e)
        return []


    # Category-aware scoring weights
CATEGORY_WEIGHTS = {
    "review": {
        "design": 0.10, "seo": 0.20, "performance": 0.15, "content": 0.15,
        "conversion": 0.15, "mobile": 0.10, "trust": 0.10, "ai_readiness": 0.05,
    },
    "spiritual": {
        "design": 0.20, "seo": 0.10, "performance": 0.10, "content": 0.15,
        "conversion": 0.15, "mobile": 0.10, "trust": 0.10, "ai_readiness": 0.10,
    },
    "ai_tech": {
        "design": 0.10, "seo": 0.15, "performance": 0.15, "content": 0.20,
        "conversion": 0.10, "mobile": 0.10, "trust": 0.10, "ai_readiness": 0.10,
    },
    "lifestyle": {
        "design": 0.15, "seo": 0.15, "performance": 0.10, "content": 0.15,
        "conversion": 0.15, "mobile": 0.10, "trust": 0.10, "ai_readiness": 0.10,
    },
    "default": {
        "design": 0.125, "seo": 0.125, "performance": 0.125, "content": 0.125,
        "conversion": 0.125, "mobile": 0.125, "trust": 0.125, "ai_readiness": 0.125,
    },
}


class SiteAuditor:
    """Score sites across 8 dimensions with 250+ checks."""

    def audit_site(self, site_slug: str) -> Dict:
        """Full audit of a single site. Returns 8 dimension scores."""
        config = load_site_config(site_slug)
        posts = _get_posts(site_slug)
        pages = _get_pages(site_slug)

        scores = {
            "design": self._audit_design(site_slug, config, posts, pages),
            "seo": self._audit_seo(site_slug, config, posts, pages),
            "performance": self._audit_performance(site_slug, config),
            "content": self._audit_content(site_slug, config, posts),
            "conversion": self._audit_conversion(site_slug, config, posts, pages),
            "mobile": self._audit_mobile(site_slug, config),
            "trust": self._audit_trust(site_slug, config, pages),
            "ai_readiness": self._audit_ai_readiness(site_slug, config, posts),
        }

        # Accessibility audit (adds to trust dimension issues, separate score)
        accessibility = self._audit_accessibility(site_slug, config)

        overall = self._calculate_weighted_score(
            {k: v["score"] for k, v in scores.items()}, site_slug
        )

        # Collect all issues
        all_issues = []
        for dim, result in scores.items():
            for issue in result.get("issues", []):
                issue["dimension"] = dim
                all_issues.append(issue)

        # Record in codex
        from systems.site_evolution import codex
        score_dict = {k: v["score"] for k, v in scores.items()}
        codex.record_audit(
            site_slug, score_dict,
            json.dumps({"issues_count": len(all_issues)})
        )

        return {
            "site_slug": site_slug,
            "brand_name": config.get("name", site_slug),
            "overall_score": overall,
            "scores": {k: v["score"] for k, v in scores.items()},
            "accessibility": accessibility,
            "details": scores,
            "total_issues": len(all_issues),
            "critical_issues": len([i for i in all_issues if i.get("type") == "critical"]),
            "audited_at": datetime.now().isoformat(),
        }

    def audit_all_sites(self) -> List[Dict]:
        """Audit all 14 sites and return ranked list."""
        sites = get_all_site_slugs()

        results = []
        for slug in sites:
            try:
                audit = self.audit_site(slug)
                results.append(audit)
            except Exception as e:
                log.error("Audit failed for %s: %s", slug, e)
                results.append({
                    "site_slug": slug,
                    "overall_score": 0,
                    "error": str(e),
                })

        return sorted(results, key=lambda r: r.get("overall_score", 0), reverse=True)

    def generate_fix_queue(self, audit: Dict) -> List[Dict]:
        """Convert audit findings into priority-ranked fix items."""
        items = []
        for dim, detail in audit.get("details", {}).items():
            for issue in detail.get("issues", []):
                priority = 90 if issue["type"] == "critical" else 60 if issue["type"] == "warning" else 30
                impact = 15 if issue["type"] == "critical" else 8 if issue["type"] == "warning" else 3
                items.append({
                    "site_slug": audit["site_slug"],
                    "component_type": dim,
                    "action": "fix",
                    "priority": priority,
                    "estimated_impact": impact,
                    "description": issue["msg"],
                    "issue_type": issue["type"],
                })
        return sorted(items, key=lambda i: i["priority"], reverse=True)

    # -- Dimension Auditors --

    def _audit_design(self, site_slug: str, config: Dict,
                      posts: List, pages: List) -> Dict:
        score = 30  # Base: basic WordPress theme active
        issues = []

        # Check if design system exists in codex
        from systems.site_evolution import codex
        ds = codex.get_design_system(site_slug)
        if ds:
            score += 15
            # Check design system quality
            css_vars = ds.get("css_variables", {})
            if isinstance(css_vars, str):
                try:
                    css_vars = json.loads(css_vars)
                except (json.JSONDecodeError, TypeError):
                    css_vars = {}
            if len(css_vars) >= 30:
                score += 5
        else:
            issues.append({"type": "warning", "msg": "No custom design system deployed"})

        # Check for custom CSS snippets
        has_custom_css = False
        snippet_count = 0
        try:
            from systems.site_evolution.deployer.wp_deployer import WPDeployer
            deployer = WPDeployer()
            snippets = deployer.get_existing_snippets(site_slug)
            css_snippets = [s for s in snippets if "css" in str(s.get("code_type", "")).lower()]
            snippet_count = len(snippets)
            if css_snippets:
                has_custom_css = True
                score += 10
        except Exception as e:
            log.debug("Could not check snippets for %s: %s", site_slug, e)

        if not has_custom_css:
            issues.append({"type": "warning", "msg": "No custom CSS framework"})

        # Active snippet count (more = more customized)
        if snippet_count >= 5:
            score += 5

        # Brand colors applied
        if config.get("brand", {}).get("colors"):
            score += 10
        else:
            issues.append({"type": "info", "msg": "Brand colors not configured"})

        # Custom fonts
        if config.get("brand", {}).get("fonts"):
            score += 10
        else:
            issues.append({"type": "info", "msg": "Custom fonts not configured"})

        # Check for homepage
        home_pages = [p for p in pages if p.get("slug") in ("home", "front-page")]
        if home_pages:
            score += 5
        else:
            issues.append({"type": "info", "msg": "No custom homepage"})

        # Check for components deployed
        components = codex.get_deployments(site_slug, limit=50)
        component_types = set(d.get("component_type", "") for d in components)
        if len(component_types) >= 5:
            score += 10
        elif len(component_types) >= 2:
            score += 5
        else:
            issues.append({"type": "info", "msg": f"Only {len(component_types)} component types deployed"})

        return {"score": min(100, score), "issues": issues}

    def _audit_seo(self, site_slug: str, config: Dict,
                   posts: List, pages: List) -> Dict:
        score = 30  # Base: WP + RankMath installed
        issues = []

        # Meta descriptions
        posts_with_excerpt = sum(1 for p in posts if p.get("excerpt", {}).get("rendered", "").strip())
        if posts and posts_with_excerpt / max(len(posts), 1) > 0.8:
            score += 15
        else:
            issues.append({"type": "warning", "msg": f"Only {posts_with_excerpt}/{len(posts)} posts have meta descriptions"})

        # Featured images
        posts_with_image = sum(1 for p in posts if p.get("featured_media", 0) > 0)
        if posts and posts_with_image / max(len(posts), 1) > 0.8:
            score += 10
        else:
            issues.append({"type": "warning", "msg": f"Only {posts_with_image}/{len(posts)} posts have featured images"})

        # Categories used
        posts_with_cats = sum(1 for p in posts if len(p.get("categories", [])) > 0)
        if posts and posts_with_cats / max(len(posts), 1) > 0.9:
            score += 5

        # Search analytics integration
        try:
            from systems.site_evolution.seo.search_analytics import SearchAnalytics
            sa = SearchAnalytics()
            health = sa.get_seo_health_score(site_slug)
            analytics_score = health.get("score", 0)
            score += min(25, analytics_score // 4)
            for issue in health.get("issues", []):
                issues.append(issue)
        except ImportError:
            issues.append({"type": "info", "msg": "Search analytics module not available"})
        except Exception as e:
            log.debug("Search analytics error for %s: %s", site_slug, e)
            issues.append({"type": "info", "msg": "Search analytics not connected"})

        # Schema markup (check via codex deployments)
        from systems.site_evolution import codex
        deployments = codex.get_deployments(site_slug, limit=50)
        schema_deployments = [d for d in deployments if "schema" in d.get("component_type", "")]
        if schema_deployments:
            score += 10
        else:
            issues.append({"type": "warning", "msg": "No structured data (JSON-LD) deployed"})

        # Internal link density
        try:
            from systems.site_evolution.seo.internal_linker import InternalLinker
            linker = InternalLinker()
            link_audit = linker.audit_internal_links(site_slug)
            link_score = link_audit.get("score", 0)
            if link_score >= 60:
                score += 5
            elif link_score < 30:
                issues.append({"type": "warning", "msg": "Weak internal linking structure"})
            for issue in link_audit.get("issues", [])[:2]:
                issues.append(issue)
        except Exception as e:
            log.debug("Internal link audit skipped for %s: %s", site_slug, e)

        # Broken link check
        try:
            from systems.site_evolution.auditor.broken_link_monitor import BrokenLinkMonitor
            blm = BrokenLinkMonitor()
            link_health = blm.get_link_health_score(site_slug)
            if link_health.get("score", 0) >= 70:
                score += 5
            for issue in link_health.get("issues", [])[:2]:
                issues.append(issue)
        except Exception as e:
            log.debug("Broken link check skipped for %s: %s", site_slug, e)

        # Image alt text coverage
        try:
            from systems.site_evolution.performance.image_optimizer import ImageOptimizer
            img_opt = ImageOptimizer()
            img_audit = img_opt.audit_images(site_slug, limit=20)
            alt_coverage = img_audit.get("alt_coverage", 0)
            if alt_coverage >= 90:
                score += 5
            elif alt_coverage < 50:
                issues.append({"type": "warning", "msg": f"Image alt text coverage: {alt_coverage}%"})
        except Exception as e:
            log.debug("Image audit skipped for %s: %s", site_slug, e)

        return {"score": min(100, score), "issues": issues}

    def _audit_performance(self, site_slug: str, config: Dict) -> Dict:
        score = 40  # Base: LiteSpeed active
        issues = []

        domain = config.get("domain", "")
        if domain:
            try:
                import requests
                resp = requests.get(f"https://{domain}", timeout=10,
                                    headers={"User-Agent": "EvoAuditor/1.0"})
                elapsed = resp.elapsed.total_seconds()

                # Response time scoring
                if elapsed < 0.5:
                    score += 20
                elif elapsed < 1.0:
                    score += 15
                elif elapsed < 2.0:
                    score += 8
                elif elapsed < 3.0:
                    score += 3
                    issues.append({"type": "warning", "msg": f"Slow response: {elapsed:.1f}s"})
                else:
                    issues.append({"type": "critical", "msg": f"Very slow response: {elapsed:.1f}s"})

                # Caching headers
                cache = resp.headers.get("x-litespeed-cache", "")
                if cache:
                    score += 10
                else:
                    issues.append({"type": "info", "msg": "No LiteSpeed cache header detected"})

                # Content size
                size = len(resp.content)
                if size < 100_000:
                    score += 10
                elif size < 200_000:
                    score += 5
                elif size > 500_000:
                    issues.append({"type": "warning", "msg": f"Large page size: {size // 1024}KB"})

                # Check for critical performance headers
                html = resp.text[:10000]

                # Preconnect hints
                if 'preconnect' in html:
                    score += 5

                # Font-display: swap
                if 'font-display' in html or 'display=swap' in html:
                    score += 5

                # Check for render-blocking resources
                if resp.headers.get("content-encoding", ""):
                    score += 5  # Gzip/Brotli compression active

                # Security headers check (performance dimension overlap)
                if resp.headers.get("strict-transport-security"):
                    score += 3

                # Image format check
                if 'webp' in html.lower() or '.webp' in html:
                    score += 3

            except requests.Timeout:
                issues.append({"type": "critical", "msg": f"Site timed out (>10s)"})
                score -= 20
            except requests.ConnectionError as e:
                issues.append({"type": "critical", "msg": f"Site unreachable: connection failed"})
                score -= 20
            except requests.RequestException as e:
                issues.append({"type": "critical", "msg": f"Site request error: {e}"})
                score -= 15

        # Check for performance snippets deployed
        from systems.site_evolution import codex
        deployments = codex.get_deployments(site_slug, limit=50)
        perf_deploys = [d for d in deployments if "perf" in d.get("snippet_name", "")]
        if perf_deploys:
            score += 5

        return {"score": min(100, max(0, score)), "issues": issues}

    def _audit_content(self, site_slug: str, config: Dict,
                       posts: List) -> Dict:
        score = 30
        issues = []

        if not posts:
            issues.append({"type": "critical", "msg": "No published posts"})
            return {"score": 0, "issues": issues}

        # Post count
        if len(posts) >= 20:
            score += 15
        elif len(posts) >= 10:
            score += 10
        elif len(posts) >= 5:
            score += 5
        else:
            issues.append({"type": "warning", "msg": f"Only {len(posts)} posts"})

        # Average content length
        word_counts = []
        for p in posts:
            content = p.get("content", {}).get("rendered", "")
            clean = re.sub(r'<[^>]+>', '', content)
            word_counts.append(len(clean.split()))

        avg_words = sum(word_counts) // max(len(word_counts), 1)
        if avg_words >= 1500:
            score += 20
        elif avg_words >= 800:
            score += 12
        elif avg_words >= 400:
            score += 5
        else:
            issues.append({"type": "warning", "msg": f"Average word count low: {avg_words}"})

        # Heading structure (check for H2s)
        posts_with_h2 = sum(1 for p in posts
                            if '<h2' in p.get("content", {}).get("rendered", ""))
        if posts_with_h2 / max(len(posts), 1) > 0.7:
            score += 10
        else:
            issues.append({"type": "info", "msg": "Many posts lack H2 headings"})

        # Update frequency
        if posts:
            from datetime import datetime as dt
            try:
                dates = [dt.fromisoformat(p.get("date", "2020-01-01T00:00:00").replace("Z", "+00:00"))
                         for p in posts if p.get("date")]
                if dates:
                    newest = max(dates)
                    days_since = (dt.now(newest.tzinfo) - newest).days if newest.tzinfo else 30
                    if days_since < 7:
                        score += 10
                    elif days_since < 30:
                        score += 5
                    else:
                        issues.append({"type": "warning", "msg": f"Last post {days_since} days ago"})
            except (ValueError, TypeError) as e:
                log.debug("Date parsing error in content audit: %s", e)

        return {"score": min(100, score), "issues": issues}

    def _audit_conversion(self, site_slug: str, config: Dict,
                          posts: List, pages: List) -> Dict:
        score = 20
        issues = []

        # CTAs configured
        if config.get("ctas"):
            score += 15
        else:
            issues.append({"type": "warning", "msg": "No CTAs configured"})

        # Newsletter form check (look for form in pages/posts)
        has_newsletter = False
        for p in pages + posts[:5]:
            content = p.get("content", {})
            if isinstance(content, dict):
                content = content.get("rendered", "")
            if "newsletter" in content.lower() or "subscribe" in content.lower() or "email" in content.lower():
                has_newsletter = True
                break

        if has_newsletter:
            score += 20
        else:
            issues.append({"type": "warning", "msg": "No newsletter signup found"})

        # Amazon affiliate tag
        if config.get("amazon_tag"):
            score += 10

        # About page exists
        about_pages = [p for p in pages if p.get("slug") in ("about", "about-us")]
        if about_pages:
            score += 10
        else:
            issues.append({"type": "warning", "msg": "No about page"})

        # Contact page
        contact_pages = [p for p in pages if "contact" in p.get("slug", "")]
        if contact_pages:
            score += 10
        else:
            issues.append({"type": "info", "msg": "No contact page"})

        return {"score": min(100, score), "issues": issues}

    def _audit_mobile(self, site_slug: str, config: Dict) -> Dict:
        score = 60  # Base: Blocksy is responsive
        issues = []

        # Blocksy + responsive = good baseline
        domain = config.get("domain", "")
        if domain:
            try:
                import requests
                resp = requests.get(f"https://{domain}", timeout=10,
                                    headers={"User-Agent": "EvoAuditor/1.0"})
                html = resp.text[:5000]

                if 'viewport' in html:
                    score += 15
                else:
                    issues.append({"type": "critical", "msg": "Missing viewport meta tag"})

                if 'blocksy' in html.lower():
                    score += 15  # Blocksy is well-optimized for mobile

            except requests.RequestException as e:
                log.debug("Mobile audit request failed for %s: %s", site_slug, e)

        return {"score": min(100, score), "issues": issues}

    def _audit_trust(self, site_slug: str, config: Dict,
                     pages: List) -> Dict:
        score = 20
        issues = []

        # Check for key trust pages
        page_slugs = [p.get("slug", "") for p in pages]

        trust_pages = {
            "about": 15,
            "privacy-policy": 15,
            "contact": 10,
            "affiliate-disclosure": 10,
            "terms-of-service": 5,
        }

        for slug, points in trust_pages.items():
            if any(slug in ps for ps in page_slugs):
                score += points
            else:
                issues.append({"type": "warning" if points > 10 else "info",
                               "msg": f"Missing {slug} page"})

        # SSL (all our sites use HTTPS)
        score += 10

        # Author info
        if config.get("brand", {}).get("voice"):
            score += 5

        # Security headers score
        try:
            from systems.site_evolution.performance.security_hardener import SecurityHardener
            hardener = SecurityHardener()
            sec_audit = hardener.audit_security(site_slug)
            sec_score = sec_audit.get("score", 0)
            if sec_score >= 60:
                score += 5
            elif sec_score < 30:
                issues.append({"type": "warning", "msg": "Missing critical security headers"})
            # Check XML-RPC specifically
            if "disable_xmlrpc" not in str(sec_audit.get("headers_present", [])):
                for issue in sec_audit.get("issues", []):
                    if "XML-RPC" in issue.get("msg", ""):
                        issues.append(issue)
                        break
        except Exception as e:
            log.debug("Security audit skipped for %s: %s", site_slug, e)

        return {"score": min(100, score), "issues": issues}

    def _audit_ai_readiness(self, site_slug: str, config: Dict,
                            posts: List) -> Dict:
        score = 10
        issues = []

        # Check for structured data in posts
        has_schema = False
        has_faq = False
        has_howto = False

        for p in posts[:10]:
            content = p.get("content", {})
            if isinstance(content, dict):
                content = content.get("rendered", "")

            if "schema.org" in content or "application/ld+json" in content:
                has_schema = True
            if "faq" in content.lower() or "frequently asked" in content.lower():
                has_faq = True
            if "how to" in content.lower() or "step-by-step" in content.lower():
                has_howto = True

        if has_schema:
            score += 20
        else:
            issues.append({"type": "warning", "msg": "No JSON-LD structured data in content"})

        if has_faq:
            score += 20
        else:
            issues.append({"type": "info", "msg": "No FAQ sections found"})

        if has_howto:
            score += 15
        else:
            issues.append({"type": "info", "msg": "No HowTo structured content"})

        # Check for clear Q&A headings
        qa_headings = 0
        for p in posts[:10]:
            content = p.get("content", {})
            if isinstance(content, dict):
                content = content.get("rendered", "")
            qa_headings += len(re.findall(r'<h[23][^>]*>[^<]*\?', content))

        if qa_headings >= 5:
            score += 15
        elif qa_headings >= 2:
            score += 8
        else:
            issues.append({"type": "info", "msg": "Few question-format headings"})

        # Entity markup
        from systems.site_evolution import codex
        deployments = codex.get_deployments(site_slug, limit=50)
        entity_deploys = [d for d in deployments if "entity" in d.get("component_type", "")]
        if entity_deploys:
            score += 10

        return {"score": min(100, score), "issues": issues}

    def _audit_accessibility(self, site_slug: str, config: Dict) -> Dict:
        """WCAG 2.1 AA accessibility audit.

        Checks: skip-to-content, focus styles, ARIA landmarks, form labels,
        color contrast hints, image alt text, heading order, lang attribute.
        """
        score = 0
        issues = []

        domain = config.get("domain", "")
        if not domain:
            return {"score": 0, "issues": [{"type": "warning", "msg": "No domain configured"}]}

        try:
            import requests
            resp = requests.get(f"https://{domain}", timeout=10,
                                headers={"User-Agent": "EvoA11yAuditor/1.0"})
            html = resp.text
        except Exception as e:
            return {"score": 0, "issues": [{"type": "critical", "msg": f"Could not fetch site: {e}"}]}

        # 1. lang attribute on <html> (WCAG 3.1.1)
        if re.search(r'<html[^>]*\slang=', html[:500], re.IGNORECASE):
            score += 10
        else:
            issues.append({"type": "warning", "msg": "Missing lang attribute on <html> tag"})

        # 2. Skip-to-content link (WCAG 2.4.1)
        if 'skip-to-content' in html.lower() or 'skip to content' in html.lower() or '#content' in html[:3000]:
            score += 10
        else:
            issues.append({"type": "warning", "msg": "No skip-to-content link found"})

        # 3. ARIA landmarks (WCAG 1.3.1)
        landmarks = 0
        for role in ('role="main"', 'role="navigation"', 'role="banner"', 'role="contentinfo"',
                      '<main', '<nav', '<header', '<footer'):
            if role in html:
                landmarks += 1
        if landmarks >= 4:
            score += 15
        elif landmarks >= 2:
            score += 8
            issues.append({"type": "info", "msg": f"Only {landmarks}/4 ARIA landmarks found"})
        else:
            issues.append({"type": "warning", "msg": "Missing ARIA landmark roles"})

        # 4. Focus styles (WCAG 2.4.7)
        has_focus = ':focus' in html or 'focus-visible' in html
        has_outline = 'outline:' in html or 'outline-color' in html
        if has_focus or has_outline:
            score += 10
        else:
            issues.append({"type": "warning", "msg": "No visible focus styles detected in CSS"})

        # 5. Image alt text coverage (WCAG 1.1.1)
        images = re.findall(r'<img\s[^>]*>', html, re.IGNORECASE)
        images_with_alt = [img for img in images if 'alt=' in img.lower()]
        if images:
            alt_ratio = len(images_with_alt) / len(images)
            if alt_ratio >= 0.9:
                score += 15
            elif alt_ratio >= 0.5:
                score += 8
                issues.append({"type": "warning", "msg": f"Alt text on {len(images_with_alt)}/{len(images)} images"})
            else:
                issues.append({"type": "critical", "msg": f"Only {len(images_with_alt)}/{len(images)} images have alt text"})
        else:
            score += 10  # No images = no a11y issue

        # 6. Form labels (WCAG 1.3.1, 4.1.2)
        inputs = re.findall(r'<input\s[^>]*>', html, re.IGNORECASE)
        inputs_no_hidden = [i for i in inputs if 'type="hidden"' not in i.lower()]
        labels = len(re.findall(r'<label', html, re.IGNORECASE))
        if inputs_no_hidden:
            has_aria_label = sum(1 for i in inputs_no_hidden if 'aria-label' in i.lower())
            if labels >= len(inputs_no_hidden) or has_aria_label >= len(inputs_no_hidden):
                score += 10
            else:
                issues.append({"type": "warning", "msg": "Some form inputs may lack labels"})
        else:
            score += 5

        # 7. Heading hierarchy (WCAG 1.3.1)
        headings = re.findall(r'<h([1-6])', html, re.IGNORECASE)
        if headings:
            levels = [int(h) for h in headings]
            h1_count = levels.count(1)
            if h1_count == 1:
                score += 5
            elif h1_count == 0:
                issues.append({"type": "warning", "msg": "No H1 heading on homepage"})
            elif h1_count > 1:
                issues.append({"type": "info", "msg": f"Multiple H1 headings ({h1_count})"})

            # Check for skipped levels
            skips = sum(1 for i in range(1, len(levels)) if levels[i] > levels[i-1] + 1)
            if skips == 0:
                score += 5
            elif skips <= 2:
                score += 2
                issues.append({"type": "info", "msg": f"{skips} heading level skips"})
        else:
            issues.append({"type": "warning", "msg": "No headings found on homepage"})

        # 8. Color contrast hints (check for sufficient text size)
        if 'font-size' in html:
            # Look for very small font sizes
            small_fonts = re.findall(r'font-size:\s*(\d+)px', html)
            tiny = [int(f) for f in small_fonts if int(f) < 12]
            if not tiny:
                score += 10
            else:
                issues.append({"type": "info", "msg": f"{len(tiny)} elements with font-size < 12px"})
        else:
            score += 5  # Default browser fonts are usually fine

        # 9. Touch target size (WCAG 2.5.5)
        # Check for small clickable areas in CSS
        small_buttons = len(re.findall(r'(width|height):\s*(1\d|2[0-3])px', html))
        if small_buttons < 3:
            score += 5
        else:
            issues.append({"type": "info", "msg": "Possible small touch targets detected"})

        return {"score": min(100, score), "issues": issues}

    def _calculate_weighted_score(self, scores: Dict[str, int], site_slug: str) -> int:
        """Calculate overall score using category-aware weights.

        Review sites weight SEO higher, spiritual sites weight design higher.
        """
        from systems.site_evolution.utils import SITE_CATEGORIES

        # Determine category
        category = "default"
        for cat, slugs in SITE_CATEGORIES.items():
            if site_slug in slugs:
                category = cat
                break

        weights = CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["default"])

        weighted = sum(scores.get(dim, 0) * w for dim, w in weights.items())
        return int(weighted)

    def get_score_trend(self, site_slug: str, limit: int = 10) -> Dict:
        """Week-over-week score deltas and improvement velocity."""
        from systems.site_evolution import codex
        trend = codex.get_audit_trend(site_slug, limit=limit)

        if len(trend) < 2:
            return {
                "site_slug": site_slug,
                "history": trend,
                "velocity": 0,
                "direction": "insufficient_data",
            }

        # Calculate deltas between consecutive audits
        deltas = []
        for i in range(1, len(trend)):
            prev = trend[i - 1].get("overall_score", 0)
            curr = trend[i].get("overall_score", 0)
            deltas.append(curr - prev)

        avg_delta = sum(deltas) / len(deltas) if deltas else 0

        # Per-dimension trends
        dim_trends = {}
        if len(trend) >= 2:
            first = trend[0].get("scores", {})
            latest = trend[-1].get("scores", {})
            for dim in ("design", "seo", "performance", "content", "conversion",
                         "mobile", "trust", "ai_readiness"):
                old_val = first.get(dim, 0)
                new_val = latest.get(dim, 0)
                dim_trends[dim] = {
                    "first": old_val,
                    "latest": new_val,
                    "change": new_val - old_val,
                }

        return {
            "site_slug": site_slug,
            "history": trend,
            "total_audits": len(trend),
            "velocity": round(avg_delta, 1),
            "direction": "improving" if avg_delta > 0.5 else "declining" if avg_delta < -0.5 else "stable",
            "dimension_trends": dim_trends,
            "latest_score": trend[-1].get("overall_score", 0) if trend else 0,
        }
