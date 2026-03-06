"""
Site Evolution API — FastAPI router with 25+ endpoints.
Mounted at /api/evolution/ on the dashboard.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evolution", tags=["site-evolution"])


def _get_engine():
    from systems.site_evolution.orchestrator import SiteEvolutionEngine
    return SiteEvolutionEngine()


# -- Evolution --

@router.post("/evolve/{site_slug}")
async def evolve_site(site_slug: str, dry_run: bool = True):
    """Run full enhancement cycle for one site."""
    engine = _get_engine()
    return engine.evolve_site(site_slug, dry_run=dry_run)


@router.post("/evolve-all")
async def evolve_all(dry_run: bool = True):
    """Enhance all 14 sites."""
    engine = _get_engine()
    return engine.evolve_all(dry_run=dry_run)


# -- Audit --

@router.get("/audit/{site_slug}")
async def audit_site(site_slug: str):
    """Audit a site across 8 dimensions."""
    from systems.site_evolution.auditor.site_auditor import SiteAuditor
    auditor = SiteAuditor()
    return auditor.audit_site(site_slug)


@router.get("/audit-all")
async def audit_all():
    """Audit all 14 sites and return ranked list."""
    from systems.site_evolution.auditor.site_auditor import SiteAuditor
    auditor = SiteAuditor()
    return {"sites": auditor.audit_all_sites()}


# -- Deploy --

@router.post("/deploy/{site_slug}/{component}")
async def deploy_component(site_slug: str, component: str, dry_run: bool = True):
    """Generate and deploy one component to a site."""
    engine = _get_engine()
    return engine.evolve_component(site_slug, component, dry_run=dry_run)


@router.post("/deploy-css/{site_slug}")
async def deploy_css(site_slug: str, dry_run: bool = True):
    """Generate and deploy full CSS framework."""
    from systems.site_evolution.designer.design_generator import DesignGenerator
    from systems.site_evolution.designer.css_engine import CSSEngine
    from systems.site_evolution.deployer.wp_deployer import WPDeployer

    gen = DesignGenerator()
    ds = gen.generate_design_system(site_slug)
    engine = CSSEngine()
    css = engine.generate_full_stylesheet(ds)

    if dry_run:
        return {
            "site": site_slug, "dry_run": True,
            "css_lines": css.count("\n"),
            "lane": ds.style_lane,
            "preview": css[:1000],
        }

    deployer = WPDeployer()
    result = deployer.deploy_custom_css(site_slug, css)
    return {"site": site_slug, "status": "deployed", "css_lines": css.count("\n")}


@router.post("/deploy-seo/{site_slug}")
async def deploy_seo(site_slug: str, dry_run: bool = True):
    """Deploy schema + meta optimization."""
    from systems.site_evolution.seo.schema_generator import SchemaGenerator
    from systems.site_evolution.deployer.wp_deployer import WPDeployer

    gen = SchemaGenerator()
    schemas = gen.generate_site_schemas(site_slug)

    if dry_run:
        return {
            "site": site_slug, "dry_run": True,
            "schema_length": len(schemas),
            "preview": schemas[:1000],
        }

    deployer = WPDeployer()
    deployer.deploy_snippet(
        site_slug, f"{site_slug[:4]}-schema-v1",
        schemas, code_type="html", location="site_wide_header"
    )
    return {"site": site_slug, "status": "deployed"}


# -- Queue --

@router.get("/queue/{site_slug}")
async def get_queue(site_slug: str, limit: int = 20):
    """View enhancement queue for a site."""
    from systems.site_evolution.queue.enhancement_queue import EnhancementQueue
    queue = EnhancementQueue()
    items = queue.get_queue(site_slug, limit=limit)
    progress = queue.get_progress(site_slug)
    return {"queue": items, "progress": progress}


@router.post("/execute/{item_id}")
async def execute_item(item_id: int, dry_run: bool = True):
    """Execute one queue item."""
    from systems.site_evolution.queue.enhancement_queue import EnhancementQueue
    queue = EnhancementQueue()
    return queue.execute_item(item_id, dry_run=dry_run)


# -- Status --

@router.get("/status/{site_slug}")
async def get_site_status(site_slug: str):
    """Site status: scores, queue, deployment history."""
    engine = _get_engine()
    return engine.get_site_status(site_slug)


@router.get("/empire")
async def get_empire_status():
    """All 14 sites ranked by score."""
    engine = _get_engine()
    return engine.get_empire_status()


@router.get("/deployments/{site_slug}")
async def get_deployments(site_slug: str, limit: int = 50):
    """Deployment history for a site."""
    from systems.site_evolution import codex
    return {"deployments": codex.get_deployments(site_slug, limit=limit)}


@router.post("/rollback/{deployment_id}")
async def rollback_deployment(deployment_id: int):
    """Rollback a deployment."""
    from systems.site_evolution import codex
    from systems.site_evolution.deployer.wp_deployer import WPDeployer

    # Look up the deployment to get site_slug
    deployment = codex.get_deployment_by_id(deployment_id)
    if not deployment:
        return {"status": "failed", "error": f"Deployment {deployment_id} not found"}

    deployer = WPDeployer()
    success = deployer.rollback(deployment["site_slug"], deployment_id)
    return {"status": "rolled_back" if success else "failed"}


@router.get("/stats")
async def get_stats():
    """System stats."""
    from systems.site_evolution import codex
    return codex.get_stats()


@router.get("/trend/{site_slug}")
async def get_audit_trend(site_slug: str, limit: int = 10):
    """Get historical audit scores for a site."""
    engine = _get_engine()
    return engine.get_audit_trend(site_slug, limit=limit)


@router.get("/activity")
async def get_recent_activity(limit: int = 20):
    """Get recent deployment and queue activity across all sites."""
    from systems.site_evolution import codex
    return {"activity": codex.get_recent_activity(limit=limit)}


@router.get("/queue-all")
async def get_all_queues():
    """Get pending queue items grouped by site."""
    from systems.site_evolution import codex
    queues = codex.get_all_queues()
    return {
        "queues": queues,
        "total_pending": sum(len(items) for items in queues.values()),
        "sites_with_pending": len(queues),
    }


@router.get("/health")
async def health_check():
    """Health check for the site evolution system."""
    from systems.site_evolution import codex
    try:
        stats = codex.get_stats()
        return {
            "status": "healthy",
            "version": "2.0.0",
            "database": "connected",
            "stats": stats,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# -- Search Analytics --

@router.get("/analytics/{site_slug}")
async def get_search_analytics(site_slug: str):
    """Get combined GSC + Bing analytics."""
    from systems.site_evolution.seo.search_analytics import SearchAnalytics
    sa = SearchAnalytics()
    return sa.get_full_analytics(site_slug)


@router.get("/analytics/{site_slug}/top-queries")
async def get_top_queries(site_slug: str, days: int = 28, limit: int = 50):
    """Get top search queries from GSC."""
    from systems.site_evolution.seo.search_analytics import SearchAnalytics
    sa = SearchAnalytics()
    return {"queries": sa.gsc_get_top_queries(site_slug, days=days, limit=limit)}


@router.get("/analytics/{site_slug}/declining")
async def get_declining_pages(site_slug: str):
    """Get declining pages from GSC."""
    from systems.site_evolution.seo.search_analytics import SearchAnalytics
    sa = SearchAnalytics()
    return {"declining": sa.gsc_get_declining_pages(site_slug)}


@router.get("/analytics/{site_slug}/rising")
async def get_rising_keywords(site_slug: str):
    """Get rising keywords from GSC."""
    from systems.site_evolution.seo.search_analytics import SearchAnalytics
    sa = SearchAnalytics()
    return {"rising": sa.gsc_get_rising_keywords(site_slug)}


@router.post("/analytics/{site_slug}/inspect")
async def inspect_url(site_slug: str, url: str = Query(...)):
    """Inspect URL indexing status via GSC."""
    from systems.site_evolution.seo.search_analytics import SearchAnalytics
    sa = SearchAnalytics()
    return sa.gsc_inspect_url(site_slug, url)


@router.post("/analytics/{site_slug}/submit-sitemap")
async def submit_sitemaps(site_slug: str):
    """Submit sitemaps to both GSC and Bing."""
    from systems.site_evolution.seo.search_analytics import SearchAnalytics
    sa = SearchAnalytics()
    return {
        "gsc": sa.gsc_submit_sitemap(site_slug),
        "bing": sa.bing_submit_sitemap(site_slug),
    }


@router.post("/analytics/{site_slug}/submit-url")
async def submit_url(site_slug: str, url: str = Query(...)):
    """Submit URL to Bing for indexing."""
    from systems.site_evolution.seo.search_analytics import SearchAnalytics
    sa = SearchAnalytics()
    return sa.bing_submit_url(site_slug, url)


# -- Internal Links --

@router.get("/links/{site_slug}")
async def audit_internal_links(site_slug: str):
    """Audit internal linking structure."""
    from systems.site_evolution.seo.internal_linker import InternalLinker
    linker = InternalLinker()
    return {
        "audit": linker.audit_internal_links(site_slug),
        "orphans": linker.find_orphan_pages(site_slug),
        "equity": linker.get_link_equity_report(site_slug),
    }


@router.post("/links/{site_slug}/deploy")
async def deploy_internal_links(site_slug: str, dry_run: bool = True):
    """Deploy auto-linker snippet."""
    from systems.site_evolution.seo.internal_linker import InternalLinker
    from systems.site_evolution.deployer.wp_deployer import WPDeployer
    linker = InternalLinker()
    snippet = linker.generate_link_injection_snippet(site_slug)
    if dry_run:
        return {"site": site_slug, "dry_run": True, "snippet_length": len(snippet)}
    deployer = WPDeployer()
    deployer.deploy_snippet(site_slug, f"{site_slug[:4]}-autolinks-v1",
                            snippet, code_type="php", location="everywhere")
    return {"site": site_slug, "status": "deployed"}


# -- Images --

@router.get("/images/{site_slug}")
async def audit_images(site_slug: str, limit: int = 50):
    """Audit image optimization."""
    from systems.site_evolution.performance.image_optimizer import ImageOptimizer
    return ImageOptimizer().audit_images(site_slug, limit)


@router.post("/images/{site_slug}/fix-alt")
async def fix_image_alt_text(site_slug: str, dry_run: bool = True):
    """Fix missing alt text on media items."""
    from systems.site_evolution.performance.image_optimizer import ImageOptimizer
    return ImageOptimizer().fix_missing_alt_text(site_slug, dry_run)


# -- Security --

@router.get("/security/{site_slug}")
async def audit_security(site_slug: str):
    """Audit security headers and vulnerabilities."""
    from systems.site_evolution.performance.security_hardener import SecurityHardener
    return SecurityHardener().audit_security(site_slug)


@router.post("/security/{site_slug}/deploy")
async def deploy_security(site_slug: str, dry_run: bool = True):
    """Deploy all security hardening snippets."""
    from systems.site_evolution.performance.security_hardener import SecurityHardener
    from systems.site_evolution.deployer.wp_deployer import WPDeployer
    hardener = SecurityHardener()
    snippets = hardener.generate_all_security_snippets(site_slug)
    if dry_run:
        return {"site": site_slug, "dry_run": True, "snippets": list(snippets.keys())}
    deployer = WPDeployer()
    for name, code in snippets.items():
        deployer.deploy_snippet(site_slug, f"{site_slug[:4]}-sec-{name}-v1",
                                code, code_type="php", location="everywhere")
    return {"site": site_slug, "status": "deployed", "count": len(snippets)}


# -- Broken Links --

@router.get("/broken-links/{site_slug}")
async def audit_broken_links(site_slug: str, limit: int = 30):
    """Crawl and check all links for a site."""
    from systems.site_evolution.auditor.broken_link_monitor import BrokenLinkMonitor
    return BrokenLinkMonitor().crawl_links(site_slug, limit)


@router.post("/broken-links/{site_slug}/fix")
async def fix_broken_links(site_slug: str, dry_run: bool = True):
    """Generate and deploy redirect snippet for broken internal links."""
    from systems.site_evolution.auditor.broken_link_monitor import BrokenLinkMonitor
    from systems.site_evolution.deployer.wp_deployer import WPDeployer
    blm = BrokenLinkMonitor()
    broken = blm.detect_broken_internal(site_slug)
    if dry_run:
        return {"site": site_slug, "dry_run": True, "broken_count": len(broken)}
    if broken:
        from systems.site_evolution.utils import get_site_domain
        domain = get_site_domain(site_slug)
        redirects = [{"from_url": b["url"].split(domain)[-1], "to_url": "/"} for b in broken[:20]]
        snippet = blm.generate_redirect_snippet(redirects)
        deployer = WPDeployer()
        deployer.deploy_snippet(site_slug, f"{site_slug[:4]}-redirects-v1",
                                snippet, code_type="php", location="everywhere")
    return {"site": site_slug, "status": "deployed", "fixed": len(broken)}


# -- Design Preview --

@router.get("/design/{site_slug}")
async def get_design_system(site_slug: str):
    """Get or generate design system for a site."""
    from systems.site_evolution.designer.design_generator import DesignGenerator
    gen = DesignGenerator()
    ds = gen.generate_design_system(site_slug)
    return {
        "site": site_slug,
        "lane": ds.style_lane,
        "css_variables": ds.css_variables,
        "typography": ds.typography_stack,
        "colors": ds.color_palette,
        "supports_dark_mode": ds.supports_dark_mode,
    }


# -- Multi-pass Evolution (v2) --

@router.post("/evolve-v2/{site_slug}")
async def evolve_site_v2(site_slug: str, dry_run: bool = True):
    """Run 6-wave multi-pass evolution for a site."""
    engine = _get_engine()
    return engine.evolve_site_v2(site_slug, dry_run=dry_run)


@router.get("/report/{site_slug}")
async def get_evolution_report(site_slug: str):
    """Generate HTML evolution report."""
    engine = _get_engine()
    html = engine.generate_evolution_report(site_slug)
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


# -- Snapshots --

@router.post("/snapshot/{site_slug}")
async def create_snapshot(site_slug: str):
    """Create a full snapshot of all site snippets for rollback."""
    engine = _get_engine()
    snapshot_id = engine.snapshot_site(site_slug)
    return {"site": site_slug, "snapshot_id": snapshot_id}


@router.get("/snapshots/{site_slug}")
async def list_snapshots(site_slug: str, limit: int = 10):
    """List available snapshots for a site."""
    from systems.site_evolution import codex
    return {"snapshots": codex.get_snapshots(site_slug, limit=limit)}


@router.post("/rollback-snapshot/{site_slug}/{snapshot_id}")
async def rollback_to_snapshot(site_slug: str, snapshot_id: int):
    """Rollback site to a previous snapshot."""
    engine = _get_engine()
    return engine.rollback_to_snapshot(site_slug, snapshot_id)


# -- Content Gaps --

@router.get("/content-gaps/{site_slug}")
async def analyze_content_gaps(site_slug: str):
    """Analyze keyword coverage and content gaps."""
    from systems.site_evolution.seo.content_gap import ContentGapAnalyzer
    analyzer = ContentGapAnalyzer()
    return {
        "coverage": analyzer.analyze_keyword_coverage(site_slug),
        "thin_content": analyzer.analyze_thin_content(site_slug),
    }


@router.get("/content-gaps/{site_slug}/suggestions")
async def get_article_suggestions(site_slug: str, limit: int = 10):
    """Get new article suggestions based on content gaps."""
    from systems.site_evolution.seo.content_gap import ContentGapAnalyzer
    analyzer = ContentGapAnalyzer()
    return {
        "suggestions": analyzer.suggest_new_articles(site_slug, max_suggestions=limit),
        "calendar": analyzer.get_content_calendar(site_slug),
    }


# -- Canonicals --

@router.get("/canonicals/{site_slug}")
async def audit_canonicals(site_slug: str):
    """Audit canonical tags and detect duplicate content."""
    from systems.site_evolution.seo.canonical_manager import CanonicalManager
    cm = CanonicalManager()
    return {
        "duplicates": cm.detect_duplicates(site_slug),
        "canonical_audit": cm.audit_canonicals(site_slug),
        "redirect_chains": cm.detect_redirect_chains(site_slug),
    }


@router.post("/canonicals/{site_slug}/deploy")
async def deploy_canonicals(site_slug: str, dry_run: bool = True):
    """Deploy canonical enforcement snippet."""
    from systems.site_evolution.seo.canonical_manager import CanonicalManager
    from systems.site_evolution.deployer.wp_deployer import WPDeployer
    cm = CanonicalManager()
    snippet = cm.generate_canonical_snippet(site_slug)
    if dry_run:
        return {"site": site_slug, "dry_run": True, "snippet_length": len(snippet)}
    deployer = WPDeployer()
    deployer.deploy_snippet(site_slug, f"{site_slug[:4]}-canonical-v1",
                            snippet, code_type="php", location="everywhere")
    return {"site": site_slug, "status": "deployed"}


# -- Affiliates --

@router.get("/affiliates/{site_slug}")
async def audit_affiliates(site_slug: str):
    """Audit affiliate link compliance."""
    from systems.site_evolution.seo.affiliate_manager import AffiliateLinkManager
    am = AffiliateLinkManager()
    return {
        "audit": am.audit_affiliate_links(site_slug),
        "health_score": am.get_affiliate_health_score(site_slug),
    }


@router.post("/affiliates/{site_slug}/deploy")
async def deploy_affiliate_compliance(site_slug: str, dry_run: bool = True):
    """Deploy affiliate compliance snippets (disclosure, nofollow, tracking)."""
    from systems.site_evolution.seo.affiliate_manager import AffiliateLinkManager
    from systems.site_evolution.deployer.wp_deployer import WPDeployer
    am = AffiliateLinkManager()
    snippets = am.generate_all_affiliate_snippets(site_slug)
    if dry_run:
        return {"site": site_slug, "dry_run": True, "snippets": list(snippets.keys())}
    deployer = WPDeployer()
    for name, code in snippets.items():
        deployer.deploy_snippet(site_slug, f"{site_slug[:4]}-aff-{name}-v1",
                                code, code_type="php", location="everywhere")
    return {"site": site_slug, "status": "deployed", "count": len(snippets)}


# -- AdSense --

@router.get("/adsense/{site_slug}")
async def audit_adsense(site_slug: str):
    """Audit ad placement and density."""
    from systems.site_evolution.monetization.adsense_optimizer import AdSenseOptimizer
    return AdSenseOptimizer().audit_ad_placement(site_slug)


@router.post("/adsense/{site_slug}/deploy")
async def deploy_adsense(site_slug: str, dry_run: bool = True):
    """Deploy optimized ad placement snippets."""
    from systems.site_evolution.monetization.adsense_optimizer import AdSenseOptimizer
    from systems.site_evolution.deployer.wp_deployer import WPDeployer
    adsense = AdSenseOptimizer()
    snippet = adsense.generate_optimal_ad_snippet(site_slug)
    if dry_run:
        return {"site": site_slug, "dry_run": True, "snippet_length": len(snippet)}
    deployer = WPDeployer()
    deployer.deploy_snippet(site_slug, f"{site_slug[:4]}-ads-v1",
                            snippet, code_type="php", location="everywhere")
    return {"site": site_slug, "status": "deployed"}


# -- Freshness --

@router.get("/freshness/{site_slug}")
async def audit_freshness(site_slug: str):
    """Audit content freshness across all posts."""
    from systems.site_evolution.auditor.freshness_tracker import FreshnessTracker
    tracker = FreshnessTracker()
    return tracker.audit_freshness(site_slug)


@router.get("/freshness/{site_slug}/priority")
async def get_update_priority(site_slug: str):
    """Get priority-ranked stale content update list."""
    from systems.site_evolution.auditor.freshness_tracker import FreshnessTracker
    tracker = FreshnessTracker()
    return {
        "priority": tracker.get_update_priority(site_slug),
        "calendar": tracker.get_seasonal_calendar(site_slug),
    }


# -- Uptime Monitoring --

@router.get("/uptime")
async def check_all_uptime():
    """Parallel ping all sites — response times, status, SSL."""
    from systems.site_evolution.monitoring.uptime_monitor import UptimeMonitor
    return UptimeMonitor().check_all_sites()


@router.get("/uptime/{site_slug}")
async def check_site_uptime(site_slug: str):
    """Check single site uptime, response time, SSL."""
    from systems.site_evolution.monitoring.uptime_monitor import UptimeMonitor
    return UptimeMonitor().check_site(site_slug)


@router.get("/uptime/{site_slug}/history")
async def get_uptime_history(site_slug: str, limit: int = 50):
    """Get historical response times for a site."""
    from systems.site_evolution.monitoring.uptime_monitor import UptimeMonitor
    return {"history": UptimeMonitor().get_response_time_history(site_slug, limit)}


@router.get("/ssl-warnings")
async def check_ssl_certificates():
    """Check SSL certificates — flag expiring within 30 days."""
    from systems.site_evolution.monitoring.uptime_monitor import UptimeMonitor
    return {"warnings": UptimeMonitor().check_ssl_certificates()}


# -- Content Quality Scoring --

@router.get("/content-score/{site_slug}/{post_id}")
async def score_post(site_slug: str, post_id: int):
    """Comprehensive quality score for a single post."""
    from systems.site_evolution.auditor.content_scorer import ContentQualityScorer
    return ContentQualityScorer().score_post(site_slug, post_id)


@router.get("/content-score/{site_slug}")
async def score_all_posts(site_slug: str, limit: int = 50):
    """Batch score all posts for a site."""
    from systems.site_evolution.auditor.content_scorer import ContentQualityScorer
    return ContentQualityScorer().score_all_posts(site_slug, limit)


# -- Accessibility --

@router.get("/accessibility/{site_slug}")
async def audit_accessibility(site_slug: str):
    """WCAG 2.1 AA accessibility audit."""
    from systems.site_evolution.auditor.site_auditor import SiteAuditor
    auditor = SiteAuditor()
    config = load_site_config(site_slug)
    return auditor._audit_accessibility(site_slug, config)


# -- Score Trends --

@router.get("/score-trend/{site_slug}")
async def get_score_trend(site_slug: str, limit: int = 10):
    """Get score trend with velocity and per-dimension changes."""
    from systems.site_evolution.auditor.site_auditor import SiteAuditor
    auditor = SiteAuditor()
    return auditor.get_score_trend(site_slug, limit=limit)


def load_site_config(site_slug: str):
    """Helper to load site config for endpoints that need it."""
    from systems.site_evolution.utils import load_site_config as _load
    return _load(site_slug)
