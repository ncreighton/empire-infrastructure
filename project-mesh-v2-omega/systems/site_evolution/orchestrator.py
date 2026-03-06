"""
Site Evolution Engine — Master Orchestrator.
Runs the full AUDIT → PLAN → DESIGN → DEPLOY → VERIFY cycle.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class SiteEvolutionEngine:
    """Master coordinator for the site evolution system."""

    def __init__(self):
        from systems.site_evolution.auditor.site_auditor import SiteAuditor
        from systems.site_evolution.queue.enhancement_queue import EnhancementQueue
        from systems.site_evolution.designer.design_generator import DesignGenerator
        from systems.site_evolution.designer.css_engine import CSSEngine
        from systems.site_evolution.designer.page_layouts import PageLayouts
        from systems.site_evolution.components.component_factory import ComponentFactory
        from systems.site_evolution.deployer.wp_deployer import WPDeployer
        from systems.site_evolution.deployer.batch_deployer import BatchDeployer
        from systems.site_evolution.deployer.content_deployer import ContentDeployer
        from systems.site_evolution.seo.schema_generator import SchemaGenerator
        from systems.site_evolution.seo.meta_optimizer import MetaOptimizer
        from systems.site_evolution.seo.llmo_optimizer import LLMOOptimizer
        from systems.site_evolution.seo.search_analytics import SearchAnalytics
        from systems.site_evolution.performance.vitals_optimizer import VitalsOptimizer

        self.auditor = SiteAuditor()
        self.queue = EnhancementQueue()
        self.designer = DesignGenerator()
        self.css_engine = CSSEngine()
        self.layouts = PageLayouts()
        self.factory = ComponentFactory()
        self.deployer = WPDeployer()
        self.batch = BatchDeployer()
        self.content = ContentDeployer()
        self.schema = SchemaGenerator()
        self.meta = MetaOptimizer()
        self.llmo = LLMOOptimizer()
        self.search = SearchAnalytics()
        self.vitals = VitalsOptimizer()

    def evolve_site(self, site_slug: str, dry_run: bool = False) -> Dict:
        """Full enhancement cycle for one site.

        1. AUDIT: Run site auditor → 8-dimension score
        2. PLAN: Generate fix queue from audit gaps
        3. DESIGN: Generate design system + components
        4. DEPLOY: Push CSS, snippets, pages
        5. VERIFY: Re-audit to confirm improvements
        6. LOG: Record changes, scores, before/after
        """
        started = datetime.now()
        log.info("Starting evolution cycle for %s (dry_run=%s)", site_slug, dry_run)

        # Publish event
        self._publish_event("evolution.site_started", {
            "site": site_slug, "dry_run": dry_run
        })

        # 1. AUDIT
        log.info("[1/6] Auditing %s...", site_slug)
        audit_before = self.auditor.audit_site(site_slug)

        # 2. PLAN
        log.info("[2/6] Planning improvements for %s (score: %d)...",
                 site_slug, audit_before["overall_score"])
        items_added = self.queue.populate_from_audit(audit_before)

        # 3. DESIGN
        log.info("[3/6] Generating design system for %s...", site_slug)
        design_system = self.designer.generate_design_system(site_slug)
        full_css = self.css_engine.generate_full_stylesheet(design_system)

        # 4. DEPLOY
        if not dry_run:
            log.info("[4/6] Deploying to %s...", site_slug)

            # Deploy CSS framework
            try:
                self.deployer.deploy_custom_css(site_slug, full_css)
                log.info("  Deployed CSS framework (%d lines)", full_css.count("\n"))
            except Exception as e:
                log.error("  CSS deployment failed: %s", e)

            # Deploy schema markup
            try:
                schemas = self.schema.generate_site_schemas(site_slug)
                self.deployer.deploy_snippet(
                    site_slug, f"{site_slug[:4]}-schema-v1",
                    schemas, code_type="html", location="site_wide_header"
                )
                log.info("  Deployed schema markup")
            except Exception as e:
                log.error("  Schema deployment failed: %s", e)

            # Deploy performance snippets
            try:
                perf = self.vitals.generate_all_performance_snippets(site_slug)
                for name, code in perf.items():
                    code_type = "css" if name == "critical_css" else "php"
                    self.deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-perf-{name}-v1",
                        code, code_type=code_type, location="site_wide_header"
                    )
                log.info("  Deployed %d performance snippets", len(perf))
            except Exception as e:
                log.error("  Performance deployment failed: %s", e)

            # Deploy ToC JS
            try:
                toc = self.factory.generate_component(site_slug, "table_of_contents")
                if toc.get("js"):
                    self.deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-toc-v1",
                        f"<script>\n{toc['js']}\n</script>",
                        code_type="html", location="site_wide_footer"
                    )
                    log.info("  Deployed auto-ToC")
            except Exception as e:
                log.error("  ToC deployment failed: %s", e)

            # Deploy cookie consent (legally required for EU)
            try:
                cookie = self.factory.generate_component(site_slug, "cookie_consent")
                if cookie.get("html"):
                    from systems.site_evolution.components.snippet_builder import SnippetBuilder
                    builder = SnippetBuilder()
                    for snippet in builder.component_to_snippets(site_slug, "cookie_consent", cookie):
                        self.deployer.deploy_snippet(
                            site_slug, snippet["title"],
                            snippet["code"], code_type=snippet["code_type"],
                            location=snippet.get("location", "site_wide_footer")
                        )
                    log.info("  Deployed cookie consent")
            except Exception as e:
                log.error("  Cookie consent deployment failed: %s", e)

            # Deploy reading progress bar + back to top
            for comp_type in ("reading_progress_bar", "back_to_top"):
                try:
                    comp = self.factory.generate_component(site_slug, comp_type)
                    from systems.site_evolution.components.snippet_builder import SnippetBuilder
                    builder = SnippetBuilder()
                    for snippet in builder.component_to_snippets(site_slug, comp_type, comp):
                        self.deployer.deploy_snippet(
                            site_slug, snippet["title"],
                            snippet["code"], code_type=snippet["code_type"],
                            location=snippet.get("location", "site_wide_footer")
                        )
                    log.info("  Deployed %s", comp_type)
                except Exception as e:
                    log.error("  %s deployment failed: %s", comp_type, e)

            # Execute top queue items
            try:
                self.queue.execute_batch(site_slug, max_items=3)
            except Exception as e:
                log.error("  Queue execution failed: %s", e)
        else:
            log.info("[4/6] DRY RUN — skipping deployment")

        # 5. VERIFY
        log.info("[5/6] Verifying %s...", site_slug)
        audit_after = self.auditor.audit_site(site_slug) if not dry_run else audit_before

        # 6. LOG
        elapsed = (datetime.now() - started).total_seconds()
        log.info("[6/6] Evolution complete for %s in %.1fs", site_slug, elapsed)

        improvement = audit_after["overall_score"] - audit_before["overall_score"]

        self._publish_event("evolution.site_completed", {
            "site": site_slug,
            "score_before": audit_before["overall_score"],
            "score_after": audit_after["overall_score"],
            "improvement": improvement,
            "elapsed_seconds": elapsed,
        })

        return {
            "site_slug": site_slug,
            "dry_run": dry_run,
            "score_before": audit_before["overall_score"],
            "score_after": audit_after["overall_score"],
            "improvement": improvement,
            "scores_before": audit_before.get("scores", {}),
            "scores_after": audit_after.get("scores", {}),
            "queue_items_added": items_added,
            "css_lines": full_css.count("\n"),
            "design_lane": design_system.style_lane,
            "elapsed_seconds": elapsed,
        }

    def evolve_all(self, dry_run: bool = False) -> Dict:
        """Run enhancement cycle for all 14 sites (lowest scores first)."""
        from systems.site_evolution.utils import get_all_site_slugs
        sites = get_all_site_slugs()

        # Get current scores to prioritize lowest
        from systems.site_evolution import codex
        audits = codex.get_all_latest_audits()
        scored = {a["site_slug"]: a["overall_score"] for a in audits}

        # Sort: unaudited first, then lowest score first
        sites.sort(key=lambda s: scored.get(s, -1))

        results = {}
        for slug in sites:
            try:
                results[slug] = self.evolve_site(slug, dry_run=dry_run)
            except Exception as e:
                log.error("Evolution failed for %s: %s", slug, e)
                results[slug] = {"error": str(e)}

        return {
            "sites_processed": len(results),
            "dry_run": dry_run,
            "results": results,
        }

    def evolve_component(self, site_slug: str, component_type: str,
                         dry_run: bool = False) -> Dict:
        """Generate and deploy one specific component."""
        comp = self.factory.generate_component(site_slug, component_type)

        if dry_run:
            return {
                "site": site_slug,
                "component": component_type,
                "dry_run": True,
                "has_html": bool(comp.get("html")),
                "has_css": bool(comp.get("css")),
                "has_js": bool(comp.get("js")),
                "preview": {
                    k: v[:300] if isinstance(v, str) else v
                    for k, v in comp.items()
                },
            }

        # Deploy
        if comp.get("css"):
            self.deployer.deploy_snippet(
                site_slug, f"{site_slug[:4]}-{component_type}-css-v1",
                comp["css"], code_type="css"
            )
        if comp.get("html"):
            location = comp.get("location", "site_wide_footer")
            self.deployer.deploy_snippet(
                site_slug, f"{site_slug[:4]}-{component_type}-html-v1",
                comp["html"], code_type="html", location=location
            )
        if comp.get("js"):
            self.deployer.deploy_snippet(
                site_slug, f"{site_slug[:4]}-{component_type}-js-v1",
                f"<script>\n{comp['js']}\n</script>",
                code_type="html", location="site_wide_footer"
            )

        self._publish_event("evolution.component_deployed", {
            "site": site_slug, "component": component_type
        })

        # Save to codex
        from systems.site_evolution import codex
        codex.save_component(
            site_slug, component_type,
            comp.get("html", ""), comp.get("css", ""), comp.get("js", ""),
            snippet_name=comp.get("snippet_name", ""),
        )

        return {
            "site": site_slug,
            "component": component_type,
            "status": "deployed",
        }

    def get_site_status(self, site_slug: str) -> Dict:
        """Current scores, pending queue, deployment history."""
        from systems.site_evolution import codex

        audit = codex.get_latest_audit(site_slug)
        queue = codex.get_queue(site_slug, limit=10)
        deployments = codex.get_deployments(site_slug, limit=10)
        design = codex.get_design_system(site_slug)

        return {
            "site_slug": site_slug,
            "latest_audit": audit,
            "pending_queue": queue,
            "recent_deployments": deployments,
            "has_design_system": design is not None,
            "design_lane": design.get("style_lane") if design else None,
        }

    def get_empire_status(self) -> Dict:
        """All 14 sites ranked by overall score."""
        from systems.site_evolution import codex

        summary = codex.get_empire_summary()
        stats = codex.get_stats()
        queues = codex.get_all_queues()
        activity = codex.get_recent_activity(limit=10)

        return {
            "sites": summary.get("sites", []),
            "total_sites": summary.get("total_sites", 0),
            "avg_score": summary.get("avg_score", 0),
            "dimension_averages": summary.get("dimension_averages", {}),
            "weakest_dimension": summary.get("weakest_dimension", ""),
            "pending_queues": {slug: len(items) for slug, items in queues.items()},
            "total_pending": sum(len(items) for items in queues.values()),
            "recent_activity": activity,
            "system_stats": stats,
        }

    def get_audit_trend(self, site_slug: str, limit: int = 10) -> Dict:
        """Get historical audit trends for a site."""
        from systems.site_evolution import codex
        trend = codex.get_audit_trend(site_slug, limit=limit)
        return {
            "site_slug": site_slug,
            "history": trend,
            "total_audits": len(trend),
        }

    # -- Multi-pass evolution (v2) --

    def evolve_site_v2(self, site_slug: str, dry_run: bool = False) -> Dict:
        """Multi-pass evolution in 6 sequential waves.

        Wave 1: Foundation — security, broken links, alt text
        Wave 2: Content — internal links, freshness
        Wave 3: SEO — schema, canonicals, meta
        Wave 4: Performance — vitals, images
        Wave 5: Conversion — email capture, CTAs
        Wave 6: Polish — cookie consent, dark mode, progress bar
        """
        started = datetime.now()
        log.info("Starting v2 multi-pass evolution for %s (dry_run=%s)", site_slug, dry_run)

        self._publish_event("evolution.v2_started", {
            "site": site_slug, "dry_run": dry_run
        })

        # Snapshot before changes
        snapshot_id = None
        if not dry_run:
            try:
                snapshot_id = self.snapshot_site(site_slug)
                log.info("Created snapshot %s for rollback", snapshot_id)
            except Exception as e:
                log.warning("Snapshot failed: %s", e)

        # Pre-audit
        audit_before = self.auditor.audit_site(site_slug)
        log.info("Pre-audit score: %d", audit_before["overall_score"])

        wave_results = {}

        # Wave 1: Foundation
        log.info("[Wave 1/6] Foundation — security, broken links, alt text")
        w1 = {"deployed": [], "errors": []}
        if not dry_run:
            # Security headers
            try:
                from systems.site_evolution.performance.security_hardener import SecurityHardener
                hardener = SecurityHardener()
                snippets = hardener.generate_all_security_snippets(site_slug)
                for name, code in snippets.items():
                    self.deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-sec-{name}-v1",
                        code, code_type="php", location="everywhere"
                    )
                    w1["deployed"].append(f"security-{name}")
            except Exception as e:
                w1["errors"].append(f"security: {e}")

            # Fix broken links
            try:
                from systems.site_evolution.auditor.broken_link_monitor import BrokenLinkMonitor
                blm = BrokenLinkMonitor()
                broken = blm.detect_broken_internal(site_slug)
                if broken:
                    from systems.site_evolution.utils import get_site_domain
                    domain = get_site_domain(site_slug)
                    redirects = [{"from_url": b["url"].split(domain)[-1], "to_url": "/"} for b in broken[:20]]
                    snippet = blm.generate_redirect_snippet(redirects)
                    self.deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-redirects-v1",
                        snippet, code_type="php", location="everywhere"
                    )
                    w1["deployed"].append("redirects")
            except Exception as e:
                w1["errors"].append(f"broken_links: {e}")

            # Fix alt text
            try:
                from systems.site_evolution.performance.image_optimizer import ImageOptimizer
                img_opt = ImageOptimizer()
                img_opt.fix_missing_alt_text(site_slug, dry_run=False)
                w1["deployed"].append("alt_text_fix")
            except Exception as e:
                w1["errors"].append(f"alt_text: {e}")
        wave_results["foundation"] = w1

        # Wave 2: Content
        log.info("[Wave 2/6] Content — internal links, freshness")
        w2 = {"deployed": [], "errors": []}
        if not dry_run:
            # Internal linker
            try:
                from systems.site_evolution.seo.internal_linker import InternalLinker
                linker = InternalLinker()
                snippet = linker.generate_link_injection_snippet(site_slug)
                self.deployer.deploy_snippet(
                    site_slug, f"{site_slug[:4]}-autolinks-v1",
                    snippet, code_type="php", location="everywhere"
                )
                w2["deployed"].append("auto_linker")
            except Exception as e:
                w2["errors"].append(f"internal_links: {e}")

            # Freshness "Last Updated" display
            try:
                from systems.site_evolution.auditor.freshness_tracker import FreshnessTracker
                tracker = FreshnessTracker()
                snippet = tracker.generate_update_date_snippet(site_slug)
                self.deployer.deploy_snippet(
                    site_slug, f"{site_slug[:4]}-lastmod-v1",
                    snippet, code_type="php", location="everywhere"
                )
                w2["deployed"].append("last_updated_display")
            except Exception as e:
                w2["errors"].append(f"freshness: {e}")
        wave_results["content"] = w2

        # Wave 3: SEO
        log.info("[Wave 3/6] SEO — schema, canonicals, meta")
        w3 = {"deployed": [], "errors": []}
        if not dry_run:
            # Schema markup
            try:
                schemas = self.schema.generate_site_schemas(site_slug)
                self.deployer.deploy_snippet(
                    site_slug, f"{site_slug[:4]}-schema-v1",
                    schemas, code_type="html", location="site_wide_header"
                )
                w3["deployed"].append("schema")
            except Exception as e:
                w3["errors"].append(f"schema: {e}")

            # Canonical enforcement
            try:
                from systems.site_evolution.seo.canonical_manager import CanonicalManager
                cm = CanonicalManager()
                snippet = cm.generate_canonical_snippet(site_slug)
                self.deployer.deploy_snippet(
                    site_slug, f"{site_slug[:4]}-canonical-v1",
                    snippet, code_type="php", location="everywhere"
                )
                w3["deployed"].append("canonical")
            except Exception as e:
                w3["errors"].append(f"canonical: {e}")

            # Affiliate compliance
            try:
                from systems.site_evolution.seo.affiliate_manager import AffiliateLinkManager
                am = AffiliateLinkManager()
                aff_snippets = am.generate_all_affiliate_snippets(site_slug)
                for name, code in aff_snippets.items():
                    self.deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-aff-{name}-v1",
                        code, code_type="php", location="everywhere"
                    )
                    w3["deployed"].append(f"affiliate-{name}")
            except Exception as e:
                w3["errors"].append(f"affiliate: {e}")

            # LLMO optimization (AI discoverability)
            try:
                llmo_snippets = self.llmo.generate_llmo_snippets(site_slug)
                if isinstance(llmo_snippets, dict):
                    for name, code in llmo_snippets.items():
                        self.deployer.deploy_snippet(
                            site_slug, f"{site_slug[:4]}-llmo-{name}-v1",
                            code, code_type="php", location="site_wide_header"
                        )
                        w3["deployed"].append(f"llmo-{name}")
            except Exception as e:
                log.debug("LLMO optimization skipped: %s", e)
        wave_results["seo"] = w3

        # Wave 4: Performance
        log.info("[Wave 4/6] Performance — vitals, images")
        w4 = {"deployed": [], "errors": []}
        if not dry_run:
            # Performance snippets
            try:
                perf = self.vitals.generate_all_performance_snippets(site_slug)
                for name, code in perf.items():
                    code_type = "css" if name == "critical_css" else "php"
                    self.deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-perf-{name}-v1",
                        code, code_type=code_type, location="site_wide_header"
                    )
                    w4["deployed"].append(f"perf-{name}")
            except Exception as e:
                w4["errors"].append(f"vitals: {e}")

            # Image optimization snippets
            try:
                from systems.site_evolution.performance.image_optimizer import ImageOptimizer
                img = ImageOptimizer()
                webp = img.generate_webp_snippet()
                self.deployer.deploy_snippet(
                    site_slug, f"{site_slug[:4]}-webp-v1",
                    webp, code_type="php", location="everywhere"
                )
                w4["deployed"].append("webp")
            except Exception as e:
                w4["errors"].append(f"images: {e}")

            # CSS framework
            try:
                ds = self.designer.generate_design_system(site_slug)
                css = self.css_engine.generate_full_stylesheet(ds)
                self.deployer.deploy_custom_css(site_slug, css)
                w4["deployed"].append("css_framework")
            except Exception as e:
                w4["errors"].append(f"css: {e}")
        wave_results["performance"] = w4

        # Wave 5: Conversion
        log.info("[Wave 5/6] Conversion — email capture, CTAs")
        w5 = {"deployed": [], "errors": []}
        if not dry_run:
            # Email capture
            try:
                from systems.site_evolution.components.email_capture import EmailCaptureSystem
                from systems.site_evolution.components.snippet_builder import SnippetBuilder
                ecs = EmailCaptureSystem()
                builder = SnippetBuilder()
                comp = ecs.generate_capture_snippet(site_slug, "scroll")
                for snippet in builder.component_to_snippets(site_slug, "email_capture", comp):
                    self.deployer.deploy_snippet(
                        site_slug, snippet["title"],
                        snippet["code"], code_type=snippet["code_type"],
                        location=snippet.get("location", "site_wide_footer")
                    )
                w5["deployed"].append("email_capture")
            except Exception as e:
                w5["errors"].append(f"email_capture: {e}")

            # AdSense optimization
            try:
                from systems.site_evolution.monetization.adsense_optimizer import AdSenseOptimizer
                adsense = AdSenseOptimizer()
                ad_snippet = adsense.generate_optimal_ad_snippet(site_slug)
                self.deployer.deploy_snippet(
                    site_slug, f"{site_slug[:4]}-ads-v1",
                    ad_snippet, code_type="php", location="everywhere"
                )
                w5["deployed"].append("adsense")
            except Exception as e:
                w5["errors"].append(f"adsense: {e}")
        wave_results["conversion"] = w5

        # Wave 6: Polish
        log.info("[Wave 6/6] Polish — cookie consent, dark mode, progress bar")
        w6 = {"deployed": [], "errors": []}
        if not dry_run:
            from systems.site_evolution.components.snippet_builder import SnippetBuilder
            builder = SnippetBuilder()

            for comp_type in ("cookie_consent", "reading_progress_bar", "back_to_top"):
                try:
                    comp = self.factory.generate_component(site_slug, comp_type)
                    for snippet in builder.component_to_snippets(site_slug, comp_type, comp):
                        self.deployer.deploy_snippet(
                            site_slug, snippet["title"],
                            snippet["code"], code_type=snippet["code_type"],
                            location=snippet.get("location", "site_wide_footer")
                        )
                    w6["deployed"].append(comp_type)
                except Exception as e:
                    w6["errors"].append(f"{comp_type}: {e}")
        wave_results["polish"] = w6

        # Post-audit
        audit_after = self.auditor.audit_site(site_slug) if not dry_run else audit_before
        elapsed = (datetime.now() - started).total_seconds()

        improvement = audit_after["overall_score"] - audit_before["overall_score"]
        total_deployed = sum(len(w.get("deployed", [])) for w in wave_results.values())
        total_errors = sum(len(w.get("errors", [])) for w in wave_results.values())

        log.info("v2 evolution complete: %d deployed, %d errors, +%d score in %.1fs",
                 total_deployed, total_errors, improvement, elapsed)

        self._publish_event("evolution.v2_completed", {
            "site": site_slug,
            "score_before": audit_before["overall_score"],
            "score_after": audit_after["overall_score"],
            "improvement": improvement,
            "total_deployed": total_deployed,
            "elapsed_seconds": elapsed,
        })

        return {
            "site_slug": site_slug,
            "dry_run": dry_run,
            "score_before": audit_before["overall_score"],
            "score_after": audit_after["overall_score"],
            "improvement": improvement,
            "scores_before": audit_before.get("scores", {}),
            "scores_after": audit_after.get("scores", {}),
            "waves": wave_results,
            "total_deployed": total_deployed,
            "total_errors": total_errors,
            "snapshot_id": snapshot_id,
            "elapsed_seconds": elapsed,
        }

    def snapshot_site(self, site_slug: str) -> Optional[int]:
        """Full snapshot of all snippets for rollback.

        Returns snapshot ID.
        """
        from systems.site_evolution import codex
        from systems.site_evolution.deployer.wp_deployer import WPDeployer

        deployer = WPDeployer()
        try:
            snippets = deployer.get_existing_snippets(site_slug)
        except Exception:
            snippets = []

        # Also grab deployment history from codex
        deployments = codex.get_deployments(site_slug, limit=200)

        snapshot_data = {
            "site_slug": site_slug,
            "taken_at": datetime.now().isoformat(),
            "snippet_count": len(snippets),
            "snippets": snippets,
            "deployment_count": len(deployments),
            "deployments": deployments,
        }

        snapshot_id = codex.save_snapshot(site_slug, json.dumps(snapshot_data))

        self._publish_event("evolution.snapshot_created", {
            "site": site_slug, "snapshot_id": snapshot_id,
            "snippet_count": len(snippets),
        })

        return snapshot_id

    def rollback_to_snapshot(self, site_slug: str, snapshot_id: int) -> Dict:
        """Restore all snippets from a snapshot.

        WARNING: This deactivates current snippets and restores snapshot state.
        """
        from systems.site_evolution import codex

        snapshot = codex.get_snapshot(snapshot_id)
        if not snapshot:
            return {"error": f"Snapshot {snapshot_id} not found"}

        data = json.loads(snapshot["snippet_data"])
        if data.get("site_slug") != site_slug:
            return {"error": "Snapshot does not match site_slug"}

        restored = 0
        errors = []

        # Re-deploy each snippet from snapshot
        for snippet in data.get("snippets", []):
            try:
                name = snippet.get("title", snippet.get("name", ""))
                code = snippet.get("code", "")
                code_type = snippet.get("code_type", "php")
                if name and code:
                    self.deployer.deploy_snippet(
                        site_slug, name, code,
                        code_type=code_type,
                        location=snippet.get("location", "everywhere")
                    )
                    restored += 1
            except Exception as e:
                errors.append(f"{name}: {e}")

        self._publish_event("evolution.rollback_completed", {
            "site": site_slug, "snapshot_id": snapshot_id,
            "restored": restored,
        })

        return {
            "site_slug": site_slug,
            "snapshot_id": snapshot_id,
            "snippets_restored": restored,
            "errors": errors,
        }

    def generate_evolution_report(self, site_slug: str) -> str:
        """Generate HTML report with before/after scores and deployment history."""
        from systems.site_evolution import codex
        from systems.site_evolution.utils import get_site_brand_name

        brand = get_site_brand_name(site_slug)
        audit = codex.get_latest_audit(site_slug)
        trend = codex.get_audit_trend(site_slug, limit=10)
        deployments = codex.get_deployments(site_slug, limit=30)
        queue = codex.get_queue(site_slug, limit=10)

        scores = audit.get("scores", {}) if audit else {}
        overall = audit.get("overall_score", 0) if audit else 0

        # Build score bars HTML
        score_bars = ""
        for dim in ("design", "seo", "performance", "content", "conversion",
                     "mobile", "trust", "ai_readiness"):
            val = scores.get(dim, 0)
            color = "#22c55e" if val >= 70 else "#eab308" if val >= 40 else "#ef4444"
            score_bars += f"""
            <div style="margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;font-size:13px">
                    <span>{dim.replace('_',' ').title()}</span><span>{val}/100</span>
                </div>
                <div style="background:#e5e7eb;border-radius:4px;height:8px">
                    <div style="background:{color};width:{val}%;height:8px;border-radius:4px"></div>
                </div>
            </div>"""

        # Build trend chart data
        trend_points = ""
        if trend:
            for i, t in enumerate(reversed(trend[-10:])):
                x = 30 + i * 50
                y = 180 - int(t.get("overall_score", 0) * 1.6)
                trend_points += f"{x},{y} "

        # Build deployment list
        deploy_rows = ""
        for d in deployments[:15]:
            deploy_rows += f"""
            <tr>
                <td style="padding:4px 8px;font-size:12px">{d.get('snippet_name','')}</td>
                <td style="padding:4px 8px;font-size:12px">{d.get('component_type','')}</td>
                <td style="padding:4px 8px;font-size:12px">{d.get('deployed_at','')[:10]}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>{brand} Evolution Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:800px;margin:0 auto;padding:20px;background:#f9fafb}}
h1{{color:#1f2937;border-bottom:2px solid #3b82f6;padding-bottom:8px}}
.card{{background:#fff;border-radius:8px;padding:16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.score-big{{font-size:48px;font-weight:bold;text-align:center;
    color:{"#22c55e" if overall >= 70 else "#eab308" if overall >= 40 else "#ef4444"}}}
table{{width:100%;border-collapse:collapse}}
th{{text-align:left;padding:4px 8px;font-size:12px;color:#6b7280;border-bottom:1px solid #e5e7eb}}
</style>
</head><body>
<h1>{brand} — Evolution Report</h1>
<p style="color:#6b7280">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="card">
<h2>Overall Score</h2>
<div class="score-big">{overall}/100</div>
</div>

<div class="card">
<h2>Dimension Scores</h2>
{score_bars}
</div>

<div class="card">
<h2>Score Trend</h2>
<svg width="100%" height="200" viewBox="0 0 550 200">
<polyline fill="none" stroke="#3b82f6" stroke-width="2" points="{trend_points}"/>
</svg>
</div>

<div class="card">
<h2>Recent Deployments ({len(deployments)})</h2>
<table>
<tr><th>Snippet</th><th>Type</th><th>Date</th></tr>
{deploy_rows}
</table>
</div>

<div class="card">
<h2>Pending Queue ({len(queue)} items)</h2>
<ul style="font-size:13px">
{''.join(f'<li>{q.get("description","")}</li>' for q in queue[:10])}
</ul>
</div>
</body></html>"""

    def _publish_event(self, event_type: str, data: Dict):
        try:
            from core.event_bus import publish
            publish(event_type, data, source="site_evolution")
        except Exception:
            pass
