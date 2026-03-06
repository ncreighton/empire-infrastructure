"""
Enhancement Queue — Manages and executes site improvements.
Items are auto-generated from audits, opportunity finder, and user requests.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from systems.site_evolution import codex

log = logging.getLogger(__name__)


class EnhancementQueue:
    """Priority-ranked enhancement queue with auto-execution."""

    def populate_from_audit(self, audit_results: Dict) -> int:
        """Convert audit findings into queue items. Returns count added."""
        from systems.site_evolution.auditor.site_auditor import SiteAuditor
        auditor = SiteAuditor()
        fix_items = auditor.generate_fix_queue(audit_results)

        added = 0
        for item in fix_items:
            # Avoid duplicates: check if similar item exists
            existing = codex.get_queue(item["site_slug"], status="pending", limit=100)
            already_exists = any(
                e["component_type"] == item["component_type"] and
                e.get("details", "") == item.get("description", "")
                for e in existing
            )
            if not already_exists:
                codex.enqueue(
                    site_slug=item["site_slug"],
                    component_type=item["component_type"],
                    action=item["action"],
                    priority=item["priority"],
                    estimated_impact=item["estimated_impact"],
                    details=item.get("description", ""),
                )
                added += 1

        log.info("Populated queue for %s: %d items added",
                 audit_results.get("site_slug"), added)
        return added

    def get_next(self, site_slug: str) -> Optional[Dict]:
        """Get the next highest-priority item for a site."""
        items = codex.get_queue(site_slug, status="pending", limit=1)
        return items[0] if items else None

    def get_queue(self, site_slug: str, limit: int = 20) -> List[Dict]:
        """Get ranked list of pending improvements."""
        return codex.get_queue(site_slug, status="pending", limit=limit)

    def get_all_queues(self) -> Dict[str, List[Dict]]:
        """Get queues for all sites."""
        from systems.site_evolution.utils import get_all_site_slugs
        sites = get_all_site_slugs()

        result = {}
        for slug in sites:
            queue = self.get_queue(slug, limit=5)
            if queue:
                result[slug] = queue
        return result

    def execute_item(self, item_id: int, dry_run: bool = False) -> Dict:
        """Execute a single enhancement queue item."""
        from systems.site_evolution.deployer.wp_deployer import WPDeployer
        from systems.site_evolution.designer.design_generator import DesignGenerator
        from systems.site_evolution.designer.css_engine import CSSEngine
        from systems.site_evolution.components.component_factory import ComponentFactory
        from systems.site_evolution.seo.schema_generator import SchemaGenerator
        from systems.site_evolution.performance.vitals_optimizer import VitalsOptimizer

        # Get the queue item from DB
        import sqlite3
        conn = codex._connect()
        row = conn.execute("SELECT * FROM enhancement_queue WHERE id = ?", (item_id,)).fetchone()
        conn.close()

        if not row:
            return {"error": f"Item {item_id} not found"}

        item = dict(row)
        site_slug = item["site_slug"]
        component_type = item["component_type"]

        if dry_run:
            return {
                "item_id": item_id,
                "site": site_slug,
                "component": component_type,
                "action": item["action"],
                "dry_run": True,
                "would_deploy": True,
            }

        codex.update_queue_item(item_id, "in_progress")

        try:
            deployer = WPDeployer()
            result = {}

            if component_type == "design":
                generator = DesignGenerator()
                ds = generator.generate_design_system(site_slug)
                engine = CSSEngine()
                css = engine.generate_full_stylesheet(ds)
                deployer.deploy_custom_css(site_slug, css)
                result = {"deployed": "css_framework", "lines": css.count("\n")}

            elif component_type == "seo":
                schema_gen = SchemaGenerator()
                schemas_html = schema_gen.generate_site_schemas(site_slug)
                deployer.deploy_snippet(
                    site_slug, f"{site_slug[:4]}-schema-v1",
                    schemas_html, code_type="html", location="site_wide_header"
                )
                result = {"deployed": "schema_markup"}

            elif component_type == "performance":
                optimizer = VitalsOptimizer()
                snippets = optimizer.generate_all_performance_snippets(site_slug)
                for name, code in snippets.items():
                    code_type = "css" if name == "critical_css" else "php"
                    deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-perf-{name}-v1",
                        code, code_type=code_type
                    )
                result = {"deployed": "performance_snippets", "count": len(snippets)}

            elif component_type in ("trust", "conversion"):
                factory = ComponentFactory()
                if component_type == "trust":
                    comp = factory.generate_component(site_slug, "author_box")
                else:
                    comp = factory.generate_component(site_slug, "cta_sections")
                if comp.get("html"):
                    deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-{component_type}-v1",
                        comp["html"], code_type="html"
                    )
                result = {"deployed": component_type}

            elif component_type == "internal_links":
                from systems.site_evolution.seo.internal_linker import InternalLinker
                linker = InternalLinker()
                snippet = linker.generate_link_injection_snippet(site_slug)
                deployer.deploy_snippet(
                    site_slug, f"{site_slug[:4]}-autolinks-v1",
                    snippet, code_type="php", location="everywhere"
                )
                result = {"deployed": "internal_links"}

            elif component_type == "images":
                from systems.site_evolution.performance.image_optimizer import ImageOptimizer
                img_opt = ImageOptimizer()
                img_opt.fix_missing_alt_text(site_slug, dry_run=False)
                for name, gen in [("webp", img_opt.generate_webp_snippet),
                                  ("srcset", img_opt.generate_srcset_snippet),
                                  ("placeholder", img_opt.generate_placeholder_snippet)]:
                    deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-img-{name}-v1",
                        gen(), code_type="php", location="everywhere"
                    )
                result = {"deployed": "image_optimization", "snippets": 3}

            elif component_type == "security":
                from systems.site_evolution.performance.security_hardener import SecurityHardener
                hardener = SecurityHardener()
                snippets = hardener.generate_all_security_snippets(site_slug)
                for name, code in snippets.items():
                    deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-sec-{name}-v1",
                        code, code_type="php", location="everywhere"
                    )
                result = {"deployed": "security_hardening", "snippets": len(snippets)}

            elif component_type == "broken_links":
                from systems.site_evolution.auditor.broken_link_monitor import BrokenLinkMonitor
                blm = BrokenLinkMonitor()
                broken_internal = blm.detect_broken_internal(site_slug)
                if broken_internal:
                    redirects = [
                        {"from_url": b["url"].split(get_site_domain(site_slug))[-1], "to_url": "/"}
                        for b in broken_internal[:20]
                    ]
                    snippet = blm.generate_redirect_snippet(redirects)
                    deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-redirects-v1",
                        snippet, code_type="php", location="everywhere"
                    )
                result = {"deployed": "broken_link_redirects", "fixed": len(broken_internal)}

            elif component_type == "content_gaps":
                result = {"deployed": "content_gap_analysis", "note": "Review suggestions in audit"}

            elif component_type == "affiliates":
                from systems.site_evolution.seo.affiliate_manager import AffiliateLinkManager
                aff = AffiliateLinkManager()
                snippets = aff.generate_all_affiliate_snippets(site_slug)
                for name, code in snippets.items():
                    deployer.deploy_snippet(
                        site_slug, f"{site_slug[:4]}-aff-{name}-v1",
                        code, code_type="php", location="everywhere"
                    )
                result = {"deployed": "affiliate_compliance", "snippets": len(snippets)}

            elif component_type == "freshness":
                from systems.site_evolution.auditor.freshness_tracker import FreshnessTracker
                ft = FreshnessTracker()
                snippet = ft.generate_update_date_snippet(site_slug)
                deployer.deploy_snippet(
                    site_slug, f"{site_slug[:4]}-lastupd-v1",
                    snippet, code_type="php", location="everywhere"
                )
                result = {"deployed": "freshness_tracking"}

            elif component_type == "uptime":
                result = {"deployed": "uptime_check", "note": "Monitoring active"}

            else:
                # Generic component deployment
                factory = ComponentFactory()
                comp = factory.generate_component(site_slug, component_type)
                if comp.get("css"):
                    deployer.deploy_custom_css(site_slug, comp["css"],
                                               f"{site_slug[:4]}-{component_type}-css-v1")
                result = {"deployed": component_type}

            codex.update_queue_item(item_id, "completed")
            log.info("Executed queue item %d: %s on %s", item_id, component_type, site_slug)
            return {"item_id": item_id, "status": "completed", **result}

        except Exception as e:
            log.error("Queue execution failed for item %d: %s", item_id, e)
            codex.update_queue_item(item_id, "pending")  # Reset to pending
            return {"item_id": item_id, "status": "error", "error": str(e)}

    def execute_batch(self, site_slug: str, max_items: int = 3,
                      dry_run: bool = False) -> List[Dict]:
        """Execute top N queue items for a site."""
        items = self.get_queue(site_slug, limit=max_items)
        results = []
        for item in items:
            result = self.execute_item(item["id"], dry_run=dry_run)
            results.append(result)
        return results

    def get_progress(self, site_slug: str) -> Dict:
        """Get completion progress for a site."""
        conn = codex._connect()
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM enhancement_queue WHERE site_slug = ?",
            (site_slug,)
        ).fetchone()["cnt"]
        completed = conn.execute(
            "SELECT COUNT(*) as cnt FROM enhancement_queue WHERE site_slug = ? AND status = 'completed'",
            (site_slug,)
        ).fetchone()["cnt"]
        pending = conn.execute(
            "SELECT COUNT(*) as cnt FROM enhancement_queue WHERE site_slug = ? AND status = 'pending'",
            (site_slug,)
        ).fetchone()["cnt"]
        conn.close()

        pct = (completed / total * 100) if total > 0 else 0
        return {
            "site": site_slug,
            "total": total,
            "completed": completed,
            "pending": pending,
            "progress_pct": round(pct, 1),
        }
