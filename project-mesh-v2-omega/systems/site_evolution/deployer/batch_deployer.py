"""
Batch Deployer — Deploy components to multiple/all sites with parallel execution.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Dict, List, Optional

from systems.site_evolution.deployer.wp_deployer import WPDeployer, _load_sites

log = logging.getLogger(__name__)

# Site categories for targeted deployment
SITE_CATEGORIES = {
    "ai_tech": ["wealthfromai", "aidiscoverydigest", "aiinactionhub", "clearainews"],
    "spiritual": ["witchcraftforbeginners", "manifestandalign"],
    "review": ["pulsegearreviews", "wearablegearreviews", "smarthomegearreviews"],
    "smart_home": ["smarthomewizards", "smarthomegearreviews", "theconnectedhaven"],
    "lifestyle": ["bulletjournals", "familyflourish", "theconnectedhaven"],
    "mythology": ["mythicalarchives"],
}


class BatchDeployer:
    """Deploy components to multiple sites in parallel."""

    def __init__(self, max_workers: int = 4):
        self.deployer = WPDeployer()
        self.max_workers = max_workers
        self.manifest: List[Dict] = []

    def deploy_to_all(self, component_type: str,
                      generator_fn: Callable[[str], Dict],
                      dry_run: bool = False) -> Dict:
        """Deploy a component to all 14 sites.

        Args:
            component_type: e.g., 'hero', 'css_framework', 'schema'
            generator_fn: function(site_slug) -> {html, css, js, snippet_name, ...}
            dry_run: if True, generate but don't push
        """
        sites = list(_load_sites().keys())
        return self._batch_deploy(sites, component_type, generator_fn, dry_run)

    def deploy_to_category(self, category: str, component_type: str,
                           generator_fn: Callable[[str], Dict],
                           dry_run: bool = False) -> Dict:
        """Deploy to a site category."""
        sites = SITE_CATEGORIES.get(category, [])
        if not sites:
            return {"error": f"Unknown category: {category}", "results": {}}
        return self._batch_deploy(sites, component_type, generator_fn, dry_run)

    def deploy_to_sites(self, site_slugs: List[str], component_type: str,
                        generator_fn: Callable[[str], Dict],
                        dry_run: bool = False) -> Dict:
        """Deploy to a specific list of sites."""
        return self._batch_deploy(site_slugs, component_type, generator_fn, dry_run)

    def _batch_deploy(self, sites: List[str], component_type: str,
                      generator_fn: Callable, dry_run: bool) -> Dict:
        results = {}
        errors = []
        started = datetime.now()

        def _process_site(site_slug: str) -> Dict:
            try:
                generated = generator_fn(site_slug)
                if dry_run:
                    return {
                        "site": site_slug,
                        "status": "dry_run",
                        "component": component_type,
                        "generated": {
                            k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
                            for k, v in generated.items()
                            if k in ("html", "css", "js", "snippet_name")
                        }
                    }

                # Deploy based on what was generated
                if "css" in generated and generated["css"]:
                    snippet_name = generated.get(
                        "snippet_name",
                        f"{site_slug}-{component_type}-v1"
                    )
                    self.deployer.deploy_snippet(
                        site_slug, snippet_name, generated["css"],
                        code_type="css", location="site_wide_header"
                    )

                if "html" in generated and generated["html"]:
                    snippet_name = generated.get(
                        "snippet_name_html",
                        f"{site_slug}-{component_type}-html-v1"
                    )
                    location = generated.get("location", "site_wide_footer")
                    self.deployer.deploy_snippet(
                        site_slug, snippet_name, generated["html"],
                        code_type="html", location=location
                    )

                if "php" in generated and generated["php"]:
                    snippet_name = generated.get(
                        "snippet_name_php",
                        f"{site_slug}-{component_type}-php-v1"
                    )
                    self.deployer.deploy_snippet(
                        site_slug, snippet_name, generated["php"],
                        code_type="php", location="site_wide_header"
                    )

                return {
                    "site": site_slug,
                    "status": "deployed",
                    "component": component_type,
                }

            except Exception as e:
                log.error("Batch deploy failed for %s: %s", site_slug, e)
                return {
                    "site": site_slug,
                    "status": "error",
                    "error": str(e),
                }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_process_site, slug): slug
                for slug in sites
            }
            for future in as_completed(futures):
                slug = futures[future]
                try:
                    result = future.result()
                    results[slug] = result
                    if result.get("status") == "error":
                        errors.append(slug)
                except Exception as e:
                    results[slug] = {"site": slug, "status": "error", "error": str(e)}
                    errors.append(slug)

        elapsed = (datetime.now() - started).total_seconds()

        entry = {
            "timestamp": started.isoformat(),
            "component_type": component_type,
            "sites_targeted": len(sites),
            "sites_deployed": len(sites) - len(errors),
            "sites_failed": len(errors),
            "dry_run": dry_run,
            "elapsed_seconds": elapsed,
        }
        self.manifest.append(entry)

        return {
            "summary": entry,
            "results": results,
            "errors": errors,
        }

    def get_manifest(self) -> List[Dict]:
        """Get deployment manifest history."""
        return self.manifest
