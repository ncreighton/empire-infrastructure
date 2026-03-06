"""Internal Link Step — Add cross-site links via Cross-Pollination Engine."""

import logging
import sys
from pathlib import Path
from typing import Dict

from .base import BaseStep

log = logging.getLogger(__name__)


class InternalLinkStep(BaseStep):
    name = "internal_link"
    description = "Add cross-site links via Cross-Pollination Engine"
    requires = ["wordpress"]

    def execute(self, context: Dict) -> Dict:
        post_url = context.get("post_url")
        site_slug = context["site_slug"]

        if not post_url:
            return {"link_status": "skipped", "reason": "No post URL available"}

        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
            from systems.cross_pollination import CrossPollinationEngine
            engine = CrossPollinationEngine()

            # Get overlapping sites for this site
            overlaps = engine.get_overlaps()
            related_sites = []
            for overlap in overlaps:
                if isinstance(overlap, dict):
                    if site_slug in (overlap.get("site_a", ""), overlap.get("site_b", "")):
                        other = overlap["site_b"] if overlap["site_a"] == site_slug else overlap["site_a"]
                        related_sites.append(other)

            return {
                "link_status": "analyzed",
                "related_sites": related_sites[:5],
                "link_suggestions": len(related_sites),
            }
        except Exception as e:
            log.debug(f"Cross-pollination integration: {e}")
            return {"link_status": "skipped", "reason": str(e)}

    def dry_run(self, context: Dict) -> Dict:
        return {
            "step": self.name,
            "action": f"Find cross-site links for new post on {context['site_slug']}",
            "status": "dry_run",
        }
