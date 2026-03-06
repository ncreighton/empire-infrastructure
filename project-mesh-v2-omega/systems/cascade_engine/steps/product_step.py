"""Product Step — Trigger digital product creation (VelvetVeil/Printables)."""

import logging
from typing import Dict

from .base import BaseStep

log = logging.getLogger(__name__)


class ProductStep(BaseStep):
    name = "product"
    description = "Trigger digital product listing (VelvetVeil/Printables)"
    requires = ["article"]

    def execute(self, context: Dict) -> Dict:
        title = context["title"]
        site_slug = context["site_slug"]

        # Check if this niche has product potential
        product_niches = {
            "witchcraftforbeginners", "bulletjournals", "manifestandalign",
            "mythicalarchives",
        }

        if site_slug not in product_niches:
            return {"product_status": "skipped", "reason": "Niche has no digital products"}

        # Queue product creation (would integrate with VelvetVeil/Printables pipeline)
        return {
            "product_status": "queued",
            "product_type": "related_printable",
            "product_title": f"{title} - Printable Guide",
            "product_note": "Queued for VelvetVeil pipeline",
        }

    def dry_run(self, context: Dict) -> Dict:
        return {
            "step": self.name,
            "action": f"Queue related product for '{context['title']}'",
            "status": "dry_run",
        }
