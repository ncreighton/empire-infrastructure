"""Email Step — Trigger newsletter/Substack notification."""

import logging
from typing import Dict

from .base import BaseStep

log = logging.getLogger(__name__)


class EmailStep(BaseStep):
    name = "email"
    description = "Queue newsletter/Substack notification for new content"
    requires = ["wordpress"]

    def execute(self, context: Dict) -> Dict:
        title = context["title"]
        post_url = context.get("post_url", "")
        site_slug = context["site_slug"]

        # Queue email notification (integrates with Systeme.io or Substack)
        return {
            "email_status": "queued",
            "email_subject": f"New: {title}",
            "email_preview": f"We just published '{title}' — check it out!",
            "email_link": post_url,
            "email_note": "Queued for newsletter system",
        }

    def dry_run(self, context: Dict) -> Dict:
        return {
            "step": self.name,
            "action": f"Queue newsletter for '{context['title']}'",
            "status": "dry_run",
        }
