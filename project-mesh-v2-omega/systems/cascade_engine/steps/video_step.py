"""Video Step — Create video via VideoForge API."""

import json
import logging
from typing import Dict
from urllib.request import urlopen, Request
from urllib.error import URLError

from .base import BaseStep

log = logging.getLogger(__name__)

VIDEOFORGE_URL = "http://localhost:8090"


class VideoStep(BaseStep):
    name = "video"
    description = "Create video via VideoForge API (port 8090)"
    requires = []

    def execute(self, context: Dict) -> Dict:
        title = context["title"]
        site_slug = context["site_slug"]

        try:
            payload = json.dumps({
                "title": title,
                "niche": site_slug,
                "platforms": ["youtube"],
            }).encode("utf-8")

            req = Request(
                f"{VIDEOFORGE_URL}/create",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=600)  # Videos take time
            data = json.loads(resp.read())

            return {
                "video_status": "created",
                "video_id": data.get("video_id"),
                "video_url": data.get("render_url"),
                "video_cost": data.get("cost_estimate"),
            }
        except (URLError, Exception) as e:
            log.warning(f"VideoForge unavailable: {e}")
            return {"video_status": "queued", "video_note": str(e)}

    def dry_run(self, context: Dict) -> Dict:
        return {
            "step": self.name,
            "action": f"Create video for '{context['title']}' via VideoForge",
            "endpoint": VIDEOFORGE_URL,
            "estimated_cost": "$0.50",
            "status": "dry_run",
        }
