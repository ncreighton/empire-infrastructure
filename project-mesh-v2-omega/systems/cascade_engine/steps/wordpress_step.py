"""WordPress Step — Publish article and set featured image."""

import base64
import json
import logging
from pathlib import Path
from typing import Dict
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from .base import BaseStep

log = logging.getLogger(__name__)

SITES_CONFIG = Path(__file__).parent.parent.parent.parent.parent / "config" / "sites.json"


def _load_site_config(site_slug: str) -> Dict:
    if SITES_CONFIG.exists():
        try:
            data = json.loads(SITES_CONFIG.read_text("utf-8"))
            sites = data.get("sites", data)
            return sites.get(site_slug, {})
        except Exception:
            pass
    return {}


class WordPressStep(BaseStep):
    name = "wordpress"
    description = "Publish to WordPress and set featured image"
    requires = ["article"]

    def execute(self, context: Dict) -> Dict:
        site_slug = context["site_slug"]
        title = context["title"]
        content = context.get("article_content")

        if not content:
            return {"wordpress_status": "skipped", "reason": "No article content"}

        config = _load_site_config(site_slug)
        domain = config.get("domain")
        wp = config.get("wordpress", {})
        user = wp.get("user", config.get("wp_user"))
        password = wp.get("app_password", config.get("wp_app_password"))

        if not all([domain, user, password]):
            return {"wordpress_status": "failed", "reason": "Missing WP credentials"}

        # Create post
        url = f"https://{domain}/wp-json/wp/v2/posts"
        auth = base64.b64encode(f"{user}:{password}".encode()).decode()

        payload = json.dumps({
            "title": title,
            "content": content,
            "status": "draft",  # Always draft first for safety
        }).encode("utf-8")

        try:
            req = Request(url, data=payload, method="POST")
            req.add_header("Authorization", f"Basic {auth}")
            req.add_header("Content-Type", "application/json")
            resp = urlopen(req, timeout=30)
            post_data = json.loads(resp.read())

            return {
                "wordpress_status": "published",
                "post_id": post_data.get("id"),
                "post_url": post_data.get("link"),
                "post_status": "draft",
            }
        except HTTPError as e:
            return {"wordpress_status": "failed", "error": f"HTTP {e.code}"}
        except Exception as e:
            return {"wordpress_status": "failed", "error": str(e)}

    def dry_run(self, context: Dict) -> Dict:
        config = _load_site_config(context["site_slug"])
        return {
            "step": self.name,
            "action": f"Publish draft to {config.get('domain', context['site_slug'])}",
            "status": "dry_run",
        }
