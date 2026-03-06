"""Article Step — Generate article content via ZimmWriter."""

import logging
import json
from pathlib import Path
from typing import Dict
from urllib.request import urlopen, Request
from urllib.error import URLError

from .base import BaseStep

log = logging.getLogger(__name__)

ZIMMWRITER_URL = "http://localhost:8765"


class ArticleStep(BaseStep):
    name = "article"
    description = "Generate article via ZimmWriter (port 8765)"
    requires = []

    def execute(self, context: Dict) -> Dict:
        title = context["title"]
        site_slug = context["site_slug"]

        # Try ZimmWriter API
        try:
            payload = json.dumps({
                "title": title,
                "site": site_slug,
                "type": "article",
            }).encode("utf-8")

            req = Request(
                f"{ZIMMWRITER_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=300)
            data = json.loads(resp.read())

            return {
                "article_content": data.get("content", ""),
                "article_word_count": data.get("word_count", 0),
                "article_status": "generated",
            }
        except (URLError, Exception) as e:
            log.warning(f"ZimmWriter unavailable ({e}), article step queued")
            return {
                "article_content": None,
                "article_status": "queued",
                "article_note": f"ZimmWriter not available: {e}",
            }

    def dry_run(self, context: Dict) -> Dict:
        return {
            "step": self.name,
            "action": f"Generate article: '{context['title']}' via ZimmWriter",
            "endpoint": ZIMMWRITER_URL,
            "status": "dry_run",
        }
