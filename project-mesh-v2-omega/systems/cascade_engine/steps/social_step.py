"""Social Step — Generate social media captions."""

import logging
from typing import Dict

from .base import BaseStep

log = logging.getLogger(__name__)


class SocialStep(BaseStep):
    name = "social"
    description = "Generate social media captions for all platforms"
    requires = ["article"]

    def execute(self, context: Dict) -> Dict:
        title = context["title"]
        post_url = context.get("post_url", "")

        # Generate platform-specific captions
        captions = {
            "twitter": self._twitter_caption(title, post_url),
            "pinterest": self._pinterest_caption(title, post_url),
            "facebook": self._facebook_caption(title, post_url),
            "instagram": self._instagram_caption(title),
        }

        return {
            "social_status": "generated",
            "social_captions": captions,
        }

    def _twitter_caption(self, title: str, url: str) -> str:
        return f"{title}\n\n{url}" if url else title

    def _pinterest_caption(self, title: str, url: str) -> str:
        return f"{title} | Click to read more"

    def _facebook_caption(self, title: str, url: str) -> str:
        return f"New article: {title}\n\nRead more: {url}" if url else title

    def _instagram_caption(self, title: str) -> str:
        return f"{title}\n\n#content #empire"

    def dry_run(self, context: Dict) -> Dict:
        return {
            "step": self.name,
            "action": "Generate captions for Twitter, Pinterest, Facebook, Instagram",
            "status": "dry_run",
        }
