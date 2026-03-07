"""Publisher — Multi-platform video export (YouTube, TikTok, WordPress)."""

import os
from pathlib import Path
import json
import logging
import requests
from datetime import datetime
from ..models import VideoPlan

logger = logging.getLogger(__name__)


class Publisher:
    """Publishes rendered videos to multiple platforms."""

    def publish(self, plan: VideoPlan, platforms: list = None) -> dict:
        """Publish a rendered video to specified platforms.

        Args:
            plan: VideoPlan with render_url set
            platforms: List of platform keys to publish to.
                       Defaults to the plan's platform.

        Returns:
            Dict of {platform: {status, url, error}}
        """
        if not plan.render_url:
            return {"error": "No render URL — video must be rendered first"}

        if not platforms:
            platforms = [plan.platform]

        results = {}
        for platform in platforms:
            if platform in ("youtube_shorts", "youtube"):
                results[platform] = self._publish_youtube(plan)
            elif platform == "tiktok":
                results[platform] = self._publish_tiktok(plan)
            elif platform in ("instagram_reels", "facebook_reels"):
                results[platform] = self._publish_meta(plan, platform)
            elif platform == "wordpress":
                results[platform] = self._publish_wordpress(plan)
            else:
                results[platform] = {"status": "unsupported", "error": f"Unknown platform: {platform}"}

        return results

    def prepare_metadata(self, plan: VideoPlan) -> dict:
        """Prepare publishing metadata from a video plan."""
        sb = plan.storyboard
        if not sb:
            return {}

        return {
            "title": sb.title,
            "description": self._build_description(plan),
            "hashtags": sb.hashtags,
            "thumbnail_concept": sb.thumbnail_concept,
            "platform": plan.platform,
            "niche": plan.niche,
        }

    def _build_description(self, plan: VideoPlan) -> str:
        """Build platform-appropriate description."""
        sb = plan.storyboard
        if not sb:
            return ""

        parts = [sb.title]

        if sb.cta_text:
            parts.append(f"\n{sb.cta_text}")

        if sb.hashtags:
            parts.append("\n" + " ".join(sb.hashtags))

        return "\n".join(parts)

    def _publish_youtube(self, plan: VideoPlan) -> dict:
        """Publish to YouTube via API.

        Note: Full YouTube upload requires OAuth2 flow.
        This provides the payload structure for integration.
        """
        sb = plan.storyboard
        if not sb:
            return {"status": "error", "error": "No storyboard"}

        # YouTube upload payload (for use with YouTube Data API v3)
        payload = {
            "snippet": {
                "title": sb.title[:100],
                "description": self._build_description(plan)[:5000],
                "tags": [h.replace("#", "") for h in sb.hashtags[:15]],
                "categoryId": "22",  # People & Blogs
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            },
        }

        if plan.format == "short":
            payload["snippet"]["description"] += "\n#Shorts"

        logger.info(f"YouTube publish payload prepared for: {sb.title}")
        return {
            "status": "payload_ready",
            "payload": payload,
            "video_url": plan.render_url,
            "note": "Requires YouTube OAuth2 upload implementation",
        }

    def _publish_tiktok(self, plan: VideoPlan) -> dict:
        """Publish to TikTok.

        TikTok publishing requires their Content Posting API.
        """
        sb = plan.storyboard
        if not sb:
            return {"status": "error", "error": "No storyboard"}

        payload = {
            "post_info": {
                "title": sb.title[:150],
                "privacy_level": "PUBLIC_TO_EVERYONE",
                "disable_duet": False,
                "disable_stitch": False,
                "disable_comment": False,
            },
            "source_info": {
                "source": "PULL_FROM_URL",
                "video_url": plan.render_url,
            },
        }

        logger.info(f"TikTok publish payload prepared for: {sb.title}")
        return {
            "status": "payload_ready",
            "payload": payload,
            "video_url": plan.render_url,
            "note": "Requires TikTok Content Posting API credentials",
        }

    def _publish_meta(self, plan: VideoPlan, platform: str) -> dict:
        """Publish to Instagram/Facebook Reels."""
        sb = plan.storyboard
        if not sb:
            return {"status": "error", "error": "No storyboard"}

        payload = {
            "caption": self._build_description(plan)[:2200],
            "video_url": plan.render_url,
            "media_type": "REELS",
        }

        logger.info(f"{platform} publish payload prepared for: {sb.title}")
        return {
            "status": "payload_ready",
            "payload": payload,
            "video_url": plan.render_url,
            "note": f"Requires Meta Graph API for {platform}",
        }

    def _publish_wordpress(self, plan: VideoPlan) -> dict:
        """Upload video to WordPress media library."""
        sb = plan.storyboard
        if not sb:
            return {"status": "error", "error": "No storyboard"}

        # Load site config
        try:
            config_path = Path(os.path.dirname(__file__)) / ".." / ".." / ".." / "config" / "sites.json"
            with open(config_path) as f:
                sites_data = json.load(f)
                sites = sites_data.get("sites", sites_data)
                site = sites.get(plan.niche)

            if not site:
                return {"status": "error", "error": f"No WordPress config for {plan.niche}"}

            domain = site["domain"]
            wp_user = site["wp_user"]
            wp_pass = site["wp_app_password"]

            # Upload video URL as WordPress media
            logger.info(f"WordPress upload prepared for {domain}: {sb.title}")
            return {
                "status": "payload_ready",
                "domain": domain,
                "video_url": plan.render_url,
                "title": sb.title,
                "note": "Use wp-json/wp/v2/media endpoint for actual upload",
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}
