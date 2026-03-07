"""Auto-Pin Connector — Generates Pinterest pins from published WordPress articles.

Combines:
- WordPress REST API to fetch published articles
- Article images pipeline (enhanced_image_gen) for pin image creation
- Site brand colors and patterns from config/sites.json

Usage:
    from connectors.auto_pin import AutoPinConnector

    connector = AutoPinConnector()

    # Generate pin data for recent articles (images + descriptions)
    pins = connector.generate_pins("witchcraftforbeginners", count=5)

    # Generate pin for a specific article
    pin = connector.generate_pin("witchcraftforbeginners", post_id=123)

    # Get pin-ready content calendar
    calendar = connector.get_pin_calendar("witchcraftforbeginners", days=7)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SITES_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "sites.json"
IMAGE_PIPELINE = Path(__file__).resolve().parent.parent.parent / "article_images_pipeline.py"

# Pinterest-optimized content templates by niche
PIN_TEMPLATES = {
    "witchcraft": {
        "hashtags": ["#witchcraft", "#witchtok", "#spells", "#moonmagick", "#babyWitch",
                     "#witchesofpinterest", "#paganism", "#moonphases", "#crystalmagick"],
        "cta_templates": [
            "Save this for your next {topic} session!",
            "Pin this to your witchcraft board for later!",
            "Bookmark this {topic} guide!",
        ],
    },
    "smart_home": {
        "hashtags": ["#smarthome", "#homeautomation", "#alexa", "#googlehome",
                     "#iot", "#smartliving", "#techlife"],
        "cta_templates": [
            "Save this for your smart home setup!",
            "Pin this {topic} guide for later!",
        ],
    },
    "ai": {
        "hashtags": ["#artificialintelligence", "#AI", "#machinelearning", "#AItools",
                     "#futuretech", "#productivity", "#AIapps"],
        "cta_templates": [
            "Save this AI {topic} guide!",
            "Pin this for your AI toolkit!",
        ],
    },
    "default": {
        "hashtags": ["#tips", "#howto", "#guide", "#lifehacks"],
        "cta_templates": [
            "Save this for later!",
            "Pin this {topic} guide!",
        ],
    },
}

SITE_TO_PIN_NICHE = {
    "witchcraftforbeginners": "witchcraft",
    "manifestandalign": "witchcraft",
    "smarthomewizards": "smart_home",
    "smarthomegearreviews": "smart_home",
    "theconnectedhaven": "smart_home",
    "wealthfromai": "ai",
    "aidiscoverydigest": "ai",
    "aiinactionhub": "ai",
    "clearainews": "ai",
}


class AutoPinConnector:
    """Generates Pinterest-ready content from published WordPress articles."""

    def __init__(self):
        self._sites_cache: dict | None = None

    def _get_site_config(self, site_slug: str) -> dict | None:
        if self._sites_cache is None:
            try:
                with open(SITES_CONFIG) as f:
                    data = json.load(f)
                self._sites_cache = data.get("sites", data)
            except Exception as e:
                logger.warning("Failed to load sites.json: %s", e)
                return None
        return self._sites_cache.get(site_slug)

    def _wp_api(self, site_slug: str, endpoint: str, **kwargs) -> dict | list | None:
        config = self._get_site_config(site_slug)
        if not config:
            return None
        domain = config.get("domain", "")
        wp = config.get("wordpress", {})
        auth = (wp.get("user", ""), wp.get("app_password", ""))
        url = f"https://{domain}/wp-json/wp/v2{endpoint}"
        try:
            resp = requests.get(url, auth=auth if all(auth) else None, timeout=15, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("WP API %s: %s", site_slug, e)
            return None

    @staticmethod
    def _strip_html(html: str) -> str:
        return unescape(re.sub(r'<[^>]+>', '', html)).strip()

    # ── Public Methods ────────────────────────────────────────────────

    def generate_pins(self, site_slug: str, count: int = 5) -> list[dict]:
        """Generate Pinterest pin data for recent articles.

        For each article, produces:
        - Pin title (optimized for Pinterest search)
        - Pin description with hashtags
        - Pin image path (generated via article_images_pipeline)
        - Source URL for the pin link
        """
        posts = self._wp_api(site_slug, "/posts", params={
            "per_page": min(count, 20),
            "status": "publish",
            "orderby": "date",
            "order": "desc",
        })

        if not posts:
            return []

        niche = SITE_TO_PIN_NICHE.get(site_slug, "default")
        template = PIN_TEMPLATES.get(niche, PIN_TEMPLATES["default"])
        pins = []

        for post in posts[:count]:
            title = self._strip_html(post.get("title", {}).get("rendered", ""))
            excerpt = self._strip_html(post.get("excerpt", {}).get("rendered", ""))
            url = post.get("link", "")
            post_id = post.get("id")

            # Build Pinterest-optimized description
            pin_desc = self._build_pin_description(title, excerpt, niche, template)

            # Generate pin image
            image_path = self._generate_pin_image(site_slug, title, post_id)

            pins.append({
                "post_id": post_id,
                "title": self._optimize_pin_title(title),
                "description": pin_desc,
                "source_url": url,
                "image_path": image_path,
                "niche": niche,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

        # Log to brain
        self._log_to_brain(site_slug, len(pins))

        return pins

    def generate_pin(self, site_slug: str, post_id: int) -> dict:
        """Generate a Pinterest pin for a specific article."""
        post = self._wp_api(site_slug, f"/posts/{post_id}")
        if not post:
            return {"status": "error", "error": f"Article {post_id} not found"}

        title = self._strip_html(post.get("title", {}).get("rendered", ""))
        excerpt = self._strip_html(post.get("excerpt", {}).get("rendered", ""))
        niche = SITE_TO_PIN_NICHE.get(site_slug, "default")
        template = PIN_TEMPLATES.get(niche, PIN_TEMPLATES["default"])

        pin_desc = self._build_pin_description(title, excerpt, niche, template)
        image_path = self._generate_pin_image(site_slug, title, post_id)

        return {
            "status": "generated",
            "post_id": post_id,
            "title": self._optimize_pin_title(title),
            "description": pin_desc,
            "source_url": post.get("link", ""),
            "image_path": image_path,
            "niche": niche,
        }

    def get_pin_calendar(self, site_slug: str, days: int = 7) -> dict:
        """Get a pin-ready content calendar for upcoming days.

        Recommends optimal posting times and which articles to pin.
        """
        articles = self.generate_pins(site_slug, count=days * 2)

        # Pinterest optimal posting: 2-5 pins per day
        # Best times: 8-11 PM, weekends perform best
        schedule = []
        for i, pin in enumerate(articles[:days]):
            day_offset = i
            schedule.append({
                "day": f"Day {day_offset + 1}",
                "pin": pin,
                "recommended_time": "8:00 PM - 11:00 PM",
                "board_suggestion": self._suggest_board(pin["niche"], pin["title"]),
            })

        return {
            "site": site_slug,
            "days": days,
            "total_pins": len(schedule),
            "schedule": schedule,
            "tips": [
                "Pin at 8-11 PM for best engagement",
                "Use rich pins for article content",
                "Add 3-5 relevant boards per pin",
                "Re-pin top performers after 30 days",
            ],
        }

    # ── Internal Helpers ──────────────────────────────────────────────

    def _optimize_pin_title(self, title: str) -> str:
        """Optimize article title for Pinterest search (max 100 chars)."""
        # Remove special characters that don't work on Pinterest
        clean = re.sub(r'[^\w\s\-:,!?\'"]', '', title)
        # Ensure it's under 100 chars
        if len(clean) > 100:
            clean = clean[:97] + "..."
        return clean

    def _build_pin_description(self, title: str, excerpt: str,
                                niche: str, template: dict) -> str:
        """Build a Pinterest-optimized description with hashtags."""
        import random

        # Start with excerpt (or trimmed title)
        desc = excerpt[:200] if excerpt else title

        # Add CTA
        cta_templates = template.get("cta_templates", ["Save this for later!"])
        topic = title.split(":")[0].strip() if ":" in title else title[:30]
        cta = random.choice(cta_templates).format(topic=topic)

        # Add hashtags (Pinterest allows up to 20, use 5-8)
        hashtags = template.get("hashtags", [])
        selected_tags = random.sample(hashtags, min(6, len(hashtags)))
        tags_str = " ".join(selected_tags)

        # Combine (Pinterest max description: 500 chars)
        full_desc = f"{desc}\n\n{cta}\n\n{tags_str}"
        return full_desc[:500]

    def _generate_pin_image(self, site_slug: str, title: str,
                             post_id: int | None = None) -> str | None:
        """Generate a Pinterest pin image using the article images pipeline."""
        if not IMAGE_PIPELINE.exists():
            logger.warning("Image pipeline not found: %s", IMAGE_PIPELINE)
            return None

        try:
            cmd = [
                "D:/Python314/python.exe",
                str(IMAGE_PIPELINE),
                "--site", site_slug,
                "--title", title,
                "--type", "pinterest_pin",
                "--enhanced",
                "--no-upload",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                # Parse output for image path
                for line in result.stdout.split("\n"):
                    if "pinterest_pin" in line.lower() and (".png" in line or ".jpg" in line):
                        path_match = re.search(r'(/tmp/[^\s]+\.(?:png|jpg))', line)
                        if path_match:
                            return path_match.group(1)
                # If no specific path found, return indication it was generated
                return f"generated:pinterest_pin:{site_slug}:{post_id or 'new'}"
            else:
                logger.warning("Image pipeline failed: %s", result.stderr[:200])
                return None
        except Exception as e:
            logger.warning("Image generation failed: %s", e)
            return None

    def _suggest_board(self, niche: str, title: str) -> str:
        """Suggest a Pinterest board based on niche and content."""
        board_map = {
            "witchcraft": ["Witchcraft & Spells", "Moon Magick", "Crystal Healing",
                          "Beginner Witch", "Pagan Living"],
            "smart_home": ["Smart Home Ideas", "Home Automation", "Tech Gadgets",
                          "Connected Home"],
            "ai": ["AI Tools & Tips", "Artificial Intelligence", "Tech Trends",
                   "Productivity"],
        }
        boards = board_map.get(niche, ["Helpful Guides"])

        # Try to match content to specific board
        title_lower = title.lower()
        if "moon" in title_lower or "lunar" in title_lower:
            return "Moon Magick"
        if "crystal" in title_lower or "gem" in title_lower:
            return "Crystal Healing"
        if "spell" in title_lower or "ritual" in title_lower:
            return "Witchcraft & Spells"
        if "alexa" in title_lower or "google" in title_lower:
            return "Home Automation"

        import random
        return random.choice(boards)

    def _log_to_brain(self, site_slug: str, pin_count: int):
        """Log pin generation to brain database."""
        try:
            from knowledge.brain_db import get_db
            conn = get_db()
            conn.execute("""
                INSERT INTO events (event_type, data, source, timestamp)
                VALUES (?, ?, 'auto_pin', ?)
            """, (
                "pins.generated",
                json.dumps({"site": site_slug, "count": pin_count}),
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("Brain logging skipped: %s", e)
