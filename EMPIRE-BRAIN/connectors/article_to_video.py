"""Article-to-Video Pipeline — Converts published WordPress articles into video scripts.

Reads articles from any Empire WordPress site via the WP REST API,
extracts key points, and feeds to VideoForge for video creation.

Usage:
    from connectors.article_to_video import ArticleToVideoPipeline

    pipeline = ArticleToVideoPipeline()

    # List recent articles from a site
    articles = pipeline.list_articles("witchcraftforbeginners", count=5)

    # Convert a specific article to a video (dry run)
    result = pipeline.convert_article("witchcraftforbeginners", post_id=123)

    # Convert by URL
    result = pipeline.convert_by_url("https://witchcraftforbeginners.com/full-moon-ritual/")

    # Batch: find recent articles and convert top candidates
    results = pipeline.batch_convert("witchcraftforbeginners", count=3)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

VIDEOFORGE_BASE = "http://localhost:8090"
SITES_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "sites.json"

# Map site slugs to VideoForge niche names
SITE_TO_NICHE = {
    "witchcraftforbeginners": "witchcraft",
    "smarthomewizards": "smart_home",
    "mythicalarchives": "mythology",
    "bulletjournals": "journaling",
    "wealthfromai": "ai_finance",
    "aidiscoverydigest": "ai_news",
    "aiinactionhub": "ai_news",
    "clearainews": "ai_news",
    "pulsegearreviews": "fitness",
    "wearablegearreviews": "fitness",
    "smarthomegearreviews": "smart_home",
    "theconnectedhaven": "smart_home",
    "manifestandalign": "witchcraft",
    "familyflourish": "family",
}


class ArticleToVideoPipeline:
    """Converts WordPress articles into video scripts for VideoForge."""

    def __init__(self, videoforge_url: str = VIDEOFORGE_BASE):
        self.videoforge_url = videoforge_url.rstrip("/")
        self._sites_cache: dict | None = None

    def _get_site_config(self, site_slug: str) -> dict | None:
        """Load site config from sites.json."""
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
        """Call WordPress REST API with auth."""
        config = self._get_site_config(site_slug)
        if not config:
            logger.warning("Site config not found: %s", site_slug)
            return None

        domain = config.get("domain", "")
        wp = config.get("wordpress", {})
        user = wp.get("user", "")
        password = wp.get("app_password", "")

        url = f"https://{domain}/wp-json/wp/v2{endpoint}"
        auth = (user, password) if user and password else None

        try:
            resp = requests.get(url, auth=auth, timeout=15, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("WP API %s: %s -> %s", site_slug, endpoint, e)
            return None

    def _videoforge(self, method: str, endpoint: str, **kwargs) -> dict | None:
        try:
            resp = requests.request(method, f"{self.videoforge_url}{endpoint}",
                                    timeout=120, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("VideoForge: %s %s -> %s", method, endpoint, e)
            return None

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags and decode entities."""
        text = re.sub(r'<[^>]+>', '', html)
        return unescape(text).strip()

    @staticmethod
    def _extract_key_points(content: str, max_points: int = 5) -> list[str]:
        """Extract key points from article content for video narration."""
        # Split by headings (h2, h3) to find sections
        sections = re.split(r'<h[23][^>]*>', content)
        points = []

        for section in sections:
            # Get heading text
            heading_match = re.search(r'^([^<]+)', section)
            if heading_match:
                heading = heading_match.group(1).strip()
                if heading and len(heading) > 5:
                    # Clean the heading
                    clean = re.sub(r'</h[23]>', '', heading).strip()
                    if clean:
                        points.append(clean)

        # If no headings found, extract first sentences
        if not points:
            text = ArticleToVideoPipeline._strip_html(content)
            sentences = re.split(r'[.!?]\s+', text)
            points = [s.strip() for s in sentences[:max_points] if len(s.strip()) > 20]

        return points[:max_points]

    # ── Public Methods ────────────────────────────────────────────────

    def list_articles(self, site_slug: str, count: int = 10,
                      category: str | None = None) -> list[dict]:
        """List recent published articles from a WordPress site."""
        params = {
            "per_page": min(count, 100),
            "status": "publish",
            "orderby": "date",
            "order": "desc",
        }
        if category:
            params["categories"] = category

        posts = self._wp_api(site_slug, "/posts", params=params)
        if not posts:
            return []

        articles = []
        for post in posts:
            title = self._strip_html(post.get("title", {}).get("rendered", ""))
            content = post.get("content", {}).get("rendered", "")
            key_points = self._extract_key_points(content)

            articles.append({
                "id": post.get("id"),
                "title": title,
                "slug": post.get("slug", ""),
                "url": post.get("link", ""),
                "date": post.get("date", ""),
                "word_count": len(self._strip_html(content).split()),
                "key_points": key_points,
                "video_potential": self._score_video_potential(title, content, key_points),
            })

        # Sort by video potential
        articles.sort(key=lambda x: x["video_potential"], reverse=True)
        return articles

    def convert_article(
        self,
        site_slug: str,
        post_id: int,
        *,
        render: bool = False,
        publish: bool = False,
    ) -> dict:
        """Convert a specific article to a video.

        Args:
            site_slug: WordPress site identifier
            post_id: WordPress post ID
            render: Whether to render via Creatomate
            publish: Whether to publish to YouTube/TikTok
        """
        # Fetch the article
        post = self._wp_api(site_slug, f"/posts/{post_id}")
        if not post:
            return {"status": "error", "error": f"Article {post_id} not found on {site_slug}"}

        title = self._strip_html(post.get("title", {}).get("rendered", ""))
        content = post.get("content", {}).get("rendered", "")
        key_points = self._extract_key_points(content)

        # Build video topic from article
        video_topic = self._build_video_topic(title, key_points)
        niche = SITE_TO_NICHE.get(site_slug, "educational")

        # Send to VideoForge
        if render:
            vf_result = self._videoforge("POST", "/create", json={
                "topic": video_topic,
                "niche": niche,
                "platform": "youtube_shorts",
                "format": "short",
                "render": True,
                "publish": publish,
            })
        else:
            vf_result = self._videoforge("POST", "/analyze", json={
                "topic": video_topic,
                "niche": niche,
                "platform": "youtube_shorts",
            })

        # Log to brain
        self._log_conversion(site_slug, post_id, title, not render)

        return {
            "status": "analyzed" if not render else "created",
            "source": {
                "site": site_slug,
                "post_id": post_id,
                "title": title,
                "url": post.get("link", ""),
                "word_count": len(self._strip_html(content).split()),
                "key_points": key_points,
            },
            "video": {
                "topic": video_topic,
                "niche": niche,
                "videoforge_result": vf_result,
            },
            "converted_at": datetime.now(timezone.utc).isoformat(),
        }

    def convert_by_url(self, url: str, *, render: bool = False, publish: bool = False) -> dict:
        """Convert an article by its URL. Auto-detects site and post ID."""
        # Extract domain from URL
        domain_match = re.search(r'https?://([^/]+)', url)
        if not domain_match:
            return {"status": "error", "error": f"Invalid URL: {url}"}

        domain = domain_match.group(1).replace("www.", "")

        # Find matching site
        site_slug = None
        for slug, niche in SITE_TO_NICHE.items():
            config = self._get_site_config(slug)
            if config and config.get("domain") == domain:
                site_slug = slug
                break

        if not site_slug:
            return {"status": "error", "error": f"No site config for domain: {domain}"}

        # Extract slug from URL
        slug_match = re.search(r'/([^/]+)/?$', url.rstrip("/"))
        article_slug = slug_match.group(1) if slug_match else ""

        # Find post by slug
        posts = self._wp_api(site_slug, "/posts", params={"slug": article_slug})
        if not posts or not isinstance(posts, list) or len(posts) == 0:
            return {"status": "error", "error": f"Article not found: {article_slug}"}

        post_id = posts[0].get("id")
        return self.convert_article(site_slug, post_id, render=render, publish=publish)

    def batch_convert(
        self,
        site_slug: str,
        count: int = 3,
        *,
        render: bool = False,
    ) -> dict:
        """Find the best recent articles and convert them to videos.

        Scores articles by video potential and converts the top candidates.
        """
        articles = self.list_articles(site_slug, count=count * 2)  # Get 2x for filtering
        top_articles = articles[:count]

        results = []
        for article in top_articles:
            result = self.convert_article(
                site_slug, article["id"], render=render,
            )
            results.append(result)

        return {
            "site": site_slug,
            "articles_scanned": len(articles),
            "videos_created": len(results),
            "rendered": render,
            "results": results,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_conversion_stats(self) -> dict:
        """Get article-to-video conversion statistics from brain."""
        try:
            from knowledge.brain_db import get_db
            conn = get_db()
            events = conn.execute("""
                SELECT data FROM events
                WHERE event_type = 'article.to_video'
                ORDER BY timestamp DESC LIMIT 50
            """).fetchall()
            conn.close()

            return {
                "total_conversions": len(events),
                "recent": [json.loads(e[0]) if e[0] else {} for e in events[:5]],
            }
        except Exception:
            return {"total_conversions": 0, "recent": []}

    # ── Internal Helpers ──────────────────────────────────────────────

    def _build_video_topic(self, title: str, key_points: list[str]) -> str:
        """Build a VideoForge-friendly topic from article title and key points."""
        # Start with the title
        topic = title

        # Add top 3 key points as context
        if key_points:
            points_text = ". ".join(key_points[:3])
            topic = f"{title}. Key points: {points_text}"

        # Cap at 500 chars for VideoForge
        return topic[:500]

    def _score_video_potential(self, title: str, content: str, key_points: list[str]) -> float:
        """Score how well an article would convert to a video (0-10)."""
        score = 5.0  # Base score

        # Bonus for how-to / tutorial content
        how_to_words = ["how to", "guide", "tutorial", "step by step", "tips", "ways to"]
        title_lower = title.lower()
        for word in how_to_words:
            if word in title_lower:
                score += 1.0
                break

        # Bonus for list content
        if re.search(r'^\d+\s', title) or "best" in title_lower:
            score += 0.5

        # Bonus for having good structure (headings = key points)
        if len(key_points) >= 3:
            score += 1.0
        elif len(key_points) >= 5:
            score += 1.5

        # Bonus for medium length (not too short, not too long)
        word_count = len(self._strip_html(content).split())
        if 500 <= word_count <= 2000:
            score += 1.0
        elif word_count < 300:
            score -= 1.0

        # Bonus for engaging title length
        if 30 <= len(title) <= 70:
            score += 0.5

        return min(round(score, 1), 10.0)

    def _log_conversion(self, site_slug: str, post_id: int, title: str, dry_run: bool):
        """Log conversion event to brain database."""
        try:
            from knowledge.brain_db import get_db
            conn = get_db()
            conn.execute("""
                INSERT INTO events (event_type, data, source, timestamp)
                VALUES (?, ?, 'article_to_video', ?)
            """, (
                "article.to_video",
                json.dumps({"site": site_slug, "post_id": post_id,
                            "title": title, "dry_run": dry_run})[:500],
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("Brain logging skipped: %s", e)
