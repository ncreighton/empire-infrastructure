#!/usr/bin/env python3
"""
Empire Arsenal — Automated Content Pipeline v2.0
=================================================
End-to-end content generation → SEO → images → WordPress → social distribution.

Architecture:
  LiteLLM Gateway (14 models) → Content Generation (Sonnet)
  → SEO Metadata (Haiku) → Image Pipeline → WordPress REST API
  → Social Distribution → Langfuse Cost Tracking → n8n Notification

Usage:
  # Single article for one site
  python content_pipeline.py --site witchcraftforbeginners --topic "Full Moon Rituals"

  # Batch: generate for multiple sites
  python content_pipeline.py --batch --sites witchcraftforbeginners,smarthomewizards

  # Auto-mode: detect content gaps and fill them
  python content_pipeline.py --auto --max-articles 5

  # Dry run (no publishing)
  python content_pipeline.py --site clearainews --topic "AI Trends" --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
import base64
import hashlib
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

ARSENAL_IP = os.getenv("ARSENAL_IP", "89.116.29.33")
LITELLM_URL = os.getenv("LITELLM_URL", f"http://{ARSENAL_IP}:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-arsenal-fec2dfe2b1256586b84b962c9d25e4e9")
LANGFUSE_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-arsenal-2024")
LANGFUSE_SECRET = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-arsenal-2024")
LANGFUSE_URL = os.getenv("LANGFUSE_HOST", f"http://{ARSENAL_IP}:3004")
N8N_URL = os.getenv("N8N_BASE_URL", f"http://{ARSENAL_IP}:5678")
N8N_API_KEY = os.getenv("N8N_API_KEY", "")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", f"http://{ARSENAL_IP}:11235")
SEARXNG_URL = os.getenv("SEARXNG_URL", f"http://{ARSENAL_IP}:8080")

# Model routing — cost-optimized
MODEL_WRITER = "claude-sonnet"         # Main article writer
MODEL_META = "claude-haiku"            # Metadata, SEO, classifications
MODEL_RESEARCH = "claude-sonnet"       # Research synthesis
MODEL_OUTLINE = "claude-haiku"         # Outlines, structures

SITES_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "sites.json"
IMAGE_PIPELINE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "article_images_pipeline.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("content-pipeline")

# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SiteConfig:
    site_id: str
    name: str
    domain: str
    wp_user: str
    wp_password: str
    amazon_tag: str
    primary_color: str
    secondary_color: str
    accent_color: str
    voice: str
    visual_style: str
    ctas: list = field(default_factory=list)
    headline_font: str = "Inter"
    body_font: str = "Inter"

    @property
    def wp_api_url(self):
        return f"https://{self.domain}/wp-json/wp/v2"

    @property
    def auth_header(self):
        creds = base64.b64encode(f"{self.wp_user}:{self.wp_password}".encode()).decode()
        return {"Authorization": f"Basic {creds}"}


@dataclass
class ArticleResult:
    site_id: str
    title: str
    content: str
    excerpt: str
    seo_title: str
    meta_description: str
    focus_keyword: str
    categories: list
    tags: list
    slug: str
    word_count: int
    affiliate_links: list = field(default_factory=list)
    internal_links: list = field(default_factory=list)
    post_id: Optional[int] = None
    post_url: Optional[str] = None
    featured_image_id: Optional[int] = None
    cost_usd: float = 0.0
    generation_time_s: float = 0.0


@dataclass
class PipelineMetrics:
    articles_generated: int = 0
    articles_published: int = 0
    images_created: int = 0
    total_words: int = 0
    total_cost_usd: float = 0.0
    total_time_s: float = 0.0
    errors: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Site Config Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_sites() -> dict[str, SiteConfig]:
    """Load all site configurations from sites.json."""
    config_path = SITES_CONFIG_PATH
    if not config_path.exists():
        # Try alternate location
        config_path = Path("D:/Claude Code Projects/config/sites.json")

    with open(config_path) as f:
        data = json.load(f)

    sites = {}
    raw_sites = data.get("sites", data)

    for site_id, cfg in raw_sites.items():
        wp = cfg.get("wordpress", {})
        brand = cfg.get("brand", {})
        colors = brand.get("colors", {})
        fonts = brand.get("fonts", {})

        sites[site_id] = SiteConfig(
            site_id=site_id,
            name=cfg.get("name", site_id),
            domain=cfg.get("domain", ""),
            wp_user=wp.get("user", ""),
            wp_password=wp.get("app_password", cfg.get("wp_app_password", "")),
            amazon_tag=cfg.get("amazon_tag", ""),
            primary_color=colors.get("primary", "#333333"),
            secondary_color=colors.get("secondary", "#666666"),
            accent_color=colors.get("accent", "#0066CC"),
            voice=brand.get("voice", "Professional"),
            visual_style=brand.get("visual_style", "modern"),
            ctas=cfg.get("ctas", []),
            headline_font=fonts.get("headline", "Inter"),
            body_font=fonts.get("body", "Inter"),
        )

    return sites


# ──────────────────────────────────────────────────────────────────────────────
# LLM Client (via LiteLLM Gateway)
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """Unified LLM client routing through Arsenal LiteLLM gateway."""

    def __init__(self):
        self.base_url = LITELLM_URL
        self.api_key = LITELLM_KEY
        self.total_cost = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def chat(self, model: str, system: str, user: str,
             max_tokens: int = 4096, temperature: float = 0.7,
             trace_name: str = "content-pipeline") -> tuple[str, dict]:
        """Send a chat completion request. Returns (content, usage_dict)."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": {
                "trace_name": trace_name,
                "trace_metadata": {"pipeline": "content-pipeline-v2"},
            },
        }

        try:
            resp = self.session.post(f"{self.base_url}/chat/completions", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            # Estimate cost (using LiteLLM pricing)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            return content, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": model,
            }

        except Exception as e:
            logger.error(f"LLM request failed ({model}): {e}")
            raise


# ──────────────────────────────────────────────────────────────────────────────
# Research Engine (SearXNG + Crawl4AI)
# ──────────────────────────────────────────────────────────────────────────────

class ResearchEngine:
    """Research topics using Arsenal's SearXNG and Crawl4AI."""

    def __init__(self):
        self.searxng_url = SEARXNG_URL
        self.crawl4ai_url = CRAWL4AI_URL

    def search(self, query: str, num_results: int = 10) -> list[dict]:
        """Search using SearXNG meta-search engine."""
        try:
            resp = requests.get(
                f"{self.searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "engines": "google,bing,duckduckgo",
                    "language": "en",
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])[:num_results]
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "engine": r.get("engine", ""),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"SearXNG search failed: {e}")
            return []

    def crawl_url(self, url: str) -> str:
        """Crawl a specific URL using Crawl4AI for deep content."""
        try:
            resp = requests.post(
                f"{self.crawl4ai_url}/crawl",
                json={
                    "urls": [url],
                    "word_count_threshold": 100,
                    "extraction_strategy": "NoExtractionStrategy",
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("markdown", data[0].get("cleaned_html", ""))[:5000]
            return data.get("markdown", "")[:5000]
        except Exception as e:
            logger.warning(f"Crawl4AI failed for {url}: {e}")
            return ""

    def deep_research(self, topic: str, niche: str) -> str:
        """Perform multi-source research on a topic."""
        queries = [
            f"{topic} {niche} guide 2026",
            f"{topic} best practices tips",
            f"{topic} common mistakes beginners",
        ]

        all_results = []
        for q in queries:
            results = self.search(q, num_results=5)
            all_results.extend(results)

        # Crawl top 3 URLs for deep content
        crawled = []
        urls_crawled = set()
        for r in all_results[:5]:
            url = r.get("url", "")
            if url and url not in urls_crawled:
                content = self.crawl_url(url)
                if content:
                    crawled.append(f"SOURCE: {r['title']}\n{content[:2000]}")
                    urls_crawled.add(url)
            if len(crawled) >= 3:
                break

        research_brief = f"## Research Results for: {topic}\n\n"
        research_brief += "### Search Results:\n"
        for r in all_results[:10]:
            research_brief += f"- **{r['title']}**: {r['snippet'][:200]}\n"

        if crawled:
            research_brief += "\n### Deep Research (Crawled Content):\n"
            for c in crawled:
                research_brief += f"\n{c}\n"

        return research_brief


# ──────────────────────────────────────────────────────────────────────────────
# Content Gap Detector
# ──────────────────────────────────────────────────────────────────────────────

class ContentGapDetector:
    """Analyze sites to find content gaps and suggest topics."""

    NICHE_TOPICS = {
        "witchcraftforbeginners": [
            "moon phase rituals", "crystal grids", "herbal magic", "tarot spreads",
            "protection spells", "altar setup", "sabbat celebrations", "candle magic",
            "sigil creation", "divination methods", "shadow work", "kitchen witchcraft",
            "elemental magic", "dream interpretation", "spirit communication",
        ],
        "smarthomewizards": [
            "home assistant automations", "zigbee vs z-wave", "smart lighting setup",
            "voice assistant comparison", "smart security cameras", "energy monitoring",
            "smart thermostat optimization", "mesh wifi networks", "smart locks",
            "home automation routines", "matter protocol", "thread networking",
        ],
        "mythicalarchives": [
            "greek mythology creatures", "norse gods", "celtic legends", "egyptian deities",
            "japanese yokai", "arthurian legend", "slavic mythology", "hindu mythology",
            "mesoamerican gods", "polynesian myths", "african folklore", "dragon lore",
        ],
        "bulletjournals": [
            "monthly spread ideas", "habit tracker layouts", "mood tracker designs",
            "future log setups", "bujo supplies guide", "minimalist spreads",
            "weekly layout templates", "goal setting pages", "collection page ideas",
            "washi tape techniques", "lettering tutorials", "budget tracker spreads",
        ],
        "wealthfromai": [
            "AI side hustles 2026", "ChatGPT money making", "AI freelancing",
            "automated income streams", "AI content business", "AI trading strategies",
            "prompt engineering careers", "AI SaaS ideas", "AI affiliate marketing",
            "AI art monetization", "AI writing income", "AI consulting business",
        ],
        "aidiscoverydigest": [
            "latest AI tools", "AI research breakthroughs", "AI model comparisons",
            "AI industry news", "AI startup funding", "AI regulations update",
            "AI ethics debates", "open source AI models", "AI hardware advances",
        ],
        "aiinactionhub": [
            "AI workflow automation", "AI in business operations", "AI case studies",
            "AI implementation guides", "AI ROI analysis", "AI tool tutorials",
            "AI productivity hacks", "enterprise AI adoption", "AI integration patterns",
        ],
        "pulsegearreviews": [
            "fitness tracker comparison", "running watch reviews", "gym equipment reviews",
            "workout headphones", "smart water bottles", "fitness app reviews",
            "recovery tools", "smart scales", "resistance band sets",
        ],
        "wearablegearreviews": [
            "smartwatch comparison 2026", "health monitoring wearables", "sleep trackers",
            "blood pressure monitors", "smart rings", "fitness bands",
            "wearable ECG devices", "continuous glucose monitors",
        ],
        "smarthomegearreviews": [
            "smart speaker reviews", "robot vacuum comparison", "smart display reviews",
            "smart plug reviews", "smart bulb comparison", "air purifier reviews",
            "smart doorbell comparison", "smart garage openers",
        ],
        "clearainews": [
            "AI news today", "AI policy updates", "tech company AI moves",
            "AI safety developments", "AI job market changes", "AI tool launches",
            "AI competition landscape", "AI investment trends",
        ],
        "theconnectedhaven": [
            "smart home for families", "connected home security", "smart kitchen guide",
            "home network setup", "smart nursery", "pet tech gadgets",
            "smart garden systems", "whole home automation",
        ],
        "manifestandalign": [
            "manifestation techniques", "law of attraction tips", "affirmation guides",
            "vision board creation", "gratitude practices", "abundance mindset",
            "chakra alignment", "meditation for manifestation", "scripting method",
        ],
        "familyflourish": [
            "family bonding activities", "parenting tips", "educational activities",
            "family meal planning", "kids craft projects", "family budget tips",
            "screen time management", "family traditions", "developmental milestones",
        ],
    }

    def __init__(self, sites: dict[str, SiteConfig], llm: LLMClient):
        self.sites = sites
        self.llm = llm

    def get_existing_posts(self, site: SiteConfig, per_page: int = 100) -> list[str]:
        """Fetch existing post titles from WordPress."""
        try:
            resp = requests.get(
                f"{site.wp_api_url}/posts",
                params={"per_page": per_page, "status": "publish", "_fields": "title"},
                headers=site.auth_header,
                timeout=30,
            )
            resp.raise_for_status()
            return [p["title"]["rendered"] for p in resp.json()]
        except Exception as e:
            logger.warning(f"Failed to fetch posts from {site.domain}: {e}")
            return []

    def detect_gaps(self, site_id: str) -> list[str]:
        """Find content gaps for a specific site."""
        if site_id not in self.sites:
            return []

        site = self.sites[site_id]
        existing = self.get_existing_posts(site)
        existing_lower = [t.lower() for t in existing]

        niche_topics = self.NICHE_TOPICS.get(site_id, [])
        gaps = []

        for topic in niche_topics:
            # Check if topic is already covered
            covered = any(topic.lower() in title for title in existing_lower)
            if not covered:
                gaps.append(topic)

        return gaps

    def suggest_topics(self, site_id: str, count: int = 5) -> list[dict]:
        """AI-powered topic suggestion based on gaps and trends."""
        gaps = self.detect_gaps(site_id)
        site = self.sites.get(site_id)
        if not site:
            return []

        prompt = f"""You are a content strategist for {site.name} ({site.domain}).
Niche voice: {site.voice}

Content gaps found (topics not yet covered):
{json.dumps(gaps[:20], indent=2)}

Generate {count} specific, SEO-optimized article topics. For each:
1. Title (compelling, keyword-rich, 50-60 chars)
2. Primary keyword (2-4 word search term)
3. Search intent (informational/commercial/transactional)
4. Estimated monthly search volume (low/medium/high)
5. Content type (guide/review/listicle/how-to/comparison)

Return as JSON array of objects with keys: title, keyword, intent, volume, type"""

        content, _ = self.llm.chat(
            model=MODEL_META,
            system="You are an expert SEO content strategist. Return only valid JSON.",
            user=prompt,
            max_tokens=1500,
            temperature=0.8,
            trace_name="content-gap-detection",
        )

        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse topic suggestions as JSON")

        return []


# ──────────────────────────────────────────────────────────────────────────────
# Article Generator (Core Pipeline)
# ──────────────────────────────────────────────────────────────────────────────

class ArticleGenerator:
    """Generate complete, SEO-optimized articles via LiteLLM."""

    def __init__(self, llm: LLMClient, research: ResearchEngine):
        self.llm = llm
        self.research = research

    def generate_outline(self, site: SiteConfig, topic: str, keyword: str,
                         research_data: str = "") -> str:
        """Generate article outline using Haiku (fast + cheap)."""
        prompt = f"""Create a detailed article outline for: "{topic}"

Site: {site.name} ({site.domain})
Brand Voice: {site.voice}
Primary Keyword: {keyword}
Target: 2000-3000 words

{f"Research Data:{chr(10)}{research_data[:3000]}" if research_data else ""}

Generate a comprehensive outline with:
1. H1 title (SEO-optimized, includes keyword)
2. Introduction hook (2-3 sentences)
3. 5-8 H2 sections with 2-3 H3 subsections each
4. Key points to cover in each section
5. Where to place affiliate product recommendations
6. Internal linking opportunities
7. FAQ section (5 questions)
8. Conclusion with CTA

Return the outline in markdown format."""

        content, _ = self.llm.chat(
            model=MODEL_OUTLINE,
            system=f"You are an expert content strategist writing for {site.name}. Voice: {site.voice}.",
            user=prompt,
            max_tokens=2000,
            temperature=0.7,
            trace_name="outline-generation",
        )
        return content

    def generate_article(self, site: SiteConfig, topic: str, keyword: str,
                         outline: str, research_data: str = "") -> ArticleResult:
        """Generate the full article using Sonnet (quality writer)."""
        start_time = time.time()

        system_prompt = f"""You are an expert content writer for {site.name} ({site.domain}).

BRAND VOICE: {site.voice}
WRITING STYLE:
- Authoritative yet accessible
- Use short paragraphs (2-3 sentences)
- Include practical, actionable advice
- Natural keyword integration (no stuffing)
- Engaging hooks and transitions
- Use data/statistics when possible
- Include personal-feeling anecdotes

FORMATTING:
- HTML format (WordPress-ready)
- Use <h2>, <h3> for headings (never <h1> - WordPress handles that)
- Use <p> for paragraphs
- Use <ul>/<ol> for lists
- Use <strong> and <em> for emphasis
- Use <blockquote> for notable quotes
- Include helpful <table> elements where appropriate
- Add <!-- wp:separator --> between major sections

SEO REQUIREMENTS:
- Primary keyword: {keyword}
- Use keyword in first 100 words
- Include keyword in at least 2 H2 headings
- Use related/LSI keywords naturally
- Target 2000-3000 words
- Flesch reading ease: 60-70

AFFILIATE INTEGRATION:
- Where relevant, mention products with placeholder: [AFFILIATE: product name]
- Amazon tag: {site.amazon_tag}
- Keep recommendations genuine and helpful

INTERNAL LINKING:
- Suggest 3-5 internal links with placeholder: [INTERNAL: suggested topic]
"""

        user_prompt = f"""Write a complete article on: "{topic}"

OUTLINE:
{outline}

{f"RESEARCH DATA:{chr(10)}{research_data[:4000]}" if research_data else ""}

Write the full article in HTML format. Make it comprehensive, engaging, and SEO-optimized.
Target 2000-3000 words. Include a compelling introduction and conclusion with CTA."""

        content, usage = self.llm.chat(
            model=MODEL_WRITER,
            system=system_prompt,
            user=user_prompt,
            max_tokens=4096,
            temperature=0.75,
            trace_name="article-generation",
        )

        # Generate SEO metadata using Haiku (cheap + fast)
        meta = self._generate_seo_meta(site, topic, keyword, content)

        generation_time = time.time() - start_time

        # Count words (strip HTML)
        text_only = re.sub(r'<[^>]+>', '', content)
        word_count = len(text_only.split())

        return ArticleResult(
            site_id=site.site_id,
            title=meta.get("seo_title", topic),
            content=content,
            excerpt=meta.get("excerpt", ""),
            seo_title=meta.get("seo_title", topic),
            meta_description=meta.get("meta_description", ""),
            focus_keyword=keyword,
            categories=meta.get("categories", ["Uncategorized"]),
            tags=meta.get("tags", []),
            slug=meta.get("slug", self._slugify(topic)),
            word_count=word_count,
            generation_time_s=generation_time,
        )

    def _generate_seo_meta(self, site: SiteConfig, topic: str, keyword: str,
                           content: str) -> dict:
        """Generate SEO metadata using Haiku (fast + cheap)."""
        prompt = f"""Generate SEO metadata for this article.
Topic: {topic}
Keyword: {keyword}
Site: {site.name}
Content preview: {content[:1000]}

Return JSON with:
- seo_title: 50-60 chars, includes keyword, compelling
- meta_description: 150-160 chars, includes keyword, with CTA
- excerpt: 2-3 sentence summary
- slug: url-friendly, includes keyword
- categories: list of 1-2 WordPress categories
- tags: list of 5-8 relevant tags

Return ONLY valid JSON, no markdown."""

        meta_text, _ = self.llm.chat(
            model=MODEL_META,
            system="You are an SEO expert. Return only valid JSON.",
            user=prompt,
            max_tokens=500,
            temperature=0.3,
            trace_name="seo-metadata",
        )

        try:
            json_match = re.search(r'\{.*\}', meta_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return {
            "seo_title": topic,
            "meta_description": f"Discover everything about {topic}. Expert guide from {site.name}.",
            "excerpt": f"A comprehensive guide to {topic}.",
            "slug": self._slugify(topic),
            "categories": ["Uncategorized"],
            "tags": [keyword],
        }

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to URL slug."""
        slug = text.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        return slug[:60]


# ──────────────────────────────────────────────────────────────────────────────
# WordPress Publisher
# ──────────────────────────────────────────────────────────────────────────────

class WordPressPublisher:
    """Publish articles to WordPress via REST API."""

    def __init__(self, sites: dict[str, SiteConfig]):
        self.sites = sites

    def ensure_category(self, site: SiteConfig, category_name: str) -> int:
        """Get or create a WordPress category, return its ID."""
        try:
            # Search for existing
            resp = requests.get(
                f"{site.wp_api_url}/categories",
                params={"search": category_name, "per_page": 5},
                headers=site.auth_header,
                timeout=15,
            )
            resp.raise_for_status()
            cats = resp.json()
            for cat in cats:
                if cat["name"].lower() == category_name.lower():
                    return cat["id"]

            # Create new
            resp = requests.post(
                f"{site.wp_api_url}/categories",
                json={"name": category_name},
                headers={**site.auth_header, "Content-Type": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()["id"]

        except Exception as e:
            logger.warning(f"Category handling failed for '{category_name}': {e}")
            return 1  # Default "Uncategorized"

    def ensure_tags(self, site: SiteConfig, tag_names: list[str]) -> list[int]:
        """Get or create WordPress tags, return their IDs."""
        tag_ids = []
        for tag_name in tag_names[:10]:  # Limit to 10 tags
            try:
                resp = requests.get(
                    f"{site.wp_api_url}/tags",
                    params={"search": tag_name, "per_page": 5},
                    headers=site.auth_header,
                    timeout=10,
                )
                resp.raise_for_status()
                tags = resp.json()

                tag_id = None
                for t in tags:
                    if t["name"].lower() == tag_name.lower():
                        tag_id = t["id"]
                        break

                if not tag_id:
                    resp = requests.post(
                        f"{site.wp_api_url}/tags",
                        json={"name": tag_name},
                        headers={**site.auth_header, "Content-Type": "application/json"},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    tag_id = resp.json()["id"]

                tag_ids.append(tag_id)
            except Exception as e:
                logger.warning(f"Tag handling failed for '{tag_name}': {e}")

        return tag_ids

    def publish(self, site: SiteConfig, article: ArticleResult,
                status: str = "draft") -> dict:
        """Publish an article to WordPress."""
        logger.info(f"Publishing to {site.domain}: {article.title}")

        # Resolve categories and tags
        cat_ids = [self.ensure_category(site, c) for c in article.categories]
        tag_ids = self.ensure_tags(site, article.tags)

        post_data = {
            "title": article.title,
            "content": article.content,
            "excerpt": article.excerpt,
            "status": status,
            "slug": article.slug,
            "categories": cat_ids,
            "tags": tag_ids,
            "meta": {
                "rank_math_focus_keyword": article.focus_keyword,
                "rank_math_title": article.seo_title,
                "rank_math_description": article.meta_description,
            },
        }

        if article.featured_image_id:
            post_data["featured_media"] = article.featured_image_id

        try:
            resp = requests.post(
                f"{site.wp_api_url}/posts",
                json=post_data,
                headers={**site.auth_header, "Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

            article.post_id = result["id"]
            article.post_url = result["link"]

            logger.info(f"✅ Published: {result['link']} (ID: {result['id']}, status: {status})")
            return result

        except Exception as e:
            logger.error(f"Failed to publish to {site.domain}: {e}")
            raise

    def set_rankmath_seo(self, site: SiteConfig, post_id: int, article: ArticleResult):
        """Set RankMath SEO fields via REST API."""
        try:
            resp = requests.post(
                f"{site.wp_api_url}/posts/{post_id}",
                json={
                    "meta": {
                        "rank_math_focus_keyword": article.focus_keyword,
                        "rank_math_title": f"{article.seo_title} - {site.name}",
                        "rank_math_description": article.meta_description,
                        "rank_math_schema_Article": json.dumps({
                            "@type": "Article",
                            "headline": article.seo_title,
                            "description": article.meta_description,
                        }),
                    },
                },
                headers={**site.auth_header, "Content-Type": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            logger.info(f"SEO metadata set for post {post_id}")
        except Exception as e:
            logger.warning(f"RankMath SEO update failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Image Pipeline Integration
# ──────────────────────────────────────────────────────────────────────────────

class ImagePipeline:
    """Generate and upload branded images for articles."""

    def __init__(self):
        self.pipeline_path = IMAGE_PIPELINE_PATH
        if not self.pipeline_path.exists():
            self.pipeline_path = Path("D:/Claude Code Projects/article_images_pipeline.py")

    def generate_and_upload(self, site_id: str, title: str,
                            post_id: Optional[int] = None) -> Optional[int]:
        """Generate images and upload to WordPress. Returns featured_media ID."""
        if not self.pipeline_path.exists():
            logger.warning(f"Image pipeline not found at {self.pipeline_path}")
            return None

        cmd = [
            sys.executable, str(self.pipeline_path),
            "--site", site_id,
            "--title", title,
            "--type", "blog_featured",
            "--enhanced",
        ]

        if post_id:
            cmd.extend(["--post-id", str(post_id)])

        try:
            logger.info(f"Generating images for: {title}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                cwd=str(self.pipeline_path.parent),
            )

            if result.returncode == 0:
                # Parse output for media ID
                for line in result.stdout.split("\n"):
                    if "media_id" in line.lower() or "featured" in line.lower():
                        # Try to extract ID
                        match = re.search(r'(?:media_id|ID)[:\s]+(\d+)', line, re.IGNORECASE)
                        if match:
                            return int(match.group(1))
                logger.info("Images generated and uploaded successfully")
                return None
            else:
                logger.warning(f"Image pipeline returned non-zero: {result.stderr[:500]}")
                return None

        except subprocess.TimeoutExpired:
            logger.warning("Image pipeline timed out")
            return None
        except Exception as e:
            logger.warning(f"Image pipeline failed: {e}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Social Distribution
# ──────────────────────────────────────────────────────────────────────────────

class SocialDistributor:
    """Distribute content via n8n webhook triggers."""

    def __init__(self):
        self.n8n_url = N8N_URL

    def distribute(self, article: ArticleResult, site: SiteConfig):
        """Trigger social distribution via n8n webhook."""
        payload = {
            "event": "article_published",
            "site_id": site.site_id,
            "site_name": site.name,
            "domain": site.domain,
            "article": {
                "title": article.title,
                "url": article.post_url,
                "excerpt": article.excerpt,
                "keyword": article.focus_keyword,
                "word_count": article.word_count,
                "categories": article.categories,
                "tags": article.tags,
            },
            "social_posts": self._generate_social_posts(article, site),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            resp = requests.post(
                f"{self.n8n_url}/webhook/content-distribution",
                json=payload,
                timeout=15,
            )
            if resp.status_code == 200:
                logger.info("Social distribution triggered via n8n")
            else:
                logger.warning(f"n8n webhook returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"Social distribution failed: {e}")

    def _generate_social_posts(self, article: ArticleResult, site: SiteConfig) -> dict:
        """Generate platform-specific social posts."""
        return {
            "twitter": f"🔥 {article.title}\n\n{article.excerpt[:200]}\n\n👉 {article.post_url}\n\n#{article.focus_keyword.replace(' ', '')}",
            "pinterest": {
                "title": article.title,
                "description": f"{article.meta_description}\n\n{' '.join('#' + t.replace(' ', '') for t in article.tags[:5])}",
                "link": article.post_url,
                "board": f"{site.name} Articles",
            },
            "facebook": f"{article.title}\n\n{article.excerpt}\n\nRead more: {article.post_url}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Master Pipeline Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class ContentPipelineOrchestrator:
    """Master orchestrator for the entire content pipeline."""

    def __init__(self):
        self.sites = load_sites()
        self.llm = LLMClient()
        self.research = ResearchEngine()
        self.generator = ArticleGenerator(self.llm, self.research)
        self.publisher = WordPressPublisher(self.sites)
        self.images = ImagePipeline()
        self.social = SocialDistributor()
        self.gap_detector = ContentGapDetector(self.sites, self.llm)
        self.metrics = PipelineMetrics()

    def run_single(self, site_id: str, topic: str, keyword: Optional[str] = None,
                   status: str = "draft", dry_run: bool = False,
                   skip_images: bool = False, skip_social: bool = False) -> ArticleResult:
        """Run the complete pipeline for a single article."""
        site = self.sites.get(site_id)
        if not site:
            raise ValueError(f"Unknown site: {site_id}. Available: {list(self.sites.keys())}")

        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting pipeline for {site.name}: {topic}")
        pipeline_start = time.time()

        # If no keyword, extract one
        if not keyword:
            keyword = self._extract_keyword(topic)

        # Stage 1: Research
        logger.info("📚 Stage 1: Researching topic...")
        research_data = self.research.deep_research(topic, site.voice)

        # Stage 2: Generate outline
        logger.info("📝 Stage 2: Generating outline...")
        outline = self.generator.generate_outline(site, topic, keyword, research_data)

        # Stage 3: Generate article
        logger.info("✍️ Stage 3: Writing article...")
        article = self.generator.generate_article(site, topic, keyword, outline, research_data)

        # Stage 4: Publish to WordPress
        if not dry_run:
            logger.info("🚀 Stage 4: Publishing to WordPress...")
            post_result = self.publisher.publish(site, article, status=status)
            self.publisher.set_rankmath_seo(site, article.post_id, article)

            # Stage 5: Generate and upload images
            if not skip_images and article.post_id:
                logger.info("🎨 Stage 5: Generating images...")
                media_id = self.images.generate_and_upload(
                    site_id, article.title, article.post_id
                )
                if media_id:
                    article.featured_image_id = media_id

            # Stage 6: Social distribution
            if not skip_social and article.post_url:
                logger.info("📱 Stage 6: Distributing to social channels...")
                self.social.distribute(article, site)
        else:
            logger.info("🏃 [DRY RUN] Skipping publish, images, and social")

        # Update metrics
        pipeline_time = time.time() - pipeline_start
        article.generation_time_s = pipeline_time
        self.metrics.articles_generated += 1
        if not dry_run:
            self.metrics.articles_published += 1
        self.metrics.total_words += article.word_count
        self.metrics.total_time_s += pipeline_time

        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║ ✅ Pipeline Complete                                         ║
║ Site: {site.name:<52} ║
║ Title: {article.title[:50]:<51} ║
║ Words: {article.word_count:<52} ║
║ Time: {pipeline_time:.1f}s{' ' * (50 - len(f'{pipeline_time:.1f}s'))} ║
║ URL: {(article.post_url or 'N/A')[:52]:<53} ║
╚══════════════════════════════════════════════════════════════╝
""")

        return article

    def run_batch(self, site_ids: list[str], topics_per_site: int = 1,
                  status: str = "draft", dry_run: bool = False) -> list[ArticleResult]:
        """Run pipeline for multiple sites using auto-detected topics."""
        results = []

        for site_id in site_ids:
            if site_id not in self.sites:
                logger.warning(f"Skipping unknown site: {site_id}")
                continue

            # Get suggested topics
            suggestions = self.gap_detector.suggest_topics(site_id, count=topics_per_site)

            if not suggestions:
                logger.warning(f"No topics suggested for {site_id}, using niche defaults")
                defaults = ContentGapDetector.NICHE_TOPICS.get(site_id, [])[:topics_per_site]
                suggestions = [{"title": t, "keyword": t} for t in defaults]

            for suggestion in suggestions[:topics_per_site]:
                try:
                    article = self.run_single(
                        site_id=site_id,
                        topic=suggestion.get("title", suggestion.get("keyword", "")),
                        keyword=suggestion.get("keyword"),
                        status=status,
                        dry_run=dry_run,
                    )
                    results.append(article)
                except Exception as e:
                    logger.error(f"Pipeline failed for {site_id}: {e}")
                    self.metrics.errors.append(f"{site_id}: {str(e)}")

        self._print_batch_summary(results)
        return results

    def run_auto(self, max_articles: int = 5, status: str = "draft",
                 dry_run: bool = False) -> list[ArticleResult]:
        """Auto-detect gaps across all sites and fill them."""
        logger.info(f"🤖 Auto-mode: Finding content gaps across {len(self.sites)} sites...")

        all_suggestions = []
        for site_id in self.sites:
            suggestions = self.gap_detector.suggest_topics(site_id, count=2)
            for s in suggestions:
                s["site_id"] = site_id
                all_suggestions.append(s)

        # Prioritize by estimated search volume
        volume_order = {"high": 0, "medium": 1, "low": 2}
        all_suggestions.sort(key=lambda x: volume_order.get(x.get("volume", "low"), 2))

        results = []
        for suggestion in all_suggestions[:max_articles]:
            try:
                article = self.run_single(
                    site_id=suggestion["site_id"],
                    topic=suggestion["title"],
                    keyword=suggestion.get("keyword"),
                    status=status,
                    dry_run=dry_run,
                )
                results.append(article)
            except Exception as e:
                logger.error(f"Auto pipeline failed: {e}")
                self.metrics.errors.append(str(e))

        self._print_batch_summary(results)
        return results

    def _extract_keyword(self, topic: str) -> str:
        """Extract primary keyword from topic."""
        # Simple extraction: use 2-3 word core phrase
        words = topic.lower().split()
        stop_words = {"the", "a", "an", "for", "to", "in", "of", "and", "or", "is", "are",
                       "how", "what", "why", "when", "where", "best", "top", "complete", "ultimate",
                       "guide", "your"}
        keywords = [w for w in words if w not in stop_words]
        return " ".join(keywords[:3])

    def _print_batch_summary(self, results: list[ArticleResult]):
        """Print batch pipeline summary."""
        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║ 📊 Batch Pipeline Summary                                   ║
╠══════════════════════════════════════════════════════════════╣
║ Articles Generated: {self.metrics.articles_generated:<39} ║
║ Articles Published: {self.metrics.articles_published:<39} ║
║ Total Words: {self.metrics.total_words:<46} ║
║ Total Time: {self.metrics.total_time_s:.1f}s{' ' * (44 - len(f'{self.metrics.total_time_s:.1f}s'))} ║
║ Errors: {len(self.metrics.errors):<51} ║
╚══════════════════════════════════════════════════════════════╝
""")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Empire Arsenal Content Pipeline v2.0")
    parser.add_argument("--site", help="Site ID (e.g., witchcraftforbeginners)")
    parser.add_argument("--topic", help="Article topic/title")
    parser.add_argument("--keyword", help="Primary SEO keyword (auto-detected if omitted)")
    parser.add_argument("--status", default="draft", choices=["draft", "publish", "pending"],
                        help="WordPress post status (default: draft)")
    parser.add_argument("--batch", action="store_true", help="Batch mode for multiple sites")
    parser.add_argument("--sites", help="Comma-separated site IDs for batch mode")
    parser.add_argument("--auto", action="store_true", help="Auto-detect gaps and fill")
    parser.add_argument("--max-articles", type=int, default=5, help="Max articles in auto mode")
    parser.add_argument("--dry-run", action="store_true", help="Generate without publishing")
    parser.add_argument("--skip-images", action="store_true", help="Skip image generation")
    parser.add_argument("--skip-social", action="store_true", help="Skip social distribution")
    parser.add_argument("--list-sites", action="store_true", help="List all available sites")
    parser.add_argument("--detect-gaps", help="Detect content gaps for a site")

    args = parser.parse_args()
    pipeline = ContentPipelineOrchestrator()

    if args.list_sites:
        print("\n📋 Available Sites:")
        for sid, site in pipeline.sites.items():
            print(f"  {sid:<25} → {site.domain} ({site.voice})")
        return

    if args.detect_gaps:
        gaps = pipeline.gap_detector.detect_gaps(args.detect_gaps)
        suggestions = pipeline.gap_detector.suggest_topics(args.detect_gaps, count=10)
        print(f"\n🔍 Content Gaps for {args.detect_gaps}:")
        for g in gaps[:15]:
            print(f"  ❌ {g}")
        if suggestions:
            print(f"\n💡 Suggested Topics:")
            for s in suggestions:
                print(f"  📝 {s.get('title', 'N/A')} [{s.get('keyword', '')}] ({s.get('volume', 'unknown')} volume)")
        return

    if args.auto:
        pipeline.run_auto(max_articles=args.max_articles, status=args.status, dry_run=args.dry_run)
    elif args.batch:
        site_ids = args.sites.split(",") if args.sites else list(pipeline.sites.keys())
        pipeline.run_batch(site_ids, status=args.status, dry_run=args.dry_run)
    elif args.site and args.topic:
        pipeline.run_single(
            site_id=args.site,
            topic=args.topic,
            keyword=args.keyword,
            status=args.status,
            dry_run=args.dry_run,
            skip_images=args.skip_images,
            skip_social=args.skip_social,
        )
    else:
        parser.print_help()
        print("\n\nExamples:")
        print('  python content_pipeline.py --site witchcraftforbeginners --topic "Full Moon Rituals"')
        print("  python content_pipeline.py --auto --max-articles 3 --dry-run")
        print("  python content_pipeline.py --batch --sites witchcraftforbeginners,smarthomewizards")
        print("  python content_pipeline.py --detect-gaps witchcraftforbeginners")


if __name__ == "__main__":
    main()
