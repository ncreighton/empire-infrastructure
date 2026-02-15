"""
Internal Linking Engine — OpenClaw Empire Edition
==================================================

Builds a link graph per site from existing WordPress posts, identifies linking
opportunities, suggests contextual internal links for new content, and can
auto-inject links into articles. Critical for SEO topical authority across
Nick Creighton's 16-site publishing empire.

The engine operates entirely on keyword overlap, category/tag matching, and
link structure analysis — no AI API calls needed. Fast, free, deterministic.

Usage:
    from src.internal_linker import get_linker

    linker = get_linker()
    graph = await linker.build_graph("witchcraft")
    report = await linker.link_health("witchcraft")
    opportunities = await linker.suggest_links("witchcraft", post_id=1234)

CLI:
    python -m src.internal_linker build --site witchcraft
    python -m src.internal_linker health --site witchcraft
    python -m src.internal_linker orphans --site witchcraft
    python -m src.internal_linker suggest --site witchcraft --post-id 1234
    python -m src.internal_linker suggest-new --site witchcraft --title "Moon Water" --keywords "moon water,lunar"
    python -m src.internal_linker inject --site witchcraft --post-id 1234 --dry-run
    python -m src.internal_linker pillars --site witchcraft
    python -m src.internal_linker clusters --site witchcraft
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("internal_linker")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SITE_REGISTRY_PATH = BASE_DIR / "configs" / "site-registry.json"
DATA_DIR = BASE_DIR / "data" / "linker"

# Ensure data directory exists on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# WP REST API pagination limit
WP_MAX_PER_PAGE = 100

# Scoring weights for combined relevance
WEIGHT_KEYWORD_OVERLAP = 0.35
WEIGHT_CATEGORY_MATCH = 0.25
WEIGHT_RECENCY = 0.15
WEIGHT_LINK_NEED = 0.25

# Thresholds
DEFAULT_OVERLINKED_THRESHOLD = 20
DEFAULT_UNDERLINKED_THRESHOLD = 2
PILLAR_MIN_INCOMING = 5

# Stopwords excluded from keyword extraction
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "also", "get", "got", "has", "had", "have", "do", "does", "did",
    "be", "been", "being", "am", "is", "are", "was", "were", "it", "its",
    "he", "she", "we", "they", "them", "his", "her", "our", "your", "my",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "if", "as", "you", "i", "me", "us", "him", "would", "could",
    "may", "might", "shall", "must", "need", "one", "two", "make",
    "like", "new", "best", "good", "great", "way", "use", "using",
}

# Max age for recency scoring (days)
RECENCY_MAX_DAYS = 365


# ---------------------------------------------------------------------------
# HTML Parser helpers (stdlib only, no BeautifulSoup)
# ---------------------------------------------------------------------------


class _LinkExtractor(HTMLParser):
    """Extract all <a href="..."> links from HTML content."""

    def __init__(self, domain: str):
        super().__init__()
        self.domain = domain.lower().rstrip("/")
        self.internal_hrefs: List[str] = []
        self.external_hrefs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        href = None
        for name, value in attrs:
            if name == "href" and value:
                href = value.strip()
                break
        if not href:
            return

        parsed = urlparse(href)
        link_domain = parsed.netloc.lower().rstrip("/")

        # Internal link: same domain, or relative URL
        if not link_domain or link_domain == self.domain or link_domain == f"www.{self.domain}":
            if parsed.path and parsed.path != "/":
                self.internal_hrefs.append(href)
        else:
            self.external_hrefs.append(href)


class _TextExtractor(HTMLParser):
    """Extract plain text from HTML, stripping all tags."""

    def __init__(self):
        super().__init__()
        self._parts: List[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


class _ParagraphExtractor(HTMLParser):
    """Extract paragraph text blocks from HTML for link injection."""

    def __init__(self):
        super().__init__()
        self.paragraphs: List[Tuple[int, int, str]] = []
        self._in_p = False
        self._depth = 0
        self._current_text: List[str] = []
        self._p_start = 0
        self._raw = ""

    def feed(self, data: str) -> None:
        self._raw = data
        super().feed(data)

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "p":
            self._in_p = True
            self._depth += 1
            self._current_text = []
            self._p_start = self.getpos()[1]

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._in_p:
            self._depth -= 1
            if self._depth <= 0:
                self._in_p = False
                self._depth = 0
                text = " ".join(self._current_text).strip()
                if text:
                    self.paragraphs.append((self._p_start, self.getpos()[1], text))

    def handle_data(self, data: str) -> None:
        if self._in_p:
            self._current_text.append(data)


def _extract_text(html: str) -> str:
    """Extract plain text from HTML."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        # Fallback: strip tags with regex
        return re.sub(r"<[^>]+>", " ", html)
    return parser.get_text()


def _extract_internal_links(html: str, domain: str) -> List[str]:
    """Extract internal link hrefs from HTML content."""
    parser = _LinkExtractor(domain)
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.internal_hrefs


def _html_word_count(html: str) -> int:
    """Count words in HTML content."""
    text = _extract_text(html)
    words = text.split()
    return len(words)


def _content_hash(content: str) -> str:
    """Generate a short hash of content for change detection."""
    return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Keyword extraction (pure Python, no AI)
# ---------------------------------------------------------------------------


def _extract_keywords_from_text(text: str, max_keywords: int = 20) -> List[str]:
    """
    Extract meaningful keywords from text.

    Tokenizes, lowercases, removes stopwords and short tokens,
    and returns the most frequent terms.
    """
    # Normalize
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = text.split()

    # Filter stopwords and short tokens
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    # Count frequencies
    freq: Dict[str, int] = {}
    for token in filtered:
        freq[token] = freq.get(token, 0) + 1

    # Sort by frequency descending, then alphabetically
    sorted_kw = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [kw for kw, _ in sorted_kw[:max_keywords]]


def _extract_keywords_from_post(
    title: str,
    categories: List[str],
    tags: List[str],
    content_text: str = "",
) -> List[str]:
    """
    Build a keyword list from post metadata and optional content.

    Title and tag words are weighted higher by appearing first in the list.
    """
    seen: Set[str] = set()
    keywords: List[str] = []

    def _add(terms: List[str]) -> None:
        for t in terms:
            t_lower = t.lower().strip()
            if t_lower and t_lower not in seen and t_lower not in STOPWORDS and len(t_lower) > 2:
                seen.add(t_lower)
                keywords.append(t_lower)

    # Title tokens (highest priority)
    title_tokens = re.sub(r"[^a-z0-9\s\-]", " ", title.lower()).split()
    _add(title_tokens)

    # Tags (high priority — authors choose these deliberately)
    for tag in tags:
        tag_tokens = re.sub(r"[^a-z0-9\s\-]", " ", tag.lower()).split()
        _add(tag_tokens)

    # Categories
    for cat in categories:
        cat_tokens = re.sub(r"[^a-z0-9\s\-]", " ", cat.lower()).split()
        _add(cat_tokens)

    # Content keywords (lower priority, fill up to 25)
    if content_text:
        content_kw = _extract_keywords_from_text(content_text, max_keywords=30)
        _add(content_kw)

    return keywords[:25]


# ---------------------------------------------------------------------------
# Async event loop helpers
# ---------------------------------------------------------------------------


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get the running event loop or create a new one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def _run_sync(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=300)
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class PostNode:
    """Represents a single post in the link graph."""

    post_id: int
    title: str
    url: str
    slug: str
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    content_hash: str = ""
    word_count: int = 0
    publish_date: str = ""  # ISO 8601
    internal_links_out: List[int] = field(default_factory=list)
    internal_links_in: List[int] = field(default_factory=list)

    @property
    def incoming_count(self) -> int:
        return len(self.internal_links_in)

    @property
    def outgoing_count(self) -> int:
        return len(self.internal_links_out)

    @property
    def total_links(self) -> int:
        return self.incoming_count + self.outgoing_count

    @property
    def is_orphan(self) -> bool:
        return self.incoming_count == 0

    def publish_datetime(self) -> Optional[datetime]:
        """Parse publish_date to a datetime object."""
        if not self.publish_date:
            return None
        try:
            # WordPress REST API returns ISO 8601 without timezone
            dt = datetime.fromisoformat(self.publish_date.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PostNode:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LinkOpportunity:
    """A suggested internal link between two posts."""

    source_post_id: int
    target_post_id: int
    anchor_text: str
    context_sentence: str
    relevance_score: float  # 0.0 to 1.0
    link_type: str  # contextual, related, pillar-to-supporting, supporting-to-pillar
    target_url: str = ""
    target_title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LinkOpportunity:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LinkGraph:
    """Complete link graph for a single site."""

    site_id: str
    nodes: Dict[int, PostNode] = field(default_factory=dict)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    orphan_pages: List[int] = field(default_factory=list)
    pillar_pages: List[int] = field(default_factory=list)
    clusters: Dict[str, List[int]] = field(default_factory=dict)
    last_updated: str = ""

    @property
    def post_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "nodes": {str(pid): node.to_dict() for pid, node in self.nodes.items()},
            "edges": self.edges,
            "orphan_pages": self.orphan_pages,
            "pillar_pages": self.pillar_pages,
            "clusters": self.clusters,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LinkGraph:
        nodes = {}
        for pid_str, node_data in data.get("nodes", {}).items():
            pid = int(pid_str)
            nodes[pid] = PostNode.from_dict(node_data)
        return cls(
            site_id=data["site_id"],
            nodes=nodes,
            edges=[tuple(e) for e in data.get("edges", [])],
            orphan_pages=data.get("orphan_pages", []),
            pillar_pages=data.get("pillar_pages", []),
            clusters=data.get("clusters", {}),
            last_updated=data.get("last_updated", ""),
        )


@dataclass
class LinkReport:
    """Link health report for a site."""

    site_id: str
    total_posts: int = 0
    total_internal_links: int = 0
    avg_links_per_post: float = 0.0
    orphan_count: int = 0
    over_linked_count: int = 0
    under_linked_count: int = 0
    pillar_count: int = 0
    cluster_count: int = 0
    opportunities: List[LinkOpportunity] = field(default_factory=list)
    health_score: float = 0.0  # 0 to 100

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["opportunities"] = [o.to_dict() for o in self.opportunities]
        return data

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Link Health Report: {self.site_id} ===",
            f"  Total posts:           {self.total_posts}",
            f"  Total internal links:  {self.total_internal_links}",
            f"  Avg links per post:    {self.avg_links_per_post:.1f}",
            f"  Orphan pages:          {self.orphan_count}",
            f"  Over-linked (>20):     {self.over_linked_count}",
            f"  Under-linked (<2):     {self.under_linked_count}",
            f"  Pillar pages:          {self.pillar_count}",
            f"  Content clusters:      {self.cluster_count}",
            f"  Opportunities found:   {len(self.opportunities)}",
            f"  Health score:          {self.health_score:.0f}/100",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# URL-to-post-ID resolution
# ---------------------------------------------------------------------------


def _slug_from_url(url: str) -> str:
    """Extract the slug from a WordPress post URL."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    # Handle trailing slashes and query strings
    parts = path.split("/")
    # The slug is typically the last segment
    slug = parts[-1] if parts else ""
    # Remove common suffixes
    slug = re.sub(r"\.(html?|php|asp)$", "", slug)
    return slug


# ---------------------------------------------------------------------------
# InternalLinker — main engine
# ---------------------------------------------------------------------------


class InternalLinker:
    """
    Internal linking engine for the OpenClaw publishing empire.

    Builds link graphs from WordPress posts, analyzes link structure,
    suggests contextual internal links, and injects them into content.
    All analysis is pure Python — no AI API calls, fast and free.

    Parameters
    ----------
    data_dir : Path, optional
        Directory for cached graph JSON files.
        Defaults to ``data/linker/`` relative to the project root.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._graphs: Dict[str, LinkGraph] = {}
        self._site_configs: Optional[Dict[str, Dict[str, Any]]] = None

    # -- Site config loading ------------------------------------------------

    def _load_site_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load site configs from registry, cached after first call."""
        if self._site_configs is not None:
            return self._site_configs

        if not SITE_REGISTRY_PATH.exists():
            logger.error("Site registry not found at %s", SITE_REGISTRY_PATH)
            self._site_configs = {}
            return self._site_configs

        with open(SITE_REGISTRY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        sites = data.get("sites", [])
        config_map: Dict[str, Dict[str, Any]] = {}
        for site in sites:
            config_map[site["id"]] = site

        self._site_configs = config_map
        return self._site_configs

    def _get_domain(self, site_id: str) -> str:
        """Get the domain for a site ID."""
        configs = self._load_site_configs()
        site = configs.get(site_id)
        if not site:
            raise ValueError(
                f"Site '{site_id}' not found in registry. "
                f"Available: {', '.join(sorted(configs.keys()))}"
            )
        return site["domain"]

    def _get_wp_client(self, site_id: str):
        """
        Lazily import and return a WordPressClient for the given site.

        This avoids a hard circular import — the wordpress_client module
        is only loaded when API calls are actually needed.
        """
        from src.wordpress_client import get_site_client
        return get_site_client(site_id)

    # -- Cache paths --------------------------------------------------------

    def _graph_path(self, site_id: str) -> Path:
        """Path to the cached graph JSON for a site."""
        return self._data_dir / f"{site_id}_graph.json"

    def _cross_site_path(self) -> Path:
        """Path to the cross-site opportunities cache."""
        return self._data_dir / "cross_site_opportunities.json"

    # -- Graph persistence --------------------------------------------------

    def _save_graph(self, graph: LinkGraph) -> None:
        """Save a link graph to the JSON cache."""
        path = self._graph_path(graph.site_id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(graph.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Saved graph for %s to %s (%d nodes)", graph.site_id, path, graph.post_count)

    def _load_graph_from_disk(self, site_id: str) -> Optional[LinkGraph]:
        """Load a link graph from the JSON cache, or None if not found."""
        path = self._graph_path(site_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            graph = LinkGraph.from_dict(data)
            logger.info(
                "Loaded cached graph for %s (%d nodes, updated %s)",
                site_id, graph.post_count, graph.last_updated,
            )
            return graph
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to load cached graph for %s: %s", site_id, exc)
            return None

    # -----------------------------------------------------------------------
    # Graph Building
    # -----------------------------------------------------------------------

    async def build_graph(self, site_id: str, max_posts: int = 500) -> LinkGraph:
        """
        Build (or rebuild) the complete link graph for a site.

        Fetches all published posts via the WP REST API (paginated),
        parses HTML content to extract existing internal links, extracts
        keywords from title + categories + tags, and builds a bidirectional
        edge map.

        Parameters
        ----------
        site_id : str
            Site identifier (e.g., "witchcraft", "smarthome").
        max_posts : int
            Maximum number of posts to fetch. Default 500.

        Returns
        -------
        LinkGraph
            Complete link graph for the site.
        """
        domain = self._get_domain(site_id)
        client = self._get_wp_client(site_id)

        logger.info("Building link graph for %s (%s), max_posts=%d", site_id, domain, max_posts)

        # Fetch categories and tags for name resolution
        categories_list = await client.get_categories()
        tags_list = await client.get_tags()

        cat_map: Dict[int, str] = {c["id"]: c.get("name", "") for c in categories_list}
        tag_map: Dict[int, str] = {t["id"]: t.get("name", "") for t in tags_list}

        # Fetch all published posts (paginated)
        all_posts: List[Dict[str, Any]] = []
        page = 1
        while len(all_posts) < max_posts:
            per_page = min(WP_MAX_PER_PAGE, max_posts - len(all_posts))
            batch = await client.list_posts(
                per_page=per_page,
                page=page,
                status="publish",
                orderby="date",
                order="desc",
            )
            if not batch:
                break
            all_posts.extend(batch)
            logger.debug(
                "Fetched page %d: %d posts (total: %d)", page, len(batch), len(all_posts)
            )
            if len(batch) < per_page:
                break
            page += 1

        logger.info("Fetched %d published posts from %s", len(all_posts), site_id)

        # Build URL-to-post-ID mapping for resolving internal links
        url_to_id: Dict[str, int] = {}
        slug_to_id: Dict[str, int] = {}

        nodes: Dict[int, PostNode] = {}

        for wp_post in all_posts:
            post_id = wp_post["id"]
            title_raw = wp_post.get("title", {})
            title = title_raw.get("rendered", "") if isinstance(title_raw, dict) else str(title_raw)
            # Strip HTML entities from title
            title = re.sub(r"<[^>]+>", "", title)
            title = title.replace("&#8217;", "'").replace("&#8211;", "-").replace("&amp;", "&")

            link = wp_post.get("link", "")
            slug = wp_post.get("slug", "")

            content_raw = wp_post.get("content", {})
            content_html = (
                content_raw.get("rendered", "")
                if isinstance(content_raw, dict)
                else str(content_raw)
            )

            # Resolve category and tag names
            cat_ids = wp_post.get("categories", [])
            tag_ids = wp_post.get("tags", [])
            cat_names = [cat_map.get(cid, "") for cid in cat_ids if cat_map.get(cid)]
            tag_names = [tag_map.get(tid, "") for tid in tag_ids if tag_map.get(tid)]

            # Extract keywords from title + categories + tags
            content_text = _extract_text(content_html)
            keywords = _extract_keywords_from_post(title, cat_names, tag_names, content_text)

            publish_date = wp_post.get("date", "")

            node = PostNode(
                post_id=post_id,
                title=title,
                url=link,
                slug=slug,
                categories=cat_names,
                tags=tag_names,
                keywords=keywords,
                content_hash=_content_hash(content_html),
                word_count=_html_word_count(content_html),
                publish_date=publish_date,
                internal_links_out=[],
                internal_links_in=[],
            )
            nodes[post_id] = node

            # Register URL and slug mappings
            if link:
                url_to_id[link.rstrip("/")] = post_id
                # Also register without trailing slash and with www
                parsed = urlparse(link)
                path_key = parsed.path.rstrip("/")
                url_to_id[path_key] = post_id
            if slug:
                slug_to_id[slug] = post_id

        # Second pass: extract internal links and resolve to post IDs
        edges: List[Tuple[int, int]] = []

        for wp_post in all_posts:
            post_id = wp_post["id"]
            content_raw = wp_post.get("content", {})
            content_html = (
                content_raw.get("rendered", "")
                if isinstance(content_raw, dict)
                else str(content_raw)
            )

            internal_hrefs = _extract_internal_links(content_html, domain)

            linked_ids: Set[int] = set()
            for href in internal_hrefs:
                target_id = self._resolve_href_to_post_id(href, domain, url_to_id, slug_to_id)
                if target_id and target_id != post_id and target_id in nodes:
                    linked_ids.add(target_id)

            if post_id in nodes:
                nodes[post_id].internal_links_out = sorted(linked_ids)
                for target_id in linked_ids:
                    edges.append((post_id, target_id))

        # Build incoming link lists
        for source_id, target_id in edges:
            if target_id in nodes and source_id not in nodes[target_id].internal_links_in:
                nodes[target_id].internal_links_in.append(source_id)

        # Sort incoming lists
        for node in nodes.values():
            node.internal_links_in.sort()

        # Identify orphans and pillars
        orphan_pages = [pid for pid, n in nodes.items() if n.is_orphan]
        pillar_pages = self._compute_pillars(nodes)

        # Cluster posts by category overlap
        clusters = self._compute_clusters(nodes)

        graph = LinkGraph(
            site_id=site_id,
            nodes=nodes,
            edges=edges,
            orphan_pages=orphan_pages,
            pillar_pages=pillar_pages,
            clusters=clusters,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        # Cache in memory and on disk
        self._graphs[site_id] = graph
        self._save_graph(graph)

        logger.info(
            "Built graph for %s: %d nodes, %d edges, %d orphans, %d pillars, %d clusters",
            site_id, graph.post_count, graph.edge_count,
            len(orphan_pages), len(pillar_pages), len(clusters),
        )
        return graph

    def build_graph_sync(self, site_id: str, max_posts: int = 500) -> LinkGraph:
        """Synchronous wrapper for build_graph()."""
        return _run_sync(self.build_graph(site_id, max_posts=max_posts))

    async def update_graph(self, site_id: str, post_ids: List[int]) -> LinkGraph:
        """
        Incrementally update the link graph for specific posts.

        Fetches only the specified posts and updates their nodes in the
        existing graph. Recalculates edges, orphans, and pillars.

        Parameters
        ----------
        site_id : str
            Site identifier.
        post_ids : list of int
            Post IDs to update.

        Returns
        -------
        LinkGraph
            Updated link graph.
        """
        graph = await self.load_graph(site_id)
        if graph is None:
            logger.info("No existing graph for %s, building from scratch", site_id)
            return await self.build_graph(site_id)

        domain = self._get_domain(site_id)
        client = self._get_wp_client(site_id)

        categories_list = await client.get_categories()
        tags_list = await client.get_tags()
        cat_map = {c["id"]: c.get("name", "") for c in categories_list}
        tag_map = {t["id"]: t.get("name", "") for t in tags_list}

        # Build URL/slug maps from existing graph
        url_to_id: Dict[str, int] = {}
        slug_to_id: Dict[str, int] = {}
        for pid, node in graph.nodes.items():
            if node.url:
                url_to_id[node.url.rstrip("/")] = pid
                parsed = urlparse(node.url)
                url_to_id[parsed.path.rstrip("/")] = pid
            if node.slug:
                slug_to_id[node.slug] = pid

        # Fetch and update each post
        for post_id in post_ids:
            try:
                wp_post = await client.list_posts(per_page=1, page=1, status="publish")
                # Use get_post directly via the raw API
                url = f"{client.config.api_url}/posts/{post_id}"
                _, wp_post_data, _ = await client._request("GET", url)
                if not isinstance(wp_post_data, dict):
                    logger.warning("Could not fetch post %d from %s", post_id, site_id)
                    continue
            except Exception as exc:
                logger.warning("Failed to fetch post %d from %s: %s", post_id, site_id, exc)
                continue

            wp_post = wp_post_data
            title_raw = wp_post.get("title", {})
            title = title_raw.get("rendered", "") if isinstance(title_raw, dict) else str(title_raw)
            title = re.sub(r"<[^>]+>", "", title)

            link = wp_post.get("link", "")
            slug = wp_post.get("slug", "")

            content_raw = wp_post.get("content", {})
            content_html = (
                content_raw.get("rendered", "")
                if isinstance(content_raw, dict)
                else str(content_raw)
            )

            cat_ids = wp_post.get("categories", [])
            tag_ids = wp_post.get("tags", [])
            cat_names = [cat_map.get(cid, "") for cid in cat_ids if cat_map.get(cid)]
            tag_names = [tag_map.get(tid, "") for tid in tag_ids if tag_map.get(tid)]

            content_text = _extract_text(content_html)
            keywords = _extract_keywords_from_post(title, cat_names, tag_names, content_text)

            # Remove old outgoing edges from this post
            graph.edges = [(s, t) for s, t in graph.edges if s != post_id]

            # Remove this post from other nodes' incoming lists
            for node in graph.nodes.values():
                if post_id in node.internal_links_in:
                    node.internal_links_in.remove(post_id)

            # Create or update the node
            node = PostNode(
                post_id=post_id,
                title=title,
                url=link,
                slug=slug,
                categories=cat_names,
                tags=tag_names,
                keywords=keywords,
                content_hash=_content_hash(content_html),
                word_count=_html_word_count(content_html),
                publish_date=wp_post.get("date", ""),
                internal_links_out=[],
                internal_links_in=[],
            )

            # Extract and resolve internal links
            internal_hrefs = _extract_internal_links(content_html, domain)
            linked_ids: Set[int] = set()
            for href in internal_hrefs:
                target_id = self._resolve_href_to_post_id(href, domain, url_to_id, slug_to_id)
                if target_id and target_id != post_id and target_id in graph.nodes:
                    linked_ids.add(target_id)

            node.internal_links_out = sorted(linked_ids)
            graph.nodes[post_id] = node

            # Register in URL maps
            if link:
                url_to_id[link.rstrip("/")] = post_id
            if slug:
                slug_to_id[slug] = post_id

            # Add new edges
            for target_id in linked_ids:
                graph.edges.append((post_id, target_id))

        # Rebuild incoming link lists for all nodes
        for node in graph.nodes.values():
            node.internal_links_in = []
        for source_id, target_id in graph.edges:
            if target_id in graph.nodes and source_id not in graph.nodes[target_id].internal_links_in:
                graph.nodes[target_id].internal_links_in.append(source_id)
        for node in graph.nodes.values():
            node.internal_links_in.sort()

        # Recompute derived data
        graph.orphan_pages = [pid for pid, n in graph.nodes.items() if n.is_orphan]
        graph.pillar_pages = self._compute_pillars(graph.nodes)
        graph.clusters = self._compute_clusters(graph.nodes)
        graph.last_updated = datetime.now(timezone.utc).isoformat()

        self._graphs[site_id] = graph
        self._save_graph(graph)

        logger.info("Updated graph for %s: refreshed %d posts", site_id, len(post_ids))
        return graph

    def update_graph_sync(self, site_id: str, post_ids: List[int]) -> LinkGraph:
        """Synchronous wrapper for update_graph()."""
        return _run_sync(self.update_graph(site_id, post_ids))

    async def load_graph(self, site_id: str) -> Optional[LinkGraph]:
        """
        Load the link graph from memory cache or disk.

        Parameters
        ----------
        site_id : str
            Site identifier.

        Returns
        -------
        LinkGraph or None
            The cached graph, or None if no cache exists.
        """
        # Check in-memory cache first
        if site_id in self._graphs:
            return self._graphs[site_id]

        # Try disk cache
        graph = self._load_graph_from_disk(site_id)
        if graph:
            self._graphs[site_id] = graph
        return graph

    def load_graph_sync(self, site_id: str) -> Optional[LinkGraph]:
        """Synchronous wrapper for load_graph()."""
        return _run_sync(self.load_graph(site_id))

    # -- Internal resolution helpers ----------------------------------------

    def _resolve_href_to_post_id(
        self,
        href: str,
        domain: str,
        url_to_id: Dict[str, int],
        slug_to_id: Dict[str, int],
    ) -> Optional[int]:
        """
        Resolve an internal href to a post ID using URL and slug maps.

        Tries full URL match first, then path-only match, then slug match.
        """
        # Normalize
        href_clean = href.rstrip("/")

        # Try full URL match
        pid = url_to_id.get(href_clean)
        if pid:
            return pid

        # Try with https:// prefix
        if not href_clean.startswith("http"):
            full_url = f"https://{domain}{href_clean}" if href_clean.startswith("/") else href_clean
            pid = url_to_id.get(full_url.rstrip("/"))
            if pid:
                return pid

        # Try path-only match
        parsed = urlparse(href_clean)
        path = parsed.path.rstrip("/")
        pid = url_to_id.get(path)
        if pid:
            return pid

        # Try slug match
        slug = _slug_from_url(href_clean)
        if slug:
            pid = slug_to_id.get(slug)
            if pid:
                return pid

        return None

    def _compute_pillars(self, nodes: Dict[int, PostNode]) -> List[int]:
        """Identify pillar pages — posts with the most incoming links."""
        if not nodes:
            return []

        # Sort by incoming count descending
        sorted_nodes = sorted(
            nodes.values(),
            key=lambda n: n.incoming_count,
            reverse=True,
        )

        # Pillar if above threshold or in top 5%
        threshold = max(PILLAR_MIN_INCOMING, 1)
        top_n = max(1, len(sorted_nodes) // 20)  # Top 5%

        pillars: List[int] = []
        for i, node in enumerate(sorted_nodes):
            if node.incoming_count >= threshold or i < top_n:
                pillars.append(node.post_id)
            else:
                break

        return pillars

    def _compute_clusters(self, nodes: Dict[int, PostNode]) -> Dict[str, List[int]]:
        """
        Group posts into content clusters based on primary category.

        Each post is assigned to its first (primary) category. Posts
        without categories go into an "uncategorized" cluster.
        """
        clusters: Dict[str, List[int]] = {}

        for pid, node in nodes.items():
            if node.categories:
                cluster_name = node.categories[0].lower().strip()
            else:
                cluster_name = "uncategorized"

            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(pid)

        # Sort post IDs within each cluster
        for name in clusters:
            clusters[name].sort()

        return clusters

    # -----------------------------------------------------------------------
    # Link Analysis
    # -----------------------------------------------------------------------

    async def find_orphans(self, site_id: str) -> List[PostNode]:
        """
        Find orphan pages — posts with zero incoming internal links.

        Parameters
        ----------
        site_id : str
            Site identifier.

        Returns
        -------
        list of PostNode
            Posts with no incoming links, sorted by publish date descending.
        """
        graph = await self._ensure_graph(site_id)
        orphans = [graph.nodes[pid] for pid in graph.orphan_pages if pid in graph.nodes]
        orphans.sort(key=lambda n: n.publish_date or "", reverse=True)
        return orphans

    def find_orphans_sync(self, site_id: str) -> List[PostNode]:
        """Synchronous wrapper for find_orphans()."""
        return _run_sync(self.find_orphans(site_id))

    async def find_over_linked(
        self, site_id: str, threshold: int = DEFAULT_OVERLINKED_THRESHOLD
    ) -> List[PostNode]:
        """
        Find over-linked pages — posts with too many outgoing links.

        Parameters
        ----------
        site_id : str
            Site identifier.
        threshold : int
            Outgoing link count above which a post is considered over-linked.

        Returns
        -------
        list of PostNode
            Over-linked posts, sorted by outgoing count descending.
        """
        graph = await self._ensure_graph(site_id)
        over = [n for n in graph.nodes.values() if n.outgoing_count > threshold]
        over.sort(key=lambda n: n.outgoing_count, reverse=True)
        return over

    def find_over_linked_sync(
        self, site_id: str, threshold: int = DEFAULT_OVERLINKED_THRESHOLD
    ) -> List[PostNode]:
        """Synchronous wrapper for find_over_linked()."""
        return _run_sync(self.find_over_linked(site_id, threshold=threshold))

    async def find_under_linked(
        self, site_id: str, threshold: int = DEFAULT_UNDERLINKED_THRESHOLD
    ) -> List[PostNode]:
        """
        Find under-linked pages — posts with too few outgoing links.

        Parameters
        ----------
        site_id : str
            Site identifier.
        threshold : int
            Outgoing link count below which a post is considered under-linked.

        Returns
        -------
        list of PostNode
            Under-linked posts, sorted by word count descending (longer posts
            should have more links).
        """
        graph = await self._ensure_graph(site_id)
        under = [n for n in graph.nodes.values() if n.outgoing_count < threshold]
        under.sort(key=lambda n: n.word_count, reverse=True)
        return under

    def find_under_linked_sync(
        self, site_id: str, threshold: int = DEFAULT_UNDERLINKED_THRESHOLD
    ) -> List[PostNode]:
        """Synchronous wrapper for find_under_linked()."""
        return _run_sync(self.find_under_linked(site_id, threshold=threshold))

    async def identify_pillars(self, site_id: str) -> List[PostNode]:
        """
        Identify pillar pages — posts with the highest incoming link counts.

        These are the site's authority/cornerstone pages that cluster posts
        should link to and from.

        Parameters
        ----------
        site_id : str
            Site identifier.

        Returns
        -------
        list of PostNode
            Pillar pages sorted by incoming count descending.
        """
        graph = await self._ensure_graph(site_id)
        pillars = [graph.nodes[pid] for pid in graph.pillar_pages if pid in graph.nodes]
        pillars.sort(key=lambda n: n.incoming_count, reverse=True)
        return pillars

    def identify_pillars_sync(self, site_id: str) -> List[PostNode]:
        """Synchronous wrapper for identify_pillars()."""
        return _run_sync(self.identify_pillars(site_id))

    async def cluster_posts(self, site_id: str) -> Dict[str, List[PostNode]]:
        """
        Group posts into content clusters by primary category.

        Parameters
        ----------
        site_id : str
            Site identifier.

        Returns
        -------
        dict
            Mapping of cluster name to list of PostNode.
        """
        graph = await self._ensure_graph(site_id)
        result: Dict[str, List[PostNode]] = {}
        for name, pids in graph.clusters.items():
            result[name] = [graph.nodes[pid] for pid in pids if pid in graph.nodes]
        return result

    def cluster_posts_sync(self, site_id: str) -> Dict[str, List[PostNode]]:
        """Synchronous wrapper for cluster_posts()."""
        return _run_sync(self.cluster_posts(site_id))

    async def link_health(self, site_id: str) -> LinkReport:
        """
        Generate a comprehensive link health report for a site.

        Calculates link density, orphan rate, over/under-linked counts,
        cluster statistics, and an overall health score from 0-100.

        Parameters
        ----------
        site_id : str
            Site identifier.

        Returns
        -------
        LinkReport
            Full health report with opportunities.
        """
        graph = await self._ensure_graph(site_id)

        total_posts = graph.post_count
        if total_posts == 0:
            return LinkReport(site_id=site_id, health_score=0.0)

        # Count total internal links
        total_links = sum(n.outgoing_count for n in graph.nodes.values())
        avg_links = total_links / total_posts if total_posts > 0 else 0.0

        orphans = [n for n in graph.nodes.values() if n.is_orphan]
        over_linked = [n for n in graph.nodes.values() if n.outgoing_count > DEFAULT_OVERLINKED_THRESHOLD]
        under_linked = [n for n in graph.nodes.values() if n.outgoing_count < DEFAULT_UNDERLINKED_THRESHOLD]

        # Generate top opportunities (for orphans and under-linked posts)
        opportunities: List[LinkOpportunity] = []
        priority_posts = sorted(
            orphans + under_linked,
            key=lambda n: (n.is_orphan, n.word_count),
            reverse=True,
        )
        # Deduplicate
        seen_ids: Set[int] = set()
        deduped: List[PostNode] = []
        for node in priority_posts:
            if node.post_id not in seen_ids:
                seen_ids.add(node.post_id)
                deduped.append(node)

        for node in deduped[:20]:  # Top 20 priority posts
            suggestions = await self.suggest_links(site_id, node.post_id, max_suggestions=3)
            opportunities.extend(suggestions)

        # Calculate health score (0-100)
        health_score = self._calculate_health_score(
            total_posts=total_posts,
            orphan_count=len(orphans),
            over_linked_count=len(over_linked),
            under_linked_count=len(under_linked),
            avg_links=avg_links,
        )

        report = LinkReport(
            site_id=site_id,
            total_posts=total_posts,
            total_internal_links=total_links,
            avg_links_per_post=round(avg_links, 2),
            orphan_count=len(orphans),
            over_linked_count=len(over_linked),
            under_linked_count=len(under_linked),
            pillar_count=len(graph.pillar_pages),
            cluster_count=len(graph.clusters),
            opportunities=opportunities,
            health_score=round(health_score, 1),
        )

        logger.info("Link health for %s: score=%.0f/100", site_id, health_score)
        return report

    def link_health_sync(self, site_id: str) -> LinkReport:
        """Synchronous wrapper for link_health()."""
        return _run_sync(self.link_health(site_id))

    def _calculate_health_score(
        self,
        total_posts: int,
        orphan_count: int,
        over_linked_count: int,
        under_linked_count: int,
        avg_links: float,
    ) -> float:
        """
        Calculate a 0-100 health score based on link structure quality.

        Factors:
        - Orphan penalty: more orphans = lower score
        - Under-linked penalty: too few links = lower score
        - Over-linked penalty: too many links = slightly lower score
        - Average links bonus: sweet spot is 3-8 links per post
        """
        if total_posts == 0:
            return 0.0

        score = 100.0

        # Orphan penalty: -2 points per percent of orphan pages
        orphan_pct = (orphan_count / total_posts) * 100
        score -= orphan_pct * 2.0

        # Under-linked penalty: -1 point per percent of under-linked pages
        under_pct = (under_linked_count / total_posts) * 100
        score -= under_pct * 1.0

        # Over-linked penalty: -0.5 points per percent of over-linked pages
        over_pct = (over_linked_count / total_posts) * 100
        score -= over_pct * 0.5

        # Average links bonus/penalty
        # Sweet spot: 3-8 links per post
        if avg_links < 1.0:
            score -= 15.0
        elif avg_links < 2.0:
            score -= 8.0
        elif avg_links < 3.0:
            score -= 3.0
        elif avg_links > 15.0:
            score -= 5.0
        elif avg_links > 10.0:
            score -= 2.0
        # 3-10 is great, no penalty

        return max(0.0, min(100.0, score))

    # -----------------------------------------------------------------------
    # Link Suggestions
    # -----------------------------------------------------------------------

    async def suggest_links(
        self,
        site_id: str,
        post_id: int,
        max_suggestions: int = 10,
    ) -> List[LinkOpportunity]:
        """
        Suggest internal links for an existing post.

        Scores potential targets by keyword overlap, category match,
        recency, and current link deficit. Returns the best candidates
        with suggested anchor text.

        Parameters
        ----------
        site_id : str
            Site identifier.
        post_id : int
            The post to find link targets for.
        max_suggestions : int
            Maximum number of suggestions. Default 10.

        Returns
        -------
        list of LinkOpportunity
            Ranked link suggestions, highest relevance first.
        """
        graph = await self._ensure_graph(site_id)

        source = graph.nodes.get(post_id)
        if not source:
            logger.warning("Post %d not found in graph for %s", post_id, site_id)
            return []

        # Exclude posts already linked
        already_linked = set(source.internal_links_out)
        candidates: List[Tuple[float, PostNode]] = []

        for pid, target in graph.nodes.items():
            if pid == post_id or pid in already_linked:
                continue

            relevance = self._combined_relevance(source, target)
            if relevance > 0.05:  # Minimum threshold
                candidates.append((relevance, target))

        # Sort by relevance descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        opportunities: List[LinkOpportunity] = []
        for relevance, target in candidates[:max_suggestions]:
            # Determine link type
            link_type = self._determine_link_type(source, target, graph)

            # Generate anchor text suggestions
            anchors = self.suggest_anchor_text(source.title, target.title, target.keywords)
            anchor = anchors[0] if anchors else target.title

            # Find a context sentence (first sentence in source that mentions
            # a keyword from the target)
            context = self._find_context_sentence(source, target)

            opportunities.append(LinkOpportunity(
                source_post_id=post_id,
                target_post_id=target.post_id,
                anchor_text=anchor,
                context_sentence=context,
                relevance_score=round(relevance, 3),
                link_type=link_type,
                target_url=target.url,
                target_title=target.title,
            ))

        return opportunities

    def suggest_links_sync(
        self, site_id: str, post_id: int, max_suggestions: int = 10
    ) -> List[LinkOpportunity]:
        """Synchronous wrapper for suggest_links()."""
        return _run_sync(self.suggest_links(site_id, post_id, max_suggestions=max_suggestions))

    async def suggest_links_for_content(
        self,
        site_id: str,
        title: str,
        content_html: str,
        keywords: Optional[List[str]] = None,
        max_suggestions: int = 10,
    ) -> List[LinkOpportunity]:
        """
        Suggest internal links for NEW content that has not been published yet.

        Finds the best existing posts to link to from the new content and
        identifies contextual insertion points within the HTML.

        Parameters
        ----------
        site_id : str
            Site identifier.
        title : str
            Title of the new content.
        content_html : str
            HTML body of the new content.
        keywords : list of str, optional
            Explicit keywords. If omitted, extracted from title and content.
        max_suggestions : int
            Maximum suggestions. Default 10.

        Returns
        -------
        list of LinkOpportunity
            Ranked suggestions with anchor text and context for insertion.
        """
        graph = await self._ensure_graph(site_id)

        # Build a virtual PostNode for the new content
        content_text = _extract_text(content_html)
        if keywords:
            kw_list = []
            for kw in keywords:
                for part in kw.split(","):
                    part = part.strip().lower()
                    if part:
                        kw_list.append(part)
            derived = _extract_keywords_from_post(title, [], [], content_text)
            # Merge explicit keywords first
            seen = set(kw_list)
            for d in derived:
                if d not in seen:
                    kw_list.append(d)
                    seen.add(d)
        else:
            kw_list = _extract_keywords_from_post(title, [], [], content_text)

        virtual_node = PostNode(
            post_id=0,
            title=title,
            url="",
            slug="",
            categories=[],
            tags=[],
            keywords=kw_list,
            content_hash="",
            word_count=_html_word_count(content_html),
            publish_date=datetime.now(timezone.utc).isoformat(),
            internal_links_out=[],
            internal_links_in=[],
        )

        candidates: List[Tuple[float, PostNode]] = []
        for pid, target in graph.nodes.items():
            relevance = self._combined_relevance(virtual_node, target)
            if relevance > 0.05:
                candidates.append((relevance, target))

        candidates.sort(key=lambda x: x[0], reverse=True)

        # Find insertion points in the new content
        content_lower = content_text.lower()
        opportunities: List[LinkOpportunity] = []

        for relevance, target in candidates[:max_suggestions]:
            anchors = self.suggest_anchor_text(title, target.title, target.keywords)
            anchor = anchors[0] if anchors else target.title

            # Try to find a natural insertion point
            context = ""
            for kw in target.keywords[:5]:
                idx = content_lower.find(kw)
                if idx >= 0:
                    # Extract surrounding sentence
                    start = max(0, content_lower.rfind(".", 0, idx) + 1)
                    end = content_lower.find(".", idx)
                    if end < 0:
                        end = min(len(content_lower), idx + 100)
                    context = content_text[start:end + 1].strip()
                    if len(context) > 200:
                        context = context[:200] + "..."
                    break

            link_type = "contextual" if context else "related"
            # Check if target is a pillar page
            if target.post_id in graph.pillar_pages:
                link_type = "supporting-to-pillar"

            opportunities.append(LinkOpportunity(
                source_post_id=0,
                target_post_id=target.post_id,
                anchor_text=anchor,
                context_sentence=context,
                relevance_score=round(relevance, 3),
                link_type=link_type,
                target_url=target.url,
                target_title=target.title,
            ))

        return opportunities

    def suggest_links_for_content_sync(
        self,
        site_id: str,
        title: str,
        content_html: str,
        keywords: Optional[List[str]] = None,
        max_suggestions: int = 10,
    ) -> List[LinkOpportunity]:
        """Synchronous wrapper for suggest_links_for_content()."""
        return _run_sync(
            self.suggest_links_for_content(
                site_id, title, content_html, keywords=keywords,
                max_suggestions=max_suggestions,
            )
        )

    def suggest_anchor_text(
        self,
        source_title: str,
        target_title: str,
        target_keywords: List[str],
    ) -> List[str]:
        """
        Generate 3-5 natural anchor text options for a link.

        Returns a ranked list from most natural to most generic.

        Parameters
        ----------
        source_title : str
            Title of the linking (source) post.
        target_title : str
            Title of the linked (target) post.
        target_keywords : list of str
            Keywords associated with the target post.

        Returns
        -------
        list of str
            Anchor text suggestions, best first.
        """
        anchors: List[str] = []
        seen: Set[str] = set()

        def _add(text: str) -> None:
            text = text.strip()
            key = text.lower()
            if key and key not in seen and len(text) > 2:
                seen.add(key)
                anchors.append(text)

        # Option 1: Extract core phrase from target title (remove common patterns)
        core = re.sub(
            r"^(how to|the ultimate|a complete|beginner'?s?|your|the|a|an)\s+",
            "",
            target_title.lower(),
            flags=re.IGNORECASE,
        ).strip()
        core = re.sub(
            r"\s+(guide|tutorial|tips|tricks|review|explained|101|for beginners)$",
            "",
            core,
            flags=re.IGNORECASE,
        ).strip()
        if core:
            _add(core)

        # Option 2: First 2-3 target keywords joined
        if len(target_keywords) >= 2:
            phrase = " ".join(target_keywords[:2])
            _add(phrase)
        if len(target_keywords) >= 3:
            phrase = " ".join(target_keywords[:3])
            _add(phrase)

        # Option 3: Full target title (exact match anchor)
        if len(target_title) <= 60:
            _add(target_title)

        # Option 4: Short version of target title (first N words)
        words = target_title.split()
        if len(words) > 4:
            _add(" ".join(words[:4]))

        # Option 5: Most specific keyword
        for kw in target_keywords:
            if len(kw) > 3:
                _add(kw)
                break

        return anchors[:5]

    # -----------------------------------------------------------------------
    # Link Injection
    # -----------------------------------------------------------------------

    def inject_links(
        self,
        content_html: str,
        opportunities: List[LinkOpportunity],
        max_links: int = 5,
    ) -> str:
        """
        Inject internal links into HTML content.

        Finds natural insertion points where anchor text matches text in
        paragraphs and wraps the first occurrence in an <a> tag. Respects
        max_links limit and will not double-link to the same URL.

        Parameters
        ----------
        content_html : str
            The HTML content to inject links into.
        opportunities : list of LinkOpportunity
            Link opportunities with anchor text and target URLs.
        max_links : int
            Maximum number of links to inject. Default 5.

        Returns
        -------
        str
            Modified HTML content with links injected.
        """
        if not opportunities or not content_html:
            return content_html

        # Track URLs already present in the content
        existing_links = set()
        link_pattern = re.compile(r'<a\s[^>]*href=["\']([^"\']+)["\']', re.IGNORECASE)
        for match in link_pattern.finditer(content_html):
            existing_links.add(match.group(1).rstrip("/"))

        injected_count = 0
        result = content_html

        for opp in opportunities:
            if injected_count >= max_links:
                break

            if not opp.target_url or not opp.anchor_text:
                continue

            # Skip if this URL is already linked
            target_clean = opp.target_url.rstrip("/")
            if target_clean in existing_links:
                logger.debug("Skipping already-linked URL: %s", target_clean)
                continue

            # Find the anchor text in the content (case-insensitive, first match only)
            anchor = opp.anchor_text
            anchor_escaped = re.escape(anchor)

            # Match the anchor text only when it's NOT inside an HTML tag or existing link
            # Pattern: anchor text that is NOT preceded by <a...> and NOT inside a tag
            pattern = re.compile(
                r'(?<![<>/"\'])(?<!</a>)'  # Not inside a tag
                r'(\b' + anchor_escaped + r'\b)'
                r'(?![^<]*>)',  # Not followed by a tag close before next tag open
                re.IGNORECASE,
            )

            match = pattern.search(result)
            if match:
                original_text = match.group(1)
                link_html = f'<a href="{opp.target_url}">{original_text}</a>'
                # Replace only the first occurrence
                result = result[:match.start(1)] + link_html + result[match.end(1):]
                existing_links.add(target_clean)
                injected_count += 1
                logger.debug(
                    "Injected link: '%s' -> %s", original_text, opp.target_url
                )
            else:
                logger.debug(
                    "Anchor text '%s' not found in content for %s", anchor, opp.target_url
                )

        logger.info("Injected %d/%d links into content", injected_count, len(opportunities))
        return result

    async def inject_related_posts_section(
        self, site_id: str, post_id: int, count: int = 3
    ) -> str:
        """
        Generate a "Related Posts" HTML section for the end of an article.

        Parameters
        ----------
        site_id : str
            Site identifier.
        post_id : int
            The post to find related posts for.
        count : int
            Number of related posts to include. Default 3.

        Returns
        -------
        str
            HTML block for a related posts section.
        """
        graph = await self._ensure_graph(site_id)
        source = graph.nodes.get(post_id)
        if not source:
            return ""

        # Get suggestions sorted by relevance
        suggestions = await self.suggest_links(site_id, post_id, max_suggestions=count * 2)

        # Filter to best matches
        related = suggestions[:count]
        if not related:
            return ""

        # Build HTML
        lines = [
            '<div class="related-posts" style="margin-top: 2em; padding: 1.5em; '
            'border-top: 2px solid #eee;">',
            '  <h3 style="margin-top: 0;">You Might Also Enjoy</h3>',
            '  <ul style="list-style: none; padding: 0;">',
        ]

        for opp in related:
            title = opp.target_title or opp.anchor_text
            url = opp.target_url
            lines.append(
                f'    <li style="margin-bottom: 0.75em;">'
                f'<a href="{url}" style="text-decoration: none; font-weight: 500;">'
                f'{title}</a></li>'
            )

        lines.append('  </ul>')
        lines.append('</div>')

        return "\n".join(lines)

    def inject_related_posts_section_sync(
        self, site_id: str, post_id: int, count: int = 3
    ) -> str:
        """Synchronous wrapper for inject_related_posts_section()."""
        return _run_sync(self.inject_related_posts_section(site_id, post_id, count=count))

    # -----------------------------------------------------------------------
    # Cross-Site Linking
    # -----------------------------------------------------------------------

    async def find_cross_site_opportunities(
        self,
        site_a: str,
        site_b: str,
        max_results: int = 20,
    ) -> List[LinkOpportunity]:
        """
        Find topically related posts between two sites.

        Useful for the witchcraft sub-niche sites that share overlapping
        topics (crystals, herbs, moon phases, tarot, etc.).

        Parameters
        ----------
        site_a : str
            First site identifier.
        site_b : str
            Second site identifier.
        max_results : int
            Maximum number of opportunities. Default 20.

        Returns
        -------
        list of LinkOpportunity
            Cross-site linking opportunities sorted by relevance.
        """
        graph_a = await self._ensure_graph(site_a)
        graph_b = await self._ensure_graph(site_b)

        if not graph_a or not graph_b:
            return []

        candidates: List[Tuple[float, PostNode, PostNode]] = []

        # Compare all posts from site A against site B
        for node_a in graph_a.nodes.values():
            for node_b in graph_b.nodes.values():
                overlap = self._keyword_overlap_score(node_a, node_b)
                cat_score = self._category_match_score(node_a, node_b)
                combined = overlap * 0.6 + cat_score * 0.4

                if combined > 0.10:  # Higher threshold for cross-site
                    candidates.append((combined, node_a, node_b))

        candidates.sort(key=lambda x: x[0], reverse=True)

        opportunities: List[LinkOpportunity] = []
        for relevance, node_a, node_b in candidates[:max_results]:
            anchors = self.suggest_anchor_text(node_a.title, node_b.title, node_b.keywords)
            anchor = anchors[0] if anchors else node_b.title

            opportunities.append(LinkOpportunity(
                source_post_id=node_a.post_id,
                target_post_id=node_b.post_id,
                anchor_text=anchor,
                context_sentence=f"Cross-site: {site_a} -> {site_b}",
                relevance_score=round(relevance, 3),
                link_type="contextual",
                target_url=node_b.url,
                target_title=node_b.title,
            ))

        # Cache results
        self._save_cross_site_opportunities(site_a, site_b, opportunities)

        return opportunities

    def find_cross_site_opportunities_sync(
        self, site_a: str, site_b: str, max_results: int = 20
    ) -> List[LinkOpportunity]:
        """Synchronous wrapper for find_cross_site_opportunities()."""
        return _run_sync(
            self.find_cross_site_opportunities(site_a, site_b, max_results=max_results)
        )

    def _save_cross_site_opportunities(
        self, site_a: str, site_b: str, opportunities: List[LinkOpportunity]
    ) -> None:
        """Append cross-site opportunities to the shared cache."""
        path = self._cross_site_path()
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except (json.JSONDecodeError, TypeError):
                existing = {}

        key = f"{site_a}_to_{site_b}"
        existing[key] = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "count": len(opportunities),
            "opportunities": [o.to_dict() for o in opportunities],
        }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Keyword / Scoring (internal, no AI needed)
    # -----------------------------------------------------------------------

    def _keyword_overlap_score(self, post_a: PostNode, post_b: PostNode) -> float:
        """
        Jaccard similarity on keyword sets.

        Returns a value between 0.0 (no overlap) and 1.0 (identical keywords).
        """
        set_a = set(post_a.keywords)
        set_b = set(post_b.keywords)

        if not set_a or not set_b:
            return 0.0

        intersection = set_a & set_b
        union = set_a | set_b

        return len(intersection) / len(union)

    def _category_match_score(self, post_a: PostNode, post_b: PostNode) -> float:
        """
        Score based on shared categories.

        Returns 1.0 if all categories match, 0.0 if none match.
        """
        cats_a = set(c.lower() for c in post_a.categories)
        cats_b = set(c.lower() for c in post_b.categories)

        if not cats_a or not cats_b:
            return 0.0

        intersection = cats_a & cats_b
        union = cats_a | cats_b

        return len(intersection) / len(union)

    def _recency_score(self, post: PostNode) -> float:
        """
        Score based on post age. Newer posts score higher.

        Returns 1.0 for posts published today, decaying to 0.0 over
        RECENCY_MAX_DAYS.
        """
        dt = post.publish_datetime()
        if not dt:
            return 0.3  # Default for unknown dates

        now = datetime.now(timezone.utc)
        age_days = (now - dt).total_seconds() / 86400

        if age_days <= 0:
            return 1.0
        if age_days >= RECENCY_MAX_DAYS:
            return 0.0

        # Exponential decay
        return math.exp(-3.0 * age_days / RECENCY_MAX_DAYS)

    def _link_need_score(self, post: PostNode) -> float:
        """
        Score based on how much a post needs more incoming links.

        Posts with zero incoming links get the highest score. Score
        decreases as incoming link count increases.
        """
        incoming = post.incoming_count
        if incoming == 0:
            return 1.0
        if incoming == 1:
            return 0.8
        if incoming <= 3:
            return 0.5
        if incoming <= 5:
            return 0.3
        return 0.1

    def _combined_relevance(self, post_a: PostNode, post_b: PostNode) -> float:
        """
        Weighted combination of all scoring factors.

        Produces a single 0-1 relevance score for linking post_a to post_b.
        """
        kw_score = self._keyword_overlap_score(post_a, post_b)
        cat_score = self._category_match_score(post_a, post_b)
        recency = self._recency_score(post_b)
        need = self._link_need_score(post_b)

        combined = (
            WEIGHT_KEYWORD_OVERLAP * kw_score
            + WEIGHT_CATEGORY_MATCH * cat_score
            + WEIGHT_RECENCY * recency
            + WEIGHT_LINK_NEED * need
        )

        return min(1.0, combined)

    # -- Internal helpers ---------------------------------------------------

    def _determine_link_type(
        self, source: PostNode, target: PostNode, graph: LinkGraph
    ) -> str:
        """Determine the semantic type of a link between two posts."""
        source_is_pillar = source.post_id in graph.pillar_pages
        target_is_pillar = target.post_id in graph.pillar_pages

        if source_is_pillar and not target_is_pillar:
            return "pillar-to-supporting"
        if not source_is_pillar and target_is_pillar:
            return "supporting-to-pillar"
        if self._keyword_overlap_score(source, target) > 0.3:
            return "contextual"
        return "related"

    def _find_context_sentence(self, source: PostNode, target: PostNode) -> str:
        """
        Find a sentence in the source that mentions a keyword from the target.

        Uses the source title and keywords since we do not store full content
        in the graph (content is hashed). Returns a description of the match.
        """
        # Since we do not store full content in the graph, we find overlap keywords
        common_kw = set(source.keywords) & set(target.keywords)
        if common_kw:
            kw_str = ", ".join(sorted(common_kw)[:3])
            return f"Shared topics: {kw_str}"

        # Fallback: describe the category/tag relationship
        common_cats = set(c.lower() for c in source.categories) & set(c.lower() for c in target.categories)
        if common_cats:
            return f"Same category: {', '.join(sorted(common_cats))}"

        common_tags = set(t.lower() for t in source.tags) & set(t.lower() for t in target.tags)
        if common_tags:
            tag_str = ", ".join(sorted(common_tags)[:3])
            return f"Shared tags: {tag_str}"

        return "Topically related content"

    async def _ensure_graph(self, site_id: str) -> LinkGraph:
        """Load graph from cache or raise if not available."""
        graph = await self.load_graph(site_id)
        if graph is None:
            raise ValueError(
                f"No link graph found for '{site_id}'. "
                f"Run `build_graph('{site_id}')` first, or use the CLI: "
                f"python -m src.internal_linker build --site {site_id}"
            )
        return graph


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_linker_instance: Optional[InternalLinker] = None


def get_linker(data_dir: Optional[Path] = None) -> InternalLinker:
    """
    Get or create the singleton InternalLinker instance.

    Parameters
    ----------
    data_dir : Path, optional
        Data directory override. Only used on first call.

    Returns
    -------
    InternalLinker
    """
    global _linker_instance
    if _linker_instance is None:
        _linker_instance = InternalLinker(data_dir=data_dir)
    return _linker_instance


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="internal_linker",
        description="Internal Linking Engine for the OpenClaw Publishing Empire",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # build
    p_build = subparsers.add_parser("build", help="Build or rebuild the link graph for a site")
    p_build.add_argument("--site", required=True, help="Site ID")
    p_build.add_argument("--max-posts", type=int, default=500, help="Max posts to fetch (default: 500)")

    # health
    p_health = subparsers.add_parser("health", help="Generate a link health report")
    p_health.add_argument("--site", required=True, help="Site ID")

    # orphans
    p_orphans = subparsers.add_parser("orphans", help="List orphan pages (no incoming links)")
    p_orphans.add_argument("--site", required=True, help="Site ID")

    # suggest
    p_suggest = subparsers.add_parser("suggest", help="Suggest internal links for a post")
    p_suggest.add_argument("--site", required=True, help="Site ID")
    p_suggest.add_argument("--post-id", type=int, required=True, help="Post ID")
    p_suggest.add_argument("--max", type=int, default=10, help="Max suggestions (default: 10)")

    # suggest-new
    p_new = subparsers.add_parser("suggest-new", help="Suggest links for new (unpublished) content")
    p_new.add_argument("--site", required=True, help="Site ID")
    p_new.add_argument("--title", required=True, help="Article title")
    p_new.add_argument("--keywords", default="", help="Comma-separated keywords")
    p_new.add_argument("--content-file", default=None, help="Path to HTML content file")
    p_new.add_argument("--max", type=int, default=10, help="Max suggestions (default: 10)")

    # inject
    p_inject = subparsers.add_parser("inject", help="Inject links into a post")
    p_inject.add_argument("--site", required=True, help="Site ID")
    p_inject.add_argument("--post-id", type=int, required=True, help="Post ID")
    p_inject.add_argument("--max-links", type=int, default=5, help="Max links to inject (default: 5)")
    p_inject.add_argument("--dry-run", action="store_true", help="Preview without applying changes")

    # pillars
    p_pillars = subparsers.add_parser("pillars", help="Identify pillar/cornerstone pages")
    p_pillars.add_argument("--site", required=True, help="Site ID")

    # clusters
    p_clusters = subparsers.add_parser("clusters", help="Show post clusters by category")
    p_clusters.add_argument("--site", required=True, help="Site ID")

    return parser


async def _run_cli(args: argparse.Namespace) -> None:
    """Execute the CLI command."""
    linker = get_linker()

    if args.command == "build":
        print(f"Building link graph for {args.site}...")
        graph = await linker.build_graph(args.site, max_posts=args.max_posts)
        print(f"Done. {graph.post_count} posts, {graph.edge_count} edges.")
        print(f"  Orphan pages: {len(graph.orphan_pages)}")
        print(f"  Pillar pages: {len(graph.pillar_pages)}")
        print(f"  Clusters: {len(graph.clusters)}")
        print(f"  Graph saved to: {linker._graph_path(args.site)}")

    elif args.command == "health":
        print(f"Analyzing link health for {args.site}...")
        report = await linker.link_health(args.site)
        print()
        print(report.summary())
        if report.opportunities:
            print(f"\n  Top opportunities:")
            for i, opp in enumerate(report.opportunities[:10], 1):
                print(
                    f"    {i}. [{opp.link_type}] Post {opp.source_post_id} -> "
                    f"Post {opp.target_post_id} (score: {opp.relevance_score:.2f})"
                )
                print(f"       Anchor: \"{opp.anchor_text}\"")
                if opp.target_title:
                    print(f"       Target: {opp.target_title}")

    elif args.command == "orphans":
        orphans = await linker.find_orphans(args.site)
        print(f"Found {len(orphans)} orphan pages on {args.site}:\n")
        for node in orphans:
            print(f"  [{node.post_id}] {node.title}")
            print(f"    URL: {node.url}")
            print(f"    Published: {node.publish_date[:10] if node.publish_date else 'unknown'}")
            print(f"    Words: {node.word_count}, Outgoing links: {node.outgoing_count}")
            print()

    elif args.command == "suggest":
        suggestions = await linker.suggest_links(args.site, args.post_id, max_suggestions=args.max)
        if not suggestions:
            print(f"No link suggestions found for post {args.post_id} on {args.site}.")
            return

        print(f"Link suggestions for post {args.post_id} on {args.site}:\n")
        for i, opp in enumerate(suggestions, 1):
            print(f"  {i}. [{opp.link_type}] -> Post {opp.target_post_id} (score: {opp.relevance_score:.2f})")
            print(f"     Target: {opp.target_title}")
            print(f"     URL: {opp.target_url}")
            print(f"     Anchor: \"{opp.anchor_text}\"")
            print(f"     Context: {opp.context_sentence}")
            print()

    elif args.command == "suggest-new":
        # Load content from file if provided
        content_html = ""
        if args.content_file:
            content_path = Path(args.content_file)
            if content_path.exists():
                content_html = content_path.read_text(encoding="utf-8")
            else:
                print(f"Content file not found: {args.content_file}")
                return

        keywords = [kw.strip() for kw in args.keywords.split(",") if kw.strip()] if args.keywords else None

        suggestions = await linker.suggest_links_for_content(
            args.site, args.title, content_html,
            keywords=keywords, max_suggestions=args.max,
        )

        if not suggestions:
            print(f"No link suggestions found for \"{args.title}\" on {args.site}.")
            return

        print(f"Link suggestions for new article \"{args.title}\" on {args.site}:\n")
        for i, opp in enumerate(suggestions, 1):
            print(f"  {i}. [{opp.link_type}] -> Post {opp.target_post_id} (score: {opp.relevance_score:.2f})")
            print(f"     Target: {opp.target_title}")
            print(f"     URL: {opp.target_url}")
            print(f"     Anchor: \"{opp.anchor_text}\"")
            if opp.context_sentence:
                print(f"     Insert near: {opp.context_sentence[:100]}")
            print()

    elif args.command == "inject":
        graph = await linker.load_graph(args.site)
        if not graph:
            print(f"No graph for {args.site}. Run 'build --site {args.site}' first.")
            return

        # Fetch the post content
        client = linker._get_wp_client(args.site)
        url = f"{client.config.api_url}/posts/{args.post_id}"
        _, wp_post, _ = await client._request("GET", url)
        if not isinstance(wp_post, dict):
            print(f"Could not fetch post {args.post_id}.")
            return

        content_raw = wp_post.get("content", {})
        content_html = (
            content_raw.get("rendered", "")
            if isinstance(content_raw, dict)
            else str(content_raw)
        )
        title = wp_post.get("title", {})
        title_text = title.get("rendered", "") if isinstance(title, dict) else str(title)

        # Get suggestions
        suggestions = await linker.suggest_links(args.site, args.post_id, max_suggestions=args.max_links * 2)
        if not suggestions:
            print(f"No linking opportunities found for post {args.post_id}.")
            return

        # Inject links
        modified = linker.inject_links(content_html, suggestions, max_links=args.max_links)

        if args.dry_run:
            print(f"DRY RUN: Would inject links into post {args.post_id} \"{title_text}\"")
            print(f"\nSuggested links:")
            for i, opp in enumerate(suggestions[:args.max_links], 1):
                print(f"  {i}. \"{opp.anchor_text}\" -> {opp.target_url}")
            print(f"\nModified content preview (first 500 chars):")
            print(modified[:500])
            if modified != content_html:
                print(f"\n[Content was modified]")
            else:
                print(f"\n[No changes made — anchor text not found in content]")
        else:
            if modified == content_html:
                print("No links could be injected (anchor text not found in content).")
                return

            # Update the post
            await client._request(
                "POST",
                f"{client.config.api_url}/posts/{args.post_id}",
                json_data={"content": modified},
            )
            print(f"Updated post {args.post_id} with injected internal links.")

    elif args.command == "pillars":
        pillars = await linker.identify_pillars(args.site)
        if not pillars:
            print(f"No pillar pages identified for {args.site}.")
            return

        print(f"Pillar pages for {args.site} ({len(pillars)} found):\n")
        for node in pillars:
            print(f"  [{node.post_id}] {node.title}")
            print(f"    URL: {node.url}")
            print(f"    Incoming links: {node.incoming_count}")
            print(f"    Outgoing links: {node.outgoing_count}")
            print(f"    Words: {node.word_count}")
            print()

    elif args.command == "clusters":
        clusters = await linker.cluster_posts(args.site)
        if not clusters:
            print(f"No clusters found for {args.site}.")
            return

        print(f"Content clusters for {args.site} ({len(clusters)} clusters):\n")
        for name, nodes in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {name} ({len(nodes)} posts)")
            for node in nodes[:5]:
                print(f"    [{node.post_id}] {node.title}")
            if len(nodes) > 5:
                print(f"    ... and {len(nodes) - 5} more")
            print()

    else:
        print("No command specified. Use --help for available commands.")


def main() -> None:
    """CLI entry point."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        _run_sync(_run_cli(args))
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("CLI error: %s", exc)
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
