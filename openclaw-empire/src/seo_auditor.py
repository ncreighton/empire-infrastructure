"""
SEO Auditor — OpenClaw Empire Edition
======================================

Crawls and audits all 16 WordPress sites for SEO health. Checks meta
descriptions, title tags, heading hierarchy, internal links, broken links,
thin content, keyword cannibalization, image alt text, page speed indicators,
schema markup, and more. Generates prioritized fix lists and weekly reports.

All sites use RankMath Pro for SEO; the auditor reads RankMath-specific
REST API fields (focus keyword, SEO score, meta description) alongside
standard WP REST API post data.

Usage:
    from src.seo_auditor import get_auditor

    auditor = get_auditor()
    audit = await auditor.crawl_site("witchcraft", max_pages=200)
    report = auditor.generate_report(site_id="witchcraft")

CLI:
    python -m src.seo_auditor audit --site witchcraft
    python -m src.seo_auditor audit --all
    python -m src.seo_auditor post --site witchcraft --post-id 1234
    python -m src.seo_auditor report --period week
    python -m src.seo_auditor issues --site witchcraft --severity critical
    python -m src.seo_auditor cannibalization --site witchcraft
    python -m src.seo_auditor action-items --count 20
    python -m src.seo_auditor history --site witchcraft
    python -m src.seo_auditor score --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("seo_auditor")
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
DATA_DIR = BASE_DIR / "data" / "seo"
AUDITS_DIR = DATA_DIR / "audits"
ISSUES_PATH = DATA_DIR / "issues.json"
HISTORY_PATH = DATA_DIR / "history.json"

# Ensure data directories exist on import
AUDITS_DIR.mkdir(parents=True, exist_ok=True)

# Anthropic model for fix suggestions (Haiku per cost optimization rules)
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# Crawl tuning
DEFAULT_MAX_PAGES = 200
CONCURRENT_SITES = 3
CONCURRENT_LINK_CHECKS = 10
REQUEST_TIMEOUT = 15  # seconds per request
SLOW_RESPONSE_THRESHOLD = 3.0  # seconds
WP_API_PER_PAGE = 100

# Content thresholds
THIN_CONTENT_WORDS = 300
SHORT_CONTENT_WORDS = 1000
MIN_META_DESC_LENGTH = 50
MAX_META_DESC_LENGTH = 160
MIN_INTERNAL_LINKS = 3
FIRST_PARAGRAPH_WORDS = 100

# Scoring weights
SCORE_DEDUCTION_CRITICAL = 5
SCORE_DEDUCTION_WARNING = 2
SCORE_DEDUCTION_INFO = 0.5
SCORE_BASE = 100

# Issue types for reference
ISSUE_TYPES = [
    "missing_meta_description",
    "long_meta_description",
    "missing_focus_keyword",
    "keyword_not_in_title",
    "keyword_not_in_first_paragraph",
    "thin_content",
    "short_content",
    "missing_h1",
    "multiple_h1",
    "broken_heading_hierarchy",
    "images_missing_alt",
    "no_internal_links",
    "few_internal_links",
    "no_external_links",
    "broken_links",
    "missing_schema",
    "keyword_cannibalization",
    "duplicate_meta_descriptions",
    "duplicate_titles",
    "no_featured_image",
    "slow_response",
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SEOIssue:
    """A single SEO issue found during an audit."""

    issue_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    site_id: str = ""
    post_id: Optional[int] = None
    url: str = ""
    issue_type: str = ""
    severity: str = "warning"  # "critical", "warning", "info"
    description: str = ""
    recommendation: str = ""
    auto_fixable: bool = False
    fixed: bool = False
    found_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    fixed_at: Optional[str] = None


@dataclass
class PageAudit:
    """Audit results for a single page/post."""

    url: str = ""
    post_id: Optional[int] = None
    title: str = ""
    meta_description: str = ""
    meta_description_length: int = 0
    has_focus_keyword: bool = False
    focus_keyword: str = ""
    heading_structure: Dict[str, int] = field(
        default_factory=lambda: {
            "h1_count": 0,
            "h2_count": 0,
            "h3_count": 0,
            "h4_count": 0,
            "h5_count": 0,
            "h6_count": 0,
        }
    )
    word_count: int = 0
    internal_links_count: int = 0
    external_links_count: int = 0
    images_without_alt: List[str] = field(default_factory=list)
    has_schema: bool = False
    schema_types: List[str] = field(default_factory=list)
    has_featured_image: bool = False
    response_time: float = 0.0
    issues: List[SEOIssue] = field(default_factory=list)


@dataclass
class SiteAudit:
    """Audit results for an entire site."""

    site_id: str = ""
    domain: str = ""
    total_pages: int = 0
    audit_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pages_audited: int = 0
    issues_by_severity: Dict[str, int] = field(
        default_factory=lambda: {"critical": 0, "warning": 0, "info": 0}
    )
    issues_by_type: Dict[str, int] = field(default_factory=dict)
    overall_score: int = 100
    top_issues: List[Dict[str, Any]] = field(default_factory=list)
    pages: List[PageAudit] = field(default_factory=list)


@dataclass
class SEOReport:
    """Aggregated SEO report across sites and time periods."""

    period: str = "week"
    report_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sites_audited: List[str] = field(default_factory=list)
    total_issues: int = 0
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    improvements_since_last: int = 0
    worst_sites: List[Dict[str, Any]] = field(default_factory=list)
    best_sites: List[Dict[str, Any]] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTML Parser — stdlib-only, no BeautifulSoup
# ---------------------------------------------------------------------------


class SEOHTMLParser(HTMLParser):
    """
    Parses HTML content and extracts SEO-relevant data:
    headings, links, images, word count, schema markup.
    """

    def __init__(self, base_domain: str = "") -> None:
        super().__init__()
        self.base_domain = base_domain.lower().rstrip("/")

        # Headings
        self.headings: List[Tuple[str, str]] = []  # [(tag, text), ...]
        self._current_heading: Optional[str] = None
        self._heading_text: List[str] = []

        # Links
        self.internal_links: List[str] = []
        self.external_links: List[str] = []
        self._current_link_href: Optional[str] = None

        # Images
        self.images: List[Dict[str, str]] = []  # [{"src": ..., "alt": ...}, ...]

        # Text content
        self._text_parts: List[str] = []
        self._in_script = False
        self._in_style = False

        # Schema (JSON-LD)
        self.schema_blocks: List[Dict[str, Any]] = []
        self._in_jsonld = False
        self._jsonld_data: List[str] = []

        # Tag depth tracking for heading hierarchy check
        self.heading_sequence: List[int] = []  # e.g. [1, 2, 3, 2, 3]

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}

        # Headings
        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._current_heading = tag_lower
            self._heading_text = []
            level = int(tag_lower[1])
            self.heading_sequence.append(level)

        # Links
        elif tag_lower == "a":
            href = attrs_dict.get("href", "")
            if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                self._current_link_href = href
                self._classify_link(href)

        # Images
        elif tag_lower == "img":
            src = attrs_dict.get("src", attrs_dict.get("data-src", ""))
            alt = attrs_dict.get("alt", "")
            self.images.append({"src": src, "alt": alt})

        # JSON-LD schema
        elif tag_lower == "script":
            script_type = attrs_dict.get("type", "")
            if script_type == "application/ld+json":
                self._in_jsonld = True
                self._jsonld_data = []

        elif tag_lower == "script":
            self._in_script = True
        elif tag_lower == "style":
            self._in_style = True

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()

        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6") and self._current_heading:
            text = " ".join(self._heading_text).strip()
            self.headings.append((self._current_heading, text))
            self._current_heading = None
            self._heading_text = []

        elif tag_lower == "script":
            if self._in_jsonld:
                raw = "".join(self._jsonld_data).strip()
                if raw:
                    try:
                        parsed = json.loads(raw)
                        self.schema_blocks.append(parsed)
                    except (json.JSONDecodeError, ValueError):
                        pass
                self._in_jsonld = False
                self._jsonld_data = []
            self._in_script = False

        elif tag_lower == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_jsonld:
            self._jsonld_data.append(data)
            return

        if self._in_script or self._in_style:
            return

        if self._current_heading is not None:
            self._heading_text.append(data)

        self._text_parts.append(data)

    def _classify_link(self, href: str) -> None:
        """Classify a link as internal or external based on domain."""
        if not self.base_domain:
            self.external_links.append(href)
            return

        parsed = urlparse(href)
        link_domain = parsed.netloc.lower().rstrip("/")

        # Relative links are internal
        if not link_domain:
            self.internal_links.append(href)
            return

        # Strip www. for comparison
        clean_base = self.base_domain.replace("www.", "")
        clean_link = link_domain.replace("www.", "")

        if clean_link == clean_base or clean_link.endswith("." + clean_base):
            self.internal_links.append(href)
        else:
            self.external_links.append(href)

    @property
    def text_content(self) -> str:
        """All visible text from the parsed HTML."""
        return " ".join(self._text_parts)

    @property
    def word_count(self) -> int:
        """Approximate word count of visible text."""
        words = re.findall(r"\b\w+\b", self.text_content)
        return len(words)

    @property
    def first_paragraph_text(self) -> str:
        """Extract roughly the first N words of visible text."""
        words = re.findall(r"\b\w+\b", self.text_content)
        return " ".join(words[:FIRST_PARAGRAPH_WORDS]).lower()

    def get_heading_counts(self) -> Dict[str, int]:
        """Return heading counts by level."""
        counts: Dict[str, int] = {
            "h1_count": 0,
            "h2_count": 0,
            "h3_count": 0,
            "h4_count": 0,
            "h5_count": 0,
            "h6_count": 0,
        }
        for tag, _ in self.headings:
            key = f"{tag}_count"
            if key in counts:
                counts[key] += 1
        return counts

    def get_schema_types(self) -> List[str]:
        """Extract @type values from all JSON-LD schema blocks."""
        types: List[str] = []
        for block in self.schema_blocks:
            if isinstance(block, dict):
                schema_type = block.get("@type")
                if isinstance(schema_type, str):
                    types.append(schema_type)
                elif isinstance(schema_type, list):
                    types.extend(schema_type)
                # Handle @graph arrays
                graph = block.get("@graph")
                if isinstance(graph, list):
                    for item in graph:
                        if isinstance(item, dict):
                            st = item.get("@type")
                            if isinstance(st, str):
                                types.append(st)
                            elif isinstance(st, list):
                                types.extend(st)
        return types

    def check_heading_hierarchy(self) -> bool:
        """
        Returns True if heading hierarchy is valid (no skipped levels).
        e.g. H1 -> H3 without H2 is invalid.
        """
        if not self.heading_sequence:
            return True

        for i in range(1, len(self.heading_sequence)):
            current = self.heading_sequence[i]
            previous = self.heading_sequence[i - 1]
            # It is fine to go same level or up (lower number).
            # Going down more than one level is a hierarchy break.
            if current > previous + 1:
                return False
        return True


# ---------------------------------------------------------------------------
# Site Registry Loader (lightweight, avoids circular import)
# ---------------------------------------------------------------------------


@dataclass
class _SiteInfo:
    """Minimal site info for the auditor (no heavy WP client dependency)."""

    site_id: str
    domain: str
    wp_user: str = ""
    app_password: str = ""
    priority: int = 99

    @property
    def api_url(self) -> str:
        return f"https://{self.domain}/wp-json/wp/v2"

    @property
    def auth_tuple(self) -> Optional[Tuple[str, str]]:
        if self.wp_user and self.app_password:
            return (self.wp_user, self.app_password)
        return None

    @property
    def is_configured(self) -> bool:
        return bool(self.wp_user and self.app_password)


def _load_sites() -> Dict[str, _SiteInfo]:
    """Load site registry into lightweight _SiteInfo objects."""
    if not SITE_REGISTRY_PATH.exists():
        logger.error("Site registry not found at %s", SITE_REGISTRY_PATH)
        return {}

    with open(SITE_REGISTRY_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    sites: Dict[str, _SiteInfo] = {}
    for entry in data.get("sites", []):
        site_id = entry.get("id", "")
        app_password = ""
        env_var = entry.get("wp_app_password_env", "")
        if env_var:
            app_password = os.getenv(env_var, "")

        sites[site_id] = _SiteInfo(
            site_id=site_id,
            domain=entry["domain"],
            wp_user=entry.get("wp_user", ""),
            app_password=app_password,
            priority=entry.get("priority", 99),
        )

    return sites


# ---------------------------------------------------------------------------
# Persistence Helpers
# ---------------------------------------------------------------------------


def _save_json(path: Path, data: Any) -> None:
    """Atomically write JSON to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from a file, returning default if missing."""
    if not path.exists():
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _serialize_audit(audit: SiteAudit) -> Dict[str, Any]:
    """Convert SiteAudit to a JSON-serializable dict."""
    result = asdict(audit)
    # Ensure nested dataclasses are also dicts
    pages = []
    for page in result.get("pages", []):
        if isinstance(page, dict):
            issues = []
            for issue in page.get("issues", []):
                if isinstance(issue, dict):
                    issues.append(issue)
                else:
                    issues.append(asdict(issue))
            page["issues"] = issues
            pages.append(page)
        else:
            pages.append(asdict(page))
    result["pages"] = pages
    return result


# ---------------------------------------------------------------------------
# SEO Auditor Core
# ---------------------------------------------------------------------------


class SEOAuditor:
    """
    Main SEO auditing engine for the OpenClaw Empire.

    Crawls WordPress sites via REST API, analyses content for SEO issues,
    detects cross-post problems like keyword cannibalization, scores sites,
    and generates prioritised reports.
    """

    def __init__(self) -> None:
        self._sites = _load_sites()
        self._active_issues: Dict[str, List[SEOIssue]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = _load_json(
            HISTORY_PATH, default={}
        )
        self._load_active_issues()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _load_active_issues(self) -> None:
        """Load active (unfixed) issues from disk."""
        raw = _load_json(ISSUES_PATH, default={})
        for site_id, issue_list in raw.items():
            self._active_issues[site_id] = [
                SEOIssue(**iss) for iss in issue_list if not iss.get("fixed", False)
            ]

    def _save_active_issues(self) -> None:
        """Persist active issues to disk."""
        serialised: Dict[str, List[Dict[str, Any]]] = {}
        for site_id, issues in self._active_issues.items():
            serialised[site_id] = [asdict(i) for i in issues]
        _save_json(ISSUES_PATH, serialised)

    def _save_history(self) -> None:
        """Persist audit history to disk."""
        _save_json(HISTORY_PATH, self._history)

    def _record_history(self, audit: SiteAudit) -> None:
        """Append a summary entry to the audit history for trending."""
        if audit.site_id not in self._history:
            self._history[audit.site_id] = []
        entry = {
            "date": audit.audit_date,
            "score": audit.overall_score,
            "pages_audited": audit.pages_audited,
            "issues_by_severity": dict(audit.issues_by_severity),
            "total_issues": sum(audit.issues_by_severity.values()),
        }
        self._history[audit.site_id].append(entry)
        # Keep at most 365 entries per site
        if len(self._history[audit.site_id]) > 365:
            self._history[audit.site_id] = self._history[audit.site_id][-365:]
        self._save_history()

    def _get_site(self, site_id: str) -> _SiteInfo:
        """Get site info by ID, raising KeyError if not found."""
        if site_id not in self._sites:
            raise KeyError(
                f"Site '{site_id}' not found in registry. "
                f"Available: {', '.join(sorted(self._sites.keys()))}"
            )
        return self._sites[site_id]

    def _make_issue(
        self,
        site_id: str,
        post_id: Optional[int],
        url: str,
        issue_type: str,
        severity: str,
        description: str,
        recommendation: str,
        auto_fixable: bool = False,
    ) -> SEOIssue:
        """Factory for creating an SEOIssue with standard fields."""
        return SEOIssue(
            site_id=site_id,
            post_id=post_id,
            url=url,
            issue_type=issue_type,
            severity=severity,
            description=description,
            recommendation=recommendation,
            auto_fixable=auto_fixable,
        )

    # -----------------------------------------------------------------------
    # Analysis: individual check functions
    # -----------------------------------------------------------------------

    def check_meta_description(
        self, post_data: Dict[str, Any], site_id: str
    ) -> List[SEOIssue]:
        """Check meta description presence and length."""
        issues: List[SEOIssue] = []
        post_id = post_data.get("id")
        url = post_data.get("link", "")

        # RankMath stores meta in yoast_head_json or rankmath fields
        meta_desc = ""
        # Try RankMath REST fields
        rm = post_data.get("rankmath", {})
        if isinstance(rm, dict):
            meta_desc = rm.get("description", "")
        # Fallback: excerpt
        if not meta_desc:
            excerpt = post_data.get("excerpt", {})
            if isinstance(excerpt, dict):
                meta_desc = _strip_html(excerpt.get("rendered", ""))
            elif isinstance(excerpt, str):
                meta_desc = _strip_html(excerpt)

        meta_len = len(meta_desc.strip())

        if meta_len == 0:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "missing_meta_description",
                    "critical",
                    "No meta description set for this post.",
                    "Add a compelling meta description between 50-160 characters "
                    "that includes the focus keyword.",
                    auto_fixable=True,
                )
            )
        elif meta_len < MIN_META_DESC_LENGTH:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "missing_meta_description",
                    "warning",
                    f"Meta description is too short ({meta_len} chars, minimum {MIN_META_DESC_LENGTH}).",
                    "Expand the meta description to at least 50 characters with "
                    "relevant keywords and a call to action.",
                    auto_fixable=True,
                )
            )
        elif meta_len > MAX_META_DESC_LENGTH:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "long_meta_description",
                    "warning",
                    f"Meta description is too long ({meta_len} chars, max {MAX_META_DESC_LENGTH}). "
                    "It will be truncated in search results.",
                    "Shorten the meta description to 160 characters or fewer.",
                    auto_fixable=True,
                )
            )

        return issues

    def check_headings(
        self, content_html: str, site_id: str, post_id: Optional[int], url: str
    ) -> List[SEOIssue]:
        """Check heading hierarchy: H1 count, level skips."""
        issues: List[SEOIssue] = []
        parser = SEOHTMLParser()
        parser.feed(content_html)

        counts = parser.get_heading_counts()

        if counts["h1_count"] == 0:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "missing_h1",
                    "critical",
                    "No H1 heading found in the content.",
                    "Add exactly one H1 heading — typically the post title "
                    "should be rendered as H1 by the theme.",
                )
            )
        elif counts["h1_count"] > 1:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "multiple_h1",
                    "warning",
                    f"Multiple H1 headings found ({counts['h1_count']}). "
                    "There should be exactly one H1 per page.",
                    "Convert extra H1 tags to H2 or lower.",
                )
            )

        if not parser.check_heading_hierarchy():
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "broken_heading_hierarchy",
                    "warning",
                    "Heading hierarchy has skipped levels (e.g. H1 -> H3 without H2). "
                    "This hurts accessibility and SEO.",
                    "Restructure headings so levels are sequential: "
                    "H1 -> H2 -> H3, not H1 -> H3.",
                )
            )

        return issues

    def check_keyword_usage(
        self, post_data: Dict[str, Any], content_html: str, site_id: str
    ) -> List[SEOIssue]:
        """Check focus keyword presence in title and first paragraph."""
        issues: List[SEOIssue] = []
        post_id = post_data.get("id")
        url = post_data.get("link", "")

        # Extract focus keyword from RankMath fields
        focus_keyword = ""
        rm = post_data.get("rankmath", {})
        if isinstance(rm, dict):
            focus_keyword = rm.get("focus_keyword", "")
        # Some RankMath REST integrations use a meta field
        meta = post_data.get("meta", {})
        if isinstance(meta, dict) and not focus_keyword:
            focus_keyword = meta.get("rank_math_focus_keyword", "")

        if not focus_keyword:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "missing_focus_keyword",
                    "critical",
                    "No RankMath focus keyword is set for this post.",
                    "Set a primary focus keyword in RankMath's SEO meta box. "
                    "Choose a keyword with search volume that matches user intent.",
                )
            )
            return issues

        keyword_lower = focus_keyword.lower().strip()

        # Check title
        title_raw = post_data.get("title", {})
        if isinstance(title_raw, dict):
            title_text = _strip_html(title_raw.get("rendered", "")).lower()
        else:
            title_text = str(title_raw).lower()

        if keyword_lower not in title_text:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "keyword_not_in_title",
                    "warning",
                    f"Focus keyword '{focus_keyword}' is not in the title.",
                    "Include the focus keyword naturally in the post title.",
                )
            )

        # Check first paragraph
        parser = SEOHTMLParser()
        parser.feed(content_html)
        first_para = parser.first_paragraph_text

        if keyword_lower not in first_para:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "keyword_not_in_first_paragraph",
                    "warning",
                    f"Focus keyword '{focus_keyword}' is not in the first "
                    f"{FIRST_PARAGRAPH_WORDS} words.",
                    "Mention the focus keyword naturally within the opening "
                    "paragraph for better on-page SEO.",
                )
            )

        return issues

    def check_content_length(
        self,
        content_html: str,
        post_data: Dict[str, Any],
        site_id: str,
    ) -> List[SEOIssue]:
        """Check for thin or short content."""
        issues: List[SEOIssue] = []
        post_id = post_data.get("id")
        url = post_data.get("link", "")

        parser = SEOHTMLParser()
        parser.feed(content_html)
        wc = parser.word_count

        if wc < THIN_CONTENT_WORDS:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "thin_content",
                    "critical",
                    f"Content is only {wc} words (minimum {THIN_CONTENT_WORDS}). "
                    "Thin content is unlikely to rank.",
                    "Expand the article to at least 300 words. Aim for 1,000+ "
                    "words for competitive keywords.",
                )
            )
        elif wc < SHORT_CONTENT_WORDS:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "short_content",
                    "warning",
                    f"Content is {wc} words — below the 1,000-word target "
                    "for articles aiming to rank.",
                    "Consider expanding with additional sections, examples, "
                    "or FAQ content.",
                )
            )

        return issues

    def check_internal_links(
        self, content_html: str, domain: str, site_id: str,
        post_id: Optional[int], url: str
    ) -> List[SEOIssue]:
        """Check internal and external link counts."""
        issues: List[SEOIssue] = []

        parser = SEOHTMLParser(base_domain=domain)
        parser.feed(content_html)

        internal_count = len(parser.internal_links)
        external_count = len(parser.external_links)

        if internal_count == 0:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "no_internal_links",
                    "critical",
                    "No internal links found in this post. Internal linking "
                    "is essential for topical authority and crawlability.",
                    "Add at least 3 internal links to related posts on the site.",
                )
            )
        elif internal_count < MIN_INTERNAL_LINKS:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "few_internal_links",
                    "warning",
                    f"Only {internal_count} internal link(s) found "
                    f"(recommended: {MIN_INTERNAL_LINKS}+).",
                    "Add more internal links to build content clusters.",
                )
            )

        if external_count == 0:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "no_external_links",
                    "info",
                    "No outbound links to external sources. Adding authoritative "
                    "external links can boost E-E-A-T signals.",
                    "Link to 1-3 authoritative external sources to demonstrate "
                    "research and build trust.",
                )
            )

        return issues

    def check_images(
        self, content_html: str, site_id: str, post_id: Optional[int], url: str
    ) -> List[SEOIssue]:
        """Check images for alt text."""
        issues: List[SEOIssue] = []

        parser = SEOHTMLParser()
        parser.feed(content_html)

        missing_alt = [
            img["src"]
            for img in parser.images
            if not img.get("alt", "").strip()
        ]

        if missing_alt:
            count = len(missing_alt)
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "images_missing_alt",
                    "warning",
                    f"{count} image(s) missing alt text. Alt text is critical "
                    "for accessibility and image SEO.",
                    "Add descriptive alt text to all images, including the "
                    "focus keyword where natural.",
                    auto_fixable=True,
                )
            )

        return issues

    def check_schema(
        self, content_html: str, post_data: Dict[str, Any], site_id: str
    ) -> List[SEOIssue]:
        """Check for structured data / schema markup."""
        issues: List[SEOIssue] = []
        post_id = post_data.get("id")
        url = post_data.get("link", "")

        parser = SEOHTMLParser()
        parser.feed(content_html)

        if not parser.schema_blocks:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "missing_schema",
                    "info",
                    "No JSON-LD schema markup detected. Schema helps search "
                    "engines understand your content and can enable rich snippets.",
                    "Add BlogPosting schema at minimum. Consider HowTo or "
                    "FAQPage schema for guides and FAQ sections.",
                )
            )

        return issues

    def check_featured_image(
        self, post_data: Dict[str, Any], site_id: str
    ) -> List[SEOIssue]:
        """Check if the post has a featured image."""
        issues: List[SEOIssue] = []
        post_id = post_data.get("id")
        url = post_data.get("link", "")

        featured_media = post_data.get("featured_media", 0)
        if not featured_media:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "no_featured_image",
                    "warning",
                    "No featured image set. Posts without featured images "
                    "get lower click-through rates in search and social.",
                    "Add a high-quality featured image (1200x630) with "
                    "descriptive alt text.",
                    auto_fixable=True,
                )
            )

        return issues

    def detect_cannibalization(
        self, site_id: str, posts: List[Dict[str, Any]]
    ) -> List[SEOIssue]:
        """
        Detect keyword cannibalization: multiple posts targeting the same
        focus keyword on a single site.
        """
        issues: List[SEOIssue] = []

        keyword_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for post in posts:
            focus_keyword = ""
            rm = post.get("rankmath", {})
            if isinstance(rm, dict):
                focus_keyword = rm.get("focus_keyword", "")
            meta = post.get("meta", {})
            if isinstance(meta, dict) and not focus_keyword:
                focus_keyword = meta.get("rank_math_focus_keyword", "")

            if focus_keyword:
                keyword_map[focus_keyword.lower().strip()].append(post)

        for keyword, matching_posts in keyword_map.items():
            if len(matching_posts) > 1:
                urls = [p.get("link", f"post #{p.get('id', '?')}") for p in matching_posts]
                post_ids = [p.get("id") for p in matching_posts]
                for post in matching_posts:
                    issues.append(
                        self._make_issue(
                            site_id,
                            post.get("id"),
                            post.get("link", ""),
                            "keyword_cannibalization",
                            "critical",
                            f"Keyword '{keyword}' is targeted by {len(matching_posts)} posts: "
                            + ", ".join(urls[:5]),
                            "Consolidate these posts into one comprehensive article, "
                            "or differentiate their focus keywords to avoid competing "
                            "with yourself in search results.",
                        )
                    )

        return issues

    def detect_duplicate_metas(
        self, site_id: str, posts: List[Dict[str, Any]]
    ) -> List[SEOIssue]:
        """Detect duplicate meta descriptions across posts on a site."""
        issues: List[SEOIssue] = []

        meta_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for post in posts:
            meta_desc = ""
            rm = post.get("rankmath", {})
            if isinstance(rm, dict):
                meta_desc = rm.get("description", "")
            if not meta_desc:
                excerpt = post.get("excerpt", {})
                if isinstance(excerpt, dict):
                    meta_desc = _strip_html(excerpt.get("rendered", ""))
                elif isinstance(excerpt, str):
                    meta_desc = _strip_html(excerpt)

            meta_desc = meta_desc.strip()
            if meta_desc and len(meta_desc) > 20:  # ignore very short/empty
                meta_map[meta_desc.lower()].append(post)

        for meta_text, matching_posts in meta_map.items():
            if len(matching_posts) > 1:
                urls = [p.get("link", f"post #{p.get('id', '?')}") for p in matching_posts]
                for post in matching_posts:
                    issues.append(
                        self._make_issue(
                            site_id,
                            post.get("id"),
                            post.get("link", ""),
                            "duplicate_meta_descriptions",
                            "warning",
                            f"Duplicate meta description shared with {len(matching_posts) - 1} "
                            f"other post(s): " + ", ".join(urls[:5]),
                            "Write a unique meta description for each post.",
                            auto_fixable=True,
                        )
                    )

        return issues

    def detect_duplicate_titles(
        self, site_id: str, posts: List[Dict[str, Any]]
    ) -> List[SEOIssue]:
        """Detect posts with identical or near-identical titles."""
        issues: List[SEOIssue] = []

        title_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for post in posts:
            title_raw = post.get("title", {})
            if isinstance(title_raw, dict):
                title_text = _strip_html(title_raw.get("rendered", ""))
            else:
                title_text = str(title_raw)

            normalised = re.sub(r"\s+", " ", title_text.strip().lower())
            if normalised:
                title_map[normalised].append(post)

        for title_text, matching_posts in title_map.items():
            if len(matching_posts) > 1:
                urls = [p.get("link", f"post #{p.get('id', '?')}") for p in matching_posts]
                for post in matching_posts:
                    issues.append(
                        self._make_issue(
                            site_id,
                            post.get("id"),
                            post.get("link", ""),
                            "duplicate_titles",
                            "warning",
                            f"Duplicate title shared with {len(matching_posts) - 1} "
                            f"other post(s): " + ", ".join(urls[:5]),
                            "Give each post a unique, descriptive title that "
                            "targets its specific angle or keyword.",
                        )
                    )

        return issues

    async def check_broken_links(
        self,
        content_html: str,
        domain: str,
        site_id: str,
        post_id: Optional[int],
        url: str,
    ) -> List[SEOIssue]:
        """
        Verify links in content with HEAD requests.
        Only checks a sample to avoid hammering external sites.
        """
        issues: List[SEOIssue] = []
        if aiohttp is None:
            logger.warning("aiohttp not available; skipping broken link checks")
            return issues

        parser = SEOHTMLParser(base_domain=domain)
        parser.feed(content_html)

        all_links = parser.internal_links + parser.external_links
        # Normalise relative links
        normalised: List[str] = []
        for href in all_links:
            if href.startswith("/"):
                normalised.append(f"https://{domain}{href}")
            elif href.startswith("http"):
                normalised.append(href)
            else:
                normalised.append(f"https://{domain}/{href}")

        # Deduplicate and limit
        unique_links = list(dict.fromkeys(normalised))[:50]

        if not unique_links:
            return issues

        sem = asyncio.Semaphore(CONCURRENT_LINK_CHECKS)
        broken: List[str] = []

        async def _check_one(link: str) -> None:
            try:
                async with sem:
                    timeout = aiohttp.ClientTimeout(total=10)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.head(
                            link, allow_redirects=True, ssl=False
                        ) as resp:
                            if resp.status >= 400:
                                broken.append(f"{link} (HTTP {resp.status})")
            except Exception:
                broken.append(f"{link} (connection error)")

        await asyncio.gather(*[_check_one(link) for link in unique_links])

        if broken:
            issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    url,
                    "broken_links",
                    "critical",
                    f"{len(broken)} broken link(s) found: " + "; ".join(broken[:10]),
                    "Fix or remove broken links. Broken links hurt user "
                    "experience and crawl budget.",
                )
            )

        return issues

    # -----------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------

    def calculate_site_score(self, audit: SiteAudit) -> int:
        """
        Calculate overall site SEO score (0-100).

        Deductions:
        - Critical issues: -5 each
        - Warning issues:  -2 each
        - Info issues:     -0.5 each

        Starting score: 100, minimum: 0.
        """
        score = float(SCORE_BASE)

        critical = audit.issues_by_severity.get("critical", 0)
        warnings = audit.issues_by_severity.get("warning", 0)
        infos = audit.issues_by_severity.get("info", 0)

        score -= critical * SCORE_DEDUCTION_CRITICAL
        score -= warnings * SCORE_DEDUCTION_WARNING
        score -= infos * SCORE_DEDUCTION_INFO

        return max(0, min(100, int(score)))

    # -----------------------------------------------------------------------
    # Page-level audit
    # -----------------------------------------------------------------------

    async def _audit_single_post(
        self,
        site: _SiteInfo,
        post_data: Dict[str, Any],
        check_links: bool = False,
    ) -> PageAudit:
        """Run all checks on a single post and return a PageAudit."""
        content_rendered = ""
        content_raw = post_data.get("content", {})
        if isinstance(content_raw, dict):
            content_rendered = content_raw.get("rendered", "")
        elif isinstance(content_raw, str):
            content_rendered = content_raw

        post_id = post_data.get("id")
        url = post_data.get("link", "")

        # Parse content once for metrics
        parser = SEOHTMLParser(base_domain=site.domain)
        parser.feed(content_rendered)

        # Title
        title_raw = post_data.get("title", {})
        if isinstance(title_raw, dict):
            title_text = _strip_html(title_raw.get("rendered", ""))
        else:
            title_text = str(title_raw)

        # Meta description
        meta_desc = ""
        rm = post_data.get("rankmath", {})
        if isinstance(rm, dict):
            meta_desc = rm.get("description", "")
        if not meta_desc:
            excerpt = post_data.get("excerpt", {})
            if isinstance(excerpt, dict):
                meta_desc = _strip_html(excerpt.get("rendered", ""))

        # Focus keyword
        focus_keyword = ""
        if isinstance(rm, dict):
            focus_keyword = rm.get("focus_keyword", "")
        meta_fields = post_data.get("meta", {})
        if isinstance(meta_fields, dict) and not focus_keyword:
            focus_keyword = meta_fields.get("rank_math_focus_keyword", "")

        # Build PageAudit
        page = PageAudit(
            url=url,
            post_id=post_id,
            title=title_text,
            meta_description=meta_desc,
            meta_description_length=len(meta_desc),
            has_focus_keyword=bool(focus_keyword),
            focus_keyword=focus_keyword,
            heading_structure=parser.get_heading_counts(),
            word_count=parser.word_count,
            internal_links_count=len(parser.internal_links),
            external_links_count=len(parser.external_links),
            images_without_alt=[
                img["src"] for img in parser.images if not img.get("alt", "").strip()
            ],
            has_schema=bool(parser.schema_blocks),
            schema_types=parser.get_schema_types(),
            has_featured_image=bool(post_data.get("featured_media", 0)),
        )

        # Collect issues from all checks
        all_issues: List[SEOIssue] = []
        all_issues.extend(self.check_meta_description(post_data, site.site_id))
        all_issues.extend(
            self.check_headings(content_rendered, site.site_id, post_id, url)
        )
        all_issues.extend(
            self.check_keyword_usage(post_data, content_rendered, site.site_id)
        )
        all_issues.extend(
            self.check_content_length(content_rendered, post_data, site.site_id)
        )
        all_issues.extend(
            self.check_internal_links(
                content_rendered, site.domain, site.site_id, post_id, url
            )
        )
        all_issues.extend(
            self.check_images(content_rendered, site.site_id, post_id, url)
        )
        all_issues.extend(
            self.check_schema(content_rendered, post_data, site.site_id)
        )
        all_issues.extend(self.check_featured_image(post_data, site.site_id))

        if check_links:
            broken_issues = await self.check_broken_links(
                content_rendered, site.domain, site.site_id, post_id, url
            )
            all_issues.extend(broken_issues)

        page.issues = all_issues
        return page

    async def audit_post(self, site_id: str, post_id: int) -> PageAudit:
        """
        Deep audit of a single post by ID.

        Fetches the post via WP REST API and runs all checks including
        broken link verification.
        """
        if aiohttp is None:
            raise ImportError("aiohttp is required: pip install aiohttp")

        site = self._get_site(site_id)
        if not site.is_configured:
            raise ValueError(f"Site '{site_id}' has no API credentials configured.")

        api_url = f"{site.api_url}/posts/{post_id}"
        headers = {}
        auth = None
        if site.auth_tuple:
            auth = aiohttp.BasicAuth(*site.auth_tuple)

        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            start = time.monotonic()
            async with session.get(api_url, auth=auth, ssl=False) as resp:
                elapsed = time.monotonic() - start
                if resp.status != 200:
                    raise ValueError(
                        f"Failed to fetch post {post_id} from {site_id}: "
                        f"HTTP {resp.status}"
                    )
                post_data = await resp.json()

        page = await self._audit_single_post(site, post_data, check_links=True)
        page.response_time = elapsed

        # Check slow response
        if elapsed > SLOW_RESPONSE_THRESHOLD:
            page.issues.append(
                self._make_issue(
                    site_id,
                    post_id,
                    page.url,
                    "slow_response",
                    "warning",
                    f"Page took {elapsed:.1f}s to respond "
                    f"(threshold: {SLOW_RESPONSE_THRESHOLD}s).",
                    "Check LiteSpeed Cache configuration, image optimization, "
                    "and server performance.",
                )
            )

        logger.info(
            "Audited post %d on %s: %d issues found",
            post_id,
            site_id,
            len(page.issues),
        )
        return page

    # -----------------------------------------------------------------------
    # Site-level crawl
    # -----------------------------------------------------------------------

    async def crawl_site(
        self, site_id: str, max_pages: int = DEFAULT_MAX_PAGES
    ) -> SiteAudit:
        """
        Full site audit: fetches posts via WP REST API (paginated),
        runs all checks, detects cross-post issues, scores the site.
        """
        if aiohttp is None:
            raise ImportError("aiohttp is required: pip install aiohttp")

        site = self._get_site(site_id)
        if not site.is_configured:
            raise ValueError(f"Site '{site_id}' has no API credentials configured.")

        logger.info("Starting crawl of %s (max %d pages)...", site.domain, max_pages)

        # Fetch all posts (paginated)
        all_posts: List[Dict[str, Any]] = []
        page_num = 1
        auth = aiohttp.BasicAuth(*site.auth_tuple) if site.auth_tuple else None
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            while len(all_posts) < max_pages:
                per_page = min(WP_API_PER_PAGE, max_pages - len(all_posts))
                api_url = (
                    f"{site.api_url}/posts?"
                    f"per_page={per_page}&page={page_num}"
                    f"&status=publish&_fields=id,title,content,excerpt,link,"
                    f"featured_media,meta,date"
                )
                try:
                    start = time.monotonic()
                    async with session.get(api_url, auth=auth, ssl=False) as resp:
                        elapsed = time.monotonic() - start
                        if resp.status == 400:
                            # Past the last page
                            break
                        if resp.status != 200:
                            logger.warning(
                                "HTTP %d fetching page %d of %s",
                                resp.status,
                                page_num,
                                site_id,
                            )
                            break
                        posts = await resp.json()
                        if not posts:
                            break

                        # Attach response time to each post for later checks
                        avg_time = elapsed / len(posts) if posts else elapsed
                        for p in posts:
                            p["_response_time"] = avg_time

                        all_posts.extend(posts)
                        total_pages_header = resp.headers.get("X-WP-TotalPages", "1")
                        total_posts_header = resp.headers.get("X-WP-Total", "0")

                        if page_num >= int(total_pages_header):
                            break
                except asyncio.TimeoutError:
                    logger.warning("Timeout fetching page %d of %s", page_num, site_id)
                    break
                except Exception as exc:
                    logger.error(
                        "Error fetching page %d of %s: %s", page_num, site_id, exc
                    )
                    break

                page_num += 1

        logger.info("Fetched %d posts from %s", len(all_posts), site.domain)

        # Audit each post
        audit = SiteAudit(
            site_id=site_id,
            domain=site.domain,
            total_pages=int(total_posts_header) if all_posts else 0,
            pages_audited=len(all_posts),
        )

        all_issues: List[SEOIssue] = []

        for post_data in all_posts:
            page_audit = await self._audit_single_post(
                site, post_data, check_links=False
            )
            page_audit.response_time = post_data.get("_response_time", 0.0)

            # Slow response check
            if page_audit.response_time > SLOW_RESPONSE_THRESHOLD:
                page_audit.issues.append(
                    self._make_issue(
                        site_id,
                        post_data.get("id"),
                        page_audit.url,
                        "slow_response",
                        "warning",
                        f"Page took {page_audit.response_time:.1f}s to respond.",
                        "Check LiteSpeed Cache and server performance.",
                    )
                )

            audit.pages.append(page_audit)
            all_issues.extend(page_audit.issues)

        # Cross-post checks
        cannibalization_issues = self.detect_cannibalization(site_id, all_posts)
        duplicate_meta_issues = self.detect_duplicate_metas(site_id, all_posts)
        duplicate_title_issues = self.detect_duplicate_titles(site_id, all_posts)

        all_issues.extend(cannibalization_issues)
        all_issues.extend(duplicate_meta_issues)
        all_issues.extend(duplicate_title_issues)

        # Aggregate issue counts
        severity_counts: Dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
        type_counts: Dict[str, int] = defaultdict(int)

        for issue in all_issues:
            severity_counts[issue.severity] = (
                severity_counts.get(issue.severity, 0) + 1
            )
            type_counts[issue.issue_type] += 1

        audit.issues_by_severity = severity_counts
        audit.issues_by_type = dict(type_counts)

        # Score
        audit.overall_score = self.calculate_site_score(audit)

        # Top issues (most frequent types, sorted by severity)
        severity_rank = {"critical": 0, "warning": 1, "info": 2}
        type_severity: Dict[str, str] = {}
        for issue in all_issues:
            existing = type_severity.get(issue.issue_type, "info")
            if severity_rank.get(issue.severity, 2) < severity_rank.get(existing, 2):
                type_severity[issue.issue_type] = issue.severity

        top = sorted(
            type_counts.items(),
            key=lambda x: (severity_rank.get(type_severity.get(x[0], "info"), 2), -x[1]),
        )
        audit.top_issues = [
            {
                "issue_type": it,
                "count": ct,
                "severity": type_severity.get(it, "info"),
            }
            for it, ct in top[:10]
        ]

        # Store active issues
        self._active_issues[site_id] = all_issues
        self._save_active_issues()

        # Save audit file
        audit_file = AUDITS_DIR / f"{site_id}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
        _save_json(audit_file, _serialize_audit(audit))

        # Record history
        self._record_history(audit)

        logger.info(
            "Audit complete for %s: score=%d, critical=%d, warning=%d, info=%d",
            site.domain,
            audit.overall_score,
            severity_counts["critical"],
            severity_counts["warning"],
            severity_counts["info"],
        )

        return audit

    async def crawl_all_sites(
        self, max_pages_per_site: int = 100
    ) -> Dict[str, SiteAudit]:
        """
        Audit all 16 sites in parallel with a concurrency limit.

        Parameters
        ----------
        max_pages_per_site : int
            Maximum posts to audit per site.

        Returns
        -------
        dict mapping site_id to SiteAudit
        """
        sem = asyncio.Semaphore(CONCURRENT_SITES)
        results: Dict[str, SiteAudit] = {}

        async def _crawl_one(site_id: str) -> None:
            async with sem:
                try:
                    site = self._sites[site_id]
                    if not site.is_configured:
                        logger.warning(
                            "Skipping %s: no API credentials", site_id
                        )
                        return
                    audit = await self.crawl_site(site_id, max_pages_per_site)
                    results[site_id] = audit
                except Exception as exc:
                    logger.error("Failed to audit %s: %s", site_id, exc)

        await asyncio.gather(
            *[_crawl_one(sid) for sid in self._sites.keys()]
        )

        return results

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def generate_report(
        self, site_id: Optional[str] = None, period: str = "week"
    ) -> SEOReport:
        """
        Generate an aggregated SEO report.

        Parameters
        ----------
        site_id : str, optional
            If provided, report on a single site. Otherwise all sites.
        period : str
            Report period label: "week", "month", "quarter".
        """
        report = SEOReport(period=period)

        target_sites = [site_id] if site_id else list(self._active_issues.keys())
        report.sites_audited = target_sites

        total_critical = 0
        total_warning = 0
        total_info = 0
        site_scores: List[Dict[str, Any]] = []

        for sid in target_sites:
            issues = self._active_issues.get(sid, [])
            c = sum(1 for i in issues if i.severity == "critical")
            w = sum(1 for i in issues if i.severity == "warning")
            n = sum(1 for i in issues if i.severity == "info")
            total_critical += c
            total_warning += w
            total_info += n

            # Get latest score from history
            hist = self._history.get(sid, [])
            latest_score = hist[-1]["score"] if hist else 100
            site_scores.append(
                {"site_id": sid, "score": latest_score, "critical": c, "warning": w, "info": n}
            )

        report.total_issues = total_critical + total_warning + total_info
        report.critical_count = total_critical
        report.warning_count = total_warning
        report.info_count = total_info

        # Improvements since last report
        improvements = 0
        for sid in target_sites:
            hist = self._history.get(sid, [])
            if len(hist) >= 2:
                prev_issues = hist[-2].get("total_issues", 0)
                curr_issues = hist[-1].get("total_issues", 0)
                if curr_issues < prev_issues:
                    improvements += prev_issues - curr_issues

        report.improvements_since_last = improvements

        # Sort sites by score
        sorted_sites = sorted(site_scores, key=lambda x: x["score"])
        report.worst_sites = sorted_sites[:5]
        report.best_sites = sorted(site_scores, key=lambda x: -x["score"])[:5]

        # Action items (critical first)
        report.action_items = self.get_action_items(site_id=site_id, max_items=20)

        return report

    def format_report(self, report: SEOReport, style: str = "text") -> str:
        """
        Format an SEOReport into a human-readable string.

        Parameters
        ----------
        report : SEOReport
        style : str
            "text" for plain text, "markdown" for Markdown.
        """
        if style == "markdown":
            return self._format_report_markdown(report)
        return self._format_report_text(report)

    def _format_report_text(self, report: SEOReport) -> str:
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append(f"  SEO AUDIT REPORT — {report.period.upper()}")
        lines.append(f"  Generated: {report.report_date[:10]}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Sites Audited:  {len(report.sites_audited)}")
        lines.append(f"Total Issues:   {report.total_issues}")
        lines.append(f"  Critical:     {report.critical_count}")
        lines.append(f"  Warning:      {report.warning_count}")
        lines.append(f"  Info:         {report.info_count}")
        lines.append(f"Improvements:   {report.improvements_since_last} issues fixed")
        lines.append("")

        if report.worst_sites:
            lines.append("--- WORST PERFORMING SITES ---")
            for s in report.worst_sites:
                lines.append(
                    f"  {s['site_id']:25s} Score: {s['score']:3d}  "
                    f"C:{s['critical']} W:{s['warning']} I:{s['info']}"
                )
            lines.append("")

        if report.best_sites:
            lines.append("--- BEST PERFORMING SITES ---")
            for s in report.best_sites:
                lines.append(
                    f"  {s['site_id']:25s} Score: {s['score']:3d}  "
                    f"C:{s['critical']} W:{s['warning']} I:{s['info']}"
                )
            lines.append("")

        if report.action_items:
            lines.append("--- TOP ACTION ITEMS ---")
            for idx, item in enumerate(report.action_items, 1):
                fix_tag = " [AUTO-FIXABLE]" if item.get("auto_fixable") else ""
                lines.append(
                    f"  {idx:2d}. [{item['severity'].upper()}] "
                    f"{item['site_id']} — {item['issue_type']}{fix_tag}"
                )
                lines.append(f"      {item['description'][:100]}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_report_markdown(self, report: SEOReport) -> str:
        lines: List[str] = []
        lines.append(f"# SEO Audit Report — {report.period.capitalize()}")
        lines.append(f"*Generated: {report.report_date[:10]}*")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- **Sites Audited:** {len(report.sites_audited)}")
        lines.append(f"- **Total Issues:** {report.total_issues}")
        lines.append(f"  - Critical: {report.critical_count}")
        lines.append(f"  - Warning: {report.warning_count}")
        lines.append(f"  - Info: {report.info_count}")
        lines.append(
            f"- **Improvements Since Last:** {report.improvements_since_last} issues fixed"
        )
        lines.append("")

        if report.worst_sites:
            lines.append("## Worst Performing Sites")
            lines.append("| Site | Score | Critical | Warning | Info |")
            lines.append("|------|-------|----------|---------|------|")
            for s in report.worst_sites:
                lines.append(
                    f"| {s['site_id']} | {s['score']} | {s['critical']} "
                    f"| {s['warning']} | {s['info']} |"
                )
            lines.append("")

        if report.best_sites:
            lines.append("## Best Performing Sites")
            lines.append("| Site | Score | Critical | Warning | Info |")
            lines.append("|------|-------|----------|---------|------|")
            for s in report.best_sites:
                lines.append(
                    f"| {s['site_id']} | {s['score']} | {s['critical']} "
                    f"| {s['warning']} | {s['info']} |"
                )
            lines.append("")

        if report.action_items:
            lines.append("## Top Action Items")
            for idx, item in enumerate(report.action_items, 1):
                fix_tag = " `AUTO-FIXABLE`" if item.get("auto_fixable") else ""
                sev = item["severity"].upper()
                lines.append(
                    f"{idx}. **[{sev}]** {item['site_id']} — "
                    f"`{item['issue_type']}`{fix_tag}"
                )
                lines.append(f"   {item['description'][:120]}")
            lines.append("")

        return "\n".join(lines)

    def get_action_items(
        self,
        site_id: Optional[str] = None,
        max_items: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Return prioritised action items: critical first, then auto-fixable.

        Parameters
        ----------
        site_id : str, optional
            Filter to a single site.
        max_items : int
            Maximum items to return.
        """
        target_sites = [site_id] if site_id else list(self._active_issues.keys())

        all_issues: List[SEOIssue] = []
        for sid in target_sites:
            all_issues.extend(self._active_issues.get(sid, []))

        # Sort: critical first, then auto-fixable, then by type
        severity_rank = {"critical": 0, "warning": 1, "info": 2}
        sorted_issues = sorted(
            all_issues,
            key=lambda i: (
                severity_rank.get(i.severity, 2),
                0 if i.auto_fixable else 1,
                i.issue_type,
            ),
        )

        # Deduplicate by (site_id, issue_type, post_id) — keep first
        seen: set = set()
        items: List[Dict[str, Any]] = []

        for issue in sorted_issues:
            key = (issue.site_id, issue.issue_type, issue.post_id)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "issue_id": issue.issue_id,
                    "site_id": issue.site_id,
                    "post_id": issue.post_id,
                    "url": issue.url,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "description": issue.description,
                    "recommendation": issue.recommendation,
                    "auto_fixable": issue.auto_fixable,
                }
            )
            if len(items) >= max_items:
                break

        return items

    def compare_audits(
        self, site_id: str, audit1_date: str, audit2_date: str
    ) -> Dict[str, Any]:
        """
        Compare two audit snapshots for a site, returning improvements
        and regressions.

        Parameters
        ----------
        site_id : str
        audit1_date : str
            Earlier date (YYYY-MM-DD).
        audit2_date : str
            Later date (YYYY-MM-DD).

        Returns
        -------
        dict with keys: score_change, new_issues, resolved_issues,
        issue_type_changes.
        """
        file1 = AUDITS_DIR / f"{site_id}_{audit1_date}.json"
        file2 = AUDITS_DIR / f"{site_id}_{audit2_date}.json"

        audit1_data = _load_json(file1)
        audit2_data = _load_json(file2)

        if not audit1_data or not audit2_data:
            return {
                "error": "One or both audit files not found.",
                "available_files": [
                    f.name
                    for f in AUDITS_DIR.glob(f"{site_id}_*.json")
                ],
            }

        score1 = audit1_data.get("overall_score", 0)
        score2 = audit2_data.get("overall_score", 0)

        types1 = audit1_data.get("issues_by_type", {})
        types2 = audit2_data.get("issues_by_type", {})

        all_types = set(types1.keys()) | set(types2.keys())
        type_changes: Dict[str, Dict[str, int]] = {}
        for itype in sorted(all_types):
            c1 = types1.get(itype, 0)
            c2 = types2.get(itype, 0)
            if c1 != c2:
                type_changes[itype] = {"before": c1, "after": c2, "change": c2 - c1}

        sev1 = audit1_data.get("issues_by_severity", {})
        sev2 = audit2_data.get("issues_by_severity", {})
        total1 = sum(sev1.values())
        total2 = sum(sev2.values())

        return {
            "site_id": site_id,
            "date_before": audit1_date,
            "date_after": audit2_date,
            "score_before": score1,
            "score_after": score2,
            "score_change": score2 - score1,
            "total_issues_before": total1,
            "total_issues_after": total2,
            "new_issues": max(0, total2 - total1),
            "resolved_issues": max(0, total1 - total2),
            "issue_type_changes": type_changes,
        }

    # -----------------------------------------------------------------------
    # Fix Suggestions (Claude Haiku)
    # -----------------------------------------------------------------------

    async def suggest_meta_description(
        self, title: str, content_snippet: str, keywords: List[str]
    ) -> str:
        """
        Use Claude Haiku to generate a meta description suggestion.

        Parameters
        ----------
        title : str
            Post title.
        content_snippet : str
            First ~200 words of content.
        keywords : list of str
            Target keywords to include.

        Returns
        -------
        str : Suggested meta description (120-155 chars).
        """
        try:
            import anthropic
        except ImportError:
            return (
                f"Write a 120-155 character meta description for '{title}' "
                f"including keywords: {', '.join(keywords)}"
            )

        client = anthropic.Anthropic()
        prompt = (
            f"Write a compelling meta description for this blog post. "
            f"Requirements:\n"
            f"- Between 120 and 155 characters\n"
            f"- Include the keyword '{keywords[0]}' naturally\n"
            f"- Include a call to action\n"
            f"- Do NOT use quotes\n\n"
            f"Title: {title}\n"
            f"Content preview: {content_snippet[:500]}\n"
            f"Keywords: {', '.join(keywords)}\n\n"
            f"Reply with ONLY the meta description, nothing else."
        )

        try:
            message = client.messages.create(
                model=MODEL_HAIKU,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as exc:
            logger.error("Haiku meta description generation failed: %s", exc)
            return (
                f"Write a 120-155 character meta description for '{title}' "
                f"including keywords: {', '.join(keywords)}"
            )

    async def suggest_alt_text(
        self, image_context: str, article_title: str
    ) -> str:
        """
        Use Claude Haiku to suggest alt text for an image.

        Parameters
        ----------
        image_context : str
            Surrounding text or image filename for context.
        article_title : str
            Title of the article the image appears in.

        Returns
        -------
        str : Suggested alt text.
        """
        try:
            import anthropic
        except ImportError:
            return f"Image related to {article_title}"

        client = anthropic.Anthropic()
        prompt = (
            f"Write descriptive alt text for an image in a blog post.\n"
            f"Requirements:\n"
            f"- Under 125 characters\n"
            f"- Descriptive and specific\n"
            f"- Naturally include the topic\n\n"
            f"Article title: {article_title}\n"
            f"Image context: {image_context}\n\n"
            f"Reply with ONLY the alt text, nothing else."
        )

        try:
            message = client.messages.create(
                model=MODEL_HAIKU,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as exc:
            logger.error("Haiku alt text generation failed: %s", exc)
            return f"Image related to {article_title}"

    def get_auto_fixable(self, site_id: str) -> List[SEOIssue]:
        """Return all auto-fixable issues for a site."""
        issues = self._active_issues.get(site_id, [])
        return [i for i in issues if i.auto_fixable and not i.fixed]

    # -----------------------------------------------------------------------
    # History & Trends
    # -----------------------------------------------------------------------

    def get_audit_history(
        self, site_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Return recent audit history summaries for a site.

        Parameters
        ----------
        site_id : str
        limit : int
            Maximum entries to return (most recent first).
        """
        hist = self._history.get(site_id, [])
        return list(reversed(hist[-limit:]))

    def get_issue_trends(
        self, site_id: str, days: int = 90
    ) -> Dict[str, Any]:
        """
        Return issue count trends over time for a site.

        Parameters
        ----------
        site_id : str
        days : int
            Number of days to look back.

        Returns
        -------
        dict with keys: dates, scores, critical_counts, warning_counts,
        info_counts, total_counts.
        """
        hist = self._history.get(site_id, [])

        # Filter to requested time window
        cutoff = datetime.now(timezone.utc).isoformat()[:10]
        try:
            from datetime import timedelta

            cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
            cutoff = cutoff_dt.isoformat()
        except Exception:
            pass

        filtered = [h for h in hist if h.get("date", "") >= cutoff]

        return {
            "site_id": site_id,
            "days": days,
            "data_points": len(filtered),
            "dates": [h.get("date", "")[:10] for h in filtered],
            "scores": [h.get("score", 0) for h in filtered],
            "critical_counts": [
                h.get("issues_by_severity", {}).get("critical", 0) for h in filtered
            ],
            "warning_counts": [
                h.get("issues_by_severity", {}).get("warning", 0) for h in filtered
            ],
            "info_counts": [
                h.get("issues_by_severity", {}).get("info", 0) for h in filtered
            ],
            "total_counts": [h.get("total_issues", 0) for h in filtered],
        }

    # -----------------------------------------------------------------------
    # Quick scores (summary without full audit)
    # -----------------------------------------------------------------------

    def get_all_scores(self) -> List[Dict[str, Any]]:
        """
        Return the latest score for every site with history data.
        Useful for a quick dashboard view.
        """
        scores: List[Dict[str, Any]] = []
        for site_id, hist in self._history.items():
            if hist:
                latest = hist[-1]
                scores.append(
                    {
                        "site_id": site_id,
                        "score": latest.get("score", 0),
                        "date": latest.get("date", "")[:10],
                        "total_issues": latest.get("total_issues", 0),
                        "issues_by_severity": latest.get("issues_by_severity", {}),
                    }
                )

        scores.sort(key=lambda x: x["score"])
        return scores


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


_STRIP_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = _STRIP_RE.sub(" ", html)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_auditor_instance: Optional[SEOAuditor] = None


def get_auditor() -> SEOAuditor:
    """Return the singleton SEOAuditor instance."""
    global _auditor_instance
    if _auditor_instance is None:
        _auditor_instance = SEOAuditor()
    return _auditor_instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seo_auditor",
        description="SEO Auditor for the OpenClaw Empire — audit, score, "
        "and generate reports for 16 WordPress sites.",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # audit
    audit_p = sub.add_parser("audit", help="Full site audit")
    audit_grp = audit_p.add_mutually_exclusive_group(required=True)
    audit_grp.add_argument("--site", type=str, help="Site ID to audit")
    audit_grp.add_argument(
        "--all", action="store_true", dest="audit_all", help="Audit all 16 sites"
    )
    audit_p.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f"Max pages per site (default: {DEFAULT_MAX_PAGES})",
    )

    # post
    post_p = sub.add_parser("post", help="Audit a single post")
    post_p.add_argument("--site", type=str, required=True, help="Site ID")
    post_p.add_argument(
        "--post-id", type=int, required=True, help="WordPress post ID"
    )

    # report
    report_p = sub.add_parser("report", help="Generate SEO report")
    report_p.add_argument(
        "--period",
        type=str,
        default="week",
        choices=["week", "month", "quarter"],
        help="Report period (default: week)",
    )
    report_p.add_argument(
        "--site", type=str, default=None, help="Filter to a single site"
    )
    report_p.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "markdown"],
        dest="report_format",
        help="Output format (default: text)",
    )

    # issues
    issues_p = sub.add_parser("issues", help="List active issues")
    issues_p.add_argument("--site", type=str, default=None, help="Filter by site")
    issues_p.add_argument(
        "--severity",
        type=str,
        default=None,
        choices=["critical", "warning", "info"],
        help="Filter by severity",
    )

    # cannibalization
    cann_p = sub.add_parser(
        "cannibalization", help="Check keyword cannibalization"
    )
    cann_p.add_argument("--site", type=str, required=True, help="Site ID")

    # action-items
    act_p = sub.add_parser("action-items", help="Prioritised action items")
    act_p.add_argument(
        "--count", type=int, default=20, help="Max items (default: 20)"
    )
    act_p.add_argument("--site", type=str, default=None, help="Filter by site")

    # history
    hist_p = sub.add_parser("history", help="Audit score history")
    hist_p.add_argument("--site", type=str, required=True, help="Site ID")
    hist_p.add_argument(
        "--limit", type=int, default=10, help="Max entries (default: 10)"
    )

    # score
    score_p = sub.add_parser("score", help="Quick scores for all sites")
    score_p.add_argument(
        "--all",
        action="store_true",
        dest="score_all",
        help="Show all site scores",
    )

    return parser


async def _run_cli(args: argparse.Namespace) -> None:
    """Execute the CLI command."""
    auditor = get_auditor()

    if args.command == "audit":
        if args.audit_all:
            print(f"Auditing all sites (max {args.max_pages} pages each)...")
            results = await auditor.crawl_all_sites(
                max_pages_per_site=args.max_pages
            )
            print(f"\nCompleted {len(results)} site audits:\n")
            for sid, audit in sorted(
                results.items(), key=lambda x: x[1].overall_score
            ):
                sev = audit.issues_by_severity
                print(
                    f"  {sid:25s} Score: {audit.overall_score:3d}  "
                    f"Pages: {audit.pages_audited:3d}  "
                    f"C:{sev.get('critical', 0)} W:{sev.get('warning', 0)} "
                    f"I:{sev.get('info', 0)}"
                )
        else:
            print(f"Auditing {args.site} (max {args.max_pages} pages)...")
            audit = await auditor.crawl_site(args.site, args.max_pages)
            sev = audit.issues_by_severity
            print(f"\nAudit complete for {audit.domain}:")
            print(f"  Score:     {audit.overall_score}/100")
            print(f"  Pages:     {audit.pages_audited}")
            print(f"  Critical:  {sev.get('critical', 0)}")
            print(f"  Warning:   {sev.get('warning', 0)}")
            print(f"  Info:      {sev.get('info', 0)}")
            if audit.top_issues:
                print("\n  Top Issues:")
                for ti in audit.top_issues:
                    print(
                        f"    [{ti['severity'].upper():8s}] "
                        f"{ti['issue_type']} ({ti['count']}x)"
                    )

    elif args.command == "post":
        print(f"Auditing post {args.post_id} on {args.site}...")
        page = await auditor.audit_post(args.site, args.post_id)
        print(f"\nPost: {page.title}")
        print(f"URL:  {page.url}")
        print(f"Words: {page.word_count}  |  Response: {page.response_time:.2f}s")
        print(f"Focus keyword: {page.focus_keyword or '(none)'}")
        print(
            f"Internal links: {page.internal_links_count}  |  "
            f"External links: {page.external_links_count}"
        )
        print(f"Images without alt: {len(page.images_without_alt)}")
        print(f"Schema: {', '.join(page.schema_types) if page.schema_types else 'None'}")
        if page.issues:
            print(f"\nIssues ({len(page.issues)}):")
            for issue in page.issues:
                print(f"  [{issue.severity.upper():8s}] {issue.issue_type}")
                print(f"    {issue.description}")
                print(f"    -> {issue.recommendation}")

    elif args.command == "report":
        report = auditor.generate_report(
            site_id=args.site, period=args.period
        )
        print(auditor.format_report(report, style=args.report_format))

    elif args.command == "issues":
        target_sites = (
            [args.site] if args.site else list(auditor._active_issues.keys())
        )
        total = 0
        for sid in sorted(target_sites):
            issues = auditor._active_issues.get(sid, [])
            if args.severity:
                issues = [i for i in issues if i.severity == args.severity]
            if not issues:
                continue
            print(f"\n--- {sid} ({len(issues)} issues) ---")
            for issue in issues:
                fix_tag = " [AUTO-FIX]" if issue.auto_fixable else ""
                print(
                    f"  [{issue.severity.upper():8s}] {issue.issue_type}{fix_tag}"
                )
                if issue.url:
                    print(f"    URL: {issue.url}")
                print(f"    {issue.description}")
            total += len(issues)
        print(f"\nTotal: {total} issues")

    elif args.command == "cannibalization":
        # Load issues for this site
        issues = auditor._active_issues.get(args.site, [])
        cann = [i for i in issues if i.issue_type == "keyword_cannibalization"]
        if not cann:
            print(f"No keyword cannibalization detected on {args.site}.")
        else:
            print(
                f"Found {len(cann)} cannibalization issue(s) on {args.site}:\n"
            )
            for issue in cann:
                print(f"  Post: {issue.url or f'ID {issue.post_id}'}")
                print(f"  {issue.description}")
                print(f"  -> {issue.recommendation}")
                print()

    elif args.command == "action-items":
        items = auditor.get_action_items(
            site_id=args.site, max_items=args.count
        )
        if not items:
            print("No action items found.")
        else:
            print(f"Top {len(items)} action items:\n")
            for idx, item in enumerate(items, 1):
                fix_tag = " [AUTO-FIXABLE]" if item.get("auto_fixable") else ""
                print(
                    f"  {idx:2d}. [{item['severity'].upper():8s}] "
                    f"{item['site_id']} — {item['issue_type']}{fix_tag}"
                )
                print(f"      {item['description'][:120]}")
                print(f"      -> {item['recommendation'][:120]}")
                print()

    elif args.command == "history":
        history = auditor.get_audit_history(args.site, limit=args.limit)
        if not history:
            print(f"No audit history for {args.site}.")
        else:
            print(f"Audit history for {args.site} (last {len(history)}):\n")
            print(f"  {'Date':12s} {'Score':>6s} {'Pages':>6s} {'Crit':>5s} {'Warn':>5s} {'Info':>5s}")
            print(f"  {'-' * 46}")
            for entry in history:
                sev = entry.get("issues_by_severity", {})
                print(
                    f"  {entry['date'][:10]:12s} "
                    f"{entry['score']:6d} "
                    f"{entry.get('pages_audited', 0):6d} "
                    f"{sev.get('critical', 0):5d} "
                    f"{sev.get('warning', 0):5d} "
                    f"{sev.get('info', 0):5d}"
                )

    elif args.command == "score":
        scores = auditor.get_all_scores()
        if not scores:
            print("No audit history found. Run 'audit --all' first.")
        else:
            print("Site SEO Scores (from audit history):\n")
            print(f"  {'Site':25s} {'Score':>6s} {'Issues':>7s} {'Date':12s}")
            print(f"  {'-' * 54}")
            for s in scores:
                print(
                    f"  {s['site_id']:25s} "
                    f"{s['score']:6d} "
                    f"{s['total_issues']:7d} "
                    f"{s['date']:12s}"
                )

    else:
        print("No command specified. Use --help for usage.")


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        asyncio.run(_run_cli(args))
    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as exc:
        logger.error("Command failed: %s", exc, exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
