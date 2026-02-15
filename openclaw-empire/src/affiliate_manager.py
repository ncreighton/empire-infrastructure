"""
Affiliate Link Management System — OpenClaw Empire Edition

Tracks affiliate links across 16 WordPress sites. Detects broken/expired links,
manages replacements, tracks performance, suggests placements, and alerts on issues.
Programs: Amazon Associates, ShareASale, CJ Affiliate, Impact, Content Egg.
Data: data/affiliate/

Usage:
    from src.affiliate_manager import get_manager
    manager = get_manager()
    links = await manager.scan_site("witchcraft")     # discover links
    checks = await manager.check_site_links("witchcraft")  # health check
    report = manager.monthly_report()                  # performance report

CLI:
    python -m src.affiliate_manager scan --site witchcraft
    python -m src.affiliate_manager check --site witchcraft
    python -m src.affiliate_manager broken
    python -m src.affiliate_manager report --period month
    python -m src.affiliate_manager top --count 20 --days 30
    python -m src.affiliate_manager suggest --site witchcraft --post-id 1234
    python -m src.affiliate_manager replace --link-id ID --new-url "..."
    python -m src.affiliate_manager programs
    python -m src.affiliate_manager stats --program amazon --days 30
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("affiliate_manager")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\Claude Code Projects\openclaw-empire")
AFFILIATE_DATA_DIR = PROJECT_ROOT / "data" / "affiliate"
LINKS_FILE = AFFILIATE_DATA_DIR / "links.json"
CHECKS_FILE = AFFILIATE_DATA_DIR / "checks.json"
EARNINGS_FILE = AFFILIATE_DATA_DIR / "earnings.json"
CONFIG_FILE = AFFILIATE_DATA_DIR / "config.json"
REPORTS_DIR = AFFILIATE_DATA_DIR / "reports"
SITE_REGISTRY_PATH = PROJECT_ROOT / "configs" / "site-registry.json"

# Ensure directories exist on import
AFFILIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SITE_IDS = [
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
]

DOMAIN_MAP: dict[str, str] = {
    "witchcraftforbeginners.com": "witchcraft",
    "smarthomewizards.com": "smarthome",
    "aiinactionhub.com": "aiaction",
    "aidiscoverydigest.com": "aidiscovery",
    "wealthfromai.com": "wealthai",
    "family-flourish.com": "family",
    "mythicalarchives.com": "mythical",
    "bulletjournals.net": "bulletjournals",
    "crystalwitchcraft.com": "crystalwitchcraft",
    "herbalwitchery.com": "herbalwitchery",
    "moonphasewitch.com": "moonphasewitch",
    "tarotforbeginners.net": "tarotbeginners",
    "spellsandrituals.com": "spellsrituals",
    "paganpathways.net": "paganpathways",
    "witchyhomedecor.com": "witchyhomedecor",
    "seasonalwitchcraft.com": "seasonalwitchcraft",
}

# Reverse map: site_id -> domain
SITE_DOMAIN_MAP: dict[str, str] = {v: k for k, v in DOMAIN_MAP.items()}

MAX_LINKS = 10000
MAX_CHECKS = 5000
LINK_CHECK_TIMEOUT = 15
LINK_CHECK_CONCURRENCY = 10
WP_MAX_PER_PAGE = 100

AMAZON_DOG_PAGE_MARKERS = [
    "Sorry, we couldn't find that page",
    "looking for something",
    "The Web address you entered is not a functioning page",
]

AMAZON_DOMAINS = {"amzn.to", "amazon.com", "amazon.co.uk", "amazon.ca", "amazon.de"}
AMAZON_PATH_PATTERNS = ["/dp/", "/gp/product/", "/gp/aw/d/", "/exec/obidos/"]
SHAREASALE_DOMAINS = {"shareasale.com", "shareasale-analytics.com"}
CJ_DOMAINS = {"anrdoezrs.net", "jdoqocy.com", "tkqlhce.com", "dpbolvw.net",
               "kqzyfj.com", "commission-junction.com", "cj.com"}
IMPACT_DOMAINS = {"impact.com", "sjv.io", "goto.target.com", "goto.walmart.com"}
CONTENT_EGG_DOMAINS = {"contentegg.com"}
ASIN_RE = re.compile(r"(?:dp|gp/product|gp/aw/d)/([A-Z0-9]{10})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _today_iso() -> str:
    return _now_utc().strftime("%Y-%m-%d")


def _days_ago_iso(days: int) -> str:
    return (_now_utc() - timedelta(days=days)).strftime("%Y-%m-%d")


def _make_id() -> str:
    return str(uuid.uuid4())


class _LinkExtractor(HTMLParser):
    """Extract <a> tags with href and their anchor text from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._current_href: Optional[str] = None
        self._current_text_parts: list[str] = []
        self._in_a = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "a":
            attr_dict = dict(attrs)
            href = attr_dict.get("href", "")
            if href and href.startswith(("http://", "https://")):
                self._current_href = href
                self._current_text_parts = []
                self._in_a = True

    def handle_data(self, data: str) -> None:
        if self._in_a:
            self._current_text_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_a:
            anchor = " ".join(self._current_text_parts).strip()
            if self._current_href:
                self.links.append({
                    "href": self._current_href,
                    "anchor": anchor or "(no text)",
                })
            self._in_a = False
            self._current_href = None
            self._current_text_parts = []


def _extract_links_from_html(html: str) -> list[dict[str, str]]:
    """Parse HTML and return list of dicts with 'href' and 'anchor' keys."""
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.links


def _get_domain(url: str) -> str:
    """Extract the domain from a URL, stripping www. prefix."""
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return ""


def _identify_affiliate_program(url: str) -> Optional[str]:
    """Identify the affiliate program from a URL. Returns program name or None."""
    domain = _get_domain(url)
    if not domain:
        return None

    # Amazon
    if domain in AMAZON_DOMAINS or "amazon" in domain:
        return "amazon"
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "tag" in query and "amazon" in domain:
        return "amazon"
    for pattern in AMAZON_PATH_PATTERNS:
        if pattern in (parsed.path or ""):
            return "amazon"

    # ShareASale
    if domain in SHAREASALE_DOMAINS:
        return "shareasale"

    # CJ Affiliate
    if domain in CJ_DOMAINS:
        return "cj"

    # Impact
    if domain in IMPACT_DOMAINS or "impact" in domain:
        return "impact"

    # Content Egg redirects
    if domain in CONTENT_EGG_DOMAINS:
        return "contentegg"

    # Check for common affiliate query parameters
    if query.get("ref") or query.get("aff") or query.get("affiliate"):
        return "direct"

    return None


def _is_external_link(url: str, site_domain: str) -> bool:
    """Check whether a URL is external relative to the given site domain."""
    link_domain = _get_domain(url)
    if not link_domain:
        return False
    # Strip www from site domain too
    clean_site = site_domain.lower()
    if clean_site.startswith("www."):
        clean_site = clean_site[4:]
    return link_domain != clean_site


# ===================================================================
# Data Classes
# ===================================================================


def _dc_from_dict(cls, data: dict):
    """Shared from_dict: filter keys to only valid dataclass fields."""
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in dict(data).items() if k in valid})


@dataclass
class AffiliateLink:
    """A tracked affiliate link on one of the 16 sites."""
    link_id: str
    site_id: str
    post_id: int
    url: str
    affiliate_program: str  # "amazon", "shareasale", "cj", "impact", "direct", "contentegg"
    product_name: Optional[str] = None
    anchor_text: str = ""
    location_in_post: str = "body"  # "body", "sidebar", "header", "footer"
    status: str = "active"  # "active", "broken", "expired", "replaced"
    last_checked: str = ""
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    created_at: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.link_id:
            self.link_id = _make_id()
        self.revenue = round(float(self.revenue), 2)

    @property
    def conversion_rate(self) -> float:
        return round(self.conversions / self.clicks * 100, 2) if self.clicks > 0 else 0.0

    @property
    def revenue_per_click(self) -> float:
        return round(self.revenue / self.clicks, 4) if self.clicks > 0 else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AffiliateLink:
        return _dc_from_dict(cls, data)


@dataclass
class LinkCheck:
    """Result of a single link health check."""
    link_id: str
    checked_at: str
    status_code: int
    redirect_url: Optional[str] = None
    response_time_ms: float = 0.0
    is_valid: bool = True
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.checked_at:
            self.checked_at = _now_iso()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LinkCheck:
        return _dc_from_dict(cls, data)


@dataclass
class AffiliateProgramConfig:
    """Configuration for an affiliate program."""
    program_name: str
    api_key_env: Optional[str] = None
    tracking_id: Optional[str] = None
    commission_rate: Optional[float] = None
    cookie_days: Optional[int] = None
    payment_threshold: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AffiliateProgramConfig:
        return _dc_from_dict(cls, data)


@dataclass
class PlacementSuggestion:
    """A suggestion for a new affiliate link placement."""
    site_id: str
    post_id: int
    product_name: str
    suggested_anchor: str
    suggested_url: str
    relevance_score: float
    estimated_ctr: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PlacementSuggestion:
        return _dc_from_dict(cls, data)


@dataclass
class AffiliateReport:
    """Aggregate affiliate performance report."""
    period: str
    total_links: int = 0
    active_links: int = 0
    broken_links: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    conversion_rate: float = 0.0
    top_products: list = field(default_factory=list)
    top_posts: list = field(default_factory=list)
    issues: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.revenue = round(float(self.revenue), 2)
        if self.clicks > 0:
            self.conversion_rate = round(self.conversions / self.clicks * 100, 2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AffiliateReport:
        return _dc_from_dict(cls, data)


def _load_site_registry() -> list[dict]:
    """Load site registry and return list of site dicts."""
    raw = _load_json(SITE_REGISTRY_PATH, {})
    return raw.get("sites", [])


def _get_site_config(site_id: str) -> Optional[dict]:
    """Get a single site's config from the registry."""
    sites = _load_site_registry()
    for site in sites:
        if site.get("id") == site_id:
            return site
    return None


def _get_site_auth(site_id: str) -> tuple[str, str, str]:
    """Return (domain, wp_user, app_password) for a site.

    App password is read from the environment variable specified in the registry.
    Returns empty strings if not configured.
    """
    config = _get_site_config(site_id)
    if not config:
        return ("", "", "")
    domain = config.get("domain", "")
    wp_user = config.get("wp_user", "")
    env_key = config.get("wp_app_password_env", "")
    app_password = os.environ.get(env_key, "") if env_key else ""
    return (domain, wp_user, app_password)


def _make_auth_header(wp_user: str, app_password: str) -> str:
    """Build Basic auth header value."""
    if not wp_user or not app_password:
        return ""
    creds = f"{wp_user}:{app_password}"
    encoded = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
    return f"Basic {encoded}"


def _is_amazon_link(url: str) -> bool:
    """Check if a URL is an Amazon affiliate link."""
    domain = _get_domain(url)
    if domain in AMAZON_DOMAINS or "amazon" in domain:
        return True
    parsed = urlparse(url)
    for pattern in AMAZON_PATH_PATTERNS:
        if pattern in (parsed.path or ""):
            return True
    return False


def _extract_asin(url: str) -> Optional[str]:
    """Extract the ASIN from an Amazon product URL."""
    match = ASIN_RE.search(url)
    if match:
        return match.group(1)
    # Try path segments: /dp/ASIN or /product/ASIN
    parsed = urlparse(url)
    parts = (parsed.path or "").split("/")
    for i, part in enumerate(parts):
        if part in ("dp", "product") and i + 1 < len(parts):
            candidate = parts[i + 1]
            if re.match(r"^[A-Z0-9]{10}$", candidate):
                return candidate
    return None


def _build_amazon_link(asin: str, tracking_id: str) -> str:
    """Build a clean Amazon affiliate link from ASIN and tracking ID."""
    return f"https://www.amazon.com/dp/{asin}?tag={tracking_id}"


async def _check_amazon_product(asin: str, session: aiohttp.ClientSession) -> dict:
    """Check Amazon product availability via a GET request.

    Returns a dict with keys: available (bool), title (str/None), error (str/None).
    """
    url = f"https://www.amazon.com/dp/{asin}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=LINK_CHECK_TIMEOUT),
                               allow_redirects=True) as resp:
            if resp.status == 404:
                return {"available": False, "title": None, "error": "Product not found (404)"}
            text = await resp.text()
            # Check for Amazon dog page (removed product)
            for marker in AMAZON_DOG_PAGE_MARKERS:
                if marker.lower() in text.lower():
                    return {"available": False, "title": None, "error": "Product removed (dog page)"}
            # Try to extract title
            title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else None
            return {"available": True, "title": title, "error": None}
    except Exception as exc:
        return {"available": False, "title": None, "error": str(exc)}


# ===================================================================
# AffiliateManager
# ===================================================================

class AffiliateManager:
    """
    Central affiliate link management engine for the empire.

    Handles link discovery, health checking, replacement, performance
    tracking, placement suggestions, reporting, and alerting.
    """

    def __init__(self) -> None:
        self._links: dict[str, AffiliateLink] = {}
        self._checks: list[dict] = []
        self._earnings: list[dict] = []
        self._program_configs: dict[str, AffiliateProgramConfig] = {}
        self._load_all()
        logger.info("AffiliateManager initialized — data dir: %s", AFFILIATE_DATA_DIR)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all persisted data from disk."""
        # Links
        raw_links = _load_json(LINKS_FILE, [])
        if isinstance(raw_links, list):
            for item in raw_links[-MAX_LINKS:]:
                try:
                    link = AffiliateLink.from_dict(item)
                    self._links[link.link_id] = link
                except (TypeError, KeyError) as exc:
                    logger.warning("Skipping malformed link: %s", exc)

        # Checks
        raw_checks = _load_json(CHECKS_FILE, [])
        if isinstance(raw_checks, list):
            self._checks = raw_checks[-MAX_CHECKS:]

        # Earnings
        self._earnings = _load_json(EARNINGS_FILE, [])
        if not isinstance(self._earnings, list):
            self._earnings = []

        # Config
        raw_config = _load_json(CONFIG_FILE, {})
        programs = raw_config.get("programs", {})
        for name, pdata in programs.items():
            try:
                pdata["program_name"] = name
                self._program_configs[name] = AffiliateProgramConfig.from_dict(pdata)
            except (TypeError, KeyError):
                pass

        # Apply defaults if no program configs exist
        if not self._program_configs:
            self._program_configs = self._default_program_configs()
            self._save_config()

    def _save_links(self) -> None:
        """Persist all links to disk, bounded at MAX_LINKS."""
        all_links = list(self._links.values())
        # If over limit, keep the most recent by created_at
        if len(all_links) > MAX_LINKS:
            all_links.sort(key=lambda l: l.created_at, reverse=True)
            all_links = all_links[:MAX_LINKS]
            self._links = {l.link_id: l for l in all_links}
        _save_json(LINKS_FILE, [l.to_dict() for l in all_links])

    def _save_checks(self) -> None:
        """Persist check history, bounded at MAX_CHECKS."""
        self._checks = self._checks[-MAX_CHECKS:]
        _save_json(CHECKS_FILE, self._checks)

    def _save_earnings(self) -> None:
        """Persist earnings data."""
        _save_json(EARNINGS_FILE, self._earnings)

    def _save_config(self) -> None:
        """Persist program configs."""
        config = {
            "programs": {name: pc.to_dict() for name, pc in self._program_configs.items()},
        }
        _save_json(CONFIG_FILE, config)

    def _default_program_configs(self) -> dict[str, AffiliateProgramConfig]:
        """Return default affiliate program configurations."""
        # (name, api_key_env, tracking_id_env, commission, cookie_days, payout_min)
        defaults = [
            ("amazon",     "AMAZON_PAAPI_KEY",    "AMAZON_TRACKING_ID",     0.04, 1,  10.0),
            ("shareasale", "SHAREASALE_API_KEY",   "SHAREASALE_AFFILIATE_ID",0.10, 30, 50.0),
            ("cj",         "CJ_API_KEY",           "CJ_WEBSITE_ID",         0.08, 45, 50.0),
            ("impact",     "IMPACT_API_KEY",       "IMPACT_ACCOUNT_SID",    0.07, 30, 25.0),
            ("direct",     None,                   None,                     0.05, 30, None),
            ("contentegg", None,                   None,                     0.05, 30, None),
        ]
        result: dict[str, AffiliateProgramConfig] = {}
        for name, api_env, tid_env, rate, cookies, payout in defaults:
            result[name] = AffiliateProgramConfig(
                program_name=name, api_key_env=api_env,
                tracking_id=os.environ.get(tid_env, "") if tid_env else None,
                commission_rate=rate, cookie_days=cookies, payment_threshold=payout,
            )
        return result

    # -- Link Discovery ---------------------------------------------------

    @staticmethod
    def _build_wp_headers(wp_user: str, app_password: str) -> dict[str, str]:
        """Build HTTP headers for WP REST API requests."""
        headers: dict[str, str] = {"User-Agent": "OpenClaw-AffiliateManager/1.0"}
        auth = _make_auth_header(wp_user, app_password)
        if auth:
            headers["Authorization"] = auth
        return headers

    async def scan_site(self, site_id: str, max_posts: int = 200) -> list[AffiliateLink]:
        """Fetch posts via WP REST API, discover affiliate links by URL pattern, store results."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required. Install: pip install aiohttp")
        domain, wp_user, app_password = _get_site_auth(site_id)
        if not domain:
            logger.error("Site '%s' not found in registry", site_id)
            return []

        api_url = f"https://{domain}/wp-json/wp/v2"
        headers = self._build_wp_headers(wp_user, app_password)

        discovered: list[AffiliateLink] = []
        page = 1
        fetched = 0

        async with aiohttp.ClientSession() as session:
            while fetched < max_posts:
                per_page = min(WP_MAX_PER_PAGE, max_posts - fetched)
                params = {
                    "per_page": per_page,
                    "page": page,
                    "status": "publish",
                    "_fields": "id,title,content,link",
                }
                try:
                    async with session.get(
                        f"{api_url}/posts", headers=headers, params=params,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status == 401:
                            logger.warning("Auth failed for %s (401). Trying without auth.", site_id)
                            headers.pop("Authorization", None)
                            continue
                        if resp.status != 200:
                            logger.warning("WP API error for %s: HTTP %d", site_id, resp.status)
                            break
                        posts = await resp.json()
                        if not posts:
                            break
                except Exception as exc:
                    logger.error("Failed to fetch posts for %s: %s", site_id, exc)
                    break

                for post in posts:
                    post_id = post.get("id", 0)
                    html_content = post.get("content", {}).get("rendered", "")
                    post_links = self._discover_links_in_post(site_id, post_id, html_content, domain)
                    discovered.extend(post_links)

                fetched += len(posts)
                if len(posts) < per_page:
                    break
                page += 1

        # Merge discovered links with existing database
        new_count = 0
        for link in discovered:
            existing = self._find_existing_link(link.site_id, link.post_id, link.url)
            if existing is None:
                self._links[link.link_id] = link
                new_count += 1
            else:
                # Update anchor text and last_checked but keep stats
                existing.anchor_text = link.anchor_text
                existing.last_checked = _now_iso()

        self._save_links()
        logger.info("Scanned %s: %d posts, %d links found, %d new",
                     site_id, fetched, len(discovered), new_count)
        return discovered

    def scan_site_sync(self, site_id: str, max_posts: int = 200) -> list[AffiliateLink]:
        """Synchronous wrapper for scan_site."""
        return asyncio.run(self.scan_site(site_id, max_posts))

    async def scan_all_sites(self) -> dict[str, list[AffiliateLink]]:
        """Scan all 16 sites for affiliate links."""
        results: dict[str, list[AffiliateLink]] = {}
        for site_id in ALL_SITE_IDS:
            try:
                links = await self.scan_site(site_id)
                results[site_id] = links
            except Exception as exc:
                logger.error("Failed to scan %s: %s", site_id, exc)
                results[site_id] = []
        return results

    def scan_all_sites_sync(self) -> dict[str, list[AffiliateLink]]:
        """Synchronous wrapper for scan_all_sites."""
        return asyncio.run(self.scan_all_sites())

    async def scan_post(self, site_id: str, post_id: int) -> list[AffiliateLink]:
        """Scan a single post for affiliate links."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required. Install: pip install aiohttp")
        domain, wp_user, app_password = _get_site_auth(site_id)
        if not domain:
            return []

        api_url = f"https://{domain}/wp-json/wp/v2"
        headers = self._build_wp_headers(wp_user, app_password)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{api_url}/posts/{post_id}", headers=headers,
                    params={"_fields": "id,title,content,link"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Failed to fetch post %d from %s: HTTP %d",
                                       post_id, site_id, resp.status)
                        return []
                    post = await resp.json()
            except Exception as exc:
                logger.error("Failed to fetch post %d from %s: %s", post_id, site_id, exc)
                return []

        html_content = post.get("content", {}).get("rendered", "")
        discovered = self._discover_links_in_post(site_id, post_id, html_content, domain)

        # Merge with database
        for link in discovered:
            existing = self._find_existing_link(link.site_id, link.post_id, link.url)
            if existing is None:
                self._links[link.link_id] = link

        self._save_links()
        return discovered

    def scan_post_sync(self, site_id: str, post_id: int) -> list[AffiliateLink]:
        """Synchronous wrapper for scan_post."""
        return asyncio.run(self.scan_post(site_id, post_id))

    def _discover_links_in_post(
        self, site_id: str, post_id: int, html: str, site_domain: str,
    ) -> list[AffiliateLink]:
        """Extract affiliate links from a post's HTML content."""
        raw_links = _extract_links_from_html(html)
        discovered: list[AffiliateLink] = []

        for raw in raw_links:
            href = raw["href"]
            anchor = raw["anchor"]

            # Skip internal links
            if not _is_external_link(href, site_domain):
                continue

            # Identify affiliate program
            program = _identify_affiliate_program(href)
            if program is None:
                continue  # Not an affiliate link

            link = AffiliateLink(
                link_id=_make_id(),
                site_id=site_id,
                post_id=post_id,
                url=href,
                affiliate_program=program,
                product_name=self._guess_product_name(href, anchor),
                anchor_text=anchor,
                location_in_post="body",
                status="active",
                last_checked=_now_iso(),
                created_at=_now_iso(),
            )
            discovered.append(link)

        return discovered

    def _guess_product_name(self, url: str, anchor: str) -> Optional[str]:
        """Try to guess a product name from the URL path or anchor text."""
        if anchor and anchor != "(no text)" and len(anchor) > 3:
            return anchor[:120]
        # Try extracting from URL path
        parsed = urlparse(url)
        path = parsed.path or ""
        parts = [p for p in path.split("/") if p and p not in ("dp", "gp", "product", "r.cfm")]
        if parts:
            # Convert URL slug to readable name
            candidate = parts[-1].replace("-", " ").replace("_", " ").title()
            if len(candidate) > 3 and not re.match(r"^[A-Z0-9]{10}$", candidate.strip()):
                return candidate[:120]
        return None

    def _find_existing_link(
        self, site_id: str, post_id: int, url: str,
    ) -> Optional[AffiliateLink]:
        """Find an existing link in the database by site+post+url."""
        for link in self._links.values():
            if link.site_id == site_id and link.post_id == post_id and link.url == url:
                return link
        return None

    # -- Link Health Checking -----------------------------------------------

    async def check_link(self, link: AffiliateLink) -> LinkCheck:
        """Check a single affiliate link's health via HEAD request (falls back to GET)."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required. Install: pip install aiohttp")

        start_time = time.monotonic()
        check = LinkCheck(link_id=link.link_id, checked_at=_now_iso(), status_code=0, is_valid=False)
        ok_codes = (200, 301, 302, 307, 308)
        hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Accept": "*/*"}
        timeout = aiohttp.ClientTimeout(total=LINK_CHECK_TIMEOUT)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.head(link.url, headers=hdrs, allow_redirects=True, timeout=timeout) as resp:
                    check.status_code = resp.status
                    check.response_time_ms = round((time.monotonic() - start_time) * 1000, 1)
                    final = str(resp.url)
                    if final != link.url:
                        check.redirect_url = final
                    if resp.status in ok_codes:
                        check.is_valid = True
                    elif resp.status == 405:  # HEAD not allowed, try GET
                        async with session.get(link.url, headers=hdrs, allow_redirects=True, timeout=timeout) as gr:
                            check.status_code = gr.status
                            check.response_time_ms = round((time.monotonic() - start_time) * 1000, 1)
                            gf = str(gr.url)
                            if gf != link.url:
                                check.redirect_url = gf
                            check.is_valid = gr.status in ok_codes
                    elif resp.status in (404, 410):
                        check.error = f"{'Not found' if resp.status == 404 else 'Gone'} ({resp.status})"
                    elif resp.status >= 500:
                        check.error = f"Server error ({resp.status})"
                    else:
                        check.error = f"Unexpected status {resp.status}"
            except asyncio.TimeoutError:
                check.error = "Request timed out"
                check.response_time_ms = LINK_CHECK_TIMEOUT * 1000
            except (aiohttp.ClientError, Exception) as exc:
                check.error = f"{type(exc).__name__}: {exc}"

            # Amazon dog page detection
            if check.is_valid and _is_amazon_link(link.url):
                asin = _extract_asin(link.url)
                if asin:
                    amz = await _check_amazon_product(asin, session)
                    if not amz["available"]:
                        check.is_valid = False
                        check.error = amz.get("error", "Amazon product unavailable")

        # Update link status
        if check.is_valid:
            link.status = "active"
        elif check.status_code in (404, 410) or (check.error and "removed" in check.error.lower()):
            link.status = "broken"
        link.last_checked = check.checked_at

        self._checks.append(check.to_dict())
        self._save_checks()
        self._save_links()
        return check

    async def check_site_links(self, site_id: str) -> list[LinkCheck]:
        """Check all links for a site in parallel with concurrency limit."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required. Install: pip install aiohttp")

        site_links = [l for l in self._links.values() if l.site_id == site_id]
        if not site_links:
            logger.info("No links to check for site '%s'", site_id)
            return []

        semaphore = asyncio.Semaphore(LINK_CHECK_CONCURRENCY)
        results: list[LinkCheck] = []

        async def _check_with_semaphore(link: AffiliateLink) -> LinkCheck:
            async with semaphore:
                return await self.check_link(link)

        tasks = [_check_with_semaphore(link) for link in site_links]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        valid_count = sum(1 for r in results if r.is_valid)
        broken_count = len(results) - valid_count
        logger.info("Checked %d links for %s: %d valid, %d broken",
                     len(results), site_id, valid_count, broken_count)
        return results

    def check_site_links_sync(self, site_id: str) -> list[LinkCheck]:
        """Synchronous wrapper for check_site_links."""
        return asyncio.run(self.check_site_links(site_id))

    async def check_all_links(self) -> dict[str, list[LinkCheck]]:
        """Check all links across all sites."""
        results: dict[str, list[LinkCheck]] = {}
        for site_id in ALL_SITE_IDS:
            site_links = [l for l in self._links.values() if l.site_id == site_id]
            if site_links:
                checks = await self.check_site_links(site_id)
                results[site_id] = checks
        return results

    def check_all_links_sync(self) -> dict[str, list[LinkCheck]]:
        """Synchronous wrapper for check_all_links."""
        return asyncio.run(self.check_all_links())

    def get_broken_links(self, site_id: Optional[str] = None) -> list[AffiliateLink]:
        """Return all links with status 'broken'."""
        links = self._links.values()
        if site_id:
            links = [l for l in links if l.site_id == site_id]
        return [l for l in links if l.status == "broken"]

    def get_expired_links(self, site_id: Optional[str] = None) -> list[AffiliateLink]:
        """Return all links with status 'expired'."""
        links = self._links.values()
        if site_id:
            links = [l for l in links if l.site_id == site_id]
        return [l for l in links if l.status == "expired"]

    # -- Link Management ----------------------------------------------------

    def add_link(
        self,
        site_id: str,
        post_id: int,
        url: str,
        program: str,
        product_name: Optional[str] = None,
        anchor_text: str = "",
    ) -> AffiliateLink:
        """Manually add an affiliate link to the database."""
        link = AffiliateLink(
            link_id=_make_id(),
            site_id=site_id,
            post_id=post_id,
            url=url,
            affiliate_program=program,
            product_name=product_name,
            anchor_text=anchor_text,
            location_in_post="body",
            status="active",
            created_at=_now_iso(),
        )
        self._links[link.link_id] = link
        self._save_links()
        logger.info("Added link: %s (%s) for %s post %d", url, program, site_id, post_id)
        return link

    def update_link(self, link_id: str, **kwargs: Any) -> AffiliateLink:
        """Update fields on an existing link."""
        link = self._links.get(link_id)
        if link is None:
            raise ValueError(f"Link not found: {link_id}")

        for key, value in kwargs.items():
            if hasattr(link, key):
                setattr(link, key, value)
            else:
                logger.warning("Unknown field '%s' on AffiliateLink, ignoring", key)

        self._save_links()
        return link

    def replace_link(self, link_id: str, new_url: str, reason: str = "") -> AffiliateLink:
        """Replace a link's URL, preserving history in metadata."""
        link = self._links.get(link_id)
        if link is None:
            raise ValueError(f"Link not found: {link_id}")

        old_url = link.url
        replacements = link.metadata.get("replacements", [])
        replacements.append({
            "old_url": old_url,
            "new_url": new_url,
            "reason": reason,
            "replaced_at": _now_iso(),
        })
        link.metadata["replacements"] = replacements
        link.url = new_url
        link.status = "active"
        link.last_checked = ""

        # Re-identify program from new URL
        new_program = _identify_affiliate_program(new_url)
        if new_program:
            link.affiliate_program = new_program

        self._save_links()
        logger.info("Replaced link %s: %s -> %s (reason: %s)", link_id, old_url, new_url, reason)
        return link

    def deactivate_link(self, link_id: str, reason: str = "") -> AffiliateLink:
        """Mark a link as expired/deactivated."""
        link = self._links.get(link_id)
        if link is None:
            raise ValueError(f"Link not found: {link_id}")

        link.status = "expired"
        link.metadata["deactivated_at"] = _now_iso()
        link.metadata["deactivation_reason"] = reason

        self._save_links()
        logger.info("Deactivated link %s (reason: %s)", link_id, reason)
        return link

    def get_links(
        self,
        site_id: Optional[str] = None,
        program: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[AffiliateLink]:
        """Query links with optional filters."""
        results = list(self._links.values())
        if site_id:
            results = [l for l in results if l.site_id == site_id]
        if program:
            results = [l for l in results if l.affiliate_program == program]
        if status:
            results = [l for l in results if l.status == status]
        return results

    def search_links(self, query: str) -> list[AffiliateLink]:
        """Search links by product name, URL, or anchor text."""
        query_lower = query.lower()
        results: list[AffiliateLink] = []
        for link in self._links.values():
            searchable = " ".join([
                link.url,
                link.product_name or "",
                link.anchor_text,
                link.affiliate_program,
            ]).lower()
            if query_lower in searchable:
                results.append(link)
        return results

    # -- Performance Tracking -----------------------------------------------

    def record_click(self, link_id: str) -> AffiliateLink:
        """Increment the click count for a link."""
        link = self._links.get(link_id)
        if link is None:
            raise ValueError(f"Link not found: {link_id}")
        link.clicks += 1
        self._save_links()
        return link

    def record_conversion(self, link_id: str, amount: float) -> AffiliateLink:
        """Record a conversion (sale) for a link."""
        link = self._links.get(link_id)
        if link is None:
            raise ValueError(f"Link not found: {link_id}")
        link.conversions += 1
        link.revenue = round(link.revenue + amount, 2)
        self._save_links()
        logger.info("Recorded conversion for %s: $%.2f (total: $%.2f)",
                     link_id, amount, link.revenue)
        return link

    def import_earnings(self, program: str, data: list[dict]) -> int:
        """Import earnings and match to existing links by URL or product name.
        Returns count of matched links."""
        matched = 0
        for entry in data:
            product_name = entry.get("product_name", "")
            url = entry.get("url", "")
            amount = float(entry.get("amount", 0))

            # Match by URL first, then by product name
            ml = None
            if url:
                ml = next((l for l in self._links.values()
                           if l.affiliate_program == program and l.url == url), None)
            if ml is None and product_name:
                pn = product_name.lower()
                ml = next((l for l in self._links.values()
                           if l.affiliate_program == program and l.product_name
                           and pn in l.product_name.lower()), None)
            if ml:
                ml.conversions += 1
                ml.revenue = round(ml.revenue + amount, 2)
                matched += 1

            self._earnings.append({
                "program": program, "product_name": product_name, "url": url,
                "amount": amount, "date": entry.get("date", _today_iso()),
                "matched_link_id": ml.link_id if ml else None, "imported_at": _now_iso(),
            })

        self._save_links()
        self._save_earnings()
        logger.info("Imported %d earnings for %s, matched %d", len(data), program, matched)
        return matched

    def get_performance(
        self,
        site_id: Optional[str] = None,
        program: Optional[str] = None,
        days: int = 30,
    ) -> dict:
        """Get aggregate performance metrics."""
        links = self.get_links(site_id=site_id, program=program)
        cutoff = _days_ago_iso(days)

        # Filter earnings within the date range
        period_earnings = [
            e for e in self._earnings
            if e.get("date", "") >= cutoff
            and (program is None or e.get("program") == program)
        ]
        period_revenue = sum(e.get("amount", 0) for e in period_earnings)

        total_clicks = sum(l.clicks for l in links)
        total_conversions = sum(l.conversions for l in links)
        total_revenue = sum(l.revenue for l in links)
        active_count = sum(1 for l in links if l.status == "active")
        broken_count = sum(1 for l in links if l.status == "broken")

        return {
            "total_links": len(links),
            "active_links": active_count,
            "broken_links": broken_count,
            "total_clicks": total_clicks,
            "total_conversions": total_conversions,
            "total_revenue": round(total_revenue, 2),
            "period_revenue": round(period_revenue, 2),
            "period_days": days,
            "conversion_rate": round(total_conversions / total_clicks * 100, 2) if total_clicks > 0 else 0.0,
            "revenue_per_click": round(total_revenue / total_clicks, 4) if total_clicks > 0 else 0.0,
        }

    def top_performing_links(self, count: int = 20, days: int = 30) -> list[AffiliateLink]:
        """Return top performing links sorted by revenue."""
        links = [l for l in self._links.values() if l.status == "active"]
        links.sort(key=lambda l: l.revenue, reverse=True)
        return links[:count]

    def top_performing_posts(self, count: int = 10, days: int = 30) -> list[dict]:
        """Return top performing posts by aggregate affiliate revenue."""
        stats: dict[str, dict] = {}
        for link in self._links.values():
            key = f"{link.site_id}:{link.post_id}"
            s = stats.setdefault(key, {"site_id": link.site_id, "post_id": link.post_id,
                                       "total_clicks": 0, "total_conversions": 0,
                                       "total_revenue": 0.0, "link_count": 0})
            s["total_clicks"] += link.clicks
            s["total_conversions"] += link.conversions
            s["total_revenue"] = round(s["total_revenue"] + link.revenue, 2)
            s["link_count"] += 1
        return sorted(stats.values(), key=lambda p: p["total_revenue"], reverse=True)[:count]

    def conversion_rate_by_program(self, days: int = 30) -> dict[str, float]:
        """Return conversion rate per affiliate program."""
        program_clicks: dict[str, int] = {}
        program_conversions: dict[str, int] = {}

        for link in self._links.values():
            prog = link.affiliate_program
            program_clicks[prog] = program_clicks.get(prog, 0) + link.clicks
            program_conversions[prog] = program_conversions.get(prog, 0) + link.conversions

        result: dict[str, float] = {}
        for prog in program_clicks:
            clicks = program_clicks[prog]
            convs = program_conversions.get(prog, 0)
            result[prog] = round(convs / clicks * 100, 2) if clicks > 0 else 0.0

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    # -- Placement Suggestions ----------------------------------------------

    async def suggest_placements(self, site_id: str, post_id: int) -> list[PlacementSuggestion]:
        """Analyze post content and suggest affiliate link placements for unlinked product mentions."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required. Install: pip install aiohttp")
        domain, wp_user, app_password = _get_site_auth(site_id)
        if not domain:
            return []

        headers = self._build_wp_headers(wp_user, app_password)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"https://{domain}/wp-json/wp/v2/posts/{post_id}", headers=headers,
                    params={"_fields": "id,title,content,categories"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        return []
                    post = await resp.json()
            except Exception as exc:
                logger.error("Failed to fetch post %d from %s: %s", post_id, site_id, exc)
                return []

        html_content = post.get("content", {}).get("rendered", "")
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_content)).strip().lower()

        site_config = _get_site_config(site_id)
        niche = site_config.get("niche", "") if site_config else ""
        niche_keywords = self._get_niche_product_keywords(niche)

        suggestions: list[PlacementSuggestion] = []
        post_links = [l for l in self._links.values()
                      if l.site_id == site_id and l.post_id == post_id]
        for keyword, info in niche_keywords.items():
            if keyword.lower() not in text:
                continue
            if any(keyword.lower() in (l.product_name or "").lower() for l in post_links):
                continue
            suggestions.append(PlacementSuggestion(
                site_id=site_id, post_id=post_id,
                product_name=info.get("name", keyword), suggested_anchor=info.get("anchor", keyword),
                suggested_url=info.get("url", ""), relevance_score=info.get("relevance", 0.7),
                estimated_ctr=info.get("ctr", 0.02), reason=info.get("reason", f"Mentions '{keyword}' without link"),
            ))
        suggestions.sort(key=lambda s: s.relevance_score, reverse=True)
        return suggestions[:10]

    def suggest_placements_sync(self, site_id: str, post_id: int) -> list[PlacementSuggestion]:
        """Synchronous wrapper for suggest_placements."""
        return asyncio.run(self.suggest_placements(site_id, post_id))

    def find_unlinked_mentions(self, site_id: str) -> list[dict]:
        """Find posts that were scanned but have no active affiliate links."""
        all_post_ids = {l.post_id for l in self._links.values() if l.site_id == site_id}
        active_posts = {l.post_id for l in self._links.values()
                        if l.site_id == site_id and l.status == "active"}
        return [{"site_id": site_id, "post_id": pid, "reason": "No active affiliate links"}
                for pid in (all_post_ids - active_posts)]

    def _get_niche_product_keywords(self, niche: str) -> dict[str, dict]:
        """Return product keyword -> info mapping for a niche.

        Each value has: name, anchor, url, relevance, ctr, reason.
        Compact format: (name, anchor, relevance, ctr, reason_suffix).
        """
        def _kw(name: str, anchor: str, rel: float, ctr: float, reason: str) -> dict:
            return {"name": name, "anchor": anchor, "url": "", "relevance": rel,
                    "ctr": ctr, "reason": reason}

        niche_maps: dict[str, dict[str, dict]] = {
            "smart-home-tech": {
                "echo dot":        _kw("Amazon Echo Dot", "Echo Dot", 0.9, 0.04, "Smart speaker mention"),
                "ring doorbell":   _kw("Ring Video Doorbell", "Ring Doorbell", 0.9, 0.03, "Smart doorbell mention"),
                "nest thermostat": _kw("Google Nest Thermostat", "Nest Thermostat", 0.85, 0.03, "Smart thermostat mention"),
                "philips hue":     _kw("Philips Hue Smart Bulbs", "Philips Hue", 0.85, 0.03, "Smart lighting mention"),
                "smart plug":      _kw("Smart Plug", "smart plug", 0.75, 0.02, "Smart plug mention"),
                "alexa":           _kw("Amazon Alexa Device", "Alexa device", 0.8, 0.03, "Alexa product mention"),
            },
            "witchcraft-spirituality": {
                "candle":          _kw("Ritual Candles", "ritual candles", 0.8, 0.025, "Candle product mention"),
                "crystal":         _kw("Healing Crystals Set", "crystal set", 0.85, 0.03, "Crystal product mention"),
                "tarot deck":      _kw("Tarot Card Deck", "tarot deck", 0.9, 0.035, "Tarot deck mention"),
                "book of shadows": _kw("Book of Shadows Journal", "Book of Shadows", 0.85, 0.03, "Witchcraft journal mention"),
                "incense":         _kw("Incense and Holders", "incense", 0.75, 0.02, "Incense product mention"),
                "altar":           _kw("Altar Supplies", "altar supplies", 0.7, 0.02, "Altar supplies mention"),
            },
            "crystal-magic": {
                "amethyst":     _kw("Amethyst Crystal", "amethyst", 0.9, 0.03, "Crystal product mention"),
                "rose quartz":  _kw("Rose Quartz Crystal", "rose quartz", 0.9, 0.03, "Crystal product mention"),
                "clear quartz": _kw("Clear Quartz Crystal", "clear quartz", 0.9, 0.03, "Crystal product mention"),
                "crystal grid": _kw("Crystal Grid Kit", "crystal grid kit", 0.85, 0.025, "Crystal grid mention"),
            },
            "ai-technology": {
                "chatgpt":    _kw("ChatGPT Plus Subscription", "ChatGPT Plus", 0.8, 0.02, "AI tool mention"),
                "midjourney": _kw("Midjourney Subscription", "Midjourney", 0.8, 0.02, "AI tool mention"),
                "gpu":        _kw("NVIDIA GPU", "GPU", 0.7, 0.015, "Hardware product mention"),
            },
            "productivity-journaling": {
                "leuchtturm": _kw("Leuchtturm1917 Notebook", "Leuchtturm1917", 0.9, 0.04, "Popular journal notebook"),
                "micron pen": _kw("Sakura Pigma Micron Pens", "Micron pens", 0.85, 0.03, "Popular journaling pens"),
                "washi tape": _kw("Washi Tape Set", "washi tape", 0.8, 0.025, "Journal decoration product"),
                "stencil":    _kw("Journal Stencil Set", "bullet journal stencils", 0.8, 0.025, "Journal stencil mention"),
            },
        }

        keywords = niche_maps.get(niche, {})
        # Generic commercial-intent keywords for any niche
        generic = {
            "best seller": _kw("Best Seller", "best seller", 0.5, 0.015, "Commercial intent keyword"),
            "on amazon":   _kw("Amazon Product", "on Amazon", 0.6, 0.02, "Amazon mention without link"),
            "buy now":     _kw("Product Purchase", "buy now", 0.6, 0.02, "Purchase intent without link"),
        }
        return {**generic, **keywords}

    # -- Reporting ----------------------------------------------------------

    def daily_report(self) -> AffiliateReport:
        """Generate a report for today."""
        return self._build_report("day")

    def weekly_report(self) -> AffiliateReport:
        """Generate a report for the current week."""
        return self._build_report("week")

    def monthly_report(self) -> AffiliateReport:
        """Generate a report for the current month."""
        return self._build_report("month")

    def _build_report(self, period: str) -> AffiliateReport:
        """Build an AffiliateReport for the given period."""
        all_links = list(self._links.values())
        top = sorted(all_links, key=lambda l: l.revenue, reverse=True)[:10]
        report = AffiliateReport(
            period=period,
            total_links=len(all_links),
            active_links=sum(1 for l in all_links if l.status == "active"),
            broken_links=sum(1 for l in all_links if l.status == "broken"),
            clicks=sum(l.clicks for l in all_links),
            conversions=sum(l.conversions for l in all_links),
            revenue=round(sum(l.revenue for l in all_links), 2),
            top_products=[
                {"link_id": l.link_id, "product_name": l.product_name or l.anchor_text,
                 "site_id": l.site_id, "program": l.affiliate_program,
                 "clicks": l.clicks, "conversions": l.conversions, "revenue": l.revenue}
                for l in top if l.revenue > 0
            ],
            top_posts=self.top_performing_posts(count=10),
            issues=self.check_for_issues(),
        )
        _save_json(REPORTS_DIR / f"{period}-{_today_iso()}.json", report.to_dict())
        return report

    def format_report(self, report: AffiliateReport, style: str = "text") -> str:
        """Format an AffiliateReport for display.

        Styles: 'text' (plain text), 'markdown', 'json'.
        """
        if style == "json":
            return json.dumps(report.to_dict(), indent=2, default=str)

        if style == "markdown":
            return self._format_report_markdown(report)

        return self._format_report_text(report)

    def _format_report_text(self, report: AffiliateReport) -> str:
        """Plain text report for WhatsApp / Telegram."""
        lines = [
            f"AFFILIATE REPORT ({report.period.upper()})", f"Generated: {_today_iso()}",
            "=" * 40, "",
            f"  Links: {report.total_links} (active:{report.active_links} broken:{report.broken_links})",
            f"  Clicks: {report.clicks:,}  Conversions: {report.conversions:,}  Rate: {report.conversion_rate:.2f}%",
            f"  Revenue: ${report.revenue:,.2f}", "",
        ]
        if report.top_products:
            lines.append("TOP PRODUCTS:")
            for i, p in enumerate(report.top_products[:5], 1):
                lines.append(f"  {i}. {p.get('product_name','?')[:30]:<32} ${p.get('revenue',0):>8,.2f}")
            lines.append("")
        if report.top_posts:
            lines.append("TOP POSTS:")
            for i, p in enumerate(report.top_posts[:5], 1):
                lines.append(f"  {i}. {p.get('site_id','?')}/{p.get('post_id','?'):<18} "
                             f"${p.get('total_revenue',0):>8,.2f}")
            lines.append("")
        if report.issues:
            lines.append(f"ISSUES ({len(report.issues)}):")
            for issue in report.issues[:5]:
                lines.append(f"  [{issue.get('severity','info').upper()}] {issue.get('message','')}")
        return "\n".join(lines)

    def _format_report_markdown(self, report: AffiliateReport) -> str:
        """Rich markdown report."""
        lines: list[str] = [
            f"# Affiliate Report: {report.period.title()}",
            f"**Generated:** {_today_iso()}", "",
            "## Overview",
            "| Metric | Value |", "|--------|-------|",
            f"| Total Links | {report.total_links} |",
            f"| Active | {report.active_links} |",
            f"| Broken | {report.broken_links} |",
            f"| Clicks | {report.clicks:,} |",
            f"| Conversions | {report.conversions:,} |",
            f"| Revenue | ${report.revenue:,.2f} |",
            f"| Conv Rate | {report.conversion_rate:.2f}% |", "",
        ]
        if report.top_products:
            lines += ["## Top Products", "| # | Product | Program | Revenue |",
                       "|---|---------|---------|---------|"]
            for i, p in enumerate(report.top_products[:10], 1):
                lines.append(f"| {i} | {p.get('product_name','?')[:40]} | "
                             f"{p.get('program','?')} | ${p.get('revenue',0):,.2f} |")
            lines.append("")
        if report.top_posts:
            lines += ["## Top Posts", "| # | Site | Post | Revenue |",
                       "|---|------|------|---------|"]
            for i, p in enumerate(report.top_posts[:10], 1):
                lines.append(f"| {i} | {p.get('site_id','?')} | {p.get('post_id','?')} | "
                             f"${p.get('total_revenue',0):,.2f} |")
            lines.append("")
        if report.issues:
            lines.append(f"## Issues ({len(report.issues)})")
            for issue in report.issues:
                sev = issue.get("severity", "info")
                icon = "!!!" if sev == "critical" else "!" if sev == "warning" else "-"
                lines.append(f"- {icon} **[{sev.upper()}]** {issue.get('message', '')}")
        return "\n".join(lines)

    def _aggregate_breakdown(self, key_fn, extra_fn=None) -> dict:
        """Shared helper for program_breakdown and site_breakdown."""
        breakdown: dict[str, dict] = {}
        for link in self._links.values():
            key = key_fn(link)
            if key not in breakdown:
                stats = {"total_links": 0, "active_links": 0, "broken_links": 0,
                         "clicks": 0, "conversions": 0, "revenue": 0.0}
                if extra_fn:
                    stats.update(extra_fn(link))
                breakdown[key] = stats
            s = breakdown[key]
            s["total_links"] += 1
            if link.status == "active":
                s["active_links"] += 1
            elif link.status == "broken":
                s["broken_links"] += 1
            s["clicks"] += link.clicks
            s["conversions"] += link.conversions
            s["revenue"] = round(s["revenue"] + link.revenue, 2)
        for s in breakdown.values():
            s["conversion_rate"] = round(s["conversions"] / s["clicks"] * 100, 2) if s["clicks"] > 0 else 0.0
        return dict(sorted(breakdown.items(), key=lambda x: x[1]["revenue"], reverse=True))

    def program_breakdown(self, days: int = 30) -> dict:
        """Revenue and link counts broken down by affiliate program."""
        return self._aggregate_breakdown(
            key_fn=lambda l: l.affiliate_program,
            extra_fn=lambda l: {"program": l.affiliate_program},
        )

    def site_breakdown(self, days: int = 30) -> dict:
        """Revenue and link counts broken down by site."""
        return self._aggregate_breakdown(
            key_fn=lambda l: l.site_id,
            extra_fn=lambda l: {"site_id": l.site_id, "domain": SITE_DOMAIN_MAP.get(l.site_id, "")},
        )

    # -- Alerts -------------------------------------------------------------

    def check_for_issues(self) -> list[dict]:
        """Run all issue detection rules. Returns list of dicts with severity, message, category, details."""
        issues: list[dict] = []

        def _issue(sev: str, msg: str, cat: str, **details: Any) -> None:
            issues.append({"severity": sev, "message": msg, "category": cat, "details": details})

        # 1. Broken links (critical if >5)
        broken = self.get_broken_links()
        if broken:
            _issue("critical" if len(broken) > 5 else "warning",
                   f"{len(broken)} broken affiliate link(s) across all sites", "broken_links",
                   count=len(broken), sites=list({l.site_id for l in broken}),
                   sample=[{"id": l.link_id, "url": l.url[:80], "site": l.site_id} for l in broken[:5]])

        # 2. Expired links
        expired = self.get_expired_links()
        if expired:
            _issue("warning", f"{len(expired)} expired link(s) need replacement", "expired_links",
                   count=len(expired), sites=list({l.site_id for l in expired}))

        # 3. Low conversion (50+ clicks, 0 conversions)
        for link in self._links.values():
            if link.clicks >= 50 and link.conversions == 0 and link.status == "active":
                _issue("info", f"Low-converting: '{link.product_name or link.anchor_text}' "
                       f"on {link.site_id} ({link.clicks} clicks, 0 conv)", "low_conversion",
                       link_id=link.link_id, site_id=link.site_id, post_id=link.post_id)

        # 4. Sites with no tracked links
        sites_with = {l.site_id for l in self._links.values() if l.status == "active"}
        sites_without = set(ALL_SITE_IDS) - sites_with
        if sites_without and self._links:
            _issue("warning", f"{len(sites_without)} site(s) have no tracked affiliate links",
                   "missing_links", sites=sorted(sites_without))

        # 5. Stale checks (>7 days old)
        stale_cutoff = _days_ago_iso(7)
        stale = [l for l in self._links.values()
                 if l.status == "active" and l.last_checked and l.last_checked < stale_cutoff]
        if stale:
            _issue("info", f"{len(stale)} active link(s) not checked in 7+ days", "stale_checks",
                   count=len(stale), sites=list({l.site_id for l in stale}))

        # 6. Low commission rates
        for name, cfg in self._program_configs.items():
            if cfg.commission_rate is not None and cfg.commission_rate < 0.02:
                _issue("info", f"Low commission for '{name}': {cfg.commission_rate*100:.1f}%",
                       "program_change", program=name, rate=cfg.commission_rate)

        return issues

    # -- Async Wrappers -----------------------------------------------------

    async def aget_broken_links(self, site_id: Optional[str] = None) -> list[AffiliateLink]:
        """Async wrapper for get_broken_links."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_broken_links(site_id))

    async def aget_performance(self, site_id: Optional[str] = None,
                               program: Optional[str] = None, days: int = 30) -> dict:
        """Async wrapper for get_performance."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_performance(site_id, program, days))

    async def adaily_report(self) -> AffiliateReport:
        """Async wrapper for daily_report."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.daily_report)

    async def aweekly_report(self) -> AffiliateReport:
        """Async wrapper for weekly_report."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.weekly_report)

    async def amonthly_report(self) -> AffiliateReport:
        """Async wrapper for monthly_report."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.monthly_report)

    async def aformat_report(self, report: AffiliateReport, style: str = "text") -> str:
        """Async wrapper for format_report."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.format_report(report, style))


_manager_instance: Optional[AffiliateManager] = None


def get_manager() -> AffiliateManager:
    """Return the singleton AffiliateManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = AffiliateManager()
    return _manager_instance


def _cli_main() -> None:
    """CLI entry point: python -m src.affiliate_manager <command> [options]."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="affiliate_manager",
        description="OpenClaw Empire Affiliate Link Manager -- CLI Interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- scan ---
    p_scan = subparsers.add_parser("scan", help="Discover affiliate links on a site")
    p_scan.add_argument("--site", required=True, help="Site ID to scan")
    p_scan.add_argument("--max-posts", type=int, default=200, help="Max posts to scan (default: 200)")

    # --- check ---
    p_check = subparsers.add_parser("check", help="Check link health for a site")
    p_check.add_argument("--site", required=True, help="Site ID to check")

    # --- broken ---
    p_broken = subparsers.add_parser("broken", help="List all broken links")
    p_broken.add_argument("--site", help="Filter by site ID (optional)")

    # --- report ---
    p_report = subparsers.add_parser("report", help="Generate performance report")
    p_report.add_argument("--period", choices=["day", "week", "month"],
                          default="month", help="Report period (default: month)")
    p_report.add_argument("--format", choices=["text", "markdown", "json"],
                          default="text", help="Output format (default: text)")

    # --- top ---
    p_top = subparsers.add_parser("top", help="Top performing links")
    p_top.add_argument("--count", type=int, default=20, help="Number of results (default: 20)")
    p_top.add_argument("--days", type=int, default=30, help="Look-back period in days (default: 30)")

    # --- suggest ---
    p_suggest = subparsers.add_parser("suggest", help="Suggest affiliate placements for a post")
    p_suggest.add_argument("--site", required=True, help="Site ID")
    p_suggest.add_argument("--post-id", type=int, required=True, help="WordPress post ID")

    # --- replace ---
    p_replace = subparsers.add_parser("replace", help="Replace a link URL")
    p_replace.add_argument("--link-id", required=True, help="Link ID to replace")
    p_replace.add_argument("--new-url", required=True, help="New URL")
    p_replace.add_argument("--reason", default="", help="Reason for replacement")

    # --- programs ---
    subparsers.add_parser("programs", help="List affiliate program configurations")

    # --- stats ---
    p_stats = subparsers.add_parser("stats", help="Stats for a specific program")
    p_stats.add_argument("--program", required=True, help="Program name (amazon, shareasale, cj, impact)")
    p_stats.add_argument("--days", type=int, default=30, help="Look-back period in days (default: 30)")

    # --- search ---
    p_search = subparsers.add_parser("search", help="Search links by product, URL, or anchor text")
    p_search.add_argument("query", help="Search query string")

    # --- sites ---
    subparsers.add_parser("sites", help="Show per-site breakdown")

    # --- issues ---
    subparsers.add_parser("issues", help="Check for current issues")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    manager = get_manager()

    if args.command == "scan":
        print(f"Scanning {args.site} (max {args.max_posts} posts)...")
        links = manager.scan_site_sync(args.site, max_posts=args.max_posts)
        prog_counts: dict[str, int] = {}
        for link in links:
            prog_counts[link.affiliate_program] = prog_counts.get(link.affiliate_program, 0) + 1
        print(f"Discovered {len(links)} affiliate links:")
        for prog, count in sorted(prog_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {prog}: {count}")
        print(f"Total tracked for {args.site}: {len(manager.get_links(site_id=args.site))}")

    elif args.command == "check":
        print(f"Checking link health for {args.site}...")
        checks = manager.check_site_links_sync(args.site)
        valid = sum(1 for c in checks if c.is_valid)
        print(f"Results: {valid} valid, {len(checks)-valid} broken / {len(checks)} total")
        for check in checks:
            if not check.is_valid:
                link = manager._links.get(check.link_id)
                print(f"  [{check.status_code}] {(link.url[:60] if link else check.link_id)}"
                      f"{('  ' + check.error) if check.error else ''}")

    elif args.command == "broken":
        broken = manager.get_broken_links(site_id=getattr(args, "site", None))
        if not broken:
            print("No broken links found.")
        else:
            print(f"BROKEN LINKS ({len(broken)})\n" + "=" * 70)
            for link in broken:
                name = (link.product_name or link.anchor_text)[:35]
                print(f"  [{link.site_id}] {name:<37} {link.url[:50]}")

    elif args.command == "report":
        report_fn = {"day": manager.daily_report, "week": manager.weekly_report}.get(
            args.period, manager.monthly_report)
        print(manager.format_report(report_fn(), style=args.format))

    elif args.command == "top":
        links = manager.top_performing_links(count=args.count, days=args.days)
        print(f"TOP {args.count} AFFILIATE LINKS (last {args.days} days)\n" + "=" * 70)
        for i, link in enumerate(links, 1):
            name = (link.product_name or link.anchor_text or "Unknown")[:35]
            print(f"  {i:>2}. {name:<37} ${link.revenue:>8,.2f}  "
                  f"({link.clicks} clicks, {link.conversions} conv)")

    elif args.command == "suggest":
        suggestions = manager.suggest_placements_sync(args.site, args.post_id)
        if not suggestions:
            print("No placement suggestions found.")
        else:
            print(f"SUGGESTED PLACEMENTS ({len(suggestions)}):\n" + "=" * 60)
            for i, s in enumerate(suggestions, 1):
                print(f"  {i}. {s.product_name} (rel:{s.relevance_score:.0%} ctr:{s.estimated_ctr:.1%})")
                print(f"     {s.reason}")

    elif args.command == "replace":
        try:
            link = manager.replace_link(args.link_id, args.new_url, args.reason)
            print(f"Replaced: {link.link_id} -> {link.url} ({link.affiliate_program})")
        except ValueError as exc:
            print(f"Error: {exc}"); sys.exit(1)

    elif args.command == "programs":
        print("AFFILIATE PROGRAM CONFIGURATIONS\n" + "=" * 60)
        for name, cfg in manager._program_configs.items():
            parts = [f"{name.upper()}:"]
            if cfg.commission_rate is not None:
                parts.append(f"rate={cfg.commission_rate*100:.1f}%")
            if cfg.cookie_days is not None:
                parts.append(f"cookies={cfg.cookie_days}d")
            if cfg.payment_threshold is not None:
                parts.append(f"payout=${cfg.payment_threshold:.0f}")
            if cfg.api_key_env:
                parts.append(f"env={cfg.api_key_env}({'SET' if os.environ.get(cfg.api_key_env) else 'NOT SET'})")
            print(f"  {' | '.join(parts)}")

    elif args.command == "stats":
        perf = manager.get_performance(program=args.program, days=args.days)
        cr = manager.conversion_rate_by_program(days=args.days)
        print(f"STATS: {args.program.upper()} (last {args.days} days)\n" + "=" * 45)
        for label, key in [("Links", "total_links"), ("Active", "active_links"),
                           ("Broken", "broken_links")]:
            print(f"  {label + ':':<17} {perf[key]}")
        print(f"  {'Clicks:':<17} {perf['total_clicks']:,}")
        print(f"  {'Conversions:':<17} {perf['total_conversions']:,}")
        print(f"  {'Revenue:':<17} ${perf['total_revenue']:,.2f}")
        print(f"  {'Conv rate:':<17} {cr.get(args.program, 0.0):.2f}%")

    elif args.command == "search":
        results = manager.search_links(args.query)
        if not results:
            print(f"No links found matching '{args.query}'")
        else:
            print(f"SEARCH: '{args.query}' ({len(results)} found)\n" + "=" * 70)
            for link in results[:20]:
                name = (link.product_name or link.anchor_text or "?")[:35]
                print(f"  [{link.status.upper():>8}] {name:<37} ({link.affiliate_program})")
                print(f"           {link.site_id}:{link.post_id}  {link.url[:55]}")

    elif args.command == "sites":
        breakdown = manager.site_breakdown()
        if not breakdown:
            print("No data. Run 'scan --site <id>' first.")
        else:
            print("AFFILIATE LINKS BY SITE\n" + "=" * 70)
            hdr = f"  {'Site':<22} {'Links':>5} {'OK':>4} {'Bad':>4} {'Clicks':>7} {'Rev':>9}"
            print(hdr + "\n" + "-" * 70)
            t_l = t_a = t_b = t_c = 0; t_r = 0.0
            for sid, s in breakdown.items():
                print(f"  {sid:<22} {s['total_links']:>5} {s['active_links']:>4} "
                      f"{s['broken_links']:>4} {s['clicks']:>7,} ${s['revenue']:>8,.2f}")
                t_l += s["total_links"]; t_a += s["active_links"]
                t_b += s["broken_links"]; t_c += s["clicks"]; t_r += s["revenue"]
            print("-" * 70)
            print(f"  {'TOTAL':<22} {t_l:>5} {t_a:>4} {t_b:>4} {t_c:>7,} ${t_r:>8,.2f}")

    elif args.command == "issues":
        issues = manager.check_for_issues()
        if not issues:
            print("No issues detected.")
        else:
            print(f"AFFILIATE ISSUES ({len(issues)})\n" + "=" * 60)
            for issue in issues:
                sev = issue.get("severity", "info").upper()
                pfx = "!!!" if sev == "CRITICAL" else " ! " if sev == "WARNING" else "   "
                print(f"  {pfx} [{sev}] {issue.get('message', '')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli_main()
