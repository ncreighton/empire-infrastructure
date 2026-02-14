"""
WordPress REST API Client for the OpenClaw Publishing Empire.

Comprehensive async/sync client for managing all 16 WordPress sites through
the WP REST API. Handles post CRUD, media uploads, RankMath SEO fields,
LiteSpeed cache purging, bulk operations, and site health checks.

Usage:
    # Quick publish
    from src.wordpress_client import publish_to_site
    publish_to_site("witchcraft", "Full Moon Ritual Guide", "<p>Content...</p>")

    # Full client
    from src.wordpress_client import get_site_client
    client = get_site_client("smarthome")
    post = client.create_post_sync("Smart Lock Review", "<p>...</p>", status="publish")

    # Empire-wide operations
    from src.wordpress_client import get_empire_manager
    manager = get_empire_manager()
    health = manager.health_check_all_sync()

CLI:
    python -m src.wordpress_client health
    python -m src.wordpress_client dashboard
    python -m src.wordpress_client gaps
    python -m src.wordpress_client publish --site witchcraft --title "..." --content "..."
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("wordpress_client")
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
# Constants
# ---------------------------------------------------------------------------

SITE_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "configs" / "site-registry.json"

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

# Posting frequency targets (posts per week)
FREQUENCY_TARGETS: Dict[str, float] = {
    "daily": 7.0,
    "3x-weekly": 3.0,
    "2x-weekly": 2.0,
    "weekly": 1.0,
}

# Default RankMath robots meta
DEFAULT_ROBOTS_META = "index,follow,max-snippet:-1,max-image-preview:large,max-video-preview:-1"

# WP REST API pagination limit
WP_MAX_PER_PAGE = 100


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WordPressError(Exception):
    """Base exception for WordPress API errors."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class AuthenticationError(WordPressError):
    """Raised on 401/403 responses."""
    pass


class NotFoundError(WordPressError):
    """Raised on 404 responses."""
    pass


class RateLimitError(WordPressError):
    """Raised on 429 responses after all retries exhausted."""
    pass


class SiteNotConfiguredError(WordPressError):
    """Raised when a site lacks credentials."""
    pass


class SiteNotFoundError(WordPressError):
    """Raised when a site ID is not in the registry."""
    pass


# ---------------------------------------------------------------------------
# SiteConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class SiteConfig:
    """Configuration for a single WordPress site."""

    domain: str
    site_id: str
    brand_color: str = "#000000"
    accent_color: str = "#666666"
    voice: str = ""
    niche: str = ""
    posting_frequency: str = "2x-weekly"
    priority: int = 99
    theme: str = "blocksy"
    wp_user: str = ""
    app_password: str = ""
    flagship: bool = False

    @property
    def base_url(self) -> str:
        """WP REST API root URL."""
        return f"https://{self.domain}/wp-json"

    @property
    def api_url(self) -> str:
        """WP REST API v2 base URL."""
        return f"https://{self.domain}/wp-json/wp/v2"

    @property
    def auth_header(self) -> str:
        """Base64-encoded Basic auth header value."""
        if not self.wp_user or not self.app_password:
            return ""
        credentials = f"{self.wp_user}:{self.app_password}"
        encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        return f"Basic {encoded}"

    @property
    def is_configured(self) -> bool:
        """Whether this site has valid credentials for API access."""
        return bool(self.wp_user and self.app_password)

    @property
    def target_posts_per_week(self) -> float:
        """Target number of posts per week based on posting_frequency."""
        return FREQUENCY_TARGETS.get(self.posting_frequency, 2.0)

    def __repr__(self) -> str:
        configured = "configured" if self.is_configured else "no-creds"
        return f"SiteConfig({self.site_id!r}, {self.domain!r}, {configured})"


# ---------------------------------------------------------------------------
# Site registry loader
# ---------------------------------------------------------------------------


def load_site_registry(registry_path: Optional[Path] = None) -> List[SiteConfig]:
    """
    Load all site configurations from the site-registry.json file.

    Reads application passwords from environment variables using each site's
    ``wp_app_password_env`` field. Sites without credentials are still loaded
    but marked as unconfigured.

    Parameters
    ----------
    registry_path : Path, optional
        Path to the site-registry.json file. Defaults to the standard
        location relative to this module.

    Returns
    -------
    list of SiteConfig
        All sites from the registry, sorted by priority.
    """
    path = registry_path or SITE_REGISTRY_PATH

    if not path.exists():
        logger.error("Site registry not found at %s", path)
        raise FileNotFoundError(f"Site registry not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    sites_data = data.get("sites", [])
    configs: List[SiteConfig] = []

    for entry in sites_data:
        # Resolve the application password from environment variable
        app_password = ""
        env_var = entry.get("wp_app_password_env", "")
        if env_var:
            app_password = os.getenv(env_var, "")
            if not app_password:
                logger.debug(
                    "Site %s: env var %s not set, site will be unconfigured",
                    entry.get("id", "unknown"),
                    env_var,
                )

        config = SiteConfig(
            domain=entry["domain"],
            site_id=entry["id"],
            brand_color=entry.get("brand_color", "#000000"),
            accent_color=entry.get("accent_color", "#666666"),
            voice=entry.get("voice", ""),
            niche=entry.get("niche", ""),
            posting_frequency=entry.get("posting_frequency", "2x-weekly"),
            priority=entry.get("priority", 99),
            theme=entry.get("theme", "blocksy"),
            wp_user=entry.get("wp_user", ""),
            app_password=app_password,
            flagship=entry.get("flagship", False),
        )
        configs.append(config)

    configs.sort(key=lambda c: c.priority)
    logger.info(
        "Loaded %d sites from registry (%d configured)",
        len(configs),
        sum(1 for c in configs if c.is_configured),
    )
    return configs


# ---------------------------------------------------------------------------
# Async event loop helpers
# ---------------------------------------------------------------------------


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get the running event loop or create a new one."""
    try:
        loop = asyncio.get_running_loop()
        return loop
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
        # We are inside an already-running loop (e.g. Jupyter, nested call).
        # Create a new loop in a thread to avoid deadlock.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=120)
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# WordPressClient — per-site async client
# ---------------------------------------------------------------------------


class WordPressClient:
    """
    Async WordPress REST API client for a single site.

    All public methods come in async (default) and sync variants. Sync
    variants are named with a ``_sync`` suffix and internally call the
    async version via ``asyncio.run``.

    Parameters
    ----------
    config : SiteConfig
        Site configuration including domain and credentials.
    timeout : int
        Request timeout in seconds. Default 30.

    Examples
    --------
    >>> client = WordPressClient(config)
    >>> post = await client.create_post("Title", "<p>Body</p>", status="publish")
    >>> post_sync = client.create_post_sync("Title", "<p>Body</p>")
    """

    def __init__(self, config: SiteConfig, timeout: int = 30):
        self.config = config
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._categories_cache: Optional[List[Dict[str, Any]]] = None
        self._tags_cache: Optional[List[Dict[str, Any]]] = None

    # -- Session management -------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if aiohttp is None:
            raise ImportError(
                "aiohttp is required for WordPressClient. "
                "Install it with: pip install aiohttp"
            )

        if self._session is None or self._session.closed:
            headers = {
                "User-Agent": "OpenClaw-Empire/1.0",
                "Accept": "application/json",
            }
            if self.config.auth_header:
                headers["Authorization"] = self.config.auth_header

            connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=headers,
                connector=connector,
                timeout=timeout_config,
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def close_sync(self) -> None:
        """Synchronous wrapper for close()."""
        _run_sync(self.close())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # -- Core HTTP methods with retry ---------------------------------------

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Any, Dict[str, str]]:
        """
        Make an HTTP request with exponential backoff retry on transient errors.

        Returns
        -------
        tuple of (status_code, response_json_or_text, response_headers)

        Raises
        ------
        AuthenticationError
            On 401 or 403 responses.
        NotFoundError
            On 404 responses.
        RateLimitError
            On 429 after all retries exhausted.
        WordPressError
            On other non-2xx responses after retries.
        """
        if not self.config.is_configured:
            raise SiteNotConfiguredError(
                f"Site {self.config.site_id!r} ({self.config.domain}) has no credentials configured. "
                f"Set the environment variable for this site's application password."
            )

        session = await self._get_session()
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                logger.debug(
                    "API %s %s (attempt %d/%d) site=%s",
                    method.upper(),
                    url,
                    attempt + 1,
                    MAX_RETRIES + 1,
                    self.config.site_id,
                )

                kwargs: Dict[str, Any] = {}
                if json_data is not None:
                    kwargs["json"] = json_data
                if data is not None:
                    kwargs["data"] = data
                if headers is not None:
                    kwargs["headers"] = headers
                if params is not None:
                    # Filter out None values from params
                    kwargs["params"] = {
                        k: v for k, v in params.items() if v is not None
                    }

                async with session.request(method, url, **kwargs) as resp:
                    status = resp.status
                    resp_headers = dict(resp.headers)

                    # Try to parse JSON, fall back to text
                    try:
                        body = await resp.json(content_type=None)
                    except (json.JSONDecodeError, ValueError):
                        body = await resp.text()

                    # Handle specific error codes
                    if status == 401 or status == 403:
                        raise AuthenticationError(
                            f"Authentication failed for {self.config.site_id} "
                            f"({self.config.domain}): HTTP {status}",
                            status_code=status,
                            response_body=str(body),
                        )

                    if status == 404:
                        raise NotFoundError(
                            f"Resource not found: {url}",
                            status_code=404,
                            response_body=str(body),
                        )

                    # Retry on transient errors
                    if status in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        # Respect Retry-After header if present
                        retry_after = resp_headers.get("Retry-After")
                        if retry_after:
                            try:
                                delay = max(delay, float(retry_after))
                            except ValueError:
                                pass
                        logger.warning(
                            "Retryable error %d from %s, retrying in %.1fs",
                            status,
                            url,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    # After all retries, raise on rate limit
                    if status == 429:
                        raise RateLimitError(
                            f"Rate limited by {self.config.domain} after {MAX_RETRIES} retries",
                            status_code=429,
                            response_body=str(body),
                        )

                    # Raise on other non-2xx after retries
                    if status >= 400:
                        error_msg = body
                        if isinstance(body, dict):
                            error_msg = body.get("message", str(body))
                        raise WordPressError(
                            f"HTTP {status} from {self.config.domain}: {error_msg}",
                            status_code=status,
                            response_body=str(body),
                        )

                    return status, body, resp_headers

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Network error on %s (%s), retrying in %.1fs: %s",
                        url,
                        type(exc).__name__,
                        delay,
                        str(exc),
                    )
                    await asyncio.sleep(delay)
                else:
                    raise WordPressError(
                        f"Network error after {MAX_RETRIES} retries for "
                        f"{self.config.site_id} ({self.config.domain}): {exc}"
                    ) from exc

        # Should not reach here, but just in case
        raise WordPressError(
            f"Request failed after {MAX_RETRIES} retries: {last_error}"
        )

    async def _get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """GET request to a WP REST API v2 endpoint."""
        url = f"{self.config.api_url}/{endpoint}"
        _, body, _ = await self._request("GET", url, params=params)
        return body

    async def _post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """POST request to a WP REST API v2 endpoint."""
        url = f"{self.config.api_url}/{endpoint}"
        _, body, _ = await self._request(
            "POST", url, json_data=json_data, data=data, headers=headers
        )
        return body

    async def _delete(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """DELETE request to a WP REST API v2 endpoint."""
        url = f"{self.config.api_url}/{endpoint}"
        _, body, _ = await self._request("DELETE", url, params=params)
        return body

    async def _request_full(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Tuple[int, Any, Dict[str, str]]:
        """Full request returning status, body, and headers."""
        return await self._request(method, url, **kwargs)

    # -----------------------------------------------------------------------
    # Posts
    # -----------------------------------------------------------------------

    async def create_post(
        self,
        title: str,
        content: str,
        status: str = "draft",
        categories: Optional[List[int]] = None,
        tags: Optional[List[int]] = None,
        meta: Optional[Dict[str, Any]] = None,
        slug: Optional[str] = None,
        excerpt: Optional[str] = None,
        featured_media: Optional[int] = None,
        author: Optional[int] = None,
        comment_status: str = "open",
        ping_status: str = "closed",
    ) -> Dict[str, Any]:
        """
        Create a new WordPress post.

        Parameters
        ----------
        title : str
            Post title.
        content : str
            Post content (HTML).
        status : str
            One of: draft, publish, future, pending, private.
        categories : list of int, optional
            Category IDs to assign.
        tags : list of int, optional
            Tag IDs to assign.
        meta : dict, optional
            Post meta fields (key-value pairs).
        slug : str, optional
            URL slug. Auto-generated from title if omitted.
        excerpt : str, optional
            Post excerpt.
        featured_media : int, optional
            Media ID for featured image.
        author : int, optional
            Author user ID.
        comment_status : str
            "open" or "closed". Default "open".
        ping_status : str
            "open" or "closed". Default "closed".

        Returns
        -------
        dict
            Full post object from the API including id, link, status, etc.
        """
        payload: Dict[str, Any] = {
            "title": title,
            "content": content,
            "status": status,
            "comment_status": comment_status,
            "ping_status": ping_status,
        }

        if categories:
            payload["categories"] = categories
        if tags:
            payload["tags"] = tags
        if meta:
            payload["meta"] = meta
        if slug:
            payload["slug"] = slug
        if excerpt:
            payload["excerpt"] = excerpt
        if featured_media is not None:
            payload["featured_media"] = featured_media
        if author is not None:
            payload["author"] = author

        result = await self._post("posts", json_data=payload)
        logger.info(
            "Created post %s on %s: %s (status=%s)",
            result.get("id"),
            self.config.site_id,
            title[:60],
            status,
        )
        return result

    def create_post_sync(
        self,
        title: str,
        content: str,
        status: str = "draft",
        categories: Optional[List[int]] = None,
        tags: Optional[List[int]] = None,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for create_post()."""
        return _run_sync(
            self.create_post(
                title, content, status=status,
                categories=categories, tags=tags, meta=meta, **kwargs
            )
        )

    async def update_post(
        self, post_id: int, **kwargs
    ) -> Dict[str, Any]:
        """
        Update an existing post.

        Parameters
        ----------
        post_id : int
            The post ID to update.
        **kwargs
            Any valid post fields: title, content, status, categories,
            tags, meta, slug, excerpt, featured_media, etc.

        Returns
        -------
        dict
            Updated post object.
        """
        result = await self._post(f"posts/{post_id}", json_data=kwargs)
        logger.info(
            "Updated post %d on %s: fields=%s",
            post_id,
            self.config.site_id,
            list(kwargs.keys()),
        )
        return result

    def update_post_sync(self, post_id: int, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for update_post()."""
        return _run_sync(self.update_post(post_id, **kwargs))

    async def get_post(self, post_id: int) -> Dict[str, Any]:
        """
        Retrieve a single post by ID.

        Parameters
        ----------
        post_id : int
            The post ID.

        Returns
        -------
        dict
            Full post object.

        Raises
        ------
        NotFoundError
            If the post does not exist.
        """
        result = await self._get(f"posts/{post_id}")
        logger.debug("Retrieved post %d from %s", post_id, self.config.site_id)
        return result

    def get_post_sync(self, post_id: int) -> Dict[str, Any]:
        """Synchronous wrapper for get_post()."""
        return _run_sync(self.get_post(post_id))

    async def list_posts(
        self,
        per_page: int = 10,
        page: int = 1,
        status: str = "any",
        search: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        categories: Optional[List[int]] = None,
        tags: Optional[List[int]] = None,
        orderby: str = "date",
        order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """
        List posts with filtering and pagination.

        Parameters
        ----------
        per_page : int
            Number of posts per page (max 100).
        page : int
            Page number (1-indexed).
        status : str
            Post status filter: publish, draft, future, pending, private, any.
        search : str, optional
            Search query string.
        after : str, optional
            ISO 8601 date — only posts published after this date.
        before : str, optional
            ISO 8601 date — only posts published before this date.
        categories : list of int, optional
            Filter by category IDs.
        tags : list of int, optional
            Filter by tag IDs.
        orderby : str
            Field to order by: date, title, id, modified, relevance.
        order : str
            Sort direction: asc or desc.

        Returns
        -------
        list of dict
            List of post objects.
        """
        params: Dict[str, Any] = {
            "per_page": min(per_page, WP_MAX_PER_PAGE),
            "page": page,
            "status": status,
            "orderby": orderby,
            "order": order,
        }
        if search:
            params["search"] = search
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        if categories:
            params["categories"] = ",".join(str(c) for c in categories)
        if tags:
            params["tags"] = ",".join(str(t) for t in tags)

        result = await self._get("posts", params=params)
        logger.debug(
            "Listed %d posts from %s (page %d)",
            len(result) if isinstance(result, list) else 0,
            self.config.site_id,
            page,
        )
        return result if isinstance(result, list) else []

    def list_posts_sync(
        self,
        per_page: int = 10,
        page: int = 1,
        status: str = "any",
        search: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for list_posts()."""
        return _run_sync(
            self.list_posts(
                per_page=per_page, page=page, status=status,
                search=search, after=after, before=before, **kwargs
            )
        )

    async def delete_post(self, post_id: int, force: bool = False) -> bool:
        """
        Delete a post.

        Parameters
        ----------
        post_id : int
            The post ID to delete.
        force : bool
            If True, permanently delete (bypass trash). Default False.

        Returns
        -------
        bool
            True if deletion was successful.
        """
        params = {"force": "true"} if force else {}
        try:
            await self._delete(f"posts/{post_id}", params=params)
            logger.info(
                "Deleted post %d from %s (force=%s)",
                post_id,
                self.config.site_id,
                force,
            )
            return True
        except WordPressError as exc:
            logger.warning(
                "Failed to delete post %d from %s: %s",
                post_id,
                self.config.site_id,
                exc,
            )
            return False

    def delete_post_sync(self, post_id: int, force: bool = False) -> bool:
        """Synchronous wrapper for delete_post()."""
        return _run_sync(self.delete_post(post_id, force=force))

    async def schedule_post(
        self,
        title: str,
        content: str,
        publish_date: Union[str, datetime],
        categories: Optional[List[int]] = None,
        tags: Optional[List[int]] = None,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Schedule a post for future publication.

        Parameters
        ----------
        title : str
            Post title.
        content : str
            Post content (HTML).
        publish_date : str or datetime
            ISO 8601 date string or datetime object for when to publish.
            The WP REST API expects the date in the site's local timezone
            via the ``date`` field.
        categories : list of int, optional
            Category IDs.
        tags : list of int, optional
            Tag IDs.
        meta : dict, optional
            Post meta fields.
        **kwargs
            Additional post fields.

        Returns
        -------
        dict
            Created post object with status "future".
        """
        if isinstance(publish_date, datetime):
            date_str = publish_date.isoformat()
        else:
            date_str = publish_date

        result = await self.create_post(
            title=title,
            content=content,
            status="future",
            categories=categories,
            tags=tags,
            meta=meta,
            **kwargs,
        )

        # Set the date after creation (WP may need it separately for future)
        if result.get("status") != "future" or True:
            # Always update with the exact date to be safe
            result = await self.update_post(result["id"], date=date_str)

        logger.info(
            "Scheduled post %d on %s for %s",
            result.get("id"),
            self.config.site_id,
            date_str,
        )
        return result

    def schedule_post_sync(
        self,
        title: str,
        content: str,
        publish_date: Union[str, datetime],
        **kwargs,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for schedule_post()."""
        return _run_sync(
            self.schedule_post(title, content, publish_date, **kwargs)
        )

    # -----------------------------------------------------------------------
    # Media
    # -----------------------------------------------------------------------

    async def upload_media(
        self,
        file_path: Union[str, Path],
        alt_text: Optional[str] = None,
        caption: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a media file (image, PDF, etc.) to the WordPress media library.

        Parameters
        ----------
        file_path : str or Path
            Path to the local file to upload.
        alt_text : str, optional
            Alt text for accessibility and SEO.
        caption : str, optional
            Media caption.
        title : str, optional
            Media title. Defaults to filename without extension.
        description : str, optional
            Media description.

        Returns
        -------
        dict
            Media object with keys: id, source_url, title, alt_text, etc.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Media file not found: {file_path}")

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "application/octet-stream"

        filename = file_path.name

        # Read the file
        with open(file_path, "rb") as fh:
            file_data = fh.read()

        # Upload via multipart would be ideal, but WP REST API also accepts
        # raw binary with Content-Disposition header
        url = f"{self.config.api_url}/media"
        upload_headers = {
            "Content-Type": mime_type,
            "Content-Disposition": f'attachment; filename="{filename}"',
        }

        _, result, _ = await self._request(
            "POST",
            url,
            data=file_data,
            headers=upload_headers,
        )

        media_id = result.get("id")
        logger.info(
            "Uploaded media %s to %s: id=%s, url=%s",
            filename,
            self.config.site_id,
            media_id,
            result.get("source_url", ""),
        )

        # Update alt text, caption, title, description if provided
        update_fields: Dict[str, Any] = {}
        if alt_text is not None:
            update_fields["alt_text"] = alt_text
        if caption is not None:
            update_fields["caption"] = caption
        if title is not None:
            update_fields["title"] = title
        if description is not None:
            update_fields["description"] = description

        if update_fields and media_id:
            await self._post(f"media/{media_id}", json_data=update_fields)
            logger.debug("Updated media %d metadata: %s", media_id, list(update_fields.keys()))

        return result

    def upload_media_sync(
        self,
        file_path: Union[str, Path],
        alt_text: Optional[str] = None,
        caption: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for upload_media()."""
        return _run_sync(
            self.upload_media(file_path, alt_text=alt_text, caption=caption, **kwargs)
        )

    async def set_featured_image(self, post_id: int, media_id: int) -> bool:
        """
        Set the featured image on a post.

        Parameters
        ----------
        post_id : int
            The post to update.
        media_id : int
            The media attachment ID to set as featured.

        Returns
        -------
        bool
            True if successful.
        """
        try:
            await self.update_post(post_id, featured_media=media_id)
            logger.info(
                "Set featured image %d on post %d (%s)",
                media_id,
                post_id,
                self.config.site_id,
            )
            return True
        except WordPressError as exc:
            logger.warning(
                "Failed to set featured image on post %d: %s", post_id, exc
            )
            return False

    def set_featured_image_sync(self, post_id: int, media_id: int) -> bool:
        """Synchronous wrapper for set_featured_image()."""
        return _run_sync(self.set_featured_image(post_id, media_id))

    async def upload_and_set_featured(
        self,
        post_id: int,
        image_path: Union[str, Path],
        alt_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload an image and set it as the featured image on a post.

        Combines upload_media() and set_featured_image() in one call.

        Parameters
        ----------
        post_id : int
            The post ID.
        image_path : str or Path
            Path to the image file.
        alt_text : str, optional
            Alt text for the image.

        Returns
        -------
        dict
            The uploaded media object.
        """
        media = await self.upload_media(image_path, alt_text=alt_text)
        media_id = media.get("id")
        if media_id:
            await self.set_featured_image(post_id, media_id)
        return media

    def upload_and_set_featured_sync(
        self,
        post_id: int,
        image_path: Union[str, Path],
        alt_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for upload_and_set_featured()."""
        return _run_sync(
            self.upload_and_set_featured(post_id, image_path, alt_text=alt_text)
        )

    # -----------------------------------------------------------------------
    # Categories & Tags
    # -----------------------------------------------------------------------

    async def get_categories(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get all categories for this site.

        Results are cached for the lifetime of the client instance.

        Parameters
        ----------
        force_refresh : bool
            If True, bypass cache and fetch fresh data.

        Returns
        -------
        list of dict
            Category objects with id, name, slug, count, parent, etc.
        """
        if self._categories_cache is not None and not force_refresh:
            return self._categories_cache

        all_categories: List[Dict[str, Any]] = []
        page = 1

        while True:
            batch = await self._get(
                "categories", params={"per_page": WP_MAX_PER_PAGE, "page": page}
            )
            if not isinstance(batch, list) or len(batch) == 0:
                break
            all_categories.extend(batch)
            if len(batch) < WP_MAX_PER_PAGE:
                break
            page += 1

        self._categories_cache = all_categories
        logger.debug(
            "Fetched %d categories from %s", len(all_categories), self.config.site_id
        )
        return all_categories

    def get_categories_sync(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_categories()."""
        return _run_sync(self.get_categories(force_refresh=force_refresh))

    async def create_category(
        self,
        name: str,
        slug: Optional[str] = None,
        parent: Optional[int] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new category.

        Parameters
        ----------
        name : str
            Category name.
        slug : str, optional
            URL slug. Auto-generated from name if omitted.
        parent : int, optional
            Parent category ID for hierarchy.
        description : str, optional
            Category description.

        Returns
        -------
        dict
            Created category object.
        """
        payload: Dict[str, Any] = {"name": name}
        if slug:
            payload["slug"] = slug
        if parent is not None:
            payload["parent"] = parent
        if description:
            payload["description"] = description

        result = await self._post("categories", json_data=payload)

        # Invalidate cache
        self._categories_cache = None

        logger.info(
            "Created category '%s' (id=%s) on %s",
            name,
            result.get("id"),
            self.config.site_id,
        )
        return result

    def create_category_sync(
        self, name: str, slug: Optional[str] = None, parent: Optional[int] = None, **kwargs
    ) -> Dict[str, Any]:
        """Synchronous wrapper for create_category()."""
        return _run_sync(
            self.create_category(name, slug=slug, parent=parent, **kwargs)
        )

    async def ensure_category(self, name: str) -> Dict[str, Any]:
        """
        Get a category by name, creating it if it does not exist.

        Case-insensitive name matching against existing categories.

        Parameters
        ----------
        name : str
            Category name.

        Returns
        -------
        dict
            The existing or newly created category object.
        """
        categories = await self.get_categories()
        name_lower = name.lower().strip()

        for cat in categories:
            cat_name = cat.get("name", "")
            # WP returns names HTML-encoded sometimes
            if isinstance(cat_name, dict):
                cat_name = cat_name.get("rendered", "")
            if cat_name.lower().strip() == name_lower:
                logger.debug(
                    "Category '%s' already exists (id=%s) on %s",
                    name,
                    cat.get("id"),
                    self.config.site_id,
                )
                return cat

        return await self.create_category(name)

    def ensure_category_sync(self, name: str) -> Dict[str, Any]:
        """Synchronous wrapper for ensure_category()."""
        return _run_sync(self.ensure_category(name))

    async def get_tags(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get all tags for this site.

        Results are cached for the lifetime of the client instance.

        Parameters
        ----------
        force_refresh : bool
            If True, bypass cache and fetch fresh data.

        Returns
        -------
        list of dict
            Tag objects with id, name, slug, count, etc.
        """
        if self._tags_cache is not None and not force_refresh:
            return self._tags_cache

        all_tags: List[Dict[str, Any]] = []
        page = 1

        while True:
            batch = await self._get(
                "tags", params={"per_page": WP_MAX_PER_PAGE, "page": page}
            )
            if not isinstance(batch, list) or len(batch) == 0:
                break
            all_tags.extend(batch)
            if len(batch) < WP_MAX_PER_PAGE:
                break
            page += 1

        self._tags_cache = all_tags
        logger.debug(
            "Fetched %d tags from %s", len(all_tags), self.config.site_id
        )
        return all_tags

    def get_tags_sync(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_tags()."""
        return _run_sync(self.get_tags(force_refresh=force_refresh))

    async def create_tag(
        self,
        name: str,
        slug: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new tag.

        Parameters
        ----------
        name : str
            Tag name.
        slug : str, optional
            URL slug.
        description : str, optional
            Tag description.

        Returns
        -------
        dict
            Created tag object.
        """
        payload: Dict[str, Any] = {"name": name}
        if slug:
            payload["slug"] = slug
        if description:
            payload["description"] = description

        result = await self._post("tags", json_data=payload)

        # Invalidate cache
        self._tags_cache = None

        logger.info(
            "Created tag '%s' (id=%s) on %s",
            name,
            result.get("id"),
            self.config.site_id,
        )
        return result

    def create_tag_sync(
        self, name: str, slug: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Synchronous wrapper for create_tag()."""
        return _run_sync(self.create_tag(name, slug=slug, **kwargs))

    async def ensure_tag(self, name: str) -> Dict[str, Any]:
        """
        Get a tag by name, creating it if it does not exist.

        Case-insensitive name matching against existing tags.

        Parameters
        ----------
        name : str
            Tag name.

        Returns
        -------
        dict
            The existing or newly created tag object.
        """
        tags = await self.get_tags()
        name_lower = name.lower().strip()

        for tag in tags:
            tag_name = tag.get("name", "")
            if isinstance(tag_name, dict):
                tag_name = tag_name.get("rendered", "")
            if tag_name.lower().strip() == name_lower:
                logger.debug(
                    "Tag '%s' already exists (id=%s) on %s",
                    name,
                    tag.get("id"),
                    self.config.site_id,
                )
                return tag

        return await self.create_tag(name)

    def ensure_tag_sync(self, name: str) -> Dict[str, Any]:
        """Synchronous wrapper for ensure_tag()."""
        return _run_sync(self.ensure_tag(name))

    # -----------------------------------------------------------------------
    # RankMath SEO
    # -----------------------------------------------------------------------

    async def set_seo(
        self,
        post_id: int,
        focus_keyword: str,
        meta_title: Optional[str] = None,
        meta_description: Optional[str] = None,
        schema_type: str = "BlogPosting",
        robots: Optional[str] = None,
        canonical_url: Optional[str] = None,
        og_title: Optional[str] = None,
        og_description: Optional[str] = None,
    ) -> bool:
        """
        Set RankMath SEO metadata on a post.

        RankMath stores SEO data as post meta fields accessible via the
        WP REST API ``meta`` parameter. This method sets the focus keyword,
        meta title/description, schema type, and robots directives.

        Parameters
        ----------
        post_id : int
            The post ID to update.
        focus_keyword : str
            Primary SEO focus keyword.
        meta_title : str, optional
            SEO title (max ~60 chars recommended). If omitted, RankMath
            will use its default pattern.
        meta_description : str, optional
            SEO meta description (max ~160 chars recommended).
        schema_type : str
            Schema.org type: BlogPosting, HowTo, FAQPage, Product, Article.
            Default "BlogPosting".
        robots : str, optional
            Robots meta directives. Default is a comprehensive index/follow
            with rich snippet allowances.
        canonical_url : str, optional
            Canonical URL override.
        og_title : str, optional
            Open Graph title override.
        og_description : str, optional
            Open Graph description override.

        Returns
        -------
        bool
            True if SEO fields were set successfully.
        """
        meta: Dict[str, Any] = {
            "rank_math_focus_keyword": focus_keyword,
            "rank_math_robots": robots or DEFAULT_ROBOTS_META,
        }

        if meta_title is not None:
            meta["rank_math_title"] = meta_title
        if meta_description is not None:
            meta["rank_math_description"] = meta_description
        if canonical_url is not None:
            meta["rank_math_canonical_url"] = canonical_url
        if og_title is not None:
            meta["rank_math_facebook_title"] = og_title
            meta["rank_math_twitter_title"] = og_title
        if og_description is not None:
            meta["rank_math_facebook_description"] = og_description
            meta["rank_math_twitter_description"] = og_description

        # RankMath schema type — stored as a JSON string in meta
        if schema_type:
            schema_data = {
                "@type": schema_type,
                "isPrimary": True,
            }
            meta["rank_math_schema_BlogPosting"] = json.dumps(schema_data)

        try:
            await self.update_post(post_id, meta=meta)
            logger.info(
                "Set SEO on post %d (%s): keyword='%s', schema=%s",
                post_id,
                self.config.site_id,
                focus_keyword,
                schema_type,
            )
            return True
        except WordPressError as exc:
            logger.warning(
                "Failed to set SEO on post %d (%s): %s",
                post_id,
                self.config.site_id,
                exc,
            )
            return False

    def set_seo_sync(
        self,
        post_id: int,
        focus_keyword: str,
        meta_title: Optional[str] = None,
        meta_description: Optional[str] = None,
        schema_type: str = "BlogPosting",
        **kwargs,
    ) -> bool:
        """Synchronous wrapper for set_seo()."""
        return _run_sync(
            self.set_seo(
                post_id,
                focus_keyword,
                meta_title=meta_title,
                meta_description=meta_description,
                schema_type=schema_type,
                **kwargs,
            )
        )

    # -----------------------------------------------------------------------
    # Cache Management
    # -----------------------------------------------------------------------

    async def purge_cache(self) -> bool:
        """
        Purge all LiteSpeed Cache for the site.

        Tries the LiteSpeed developer API endpoint first, falls back to
        the query-string purge method.

        Returns
        -------
        bool
            True if purge was successful or if the endpoint responded.
        """
        # Method 1: LiteSpeed Developer API (if available)
        try:
            url = f"{self.config.base_url}/developer/v1/litespeed/purge_all"
            status, body, _ = await self._request_full("POST", url)
            if status < 400:
                logger.info("Purged LiteSpeed cache for %s via developer API", self.config.site_id)
                return True
        except (WordPressError, NotFoundError):
            pass

        # Method 2: Query string trigger
        try:
            url = f"https://{self.config.domain}/?litespeed_purge=all"
            session = await self._get_session()
            async with session.get(url) as resp:
                logger.info(
                    "Purged LiteSpeed cache for %s via query string (status=%d)",
                    self.config.site_id,
                    resp.status,
                )
                return resp.status < 400
        except Exception as exc:
            logger.warning(
                "Cache purge failed for %s: %s", self.config.site_id, exc
            )
            return False

    def purge_cache_sync(self) -> bool:
        """Synchronous wrapper for purge_cache()."""
        return _run_sync(self.purge_cache())

    async def purge_post_cache(self, post_id: int) -> bool:
        """
        Purge cache for a specific post.

        Attempts to use the LiteSpeed per-post purge, falls back to
        fetching the post URL with the purge query string.

        Parameters
        ----------
        post_id : int
            The post ID whose cache should be purged.

        Returns
        -------
        bool
            True if purge was attempted.
        """
        try:
            # Try to get the post URL for targeted purge
            post = await self.get_post(post_id)
            post_link = post.get("link", "")
            if post_link:
                separator = "&" if "?" in post_link else "?"
                purge_url = f"{post_link}{separator}litespeed_purge=all"
                session = await self._get_session()
                async with session.get(purge_url) as resp:
                    logger.info(
                        "Purged cache for post %d on %s (status=%d)",
                        post_id,
                        self.config.site_id,
                        resp.status,
                    )
                    return resp.status < 400
        except Exception as exc:
            logger.warning(
                "Post cache purge failed for post %d on %s: %s",
                post_id,
                self.config.site_id,
                exc,
            )
        return False

    def purge_post_cache_sync(self, post_id: int) -> bool:
        """Synchronous wrapper for purge_post_cache()."""
        return _run_sync(self.purge_post_cache(post_id))

    # -----------------------------------------------------------------------
    # Site Health & Info
    # -----------------------------------------------------------------------

    async def get_site_info(self) -> Dict[str, Any]:
        """
        Get site information from the WP REST API root endpoint.

        Returns the raw response from ``/wp-json/``, which includes
        site name, description, URL, namespaces, authentication info, etc.

        Returns
        -------
        dict
            Site information object.
        """
        url = self.config.base_url
        _, body, _ = await self._request("GET", url)
        return body if isinstance(body, dict) else {}

    def get_site_info_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for get_site_info()."""
        return _run_sync(self.get_site_info())

    async def check_health(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check on the site.

        Checks reachability, response time, WordPress version, active theme,
        plugin count, post count, and last published date.

        Returns
        -------
        dict
            Health report with keys:
            - site_id (str)
            - domain (str)
            - reachable (bool)
            - response_time_ms (int)
            - wp_version (str)
            - site_name (str)
            - active_theme (str)
            - post_count (int)
            - last_published (str or None)
            - error (str or None)
        """
        result: Dict[str, Any] = {
            "site_id": self.config.site_id,
            "domain": self.config.domain,
            "reachable": False,
            "response_time_ms": 0,
            "wp_version": "",
            "site_name": "",
            "active_theme": self.config.theme,
            "post_count": 0,
            "last_published": None,
            "error": None,
        }

        start = time.monotonic()

        try:
            # Check basic reachability via the REST API root
            info = await self.get_site_info()
            elapsed = time.monotonic() - start
            result["reachable"] = True
            result["response_time_ms"] = int(elapsed * 1000)
            result["site_name"] = info.get("name", "")

            # Extract WP version from namespaces or generator
            # The root endpoint may not expose version directly for security,
            # but some sites do via the description or gmw data
            wp_version = info.get("wp_version", "")
            if not wp_version:
                # Try to infer from namespaces
                namespaces = info.get("namespaces", [])
                if "wp/v2" in namespaces:
                    wp_version = "5.0+"  # REST API v2 available
            result["wp_version"] = wp_version

        except WordPressError as exc:
            elapsed = time.monotonic() - start
            result["response_time_ms"] = int(elapsed * 1000)
            result["error"] = str(exc)
            logger.warning("Health check failed for %s: %s", self.config.site_id, exc)
            return result
        except Exception as exc:
            elapsed = time.monotonic() - start
            result["response_time_ms"] = int(elapsed * 1000)
            result["error"] = f"Unexpected error: {exc}"
            logger.warning("Health check error for %s: %s", self.config.site_id, exc)
            return result

        # Get post stats if site is configured
        if self.config.is_configured:
            try:
                # Fetch latest published post
                recent = await self.list_posts(per_page=1, page=1, status="publish")
                if recent:
                    result["last_published"] = recent[0].get("date", None)

                # Get total post count from headers
                # We need the full response for X-WP-Total header
                url = f"{self.config.api_url}/posts"
                params = {"per_page": 1, "status": "publish"}
                _, _, headers = await self._request_full("GET", url, params=params)
                total = headers.get("X-WP-Total", headers.get("x-wp-total", "0"))
                result["post_count"] = int(total)

            except (WordPressError, SiteNotConfiguredError) as exc:
                logger.debug(
                    "Could not get post stats for %s: %s",
                    self.config.site_id,
                    exc,
                )

        return result

    def check_health_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for check_health()."""
        return _run_sync(self.check_health())

    # -----------------------------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------------------------

    async def publish(
        self,
        title: str,
        content: str,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        focus_keyword: Optional[str] = None,
        meta_description: Optional[str] = None,
        schema_type: str = "BlogPosting",
        featured_image_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        High-level publish method: create post, set categories/tags by name,
        set SEO, and optionally upload a featured image.

        This is the recommended method for most publishing workflows.

        Parameters
        ----------
        title : str
            Post title.
        content : str
            Post HTML content.
        categories : list of str, optional
            Category names (will be created if they don't exist).
        tags : list of str, optional
            Tag names (will be created if they don't exist).
        focus_keyword : str, optional
            SEO focus keyword for RankMath.
        meta_description : str, optional
            SEO meta description.
        schema_type : str
            Schema.org type for RankMath.
        featured_image_path : str or Path, optional
            Path to an image file to set as featured.
        **kwargs
            Additional fields passed to create_post().

        Returns
        -------
        dict
            Created post object, enriched with ``_seo_set`` and
            ``_featured_media`` keys.
        """
        # Resolve category IDs from names
        cat_ids: List[int] = []
        if categories:
            for name in categories:
                cat = await self.ensure_category(name)
                cat_ids.append(cat["id"])

        # Resolve tag IDs from names
        tag_ids: List[int] = []
        if tags:
            for name in tags:
                tag = await self.ensure_tag(name)
                tag_ids.append(tag["id"])

        # Create the post
        post = await self.create_post(
            title=title,
            content=content,
            status="publish",
            categories=cat_ids or None,
            tags=tag_ids or None,
            **kwargs,
        )

        post_id = post["id"]

        # Set SEO if keyword provided
        if focus_keyword:
            seo_ok = await self.set_seo(
                post_id,
                focus_keyword=focus_keyword,
                meta_description=meta_description,
                schema_type=schema_type,
            )
            post["_seo_set"] = seo_ok

        # Upload and set featured image
        if featured_image_path:
            media = await self.upload_and_set_featured(
                post_id,
                featured_image_path,
                alt_text=focus_keyword or title,
            )
            post["_featured_media"] = media

        logger.info(
            "Published '%s' to %s (id=%d, cats=%s, tags=%s)",
            title[:60],
            self.config.site_id,
            post_id,
            cat_ids,
            tag_ids,
        )
        return post

    def publish_sync(
        self,
        title: str,
        content: str,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        focus_keyword: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for publish()."""
        return _run_sync(
            self.publish(
                title, content,
                categories=categories, tags=tags,
                focus_keyword=focus_keyword, **kwargs
            )
        )

    def __repr__(self) -> str:
        return (
            f"WordPressClient(site_id={self.config.site_id!r}, "
            f"domain={self.config.domain!r}, "
            f"configured={self.config.is_configured})"
        )


# ---------------------------------------------------------------------------
# EmpireManager — manages all 16 sites
# ---------------------------------------------------------------------------


class EmpireManager:
    """
    Central manager for all WordPress sites in the empire.

    Loads the site registry, creates a WordPressClient for each site, and
    provides bulk operations like health checks, cache purging, and
    publishing dashboards.

    Examples
    --------
    >>> manager = EmpireManager()
    >>> client = manager.get_site("witchcraft")
    >>> health = manager.health_check_all_sync()
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self._configs: List[SiteConfig] = load_site_registry(registry_path)
        self._clients: Dict[str, WordPressClient] = {}

        for config in self._configs:
            self._clients[config.site_id] = WordPressClient(config)

        logger.info(
            "EmpireManager initialized: %d sites, %d configured",
            len(self._clients),
            len(self.get_configured_sites()),
        )

    def get_site(self, site_id: str) -> WordPressClient:
        """
        Get the WordPressClient for a specific site.

        Parameters
        ----------
        site_id : str
            Site identifier (e.g., "witchcraft", "smarthome").

        Returns
        -------
        WordPressClient

        Raises
        ------
        SiteNotFoundError
            If the site_id is not in the registry.
        """
        client = self._clients.get(site_id)
        if client is None:
            available = ", ".join(sorted(self._clients.keys()))
            raise SiteNotFoundError(
                f"Site '{site_id}' not found in registry. Available: {available}"
            )
        return client

    def get_all_sites(self) -> Dict[str, WordPressClient]:
        """
        Get all site clients (including unconfigured ones).

        Returns
        -------
        dict
            Mapping of site_id to WordPressClient.
        """
        return dict(self._clients)

    def get_configured_sites(self) -> Dict[str, WordPressClient]:
        """
        Get only sites that have valid credentials.

        Returns
        -------
        dict
            Mapping of site_id to WordPressClient for configured sites only.
        """
        return {
            sid: client
            for sid, client in self._clients.items()
            if client.config.is_configured
        }

    def get_site_config(self, site_id: str) -> SiteConfig:
        """
        Get the SiteConfig for a site.

        Parameters
        ----------
        site_id : str
            Site identifier.

        Returns
        -------
        SiteConfig
        """
        client = self.get_site(site_id)
        return client.config

    def list_site_ids(self) -> List[str]:
        """Return all site IDs sorted by priority."""
        return [c.site_id for c in self._configs]

    # -- Bulk async operations ----------------------------------------------

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run health checks on all sites in parallel.

        Returns
        -------
        dict
            Mapping of site_id to health check result dict.
        """
        tasks = {}
        for site_id, client in self._clients.items():
            tasks[site_id] = asyncio.ensure_future(client.check_health())

        results: Dict[str, Dict[str, Any]] = {}
        for site_id, task in tasks.items():
            try:
                results[site_id] = await task
            except Exception as exc:
                results[site_id] = {
                    "site_id": site_id,
                    "domain": self._clients[site_id].config.domain,
                    "reachable": False,
                    "error": str(exc),
                }

        return results

    def health_check_all_sync(self) -> Dict[str, Dict[str, Any]]:
        """Synchronous wrapper for health_check_all()."""
        return _run_sync(self.health_check_all())

    async def bulk_purge_cache(self) -> Dict[str, bool]:
        """
        Purge LiteSpeed cache on all configured sites in parallel.

        Returns
        -------
        dict
            Mapping of site_id to purge success boolean.
        """
        configured = self.get_configured_sites()
        tasks = {
            sid: asyncio.ensure_future(client.purge_cache())
            for sid, client in configured.items()
        }

        results: Dict[str, bool] = {}
        for sid, task in tasks.items():
            try:
                results[sid] = await task
            except Exception:
                results[sid] = False

        return results

    def bulk_purge_cache_sync(self) -> Dict[str, bool]:
        """Synchronous wrapper for bulk_purge_cache()."""
        return _run_sync(self.bulk_purge_cache())

    async def publishing_dashboard(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """
        Generate a publishing dashboard showing activity over the last N days.

        For each configured site, returns:
        - post_count: number of posts published in the period
        - last_published: date of the most recent post
        - target_per_week: expected posts per week
        - on_track: whether the site is meeting its posting frequency
        - posts: list of recent post titles and dates

        Parameters
        ----------
        days : int
            Look-back period in days. Default 7.

        Returns
        -------
        dict
            Mapping of site_id to dashboard data.
        """
        after_date = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()

        configured = self.get_configured_sites()
        tasks = {}

        for sid, client in configured.items():
            tasks[sid] = asyncio.ensure_future(
                client.list_posts(
                    per_page=50, status="publish", after=after_date
                )
            )

        results: Dict[str, Dict[str, Any]] = {}

        for sid, task in tasks.items():
            client = configured[sid]
            target = client.config.target_posts_per_week
            expected_in_period = target * (days / 7.0)

            try:
                posts = await task
                post_count = len(posts)

                # Extract post info
                post_summaries = []
                for p in posts:
                    title = p.get("title", {})
                    if isinstance(title, dict):
                        title = title.get("rendered", "Untitled")
                    post_summaries.append({
                        "id": p.get("id"),
                        "title": title,
                        "date": p.get("date"),
                        "link": p.get("link"),
                    })

                last_published = posts[0].get("date") if posts else None

                results[sid] = {
                    "site_id": sid,
                    "domain": client.config.domain,
                    "post_count": post_count,
                    "target_per_week": target,
                    "expected_in_period": round(expected_in_period, 1),
                    "on_track": post_count >= expected_in_period * 0.8,
                    "last_published": last_published,
                    "posting_frequency": client.config.posting_frequency,
                    "posts": post_summaries,
                }

            except (WordPressError, SiteNotConfiguredError) as exc:
                results[sid] = {
                    "site_id": sid,
                    "domain": client.config.domain,
                    "error": str(exc),
                    "post_count": 0,
                    "target_per_week": target,
                    "expected_in_period": round(expected_in_period, 1),
                    "on_track": False,
                    "last_published": None,
                    "posting_frequency": client.config.posting_frequency,
                    "posts": [],
                }

        return results

    def publishing_dashboard_sync(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """Synchronous wrapper for publishing_dashboard()."""
        return _run_sync(self.publishing_dashboard(days=days))

    async def find_content_gaps(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Identify sites that are below their target posting frequency.

        Parameters
        ----------
        days : int
            Look-back period in days. Default 7.

        Returns
        -------
        list of dict
            Sites below target, sorted by severity (most behind first).
            Each dict contains: site_id, domain, post_count, target,
            expected, deficit, posting_frequency.
        """
        dashboard = await self.publishing_dashboard(days=days)
        gaps: List[Dict[str, Any]] = []

        for sid, data in dashboard.items():
            if data.get("error"):
                gaps.append({
                    "site_id": sid,
                    "domain": data["domain"],
                    "post_count": 0,
                    "target_per_week": data["target_per_week"],
                    "expected_in_period": data["expected_in_period"],
                    "deficit": data["expected_in_period"],
                    "posting_frequency": data["posting_frequency"],
                    "error": data["error"],
                })
                continue

            if not data["on_track"]:
                deficit = data["expected_in_period"] - data["post_count"]
                gaps.append({
                    "site_id": sid,
                    "domain": data["domain"],
                    "post_count": data["post_count"],
                    "target_per_week": data["target_per_week"],
                    "expected_in_period": data["expected_in_period"],
                    "deficit": round(deficit, 1),
                    "posting_frequency": data["posting_frequency"],
                    "last_published": data["last_published"],
                })

        gaps.sort(key=lambda g: g.get("deficit", 0), reverse=True)
        return gaps

    def find_content_gaps_sync(self, days: int = 7) -> List[Dict[str, Any]]:
        """Synchronous wrapper for find_content_gaps()."""
        return _run_sync(self.find_content_gaps(days=days))

    async def bulk_operation(
        self,
        site_ids: List[str],
        operation: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run an operation across multiple sites in parallel.

        Supported operations:
        - "purge_cache": Purge LiteSpeed cache
        - "health_check": Run health check
        - "list_posts": List recent posts (pass per_page, status, etc.)
        - "publish": Create and publish a post (pass title, content, etc.)

        Parameters
        ----------
        site_ids : list of str
            Site identifiers to operate on.
        operation : str
            Operation name.
        **kwargs
            Arguments passed to the operation method.

        Returns
        -------
        dict
            Mapping of site_id to operation result.
        """
        tasks: Dict[str, asyncio.Task] = {}

        for sid in site_ids:
            client = self.get_site(sid)

            if operation == "purge_cache":
                tasks[sid] = asyncio.ensure_future(client.purge_cache())
            elif operation == "health_check":
                tasks[sid] = asyncio.ensure_future(client.check_health())
            elif operation == "list_posts":
                tasks[sid] = asyncio.ensure_future(client.list_posts(**kwargs))
            elif operation == "publish":
                tasks[sid] = asyncio.ensure_future(client.publish(**kwargs))
            else:
                raise ValueError(f"Unknown operation: {operation}")

        results: Dict[str, Any] = {}
        for sid, task in tasks.items():
            try:
                results[sid] = await task
            except Exception as exc:
                results[sid] = {"error": str(exc)}

        return results

    def bulk_operation_sync(
        self, site_ids: List[str], operation: str, **kwargs
    ) -> Dict[str, Any]:
        """Synchronous wrapper for bulk_operation()."""
        return _run_sync(self.bulk_operation(site_ids, operation, **kwargs))

    async def close_all(self) -> None:
        """Close all HTTP sessions."""
        for client in self._clients.values():
            await client.close()

    def close_all_sync(self) -> None:
        """Synchronous wrapper for close_all()."""
        _run_sync(self.close_all())

    def __repr__(self) -> str:
        total = len(self._clients)
        configured = len(self.get_configured_sites())
        return f"EmpireManager({total} sites, {configured} configured)"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_empire_manager: Optional[EmpireManager] = None


def get_empire_manager(registry_path: Optional[Path] = None) -> EmpireManager:
    """
    Get or create the singleton EmpireManager instance.

    Parameters
    ----------
    registry_path : Path, optional
        Path to site-registry.json. Only used on first call.

    Returns
    -------
    EmpireManager
    """
    global _empire_manager
    if _empire_manager is None:
        _empire_manager = EmpireManager(registry_path=registry_path)
    return _empire_manager


def get_site_client(site_id: str) -> WordPressClient:
    """
    Convenience function to get a WordPressClient for a specific site.

    Parameters
    ----------
    site_id : str
        Site identifier.

    Returns
    -------
    WordPressClient
    """
    manager = get_empire_manager()
    return manager.get_site(site_id)


def publish_to_site(
    site_id: str,
    title: str,
    content: str,
    status: str = "publish",
    categories: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    focus_keyword: Optional[str] = None,
    meta_description: Optional[str] = None,
    featured_image_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Module-level convenience function to publish a post to a specific site.

    This is the simplest way to publish content from external scripts.

    Parameters
    ----------
    site_id : str
        Target site identifier.
    title : str
        Post title.
    content : str
        Post content (HTML).
    status : str
        Post status. Default "publish".
    categories : list of str, optional
        Category names (created if needed).
    tags : list of str, optional
        Tag names (created if needed).
    focus_keyword : str, optional
        SEO focus keyword.
    meta_description : str, optional
        SEO meta description.
    featured_image_path : str or Path, optional
        Path to featured image file.
    **kwargs
        Additional post fields.

    Returns
    -------
    dict
        Created post object.

    Examples
    --------
    >>> from src.wordpress_client import publish_to_site
    >>> post = publish_to_site(
    ...     "witchcraft",
    ...     "Full Moon Ritual Guide",
    ...     "<p>The full moon is the most powerful time...</p>",
    ...     categories=["Moon Magic", "Rituals"],
    ...     focus_keyword="full moon ritual",
    ... )
    >>> print(post["id"], post["link"])
    """
    client = get_site_client(site_id)

    if status == "publish" and (categories or tags or focus_keyword or featured_image_path):
        # Use the high-level publish method
        return client.publish_sync(
            title=title,
            content=content,
            categories=categories,
            tags=tags,
            focus_keyword=focus_keyword,
            meta_description=meta_description,
            featured_image_path=featured_image_path,
            **kwargs,
        )
    else:
        return client.create_post_sync(
            title=title,
            content=content,
            status=status,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def _format_table(headers: List[str], rows: List[List[str]], min_widths: Optional[List[int]] = None) -> str:
    """Format data as an aligned ASCII table."""
    if not rows:
        return "(no data)"

    # Calculate column widths
    widths = [len(h) for h in headers]
    if min_widths:
        widths = [max(w, mw) for w, mw in zip(widths, min_widths)]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
            else:
                widths.append(len(str(cell)))

    # Build table
    lines: List[str] = []
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header_line)
    lines.append(separator)

    for row in rows:
        cells = []
        for i in range(len(headers)):
            val = str(row[i]) if i < len(row) else ""
            cells.append(val.ljust(widths[i]))
        lines.append(" | ".join(cells))

    return "\n".join(lines)


def _cli_health(args: List[str]) -> None:
    """Run health checks on all sites."""
    print("Running health checks on all sites...\n")
    manager = get_empire_manager()

    try:
        results = manager.health_check_all_sync()
    finally:
        manager.close_all_sync()

    headers = ["Site ID", "Domain", "Status", "Time (ms)", "Posts", "Last Published"]
    rows = []

    for sid in manager.list_site_ids():
        data = results.get(sid, {})
        status = "OK" if data.get("reachable") else "DOWN"
        if data.get("error"):
            status = f"ERR: {data['error'][:40]}"

        last_pub = data.get("last_published", "")
        if last_pub and isinstance(last_pub, str) and len(last_pub) > 10:
            last_pub = last_pub[:10]  # Show date only

        rows.append([
            sid,
            data.get("domain", ""),
            status,
            str(data.get("response_time_ms", "")),
            str(data.get("post_count", "")),
            last_pub or "N/A",
        ])

    print(_format_table(headers, rows))

    # Summary
    total = len(results)
    reachable = sum(1 for r in results.values() if r.get("reachable"))
    print(f"\n{reachable}/{total} sites reachable")


def _cli_dashboard(args: List[str]) -> None:
    """Show publishing dashboard."""
    days = 7
    if args and args[0].isdigit():
        days = int(args[0])

    print(f"Publishing dashboard (last {days} days)...\n")
    manager = get_empire_manager()

    try:
        dashboard = manager.publishing_dashboard_sync(days=days)
    finally:
        manager.close_all_sync()

    headers = ["Site ID", "Domain", "Posts", "Target", "Expected", "Status", "Last Published"]
    rows = []

    for sid in manager.list_site_ids():
        data = dashboard.get(sid)
        if not data:
            continue

        if data.get("error"):
            status = "ERROR"
        elif data.get("on_track"):
            status = "ON TRACK"
        else:
            status = "BEHIND"

        last_pub = data.get("last_published", "")
        if last_pub and isinstance(last_pub, str) and len(last_pub) > 10:
            last_pub = last_pub[:10]

        rows.append([
            sid,
            data.get("domain", ""),
            str(data.get("post_count", 0)),
            str(data.get("target_per_week", "")),
            str(data.get("expected_in_period", "")),
            status,
            last_pub or "N/A",
        ])

    print(_format_table(headers, rows))


def _cli_gaps(args: List[str]) -> None:
    """Show content gaps."""
    days = 7
    if args and args[0].isdigit():
        days = int(args[0])

    print(f"Content gaps (last {days} days)...\n")
    manager = get_empire_manager()

    try:
        gaps = manager.find_content_gaps_sync(days=days)
    finally:
        manager.close_all_sync()

    if not gaps:
        print("All sites are on track! No content gaps detected.")
        return

    headers = ["Site ID", "Domain", "Posts", "Expected", "Deficit", "Frequency"]
    rows = []

    for gap in gaps:
        rows.append([
            gap["site_id"],
            gap["domain"],
            str(gap["post_count"]),
            str(gap["expected_in_period"]),
            str(gap["deficit"]),
            gap["posting_frequency"],
        ])

    print(_format_table(headers, rows))
    print(f"\n{len(gaps)} site(s) behind schedule")


def _cli_publish(args: List[str]) -> None:
    """Publish a post from the command line."""
    # Parse arguments
    site_id = None
    title = None
    content = None
    status = "publish"

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--site" and i + 1 < len(args):
            site_id = args[i + 1]
            i += 2
        elif arg == "--title" and i + 1 < len(args):
            title = args[i + 1]
            i += 2
        elif arg == "--content" and i + 1 < len(args):
            content = args[i + 1]
            i += 2
        elif arg == "--status" and i + 1 < len(args):
            status = args[i + 1]
            i += 2
        elif arg == "--content-file" and i + 1 < len(args):
            content_path = Path(args[i + 1])
            if content_path.exists():
                content = content_path.read_text(encoding="utf-8")
            else:
                print(f"Error: Content file not found: {args[i + 1]}")
                sys.exit(1)
            i += 2
        else:
            i += 1

    if not site_id:
        print("Error: --site is required")
        print("Usage: python -m src.wordpress_client publish --site ID --title \"...\" --content \"...\"")
        sys.exit(1)

    if not title:
        print("Error: --title is required")
        sys.exit(1)

    if not content:
        print("Error: --content or --content-file is required")
        sys.exit(1)

    print(f"Publishing to {site_id}...")

    try:
        post = publish_to_site(site_id, title, content, status=status)
        print(f"Published successfully!")
        print(f"  Post ID: {post.get('id')}")
        print(f"  Status:  {post.get('status')}")
        print(f"  Link:    {post.get('link', 'N/A')}")
    except (SiteNotFoundError, SiteNotConfiguredError, WordPressError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    finally:
        manager = get_empire_manager()
        manager.close_all_sync()


def _cli_list_sites(args: List[str]) -> None:
    """List all sites in the registry."""
    manager = get_empire_manager()

    headers = ["Site ID", "Domain", "Theme", "Niche", "Frequency", "Priority", "Configured"]
    rows = []

    for sid in manager.list_site_ids():
        config = manager.get_site_config(sid)
        rows.append([
            config.site_id,
            config.domain,
            config.theme,
            config.niche,
            config.posting_frequency,
            str(config.priority),
            "Yes" if config.is_configured else "No",
        ])

    print(_format_table(headers, rows))
    print(f"\n{len(rows)} sites total")


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]

    if not args:
        print("OpenClaw Empire — WordPress Client")
        print("=" * 40)
        print()
        print("Commands:")
        print("  health              Check health of all sites")
        print("  dashboard [days]    Show publishing dashboard (default: 7 days)")
        print("  gaps [days]         Show content gaps (default: 7 days)")
        print("  sites               List all sites in the registry")
        print("  publish             Publish a post")
        print("    --site ID         Site identifier (required)")
        print("    --title \"...\"     Post title (required)")
        print("    --content \"...\"   Post content HTML (required)")
        print("    --content-file F  Read content from file")
        print("    --status STATUS   Post status (default: publish)")
        print()
        print("Examples:")
        print("  python -m src.wordpress_client health")
        print("  python -m src.wordpress_client dashboard 14")
        print("  python -m src.wordpress_client publish --site witchcraft --title \"Moon Ritual\" --content \"<p>...</p>\"")
        sys.exit(0)

    command = args[0].lower()
    remaining = args[1:]

    commands = {
        "health": _cli_health,
        "dashboard": _cli_dashboard,
        "gaps": _cli_gaps,
        "publish": _cli_publish,
        "sites": _cli_list_sites,
    }

    handler = commands.get(command)
    if handler is None:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(commands.keys())}")
        sys.exit(1)

    handler(remaining)


if __name__ == "__main__":
    main()
