"""WordPress REST API client for all empire sites."""

import json
import logging
from base64 import b64encode
from pathlib import Path

import httpx

from bot.config import SITES_CONFIG_PATH, HTTP_TIMEOUT

logger = logging.getLogger(__name__)

_sites_cache: dict | None = None


def _load_sites() -> dict:
    """Load sites config, with caching."""
    global _sites_cache
    if _sites_cache is not None:
        return _sites_cache

    config_path = Path(SITES_CONFIG_PATH)
    if not config_path.exists():
        # Try common alternative paths
        for alt in [Path("/app/config/sites.json"), Path("config/sites.json"),
                    Path(r"D:\Claude Code Projects\config\sites.json")]:
            if alt.exists():
                config_path = alt
                break

    if not config_path.exists():
        logger.error("Sites config not found at %s", config_path)
        return {}

    with open(config_path) as f:
        data = json.load(f)

    _sites_cache = data.get("sites", data)
    return _sites_cache


def get_site_ids() -> list[str]:
    """Get all available site IDs."""
    return list(_load_sites().keys())


def get_site_config(site_id: str) -> dict | None:
    """Get config for a specific site."""
    return _load_sites().get(site_id)


def _auth_header(site: dict) -> str:
    """Build Basic auth header from site config."""
    wp = site.get("wordpress", {})
    user = wp.get("user", site.get("wp_user", ""))
    password = wp.get("app_password", site.get("wp_app_password", ""))
    token = b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


class WPClient:
    """WordPress REST API client for a specific site."""

    def __init__(self, site_id: str):
        self.site_id = site_id
        self.site = get_site_config(site_id)
        if not self.site:
            raise ValueError(f"Unknown site: {site_id}")
        self.domain = self.site.get("domain", "")
        self.base_url = f"https://{self.domain}/wp-json/wp/v2"
        self.auth = _auth_header(self.site)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            headers={"Authorization": self.auth},
        )

    async def posts(self, per_page: int = 10, page: int = 1,
                    status: str = "publish") -> list[dict]:
        """Get recent posts."""
        async with self._client() as c:
            r = await c.get(f"{self.base_url}/posts", params={
                "per_page": per_page, "page": page, "status": status,
                "_fields": "id,title,date,status,link",
            })
            r.raise_for_status()
            return r.json()

    async def post_count(self) -> int:
        """Get total published post count."""
        async with self._client() as c:
            r = await c.head(f"{self.base_url}/posts", params={
                "per_page": 1, "status": "publish",
            })
            return int(r.headers.get("X-WP-Total", 0))

    async def pages(self, per_page: int = 10) -> list[dict]:
        """Get pages."""
        async with self._client() as c:
            r = await c.get(f"{self.base_url}/pages", params={
                "per_page": per_page,
                "_fields": "id,title,date,status,link",
            })
            r.raise_for_status()
            return r.json()

    async def clear_cache(self) -> bool:
        """Try to clear various cache plugins."""
        async with self._client() as c:
            # Try WP Super Cache
            try:
                r = await c.delete(f"https://{self.domain}/wp-json/wp-super-cache/v1/cache")
                if r.status_code < 400:
                    return True
            except Exception:
                pass
            # Try LiteSpeed Cache
            try:
                r = await c.post(f"https://{self.domain}/wp-json/litespeed/v1/purge_all")
                if r.status_code < 400:
                    return True
            except Exception:
                pass
        return False

    async def site_health(self) -> dict:
        """Basic site health check."""
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"https://{self.domain}/", follow_redirects=True)
                return {
                    "domain": self.domain,
                    "status_code": r.status_code,
                    "ok": r.status_code == 200,
                    "response_time_ms": int(r.elapsed.total_seconds() * 1000),
                }
        except Exception as e:
            return {"domain": self.domain, "ok": False, "error": str(e)}
