"""Async HTTP client for Empire Dashboard API."""

import logging

import httpx

from bot.config import DASHBOARD_URL, HTTP_TIMEOUT

logger = logging.getLogger(__name__)


class DashboardClient:
    """Wraps the Empire Dashboard API."""

    def __init__(self, base_url: str = DASHBOARD_URL):
        self.base_url = base_url.rstrip("/")

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=HTTP_TIMEOUT)

    async def _get(self, path: str, params: dict | None = None) -> dict:
        async with self._client() as c:
            r = await c.get(path, params=params)
            r.raise_for_status()
            return r.json()

    async def health(self) -> dict:
        """Get overall dashboard health."""
        return await self._get("/api/health")

    async def services(self) -> dict:
        """Get all service health statuses."""
        return await self._get("/api/health/services")

    async def alerts(self, limit: int = 20) -> dict:
        """Get recent alerts."""
        return await self._get("/api/alerts", {"limit": limit})

    async def screenpipe_search(self, query: str, limit: int = 10) -> dict:
        """Search Screenpipe OCR data via dashboard proxy."""
        return await self._get("/api/screenpipe/search", {"q": query, "limit": limit})
