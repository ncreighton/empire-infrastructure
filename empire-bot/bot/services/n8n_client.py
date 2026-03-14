"""Async HTTP client for n8n workflow automation API."""

import logging

import httpx

from bot.config import N8N_URL, N8N_API_KEY, HTTP_TIMEOUT

logger = logging.getLogger(__name__)


class N8nClient:
    """Wraps the n8n REST API."""

    def __init__(self, base_url: str = N8N_URL, api_key: str = N8N_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-N8N-API-KEY": api_key} if api_key else {}

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url, timeout=HTTP_TIMEOUT, headers=self.headers,
        )

    async def _get(self, path: str, params: dict | None = None) -> dict:
        async with self._client() as c:
            r = await c.get(path, params=params)
            r.raise_for_status()
            return r.json()

    async def _post(self, path: str, json: dict | None = None) -> dict:
        async with self._client() as c:
            r = await c.post(path, json=json or {})
            r.raise_for_status()
            return r.json()

    async def workflows(self, active: bool | None = None) -> dict:
        """List all workflows."""
        params = {}
        if active is not None:
            params["active"] = str(active).lower()
        return await self._get("/api/v1/workflows", params or None)

    async def workflow(self, workflow_id: str) -> dict:
        """Get a specific workflow."""
        return await self._get(f"/api/v1/workflows/{workflow_id}")

    async def activate(self, workflow_id: str) -> dict:
        """Activate a workflow."""
        return await self._post(f"/api/v1/workflows/{workflow_id}/activate")

    async def deactivate(self, workflow_id: str) -> dict:
        """Deactivate a workflow."""
        return await self._post(f"/api/v1/workflows/{workflow_id}/deactivate")

    async def executions(self, workflow_id: str | None = None, limit: int = 10) -> dict:
        """List recent executions."""
        params = {"limit": limit}
        if workflow_id:
            params["workflowId"] = workflow_id
        return await self._get("/api/v1/executions", params)

    async def health(self) -> bool:
        """Check if n8n is responding."""
        try:
            async with self._client() as c:
                r = await c.get("/healthz")
                return r.status_code == 200
        except Exception:
            return False
