"""Async HTTP client for Brain MCP (all 50+ endpoints)."""

import logging
from typing import Any

import httpx

from bot.config import BRAIN_MCP_URL, HTTP_TIMEOUT

logger = logging.getLogger(__name__)


class BrainClient:
    """Wraps the Brain MCP API at BRAIN_MCP_URL."""

    def __init__(self, base_url: str = BRAIN_MCP_URL):
        self.base_url = base_url.rstrip("/")

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=HTTP_TIMEOUT)

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

    # ── Health & Status ──────────────────────────────────────────────

    async def health(self) -> dict:
        return await self._get("/health")

    async def stats(self) -> dict:
        return await self._get("/tools/brain_stats")

    # ── Core Brain ───────────────────────────────────────────────────

    async def query(self, query: str, limit: int = 20) -> dict:
        return await self._post("/tools/brain_query", {"query": query, "limit": limit})

    async def projects(self, category: str | None = None) -> dict:
        params = {"category": category} if category else None
        return await self._get("/tools/brain_projects", params)

    async def skills(self, category: str | None = None, project: str | None = None) -> dict:
        params = {}
        if category:
            params["category"] = category
        if project:
            params["project"] = project
        return await self._get("/tools/brain_skills", params or None)

    async def learn(self, content: str, source: str = "", category: str = "lesson",
                    confidence: float = 0.8) -> dict:
        return await self._post("/tools/brain_learn", {
            "content": content, "source": source,
            "category": category, "confidence": confidence,
        })

    async def patterns(self, pattern_type: str | None = None) -> dict:
        params = {"pattern_type": pattern_type} if pattern_type else None
        return await self._get("/tools/brain_patterns", params)

    async def opportunities(self, status: str = "open") -> dict:
        return await self._get("/tools/brain_opportunities", {"status": status})

    async def cross_reference(self, topic: str) -> dict:
        return await self._post("/tools/brain_cross_reference", {"topic": topic})

    async def briefing(self) -> dict:
        return await self._get("/tools/brain_briefing")

    async def brain_health(self) -> dict:
        return await self._get("/tools/brain_health")

    async def search_solution(self, query: str, limit: int = 20) -> dict:
        return await self._post("/tools/brain_solution", {"query": query, "limit": limit})

    async def record_solution(self, problem: str, solution: str = "",
                              language: str = "python", project: str = "") -> dict:
        return await self._post("/tools/brain_record_solution", {
            "problem": problem, "solution": solution,
            "language": language, "project": project,
        })

    async def amplify(self, data: dict, context: str = "", quick: bool = True) -> dict:
        return await self._post("/tools/brain_amplify", {
            "data": data, "context": context, "quick": quick,
        })

    async def site_context(self, site_id: str) -> dict:
        return await self._get("/tools/brain_site_context", {"site_id": site_id})

    async def forecast(self) -> dict:
        return await self._get("/tools/brain_forecast")

    async def scan(self, project: str | None = None) -> dict:
        payload = {"project": project} if project else {}
        return await self._post("/tools/brain_scan", payload)

    # ── Events ───────────────────────────────────────────────────────

    async def events(self, limit: int = 50, event_type: str | None = None) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if event_type:
            params["event_type"] = event_type
        return await self._get("/events", params)

    # ── Evolution Engine ─────────────────────────────────────────────

    async def evolution_status(self) -> dict:
        return await self._get("/tools/brain_evolution_status")

    async def discoveries(self, status: str | None = None, discovery_type: str | None = None,
                          limit: int = 50) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if discovery_type:
            params["discovery_type"] = discovery_type
        return await self._get("/tools/brain_discoveries", params)

    async def discovery_update(self, item_id: int, status: str) -> dict:
        return await self._post("/tools/brain_discovery_update", {"id": item_id, "status": status})

    async def ideas(self, status: str | None = None, idea_type: str | None = None,
                    limit: int = 50) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if idea_type:
            params["idea_type"] = idea_type
        return await self._get("/tools/brain_ideas", params)

    async def idea_update(self, item_id: int, status: str) -> dict:
        return await self._post("/tools/brain_idea_update", {"id": item_id, "status": status})

    async def enhancements(self, status: str | None = None, project: str | None = None,
                           enhancement_type: str | None = None, limit: int = 50) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if project:
            params["project"] = project
        if enhancement_type:
            params["enhancement_type"] = enhancement_type
        return await self._get("/tools/brain_enhancements", params)

    async def enhancement_update(self, item_id: int, status: str) -> dict:
        return await self._post("/tools/brain_enhancement_update", {"id": item_id, "status": status})

    async def evolve(self, cycle: str = "quick", project: str | None = None) -> dict:
        payload: dict[str, Any] = {"cycle": cycle}
        if project:
            payload["project"] = project
        return await self._post("/tools/brain_evolve", payload)

    async def adoption_metrics(self) -> dict:
        return await self._get("/tools/brain_adoption_metrics")

    async def invalidate_cycle(self, evolution_id: int) -> dict:
        return await self._post("/tools/brain_invalidate_cycle", {"evolution_id": evolution_id})

    async def sync_evolution(self) -> dict:
        return await self._post("/tools/brain_sync_evolution")

    async def auto_apply(self, dry_run: bool = True) -> dict:
        return await self._post("/tools/brain_auto_apply", {"dry_run": dry_run})

    # ── Grimoire Connector ───────────────────────────────────────────

    async def grimoire_health(self) -> dict:
        return await self._get("/tools/brain_grimoire_health")

    async def grimoire_recommend(self, query: str) -> dict:
        return await self._post("/tools/brain_grimoire_recommend", {"query": query})

    async def grimoire_ideas(self, count: int = 5) -> dict:
        return await self._get("/tools/brain_grimoire_ideas", {"count": count})

    async def grimoire_insights(self) -> dict:
        return await self._get("/tools/brain_grimoire_insights")

    async def grimoire_sync(self) -> dict:
        return await self._post("/tools/brain_grimoire_sync")

    # ── Witchcraft Video Pipeline ────────────────────────────────────

    async def witchcraft_topics(self, count: int = 5) -> dict:
        return await self._get("/tools/brain_witchcraft_topics", {"count": count})

    async def witchcraft_video(self, topic: str, render: bool = False,
                               publish: bool = False) -> dict:
        return await self._post("/tools/brain_witchcraft_video", {
            "topic": topic, "render": render, "publish": publish,
        })

    async def witchcraft_calendar(self, days: int = 14) -> dict:
        return await self._get("/tools/brain_witchcraft_calendar", {"days": days})

    # ── Article-to-Video Pipeline ────────────────────────────────────

    async def article_list(self, site: str, count: int = 10) -> dict:
        return await self._get("/tools/brain_article_list", {"site": site, "count": count})

    async def article_to_video(self, site: str, post_id: int, render: bool = False) -> dict:
        return await self._post("/tools/brain_article_to_video", {
            "site": site, "post_id": post_id, "render": render,
        })

    # ── Auto-Pin Connector ───────────────────────────────────────────

    async def auto_pins(self, site: str, count: int = 5) -> dict:
        return await self._get("/tools/brain_auto_pins", {"site": site, "count": count})

    async def pin_calendar(self, site: str, days: int = 7) -> dict:
        return await self._get("/tools/brain_pin_calendar", {"site": site, "days": days})

    # ── Credit Optimizer ─────────────────────────────────────────────

    async def credit_status(self) -> dict:
        return await self._get("/tools/brain_credit_status")

    async def credit_report(self) -> dict:
        return await self._get("/tools/brain_credit_report")

    async def credit_analysis(self) -> dict:
        return await self._get("/tools/brain_credit_analysis")

    async def claude_md_sizes(self, limit: int = 15) -> dict:
        return await self._get("/tools/brain_claude_md_sizes", {"limit": limit})
