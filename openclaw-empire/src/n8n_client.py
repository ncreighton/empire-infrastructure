"""
n8n Webhook Integration Client -- OpenClaw Empire
===================================================

Bidirectional webhook integration between the OpenClaw intelligence system and
the n8n workflow automation server running on Nick Creighton's Contabo VPS.

n8n is the workflow backbone for the 16-site WordPress publishing empire.  It
handles content generation pipelines, publishing, site monitoring, revenue
tracking, KDP book operations, SEO auditing, and more.

Outbound (OpenClaw -> n8n):
    Trigger n8n webhooks to kick off workflows -- content generation, publishing,
    monitoring, revenue reports, KDP pipelines, and SEO audits.

Inbound (n8n -> OpenClaw):
    Receive webhook callbacks from n8n when workflows complete.  HMAC-SHA256
    signature verification protects all inbound routes.

Management:
    List, inspect, activate, and deactivate n8n workflows and executions via
    the n8n REST API.

Usage:
    from src.n8n_client import get_n8n_client, get_empire_integration

    # Low-level webhook trigger
    client = get_n8n_client()
    result = await client.trigger_content_pipeline("witchcraft", "Full Moon Ritual Guide")

    # High-level empire orchestration
    empire = get_empire_integration()
    report = await empire.daily_revenue_summary()

CLI:
    python -m src.n8n_client status
    python -m src.n8n_client trigger content --site witchcraft --topic "moon water"
    python -m src.n8n_client trigger publish --site witchcraft --post-id 1234
    python -m src.n8n_client trigger monitor --all
    python -m src.n8n_client trigger revenue --period week
    python -m src.n8n_client executions --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("n8n_client")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    ))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N8N_BASE_URL = os.getenv(
    "N8N_BASE_URL", "http://vmi2976539.contaboserver.net:5678"
)
N8N_WEBHOOK_BASE = f"{N8N_BASE_URL}/webhook"
N8N_API_BASE = f"{N8N_BASE_URL}/api/v1"
N8N_API_KEY = os.getenv("N8N_API_KEY", "")
N8N_WEBHOOK_SECRET = os.getenv("N8N_WEBHOOK_SECRET", "")

# Webhook paths as defined in the n8n workflow triggers
WEBHOOK_PATHS = {
    "content": "openclaw-content",
    "publish": "openclaw-publish",
    "kdp": "openclaw-kdp",
    "monitor": "openclaw-monitor",
    "revenue": "openclaw-revenue",
    "audit": "openclaw-audit",
}

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds, doubles on each retry

# Timeouts
WEBHOOK_TIMEOUT = 30  # seconds
API_TIMEOUT = 10  # seconds

# Valid event types for the callback registry
CALLBACK_EVENT_TYPES = frozenset({
    "content_ready",
    "publish_complete",
    "monitor_alert",
    "revenue_update",
    "kdp_stage_complete",
    "audit_complete",
})

# All 16 empire site IDs for validation
EMPIRE_SITE_IDS = frozenset({
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
})


def _now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _gen_request_id() -> str:
    """Generate a short unique request identifier."""
    return f"req-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class WebhookPayload:
    """Structured payload for an outbound n8n webhook trigger."""
    webhook_path: str
    data: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = WEBHOOK_TIMEOUT

    def full_url(self) -> str:
        """Build the complete webhook URL."""
        path = self.webhook_path.lstrip("/")
        return f"{N8N_WEBHOOK_BASE}/{path}"


@dataclass
class WebhookResponse:
    """Response from an n8n webhook trigger or API call."""
    success: bool
    status_code: int
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    response_time_ms: float = 0.0

    def __bool__(self) -> bool:
        return self.success


@dataclass
class WorkflowInfo:
    """Metadata about an n8n workflow."""
    id: str
    name: str
    active: bool
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> WorkflowInfo:
        """Construct from n8n API response JSON."""
        tags_raw = data.get("tags", [])
        tag_names = []
        for t in tags_raw:
            if isinstance(t, dict):
                tag_names.append(t.get("name", str(t.get("id", ""))))
            else:
                tag_names.append(str(t))
        return cls(
            id=str(data.get("id", "")),
            name=data.get("name", ""),
            active=bool(data.get("active", False)),
            created_at=data.get("createdAt", data.get("created_at", "")),
            updated_at=data.get("updatedAt", data.get("updated_at", "")),
            tags=tag_names,
        )


# ---------------------------------------------------------------------------
# Async HTTP helpers (aiohttp-based)
# ---------------------------------------------------------------------------

async def _get_session():
    """
    Create an aiohttp ClientSession.  Imported lazily so the module can be
    loaded even when aiohttp is not installed (e.g. for type-checking).
    """
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=max(WEBHOOK_TIMEOUT, API_TIMEOUT) + 5)
    return aiohttp.ClientSession(timeout=timeout)


async def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: int = WEBHOOK_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> WebhookResponse:
    """
    Perform an HTTP request with exponential-backoff retry on 5xx errors.

    Returns a WebhookResponse regardless of success or failure.
    """
    import aiohttp

    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)

    last_error: Optional[str] = None
    last_status: int = 0

    for attempt in range(max_retries):
        start = time.monotonic()
        session = await _get_session()
        try:
            async with session:
                req_timeout = aiohttp.ClientTimeout(total=timeout)
                async with session.request(
                    method,
                    url,
                    headers=merged_headers,
                    json=json_data,
                    timeout=req_timeout,
                ) as resp:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    last_status = resp.status

                    # Try to parse response body as JSON
                    body: Optional[Dict[str, Any]] = None
                    try:
                        body = await resp.json(content_type=None)
                    except Exception:
                        raw = await resp.text()
                        body = {"raw": raw} if raw else None

                    if 200 <= resp.status < 300:
                        return WebhookResponse(
                            success=True,
                            status_code=resp.status,
                            data=body,
                            response_time_ms=round(elapsed_ms, 2),
                        )

                    # 4xx -- client error, do not retry
                    if 400 <= resp.status < 500:
                        error_msg = f"HTTP {resp.status}"
                        if body and isinstance(body, dict):
                            error_msg += f": {body.get('message', body.get('error', ''))}"
                        return WebhookResponse(
                            success=False,
                            status_code=resp.status,
                            data=body,
                            error=error_msg,
                            response_time_ms=round(elapsed_ms, 2),
                        )

                    # 5xx -- server error, retry with backoff
                    last_error = f"HTTP {resp.status}"
                    if body and isinstance(body, dict):
                        last_error += f": {body.get('message', body.get('error', ''))}"

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            last_error = f"Request timed out after {timeout}s"
            last_status = 0
        except aiohttp.ClientError as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            last_error = f"Connection error: {exc}"
            last_status = 0
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            last_error = f"Unexpected error: {exc}"
            last_status = 0

        # Backoff before retry (skip on last attempt)
        if attempt < max_retries - 1:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "Request to %s failed (attempt %d/%d): %s -- retrying in %.1fs",
                url, attempt + 1, max_retries, last_error, delay,
            )
            await asyncio.sleep(delay)

    logger.error("Request to %s failed after %d attempts: %s", url, max_retries, last_error)
    return WebhookResponse(
        success=False,
        status_code=last_status,
        error=last_error,
        response_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Sync wrapper helper
# ---------------------------------------------------------------------------

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _run_sync(coro: Awaitable[Any]) -> Any:
    """
    Run an async coroutine synchronously.

    If there is already a running event loop (e.g. inside Jupyter or an async
    framework), a new loop is spun up in a background thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already inside an event loop -- run in a dedicated thread
        future = _thread_pool.submit(asyncio.run, coro)
        return future.result(timeout=120)
    else:
        return asyncio.run(coro)


# ===================================================================
# N8nClient -- Core Integration
# ===================================================================

class N8nClient:
    """
    Async-first n8n webhook and API client for the OpenClaw Empire.

    Provides:
      - Outbound webhook triggers for all empire workflow types
      - n8n REST API access for workflow and execution management
      - Inbound webhook handler registration with HMAC verification
      - Callback registry for event-driven automation

    Every async method has a corresponding ``*_sync`` wrapper for use
    from synchronous code.
    """

    def __init__(
        self,
        webhook_base: str = N8N_WEBHOOK_BASE,
        api_base: str = N8N_API_BASE,
        api_key: str = "",
        webhook_secret: str = "",
    ) -> None:
        self.webhook_base = webhook_base.rstrip("/")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or N8N_API_KEY
        self.webhook_secret = webhook_secret or N8N_WEBHOOK_SECRET

        # Callback registry: event_type -> list of async callables
        self._callbacks: Dict[str, List[Callable[..., Awaitable[Any]]]] = {
            evt: [] for evt in CALLBACK_EVENT_TYPES
        }

        # Registered inbound webhook handlers: path -> handler info
        self._webhook_handlers: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "N8nClient initialized (webhook=%s, api=%s, key=%s)",
            self.webhook_base,
            self.api_base,
            "set" if self.api_key else "NOT SET",
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _webhook_headers(self) -> Dict[str, str]:
        """Standard headers for outbound webhook requests."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.webhook_secret:
            headers["X-Webhook-Secret"] = self.webhook_secret
        return headers

    def _api_headers(self) -> Dict[str, str]:
        """Standard headers for n8n REST API requests."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-N8N-API-KEY"] = self.api_key
        return headers

    def _build_webhook_url(self, path: str) -> str:
        """Build full webhook URL from a path segment."""
        path = path.lstrip("/")
        return f"{self.webhook_base}/{path}"

    async def _trigger_webhook(self, payload: WebhookPayload) -> WebhookResponse:
        """Send a webhook trigger to n8n."""
        url = payload.full_url()
        merged_headers = self._webhook_headers()
        if payload.headers:
            merged_headers.update(payload.headers)

        logger.info("Triggering webhook: %s", url)
        logger.debug("Payload: %s", json.dumps(payload.data, default=str)[:500])

        response = await _request_with_retry(
            "POST",
            url,
            headers=merged_headers,
            json_data=payload.data,
            timeout=payload.timeout,
        )

        if response.success:
            logger.info(
                "Webhook %s succeeded (HTTP %d, %.0fms)",
                payload.webhook_path, response.status_code, response.response_time_ms,
            )
        else:
            logger.error(
                "Webhook %s failed: %s (HTTP %d, %.0fms)",
                payload.webhook_path, response.error, response.status_code,
                response.response_time_ms,
            )

        return response

    async def _api_request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> WebhookResponse:
        """Make a request to the n8n REST API."""
        url = f"{self.api_base}/{path.lstrip('/')}"

        # Append query parameters for GET requests
        if params:
            query_parts = []
            for k, v in params.items():
                if v is not None:
                    query_parts.append(f"{k}={v}")
            if query_parts:
                url += "?" + "&".join(query_parts)

        return await _request_with_retry(
            method,
            url,
            headers=self._api_headers(),
            json_data=json_data,
            timeout=API_TIMEOUT,
            max_retries=2,  # API calls get fewer retries
        )

    # ===================================================================
    # Outbound Webhook Triggers (OpenClaw -> n8n)
    # ===================================================================

    async def trigger_content_pipeline(
        self,
        site_id: str,
        topic: str,
        action: str = "generate-article",
        keywords: Optional[List[str]] = None,
        word_count: int = 2500,
        voice: Optional[str] = None,
    ) -> WebhookResponse:
        """
        Trigger the content generation pipeline in n8n.

        Kicks off article generation for a specific empire site.  n8n will
        generate the content using the configured AI model and brand voice,
        then optionally call back with the result.

        Args:
            site_id:    Empire site identifier (e.g. "witchcraft", "smarthome").
            topic:      Article topic or title seed.
            action:     Pipeline action -- "generate-article", "outline-only",
                        "rewrite", "expand".
            keywords:   Target SEO keywords for RankMath optimization.
            word_count: Target article length (default 2500 words).
            voice:      Override brand voice (normally loaded from site config).

        Returns:
            WebhookResponse with n8n's acknowledgement or immediate result.
        """
        payload = WebhookPayload(
            webhook_path=WEBHOOK_PATHS["content"],
            data={
                "site": site_id,
                "topic": topic,
                "action": action,
                "keywords": keywords or [],
                "word_count": word_count,
                "voice": voice,
                "request_id": _gen_request_id(),
                "timestamp": _now_iso(),
            },
        )
        return await self._trigger_webhook(payload)

    def trigger_content_pipeline_sync(
        self,
        site_id: str,
        topic: str,
        action: str = "generate-article",
        keywords: Optional[List[str]] = None,
        word_count: int = 2500,
        voice: Optional[str] = None,
    ) -> WebhookResponse:
        """Synchronous wrapper for :meth:`trigger_content_pipeline`."""
        return _run_sync(self.trigger_content_pipeline(
            site_id, topic, action, keywords, word_count, voice,
        ))

    async def trigger_publish(
        self,
        site_id: str,
        post_id: Optional[int] = None,
        title: Optional[str] = None,
        content: Optional[str] = None,
        action: str = "publish",
        schedule_date: Optional[str] = None,
    ) -> WebhookResponse:
        """
        Trigger WordPress publishing via n8n.

        Can publish immediately, schedule for a future date, update an existing
        post, or draft new content.

        Args:
            site_id:       Empire site identifier.
            post_id:       Existing WordPress post ID to update (optional).
            title:         Post title (required for new posts).
            content:       Full HTML content body (required for new posts).
            action:        "publish", "draft", "schedule", "update".
            schedule_date: ISO 8601 date for scheduled publishing.

        Returns:
            WebhookResponse with the published post URL or error.
        """
        payload = WebhookPayload(
            webhook_path=WEBHOOK_PATHS["publish"],
            data={
                "site": site_id,
                "post_id": post_id,
                "title": title,
                "content": content,
                "action": action,
                "schedule_date": schedule_date,
                "request_id": _gen_request_id(),
                "timestamp": _now_iso(),
            },
        )
        return await self._trigger_webhook(payload)

    def trigger_publish_sync(
        self,
        site_id: str,
        post_id: Optional[int] = None,
        title: Optional[str] = None,
        content: Optional[str] = None,
        action: str = "publish",
        schedule_date: Optional[str] = None,
    ) -> WebhookResponse:
        """Synchronous wrapper for :meth:`trigger_publish`."""
        return _run_sync(self.trigger_publish(
            site_id, post_id, title, content, action, schedule_date,
        ))

    async def trigger_kdp(
        self,
        title: str,
        niche: str,
        action: str = "generate-outline",
        chapters: Optional[List[str]] = None,
        project_id: Optional[str] = None,
    ) -> WebhookResponse:
        """
        Trigger KDP (Kindle Direct Publishing) pipeline in n8n.

        Manages the book creation pipeline -- outline generation, chapter
        drafting, formatting, cover creation, and submission prep.

        Args:
            title:      Book title.
            niche:      Target niche/category.
            action:     "generate-outline", "draft-chapter", "format",
                        "cover-brief", "full-pipeline".
            chapters:   List of chapter titles (for draft-chapter action).
            project_id: Existing KDP project ID (for continuing work).

        Returns:
            WebhookResponse with pipeline status or generated content.
        """
        payload = WebhookPayload(
            webhook_path=WEBHOOK_PATHS["kdp"],
            data={
                "title": title,
                "niche": niche,
                "action": action,
                "chapters": chapters or [],
                "project_id": project_id,
                "request_id": _gen_request_id(),
                "timestamp": _now_iso(),
            },
        )
        return await self._trigger_webhook(payload)

    def trigger_kdp_sync(
        self,
        title: str,
        niche: str,
        action: str = "generate-outline",
        chapters: Optional[List[str]] = None,
        project_id: Optional[str] = None,
    ) -> WebhookResponse:
        """Synchronous wrapper for :meth:`trigger_kdp`."""
        return _run_sync(self.trigger_kdp(title, niche, action, chapters, project_id))

    async def trigger_monitor(
        self,
        site_id: Optional[str] = None,
        action: str = "health-check",
        check_type: str = "full",
    ) -> WebhookResponse:
        """
        Trigger site monitoring/health checks via n8n.

        Can check a single site or the entire empire.  Monitoring covers
        uptime, page speed, SSL status, plugin updates, and security scans.

        Args:
            site_id:    Site to monitor, or None for all 16 sites.
            action:     "health-check", "speed-test", "ssl-check",
                        "plugin-audit", "security-scan".
            check_type: "full", "quick", "deep".

        Returns:
            WebhookResponse with health check results.
        """
        payload = WebhookPayload(
            webhook_path=WEBHOOK_PATHS["monitor"],
            data={
                "site": site_id,
                "action": action,
                "check_type": check_type,
                "request_id": _gen_request_id(),
                "timestamp": _now_iso(),
            },
            timeout=60 if site_id is None else WEBHOOK_TIMEOUT,
        )
        return await self._trigger_webhook(payload)

    def trigger_monitor_sync(
        self,
        site_id: Optional[str] = None,
        action: str = "health-check",
        check_type: str = "full",
    ) -> WebhookResponse:
        """Synchronous wrapper for :meth:`trigger_monitor`."""
        return _run_sync(self.trigger_monitor(site_id, action, check_type))

    async def trigger_revenue(
        self,
        action: str = "daily-report",
        period: str = "today",
        source: Optional[str] = None,
    ) -> WebhookResponse:
        """
        Trigger revenue reporting via n8n.

        Aggregates revenue data from all sources -- AdSense, Amazon Associates,
        Etsy, KDP royalties, and other affiliate programs.

        Args:
            action:  "daily-report", "weekly-summary", "monthly-report",
                     "ytd-report", "source-breakdown".
            period:  "today", "yesterday", "week", "month", "quarter", "year".
            source:  Filter by revenue source (e.g. "adsense", "amazon",
                     "etsy", "kdp").  None for all sources.

        Returns:
            WebhookResponse with revenue data.
        """
        payload = WebhookPayload(
            webhook_path=WEBHOOK_PATHS["revenue"],
            data={
                "action": action,
                "period": period,
                "source": source,
                "request_id": _gen_request_id(),
                "timestamp": _now_iso(),
            },
        )
        return await self._trigger_webhook(payload)

    def trigger_revenue_sync(
        self,
        action: str = "daily-report",
        period: str = "today",
        source: Optional[str] = None,
    ) -> WebhookResponse:
        """Synchronous wrapper for :meth:`trigger_revenue`."""
        return _run_sync(self.trigger_revenue(action, period, source))

    async def trigger_audit(
        self,
        site_id: str,
        action: str = "seo-audit",
        report_type: str = "full",
    ) -> WebhookResponse:
        """
        Trigger SEO/content audit via n8n.

        Runs a comprehensive site audit covering on-page SEO, schema markup,
        internal linking, content gaps, RankMath score optimization, and
        E-E-A-T signal analysis.

        Args:
            site_id:     Empire site identifier.
            action:      "seo-audit", "content-audit", "technical-audit",
                         "link-audit", "rankmath-audit".
            report_type: "full", "summary", "actionable-only".

        Returns:
            WebhookResponse with audit results or report URL.
        """
        payload = WebhookPayload(
            webhook_path=WEBHOOK_PATHS["audit"],
            data={
                "site": site_id,
                "action": action,
                "report_type": report_type,
                "request_id": _gen_request_id(),
                "timestamp": _now_iso(),
            },
        )
        return await self._trigger_webhook(payload)

    def trigger_audit_sync(
        self,
        site_id: str,
        action: str = "seo-audit",
        report_type: str = "full",
    ) -> WebhookResponse:
        """Synchronous wrapper for :meth:`trigger_audit`."""
        return _run_sync(self.trigger_audit(site_id, action, report_type))

    async def trigger_custom(
        self,
        webhook_path: str,
        data: Dict[str, Any],
    ) -> WebhookResponse:
        """
        Trigger a custom n8n webhook.

        For ad-hoc or newly created workflows that don't have a dedicated
        trigger method.

        Args:
            webhook_path: Webhook path (e.g. "my-custom-workflow").
            data:         Arbitrary payload data.

        Returns:
            WebhookResponse from n8n.
        """
        # Inject standard metadata
        enriched_data = {
            **data,
            "request_id": data.get("request_id", _gen_request_id()),
            "timestamp": data.get("timestamp", _now_iso()),
        }
        payload = WebhookPayload(webhook_path=webhook_path, data=enriched_data)
        return await self._trigger_webhook(payload)

    def trigger_custom_sync(
        self,
        webhook_path: str,
        data: Dict[str, Any],
    ) -> WebhookResponse:
        """Synchronous wrapper for :meth:`trigger_custom`."""
        return _run_sync(self.trigger_custom(webhook_path, data))

    # ===================================================================
    # n8n REST API -- Workflow Management
    # ===================================================================

    async def list_workflows(
        self,
        active_only: bool = False,
    ) -> List[WorkflowInfo]:
        """
        List all workflows configured in n8n.

        Args:
            active_only: If True, return only workflows with active=True.

        Returns:
            List of WorkflowInfo objects sorted by name.
        """
        response = await self._api_request("GET", "/workflows")

        if not response.success:
            logger.error("Failed to list workflows: %s", response.error)
            return []

        raw_data = response.data or {}
        # n8n API may return {"data": [...]} or just [...]
        workflows_raw = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data

        if not isinstance(workflows_raw, list):
            logger.warning("Unexpected workflows response format: %s", type(workflows_raw))
            return []

        workflows = [WorkflowInfo.from_api(w) for w in workflows_raw]

        if active_only:
            workflows = [w for w in workflows if w.active]

        workflows.sort(key=lambda w: w.name.lower())
        logger.info("Listed %d workflows (%d active)", len(workflows),
                     sum(1 for w in workflows if w.active))
        return workflows

    def list_workflows_sync(self, active_only: bool = False) -> List[WorkflowInfo]:
        """Synchronous wrapper for :meth:`list_workflows`."""
        return _run_sync(self.list_workflows(active_only))

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowInfo]:
        """
        Get details for a specific workflow.

        Args:
            workflow_id: The n8n workflow ID.

        Returns:
            WorkflowInfo or None if not found.
        """
        response = await self._api_request("GET", f"/workflows/{workflow_id}")

        if not response.success:
            logger.error("Failed to get workflow %s: %s", workflow_id, response.error)
            return None

        return WorkflowInfo.from_api(response.data or {})

    def get_workflow_sync(self, workflow_id: str) -> Optional[WorkflowInfo]:
        """Synchronous wrapper for :meth:`get_workflow`."""
        return _run_sync(self.get_workflow(workflow_id))

    async def activate_workflow(self, workflow_id: str) -> bool:
        """
        Activate an n8n workflow.

        Args:
            workflow_id: The n8n workflow ID.

        Returns:
            True if activation succeeded, False otherwise.
        """
        response = await self._api_request(
            "PATCH",
            f"/workflows/{workflow_id}",
            json_data={"active": True},
        )

        if response.success:
            logger.info("Activated workflow %s", workflow_id)
            return True
        else:
            logger.error("Failed to activate workflow %s: %s", workflow_id, response.error)
            return False

    def activate_workflow_sync(self, workflow_id: str) -> bool:
        """Synchronous wrapper for :meth:`activate_workflow`."""
        return _run_sync(self.activate_workflow(workflow_id))

    async def deactivate_workflow(self, workflow_id: str) -> bool:
        """
        Deactivate an n8n workflow.

        Args:
            workflow_id: The n8n workflow ID.

        Returns:
            True if deactivation succeeded, False otherwise.
        """
        response = await self._api_request(
            "PATCH",
            f"/workflows/{workflow_id}",
            json_data={"active": False},
        )

        if response.success:
            logger.info("Deactivated workflow %s", workflow_id)
            return True
        else:
            logger.error("Failed to deactivate workflow %s: %s", workflow_id, response.error)
            return False

    def deactivate_workflow_sync(self, workflow_id: str) -> bool:
        """Synchronous wrapper for :meth:`deactivate_workflow`."""
        return _run_sync(self.deactivate_workflow(workflow_id))

    async def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        List recent workflow executions.

        Args:
            workflow_id: Filter by workflow ID (optional).
            status:      Filter by status -- "success", "error", "waiting".
            limit:       Max results to return (default 20).

        Returns:
            List of execution records from n8n.
        """
        params: Dict[str, Any] = {"limit": limit}
        if workflow_id:
            params["workflowId"] = workflow_id
        if status:
            params["status"] = status

        response = await self._api_request("GET", "/executions", params=params)

        if not response.success:
            logger.error("Failed to list executions: %s", response.error)
            return []

        raw_data = response.data or {}
        executions = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data

        if not isinstance(executions, list):
            return []

        logger.info("Listed %d executions", len(executions))
        return executions

    def list_executions_sync(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for :meth:`list_executions`."""
        return _run_sync(self.list_executions(workflow_id, status, limit))

    async def get_execution(self, execution_id: str) -> Dict[str, Any]:
        """
        Get details of a specific execution.

        Args:
            execution_id: The n8n execution ID.

        Returns:
            Execution record dict, or empty dict on failure.
        """
        response = await self._api_request("GET", f"/executions/{execution_id}")

        if not response.success:
            logger.error("Failed to get execution %s: %s", execution_id, response.error)
            return {}

        return response.data or {}

    def get_execution_sync(self, execution_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`get_execution`."""
        return _run_sync(self.get_execution(execution_id))

    # ===================================================================
    # Inbound Webhooks (n8n -> OpenClaw)
    # ===================================================================

    def verify_webhook_signature(
        self,
        payload: Union[bytes, str],
        signature: str,
        secret: Optional[str] = None,
    ) -> bool:
        """
        Verify HMAC-SHA256 signature on an inbound webhook from n8n.

        Args:
            payload:   Raw request body (bytes or str).
            signature: The signature header value (hex-encoded HMAC).
            secret:    Shared secret for verification.  Defaults to
                       the instance webhook_secret.

        Returns:
            True if the signature is valid, False otherwise.
        """
        secret = secret or self.webhook_secret
        if not secret:
            logger.warning("No webhook secret configured -- skipping verification")
            return True

        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        expected = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        is_valid = hmac.compare_digest(expected, signature)

        if not is_valid:
            logger.warning("Webhook signature verification FAILED")

        return is_valid

    def create_webhook_handler(
        self,
        path: str,
        callback: Callable[..., Awaitable[Any]],
        secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a handler for inbound n8n webhook callbacks.

        This creates an internal registration that :class:`WebhookServer`
        uses when mounting routes on a FastAPI application.

        Args:
            path:     URL path segment (e.g. "content-ready").
            callback: Async function to call with the parsed JSON body.
            secret:   Optional per-handler HMAC secret.  Falls back to
                      the instance-level webhook_secret.

        Returns:
            Handler registration info dict.
        """
        handler_id = f"handler-{uuid.uuid4().hex[:8]}"
        handler_info = {
            "id": handler_id,
            "path": path,
            "callback": callback,
            "secret": secret or self.webhook_secret,
            "created_at": _now_iso(),
        }
        self._webhook_handlers[path] = handler_info

        logger.info(
            "Registered webhook handler: path=/%s  id=%s  signed=%s",
            path, handler_id, bool(handler_info["secret"]),
        )
        return {k: v for k, v in handler_info.items() if k != "callback"}

    def get_webhook_handlers(self) -> Dict[str, Dict[str, Any]]:
        """Return all registered webhook handlers (without callback references)."""
        return {
            path: {k: v for k, v in info.items() if k != "callback"}
            for path, info in self._webhook_handlers.items()
        }

    # ===================================================================
    # Callback Registry
    # ===================================================================

    def register_callback(
        self,
        event_type: str,
        callback_fn: Callable[..., Awaitable[Any]],
    ) -> None:
        """
        Register an async callback for a specific event type.

        Supported event types:
            - content_ready       -- n8n finished generating content
            - publish_complete    -- WordPress publish succeeded
            - monitor_alert       -- site health issue detected
            - revenue_update      -- new revenue data available
            - kdp_stage_complete  -- KDP pipeline stage finished
            - audit_complete      -- SEO audit results ready

        Args:
            event_type:  One of the supported event types above.
            callback_fn: Async callable that receives the event data dict.

        Raises:
            ValueError: If event_type is not recognized.
        """
        if event_type not in CALLBACK_EVENT_TYPES:
            raise ValueError(
                f"Unknown event type '{event_type}'.  "
                f"Valid types: {sorted(CALLBACK_EVENT_TYPES)}"
            )

        self._callbacks[event_type].append(callback_fn)
        logger.info(
            "Registered callback for '%s' (total: %d)",
            event_type, len(self._callbacks[event_type]),
        )

    async def dispatch_callback(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Fire all registered callbacks for an event type.

        Each callback is invoked with the event data dict.  Errors in
        individual callbacks are caught and logged without affecting others.

        Args:
            event_type: The event type to dispatch.
            data:       Event payload from n8n.

        Returns:
            List of result dicts with callback outcomes.
        """
        if event_type not in CALLBACK_EVENT_TYPES:
            logger.warning("Dispatch for unknown event type: %s", event_type)
            return []

        handlers = self._callbacks.get(event_type, [])
        if not handlers:
            logger.debug("No callbacks registered for '%s'", event_type)
            return []

        logger.info(
            "Dispatching '%s' to %d callback(s)", event_type, len(handlers),
        )

        results: List[Dict[str, Any]] = []
        for i, handler in enumerate(handlers):
            try:
                result = await handler(data)
                results.append({
                    "handler_index": i,
                    "success": True,
                    "result": result,
                })
            except Exception as exc:
                logger.error(
                    "Callback %d for '%s' raised %s: %s",
                    i, event_type, type(exc).__name__, exc,
                )
                results.append({
                    "handler_index": i,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        succeeded = sum(1 for r in results if r["success"])
        logger.info(
            "Dispatched '%s': %d/%d succeeded", event_type, succeeded, len(results),
        )
        return results

    def dispatch_callback_sync(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for :meth:`dispatch_callback`."""
        return _run_sync(self.dispatch_callback(event_type, data))


# ===================================================================
# WebhookServer -- Inbound Route Mounter
# ===================================================================

class WebhookServer:
    """
    Lightweight helper that mounts inbound webhook routes on an existing
    FastAPI application.

    n8n workflow nodes can POST to these routes when workflows complete,
    errors occur, or status updates are available.  All inbound requests
    are verified via HMAC-SHA256 signatures.

    Usage:
        from fastapi import FastAPI
        from src.n8n_client import get_n8n_client, WebhookServer

        app = FastAPI()
        client = get_n8n_client()

        server = WebhookServer(client)
        server.mount(app, prefix="/webhook/n8n")
    """

    def __init__(self, client: N8nClient) -> None:
        self.client = client
        self._mounted = False

    def mount(self, app: Any, prefix: str = "/webhook/n8n") -> None:
        """
        Add inbound webhook routes to a FastAPI application.

        Creates a catch-all POST route at ``{prefix}/{event_type}`` that:
          1. Reads the raw request body
          2. Verifies the HMAC-SHA256 signature (``X-Webhook-Signature`` header)
          3. Parses the JSON body
          4. Dispatches to registered callbacks via the N8nClient

        Also adds a GET ``{prefix}/health`` endpoint for connectivity checks.

        Args:
            app:    FastAPI application instance.
            prefix: URL prefix for all inbound webhook routes.
        """
        # Import FastAPI types lazily
        from fastapi import Request
        from fastapi.responses import JSONResponse

        prefix = prefix.rstrip("/")

        @app.get(f"{prefix}/health")
        async def n8n_webhook_health():
            """Health check for n8n webhook receiver."""
            handlers = self.client.get_webhook_handlers()
            callback_counts = {
                evt: len(cbs) for evt, cbs in self.client._callbacks.items() if cbs
            }
            return {
                "status": "ok",
                "service": "n8n-webhook-receiver",
                "registered_handlers": len(handlers),
                "callback_counts": callback_counts,
                "timestamp": _now_iso(),
            }

        @app.post(f"{prefix}/{{event_type}}")
        async def n8n_webhook_receiver(event_type: str, request: Request):
            """
            Receive and dispatch inbound n8n webhook callbacks.

            Validates the HMAC signature, parses the body, and dispatches
            to all registered callbacks for the event type.
            """
            # Read raw body for signature verification
            raw_body = await request.body()
            signature = request.headers.get("X-Webhook-Signature", "")

            # Determine which secret to use (handler-specific or default)
            handler_info = self.client._webhook_handlers.get(event_type)
            secret = None
            if handler_info:
                secret = handler_info.get("secret")

            # Verify signature
            if secret or self.client.webhook_secret:
                effective_secret = secret or self.client.webhook_secret
                if not self.client.verify_webhook_signature(
                    raw_body, signature, effective_secret
                ):
                    logger.warning(
                        "Rejected inbound webhook /%s -- invalid signature",
                        event_type,
                    )
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Invalid webhook signature"},
                    )

            # Parse body
            try:
                data = json.loads(raw_body) if raw_body else {}
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid JSON body"},
                )

            logger.info(
                "Received inbound webhook: event=%s  keys=%s",
                event_type, list(data.keys()) if isinstance(data, dict) else "non-dict",
            )

            # Dispatch to handler-specific callback if registered
            if handler_info and "callback" in handler_info:
                try:
                    await handler_info["callback"](data)
                except Exception as exc:
                    logger.error(
                        "Handler callback for /%s failed: %s", event_type, exc,
                    )

            # Dispatch to event-type callbacks
            results = []
            if event_type in CALLBACK_EVENT_TYPES:
                results = await self.client.dispatch_callback(event_type, data)

            return {
                "received": True,
                "event_type": event_type,
                "callbacks_fired": len(results),
                "timestamp": _now_iso(),
            }

        self._mounted = True
        logger.info(
            "WebhookServer mounted at %s (handlers: %d, event types: %d)",
            prefix,
            len(self.client._webhook_handlers),
            len([e for e in CALLBACK_EVENT_TYPES if self.client._callbacks.get(e)]),
        )


# ===================================================================
# N8nEmpireIntegration -- High-Level Orchestrator
# ===================================================================

class N8nEmpireIntegration:
    """
    High-level orchestrator wrapping :class:`N8nClient` with empire-specific
    composite workflows.

    Provides multi-step operations that coordinate multiple n8n webhook
    triggers, handle callbacks, and return consolidated results.

    Usage:
        from src.n8n_client import get_empire_integration

        empire = get_empire_integration()
        result = await empire.generate_and_publish("witchcraft", "Full Moon Ritual")
        summary = await empire.daily_revenue_summary()
    """

    def __init__(self, client: Optional[N8nClient] = None) -> None:
        self.client = client or get_n8n_client()

        # Track pending multi-step operations
        self._pending_operations: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------
    # Composite Workflows
    # -------------------------------------------------------------------

    async def generate_and_publish(
        self,
        site_id: str,
        topic: str,
        keywords: Optional[List[str]] = None,
        word_count: int = 2500,
        auto_publish: bool = True,
        schedule_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full content pipeline: generate article via n8n, then publish.

        Steps:
          1. Trigger content generation pipeline
          2. If successful, trigger publish workflow
          3. Return consolidated result

        For truly async pipelines where n8n generates content asynchronously
        and calls back, register a ``content_ready`` callback instead.

        Args:
            site_id:       Empire site identifier.
            topic:         Article topic.
            keywords:      SEO target keywords.
            word_count:    Target article length.
            auto_publish:  If True, immediately publish after generation.
            schedule_date: Schedule for future publishing instead.

        Returns:
            Dict with content generation and publishing results.
        """
        operation_id = _gen_request_id()
        result: Dict[str, Any] = {
            "operation_id": operation_id,
            "site_id": site_id,
            "topic": topic,
            "started_at": _now_iso(),
            "steps": {},
        }

        # Step 1: Generate content
        logger.info(
            "[%s] Starting generate_and_publish for '%s' on %s",
            operation_id, topic, site_id,
        )

        content_response = await self.client.trigger_content_pipeline(
            site_id=site_id,
            topic=topic,
            keywords=keywords,
            word_count=word_count,
        )

        result["steps"]["content_generation"] = {
            "success": content_response.success,
            "status_code": content_response.status_code,
            "response_time_ms": content_response.response_time_ms,
            "data": content_response.data,
            "error": content_response.error,
        }

        if not content_response.success:
            result["success"] = False
            result["error"] = f"Content generation failed: {content_response.error}"
            result["completed_at"] = _now_iso()
            logger.error("[%s] Content generation failed: %s", operation_id, content_response.error)
            return result

        # Step 2: Publish (if content generation returned content directly)
        if auto_publish or schedule_date:
            # Extract generated content from n8n response
            content_data = content_response.data or {}
            title = content_data.get("title", topic)
            content = content_data.get("content", "")
            post_id = content_data.get("post_id")

            action = "schedule" if schedule_date else "publish"

            publish_response = await self.client.trigger_publish(
                site_id=site_id,
                post_id=post_id,
                title=title,
                content=content,
                action=action,
                schedule_date=schedule_date,
            )

            result["steps"]["publishing"] = {
                "success": publish_response.success,
                "status_code": publish_response.status_code,
                "response_time_ms": publish_response.response_time_ms,
                "data": publish_response.data,
                "error": publish_response.error,
            }

            if not publish_response.success:
                result["success"] = False
                result["error"] = f"Publishing failed: {publish_response.error}"
            else:
                result["success"] = True
                result["post_url"] = (publish_response.data or {}).get("url", "")
        else:
            result["success"] = True
            result["note"] = "Content generated; auto_publish=False, not published"

        result["completed_at"] = _now_iso()
        logger.info(
            "[%s] generate_and_publish completed: success=%s",
            operation_id, result["success"],
        )
        return result

    def generate_and_publish_sync(
        self,
        site_id: str,
        topic: str,
        keywords: Optional[List[str]] = None,
        word_count: int = 2500,
        auto_publish: bool = True,
        schedule_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`generate_and_publish`."""
        return _run_sync(self.generate_and_publish(
            site_id, topic, keywords, word_count, auto_publish, schedule_date,
        ))

    async def health_check_empire(self) -> Dict[str, Any]:
        """
        Trigger health checks across all 16 empire sites.

        Sends a monitoring webhook for each site and aggregates the results
        into a consolidated empire health report.

        Returns:
            Dict with per-site health status and empire-wide summary.
        """
        logger.info("Starting empire-wide health check (all %d sites)", len(EMPIRE_SITE_IDS))
        start = time.monotonic()

        # Trigger individual site checks concurrently
        tasks = []
        for site_id in sorted(EMPIRE_SITE_IDS):
            tasks.append(self._check_single_site(site_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        site_results: Dict[str, Any] = {}
        healthy = 0
        unhealthy = 0
        errors = 0

        for site_id, result in zip(sorted(EMPIRE_SITE_IDS), results):
            if isinstance(result, Exception):
                site_results[site_id] = {
                    "status": "error",
                    "error": str(result),
                }
                errors += 1
            elif isinstance(result, dict):
                site_results[site_id] = result
                if result.get("success"):
                    healthy += 1
                else:
                    unhealthy += 1
            else:
                site_results[site_id] = {"status": "unknown"}
                errors += 1

        elapsed_ms = (time.monotonic() - start) * 1000

        report = {
            "empire_health": {
                "total_sites": len(EMPIRE_SITE_IDS),
                "healthy": healthy,
                "unhealthy": unhealthy,
                "errors": errors,
                "check_duration_ms": round(elapsed_ms, 2),
                "timestamp": _now_iso(),
            },
            "sites": site_results,
        }

        logger.info(
            "Empire health check complete: %d healthy, %d unhealthy, %d errors (%.0fms)",
            healthy, unhealthy, errors, elapsed_ms,
        )
        return report

    def health_check_empire_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`health_check_empire`."""
        return _run_sync(self.health_check_empire())

    async def _check_single_site(self, site_id: str) -> Dict[str, Any]:
        """Check health of a single site via the monitor webhook."""
        response = await self.client.trigger_monitor(
            site_id=site_id,
            action="health-check",
            check_type="quick",
        )
        return {
            "success": response.success,
            "status_code": response.status_code,
            "response_time_ms": response.response_time_ms,
            "data": response.data,
            "error": response.error,
        }

    async def daily_revenue_summary(self) -> Dict[str, Any]:
        """
        Generate a daily revenue summary across all empire revenue sources.

        Triggers the revenue webhook and returns a formatted summary.

        Returns:
            Dict with revenue data and summary statistics.
        """
        logger.info("Requesting daily revenue summary")

        response = await self.client.trigger_revenue(
            action="daily-report",
            period="today",
        )

        result: Dict[str, Any] = {
            "request_time": _now_iso(),
            "success": response.success,
        }

        if response.success:
            result["revenue_data"] = response.data
            result["response_time_ms"] = response.response_time_ms
        else:
            result["error"] = response.error

        return result

    def daily_revenue_summary_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`daily_revenue_summary`."""
        return _run_sync(self.daily_revenue_summary())

    async def bulk_content_pipeline(
        self,
        tasks: List[Dict[str, Any]],
        rate_limit_seconds: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple content tasks through n8n with rate limiting.

        Each task dict should contain at minimum:
            - site_id (str)
            - topic (str)

        Optional keys: keywords (list), word_count (int), voice (str).

        Args:
            tasks:              List of content task dicts.
            rate_limit_seconds: Delay between each webhook trigger to avoid
                                overwhelming n8n (default 2 seconds).

        Returns:
            List of result dicts, one per task.
        """
        if not tasks:
            logger.warning("bulk_content_pipeline called with empty task list")
            return []

        logger.info(
            "Starting bulk content pipeline: %d tasks, %.1fs rate limit",
            len(tasks), rate_limit_seconds,
        )

        results: List[Dict[str, Any]] = []

        for i, task in enumerate(tasks):
            site_id = task.get("site_id", "")
            topic = task.get("topic", "")

            if not site_id or not topic:
                results.append({
                    "task_index": i,
                    "site_id": site_id,
                    "topic": topic,
                    "success": False,
                    "error": "Missing required 'site_id' or 'topic'",
                })
                continue

            logger.info(
                "Bulk task %d/%d: site=%s topic='%s'",
                i + 1, len(tasks), site_id, topic[:60],
            )

            response = await self.client.trigger_content_pipeline(
                site_id=site_id,
                topic=topic,
                keywords=task.get("keywords"),
                word_count=task.get("word_count", 2500),
                voice=task.get("voice"),
            )

            results.append({
                "task_index": i,
                "site_id": site_id,
                "topic": topic,
                "success": response.success,
                "status_code": response.status_code,
                "response_time_ms": response.response_time_ms,
                "data": response.data,
                "error": response.error,
            })

            # Rate limit between tasks (skip after last task)
            if i < len(tasks) - 1 and rate_limit_seconds > 0:
                await asyncio.sleep(rate_limit_seconds)

        succeeded = sum(1 for r in results if r["success"])
        logger.info(
            "Bulk content pipeline complete: %d/%d succeeded", succeeded, len(results),
        )
        return results

    def bulk_content_pipeline_sync(
        self,
        tasks: List[Dict[str, Any]],
        rate_limit_seconds: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for :meth:`bulk_content_pipeline`."""
        return _run_sync(self.bulk_content_pipeline(tasks, rate_limit_seconds))


# ===================================================================
# Singletons
# ===================================================================

_n8n_client_instance: Optional[N8nClient] = None
_empire_integration_instance: Optional[N8nEmpireIntegration] = None


def get_n8n_client() -> N8nClient:
    """
    Return a module-level singleton :class:`N8nClient`.

    Thread-safe for typical usage patterns.  The singleton is created on
    first call using environment-variable configuration.
    """
    global _n8n_client_instance
    if _n8n_client_instance is None:
        _n8n_client_instance = N8nClient()
    return _n8n_client_instance


def get_empire_integration() -> N8nEmpireIntegration:
    """
    Return a module-level singleton :class:`N8nEmpireIntegration`.

    Uses the singleton :class:`N8nClient` internally.
    """
    global _empire_integration_instance
    if _empire_integration_instance is None:
        _empire_integration_instance = N8nEmpireIntegration(client=get_n8n_client())
    return _empire_integration_instance


# ===================================================================
# CLI
# ===================================================================

def _format_workflow_table(workflows: List[WorkflowInfo]) -> str:
    """Format workflows as a readable ASCII table."""
    if not workflows:
        return "  (no workflows found)"

    # Compute column widths
    id_width = max(len(w.id) for w in workflows)
    name_width = max(len(w.name) for w in workflows)
    id_width = max(id_width, 2)
    name_width = min(max(name_width, 4), 50)

    lines = []
    header = f"  {'ID':<{id_width}}  {'Name':<{name_width}}  {'Status':<10}  Tags"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for w in workflows:
        status = "ACTIVE" if w.active else "inactive"
        name_display = w.name[:name_width]
        tags_display = ", ".join(w.tags) if w.tags else "-"
        lines.append(f"  {w.id:<{id_width}}  {name_display:<{name_width}}  {status:<10}  {tags_display}")

    return "\n".join(lines)


def _format_executions_table(executions: List[Dict[str, Any]]) -> str:
    """Format executions as a readable list."""
    if not executions:
        return "  (no executions found)"

    lines = []
    for ex in executions:
        ex_id = ex.get("id", "?")
        wf_name = ex.get("workflowData", {}).get("name", ex.get("workflowId", "?"))
        status = ex.get("status", ex.get("finished", "?"))
        started = ex.get("startedAt", "?")
        if isinstance(started, str) and len(started) > 19:
            started = started[:19]
        lines.append(f"  [{ex_id}]  {wf_name}  status={status}  started={started}")

    return "\n".join(lines)


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.n8n_client",
        description="OpenClaw Empire -- n8n Webhook Integration Client",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- status ---
    subparsers.add_parser("status", help="List all n8n workflows and their status")

    # --- trigger ---
    trigger_parser = subparsers.add_parser("trigger", help="Trigger an n8n webhook")
    trigger_sub = trigger_parser.add_subparsers(dest="trigger_type", help="Webhook type")

    # trigger content
    content_parser = trigger_sub.add_parser("content", help="Trigger content pipeline")
    content_parser.add_argument("--site", required=True, help="Site ID")
    content_parser.add_argument("--topic", required=True, help="Article topic")
    content_parser.add_argument("--action", default="generate-article", help="Pipeline action")
    content_parser.add_argument("--keywords", nargs="*", help="SEO keywords")
    content_parser.add_argument("--word-count", type=int, default=2500, help="Target word count")
    content_parser.add_argument("--voice", help="Override brand voice")

    # trigger publish
    publish_parser = trigger_sub.add_parser("publish", help="Trigger publishing")
    publish_parser.add_argument("--site", required=True, help="Site ID")
    publish_parser.add_argument("--post-id", type=int, help="WordPress post ID")
    publish_parser.add_argument("--title", help="Post title")
    publish_parser.add_argument("--action", default="publish", help="publish/draft/schedule/update")
    publish_parser.add_argument("--schedule-date", help="ISO 8601 schedule date")

    # trigger kdp
    kdp_parser = trigger_sub.add_parser("kdp", help="Trigger KDP pipeline")
    kdp_parser.add_argument("--title", required=True, help="Book title")
    kdp_parser.add_argument("--niche", required=True, help="Target niche")
    kdp_parser.add_argument("--action", default="generate-outline", help="Pipeline action")
    kdp_parser.add_argument("--chapters", nargs="*", help="Chapter titles")

    # trigger monitor
    monitor_parser = trigger_sub.add_parser("monitor", help="Trigger monitoring")
    monitor_parser.add_argument("--site", help="Site ID (omit for all sites)")
    monitor_parser.add_argument("--all", action="store_true", help="Monitor all 16 sites")
    monitor_parser.add_argument("--action", default="health-check", help="Monitor action")
    monitor_parser.add_argument("--check-type", default="full", help="full/quick/deep")

    # trigger revenue
    revenue_parser = trigger_sub.add_parser("revenue", help="Trigger revenue report")
    revenue_parser.add_argument("--action", default="daily-report", help="Report action")
    revenue_parser.add_argument("--period", default="today", help="today/yesterday/week/month/quarter/year")
    revenue_parser.add_argument("--source", help="Revenue source filter")

    # trigger audit
    audit_parser = trigger_sub.add_parser("audit", help="Trigger SEO audit")
    audit_parser.add_argument("--site", required=True, help="Site ID")
    audit_parser.add_argument("--action", default="seo-audit", help="Audit action")
    audit_parser.add_argument("--report-type", default="full", help="full/summary/actionable-only")

    # --- executions ---
    exec_parser = subparsers.add_parser("executions", help="List recent executions")
    exec_parser.add_argument("--workflow-id", help="Filter by workflow ID")
    exec_parser.add_argument("--status", help="Filter by status (success/error/waiting)")
    exec_parser.add_argument("--limit", type=int, default=10, help="Max results")

    return parser


async def _cli_main(args: argparse.Namespace) -> int:
    """Execute the CLI command (async)."""
    client = get_n8n_client()

    if args.command == "status":
        print("Fetching n8n workflows...")
        workflows = await client.list_workflows()
        active = sum(1 for w in workflows if w.active)
        print(f"\nn8n Workflows ({len(workflows)} total, {active} active):\n")
        print(_format_workflow_table(workflows))
        return 0

    elif args.command == "trigger":
        if args.trigger_type == "content":
            print(f"Triggering content pipeline: site={args.site} topic='{args.topic}'")
            response = await client.trigger_content_pipeline(
                site_id=args.site,
                topic=args.topic,
                action=args.action,
                keywords=args.keywords,
                word_count=args.word_count,
                voice=args.voice,
            )

        elif args.trigger_type == "publish":
            print(f"Triggering publish: site={args.site}")
            response = await client.trigger_publish(
                site_id=args.site,
                post_id=args.post_id,
                title=args.title,
                action=args.action,
                schedule_date=args.schedule_date,
            )

        elif args.trigger_type == "kdp":
            print(f"Triggering KDP pipeline: title='{args.title}' niche='{args.niche}'")
            response = await client.trigger_kdp(
                title=args.title,
                niche=args.niche,
                action=args.action,
                chapters=args.chapters,
            )

        elif args.trigger_type == "monitor":
            site = None if args.all else args.site
            label = "all sites" if site is None else f"site={site}"
            print(f"Triggering monitor: {label}")
            response = await client.trigger_monitor(
                site_id=site,
                action=args.action,
                check_type=args.check_type,
            )

        elif args.trigger_type == "revenue":
            print(f"Triggering revenue report: period={args.period}")
            response = await client.trigger_revenue(
                action=args.action,
                period=args.period,
                source=args.source,
            )

        elif args.trigger_type == "audit":
            print(f"Triggering audit: site={args.site}")
            response = await client.trigger_audit(
                site_id=args.site,
                action=args.action,
                report_type=args.report_type,
            )

        else:
            print("Error: specify a trigger type (content/publish/kdp/monitor/revenue/audit)")
            return 1

        # Display result
        print(f"\nResult: {'SUCCESS' if response.success else 'FAILED'}")
        print(f"  HTTP Status:   {response.status_code}")
        print(f"  Response Time: {response.response_time_ms:.0f}ms")
        if response.error:
            print(f"  Error:         {response.error}")
        if response.data:
            print(f"  Data:          {json.dumps(response.data, indent=2, default=str)[:1000]}")

        return 0 if response.success else 1

    elif args.command == "executions":
        print("Fetching recent executions...")
        executions = await client.list_executions(
            workflow_id=args.workflow_id,
            status=args.status,
            limit=args.limit,
        )
        print(f"\nRecent Executions ({len(executions)}):\n")
        print(_format_executions_table(executions))
        return 0

    else:
        print("No command specified. Use --help for usage.")
        return 1


def cli_entry() -> None:
    """CLI entry point for ``python -m src.n8n_client``."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    exit_code = asyncio.run(_cli_main(args))
    sys.exit(exit_code)


# ===================================================================
# Module entry point
# ===================================================================

if __name__ == "__main__":
    cli_entry()
