"""n8n workflow health checks — verify recent executions via n8n API.

SCAN tier: runs every 30 minutes.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from openclaw.models import CheckResult, HealthCheck, HeartbeatTier

logger = logging.getLogger(__name__)

# Known workflow IDs from EMPIRE-BRAIN
_KNOWN_WORKFLOWS = {
    "ROrZ0Gn2YMpt3o4o": "Data Receiver",
    "cFeiIJsXJD273T5B": "Pattern Detector",
    "MmfLLGeDzoUMmXGt": "Opportunity Finder",
    "c0XCDohiUC3sjC3M": "Morning Briefing",
    "bkUxMuGgcqxRWHKQ": "Init Schema",
}

_N8N_BASE = os.environ.get(
    "N8N_BASE_URL",
    "http://empire-n8n:5678" if os.path.exists("/.dockerenv") else "http://localhost:5678",
)
_TIMEOUT = 10.0


async def check_workflows() -> list[HealthCheck]:
    """Check n8n workflow health via API.

    Queries recent executions and flags errors.

    Returns:
        List of HealthCheck results (one overall + per-workflow if errors).
    """
    api_key = os.environ.get("N8N_API_KEY", "")
    if not api_key:
        return [HealthCheck(
            name="n8n:api",
            tier=HeartbeatTier.SCAN,
            result=CheckResult.UNKNOWN,
            message="N8N_API_KEY not configured",
        )]

    checks: list[HealthCheck] = []
    headers = {"X-N8N-API-KEY": api_key}

    try:
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            # Check for recent errors
            resp = await client.get(
                f"{_N8N_BASE}/api/v1/executions",
                params={"limit": 10, "status": "error"},
                headers=headers,
            )

            if resp.status_code != 200:
                checks.append(HealthCheck(
                    name="n8n:api",
                    tier=HeartbeatTier.SCAN,
                    result=CheckResult.DEGRADED,
                    message=f"n8n API returned {resp.status_code}",
                    duration_ms=(time.monotonic() - start) * 1000,
                ))
                return checks

            data = resp.json()
            executions = data.get("data", [])

            if not executions:
                checks.append(HealthCheck(
                    name="n8n:overall",
                    tier=HeartbeatTier.SCAN,
                    result=CheckResult.HEALTHY,
                    message="No recent execution errors",
                    duration_ms=(time.monotonic() - start) * 1000,
                ))
            else:
                # Group errors by workflow
                error_workflows: dict[str, int] = {}
                for ex in executions:
                    wf_id = ex.get("workflowId", "unknown")
                    wf_name = _KNOWN_WORKFLOWS.get(wf_id, wf_id)
                    error_workflows[wf_name] = error_workflows.get(wf_name, 0) + 1

                checks.append(HealthCheck(
                    name="n8n:overall",
                    tier=HeartbeatTier.SCAN,
                    result=CheckResult.DEGRADED,
                    message=f"{len(executions)} recent error(s) across {len(error_workflows)} workflow(s)",
                    details={"error_workflows": error_workflows},
                    duration_ms=(time.monotonic() - start) * 1000,
                ))

                for wf_name, count in error_workflows.items():
                    checks.append(HealthCheck(
                        name=f"n8n:{wf_name}",
                        tier=HeartbeatTier.SCAN,
                        result=CheckResult.DEGRADED,
                        message=f"{count} recent error(s)",
                    ))

    except httpx.ConnectError:
        checks.append(HealthCheck(
            name="n8n:api",
            tier=HeartbeatTier.SCAN,
            result=CheckResult.DOWN,
            message="Cannot connect to n8n API (port 5679)",
        ))
    except Exception as e:
        checks.append(HealthCheck(
            name="n8n:api",
            tier=HeartbeatTier.SCAN,
            result=CheckResult.DOWN,
            message=f"n8n check failed: {str(e)[:80]}",
        ))

    return checks
