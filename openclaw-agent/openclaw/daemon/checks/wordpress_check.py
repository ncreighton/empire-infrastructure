"""WordPress site health checks — HTTP 200 + REST API verification.

PULSE tier: runs every 5 minutes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from openclaw.models import CheckResult, HealthCheck, HeartbeatTier

logger = logging.getLogger(__name__)

# Timeout per site (seconds)
_TIMEOUT = 10.0

# Response time threshold for DEGRADED (seconds)
_SLOW_THRESHOLD = 3.0


async def _check_single_site(
    client: httpx.AsyncClient,
    domain: str,
) -> HealthCheck:
    """Check a single WordPress site for HTTP 200 + REST API health."""
    check = HealthCheck(
        name=f"wp:{domain}",
        tier=HeartbeatTier.PULSE,
        result=CheckResult.UNKNOWN,
    )
    start = time.monotonic()
    details: dict[str, Any] = {}

    try:
        # Homepage check
        resp = await client.get(f"https://{domain}/", follow_redirects=True)
        details["status_code"] = resp.status_code
        elapsed = time.monotonic() - start
        details["response_time_s"] = round(elapsed, 2)

        if resp.status_code >= 500:
            check.result = CheckResult.DOWN
            check.message = f"HTTP {resp.status_code}"
        elif resp.status_code >= 400:
            check.result = CheckResult.DEGRADED
            check.message = f"HTTP {resp.status_code}"
        elif elapsed > _SLOW_THRESHOLD:
            check.result = CheckResult.DEGRADED
            check.message = f"Slow response: {elapsed:.1f}s"
        else:
            check.result = CheckResult.HEALTHY
            check.message = f"OK ({elapsed:.1f}s)"

        # REST API check (only if homepage is up)
        if check.result != CheckResult.DOWN:
            try:
                api_resp = await client.get(
                    f"https://{domain}/wp-json/wp/v2/posts?per_page=1",
                    follow_redirects=True,
                )
                details["api_status"] = api_resp.status_code
                if api_resp.status_code != 200:
                    if check.result == CheckResult.HEALTHY:
                        check.result = CheckResult.DEGRADED
                        check.message += f" (REST API: {api_resp.status_code})"
            except Exception as e:
                details["api_error"] = str(e)[:100]
                if check.result == CheckResult.HEALTHY:
                    check.result = CheckResult.DEGRADED
                    check.message += " (REST API unreachable)"

    except httpx.TimeoutException:
        check.result = CheckResult.DOWN
        check.message = f"Timeout after {_TIMEOUT}s"
    except httpx.ConnectError as e:
        check.result = CheckResult.DOWN
        check.message = f"Connection failed: {str(e)[:80]}"
    except Exception as e:
        check.result = CheckResult.DOWN
        check.message = f"Error: {str(e)[:80]}"

    check.duration_ms = (time.monotonic() - start) * 1000
    check.details = details
    return check


async def check_all_sites(domains: list[str]) -> list[HealthCheck]:
    """Check all WordPress sites concurrently.

    Args:
        domains: List of domain names (e.g., ["smarthomewizards.com"]).

    Returns:
        List of HealthCheck results, one per domain.
    """
    if not domains:
        return []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        tasks = [_check_single_site(client, d) for d in domains]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    checks = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            checks.append(HealthCheck(
                name=f"wp:{domains[i]}",
                tier=HeartbeatTier.PULSE,
                result=CheckResult.DOWN,
                message=f"Check failed: {str(result)[:80]}",
            ))
        else:
            checks.append(result)

    return checks
