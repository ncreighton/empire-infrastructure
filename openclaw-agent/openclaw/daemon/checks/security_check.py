"""Security health checks — WordPress plugin audit.

DAILY tier: runs every 24 hours.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx

from openclaw.models import CheckResult, HealthCheck, HeartbeatTier

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0


async def _check_site_plugins(
    client: httpx.AsyncClient,
    domain: str,
    wp_user: str,
    wp_pass: str,
) -> HealthCheck:
    """Check plugin list for a single WordPress site."""
    check = HealthCheck(
        name=f"security:{domain}",
        tier=HeartbeatTier.DAILY,
        result=CheckResult.UNKNOWN,
    )
    start = time.monotonic()

    try:
        auth = (wp_user, wp_pass)
        resp = await client.get(
            f"https://{domain}/wp-json/wp/v2/plugins",
            auth=auth,
            follow_redirects=True,
        )

        if resp.status_code == 401:
            check.result = CheckResult.UNKNOWN
            check.message = "Plugin endpoint requires authentication"
        elif resp.status_code != 200:
            check.result = CheckResult.DEGRADED
            check.message = f"Plugin API returned {resp.status_code}"
        else:
            plugins = resp.json()
            active_plugins = [p for p in plugins if p.get("status") == "active"]
            check.result = CheckResult.HEALTHY
            check.message = f"{len(active_plugins)} active plugins"
            check.details = {
                "total_plugins": len(plugins),
                "active_plugins": len(active_plugins),
                "plugin_names": [p.get("name", "?") for p in active_plugins],
            }

    except Exception as e:
        check.result = CheckResult.UNKNOWN
        check.message = f"Plugin check failed: {str(e)[:80]}"

    check.duration_ms = (time.monotonic() - start) * 1000
    return check


async def check_plugin_security() -> list[HealthCheck]:
    """Check WordPress plugin security across all sites.

    Reads credentials from config/sites.json to access plugin endpoints.

    Returns:
        List of HealthCheck results.
    """
    # Load site configs
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent.parent / "config" / "sites.json",
        Path("D:/Claude Code Projects/config/sites.json"),
    ]
    sites_data = None
    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    sites_data = json.load(f)
                break
            except (json.JSONDecodeError, OSError):
                pass

    if not sites_data:
        return [HealthCheck(
            name="security:plugins",
            tier=HeartbeatTier.DAILY,
            result=CheckResult.UNKNOWN,
            message="config/sites.json not found — cannot check plugin security",
        )]

    sites = sites_data.get("sites", sites_data)
    checks: list[HealthCheck] = []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        tasks = []
        for site_id, config in sites.items():
            domain = config.get("domain", "")
            wp = config.get("wordpress", {})
            wp_user = wp.get("user", "")
            wp_pass = wp.get("app_password", "")
            if domain and wp_user and wp_pass:
                tasks.append(_check_site_plugins(client, domain, wp_user, wp_pass))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    checks.append(HealthCheck(
                        name="security:error",
                        tier=HeartbeatTier.DAILY,
                        result=CheckResult.DOWN,
                        message=str(result)[:80],
                    ))
                else:
                    checks.append(result)

    if not checks:
        checks.append(HealthCheck(
            name="security:plugins",
            tier=HeartbeatTier.DAILY,
            result=CheckResult.UNKNOWN,
            message="No sites with WordPress credentials configured",
        ))

    return checks
