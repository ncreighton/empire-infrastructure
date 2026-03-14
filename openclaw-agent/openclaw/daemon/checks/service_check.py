"""Empire service health checks — TCP port + /health endpoint.

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

_TIMEOUT = 5.0


async def _check_single_service(
    name: str,
    port: int,
    host: str = "127.0.0.1",
) -> HealthCheck:
    """Check a single service via TCP connect + optional /health HTTP GET."""
    check = HealthCheck(
        name=f"service:{name}",
        tier=HeartbeatTier.PULSE,
        result=CheckResult.UNKNOWN,
    )
    start = time.monotonic()
    details: dict[str, Any] = {"port": port, "host": host}

    # TCP connect check
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=_TIMEOUT,
        )
        writer.close()
        await writer.wait_closed()
        details["tcp"] = "connected"
    except (asyncio.TimeoutError, OSError, ConnectionRefusedError) as e:
        check.result = CheckResult.DOWN
        check.message = f"Port {port} unreachable: {type(e).__name__}"
        check.duration_ms = (time.monotonic() - start) * 1000
        check.details = details
        return check

    # HTTP health check — try / first, then /health
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"http://{host}:{port}/")
            if resp.status_code == 200:
                check.result = CheckResult.HEALTHY
                check.message = f"OK (port {port})"
                details["health_status"] = 200
                details["health_path"] = "/"
            else:
                # Root didn't work, try /health
                resp = await client.get(f"http://{host}:{port}/health")
                details["health_status"] = resp.status_code
                details["health_path"] = "/health"
                if resp.status_code == 200:
                    check.result = CheckResult.HEALTHY
                    check.message = f"OK (port {port})"
                else:
                    check.result = CheckResult.DEGRADED
                    check.message = f"Health endpoint returned {resp.status_code}"
    except Exception:
        # Service is up (TCP connected) but HTTP failed — still healthy
        check.result = CheckResult.HEALTHY
        check.message = f"TCP OK (port {port}, no health endpoint)"

    check.duration_ms = (time.monotonic() - start) * 1000
    check.details = details
    return check


async def check_all_services(
    service_ports: dict[str, int],
    host: str = "127.0.0.1",
    service_hosts: dict[str, str] | None = None,
) -> list[HealthCheck]:
    """Check all empire services concurrently.

    Args:
        service_ports: Mapping of service name to port number.
        host: Default host address (default localhost).
        service_hosts: Per-service host overrides (service_name → host).

    Returns:
        List of HealthCheck results, one per service.
    """
    if not service_ports:
        return []

    hosts = service_hosts or {}
    tasks = [
        _check_single_service(name, port, hosts.get(name, host))
        for name, port in service_ports.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    checks = []
    for i, (name, port) in enumerate(service_ports.items()):
        result = results[i]
        if isinstance(result, Exception):
            checks.append(HealthCheck(
                name=f"service:{name}",
                tier=HeartbeatTier.PULSE,
                result=CheckResult.DOWN,
                message=f"Check failed: {str(result)[:80]}",
            ))
        else:
            checks.append(result)

    return checks
