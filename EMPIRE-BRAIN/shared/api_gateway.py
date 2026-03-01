"""
Empire API Gateway — Lightweight reverse proxy for all FastAPI services.

Single entrypoint on port 8888 that routes to all empire services:
    /api/brain/*     → localhost:8200
    /api/grimoire/*  → localhost:8080
    /api/videoforge/* → localhost:8090
    /api/bmc/*       → localhost:8095
    /api/dashboard/* → localhost:8000
    /api/vision/*    → localhost:8002
    /api/gateway/*   → self (health, service registry, aggregate stats)

Features:
- Unified health check (all services at once)
- Request logging with timing
- Service registry with auto-discovery
- Aggregate stats endpoint

Start:
    cd EMPIRE-BRAIN
    PYTHONPATH=. python -m uvicorn shared.api_gateway:app --port 8888
"""
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [gateway] %(levelname)s: %(message)s")
log = logging.getLogger("gateway")

app = FastAPI(title="Empire API Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Service Registry ---

SERVICES = {
    "brain":      {"port": 8200, "health": "/health", "description": "EMPIRE-BRAIN MCP Server"},
    "grimoire":   {"port": 8080, "health": "/health", "description": "Grimoire Intelligence API"},
    "videoforge": {"port": 8090, "health": "/health", "description": "VideoForge Engine API"},
    "bmc":        {"port": 8095, "health": "/health", "description": "BMC Webhook Handler"},
    "dashboard":  {"port": 8000, "health": "/health", "description": "Empire Dashboard"},
    "vision":     {"port": 8002, "health": "/health", "description": "Vision Analysis Service"},
}

# Request stats
_stats = {
    "total_requests": 0,
    "by_service": {name: 0 for name in SERVICES},
    "errors": 0,
    "started_at": datetime.now().isoformat(),
}


async def _proxy(service_name: str, path: str, request: Request) -> Response:
    """Proxy a request to a backend service."""
    import httpx

    svc = SERVICES.get(service_name)
    if not svc:
        return Response(content=f'{{"error": "Unknown service: {service_name}"}}',
                        status_code=404, media_type="application/json")

    url = f"http://localhost:{svc['port']}{path}"
    start = time.time()
    _stats["total_requests"] += 1
    _stats["by_service"][service_name] = _stats["by_service"].get(service_name, 0) + 1

    try:
        body = await request.body()
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length")}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body if body else None,
                params=dict(request.query_params),
            )

        elapsed = (time.time() - start) * 1000
        log.info(f"{request.method} /api/{service_name}{path} → {resp.status_code} ({elapsed:.0f}ms)")

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except Exception as e:
        _stats["errors"] += 1
        elapsed = (time.time() - start) * 1000
        log.error(f"{request.method} /api/{service_name}{path} → ERROR: {e} ({elapsed:.0f}ms)")
        return Response(
            content=f'{{"error": "Service unavailable: {service_name}", "detail": "{e}"}}',
            status_code=502,
            media_type="application/json",
        )


# --- Service Routes ---

@app.api_route("/api/brain/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_brain(path: str, request: Request):
    return await _proxy("brain", f"/{path}", request)

@app.api_route("/api/grimoire/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_grimoire(path: str, request: Request):
    return await _proxy("grimoire", f"/{path}", request)

@app.api_route("/api/videoforge/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_videoforge(path: str, request: Request):
    return await _proxy("videoforge", f"/{path}", request)

@app.api_route("/api/bmc/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_bmc(path: str, request: Request):
    return await _proxy("bmc", f"/{path}", request)

@app.api_route("/api/dashboard/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_dashboard(path: str, request: Request):
    return await _proxy("dashboard", f"/{path}", request)

@app.api_route("/api/vision/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_vision(path: str, request: Request):
    return await _proxy("vision", f"/{path}", request)


# --- Gateway Endpoints ---

@app.get("/api/gateway/health")
async def gateway_health():
    """Check health of all registered services."""
    import httpx

    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, svc in SERVICES.items():
            url = f"http://localhost:{svc['port']}{svc['health']}"
            try:
                resp = await client.get(url)
                results[name] = {
                    "status": "ok" if resp.status_code == 200 else "degraded",
                    "port": svc["port"],
                    "response_ms": resp.elapsed.total_seconds() * 1000 if hasattr(resp, 'elapsed') else 0,
                }
            except Exception:
                results[name] = {"status": "offline", "port": svc["port"]}

    online = sum(1 for r in results.values() if r["status"] == "ok")
    return {
        "gateway": "ok",
        "services": results,
        "summary": f"{online}/{len(SERVICES)} services online",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/gateway/services")
async def list_services():
    """List all registered services."""
    return {
        "services": {
            name: {
                "port": svc["port"],
                "health_path": svc["health"],
                "description": svc["description"],
                "proxy_prefix": f"/api/{name}/",
            }
            for name, svc in SERVICES.items()
        }
    }


@app.get("/api/gateway/stats")
async def gateway_stats():
    """Get gateway request statistics."""
    return {
        **_stats,
        "uptime_since": _stats["started_at"],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "empire-api-gateway",
        "port": 8888,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
