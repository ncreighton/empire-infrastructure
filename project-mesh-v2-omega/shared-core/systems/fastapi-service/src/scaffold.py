"""
fastapi-service -- Common FastAPI app scaffold for empire services.
Extracted from grimoire-intelligence/api/app.py and other FastAPI services.

Provides:
- create_app(): standard FastAPI app with CORS, timing, error handling
- add_health_endpoint(): standard /health endpoint
- to_dict(): recursive dataclass/enum serializer
- register_routes(): helper for adding route groups

Used by: Grimoire (8080), VideoForge (8090), BMC Webhook (8095),
Empire Dashboard (8000).

Pattern: All empire services follow the same scaffold:
1. create_app() with title, version, port
2. Add service-specific routes
3. Run with uvicorn on the designated port
"""

import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

log = logging.getLogger(__name__)


def create_app(
    title: str,
    version: str = "1.0.0",
    description: str = "",
    port: int = 8000,
    cors_origins: Optional[List[str]] = None,
    debug: bool = False,
) -> "FastAPI":
    """Create a standard FastAPI application with common middleware.

    Includes:
    - CORS middleware (permissive by default for local services)
    - Request timing middleware (X-Process-Time header)
    - Global exception handler with logging
    - Health endpoint at /health
    - Root endpoint at / with service info

    Args:
        title: Service title (shown in docs)
        version: Service version string
        description: Service description
        port: Port number (informational, used in health response)
        cors_origins: Allowed CORS origins (default: ["*"])
        debug: Enable debug mode with detailed error responses

    Returns:
        Configured FastAPI application instance.
    """
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title=title,
        version=version,
        description=description,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        response.headers["X-Process-Time"] = f"{elapsed:.3f}"
        return response

    # Global error handler
    @app.exception_handler(Exception)
    async def global_error_handler(request: Request, exc: Exception):
        log.error("Unhandled error on %s %s: %s",
                  request.method, request.url.path, exc, exc_info=True)
        detail = str(exc) if debug else "Internal server error"
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": detail},
        )

    # Health endpoint
    startup_time = datetime.now().isoformat()

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "service": title,
            "version": version,
            "port": port,
            "started_at": startup_time,
            "timestamp": datetime.now().isoformat(),
        }

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": title,
            "version": version,
            "status": "running",
        }

    return app


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses, enums, and complex objects to dicts.

    Extracted from Grimoire API -- handles the serialization of
    dataclass-heavy response objects (RitualPlan, QualityScore, etc.)
    that FastAPI's default JSON encoder cannot handle.

    Usage:
        @app.post("/endpoint")
        def my_endpoint():
            result = engine.process()
            return to_dict(result)
    """
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = to_dict(value)
        return result
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return obj


def add_health_endpoint(app: "FastAPI", service_name: str,
                        version: str = "1.0.0",
                        port: int = 8000) -> None:
    """Add a standard health endpoint to an existing app.

    Use this when create_app() was not used (e.g., manual FastAPI setup).
    """
    startup_time = datetime.now().isoformat()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": service_name,
            "version": version,
            "port": port,
            "started_at": startup_time,
            "timestamp": datetime.now().isoformat(),
        }


def run_service(app: "FastAPI", host: str = "0.0.0.0",
                port: int = 8000) -> None:
    """Run a FastAPI service with uvicorn.

    Convenience wrapper for the common startup pattern.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


def filter_dict(data: Dict, query: Optional[str]) -> Dict:
    """Filter a dict of items by name or properties matching a query.

    Extracted from Grimoire API knowledge endpoints.
    Useful for searchable knowledge base endpoints.

    Args:
        data: Dict of keyed items (each value is a dict)
        query: Search string (case-insensitive substring match)

    Returns:
        Filtered dict containing only matching items.
    """
    if not query:
        return data
    q = query.lower()
    result = {}
    for key, val in data.items():
        if q in key.lower():
            result[key] = val
        elif isinstance(val, dict):
            name = val.get("name", "")
            props = val.get("properties", val.get("magical_properties", []))
            if q in name.lower() or any(q in str(p).lower() for p in props):
                result[key] = val
    return result
