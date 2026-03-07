"""FastAPI service boilerplate — shared across all Empire microservices.

Eliminates ~150 lines of repeated setup per service (5+ services = 750+ lines saved).

Usage:
    from empire_utils import create_empire_app, to_dict

    app = create_empire_app(
        title="Grimoire Intelligence API",
        description="Witchcraft practice companion",
        version="1.0.0",
        service_name="grimoire",
        port=8080,
        endpoints={
            "POST /consult": "Ask the grimoire anything",
            "GET /energy": "Current magical energy",
        },
    )

    @app.post("/consult")
    def consult(req: ConsultRequest):
        result = engine.consult(req.query)
        return to_dict(result)
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses/enums to JSON-serializable dicts."""
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return str(obj)


def create_empire_app(
    *,
    title: str,
    description: str = "",
    version: str = "1.0.0",
    service_name: str,
    port: int = 8000,
    endpoints: dict[str, str] | None = None,
) -> FastAPI:
    """Create a FastAPI app with standard Empire middleware and endpoints.

    Includes:
    - CORS middleware (allow all origins)
    - Root endpoint listing all routes
    - Health check endpoint at /health
    """
    app = FastAPI(title=title, description=description, version=version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        routes = endpoints or {}
        routes["GET /health"] = "Health check"
        return {
            "service": title,
            "version": version,
            "endpoints": routes,
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": service_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return app
