"""
fastapi-service — Common FastAPI app scaffold used by Grimoire, VideoForge, Dashboard, BMC.
Provides health endpoint, CORS, error handling, and startup patterns.
"""

import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)


def create_app(
    title: str,
    version: str = "1.0.0",
    description: str = "",
    port: int = 8000,
    cors_origins: Optional[list] = None,
):
    """Create a standard FastAPI application with common middleware."""
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    app = FastAPI(title=title, version=version, description=description)

    # CORS
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
        log.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
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
            "timestamp": datetime.now().isoformat()
        }

    return app
