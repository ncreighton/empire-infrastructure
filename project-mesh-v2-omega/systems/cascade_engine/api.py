"""Cascade Engine API — FastAPI routes."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/cascade", tags=["cascade"])

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from .engine import CascadeEngine
        _engine = CascadeEngine()
    return _engine


class CascadeTrigger(BaseModel):
    site_slug: str
    title: str
    template: str = "full"
    subtitle: Optional[str] = None
    dry_run: bool = False


@router.post("/trigger")
async def trigger_cascade(req: CascadeTrigger):
    """Trigger a new content cascade."""
    return _get_engine().trigger(
        req.site_slug, req.title, req.template,
        req.dry_run, req.subtitle
    )


@router.get("/{cascade_id}")
async def get_cascade(cascade_id: int):
    """Get cascade status and steps."""
    result = _get_engine().get_cascade(cascade_id)
    return result or {"error": "Cascade not found"}


@router.post("/{cascade_id}/retry/{step_name}")
async def retry_step(cascade_id: int, step_name: str):
    """Retry a failed step."""
    return _get_engine().retry_step(cascade_id, step_name)


@router.get("/")
async def recent_cascades(limit: int = 20):
    """Get recent cascades."""
    return {"cascades": _get_engine().get_recent(limit)}


@router.get("/templates/list")
async def list_templates():
    """List available cascade templates."""
    return {"templates": _get_engine().get_templates()}


@router.get("/stats/summary")
async def stats():
    return _get_engine().get_stats()
