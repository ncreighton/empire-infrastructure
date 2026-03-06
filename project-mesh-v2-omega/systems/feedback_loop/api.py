"""Feedback Loop API — FastAPI routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/loop", tags=["feedback_loop"])

_loop = None


def _get_loop():
    global _loop
    if _loop is None:
        from .loop import FeedbackLoop
        _loop = FeedbackLoop()
    return _loop


@router.post("/cycle")
async def run_cycle(dry_run: bool = False):
    """Run one full feedback loop cycle."""
    return _get_loop().run_cycle(dry_run)


@router.get("/cycles")
async def list_cycles(limit: int = 20):
    """List cycle history."""
    return {"cycles": _get_loop().get_cycles(limit)}


@router.get("/cycle/{cycle_id}")
async def get_cycle(cycle_id: int):
    """Get full cycle details."""
    result = _get_loop().get_cycle(cycle_id)
    return result or {"error": "Cycle not found"}


@router.get("/compounding")
async def compounding_metrics():
    """Get compounding improvement metrics."""
    return {"metrics": _get_loop().get_compounding_metrics()}


@router.get("/stats")
async def stats():
    return _get_loop().get_stats()
