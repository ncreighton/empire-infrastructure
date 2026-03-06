"""Cross-Pollination API — FastAPI routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/pollination", tags=["pollination"])

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from .pollinator import CrossPollinationEngine
        _engine = CrossPollinationEngine()
    return _engine


@router.get("/overlaps")
async def get_overlaps():
    """Get keyword overlap analysis between sites."""
    return {"overlaps": _get_engine().get_overlaps()}


@router.post("/detect")
async def detect_overlaps():
    """Run overlap detection scan."""
    return _get_engine().detect_overlaps()


@router.get("/clusters")
async def get_clusters():
    """Get niche clusters."""
    return {"clusters": _get_engine().get_clusters()}


@router.get("/suggest/{source}/{target}")
async def suggest_links(source: str, target: str):
    """Suggest cross-site links."""
    return {"suggestions": _get_engine().suggest_links(source, target)}


@router.post("/inject/{promo_id}")
async def inject_link(promo_id: int):
    """Inject a cross-site link."""
    return _get_engine().inject_link(promo_id)


@router.get("/stats")
async def stats():
    return _get_engine().get_stats()
