"""Economics Engine API — FastAPI routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/economics", tags=["economics"])

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from .economics import EconomicsEngine
        _engine = EconomicsEngine()
    return _engine


@router.get("/empire")
async def empire_pnl():
    """Full empire P&L."""
    return _get_engine().get_empire_pnl()


@router.get("/site/{site_slug}")
async def site_pnl(site_slug: str):
    """Site-level P&L."""
    result = _get_engine().get_site_pnl(site_slug)
    return result or {"error": "No data for this site"}


@router.get("/allocation")
async def allocation():
    """Investment allocation recommendations."""
    return {"recommendations": _get_engine().get_allocation_recommendations()}


@router.get("/top-articles")
async def top_articles(site: str = None):
    """Top articles by ROI."""
    return {"articles": _get_engine().get_top_articles(site)}


@router.get("/costs")
async def cost_reference():
    """API cost reference table."""
    return _get_engine().get_cost_reference()


@router.get("/stats")
async def stats():
    return _get_engine().get_stats()
