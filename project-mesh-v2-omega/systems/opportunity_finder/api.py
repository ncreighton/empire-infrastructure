"""Opportunity Finder API — FastAPI routes."""

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/opportunity", tags=["opportunity"])

_finder = None


def _get_finder():
    global _finder
    if _finder is None:
        from .finder import OpportunityFinder
        _finder = OpportunityFinder()
    return _finder


@router.get("/scan")
async def run_scan():
    """Run a full opportunity scan across all sites."""
    return _get_finder().run_daily_scan()


@router.get("/scan/{site_slug}")
async def scan_site(site_slug: str):
    """Scan opportunities for a specific site."""
    return _get_finder().scan_site(site_slug)


@router.get("/queue")
async def get_queue(site: str = None, limit: int = 20):
    """Get the prioritized opportunity queue."""
    return {"opportunities": _get_finder().get_queue(site, limit)}


@router.get("/queue/{site_slug}")
async def get_site_queue(site_slug: str, limit: int = 20):
    """Get opportunity queue for a specific site."""
    return {"opportunities": _get_finder().get_queue(site_slug, limit)}


@router.get("/cross-site")
async def cross_site():
    """Get cross-site keyword opportunities."""
    return {"cross_site": _get_finder().get_cross_site()}


@router.get("/seasonal")
async def seasonal(months: int = 3):
    """Get upcoming seasonal opportunities."""
    return {"seasonal": _get_finder().get_upcoming_seasonal(months)}


@router.get("/stats")
async def stats():
    """Opportunity system statistics."""
    return _get_finder().get_stats()
