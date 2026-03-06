"""Self-Healing API — FastAPI routes for healing operations."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/healing", tags=["healing"])

_healer = None


def _get_healer():
    global _healer
    if _healer is None:
        from .healer import SelfHealer
        _healer = SelfHealer()
    return _healer


@router.get("/check")
async def healing_check():
    """Run full healing check and auto-remediate."""
    return _get_healer().run_full_check()


@router.post("/heal")
async def trigger_heal():
    """Force healing pass on all services."""
    healer = _get_healer()
    result = healer._check_and_heal_services()
    return {"action": "heal", "result": result}


@router.get("/history")
async def healing_history(limit: int = 50):
    """Get recent healing events."""
    return {"events": _get_healer().get_history(limit)}


@router.get("/services")
async def service_status():
    """Current service health status."""
    return _get_healer()._check_and_heal_services()


@router.get("/wordpress")
async def wordpress_health():
    """WordPress site health."""
    return _get_healer()._check_wordpress()


@router.get("/traffic/{site_slug}")
async def investigate_traffic(site_slug: str):
    """Investigate traffic drops for a site."""
    return _get_healer().investigate_traffic(site_slug)


@router.get("/api-keys")
async def api_key_status():
    """API key health status."""
    return _get_healer().api_key_manager.get_summary()


@router.get("/stats")
async def healing_stats():
    """Healing system statistics."""
    return _get_healer().get_stats()
