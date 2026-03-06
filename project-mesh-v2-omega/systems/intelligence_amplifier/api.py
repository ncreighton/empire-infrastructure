"""Intelligence Amplifier API — FastAPI routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/intelligence", tags=["intelligence"])

_amp = None


def _get_amp():
    global _amp
    if _amp is None:
        from .amplifier import IntelligenceAmplifier
        _amp = IntelligenceAmplifier()
    return _amp


@router.get("/analyze/{site_slug}")
async def analyze_site(site_slug: str):
    """Analyze article performance for a site."""
    return _get_amp().analyze_site(site_slug)


@router.get("/playbook/{niche}")
async def get_playbook(niche: str):
    """Get the niche playbook."""
    result = _get_amp().get_playbook(niche)
    return result or {"error": "No playbook found for this niche"}


@router.get("/playbooks")
async def all_playbooks():
    """Get all niche playbooks."""
    return {"playbooks": _get_amp().codex.get_all_playbooks()}


@router.get("/decaying")
async def decaying_articles(site: str = None, limit: int = 20):
    """Get articles with high decay rates."""
    return {"decaying": _get_amp().get_decaying(site)}


@router.get("/decaying/{site_slug}")
async def decaying_for_site(site_slug: str):
    """Get decaying articles for a specific site."""
    return {"decaying": _get_amp().get_decaying(site_slug)}


@router.get("/stats")
async def stats():
    """Intelligence system statistics."""
    return _get_amp().get_stats()
