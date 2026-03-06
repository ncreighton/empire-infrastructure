"""Project Launcher API — FastAPI routes."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/launcher", tags=["launcher"])

_launcher = None


def _get_launcher():
    global _launcher
    if _launcher is None:
        from .launcher import ProjectLauncher
        _launcher = ProjectLauncher()
    return _launcher


class ResearchRequest(BaseModel):
    niche: str
    dry_run: bool = False


@router.post("/research")
async def research_niche(req: ResearchRequest):
    """Research a niche for launch viability."""
    return _get_launcher().research_niche(req.niche, req.dry_run)


@router.post("/launch/{proposal_id}")
async def launch_site(proposal_id: int):
    """Launch an approved proposal."""
    return _get_launcher().launch_site(proposal_id)


@router.get("/proposals")
async def list_proposals(status: str = None):
    """List all proposals."""
    return {"proposals": _get_launcher().get_proposals(status)}


@router.get("/proposal/{proposal_id}")
async def get_proposal(proposal_id: int):
    """Get a specific proposal."""
    result = _get_launcher().get_proposal(proposal_id)
    return result or {"error": "Proposal not found"}


@router.get("/stats")
async def stats():
    return _get_launcher().get_stats()
