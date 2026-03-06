"""Enhancement Enhancer API — FastAPI routes."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/api/enhancer", tags=["enhancer"])

_enhancer = None


def _get_enhancer():
    global _enhancer
    if _enhancer is None:
        from .enhancer import EnhancementEnhancer
        _enhancer = EnhancementEnhancer()
    return _enhancer


class ExperimentCreate(BaseModel):
    name: str
    pipeline: str
    variant_a: str
    variant_b: str
    metric: str = "ctr"


class PropagateRequest(BaseModel):
    target_sites: Optional[List[str]] = None


@router.get("/quality/{pipeline}")
async def quality_score(pipeline: str, site: str = None):
    """Score pipeline quality."""
    return _get_enhancer().score_pipeline_quality(pipeline, site)


@router.get("/quality/{pipeline}/trend")
async def quality_trend(pipeline: str):
    """Quality score trend."""
    return {"trend": _get_enhancer().get_quality_trend(pipeline)}


@router.get("/degradation/{pipeline}")
async def check_degradation(pipeline: str):
    """Check for quality degradation."""
    result = _get_enhancer().detect_degradation(pipeline)
    return result or {"status": "no_degradation"}


@router.post("/experiment")
async def create_experiment(req: ExperimentCreate):
    """Create a new A/B experiment."""
    return _get_enhancer().create_experiment(
        req.name, req.pipeline, req.variant_a, req.variant_b, req.metric
    )


@router.get("/experiments")
async def list_experiments(status: str = None):
    """List all experiments."""
    return {"experiments": _get_enhancer().get_experiments(status)}


@router.get("/experiment/{exp_id}/evaluate")
async def evaluate_experiment(exp_id: int):
    """Evaluate experiment results."""
    return _get_enhancer().evaluate_experiment(exp_id)


@router.post("/propagate/{exp_id}")
async def propagate(exp_id: int, req: PropagateRequest):
    """Propagate winning config."""
    return _get_enhancer().propagate_winner(exp_id, req.target_sites)


@router.get("/stats")
async def stats():
    return _get_enhancer().get_stats()
