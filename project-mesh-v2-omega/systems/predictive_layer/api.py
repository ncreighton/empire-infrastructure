"""Predictive Layer API — FastAPI routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/predictive", tags=["predictive"])

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from .predictor import PredictiveLayer
        _predictor = PredictiveLayer()
    return _predictor


@router.get("/anomalies")
async def detect_anomalies():
    """Check for algorithm update anomalies."""
    result = _get_predictor().detect_algorithm_update()
    return result or {"status": "no_anomalies"}


@router.get("/anomalies/history")
async def anomaly_history(limit: int = 20):
    """Historical anomaly events."""
    return {"anomalies": _get_predictor().get_anomalies(limit)}


@router.get("/decay/{site_slug}")
async def predict_decay(site_slug: str):
    """Predict content decay for a site."""
    return {"predictions": _get_predictor().predict_decay(site_slug)}


@router.get("/decay")
async def all_decay():
    """All decay predictions."""
    return {"predictions": _get_predictor().get_decay()}


@router.get("/forecast/{site_slug}")
async def forecast(site_slug: str):
    """Revenue forecast for a site."""
    return _get_predictor().forecast_revenue(site_slug)


@router.get("/accuracy")
async def forecast_accuracy():
    """Forecast accuracy metrics."""
    return _get_predictor().get_accuracy()


@router.get("/stats")
async def stats():
    return _get_predictor().get_stats()
