"""
Revenue Forecaster — Statistical Revenue Forecasting for OpenClaw Empire

Pure-Python time-series forecasting engine for predicting revenue across
all income streams in Nick Creighton's 16-site WordPress publishing empire.

Forecasting methods (all pure Python, no numpy/scipy):
    SIMPLE_MOVING_AVERAGE       — Trailing window mean
    WEIGHTED_MOVING_AVERAGE     — Linearly weighted trailing mean
    EXPONENTIAL_SMOOTHING       — Single exponential smoothing (SES / Brown)
    DOUBLE_EXPONENTIAL          — Holt's linear trend method
    TRIPLE_EXPONENTIAL          — Holt-Winters additive seasonal method
    LINEAR_REGRESSION           — Ordinary least-squares trend projection
    ENSEMBLE                    — Accuracy-weighted blend of all methods

Features:
    - Ingest data from RevenueTracker or manual entry
    - Configurable forecast horizon and confidence levels
    - Automatic seasonality detection via autocorrelation
    - Trend direction and strength analysis
    - Confidence interval fans that widen over the forecast horizon
    - Method accuracy tracking (MAE, MAPE, RMSE, R-squared)
    - Per-stream, per-site, and empire-wide forecasts
    - Revenue outlook reports with trend narratives

All data persisted to: data/revenue_forecaster/

Usage:
    from src.revenue_forecaster import get_forecaster, ForecastMethod

    forecaster = get_forecaster()

    # Forecast a single stream
    fc = await forecaster.forecast(
        stream=RevenueStream.ADSENSE,
        site_id="witchcraft",
        method=ForecastMethod.ENSEMBLE,
        horizon_days=30,
    )
    print(f"30-day projection: ${sum(p['amount'] for p in fc.predictions):,.2f}")

    # Detect seasonality
    profile = forecaster.detect_seasonality(data, Granularity.DAILY)
    print(f"Season pattern: {profile.pattern}, period: {profile.period_days}d")

    # Generate revenue outlook
    outlook = await forecaster.generate_outlook(days=30)
    print(outlook)

CLI:
    python -m src.revenue_forecaster forecast --stream adsense --site witchcraft --horizon 30
    python -m src.revenue_forecaster all-streams --horizon 30
    python -m src.revenue_forecaster seasonality --stream adsense --site witchcraft
    python -m src.revenue_forecaster accuracy --stream adsense
    python -m src.revenue_forecaster history --stream adsense --site witchcraft --days 90
    python -m src.revenue_forecaster sync
    python -m src.revenue_forecaster trend --stream adsense --site witchcraft
    python -m src.revenue_forecaster outlook --days 30
    python -m src.revenue_forecaster stats
    python -m src.revenue_forecaster add --date 2026-02-14 --stream adsense --amount 42.50 --site witchcraft
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("revenue_forecaster")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "revenue_forecaster"

FORECASTS_FILE = DATA_DIR / "forecasts.json"
DATA_POINTS_FILE = DATA_DIR / "data_points.json"
SEASONALITY_FILE = DATA_DIR / "seasonality.json"
ACCURACY_FILE = DATA_DIR / "accuracy.json"
CONFIG_FILE = DATA_DIR / "config.json"
STATS_FILE = DATA_DIR / "stats.json"

# Ensure data directory exists on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FORECASTS = 500
MAX_DATA_POINTS = 5000
MAX_ACCURACY_RECORDS = 200
DEFAULT_HORIZON_DAYS = 30
DEFAULT_SMA_WINDOW = 7
DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.1
DEFAULT_GAMMA = 0.1
DEFAULT_SEASON_PERIOD = 7
MIN_POINTS_FOR_FORECAST = 7
MIN_POINTS_FOR_SEASONALITY = 14
MIN_POINTS_FOR_REGRESSION = 3

# Confidence multipliers (z-scores)
CONFIDENCE_MULTIPLIERS: Dict[str, float] = {
    "low": 0.6745,       # 50% confidence
    "medium": 1.2816,    # 80% confidence
    "high": 1.9600,      # 95% confidence
}

ALL_SITE_IDS = [
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
]


# ---------------------------------------------------------------------------
# JSON helpers (atomic writes)
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _today_iso() -> str:
    return _now_utc().strftime("%Y-%m-%d")


def _parse_date(d: str) -> date:
    """Parse YYYY-MM-DD string to date object."""
    return date.fromisoformat(d)


def _date_range(start: str, end: str) -> list[str]:
    """Return list of ISO date strings from *start* to *end* inclusive."""
    s = _parse_date(start)
    e = _parse_date(end)
    days: list[str] = []
    current = s
    while current <= e:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _round_amount(amount: float) -> float:
    return round(float(amount), 2)


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ===================================================================
# Enums
# ===================================================================


class ForecastMethod(str, Enum):
    """Available forecasting algorithms."""
    SIMPLE_MOVING_AVERAGE = "simple_moving_average"
    WEIGHTED_MOVING_AVERAGE = "weighted_moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    DOUBLE_EXPONENTIAL = "double_exponential"
    TRIPLE_EXPONENTIAL = "triple_exponential"
    LINEAR_REGRESSION = "linear_regression"
    ENSEMBLE = "ensemble"


class Granularity(str, Enum):
    """Time granularity for aggregation and forecasting."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class SeasonPattern(str, Enum):
    """Detected seasonality patterns."""
    NONE = "none"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class RevenueStream(str, Enum):
    """Revenue streams tracked across the empire."""
    ADSENSE = "adsense"
    AFFILIATE = "affiliate"
    KDP = "kdp"
    ETSY = "etsy"
    SUBSTACK = "substack"
    SPONSORSHIP = "sponsorship"
    SERVICES = "services"
    TOTAL = "total"

    @classmethod
    def from_string(cls, value: str) -> RevenueStream:
        """Parse a stream from a loose string (case-insensitive)."""
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        # Map alternative names
        aliases: Dict[str, str] = {
            "ads": "adsense",
            "ad": "adsense",
            "google_adsense": "adsense",
            "amazon": "affiliate",
            "aff": "affiliate",
            "kindle": "kdp",
            "books": "kdp",
            "print_on_demand": "etsy",
            "pod": "etsy",
            "newsletter": "substack",
            "sponsor": "sponsorship",
            "sponsored": "sponsorship",
            "service": "services",
            "consulting": "services",
            "all": "total",
        }
        normalized = aliases.get(normalized, normalized)
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown revenue stream: {value!r}")


class ConfidenceLevel(str, Enum):
    """Confidence level for prediction intervals."""
    LOW = "low"          # 50%
    MEDIUM = "medium"    # 80%
    HIGH = "high"        # 95%

    @property
    def percentage(self) -> int:
        mapping = {"low": 50, "medium": 80, "high": 95}
        return mapping[self.value]


# ===================================================================
# Dataclasses
# ===================================================================


@dataclass
class RevenueDataPoint:
    """A single historical revenue observation."""
    date: str                              # ISO YYYY-MM-DD
    stream: str                            # RevenueStream value
    amount: float
    site_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.amount = _round_amount(self.amount)
        if isinstance(self.stream, RevenueStream):
            self.stream = self.stream.value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RevenueDataPoint:
        data = dict(data)
        if "stream" in data and isinstance(data["stream"], RevenueStream):
            data["stream"] = data["stream"].value
        return cls(**{k: v for k, v in data.items()
                      if k in ("date", "stream", "amount", "site_id", "metadata")})


@dataclass
class Forecast:
    """A complete forecast result with predictions and metadata."""
    forecast_id: str = ""
    stream: str = ""
    site_id: str = ""
    method: str = ""
    granularity: str = "daily"
    horizon_days: int = 30
    predictions: list = field(default_factory=list)   # [{date, amount, lower, upper}]
    confidence_level: str = "medium"
    accuracy_metrics: dict = field(default_factory=dict)  # {MAE, MAPE, RMSE}
    seasonality_detected: str = "none"
    trend_direction: str = "flat"                      # up, down, flat
    trend_strength: float = 0.0                        # 0.0 to 1.0
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.forecast_id:
            self.forecast_id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = _now_iso()

    @property
    def total_predicted(self) -> float:
        return _round_amount(sum(p.get("amount", 0.0) for p in self.predictions))

    @property
    def average_daily(self) -> float:
        if not self.predictions:
            return 0.0
        return _round_amount(self.total_predicted / len(self.predictions))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_predicted"] = self.total_predicted
        d["average_daily"] = self.average_daily
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Forecast:
        safe_keys = {
            "forecast_id", "stream", "site_id", "method", "granularity",
            "horizon_days", "predictions", "confidence_level", "accuracy_metrics",
            "seasonality_detected", "trend_direction", "trend_strength", "created_at",
        }
        return cls(**{k: v for k, v in data.items() if k in safe_keys})


@dataclass
class SeasonalityProfile:
    """Detected seasonality characteristics for a revenue stream."""
    stream: str = ""
    pattern: str = "none"                  # SeasonPattern value
    period_days: int = 0
    seasonal_indices: list = field(default_factory=list)  # multipliers per sub-period
    peak_periods: list = field(default_factory=list)      # indices of peak sub-periods
    trough_periods: list = field(default_factory=list)    # indices of trough sub-periods
    strength: float = 0.0                  # 0.0 to 1.0
    analyzed_at: str = ""

    def __post_init__(self) -> None:
        if not self.analyzed_at:
            self.analyzed_at = _now_iso()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SeasonalityProfile:
        safe_keys = {
            "stream", "pattern", "period_days", "seasonal_indices",
            "peak_periods", "trough_periods", "strength", "analyzed_at",
        }
        return cls(**{k: v for k, v in data.items() if k in safe_keys})


@dataclass
class ForecastAccuracy:
    """Accuracy record for a specific forecast after actuals are known."""
    forecast_id: str = ""
    mae: float = 0.0                       # Mean Absolute Error
    mape: float = 0.0                      # Mean Absolute Percentage Error (%)
    rmse: float = 0.0                      # Root Mean Squared Error
    r_squared: float = 0.0                 # Coefficient of determination
    predictions_count: int = 0
    actual_count: int = 0
    best_method: str = ""
    evaluated_at: str = ""

    def __post_init__(self) -> None:
        if not self.evaluated_at:
            self.evaluated_at = _now_iso()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ForecastAccuracy:
        safe_keys = {
            "forecast_id", "mae", "mape", "rmse", "r_squared",
            "predictions_count", "actual_count", "best_method", "evaluated_at",
        }
        return cls(**{k: v for k, v in data.items() if k in safe_keys})


# ===================================================================
# Math Utilities (Pure Python)
# ===================================================================


def _mean(values: list[float]) -> float:
    """Arithmetic mean of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float], mean_val: Optional[float] = None) -> float:
    """Population variance of a list of values."""
    if len(values) < 2:
        return 0.0
    if mean_val is None:
        mean_val = _mean(values)
    return sum((x - mean_val) ** 2 for x in values) / len(values)


def _std_dev(values: list[float], mean_val: Optional[float] = None) -> float:
    """Population standard deviation of a list of values."""
    return math.sqrt(_variance(values, mean_val))


def _linear_regression(x: list[float], y: list[float]) -> Tuple[float, float]:
    """Compute slope and intercept of OLS linear regression.

    Args:
        x: Independent variable values (e.g. time indices 0, 1, 2, ...).
        y: Dependent variable values (e.g. daily revenue).

    Returns:
        (slope, intercept)
    """
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0, _mean(y) if y else 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Numerator: sum of (xi - mean_x)(yi - mean_y)
    ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    # Denominator: sum of (xi - mean_x)^2
    ss_xx = sum((xi - mean_x) ** 2 for xi in x)

    if ss_xx == 0:
        return 0.0, mean_y

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _autocorrelation(data: list[float], lag: int) -> float:
    """Compute autocorrelation of *data* at a given *lag*.

    Returns a value between -1 and 1.  Returns 0.0 if there is
    insufficient data or zero variance.
    """
    n = len(data)
    if lag <= 0 or lag >= n or n < lag + 2:
        return 0.0

    mean_val = _mean(data)
    var_val = _variance(data, mean_val)
    if var_val == 0:
        return 0.0

    # Compute autocovariance at the given lag
    cov = sum(
        (data[i] - mean_val) * (data[i + lag] - mean_val)
        for i in range(n - lag)
    ) / (n - lag)

    return cov / var_val


def _calculate_mae(predicted: list[float], actual: list[float]) -> float:
    """Mean Absolute Error between predicted and actual values."""
    n = min(len(predicted), len(actual))
    if n == 0:
        return 0.0
    return sum(abs(p - a) for p, a in zip(predicted[:n], actual[:n])) / n


def _calculate_mape(predicted: list[float], actual: list[float]) -> float:
    """Mean Absolute Percentage Error (%) between predicted and actual values.

    Skips zero actuals to avoid division by zero.
    """
    pairs = [(p, a) for p, a in zip(predicted, actual) if a != 0.0]
    if not pairs:
        return 0.0
    return (sum(abs((a - p) / a) for p, a in pairs) / len(pairs)) * 100.0


def _calculate_rmse(predicted: list[float], actual: list[float]) -> float:
    """Root Mean Squared Error between predicted and actual values."""
    n = min(len(predicted), len(actual))
    if n == 0:
        return 0.0
    mse = sum((p - a) ** 2 for p, a in zip(predicted[:n], actual[:n])) / n
    return math.sqrt(mse)


def _calculate_r_squared(predicted: list[float], actual: list[float]) -> float:
    """Coefficient of determination (R-squared) between predicted and actual.

    Returns a value where 1.0 is a perfect fit, 0.0 is baseline mean-model,
    and negative means worse than predicting the mean.
    """
    n = min(len(predicted), len(actual))
    if n < 2:
        return 0.0

    pred = predicted[:n]
    act = actual[:n]
    mean_actual = _mean(act)

    ss_res = sum((a - p) ** 2 for p, a in zip(pred, act))
    ss_tot = sum((a - mean_actual) ** 2 for a in act)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1.0 - (ss_res / ss_tot)


def _residuals(predicted: list[float], actual: list[float]) -> list[float]:
    """Compute residuals (actual - predicted)."""
    return [a - p for p, a in zip(predicted, actual)]


def _date_offset(base_date: str, days: int) -> str:
    """Return ISO date string *days* after *base_date*."""
    d = _parse_date(base_date) + timedelta(days=days)
    return d.isoformat()


# ===================================================================
# RevenueForecaster — Main Class (Singleton)
# ===================================================================


class RevenueForecaster:
    """
    Statistical revenue forecasting engine for the empire.

    Provides multiple forecasting methods (SMA, WMA, SES, DES, TES,
    linear regression, ensemble), seasonality detection via autocorrelation,
    trend analysis, and accuracy tracking.

    All methods are pure Python with no external dependencies beyond the
    standard library.
    """

    def __init__(self) -> None:
        self._data_points: list[dict] = self._load_data_points()
        self._forecasts: list[dict] = self._load_forecasts()
        self._seasonality: dict[str, dict] = self._load_seasonality()
        self._accuracy: list[dict] = self._load_accuracy()
        self._config: dict = self._load_config()
        self._stats: dict = self._load_stats()
        logger.info("RevenueForecaster initialized — data dir: %s", DATA_DIR)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_data_points(self) -> list[dict]:
        raw = _load_json(DATA_POINTS_FILE, [])
        if isinstance(raw, list):
            return raw[-MAX_DATA_POINTS:]
        return []

    def _save_data_points(self) -> None:
        self._data_points = self._data_points[-MAX_DATA_POINTS:]
        _save_json(DATA_POINTS_FILE, self._data_points)

    def _load_forecasts(self) -> list[dict]:
        raw = _load_json(FORECASTS_FILE, [])
        if isinstance(raw, list):
            return raw[-MAX_FORECASTS:]
        return []

    def _save_forecasts(self) -> None:
        self._forecasts = self._forecasts[-MAX_FORECASTS:]
        _save_json(FORECASTS_FILE, self._forecasts)

    def _load_seasonality(self) -> dict[str, dict]:
        raw = _load_json(SEASONALITY_FILE, {})
        return raw if isinstance(raw, dict) else {}

    def _save_seasonality(self) -> None:
        _save_json(SEASONALITY_FILE, self._seasonality)

    def _load_accuracy(self) -> list[dict]:
        raw = _load_json(ACCURACY_FILE, [])
        if isinstance(raw, list):
            return raw[-MAX_ACCURACY_RECORDS:]
        return []

    def _save_accuracy(self) -> None:
        self._accuracy = self._accuracy[-MAX_ACCURACY_RECORDS:]
        _save_json(ACCURACY_FILE, self._accuracy)

    def _load_config(self) -> dict:
        defaults = {
            "default_horizon_days": DEFAULT_HORIZON_DAYS,
            "default_method": ForecastMethod.ENSEMBLE.value,
            "default_confidence": ConfidenceLevel.MEDIUM.value,
            "sma_window": DEFAULT_SMA_WINDOW,
            "alpha": DEFAULT_ALPHA,
            "beta": DEFAULT_BETA,
            "gamma": DEFAULT_GAMMA,
            "season_period": DEFAULT_SEASON_PERIOD,
            "ensemble_methods": [
                ForecastMethod.SIMPLE_MOVING_AVERAGE.value,
                ForecastMethod.EXPONENTIAL_SMOOTHING.value,
                ForecastMethod.DOUBLE_EXPONENTIAL.value,
                ForecastMethod.LINEAR_REGRESSION.value,
            ],
        }
        config = _load_json(CONFIG_FILE, defaults.copy())
        for k, v in defaults.items():
            if k not in config:
                config[k] = v
        return config

    def _save_config(self) -> None:
        _save_json(CONFIG_FILE, self._config)

    def _load_stats(self) -> dict:
        defaults = {
            "total_forecasts": 0,
            "total_data_points": 0,
            "evaluations_run": 0,
            "best_overall_method": "",
            "last_forecast_at": "",
            "last_sync_at": "",
        }
        raw = _load_json(STATS_FILE, defaults.copy())
        for k, v in defaults.items():
            if k not in raw:
                raw[k] = v
        return raw

    def _save_stats(self) -> None:
        _save_json(STATS_FILE, self._stats)

    def _bump_stat(self, key: str, amount: int = 1) -> None:
        self._stats[key] = self._stats.get(key, 0) + amount
        self._save_stats()

    # ==================================================================
    # DATA INGESTION
    # ==================================================================

    def add_data_point(
        self,
        date_str: str,
        stream: str | RevenueStream,
        amount: float,
        site_id: str = "",
        metadata: Optional[dict] = None,
    ) -> RevenueDataPoint:
        """Add a single revenue data point to the historical dataset.

        Args:
            date_str: ISO YYYY-MM-DD date.
            stream: Revenue stream identifier.
            amount: Revenue amount in USD.
            site_id: Optional site identifier.
            metadata: Optional extra metadata dict.

        Returns:
            The created RevenueDataPoint.
        """
        if isinstance(stream, RevenueStream):
            stream_val = stream.value
        else:
            stream_val = RevenueStream.from_string(stream).value

        dp = RevenueDataPoint(
            date=date_str,
            stream=stream_val,
            amount=amount,
            site_id=site_id or "",
            metadata=metadata or {},
        )

        self._data_points.append(dp.to_dict())
        self._save_data_points()
        self._bump_stat("total_data_points")

        logger.info(
            "Added data point: %s %s $%.2f (%s) on %s",
            stream_val, site_id or "all", amount, date_str,
            "with metadata" if metadata else "no metadata",
        )
        return dp

    def add_data_points(self, points: list[dict]) -> int:
        """Bulk-add multiple data points.

        Each dict should have keys: date, stream, amount, and optionally
        site_id and metadata.

        Returns:
            Count of successfully added points.
        """
        added = 0
        for raw in points:
            try:
                self.add_data_point(
                    date_str=raw["date"],
                    stream=raw["stream"],
                    amount=float(raw["amount"]),
                    site_id=raw.get("site_id", ""),
                    metadata=raw.get("metadata"),
                )
                added += 1
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping malformed data point: %s — %s", raw, exc)

        logger.info("Bulk-added %d/%d data points", added, len(points))
        return added

    async def sync_from_revenue_tracker(self) -> int:
        """Synchronize historical data from the RevenueTracker module.

        Reads daily revenue files from data/revenue/daily/ and converts
        them into forecaster data points.

        Returns:
            Count of data points synced.
        """
        revenue_dir = BASE_DIR / "data" / "revenue" / "daily"
        if not revenue_dir.exists():
            logger.warning("Revenue daily dir not found: %s", revenue_dir)
            return 0

        synced = 0
        existing_dates: set[str] = set()
        for dp in self._data_points:
            key = f"{dp.get('date', '')}|{dp.get('stream', '')}|{dp.get('site_id', '')}"
            existing_dates.add(key)

        daily_files = sorted(revenue_dir.glob("*.json"))
        for daily_file in daily_files:
            try:
                raw = _load_json(daily_file, None)
                if raw is None:
                    continue

                file_date = raw.get("date", daily_file.stem)
                entries = raw.get("entries", [])

                for entry in entries:
                    stream_raw = entry.get("stream", "")
                    site_id = entry.get("site_id", "")
                    amount = float(entry.get("amount", 0.0))

                    if amount <= 0:
                        continue

                    # Map revenue_tracker stream names to forecaster streams
                    stream_mapped = self._map_tracker_stream(stream_raw)
                    if not stream_mapped:
                        continue

                    key = f"{file_date}|{stream_mapped}|{site_id or ''}"
                    if key in existing_dates:
                        continue

                    dp = RevenueDataPoint(
                        date=file_date,
                        stream=stream_mapped,
                        amount=amount,
                        site_id=site_id or "",
                        metadata=entry.get("metadata", {}),
                    )
                    self._data_points.append(dp.to_dict())
                    existing_dates.add(key)
                    synced += 1

            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Skipping revenue file %s: %s", daily_file.name, exc)

        if synced > 0:
            self._save_data_points()
            self._bump_stat("total_data_points", synced)
            self._stats["last_sync_at"] = _now_iso()
            self._save_stats()

        logger.info("Synced %d data points from RevenueTracker", synced)
        return synced

    def _map_tracker_stream(self, stream_name: str) -> str:
        """Map revenue_tracker stream names to forecaster RevenueStream values."""
        mapping: Dict[str, str] = {
            "ads": RevenueStream.ADSENSE.value,
            "adsense": RevenueStream.ADSENSE.value,
            "affiliate": RevenueStream.AFFILIATE.value,
            "kdp": RevenueStream.KDP.value,
            "etsy": RevenueStream.ETSY.value,
            "substack": RevenueStream.SUBSTACK.value,
            "youtube": RevenueStream.ADSENSE.value,  # Group with ads
            "sponsored": RevenueStream.SPONSORSHIP.value,
            "digital_products": RevenueStream.SERVICES.value,
        }
        normalized = stream_name.strip().lower()
        return mapping.get(normalized, "")

    # ------------------------------------------------------------------
    # Data retrieval and aggregation
    # ------------------------------------------------------------------

    def get_history(
        self,
        stream: Optional[str] = None,
        site_id: Optional[str] = None,
        days: int = 90,
        granularity: Granularity = Granularity.DAILY,
    ) -> list[RevenueDataPoint]:
        """Retrieve historical data points, optionally filtered and aggregated.

        Args:
            stream: Filter to a specific RevenueStream value.
            site_id: Filter to a specific site.
            days: Number of trailing days to include.
            granularity: Time aggregation level.

        Returns:
            Sorted list of RevenueDataPoint objects.
        """
        cutoff = (_now_utc().date() - timedelta(days=days)).isoformat()

        filtered: list[dict] = []
        for dp in self._data_points:
            if dp.get("date", "") < cutoff:
                continue
            if stream and dp.get("stream", "") != stream:
                continue
            if site_id and dp.get("site_id", "") != site_id:
                continue
            filtered.append(dp)

        if not filtered:
            return []

        # Sort by date
        filtered.sort(key=lambda d: d.get("date", ""))

        if granularity == Granularity.DAILY:
            # Aggregate by date (sum amounts for same date/stream/site)
            aggregated = self._aggregate_by_date(filtered)
        elif granularity == Granularity.WEEKLY:
            aggregated = self._aggregate_by_week(filtered)
        elif granularity == Granularity.MONTHLY:
            aggregated = self._aggregate_by_month(filtered)
        else:
            aggregated = filtered

        return [RevenueDataPoint.from_dict(d) for d in aggregated]

    def _aggregate_by_date(self, points: list[dict]) -> list[dict]:
        """Sum amounts by unique date."""
        by_date: Dict[str, float] = {}
        sample_stream = points[0].get("stream", "") if points else ""
        sample_site = points[0].get("site_id", "") if points else ""
        for dp in points:
            d = dp.get("date", "")
            by_date[d] = by_date.get(d, 0.0) + dp.get("amount", 0.0)
        return [
            {"date": d, "stream": sample_stream, "amount": _round_amount(amt),
             "site_id": sample_site, "metadata": {}}
            for d, amt in sorted(by_date.items())
        ]

    def _aggregate_by_week(self, points: list[dict]) -> list[dict]:
        """Sum amounts by ISO week start (Monday)."""
        by_week: Dict[str, float] = {}
        sample_stream = points[0].get("stream", "") if points else ""
        sample_site = points[0].get("site_id", "") if points else ""
        for dp in points:
            d = _parse_date(dp.get("date", _today_iso()))
            monday = d - timedelta(days=d.weekday())
            week_key = monday.isoformat()
            by_week[week_key] = by_week.get(week_key, 0.0) + dp.get("amount", 0.0)
        return [
            {"date": d, "stream": sample_stream, "amount": _round_amount(amt),
             "site_id": sample_site, "metadata": {}}
            for d, amt in sorted(by_week.items())
        ]

    def _aggregate_by_month(self, points: list[dict]) -> list[dict]:
        """Sum amounts by month (YYYY-MM-01)."""
        by_month: Dict[str, float] = {}
        sample_stream = points[0].get("stream", "") if points else ""
        sample_site = points[0].get("site_id", "") if points else ""
        for dp in points:
            d = dp.get("date", _today_iso())[:7] + "-01"
            by_month[d] = by_month.get(d, 0.0) + dp.get("amount", 0.0)
        return [
            {"date": d, "stream": sample_stream, "amount": _round_amount(amt),
             "site_id": sample_site, "metadata": {}}
            for d, amt in sorted(by_month.items())
        ]

    def aggregate_history(
        self,
        stream: Optional[str] = None,
        site_id: Optional[str] = None,
        days: int = 90,
    ) -> dict:
        """Return summary statistics for historical data.

        Returns:
            Dict with total, mean, median, std_dev, min, max, count, date_range.
        """
        history = self.get_history(stream=stream, site_id=site_id, days=days)
        if not history:
            return {
                "total": 0.0, "mean": 0.0, "median": 0.0, "std_dev": 0.0,
                "min": 0.0, "max": 0.0, "count": 0, "date_range": "",
            }

        amounts = [dp.amount for dp in history]
        amounts_sorted = sorted(amounts)
        n = len(amounts_sorted)
        median = amounts_sorted[n // 2] if n % 2 == 1 else (
            (amounts_sorted[n // 2 - 1] + amounts_sorted[n // 2]) / 2.0
        )

        return {
            "total": _round_amount(sum(amounts)),
            "mean": _round_amount(_mean(amounts)),
            "median": _round_amount(median),
            "std_dev": _round_amount(_std_dev(amounts)),
            "min": _round_amount(min(amounts)),
            "max": _round_amount(max(amounts)),
            "count": n,
            "date_range": f"{history[0].date} to {history[-1].date}",
        }

    # ==================================================================
    # FORECASTING METHODS
    # ==================================================================

    def forecast_sma(
        self,
        data: list[float],
        window: int = 7,
        horizon: int = 30,
    ) -> list[float]:
        """Simple Moving Average forecast.

        Uses the trailing *window*-period average as the forecast for all
        future periods.  When fewer than *window* points exist, uses all
        available data.

        Args:
            data: Historical values (chronological order).
            window: Number of trailing periods to average.
            horizon: Number of future periods to predict.

        Returns:
            List of *horizon* predicted values.
        """
        if not data:
            return [0.0] * horizon

        effective_window = min(window, len(data))
        trailing = data[-effective_window:]
        avg = _mean(trailing)

        return [_round_amount(avg)] * horizon

    def forecast_wma(
        self,
        data: list[float],
        weights: Optional[list[float]] = None,
        horizon: int = 30,
    ) -> list[float]:
        """Weighted Moving Average forecast.

        More recent observations receive higher weights.  If *weights* is
        not provided, uses linearly increasing weights over the last
        min(7, len(data)) periods.

        Args:
            data: Historical values (chronological order).
            weights: Custom weight vector (same length as window).
            horizon: Number of future periods to predict.

        Returns:
            List of *horizon* predicted values.
        """
        if not data:
            return [0.0] * horizon

        if weights is None:
            window = min(7, len(data))
            weights = [float(i + 1) for i in range(window)]
        else:
            window = len(weights)

        effective_window = min(window, len(data))
        if effective_window < len(weights):
            weights = weights[-effective_window:]

        trailing = data[-effective_window:]
        total_weight = sum(weights)

        if total_weight == 0:
            return [_round_amount(_mean(trailing))] * horizon

        weighted_sum = sum(w * v for w, v in zip(weights, trailing))
        avg = weighted_sum / total_weight

        return [_round_amount(avg)] * horizon

    def forecast_ses(
        self,
        data: list[float],
        alpha: float = 0.3,
        horizon: int = 30,
    ) -> list[float]:
        """Single Exponential Smoothing (SES / Brown's method).

        Produces a level forecast — all future periods receive the same
        smoothed value.  Good for data with no clear trend or seasonality.

        Args:
            data: Historical values.
            alpha: Smoothing factor (0 < alpha <= 1). Higher = more reactive.
            horizon: Number of future periods to predict.

        Returns:
            List of *horizon* predicted values.
        """
        if not data:
            return [0.0] * horizon

        alpha = max(0.01, min(1.0, alpha))

        # Initialize level to first observation
        level = data[0]

        # Update level through all observations
        for value in data[1:]:
            level = alpha * value + (1 - alpha) * level

        return [_round_amount(level)] * horizon

    def forecast_des(
        self,
        data: list[float],
        alpha: float = 0.3,
        beta: float = 0.1,
        horizon: int = 30,
    ) -> list[float]:
        """Double Exponential Smoothing (Holt's linear trend method).

        Captures level and trend.  Each future step adds the estimated
        trend increment, producing a linearly trending forecast.

        Args:
            data: Historical values.
            alpha: Level smoothing factor (0 < alpha <= 1).
            beta: Trend smoothing factor (0 < beta <= 1).
            horizon: Number of future periods to predict.

        Returns:
            List of *horizon* predicted values.
        """
        if not data:
            return [0.0] * horizon
        if len(data) == 1:
            return [_round_amount(data[0])] * horizon

        alpha = max(0.01, min(1.0, alpha))
        beta = max(0.01, min(1.0, beta))

        # Initialize level and trend
        level = data[0]
        trend = data[1] - data[0]

        for value in data[1:]:
            prev_level = level
            level = alpha * value + (1 - alpha) * (prev_level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        predictions: list[float] = []
        for h in range(1, horizon + 1):
            predicted = level + h * trend
            # Floor at zero — revenue cannot be negative
            predictions.append(_round_amount(max(0.0, predicted)))

        return predictions

    def forecast_tes(
        self,
        data: list[float],
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.1,
        period: int = 7,
        horizon: int = 30,
    ) -> list[float]:
        """Triple Exponential Smoothing (Holt-Winters additive method).

        Captures level, trend, and additive seasonal components.  The
        *period* parameter should match the detected seasonal cycle
        (e.g. 7 for weekly, 30 for monthly).

        Requires at least 2 full seasons of data.

        Args:
            data: Historical values.
            alpha: Level smoothing factor (0 < alpha <= 1).
            beta: Trend smoothing factor (0 < beta <= 1).
            gamma: Seasonal smoothing factor (0 < gamma <= 1).
            period: Seasonal period in data points.
            horizon: Number of future periods to predict.

        Returns:
            List of *horizon* predicted values.
        """
        if not data:
            return [0.0] * horizon

        n = len(data)
        period = max(2, period)

        # Fall back to DES if insufficient data for seasonal decomposition
        if n < 2 * period:
            logger.debug(
                "TES: insufficient data (%d pts, need %d). Falling back to DES.",
                n, 2 * period,
            )
            return self.forecast_des(data, alpha, beta, horizon)

        alpha = max(0.01, min(1.0, alpha))
        beta = max(0.01, min(1.0, beta))
        gamma = max(0.01, min(1.0, gamma))

        # --- Initialize seasonal indices from first two complete cycles ---
        num_complete = n // period
        # Average value per season cycle
        season_avgs: list[float] = []
        for c in range(min(num_complete, 2)):
            start_idx = c * period
            end_idx = start_idx + period
            cycle_data = data[start_idx:end_idx]
            avg = _mean(cycle_data)
            season_avgs.append(avg if avg != 0 else 1.0)

        # Compute initial seasonal indices as deviations from cycle averages
        seasonal: list[float] = [0.0] * period
        if len(season_avgs) >= 2:
            for i in range(period):
                vals: list[float] = []
                for c in range(min(num_complete, 2)):
                    idx = c * period + i
                    if idx < n:
                        vals.append(data[idx] - season_avgs[c])
                seasonal[i] = _mean(vals) if vals else 0.0
        elif len(season_avgs) == 1:
            for i in range(period):
                if i < n:
                    seasonal[i] = data[i] - season_avgs[0]

        # Initialize level and trend
        level = _mean(data[:period])
        trend = 0.0
        if num_complete >= 2:
            sum1 = sum(data[:period])
            sum2 = sum(data[period: 2 * period])
            trend = (sum2 - sum1) / (period * period)

        # --- Smooth through the entire dataset ---
        for t in range(n):
            value = data[t]
            season_idx = t % period
            prev_level = level
            prev_seasonal = seasonal[season_idx]

            level = alpha * (value - prev_seasonal) + (1 - alpha) * (prev_level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasonal[season_idx] = gamma * (value - level) + (1 - gamma) * prev_seasonal

        # --- Generate forecast ---
        predictions: list[float] = []
        for h in range(1, horizon + 1):
            season_idx = (n + h - 1) % period
            predicted = level + h * trend + seasonal[season_idx]
            predictions.append(_round_amount(max(0.0, predicted)))

        return predictions

    def forecast_linear(
        self,
        data: list[float],
        horizon: int = 30,
    ) -> list[float]:
        """Linear regression forecast.

        Fits an OLS line y = slope * x + intercept to the historical data
        and extrapolates forward.

        Args:
            data: Historical values.
            horizon: Number of future periods to predict.

        Returns:
            List of *horizon* predicted values.
        """
        if not data:
            return [0.0] * horizon
        if len(data) < MIN_POINTS_FOR_REGRESSION:
            avg = _mean(data)
            return [_round_amount(avg)] * horizon

        x = [float(i) for i in range(len(data))]
        slope, intercept = _linear_regression(x, data)

        predictions: list[float] = []
        n = len(data)
        for h in range(horizon):
            predicted = slope * (n + h) + intercept
            predictions.append(_round_amount(max(0.0, predicted)))

        return predictions

    def forecast_ensemble(
        self,
        data: list[float],
        horizon: int = 30,
    ) -> list[float]:
        """Ensemble forecast — accuracy-weighted average of multiple methods.

        Runs each configured ensemble method, then blends their forecasts
        using weights derived from historical accuracy.  If no accuracy
        history exists, uses equal weights.

        Args:
            data: Historical values.
            horizon: Number of future periods to predict.

        Returns:
            List of *horizon* predicted values.
        """
        if not data:
            return [0.0] * horizon

        method_configs = self._config.get("ensemble_methods", [
            ForecastMethod.SIMPLE_MOVING_AVERAGE.value,
            ForecastMethod.EXPONENTIAL_SMOOTHING.value,
            ForecastMethod.DOUBLE_EXPONENTIAL.value,
            ForecastMethod.LINEAR_REGRESSION.value,
        ])

        # Generate forecasts from each sub-method
        method_forecasts: Dict[str, list[float]] = {}
        for method_name in method_configs:
            try:
                preds = self._run_method(method_name, data, horizon)
                if preds:
                    method_forecasts[method_name] = preds
            except Exception as exc:
                logger.warning("Ensemble: method %s failed: %s", method_name, exc)

        if not method_forecasts:
            # Last resort: simple average
            return self.forecast_sma(data, horizon=horizon)

        # Determine weights from accuracy history
        weights = self._get_ensemble_weights(method_forecasts.keys())

        # Blend forecasts
        predictions: list[float] = []
        total_weight = sum(weights.get(m, 1.0) for m in method_forecasts)
        if total_weight == 0:
            total_weight = len(method_forecasts)

        for h in range(horizon):
            weighted_sum = 0.0
            for method_name, preds in method_forecasts.items():
                w = weights.get(method_name, 1.0)
                if h < len(preds):
                    weighted_sum += w * preds[h]
            blended = weighted_sum / total_weight
            predictions.append(_round_amount(max(0.0, blended)))

        return predictions

    def _run_method(
        self,
        method_name: str,
        data: list[float],
        horizon: int,
    ) -> list[float]:
        """Dispatch to the appropriate forecast method by name."""
        alpha = self._config.get("alpha", DEFAULT_ALPHA)
        beta = self._config.get("beta", DEFAULT_BETA)
        gamma = self._config.get("gamma", DEFAULT_GAMMA)
        window = self._config.get("sma_window", DEFAULT_SMA_WINDOW)
        period = self._config.get("season_period", DEFAULT_SEASON_PERIOD)

        if method_name == ForecastMethod.SIMPLE_MOVING_AVERAGE.value:
            return self.forecast_sma(data, window=window, horizon=horizon)
        elif method_name == ForecastMethod.WEIGHTED_MOVING_AVERAGE.value:
            return self.forecast_wma(data, horizon=horizon)
        elif method_name == ForecastMethod.EXPONENTIAL_SMOOTHING.value:
            return self.forecast_ses(data, alpha=alpha, horizon=horizon)
        elif method_name == ForecastMethod.DOUBLE_EXPONENTIAL.value:
            return self.forecast_des(data, alpha=alpha, beta=beta, horizon=horizon)
        elif method_name == ForecastMethod.TRIPLE_EXPONENTIAL.value:
            return self.forecast_tes(data, alpha=alpha, beta=beta, gamma=gamma,
                                     period=period, horizon=horizon)
        elif method_name == ForecastMethod.LINEAR_REGRESSION.value:
            return self.forecast_linear(data, horizon=horizon)
        elif method_name == ForecastMethod.ENSEMBLE.value:
            return self.forecast_ensemble(data, horizon=horizon)
        else:
            raise ValueError(f"Unknown forecast method: {method_name}")

    def _get_ensemble_weights(self, method_names: Any) -> Dict[str, float]:
        """Compute accuracy-based weights for ensemble blending.

        Methods with lower MAPE get higher weight.  If no accuracy data
        exists, all methods receive equal weight of 1.0.
        """
        weights: Dict[str, float] = {}

        # Aggregate MAPE scores from accuracy records
        method_mapes: Dict[str, list[float]] = {}
        for acc in self._accuracy:
            method = acc.get("best_method", "")
            mape = acc.get("mape", 0.0)
            if method and mape > 0:
                method_mapes.setdefault(method, []).append(mape)

        if not method_mapes:
            # Equal weights when no accuracy history exists
            for m in method_names:
                weights[m] = 1.0
            return weights

        # Convert MAPE to inverse weight (lower MAPE = higher weight)
        avg_mapes: Dict[str, float] = {}
        for method, mapes in method_mapes.items():
            avg_mapes[method] = _mean(mapes)

        for m in method_names:
            m_str = m if isinstance(m, str) else m.value
            mape = avg_mapes.get(m_str, 0.0)
            if mape > 0:
                weights[m_str] = 1.0 / mape
            else:
                weights[m_str] = 1.0

        return weights

    # ==================================================================
    # CONFIDENCE INTERVALS
    # ==================================================================

    def calculate_confidence_interval(
        self,
        predictions: list[float],
        residuals_list: list[float],
        level: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    ) -> list[dict]:
        """Calculate confidence intervals for predictions with widening uncertainty.

        The uncertainty band widens over the forecast horizon, simulating
        increasing uncertainty as we project further into the future.

        Args:
            predictions: Point forecast values.
            residuals_list: Historical residuals (actual - predicted) from
                backtest, used to estimate forecast standard error.
            level: Confidence level for the interval.

        Returns:
            List of dicts: [{amount, lower, upper}, ...] for each prediction.
        """
        z = CONFIDENCE_MULTIPLIERS.get(level.value, 1.2816)

        # Base standard error from residuals
        if residuals_list:
            base_se = _std_dev(residuals_list)
        else:
            # Fallback: use 10% of mean prediction as base error
            mean_pred = _mean(predictions) if predictions else 0.0
            base_se = max(mean_pred * 0.10, 1.0)

        result: list[dict] = []
        n = len(predictions)

        for i, predicted in enumerate(predictions):
            # Widen uncertainty: sqrt(1 + step/horizon) growth factor
            horizon_factor = math.sqrt(1.0 + (i / max(n, 1)))
            se = base_se * horizon_factor

            margin = z * se
            lower = max(0.0, predicted - margin)
            upper = predicted + margin

            result.append({
                "amount": _round_amount(predicted),
                "lower": _round_amount(lower),
                "upper": _round_amount(upper),
            })

        return result

    # ==================================================================
    # SEASONALITY DETECTION
    # ==================================================================

    def detect_seasonality(
        self,
        data: list[float],
        granularity: Granularity = Granularity.DAILY,
    ) -> SeasonalityProfile:
        """Detect seasonality in time-series data via autocorrelation.

        Tests candidate periods and selects the one with the strongest
        autocorrelation signal.

        Args:
            data: Historical values.
            granularity: Time granularity of the data.

        Returns:
            SeasonalityProfile with detected pattern, period, and indices.
        """
        profile = SeasonalityProfile()

        if len(data) < MIN_POINTS_FOR_SEASONALITY:
            profile.pattern = SeasonPattern.NONE.value
            return profile

        # Define candidate periods based on granularity
        if granularity == Granularity.DAILY:
            candidates = [
                (7, SeasonPattern.WEEKLY),
                (30, SeasonPattern.MONTHLY),
                (91, SeasonPattern.QUARTERLY),
                (365, SeasonPattern.YEARLY),
            ]
        elif granularity == Granularity.WEEKLY:
            candidates = [
                (4, SeasonPattern.MONTHLY),
                (13, SeasonPattern.QUARTERLY),
                (52, SeasonPattern.YEARLY),
            ]
        elif granularity == Granularity.MONTHLY:
            candidates = [
                (3, SeasonPattern.QUARTERLY),
                (12, SeasonPattern.YEARLY),
            ]
        else:
            candidates = [(7, SeasonPattern.WEEKLY)]

        best_acf = 0.0
        best_period = 0
        best_pattern = SeasonPattern.NONE

        for period_len, pattern in candidates:
            if len(data) < 2 * period_len:
                continue

            acf = _autocorrelation(data, period_len)
            if acf > best_acf:
                best_acf = acf
                best_period = period_len
                best_pattern = pattern

        # Significance threshold: autocorrelation must exceed noise level
        significance_threshold = 2.0 / math.sqrt(len(data))

        if best_acf > significance_threshold and best_period > 0:
            profile.pattern = best_pattern.value
            profile.period_days = best_period
            profile.strength = _round_amount(min(1.0, best_acf))

            # Compute seasonal indices
            indices = self.get_seasonal_indices(data, best_period)
            profile.seasonal_indices = [_round_amount(idx) for idx in indices]

            # Find peak and trough periods
            if indices:
                mean_idx = _mean(indices)
                peaks: list[int] = []
                troughs: list[int] = []
                for i, idx in enumerate(indices):
                    if idx > mean_idx * 1.05:
                        peaks.append(i)
                    elif idx < mean_idx * 0.95:
                        troughs.append(i)
                profile.peak_periods = peaks
                profile.trough_periods = troughs
        else:
            profile.pattern = SeasonPattern.NONE.value
            profile.strength = _round_amount(best_acf) if best_acf > 0 else 0.0

        profile.analyzed_at = _now_iso()

        # Cache the profile
        cache_key = f"{granularity.value}_{best_pattern.value}"
        self._seasonality[cache_key] = profile.to_dict()
        self._save_seasonality()

        return profile

    def get_seasonal_indices(self, data: list[float], period: int) -> list[float]:
        """Compute seasonal indices (multiplicative) for each sub-period.

        Returns a list of *period* values.  A value >1.0 indicates
        the sub-period is above-average; <1.0 indicates below-average.

        Args:
            data: Historical values.
            period: Seasonal period (number of sub-periods per cycle).

        Returns:
            List of *period* seasonal index multipliers.
        """
        if not data or period < 2:
            return []

        n = len(data)
        overall_mean = _mean(data)

        if overall_mean == 0:
            return [1.0] * period

        # Collect values for each sub-period position
        sub_period_values: list[list[float]] = [[] for _ in range(period)]
        for i, value in enumerate(data):
            sub_period_values[i % period].append(value)

        # Compute index as ratio of sub-period average to overall average
        indices: list[float] = []
        for bucket in sub_period_values:
            if bucket:
                sub_mean = _mean(bucket)
                idx = sub_mean / overall_mean
            else:
                idx = 1.0
            indices.append(idx)

        # Normalize so indices average to 1.0
        idx_mean = _mean(indices)
        if idx_mean > 0:
            indices = [idx / idx_mean for idx in indices]

        return indices

    # ==================================================================
    # TREND DETECTION
    # ==================================================================

    def detect_trend(self, data: list[float]) -> Tuple[str, float]:
        """Detect the overall trend direction and strength in the data.

        Uses linear regression slope normalized by the data's standard
        deviation to determine direction (up/down/flat) and strength
        (0.0 to 1.0).

        Args:
            data: Historical values.

        Returns:
            (direction, strength) where direction is "up", "down", or "flat"
            and strength is between 0.0 and 1.0.
        """
        if len(data) < MIN_POINTS_FOR_REGRESSION:
            return "flat", 0.0

        x = [float(i) for i in range(len(data))]
        slope, intercept = _linear_regression(x, data)

        # Normalize slope by data magnitude for strength assessment
        data_std = _std_dev(data)
        data_mean = _mean(data)

        if data_mean == 0 and data_std == 0:
            return "flat", 0.0

        # Strength: how much the regression line rises/falls per period
        # relative to the data's variability
        if data_std > 0:
            normalized_slope = abs(slope) / data_std
        else:
            normalized_slope = abs(slope) / max(abs(data_mean), 1.0)

        # Cap strength at 1.0
        strength = min(1.0, normalized_slope)

        # Threshold for "meaningful" trend
        noise_threshold = 0.05  # 5% of std dev

        if data_std > 0:
            relative_slope = abs(slope) / data_std
        else:
            relative_slope = 0.0

        if relative_slope < noise_threshold:
            direction = "flat"
            strength = 0.0
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"

        return direction, _round_amount(strength)

    # ==================================================================
    # HIGH-LEVEL FORECAST API
    # ==================================================================

    async def forecast(
        self,
        stream: str | RevenueStream = RevenueStream.TOTAL,
        site_id: str = "",
        method: str | ForecastMethod = ForecastMethod.ENSEMBLE,
        horizon_days: int = 30,
        confidence: str | ConfidenceLevel = ConfidenceLevel.MEDIUM,
        granularity: Granularity = Granularity.DAILY,
    ) -> Forecast:
        """Generate a complete forecast for a revenue stream.

        This is the primary high-level method.  It retrieves historical
        data, runs the selected forecasting method, computes confidence
        intervals, detects seasonality and trend, and persists the result.

        Args:
            stream: Revenue stream to forecast.
            site_id: Optional site filter.
            method: Forecasting method to use.
            horizon_days: Number of days to forecast ahead.
            confidence: Confidence level for prediction intervals.
            granularity: Time aggregation level.

        Returns:
            A complete Forecast object with predictions and metadata.
        """
        if isinstance(stream, RevenueStream):
            stream_val = stream.value
        else:
            stream_val = RevenueStream.from_string(stream).value

        if isinstance(method, ForecastMethod):
            method_val = method.value
        else:
            method_val = method

        if isinstance(confidence, ConfidenceLevel):
            conf_level = confidence
        else:
            conf_level = ConfidenceLevel(confidence)

        # Retrieve historical data
        lookback_days = max(horizon_days * 3, 90)
        history = self.get_history(
            stream=stream_val if stream_val != RevenueStream.TOTAL.value else None,
            site_id=site_id or None,
            days=lookback_days,
            granularity=granularity,
        )

        data_values = [dp.amount for dp in history]

        if len(data_values) < MIN_POINTS_FOR_FORECAST:
            logger.warning(
                "Insufficient data for forecast: %d points (need %d)",
                len(data_values), MIN_POINTS_FOR_FORECAST,
            )
            # Return empty forecast
            fc = Forecast(
                stream=stream_val,
                site_id=site_id,
                method=method_val,
                granularity=granularity.value,
                horizon_days=horizon_days,
                confidence_level=conf_level.value,
                accuracy_metrics={"MAE": 0, "MAPE": 0, "RMSE": 0},
            )
            self._forecasts.append(fc.to_dict())
            self._save_forecasts()
            return fc

        # Run forecast
        predictions_raw = self._run_method(method_val, data_values, horizon_days)

        # Compute residuals from a backtest for confidence intervals
        backtest_residuals = self._backtest_residuals(data_values, method_val)

        # Build confidence intervals
        predictions_with_ci = self.calculate_confidence_interval(
            predictions_raw, backtest_residuals, conf_level,
        )

        # Add dates to predictions
        last_date = history[-1].date if history else _today_iso()
        for i, pred in enumerate(predictions_with_ci):
            pred["date"] = _date_offset(last_date, i + 1)

        # Detect seasonality
        season_profile = self.detect_seasonality(data_values, granularity)

        # Detect trend
        trend_dir, trend_str = self.detect_trend(data_values)

        # Compute accuracy metrics from backtest
        accuracy = self._backtest_accuracy(data_values, method_val)

        # Build forecast object
        fc = Forecast(
            stream=stream_val,
            site_id=site_id,
            method=method_val,
            granularity=granularity.value,
            horizon_days=horizon_days,
            predictions=predictions_with_ci,
            confidence_level=conf_level.value,
            accuracy_metrics=accuracy,
            seasonality_detected=season_profile.pattern,
            trend_direction=trend_dir,
            trend_strength=trend_str,
        )

        # Persist
        self._forecasts.append(fc.to_dict())
        self._save_forecasts()
        self._bump_stat("total_forecasts")
        self._stats["last_forecast_at"] = _now_iso()
        self._save_stats()

        logger.info(
            "Generated %s forecast for %s/%s: %d-day horizon, "
            "total=$%.2f, trend=%s (%.2f), season=%s",
            method_val, stream_val, site_id or "all",
            horizon_days, fc.total_predicted,
            trend_dir, trend_str, season_profile.pattern,
        )

        return fc

    async def forecast_all_streams(
        self,
        site_id: str = "",
        horizon_days: int = 30,
        method: str | ForecastMethod = ForecastMethod.ENSEMBLE,
    ) -> dict[str, Forecast]:
        """Generate forecasts for all revenue streams.

        Args:
            site_id: Optional site filter.
            horizon_days: Forecast horizon.
            method: Forecasting method.

        Returns:
            Dict mapping stream name to Forecast object.
        """
        results: dict[str, Forecast] = {}

        for stream in RevenueStream:
            if stream == RevenueStream.TOTAL:
                continue
            try:
                fc = await self.forecast(
                    stream=stream,
                    site_id=site_id,
                    method=method,
                    horizon_days=horizon_days,
                )
                results[stream.value] = fc
            except Exception as exc:
                logger.warning("Failed to forecast %s: %s", stream.value, exc)

        # Also forecast total
        try:
            total_fc = await self.forecast(
                stream=RevenueStream.TOTAL,
                site_id=site_id,
                method=method,
                horizon_days=horizon_days,
            )
            results[RevenueStream.TOTAL.value] = total_fc
        except Exception as exc:
            logger.warning("Failed to forecast total: %s", exc)

        logger.info(
            "Forecast all streams: %d streams, site=%s, horizon=%d days",
            len(results), site_id or "all", horizon_days,
        )
        return results

    async def forecast_site(
        self,
        site_id: str,
        horizon_days: int = 30,
        method: str | ForecastMethod = ForecastMethod.ENSEMBLE,
    ) -> dict[str, Forecast]:
        """Generate forecasts for all streams of a specific site.

        Args:
            site_id: Site identifier.
            horizon_days: Forecast horizon.
            method: Forecasting method.

        Returns:
            Dict mapping stream name to Forecast object.
        """
        return await self.forecast_all_streams(
            site_id=site_id,
            horizon_days=horizon_days,
            method=method,
        )

    # ------------------------------------------------------------------
    # Backtesting helpers
    # ------------------------------------------------------------------

    def _backtest_residuals(
        self,
        data: list[float],
        method_name: str,
        test_fraction: float = 0.2,
    ) -> list[float]:
        """Run a backtest and return residuals (actual - predicted).

        Splits data into train/test, generates forecast on train,
        and computes residuals against the test set.
        """
        n = len(data)
        test_size = max(1, int(n * test_fraction))
        if n < test_size + MIN_POINTS_FOR_FORECAST:
            return []

        train = data[:n - test_size]
        test = data[n - test_size:]

        try:
            predictions = self._run_method(method_name, train, test_size)
        except Exception:
            return []

        return _residuals(predictions, test)

    def _backtest_accuracy(
        self,
        data: list[float],
        method_name: str,
        test_fraction: float = 0.2,
    ) -> dict:
        """Run a backtest and return accuracy metrics dict."""
        n = len(data)
        test_size = max(1, int(n * test_fraction))
        if n < test_size + MIN_POINTS_FOR_FORECAST:
            return {"MAE": 0.0, "MAPE": 0.0, "RMSE": 0.0}

        train = data[:n - test_size]
        test = data[n - test_size:]

        try:
            predictions = self._run_method(method_name, train, test_size)
        except Exception:
            return {"MAE": 0.0, "MAPE": 0.0, "RMSE": 0.0}

        return {
            "MAE": _round_amount(_calculate_mae(predictions, test)),
            "MAPE": _round_amount(_calculate_mape(predictions, test)),
            "RMSE": _round_amount(_calculate_rmse(predictions, test)),
        }

    # ==================================================================
    # ACCURACY EVALUATION
    # ==================================================================

    def evaluate_accuracy(
        self,
        forecast_id: str,
        actual_values: list[dict],
    ) -> ForecastAccuracy:
        """Evaluate a past forecast against actual realized values.

        Args:
            forecast_id: ID of the forecast to evaluate.
            actual_values: List of dicts with 'date' and 'amount' keys.

        Returns:
            ForecastAccuracy with computed metrics.
        """
        # Find the forecast
        forecast_data = None
        for fc in self._forecasts:
            if fc.get("forecast_id") == forecast_id:
                forecast_data = fc
                break

        if forecast_data is None:
            raise ValueError(f"Forecast not found: {forecast_id}")

        predictions = forecast_data.get("predictions", [])

        # Align predictions with actuals by date
        actual_by_date = {a["date"]: float(a["amount"]) for a in actual_values}
        pred_values: list[float] = []
        act_values: list[float] = []

        for pred in predictions:
            pred_date = pred.get("date", "")
            if pred_date in actual_by_date:
                pred_values.append(pred.get("amount", 0.0))
                act_values.append(actual_by_date[pred_date])

        if not pred_values:
            logger.warning("No matching dates between forecast and actuals")
            return ForecastAccuracy(forecast_id=forecast_id)

        mae = _calculate_mae(pred_values, act_values)
        mape = _calculate_mape(pred_values, act_values)
        rmse = _calculate_rmse(pred_values, act_values)
        r_sq = _calculate_r_squared(pred_values, act_values)

        accuracy = ForecastAccuracy(
            forecast_id=forecast_id,
            mae=_round_amount(mae),
            mape=_round_amount(mape),
            rmse=_round_amount(rmse),
            r_squared=_round_amount(r_sq),
            predictions_count=len(pred_values),
            actual_count=len(act_values),
            best_method=forecast_data.get("method", ""),
        )

        # Persist
        self._accuracy.append(accuracy.to_dict())
        self._save_accuracy()
        self._bump_stat("evaluations_run")

        logger.info(
            "Evaluated forecast %s: MAE=%.2f, MAPE=%.1f%%, RMSE=%.2f, R2=%.3f",
            forecast_id, mae, mape, rmse, r_sq,
        )

        return accuracy

    def get_method_accuracy(
        self,
        method: Optional[str] = None,
    ) -> dict:
        """Get aggregate accuracy metrics across all evaluated forecasts.

        Args:
            method: Filter to a specific method, or None for all.

        Returns:
            Dict with per-method accuracy summaries.
        """
        method_metrics: Dict[str, list[dict]] = {}

        for acc in self._accuracy:
            m = acc.get("best_method", "unknown")
            if method and m != method:
                continue
            method_metrics.setdefault(m, []).append(acc)

        result: dict = {}
        for m, records in method_metrics.items():
            maes = [r.get("mae", 0.0) for r in records]
            mapes = [r.get("mape", 0.0) for r in records]
            rmses = [r.get("rmse", 0.0) for r in records]
            r_squareds = [r.get("r_squared", 0.0) for r in records]

            result[m] = {
                "evaluations": len(records),
                "avg_mae": _round_amount(_mean(maes)),
                "avg_mape": _round_amount(_mean(mapes)),
                "avg_rmse": _round_amount(_mean(rmses)),
                "avg_r_squared": _round_amount(_mean(r_squareds)),
                "best_mae": _round_amount(min(maes)) if maes else 0.0,
                "worst_mape": _round_amount(max(mapes)) if mapes else 0.0,
            }

        # Determine best overall method
        if result:
            best = min(result.items(), key=lambda x: x[1].get("avg_mape", float("inf")))
            self._stats["best_overall_method"] = best[0]
            self._save_stats()

        return result

    # ==================================================================
    # REPORTING & OUTLOOK
    # ==================================================================

    def get_stats(self) -> dict:
        """Return aggregate statistics about the forecaster's state."""
        stats = dict(self._stats)
        stats.update({
            "data_points_count": len(self._data_points),
            "forecasts_count": len(self._forecasts),
            "accuracy_records_count": len(self._accuracy),
            "seasonality_profiles": len(self._seasonality),
            "config": self._config,
        })

        # Method accuracy summary
        accuracy_summary = self.get_method_accuracy()
        stats["method_accuracy"] = accuracy_summary

        # Data coverage
        if self._data_points:
            dates = sorted(set(dp.get("date", "") for dp in self._data_points))
            stats["data_range"] = f"{dates[0]} to {dates[-1]}" if dates else ""
            streams = sorted(set(dp.get("stream", "") for dp in self._data_points))
            stats["streams_tracked"] = streams
            sites = sorted(set(dp.get("site_id", "") for dp in self._data_points if dp.get("site_id")))
            stats["sites_tracked"] = sites
        else:
            stats["data_range"] = ""
            stats["streams_tracked"] = []
            stats["sites_tracked"] = []

        return stats

    async def generate_outlook(self, days: int = 30) -> str:
        """Generate a human-readable revenue outlook report.

        Runs ensemble forecasts across all streams and produces a
        formatted text summary with trend narratives.

        Args:
            days: Forecast horizon in days.

        Returns:
            Formatted text outlook report.
        """
        lines: list[str] = []
        lines.append(f"REVENUE OUTLOOK — Next {days} Days")
        lines.append(f"Generated: {_now_iso()[:16]}")
        lines.append("=" * 50)
        lines.append("")

        # Forecast all streams
        all_forecasts = await self.forecast_all_streams(horizon_days=days)

        total_predicted = 0.0
        stream_summaries: list[Tuple[str, float, str, float]] = []

        for stream_name, fc in sorted(all_forecasts.items()):
            predicted = fc.total_predicted
            total_predicted += predicted
            stream_summaries.append((
                stream_name,
                predicted,
                fc.trend_direction,
                fc.trend_strength,
            ))

        # Header totals
        daily_avg = total_predicted / days if days > 0 else 0.0
        lines.append(f"PROJECTED TOTAL: ${total_predicted:,.2f}")
        lines.append(f"Daily Average:   ${daily_avg:,.2f}")
        lines.append("")

        # Per-stream breakdown
        lines.append("BY STREAM:")
        lines.append("-" * 50)
        for stream_name, predicted, trend_dir, trend_str in sorted(
            stream_summaries, key=lambda x: x[1], reverse=True,
        ):
            pct = (predicted / total_predicted * 100) if total_predicted > 0 else 0.0
            trend_arrow = {"up": "+", "down": "-", "flat": "~"}.get(trend_dir, "~")
            trend_label = f"{trend_arrow}{trend_str:.0%}" if trend_str > 0 else "flat"
            lines.append(
                f"  {stream_name:<18} ${predicted:>10,.2f}  "
                f"({pct:>4.1f}%)  trend: {trend_label}"
            )
        lines.append("")

        # Trend narrative
        lines.append("TREND ANALYSIS:")
        lines.append("-" * 50)

        uptrending = [s for s in stream_summaries if s[2] == "up" and s[3] >= 0.1]
        downtrending = [s for s in stream_summaries if s[2] == "down" and s[3] >= 0.1]
        flat = [s for s in stream_summaries if s[2] == "flat" or s[3] < 0.1]

        if uptrending:
            names = ", ".join(s[0] for s in uptrending)
            lines.append(f"  Growing:  {names}")

        if downtrending:
            names = ", ".join(s[0] for s in downtrending)
            lines.append(f"  Declining: {names}")

        if flat:
            names = ", ".join(s[0] for s in flat)
            lines.append(f"  Stable:   {names}")

        lines.append("")

        # Seasonality notes
        lines.append("SEASONALITY:")
        lines.append("-" * 50)
        season_notes: list[str] = []
        for key, profile_data in self._seasonality.items():
            pattern = profile_data.get("pattern", "none")
            strength = profile_data.get("strength", 0.0)
            if pattern != "none" and strength > 0.1:
                season_notes.append(
                    f"  {key}: {pattern} pattern (strength: {strength:.0%})"
                )
        if season_notes:
            lines.extend(season_notes)
        else:
            lines.append("  No significant seasonal patterns detected.")

        lines.append("")

        # Confidence note
        lines.append("NOTE: Forecasts use ensemble method with 80% confidence intervals.")
        lines.append("Actuals may vary. Longer horizons have wider uncertainty bands.")

        return "\n".join(lines)

    # ==================================================================
    # ASYNC WRAPPERS
    # ==================================================================

    async def async_add_data_point(
        self,
        date_str: str,
        stream: str | RevenueStream,
        amount: float,
        **kwargs: Any,
    ) -> RevenueDataPoint:
        """Async wrapper for add_data_point."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.add_data_point(date_str, stream, amount, **kwargs)
        )

    async def async_detect_seasonality(
        self,
        data: list[float],
        granularity: Granularity = Granularity.DAILY,
    ) -> SeasonalityProfile:
        """Async wrapper for detect_seasonality."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.detect_seasonality(data, granularity)
        )

    async def async_detect_trend(self, data: list[float]) -> Tuple[str, float]:
        """Async wrapper for detect_trend."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.detect_trend(data))

    async def async_evaluate_accuracy(
        self,
        forecast_id: str,
        actual_values: list[dict],
    ) -> ForecastAccuracy:
        """Async wrapper for evaluate_accuracy."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.evaluate_accuracy(forecast_id, actual_values)
        )


# ===================================================================
# Module-Level Singleton API
# ===================================================================

_forecaster_instance: Optional[RevenueForecaster] = None


def get_forecaster() -> RevenueForecaster:
    """Return the singleton RevenueForecaster instance."""
    global _forecaster_instance
    if _forecaster_instance is None:
        _forecaster_instance = RevenueForecaster()
    return _forecaster_instance


# ===================================================================
# CLI Entry Point
# ===================================================================


def main() -> None:
    """CLI entry point: python -m src.revenue_forecaster <command> [options]."""

    parser = argparse.ArgumentParser(
        prog="revenue_forecaster",
        description="OpenClaw Empire Revenue Forecaster — Statistical Forecasting CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- forecast ---
    p_fc = subparsers.add_parser("forecast", help="Generate a revenue forecast")
    p_fc.add_argument("--stream", default="total",
                       help="Revenue stream (adsense, affiliate, kdp, etsy, substack, sponsorship, services, total)")
    p_fc.add_argument("--site", default="",
                       help="Site ID filter (optional)")
    p_fc.add_argument("--method", default="ensemble",
                       choices=[m.value for m in ForecastMethod],
                       help="Forecast method (default: ensemble)")
    p_fc.add_argument("--horizon", type=int, default=30,
                       help="Forecast horizon in days (default: 30)")
    p_fc.add_argument("--confidence", default="medium",
                       choices=[c.value for c in ConfidenceLevel],
                       help="Confidence level (default: medium)")
    p_fc.add_argument("--json", action="store_true",
                       help="Output raw JSON")

    # --- all-streams ---
    p_all = subparsers.add_parser("all-streams", help="Forecast all revenue streams")
    p_all.add_argument("--site", default="",
                        help="Site ID filter (optional)")
    p_all.add_argument("--horizon", type=int, default=30,
                        help="Forecast horizon in days (default: 30)")
    p_all.add_argument("--json", action="store_true",
                        help="Output raw JSON")

    # --- seasonality ---
    p_season = subparsers.add_parser("seasonality", help="Detect seasonality patterns")
    p_season.add_argument("--stream", default="total",
                           help="Revenue stream")
    p_season.add_argument("--site", default="",
                           help="Site ID filter (optional)")
    p_season.add_argument("--days", type=int, default=180,
                           help="Days of history to analyze (default: 180)")
    p_season.add_argument("--granularity", default="daily",
                           choices=[g.value for g in Granularity],
                           help="Data granularity (default: daily)")

    # --- accuracy ---
    p_acc = subparsers.add_parser("accuracy", help="Show forecast accuracy metrics")
    p_acc.add_argument("--stream", default="",
                        help="Filter by stream (optional)")
    p_acc.add_argument("--method", default="",
                        help="Filter by method (optional)")

    # --- history ---
    p_hist = subparsers.add_parser("history", help="Show revenue history data")
    p_hist.add_argument("--stream", default="",
                         help="Revenue stream filter")
    p_hist.add_argument("--site", default="",
                         help="Site ID filter")
    p_hist.add_argument("--days", type=int, default=30,
                         help="Days of history (default: 30)")
    p_hist.add_argument("--granularity", default="daily",
                         choices=[g.value for g in Granularity],
                         help="Aggregation level (default: daily)")

    # --- sync ---
    subparsers.add_parser("sync", help="Sync data from RevenueTracker")

    # --- trend ---
    p_trend = subparsers.add_parser("trend", help="Detect revenue trend")
    p_trend.add_argument("--stream", default="total",
                          help="Revenue stream")
    p_trend.add_argument("--site", default="",
                          help="Site ID filter")
    p_trend.add_argument("--days", type=int, default=90,
                          help="Days of history (default: 90)")

    # --- outlook ---
    p_outlook = subparsers.add_parser("outlook", help="Generate revenue outlook report")
    p_outlook.add_argument("--days", type=int, default=30,
                            help="Outlook horizon in days (default: 30)")

    # --- stats ---
    subparsers.add_parser("stats", help="Show forecaster statistics")

    # --- add ---
    p_add = subparsers.add_parser("add", help="Add a revenue data point")
    p_add.add_argument("--date", required=True,
                        help="Date (YYYY-MM-DD)")
    p_add.add_argument("--stream", required=True,
                        help="Revenue stream")
    p_add.add_argument("--amount", type=float, required=True,
                        help="Amount in USD")
    p_add.add_argument("--site", default="",
                        help="Site ID (optional)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    forecaster = get_forecaster()

    # --- Dispatch ---

    if args.command == "forecast":
        fc = _run_sync(forecaster.forecast(
            stream=args.stream,
            site_id=args.site,
            method=args.method,
            horizon_days=args.horizon,
            confidence=args.confidence,
        ))

        if args.json:
            print(json.dumps(fc.to_dict(), indent=2, default=str))
        else:
            _print_forecast(fc)

    elif args.command == "all-streams":
        results = _run_sync(forecaster.forecast_all_streams(
            site_id=args.site,
            horizon_days=args.horizon,
        ))

        if args.json:
            output = {k: v.to_dict() for k, v in results.items()}
            print(json.dumps(output, indent=2, default=str))
        else:
            total_projected = 0.0
            print(f"FORECAST ALL STREAMS — {args.horizon}-day horizon")
            if args.site:
                print(f"Site: {args.site}")
            print("=" * 55)

            for stream_name, fc in sorted(results.items(), key=lambda x: x[1].total_predicted, reverse=True):
                total_projected += fc.total_predicted
                trend_arrow = {"up": "+", "down": "-", "flat": "~"}.get(fc.trend_direction, "~")
                print(
                    f"  {stream_name:<18} ${fc.total_predicted:>10,.2f}  "
                    f"(${fc.average_daily:>7,.2f}/day)  "
                    f"trend: {trend_arrow}{fc.trend_strength:.0%}"
                )

            print("=" * 55)
            daily_avg = total_projected / args.horizon if args.horizon > 0 else 0
            print(f"  {'TOTAL':<18} ${total_projected:>10,.2f}  (${daily_avg:>7,.2f}/day)")

    elif args.command == "seasonality":
        stream_val = None
        if args.stream and args.stream != "total":
            stream_val = RevenueStream.from_string(args.stream).value

        history = forecaster.get_history(
            stream=stream_val,
            site_id=args.site or None,
            days=args.days,
            granularity=Granularity(args.granularity),
        )
        data_values = [dp.amount for dp in history]

        if not data_values:
            print(f"No data found for stream={args.stream}, site={args.site or 'all'}")
            sys.exit(0)

        profile = forecaster.detect_seasonality(data_values, Granularity(args.granularity))

        print(f"SEASONALITY ANALYSIS — {args.stream}")
        if args.site:
            print(f"Site: {args.site}")
        print(f"Data points: {len(data_values)}, period: {args.days} days")
        print("=" * 45)
        print(f"  Pattern:       {profile.pattern}")
        print(f"  Period:        {profile.period_days} days")
        print(f"  Strength:      {profile.strength:.0%}")

        if profile.seasonal_indices:
            print(f"  Indices:       {len(profile.seasonal_indices)} sub-periods")
            for i, idx in enumerate(profile.seasonal_indices):
                bar = "#" * int(idx * 10)
                label = "PEAK" if i in profile.peak_periods else (
                    "TROUGH" if i in profile.trough_periods else ""
                )
                print(f"    [{i:>2}] {idx:>5.2f}x  {bar}  {label}")

        if profile.peak_periods:
            print(f"  Peak periods:  {profile.peak_periods}")
        if profile.trough_periods:
            print(f"  Trough periods: {profile.trough_periods}")

    elif args.command == "accuracy":
        accuracy = forecaster.get_method_accuracy(
            method=args.method or None,
        )

        if not accuracy:
            print("No accuracy data available yet.")
            print("Run forecasts and then evaluate them with actual values.")
            sys.exit(0)

        print("FORECAST ACCURACY BY METHOD")
        print("=" * 65)
        print(f"  {'Method':<25} {'Evals':>5}  {'MAE':>8}  {'MAPE':>7}  {'RMSE':>8}  {'R2':>6}")
        print("-" * 65)
        for method_name, metrics in sorted(accuracy.items(),
                                           key=lambda x: x[1].get("avg_mape", 999)):
            print(
                f"  {method_name:<25} {metrics['evaluations']:>5}  "
                f"${metrics['avg_mae']:>7,.2f}  {metrics['avg_mape']:>6.1f}%  "
                f"${metrics['avg_rmse']:>7,.2f}  {metrics['avg_r_squared']:>5.3f}"
            )

        best_method = forecaster._stats.get("best_overall_method", "")
        if best_method:
            print(f"\nBest method (lowest MAPE): {best_method}")

    elif args.command == "history":
        stream_val = None
        if args.stream:
            stream_val = RevenueStream.from_string(args.stream).value

        history = forecaster.get_history(
            stream=stream_val,
            site_id=args.site or None,
            days=args.days,
            granularity=Granularity(args.granularity),
        )

        if not history:
            print(f"No history found for stream={args.stream or 'all'}, site={args.site or 'all'}")
            sys.exit(0)

        print(f"REVENUE HISTORY — {args.days} days ({args.granularity})")
        if args.stream:
            print(f"Stream: {args.stream}")
        if args.site:
            print(f"Site: {args.site}")
        print(f"Data points: {len(history)}")
        print("=" * 40)

        for dp in history:
            print(f"  {dp.date}  ${dp.amount:>10,.2f}")

        # Summary
        amounts = [dp.amount for dp in history]
        total = sum(amounts)
        avg = _mean(amounts)
        print(f"\n  Total:   ${total:>10,.2f}")
        print(f"  Average: ${avg:>10,.2f}")
        print(f"  Min:     ${min(amounts):>10,.2f}")
        print(f"  Max:     ${max(amounts):>10,.2f}")

    elif args.command == "sync":
        count = _run_sync(forecaster.sync_from_revenue_tracker())
        print(f"Synced {count} data points from RevenueTracker")
        print(f"Total data points: {len(forecaster._data_points)}")

    elif args.command == "trend":
        stream_val = None
        if args.stream and args.stream != "total":
            stream_val = RevenueStream.from_string(args.stream).value

        history = forecaster.get_history(
            stream=stream_val,
            site_id=args.site or None,
            days=args.days,
        )
        data_values = [dp.amount for dp in history]

        if not data_values:
            print(f"No data found for stream={args.stream}, site={args.site or 'all'}")
            sys.exit(0)

        direction, strength = forecaster.detect_trend(data_values)

        print(f"TREND ANALYSIS — {args.stream}")
        if args.site:
            print(f"Site: {args.site}")
        print(f"Data points: {len(data_values)}, period: {args.days} days")
        print("=" * 45)

        arrow = {"up": "UPWARD", "down": "DOWNWARD", "flat": "FLAT"}.get(direction, "FLAT")
        print(f"  Direction: {arrow}")
        print(f"  Strength:  {strength:.0%}")

        # Linear regression details
        if len(data_values) >= MIN_POINTS_FOR_REGRESSION:
            x = [float(i) for i in range(len(data_values))]
            slope, intercept = _linear_regression(x, data_values)
            print(f"  Slope:     ${slope:+,.4f}/day")
            print(f"  Intercept: ${intercept:,.2f}")

            # Projected change over the period
            change = slope * len(data_values)
            print(f"  Period change: ${change:+,.2f}")

        # Recent vs older comparison
        if len(data_values) >= 14:
            half = len(data_values) // 2
            first_half_avg = _mean(data_values[:half])
            second_half_avg = _mean(data_values[half:])
            if first_half_avg > 0:
                change_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100
                print(f"\n  Recent vs older avg:")
                print(f"    Older half avg:  ${first_half_avg:>10,.2f}")
                print(f"    Recent half avg: ${second_half_avg:>10,.2f}")
                print(f"    Change:          {change_pct:+.1f}%")

    elif args.command == "outlook":
        report = _run_sync(forecaster.generate_outlook(days=args.days))
        print(report)

    elif args.command == "stats":
        stats = forecaster.get_stats()
        print("REVENUE FORECASTER STATISTICS")
        print("=" * 50)
        print(f"  Data points:       {stats.get('data_points_count', 0):,}")
        print(f"  Forecasts stored:  {stats.get('forecasts_count', 0):,}")
        print(f"  Accuracy records:  {stats.get('accuracy_records_count', 0):,}")
        print(f"  Seasonality profiles: {stats.get('seasonality_profiles', 0)}")
        print(f"  Total forecasts run:  {stats.get('total_forecasts', 0):,}")
        print(f"  Evaluations run:      {stats.get('evaluations_run', 0):,}")
        print(f"  Best method:          {stats.get('best_overall_method', 'N/A')}")
        print(f"  Last forecast:        {stats.get('last_forecast_at', 'never')[:16]}")
        print(f"  Last sync:            {stats.get('last_sync_at', 'never')[:16]}")
        print(f"  Data range:           {stats.get('data_range', 'none')}")

        if stats.get("streams_tracked"):
            print(f"  Streams tracked:      {', '.join(stats['streams_tracked'])}")
        if stats.get("sites_tracked"):
            sites = stats["sites_tracked"]
            print(f"  Sites tracked:        {len(sites)} sites")

        # Config summary
        config = stats.get("config", {})
        if config:
            print(f"\n  Config:")
            print(f"    Default horizon: {config.get('default_horizon_days', 30)} days")
            print(f"    Default method:  {config.get('default_method', 'ensemble')}")
            print(f"    SMA window:      {config.get('sma_window', 7)}")
            print(f"    Alpha:           {config.get('alpha', 0.3)}")
            print(f"    Beta:            {config.get('beta', 0.1)}")
            print(f"    Gamma:           {config.get('gamma', 0.1)}")
            print(f"    Season period:   {config.get('season_period', 7)}")

        # Method accuracy
        if stats.get("method_accuracy"):
            print(f"\n  Method Accuracy:")
            for m, metrics in sorted(stats["method_accuracy"].items(),
                                     key=lambda x: x[1].get("avg_mape", 999)):
                print(
                    f"    {m:<25} MAE=${metrics['avg_mae']:.2f}  "
                    f"MAPE={metrics['avg_mape']:.1f}%  "
                    f"R2={metrics['avg_r_squared']:.3f}"
                )

    elif args.command == "add":
        dp = forecaster.add_data_point(
            date_str=args.date,
            stream=args.stream,
            amount=args.amount,
            site_id=args.site,
        )
        print(f"Added: ${dp.amount:.2f} {dp.stream} on {dp.date}")
        if dp.site_id:
            print(f"  Site: {dp.site_id}")
        print(f"Total data points: {len(forecaster._data_points)}")

    else:
        parser.print_help()


def _print_forecast(fc: Forecast) -> None:
    """Pretty-print a Forecast to stdout."""
    print(f"REVENUE FORECAST")
    print(f"  Stream:     {fc.stream}")
    if fc.site_id:
        print(f"  Site:       {fc.site_id}")
    print(f"  Method:     {fc.method}")
    print(f"  Horizon:    {fc.horizon_days} days")
    print(f"  Confidence: {fc.confidence_level} ({ConfidenceLevel(fc.confidence_level).percentage}%)")
    print(f"  Created:    {fc.created_at[:16]}")
    print("=" * 55)

    # Summary
    print(f"  Total projected:  ${fc.total_predicted:>10,.2f}")
    print(f"  Daily average:    ${fc.average_daily:>10,.2f}")
    print(f"  Trend:            {fc.trend_direction} ({fc.trend_strength:.0%})")
    print(f"  Seasonality:      {fc.seasonality_detected}")

    # Accuracy
    acc = fc.accuracy_metrics
    if acc and any(v > 0 for v in acc.values()):
        print(f"\n  Backtest Accuracy:")
        print(f"    MAE:  ${acc.get('MAE', 0):,.2f}")
        print(f"    MAPE: {acc.get('MAPE', 0):.1f}%")
        print(f"    RMSE: ${acc.get('RMSE', 0):,.2f}")

    # Predictions (show first 10 and last 5 if long)
    predictions = fc.predictions
    if predictions:
        print(f"\n  Predictions ({len(predictions)} days):")
        print(f"  {'Date':<12} {'Amount':>10} {'Lower':>10} {'Upper':>10}")
        print(f"  {'-' * 44}")

        show_count = min(10, len(predictions))
        for pred in predictions[:show_count]:
            print(
                f"  {pred.get('date', ''):<12} "
                f"${pred.get('amount', 0):>9,.2f} "
                f"${pred.get('lower', 0):>9,.2f} "
                f"${pred.get('upper', 0):>9,.2f}"
            )

        if len(predictions) > 15:
            print(f"  {'...':<12} {'...':>10} {'...':>10} {'...':>10}")
            for pred in predictions[-5:]:
                print(
                    f"  {pred.get('date', ''):<12} "
                    f"${pred.get('amount', 0):>9,.2f} "
                    f"${pred.get('lower', 0):>9,.2f} "
                    f"${pred.get('upper', 0):>9,.2f}"
                )


if __name__ == "__main__":
    main()
