"""
Anomaly Detector — Statistical Anomaly Detection for OpenClaw Empire

Real-time statistical anomaly detection across accounts, revenue, traffic,
content metrics, and social engagement for Nick Creighton's 16-site
WordPress publishing empire.

Detection methods (all pure Python, no numpy/scipy):
    Z-SCORE          — Flag values beyond N standard deviations from mean
    IQR              — Interquartile range outlier detection
    TREND_CHANGE     — Linear regression slope reversal detection
    FLATLINE         — Detect unexpected zero/constant metric runs
    THRESHOLD        — User-configured min/max boundary alerts
    PATTERN_BREAK    — Day-of-week seasonal deviation detection

All data persisted to: data/anomalies/

Usage:
    from src.anomaly_detector import get_detector, MetricSource

    detector = get_detector()

    # Ingest a data point and check for anomalies
    anomaly = await detector.ingest(42.50, MetricSource.REVENUE, "daily_ads")
    if anomaly:
        print(f"ANOMALY: {anomaly.description}")

    # Synchronous variant
    anomaly = detector.ingest_sync(42.50, MetricSource.REVENUE, "daily_ads")

    # Batch ingest
    anomalies = await detector.ingest_batch([
        DataPoint(timestamp=_now_iso(), value=100, source=MetricSource.TRAFFIC,
                  metric_name="witchcraft_sessions"),
        DataPoint(timestamp=_now_iso(), value=5.0, source=MetricSource.REVENUE,
                  metric_name="adsense"),
    ])

    # Set custom thresholds
    detector.set_threshold(MetricSource.REVENUE, "daily_ads",
                           min_val=5.0, max_val=500.0, z_score_threshold=2.5)

    # Query anomalies
    recent = detector.get_recent(hours=24)
    critical = detector.get_anomalies(severity=AnomalySeverity.CRITICAL)

    # Health dashboard
    health = detector.get_metric_health()

CLI:
    python -m src.anomaly_detector status           # Overview of all tracked metrics
    python -m src.anomaly_detector anomalies        # List recent anomalies
    python -m src.anomaly_detector anomalies --severity critical
    python -m src.anomaly_detector summary          # 7-day anomaly summary
    python -m src.anomaly_detector summary --days 30
    python -m src.anomaly_detector profiles         # Show metric profiles
    python -m src.anomaly_detector health           # Metric health dashboard
    python -m src.anomaly_detector thresholds       # Show configured thresholds
    python -m src.anomaly_detector set-threshold --source revenue --metric daily_ads --min 5 --max 500
    python -m src.anomaly_detector acknowledge ID   # Acknowledge an anomaly
    python -m src.anomaly_detector resolve ID       # Resolve an anomaly
    python -m src.anomaly_detector ingest --source revenue --metric daily_ads --value 42.5
    python -m src.anomaly_detector rebuild          # Rebuild all metric profiles
    python -m src.anomaly_detector stats            # Aggregate statistics
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import json
import logging
import math
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("anomaly_detector")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
ANOMALY_DATA_DIR = BASE_DIR / "data" / "anomalies"
ANOMALIES_FILE = ANOMALY_DATA_DIR / "anomalies.json"
PROFILES_FILE = ANOMALY_DATA_DIR / "profiles.json"
THRESHOLDS_FILE = ANOMALY_DATA_DIR / "thresholds.json"
DATA_POINTS_FILE = ANOMALY_DATA_DIR / "data_points.json"
STATS_FILE = ANOMALY_DATA_DIR / "stats.json"

# Ensure data directory exists on import
ANOMALY_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum stored items to prevent unbounded growth
MAX_ANOMALIES = 1000
MAX_DATA_POINTS_PER_KEY = 500
MAX_ANOMALY_AGE_DAYS = 90

# Default detection thresholds
DEFAULT_ZSCORE_THRESHOLD = 3.0
DEFAULT_IQR_FACTOR = 1.5
DEFAULT_FLATLINE_COUNT = 3
DEFAULT_TREND_WINDOW = 7
DEFAULT_SEASONALITY_PERIOD = 7

# Minimum data points required before running checks
MIN_POINTS_FOR_ZSCORE = 10
MIN_POINTS_FOR_IQR = 10
MIN_POINTS_FOR_TREND = 14
MIN_POINTS_FOR_SEASONALITY = 21

# Severity escalation thresholds (z-score magnitudes)
ZSCORE_WARNING = 2.0
ZSCORE_CRITICAL = 3.5

# Day-of-week names for seasonality
DOW_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    """Return the current time in UTC, timezone-aware."""
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return _now_utc().isoformat()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string back to a timezone-aware datetime."""
    if s is None or s == "":
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _hours_ago(hours: int) -> str:
    """Return an ISO-8601 string for N hours ago."""
    return (_now_utc() - timedelta(hours=hours)).isoformat()


def _days_ago(days: int) -> str:
    """Return an ISO-8601 string for N days ago."""
    return (_now_utc() - timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# JSON persistence helpers (atomic writes)
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
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        # Atomic replace
        if os.name == "nt":
            os.replace(str(tmp), str(path))
        else:
            tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Async/sync helper
# ---------------------------------------------------------------------------

def _run_sync(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnomalyType(str, Enum):
    """Classification of detected anomaly."""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    OUTLIER = "outlier"
    FLATLINE = "flatline"
    PATTERN_BREAK = "pattern_break"
    THRESHOLD = "threshold"


class AnomalySeverity(str, Enum):
    """Severity level for an anomaly."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricSource(str, Enum):
    """Origin category for a metric data point."""
    REVENUE = "revenue"
    TRAFFIC = "traffic"
    ACCOUNTS = "accounts"
    CONTENT = "content"
    SOCIAL = "social"
    PHONE = "phone"
    API = "api"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DataPoint:
    """A single metric observation."""
    timestamp: str          # ISO-8601
    value: float
    source: MetricSource
    metric_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "source": self.source.value if isinstance(self.source, MetricSource) else str(self.source),
            "metric_name": self.metric_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DataPoint:
        """Deserialize from a dictionary."""
        source = d.get("source", "custom")
        try:
            source = MetricSource(source)
        except ValueError:
            source = MetricSource.CUSTOM
        return cls(
            timestamp=d.get("timestamp", _now_iso()),
            value=float(d.get("value", 0.0)),
            source=source,
            metric_name=d.get("metric_name", "unknown"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Anomaly:
    """A detected statistical anomaly."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    source: MetricSource
    metric_name: str
    detected_at: str        # ISO-8601
    value: float
    expected_value: float
    deviation: float        # standard deviations from mean
    z_score: float
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value if isinstance(self.anomaly_type, AnomalyType) else str(self.anomaly_type),
            "severity": self.severity.value if isinstance(self.severity, AnomalySeverity) else str(self.severity),
            "source": self.source.value if isinstance(self.source, MetricSource) else str(self.source),
            "metric_name": self.metric_name,
            "detected_at": self.detected_at,
            "value": self.value,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "z_score": self.z_score,
            "description": self.description,
            "context": self.context,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Anomaly:
        """Deserialize from a dictionary."""
        try:
            anomaly_type = AnomalyType(d.get("anomaly_type", "outlier"))
        except ValueError:
            anomaly_type = AnomalyType.OUTLIER
        try:
            severity = AnomalySeverity(d.get("severity", "info"))
        except ValueError:
            severity = AnomalySeverity.INFO
        try:
            source = MetricSource(d.get("source", "custom"))
        except ValueError:
            source = MetricSource.CUSTOM
        return cls(
            anomaly_id=d.get("anomaly_id", str(uuid.uuid4())),
            anomaly_type=anomaly_type,
            severity=severity,
            source=source,
            metric_name=d.get("metric_name", "unknown"),
            detected_at=d.get("detected_at", _now_iso()),
            value=float(d.get("value", 0.0)),
            expected_value=float(d.get("expected_value", 0.0)),
            deviation=float(d.get("deviation", 0.0)),
            z_score=float(d.get("z_score", 0.0)),
            description=d.get("description", ""),
            context=d.get("context", {}),
            acknowledged=d.get("acknowledged", False),
            resolved=d.get("resolved", False),
        )


@dataclass
class MetricProfile:
    """Statistical profile for a tracked metric."""
    source: MetricSource
    metric_name: str
    mean: float
    std_dev: float
    min_val: float
    max_val: float
    median: float
    sample_count: int
    last_updated: str       # ISO-8601
    trend_direction: str    # "up", "down", "stable"
    trend_slope: float
    seasonality_detected: bool
    day_of_week_means: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "source": self.source.value if isinstance(self.source, MetricSource) else str(self.source),
            "metric_name": self.metric_name,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "median": self.median,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
            "trend_direction": self.trend_direction,
            "trend_slope": self.trend_slope,
            "seasonality_detected": self.seasonality_detected,
            "day_of_week_means": self.day_of_week_means,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MetricProfile:
        """Deserialize from a dictionary."""
        try:
            source = MetricSource(d.get("source", "custom"))
        except ValueError:
            source = MetricSource.CUSTOM
        return cls(
            source=source,
            metric_name=d.get("metric_name", "unknown"),
            mean=float(d.get("mean", 0.0)),
            std_dev=float(d.get("std_dev", 0.0)),
            min_val=float(d.get("min_val", 0.0)),
            max_val=float(d.get("max_val", 0.0)),
            median=float(d.get("median", 0.0)),
            sample_count=int(d.get("sample_count", 0)),
            last_updated=d.get("last_updated", _now_iso()),
            trend_direction=d.get("trend_direction", "stable"),
            trend_slope=float(d.get("trend_slope", 0.0)),
            seasonality_detected=d.get("seasonality_detected", False),
            day_of_week_means=d.get("day_of_week_means", {}),
        )


# ---------------------------------------------------------------------------
# StatisticalEngine — Pure Python statistical computations
# ---------------------------------------------------------------------------

class StatisticalEngine:
    """Pure Python statistical computation engine.

    All methods are stateless and operate on lists of floats.
    No numpy, no scipy -- everything implemented from scratch.
    """

    # ── Basic statistics ──

    @staticmethod
    def mean(values: List[float]) -> float:
        """Arithmetic mean."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def std_deviation(values: List[float]) -> float:
        """Population standard deviation."""
        n = len(values)
        if n < 2:
            return 0.0
        avg = sum(values) / n
        variance = sum((x - avg) ** 2 for x in values) / n
        return math.sqrt(variance)

    @staticmethod
    def sample_std_deviation(values: List[float]) -> float:
        """Sample standard deviation (Bessel's correction)."""
        n = len(values)
        if n < 2:
            return 0.0
        avg = sum(values) / n
        variance = sum((x - avg) ** 2 for x in values) / (n - 1)
        return math.sqrt(variance)

    @staticmethod
    def median(values: List[float]) -> float:
        """Median value."""
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        mid = n // 2
        if n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return s[mid]

    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        """Compute the p-th percentile (0..100) using linear interpolation."""
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        if n == 1:
            return s[0]
        # Clamp percentile
        p = max(0.0, min(100.0, p))
        # Rank (0-indexed)
        rank = (p / 100.0) * (n - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return s[lo]
        frac = rank - lo
        return s[lo] + frac * (s[hi] - s[lo])

    @staticmethod
    def z_score(value: float, mean: float, std_dev: float) -> float:
        """Compute the z-score of a value given mean and standard deviation."""
        if std_dev == 0.0:
            if value == mean:
                return 0.0
            # Infinite deviation if std is zero but value differs
            return float("inf") if value > mean else float("-inf")
        return (value - mean) / std_dev

    # ── Moving averages ──

    @staticmethod
    def moving_average(values: List[float], window: int) -> List[float]:
        """Simple moving average with the given window size.

        Returns a list the same length as *values*. The first (window-1)
        entries use an expanding window for stability.
        """
        if not values or window < 1:
            return []
        result: List[float] = []
        running_sum = 0.0
        for i, v in enumerate(values):
            running_sum += v
            if i < window:
                result.append(running_sum / (i + 1))
            else:
                running_sum -= values[i - window]
                result.append(running_sum / window)
        return result

    @staticmethod
    def exponential_moving_average(values: List[float], alpha: float = 0.3) -> List[float]:
        """Exponential moving average (EMA).

        *alpha* is the smoothing factor between 0 and 1.  Higher alpha
        gives more weight to recent observations.
        """
        if not values:
            return []
        alpha = max(0.0, min(1.0, alpha))
        result: List[float] = [values[0]]
        for i in range(1, len(values)):
            ema = alpha * values[i] + (1.0 - alpha) * result[-1]
            result.append(ema)
        return result

    # ── Trend detection ──

    @staticmethod
    def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float]:
        """Simple linear regression returning (slope, intercept).

        Uses the ordinary least squares formula:
            slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
            intercept = mean(y) - slope * mean(x)
        """
        n = len(x)
        if n < 2 or len(y) < 2:
            return 0.0, 0.0
        n = min(n, len(y))  # handle mismatched lengths

        sum_x = sum(x[:n])
        sum_y = sum(y[:n])
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x[:n])

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-12:
            return 0.0, sum_y / n if n > 0 else 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept

    @staticmethod
    def detect_trend(values: List[float], window: int = 7) -> Tuple[str, float]:
        """Detect trend direction over the most recent *window* points.

        Returns:
            ("up" | "down" | "stable", slope)

        A slope whose absolute value is less than 1% of the mean is
        considered stable.
        """
        if len(values) < 3:
            return "stable", 0.0

        recent = values[-window:] if len(values) >= window else values[:]
        x = list(range(len(recent)))
        y = recent
        slope, _intercept = StatisticalEngine.linear_regression(
            [float(xi) for xi in x], y
        )

        # Determine significance relative to the mean
        avg = sum(recent) / len(recent) if recent else 1.0
        if avg == 0.0:
            avg = 1.0  # avoid division by zero

        relative_slope = abs(slope) / abs(avg)

        if relative_slope < 0.01:
            return "stable", slope
        elif slope > 0:
            return "up", slope
        else:
            return "down", slope

    @staticmethod
    def detect_seasonality(
        values: List[float],
        period: int = 7,
    ) -> Tuple[bool, Dict[str, float]]:
        """Detect weekly seasonality patterns.

        Groups values by position within each *period*-length cycle
        (e.g., day-of-week for period=7) and checks if inter-group
        variance is significantly larger than intra-group variance.

        Returns:
            (is_seasonal, {group_index_str: mean})
        """
        n = len(values)
        if n < period * 3:
            # Need at least 3 full cycles to detect seasonality
            return False, {}

        # Group values by cycle position
        groups: Dict[int, List[float]] = {i: [] for i in range(period)}
        for i, v in enumerate(values):
            groups[i % period].append(v)

        # Compute per-group means
        group_means: Dict[str, float] = {}
        group_mean_values: List[float] = []
        for idx in range(period):
            g = groups[idx]
            if g:
                gm = sum(g) / len(g)
                key = DOW_NAMES[idx] if idx < len(DOW_NAMES) else str(idx)
                group_means[key] = round(gm, 4)
                group_mean_values.append(gm)
            else:
                key = DOW_NAMES[idx] if idx < len(DOW_NAMES) else str(idx)
                group_means[key] = 0.0
                group_mean_values.append(0.0)

        # F-test approximation: compare between-group variance to within-group variance
        overall_mean = sum(values) / n
        k = period  # number of groups

        # Between-group sum of squares
        ss_between = sum(
            len(groups[i]) * (group_mean_values[i] - overall_mean) ** 2
            for i in range(k) if groups[i]
        )

        # Within-group sum of squares
        ss_within = 0.0
        for i in range(k):
            gm = group_mean_values[i]
            for v in groups[i]:
                ss_within += (v - gm) ** 2

        # Compute F-ratio (simplified)
        df_between = k - 1
        df_within = n - k

        if df_within <= 0 or ss_within == 0.0:
            return False, group_means

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_ratio = ms_between / ms_within

        # Use a conservative threshold: F > 3.0 suggests seasonality
        is_seasonal = f_ratio > 3.0

        return is_seasonal, group_means

    @staticmethod
    def iqr_outliers(values: List[float], factor: float = 1.5) -> List[int]:
        """Detect outlier indices using the interquartile range method.

        An outlier is any value below Q1 - factor*IQR or above Q3 + factor*IQR.
        Returns the indices of outliers in the original list.
        """
        if len(values) < 4:
            return []

        q1 = StatisticalEngine.percentile(values, 25.0)
        q3 = StatisticalEngine.percentile(values, 75.0)
        iqr = q3 - q1

        if iqr == 0.0:
            # All values in the middle 50% are the same -- no IQR-based outliers
            return []

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        return [i for i, v in enumerate(values) if v < lower or v > upper]

    # ── Utility ──

    @staticmethod
    def coefficient_of_variation(values: List[float]) -> float:
        """Coefficient of variation (CV): std_dev / |mean|."""
        if not values:
            return 0.0
        avg = sum(values) / len(values)
        if avg == 0.0:
            return 0.0
        sd = StatisticalEngine.std_deviation(values)
        return sd / abs(avg)

    @staticmethod
    def rate_of_change(values: List[float], window: int = 1) -> List[float]:
        """Compute the rate of change over a sliding window.

        Returns (len(values) - window) values, each representing
        (current - previous) / |previous|.
        """
        if len(values) <= window:
            return []
        result: List[float] = []
        for i in range(window, len(values)):
            prev = values[i - window]
            if prev == 0.0:
                roc = 0.0 if values[i] == 0.0 else float("inf")
            else:
                roc = (values[i] - prev) / abs(prev)
            result.append(roc)
        return result

    @staticmethod
    def rolling_zscore(values: List[float], window: int = 20) -> List[float]:
        """Compute z-scores using a rolling window of statistics.

        For each value, compute z-score relative to the preceding
        *window* values. Returns a list the same length as *values*;
        the first *window* entries use an expanding window.
        """
        result: List[float] = []
        for i in range(len(values)):
            start = max(0, i - window)
            subset = values[start:i]  # exclude current value
            if len(subset) < 2:
                result.append(0.0)
                continue
            avg = sum(subset) / len(subset)
            sd = StatisticalEngine.std_deviation(subset)
            result.append(StatisticalEngine.z_score(values[i], avg, sd))
        return result


# ---------------------------------------------------------------------------
# AnomalyDetector — Core detection engine
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Statistical anomaly detection engine for the OpenClaw Empire.

    Maintains metric profiles, data point history, thresholds, and a
    bounded list of detected anomalies.  All state is persisted to
    JSON files in the anomalies data directory.
    """

    def __init__(self) -> None:
        self._stats_engine = StatisticalEngine()
        self._profiles: Dict[str, MetricProfile] = {}
        self._anomalies: List[Anomaly] = []
        self._data_points: Dict[str, List[DataPoint]] = {}
        self._thresholds: Dict[str, Dict[str, float]] = {}
        self._detection_stats: Dict[str, int] = {
            "total_ingested": 0,
            "total_anomalies": 0,
            "by_type": {},
            "by_severity": {},
            "by_source": {},
        }
        self._load_state()

    # ── Persistence ──

    def _load_state(self) -> None:
        """Load all persisted state from disk."""
        # Load anomalies
        raw_anomalies = _load_json(ANOMALIES_FILE, [])
        self._anomalies = []
        for entry in raw_anomalies:
            try:
                self._anomalies.append(Anomaly.from_dict(entry))
            except Exception as exc:
                logger.warning("Skipping corrupt anomaly entry: %s", exc)

        # Load profiles
        raw_profiles = _load_json(PROFILES_FILE, {})
        self._profiles = {}
        for key, pdata in raw_profiles.items():
            try:
                self._profiles[key] = MetricProfile.from_dict(pdata)
            except Exception as exc:
                logger.warning("Skipping corrupt profile '%s': %s", key, exc)

        # Load data points
        raw_points = _load_json(DATA_POINTS_FILE, {})
        self._data_points = {}
        for key, points_list in raw_points.items():
            self._data_points[key] = []
            for pt in points_list:
                try:
                    self._data_points[key].append(DataPoint.from_dict(pt))
                except Exception as exc:
                    logger.warning("Skipping corrupt data point in '%s': %s", key, exc)

        # Load thresholds
        self._thresholds = _load_json(THRESHOLDS_FILE, {})

        # Load stats
        self._detection_stats = _load_json(STATS_FILE, {
            "total_ingested": 0,
            "total_anomalies": 0,
            "by_type": {},
            "by_severity": {},
            "by_source": {},
        })

        logger.debug(
            "Loaded state: %d anomalies, %d profiles, %d metric keys, %d thresholds",
            len(self._anomalies),
            len(self._profiles),
            len(self._data_points),
            len(self._thresholds),
        )

    def _save_anomalies(self) -> None:
        """Persist anomalies to disk."""
        _save_json(ANOMALIES_FILE, [a.to_dict() for a in self._anomalies])

    def _save_profiles(self) -> None:
        """Persist metric profiles to disk."""
        _save_json(PROFILES_FILE, {k: v.to_dict() for k, v in self._profiles.items()})

    def _save_data_points(self) -> None:
        """Persist data points to disk."""
        serialized: Dict[str, List[Dict]] = {}
        for key, points in self._data_points.items():
            serialized[key] = [p.to_dict() for p in points]
        _save_json(DATA_POINTS_FILE, serialized)

    def _save_thresholds(self) -> None:
        """Persist thresholds to disk."""
        _save_json(THRESHOLDS_FILE, self._thresholds)

    def _save_stats(self) -> None:
        """Persist detection statistics to disk."""
        _save_json(STATS_FILE, self._detection_stats)

    def _save_all(self) -> None:
        """Persist all state to disk."""
        self._save_anomalies()
        self._save_profiles()
        self._save_data_points()
        self._save_thresholds()
        self._save_stats()

    # ── Key generation ──

    @staticmethod
    def _make_key(source: MetricSource, metric_name: str) -> str:
        """Generate a composite key from source and metric name."""
        src = source.value if isinstance(source, MetricSource) else str(source)
        return f"{src}:{metric_name}"

    # ── Severity determination ──

    def _determine_severity(
        self,
        z_score: float,
        anomaly_type: AnomalyType,
        key: str,
    ) -> AnomalySeverity:
        """Determine anomaly severity based on z-score and configured thresholds."""
        abs_z = abs(z_score)

        # Check for custom z-score threshold
        custom_thresh = self._thresholds.get(key, {})
        custom_z = custom_thresh.get("z_score_threshold")

        if custom_z is not None:
            if abs_z >= custom_z * 1.5:
                return AnomalySeverity.CRITICAL
            elif abs_z >= custom_z:
                return AnomalySeverity.WARNING
            else:
                return AnomalySeverity.INFO

        # Default thresholds
        if abs_z >= ZSCORE_CRITICAL:
            return AnomalySeverity.CRITICAL
        elif abs_z >= ZSCORE_WARNING:
            return AnomalySeverity.WARNING
        else:
            return AnomalySeverity.INFO

    # ── Anomaly creation ──

    def _create_anomaly(
        self,
        anomaly_type: AnomalyType,
        source: MetricSource,
        metric_name: str,
        value: float,
        expected_value: float,
        deviation: float,
        z_score_val: float,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Anomaly:
        """Create, store, and return an Anomaly object."""
        key = self._make_key(source, metric_name)
        severity = self._determine_severity(z_score_val, anomaly_type, key)

        anomaly = Anomaly(
            anomaly_id=str(uuid.uuid4()),
            anomaly_type=anomaly_type,
            severity=severity,
            source=source,
            metric_name=metric_name,
            detected_at=_now_iso(),
            value=value,
            expected_value=round(expected_value, 4),
            deviation=round(deviation, 4),
            z_score=round(z_score_val, 4),
            description=description,
            context=context or {},
        )

        # Store anomaly
        self._anomalies.append(anomaly)

        # Enforce max size (keep most recent)
        if len(self._anomalies) > MAX_ANOMALIES:
            self._anomalies = self._anomalies[-MAX_ANOMALIES:]

        # Update stats
        self._detection_stats["total_anomalies"] = self._detection_stats.get("total_anomalies", 0) + 1
        type_key = anomaly_type.value
        self._detection_stats.setdefault("by_type", {})
        self._detection_stats["by_type"][type_key] = self._detection_stats["by_type"].get(type_key, 0) + 1
        sev_key = severity.value
        self._detection_stats.setdefault("by_severity", {})
        self._detection_stats["by_severity"][sev_key] = self._detection_stats["by_severity"].get(sev_key, 0) + 1
        src_key = source.value
        self._detection_stats.setdefault("by_source", {})
        self._detection_stats["by_source"][src_key] = self._detection_stats["by_source"].get(src_key, 0) + 1

        logger.info(
            "ANOMALY [%s] %s | %s:%s | value=%.2f expected=%.2f z=%.2f | %s",
            severity.value.upper(),
            anomaly_type.value,
            source.value,
            metric_name,
            value,
            expected_value,
            z_score_val,
            description,
        )

        return anomaly

    # ── Data point storage ──

    def _store_point(self, key: str, point: DataPoint) -> None:
        """Store a data point and enforce per-key size limits."""
        if key not in self._data_points:
            self._data_points[key] = []
        self._data_points[key].append(point)

        # Enforce max size per key
        if len(self._data_points[key]) > MAX_DATA_POINTS_PER_KEY:
            self._data_points[key] = self._data_points[key][-MAX_DATA_POINTS_PER_KEY:]

    def _get_values(self, key: str) -> List[float]:
        """Return stored values for a given key, chronologically ordered."""
        points = self._data_points.get(key, [])
        return [p.value for p in points]

    def _get_recent_values(self, key: str, count: int) -> List[float]:
        """Return the most recent N values for a key."""
        values = self._get_values(key)
        return values[-count:] if len(values) > count else values

    # ── Detection checks ──

    async def check_zscore(
        self,
        key: str,
        value: float,
        threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    ) -> Optional[Anomaly]:
        """Check if the value is a z-score outlier.

        Returns an Anomaly if the z-score exceeds the threshold, else None.
        """
        values = self._get_values(key)
        if len(values) < MIN_POINTS_FOR_ZSCORE:
            return None

        mean = self._stats_engine.mean(values)
        sd = self._stats_engine.std_deviation(values)

        if sd == 0.0:
            return None

        z = self._stats_engine.z_score(value, mean, sd)

        if abs(z) < threshold:
            return None

        # Determine type: spike or drop
        anomaly_type = AnomalyType.SPIKE if z > 0 else AnomalyType.DROP

        # Build context
        recent = values[-10:] if len(values) >= 10 else values[:]
        context = {
            "mean": round(mean, 4),
            "std_dev": round(sd, 4),
            "threshold": threshold,
            "recent_values": [round(v, 4) for v in recent],
            "sample_count": len(values),
        }

        source_str, metric_name = key.split(":", 1)
        try:
            source = MetricSource(source_str)
        except ValueError:
            source = MetricSource.CUSTOM

        direction = "above" if z > 0 else "below"
        description = (
            f"{metric_name} is {abs(z):.1f} std devs {direction} mean "
            f"(value={value:.2f}, mean={mean:.2f}, sd={sd:.2f})"
        )

        return self._create_anomaly(
            anomaly_type=anomaly_type,
            source=source,
            metric_name=metric_name,
            value=value,
            expected_value=mean,
            deviation=abs(value - mean),
            z_score_val=z,
            description=description,
            context=context,
        )

    async def check_iqr(
        self,
        key: str,
        value: float,
        factor: float = DEFAULT_IQR_FACTOR,
    ) -> Optional[Anomaly]:
        """Check if the value is an IQR outlier.

        Returns an Anomaly if the value is outside Q1 - factor*IQR or
        Q3 + factor*IQR, else None.
        """
        values = self._get_values(key)
        if len(values) < MIN_POINTS_FOR_IQR:
            return None

        q1 = self._stats_engine.percentile(values, 25.0)
        q3 = self._stats_engine.percentile(values, 75.0)
        iqr = q3 - q1

        if iqr == 0.0:
            return None

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        if lower <= value <= upper:
            return None

        # Compute z-score for severity
        mean = self._stats_engine.mean(values)
        sd = self._stats_engine.std_deviation(values)
        z = self._stats_engine.z_score(value, mean, sd) if sd > 0 else 0.0

        anomaly_type = AnomalyType.OUTLIER
        direction = "above Q3+IQR" if value > upper else "below Q1-IQR"

        source_str, metric_name = key.split(":", 1)
        try:
            source = MetricSource(source_str)
        except ValueError:
            source = MetricSource.CUSTOM

        context = {
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_fence": round(lower, 4),
            "upper_fence": round(upper, 4),
            "factor": factor,
            "sample_count": len(values),
        }

        description = (
            f"{metric_name} is an IQR outlier: {direction} "
            f"(value={value:.2f}, Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f})"
        )

        return self._create_anomaly(
            anomaly_type=anomaly_type,
            source=source,
            metric_name=metric_name,
            value=value,
            expected_value=mean,
            deviation=abs(value - mean),
            z_score_val=z,
            description=description,
            context=context,
        )

    async def check_trend_change(self, key: str) -> Optional[Anomaly]:
        """Detect a trend direction reversal.

        Compares the trend direction of the older half of the recent
        window to the newer half.  A reversal (up->down or down->up)
        triggers an anomaly.
        """
        values = self._get_values(key)
        if len(values) < MIN_POINTS_FOR_TREND:
            return None

        recent = values[-MIN_POINTS_FOR_TREND:]
        mid = len(recent) // 2
        older = recent[:mid]
        newer = recent[mid:]

        old_dir, old_slope = self._stats_engine.detect_trend(older)
        new_dir, new_slope = self._stats_engine.detect_trend(newer)

        # Only flag if direction actually reversed
        if old_dir == "stable" or new_dir == "stable":
            return None
        if old_dir == new_dir:
            return None

        source_str, metric_name = key.split(":", 1)
        try:
            source = MetricSource(source_str)
        except ValueError:
            source = MetricSource.CUSTOM

        current_value = values[-1]
        mean = self._stats_engine.mean(values)
        sd = self._stats_engine.std_deviation(values)
        z = self._stats_engine.z_score(current_value, mean, sd) if sd > 0 else 0.0

        context = {
            "old_trend": old_dir,
            "old_slope": round(old_slope, 6),
            "new_trend": new_dir,
            "new_slope": round(new_slope, 6),
            "window_size": MIN_POINTS_FOR_TREND,
            "older_values": [round(v, 4) for v in older],
            "newer_values": [round(v, 4) for v in newer],
        }

        description = (
            f"{metric_name} trend reversed from {old_dir} to {new_dir} "
            f"(old slope={old_slope:.4f}, new slope={new_slope:.4f})"
        )

        return self._create_anomaly(
            anomaly_type=AnomalyType.TREND_CHANGE,
            source=source,
            metric_name=metric_name,
            value=current_value,
            expected_value=mean,
            deviation=abs(new_slope - old_slope),
            z_score_val=z,
            description=description,
            context=context,
        )

    async def check_flatline(
        self,
        key: str,
        zero_count: int = DEFAULT_FLATLINE_COUNT,
    ) -> Optional[Anomaly]:
        """Detect an unexpected flatline (consecutive identical or zero values).

        A flatline is flagged when the last *zero_count* values are all
        identical (typically zero) and the metric historically varies.
        """
        values = self._get_values(key)
        if len(values) < zero_count + 5:
            return None

        recent = values[-zero_count:]

        # Check if all recent values are identical
        if len(set(recent)) != 1:
            return None

        flat_value = recent[0]

        # Only flag if the metric historically varies
        historical = values[:-zero_count]
        sd = self._stats_engine.std_deviation(historical)
        mean = self._stats_engine.mean(historical)

        # If historical std dev is basically zero too, the metric just does not vary
        if sd < 0.001:
            return None

        z = self._stats_engine.z_score(flat_value, mean, sd) if sd > 0 else 0.0

        source_str, metric_name = key.split(":", 1)
        try:
            source = MetricSource(source_str)
        except ValueError:
            source = MetricSource.CUSTOM

        context = {
            "flat_value": flat_value,
            "consecutive_count": zero_count,
            "historical_mean": round(mean, 4),
            "historical_std_dev": round(sd, 4),
        }

        label = "zero" if flat_value == 0.0 else f"constant ({flat_value})"
        description = (
            f"{metric_name} flatlined at {label} for {zero_count} consecutive readings "
            f"(historical mean={mean:.2f}, sd={sd:.2f})"
        )

        return self._create_anomaly(
            anomaly_type=AnomalyType.FLATLINE,
            source=source,
            metric_name=metric_name,
            value=flat_value,
            expected_value=mean,
            deviation=abs(flat_value - mean),
            z_score_val=z,
            description=description,
            context=context,
        )

    async def check_threshold(self, key: str, value: float) -> Optional[Anomaly]:
        """Check if the value crosses a configured threshold.

        Returns an Anomaly if the value is below the configured min
        or above the configured max, else None.
        """
        thresh = self._thresholds.get(key)
        if not thresh:
            return None

        min_val = thresh.get("min_val")
        max_val = thresh.get("max_val")

        violation = None
        if min_val is not None and value < min_val:
            violation = "below_min"
        elif max_val is not None and value > max_val:
            violation = "above_max"

        if violation is None:
            return None

        source_str, metric_name = key.split(":", 1)
        try:
            source = MetricSource(source_str)
        except ValueError:
            source = MetricSource.CUSTOM

        # Compute z-score if we have data
        values = self._get_values(key)
        mean = self._stats_engine.mean(values) if values else value
        sd = self._stats_engine.std_deviation(values) if values else 0.0
        z = self._stats_engine.z_score(value, mean, sd) if sd > 0 else 0.0

        if violation == "below_min":
            expected = min_val
            description = (
                f"{metric_name} dropped below minimum threshold "
                f"(value={value:.2f}, min={min_val:.2f})"
            )
        else:
            expected = max_val
            description = (
                f"{metric_name} exceeded maximum threshold "
                f"(value={value:.2f}, max={max_val:.2f})"
            )

        context = {
            "threshold_config": thresh,
            "violation": violation,
        }

        return self._create_anomaly(
            anomaly_type=AnomalyType.THRESHOLD,
            source=source,
            metric_name=metric_name,
            value=value,
            expected_value=expected,
            deviation=abs(value - expected),
            z_score_val=z,
            description=description,
            context=context,
        )

    async def check_pattern_break(self, key: str, value: float) -> Optional[Anomaly]:
        """Detect deviation from day-of-week seasonal pattern.

        If seasonality has been detected for this metric, compares the
        current value against the expected value for today's day-of-week.
        """
        profile = self._profiles.get(key)
        if profile is None or not profile.seasonality_detected:
            return None

        if not profile.day_of_week_means:
            return None

        # Get today's day-of-week
        dow_idx = _now_utc().weekday()  # 0=Monday
        dow_name = DOW_NAMES[dow_idx] if dow_idx < len(DOW_NAMES) else str(dow_idx)
        expected = profile.day_of_week_means.get(dow_name)
        if expected is None or expected == 0.0:
            return None

        # Compute deviation from expected day-of-week value
        deviation_pct = abs(value - expected) / abs(expected)

        # Only flag if deviation exceeds 50% of expected
        if deviation_pct < 0.5:
            return None

        values = self._get_values(key)
        mean = self._stats_engine.mean(values) if values else expected
        sd = self._stats_engine.std_deviation(values) if values else 0.0
        z = self._stats_engine.z_score(value, mean, sd) if sd > 0 else 0.0

        source_str, metric_name = key.split(":", 1)
        try:
            source = MetricSource(source_str)
        except ValueError:
            source = MetricSource.CUSTOM

        direction = "above" if value > expected else "below"
        context = {
            "day_of_week": dow_name,
            "expected_for_day": round(expected, 4),
            "deviation_pct": round(deviation_pct * 100, 2),
            "day_of_week_means": profile.day_of_week_means,
        }

        description = (
            f"{metric_name} is {deviation_pct*100:.0f}% {direction} expected for "
            f"{dow_name} (value={value:.2f}, expected={expected:.2f})"
        )

        return self._create_anomaly(
            anomaly_type=AnomalyType.PATTERN_BREAK,
            source=source,
            metric_name=metric_name,
            value=value,
            expected_value=expected,
            deviation=abs(value - expected),
            z_score_val=z,
            description=description,
            context=context,
        )

    async def run_all_checks(self, key: str, value: float) -> List[Anomaly]:
        """Run all anomaly detection checks for a given key and value.

        Returns a list of all anomalies detected (may be empty).
        Deduplicates by type to avoid double-counting (e.g., z-score
        and IQR often overlap).
        """
        anomalies: List[Anomaly] = []
        seen_types: set = set()

        # Run all checks; collect non-None results
        checks = [
            self.check_zscore(key, value),
            self.check_iqr(key, value),
            self.check_trend_change(key),
            self.check_flatline(key),
            self.check_threshold(key, value),
            self.check_pattern_break(key, value),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning("Detection check raised exception: %s", result)
                continue
            if result is not None and result.anomaly_type not in seen_types:
                anomalies.append(result)
                seen_types.add(result.anomaly_type)

        return anomalies

    # ── Data ingestion ──

    async def ingest(
        self,
        value: float,
        source: MetricSource,
        metric_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Anomaly]:
        """Ingest a single data point and run anomaly detection.

        Returns the most severe anomaly detected, or None.
        """
        key = self._make_key(source, metric_name)

        point = DataPoint(
            timestamp=_now_iso(),
            value=value,
            source=source,
            metric_name=metric_name,
            metadata=metadata or {},
        )
        self._store_point(key, point)

        # Update stats
        self._detection_stats["total_ingested"] = self._detection_stats.get("total_ingested", 0) + 1

        # Run detection checks
        anomalies = await self.run_all_checks(key, value)

        # Auto-rebuild profile periodically (every 50 ingestions per key)
        point_count = len(self._data_points.get(key, []))
        if point_count > 0 and point_count % 50 == 0:
            self.build_profile(key)

        # Persist state
        self._save_data_points()
        self._save_anomalies()
        self._save_stats()

        if not anomalies:
            return None

        # Return the most severe anomaly
        severity_order = {
            AnomalySeverity.CRITICAL: 3,
            AnomalySeverity.WARNING: 2,
            AnomalySeverity.INFO: 1,
        }
        anomalies.sort(key=lambda a: severity_order.get(a.severity, 0), reverse=True)
        return anomalies[0]

    def ingest_sync(
        self,
        value: float,
        source: MetricSource,
        metric_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Anomaly]:
        """Synchronous wrapper for ingest."""
        return _run_sync(self.ingest(value, source, metric_name, metadata))

    async def ingest_batch(
        self,
        points: List[DataPoint],
    ) -> List[Anomaly]:
        """Ingest multiple data points and return all anomalies detected."""
        all_anomalies: List[Anomaly] = []

        for point in points:
            anomaly = await self.ingest(
                value=point.value,
                source=point.source,
                metric_name=point.metric_name,
                metadata=point.metadata,
            )
            if anomaly is not None:
                all_anomalies.append(anomaly)

        return all_anomalies

    def ingest_batch_sync(self, points: List[DataPoint]) -> List[Anomaly]:
        """Synchronous wrapper for ingest_batch."""
        return _run_sync(self.ingest_batch(points))

    # ── Profile management ──

    def build_profile(self, key: str) -> MetricProfile:
        """Compute and store a statistical profile for the given metric key.

        The profile includes mean, std dev, min, max, median, trend
        information, and day-of-week seasonality analysis.
        """
        values = self._get_values(key)

        source_str, metric_name = key.split(":", 1)
        try:
            source = MetricSource(source_str)
        except ValueError:
            source = MetricSource.CUSTOM

        if not values:
            profile = MetricProfile(
                source=source,
                metric_name=metric_name,
                mean=0.0,
                std_dev=0.0,
                min_val=0.0,
                max_val=0.0,
                median=0.0,
                sample_count=0,
                last_updated=_now_iso(),
                trend_direction="stable",
                trend_slope=0.0,
                seasonality_detected=False,
                day_of_week_means={},
            )
            self._profiles[key] = profile
            self._save_profiles()
            return profile

        mean = self._stats_engine.mean(values)
        sd = self._stats_engine.std_deviation(values)
        med = self._stats_engine.median(values)
        min_val = min(values)
        max_val = max(values)

        # Trend detection
        trend_dir, trend_slope = self._stats_engine.detect_trend(
            values, window=DEFAULT_TREND_WINDOW
        )

        # Seasonality detection
        is_seasonal, dow_means = self._stats_engine.detect_seasonality(
            values, period=DEFAULT_SEASONALITY_PERIOD
        )

        profile = MetricProfile(
            source=source,
            metric_name=metric_name,
            mean=round(mean, 4),
            std_dev=round(sd, 4),
            min_val=round(min_val, 4),
            max_val=round(max_val, 4),
            median=round(med, 4),
            sample_count=len(values),
            last_updated=_now_iso(),
            trend_direction=trend_dir,
            trend_slope=round(trend_slope, 6),
            seasonality_detected=is_seasonal,
            day_of_week_means=dow_means,
        )

        self._profiles[key] = profile
        self._save_profiles()

        logger.debug(
            "Built profile for %s: mean=%.2f sd=%.2f trend=%s seasonal=%s (%d samples)",
            key, mean, sd, trend_dir, is_seasonal, len(values),
        )

        return profile

    def get_profile(self, key: str) -> Optional[MetricProfile]:
        """Return the stored profile for a metric key, or None."""
        return self._profiles.get(key)

    def rebuild_all_profiles(self) -> int:
        """Rebuild profiles for all tracked metrics.

        Returns the number of profiles rebuilt.
        """
        count = 0
        for key in list(self._data_points.keys()):
            self.build_profile(key)
            count += 1
        logger.info("Rebuilt %d metric profiles", count)
        return count

    # ── Threshold configuration ──

    def set_threshold(
        self,
        source: MetricSource,
        metric_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        z_score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Configure a custom threshold for a metric.

        Returns the stored threshold configuration.
        """
        key = self._make_key(source, metric_name)
        config: Dict[str, Any] = self._thresholds.get(key, {})

        if min_val is not None:
            config["min_val"] = min_val
        if max_val is not None:
            config["max_val"] = max_val
        if z_score_threshold is not None:
            config["z_score_threshold"] = z_score_threshold

        config["updated_at"] = _now_iso()
        self._thresholds[key] = config
        self._save_thresholds()

        logger.info("Set threshold for %s: %s", key, config)
        return config

    def remove_threshold(self, source: MetricSource, metric_name: str) -> bool:
        """Remove a configured threshold. Returns True if one existed."""
        key = self._make_key(source, metric_name)
        if key in self._thresholds:
            del self._thresholds[key]
            self._save_thresholds()
            return True
        return False

    def get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Return all configured thresholds."""
        return copy.deepcopy(self._thresholds)

    # ── Anomaly management ──

    def get_anomalies(
        self,
        source: Optional[MetricSource] = None,
        severity: Optional[AnomalySeverity] = None,
        anomaly_type: Optional[AnomalyType] = None,
        acknowledged: Optional[bool] = None,
        resolved: Optional[bool] = None,
        limit: int = 50,
    ) -> List[Anomaly]:
        """Query anomalies with optional filters.

        Results are returned in reverse chronological order (most recent first).
        """
        results = self._anomalies[:]

        if source is not None:
            results = [a for a in results if a.source == source]
        if severity is not None:
            results = [a for a in results if a.severity == severity]
        if anomaly_type is not None:
            results = [a for a in results if a.anomaly_type == anomaly_type]
        if acknowledged is not None:
            results = [a for a in results if a.acknowledged == acknowledged]
        if resolved is not None:
            results = [a for a in results if a.resolved == resolved]

        # Reverse chronological
        results.sort(key=lambda a: a.detected_at, reverse=True)

        return results[:limit]

    def acknowledge(self, anomaly_id: str) -> bool:
        """Acknowledge an anomaly by ID. Returns True if found."""
        for anomaly in self._anomalies:
            if anomaly.anomaly_id == anomaly_id:
                anomaly.acknowledged = True
                self._save_anomalies()
                logger.info("Acknowledged anomaly %s", anomaly_id)
                return True
        return False

    def resolve(self, anomaly_id: str) -> bool:
        """Resolve an anomaly by ID. Returns True if found."""
        for anomaly in self._anomalies:
            if anomaly.anomaly_id == anomaly_id:
                anomaly.resolved = True
                anomaly.acknowledged = True  # auto-acknowledge on resolve
                self._save_anomalies()
                logger.info("Resolved anomaly %s", anomaly_id)
                return True
        return False

    def get_recent(self, hours: int = 24) -> List[Anomaly]:
        """Return anomalies from the last N hours."""
        cutoff = _parse_iso(_hours_ago(hours))
        results: List[Anomaly] = []
        for anomaly in self._anomalies:
            dt = _parse_iso(anomaly.detected_at)
            if dt is not None and cutoff is not None and dt >= cutoff:
                results.append(anomaly)
        results.sort(key=lambda a: a.detected_at, reverse=True)
        return results

    def clear_old_anomalies(self, days: int = MAX_ANOMALY_AGE_DAYS) -> int:
        """Remove anomalies older than *days*. Returns count removed."""
        cutoff = _parse_iso(_days_ago(days))
        before = len(self._anomalies)
        self._anomalies = [
            a for a in self._anomalies
            if _parse_iso(a.detected_at) is not None
            and cutoff is not None
            and _parse_iso(a.detected_at) >= cutoff
        ]
        removed = before - len(self._anomalies)
        if removed > 0:
            self._save_anomalies()
            logger.info("Cleared %d old anomalies (>%d days)", removed, days)
        return removed

    # ── Analytics ──

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate detection statistics."""
        active_anomalies = [a for a in self._anomalies if not a.resolved]
        unacked = [a for a in active_anomalies if not a.acknowledged]

        return {
            "total_ingested": self._detection_stats.get("total_ingested", 0),
            "total_anomalies_detected": self._detection_stats.get("total_anomalies", 0),
            "active_anomalies": len(active_anomalies),
            "unacknowledged": len(unacked),
            "total_stored_anomalies": len(self._anomalies),
            "tracked_metrics": len(self._data_points),
            "profiles_built": len(self._profiles),
            "thresholds_configured": len(self._thresholds),
            "by_type": self._detection_stats.get("by_type", {}),
            "by_severity": self._detection_stats.get("by_severity", {}),
            "by_source": self._detection_stats.get("by_source", {}),
        }

    def get_anomaly_summary(self, days: int = 7) -> Dict[str, Any]:
        """Generate an anomaly summary for the last N days.

        Breaks down anomalies by type, severity, source, and metric.
        """
        cutoff = _parse_iso(_days_ago(days))
        recent = [
            a for a in self._anomalies
            if _parse_iso(a.detected_at) is not None
            and cutoff is not None
            and _parse_iso(a.detected_at) >= cutoff
        ]

        # By type
        by_type: Dict[str, int] = {}
        for a in recent:
            t = a.anomaly_type.value
            by_type[t] = by_type.get(t, 0) + 1

        # By severity
        by_severity: Dict[str, int] = {}
        for a in recent:
            s = a.severity.value
            by_severity[s] = by_severity.get(s, 0) + 1

        # By source
        by_source: Dict[str, int] = {}
        for a in recent:
            src = a.source.value
            by_source[src] = by_source.get(src, 0) + 1

        # By metric (top 10)
        by_metric: Dict[str, int] = {}
        for a in recent:
            key = f"{a.source.value}:{a.metric_name}"
            by_metric[key] = by_metric.get(key, 0) + 1
        top_metrics = dict(
            sorted(by_metric.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Resolved vs active
        resolved_count = sum(1 for a in recent if a.resolved)
        acked_count = sum(1 for a in recent if a.acknowledged and not a.resolved)
        active_count = len(recent) - resolved_count

        return {
            "period_days": days,
            "total": len(recent),
            "active": active_count,
            "acknowledged": acked_count,
            "resolved": resolved_count,
            "by_type": by_type,
            "by_severity": by_severity,
            "by_source": by_source,
            "top_metrics": top_metrics,
        }

    def get_metric_health(self) -> Dict[str, Any]:
        """Generate a health dashboard for all tracked metrics.

        Returns a dictionary keyed by metric key, each containing:
        - current value, profile stats, recent anomaly count, status
        """
        health: Dict[str, Any] = {}

        for key, points in self._data_points.items():
            if not points:
                continue

            current_value = points[-1].value
            profile = self._profiles.get(key)

            # Count recent anomalies for this metric (last 24h)
            cutoff = _parse_iso(_hours_ago(24))
            recent_anomaly_count = 0
            for a in self._anomalies:
                akey = self._make_key(a.source, a.metric_name)
                if akey == key:
                    dt = _parse_iso(a.detected_at)
                    if dt is not None and cutoff is not None and dt >= cutoff:
                        recent_anomaly_count += 1

            # Determine status
            if recent_anomaly_count == 0:
                status = "healthy"
            elif recent_anomaly_count <= 2:
                status = "warning"
            else:
                status = "critical"

            # Z-score relative to profile
            current_z = 0.0
            if profile and profile.std_dev > 0:
                current_z = self._stats_engine.z_score(
                    current_value, profile.mean, profile.std_dev
                )

            entry: Dict[str, Any] = {
                "current_value": round(current_value, 4),
                "current_z_score": round(current_z, 4),
                "data_points": len(points),
                "recent_anomalies_24h": recent_anomaly_count,
                "status": status,
                "last_updated": points[-1].timestamp,
            }

            if profile:
                entry["profile"] = {
                    "mean": profile.mean,
                    "std_dev": profile.std_dev,
                    "min": profile.min_val,
                    "max": profile.max_val,
                    "median": profile.median,
                    "trend": profile.trend_direction,
                    "seasonal": profile.seasonality_detected,
                }

            # Include threshold if configured
            thresh = self._thresholds.get(key)
            if thresh:
                entry["threshold"] = thresh

            health[key] = entry

        return health


# ===================================================================
# SINGLETON
# ===================================================================

_detector: Optional[AnomalyDetector] = None


def get_detector() -> AnomalyDetector:
    """Return the global AnomalyDetector singleton, creating it on first call."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def ingest(
    value: float,
    source: MetricSource,
    metric_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Anomaly]:
    """Convenience: ingest a data point via the singleton detector (sync)."""
    return get_detector().ingest_sync(value, source, metric_name, metadata)


def recent_anomalies(hours: int = 24) -> List[Anomaly]:
    """Convenience: get recent anomalies from the singleton."""
    return get_detector().get_recent(hours)


def health() -> Dict[str, Any]:
    """Convenience: get metric health from the singleton."""
    return get_detector().get_metric_health()


# ===================================================================
# CLI COMMAND HANDLERS
# ===================================================================

def _cmd_status(args: argparse.Namespace) -> None:
    """Show overview of all tracked metrics."""
    detector = get_detector()
    stats = detector.get_stats()

    print("\n=== Anomaly Detector Status ===\n")
    print(f"  Total Ingested:         {stats['total_ingested']:,}")
    print(f"  Total Anomalies Found:  {stats['total_anomalies_detected']:,}")
    print(f"  Active (unresolved):    {stats['active_anomalies']:,}")
    print(f"  Unacknowledged:         {stats['unacknowledged']:,}")
    print(f"  Tracked Metrics:        {stats['tracked_metrics']:,}")
    print(f"  Profiles Built:         {stats['profiles_built']:,}")
    print(f"  Thresholds Configured:  {stats['thresholds_configured']:,}")

    by_type = stats.get("by_type", {})
    if by_type:
        print("\n  Anomaly Distribution (by type):")
        for atype, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * min(count, 40)
            print(f"    {atype:<18} {count:>5}  {bar}")

    by_severity = stats.get("by_severity", {})
    if by_severity:
        print("\n  Anomaly Distribution (by severity):")
        for sev, count in sorted(by_severity.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * min(count, 40)
            print(f"    {sev:<12} {count:>5}  {bar}")

    by_source = stats.get("by_source", {})
    if by_source:
        print("\n  Anomaly Distribution (by source):")
        for src, count in sorted(by_source.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * min(count, 40)
            print(f"    {src:<12} {count:>5}  {bar}")

    print()


def _cmd_anomalies(args: argparse.Namespace) -> None:
    """List recent anomalies with optional filters."""
    detector = get_detector()

    severity_filter = None
    if args.severity:
        try:
            severity_filter = AnomalySeverity(args.severity)
        except ValueError:
            print(f"Invalid severity: {args.severity}")
            sys.exit(1)

    source_filter = None
    if args.source:
        try:
            source_filter = MetricSource(args.source)
        except ValueError:
            print(f"Invalid source: {args.source}")
            sys.exit(1)

    type_filter = None
    if args.type:
        try:
            type_filter = AnomalyType(args.type)
        except ValueError:
            print(f"Invalid type: {args.type}")
            sys.exit(1)

    anomalies = detector.get_anomalies(
        source=source_filter,
        severity=severity_filter,
        anomaly_type=type_filter,
        acknowledged=args.acknowledged if args.acknowledged is not None else None,
        limit=args.limit,
    )

    if not anomalies:
        print("No anomalies found matching filters.")
        return

    print(f"\n{'ID (short)':<12} {'Severity':<10} {'Type':<16} {'Source':<10} {'Metric':<25} {'Value':>10} {'Expected':>10} {'Z-Score':>8}")
    print("-" * 105)

    for a in anomalies:
        short_id = a.anomaly_id[:8]
        flags = ""
        if a.resolved:
            flags = " [R]"
        elif a.acknowledged:
            flags = " [A]"

        print(
            f"  {short_id:<10} {a.severity.value:<10} {a.anomaly_type.value:<16} "
            f"{a.source.value:<10} {a.metric_name:<25} {a.value:>10.2f} "
            f"{a.expected_value:>10.2f} {a.z_score:>8.2f}{flags}"
        )

    print(f"\nShowing {len(anomalies)} anomalies. [A]=Acknowledged [R]=Resolved\n")

    if args.verbose:
        print("--- Details ---\n")
        for a in anomalies:
            print(f"  [{a.anomaly_id[:8]}] {a.description}")
            print(f"    Detected: {a.detected_at}")
            if a.context:
                for ck, cv in a.context.items():
                    print(f"    {ck}: {cv}")
            print()


def _cmd_summary(args: argparse.Namespace) -> None:
    """Show anomaly summary for the last N days."""
    detector = get_detector()
    summary = detector.get_anomaly_summary(days=args.days)

    print(f"\n=== Anomaly Summary (Last {summary['period_days']} Days) ===\n")
    print(f"  Total Anomalies:   {summary['total']}")
    print(f"  Active:            {summary['active']}")
    print(f"  Acknowledged:      {summary['acknowledged']}")
    print(f"  Resolved:          {summary['resolved']}")

    by_type = summary.get("by_type", {})
    if by_type:
        print("\n  By Type:")
        for t, c in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            print(f"    {t:<18} {c}")

    by_severity = summary.get("by_severity", {})
    if by_severity:
        print("\n  By Severity:")
        for s, c in sorted(by_severity.items(), key=lambda x: x[1], reverse=True):
            print(f"    {s:<12} {c}")

    by_source = summary.get("by_source", {})
    if by_source:
        print("\n  By Source:")
        for src, c in sorted(by_source.items(), key=lambda x: x[1], reverse=True):
            print(f"    {src:<12} {c}")

    top = summary.get("top_metrics", {})
    if top:
        print("\n  Top Offending Metrics:")
        for metric, c in top.items():
            print(f"    {metric:<40} {c}")

    print()


def _cmd_profiles(args: argparse.Namespace) -> None:
    """Show metric profiles."""
    detector = get_detector()

    if not detector._profiles:
        print("No metric profiles built yet. Ingest data first, or run 'rebuild'.")
        return

    print(f"\n{'Key':<40} {'Mean':>10} {'StdDev':>10} {'Min':>10} {'Max':>10} {'Trend':<8} {'Seasonal':<9} {'Samples':>8}")
    print("-" * 115)

    for key, p in sorted(detector._profiles.items()):
        seasonal = "Yes" if p.seasonality_detected else "No"
        print(
            f"  {key:<38} {p.mean:>10.2f} {p.std_dev:>10.2f} "
            f"{p.min_val:>10.2f} {p.max_val:>10.2f} {p.trend_direction:<8} "
            f"{seasonal:<9} {p.sample_count:>8}"
        )

    print(f"\n{len(detector._profiles)} profiles total.\n")

    if args.verbose:
        for key, p in sorted(detector._profiles.items()):
            if p.day_of_week_means:
                print(f"  {key} day-of-week means:")
                for dow, avg in p.day_of_week_means.items():
                    print(f"    {dow}: {avg:.2f}")
                print()


def _cmd_health(args: argparse.Namespace) -> None:
    """Show metric health dashboard."""
    detector = get_detector()
    health_data = detector.get_metric_health()

    if not health_data:
        print("No metrics tracked yet.")
        return

    # Sort by status: critical first, then warning, then healthy
    status_order = {"critical": 0, "warning": 1, "healthy": 2}

    print(f"\n{'Metric':<40} {'Value':>10} {'Z-Score':>8} {'Points':>7} {'24h Anom':>9} {'Status':<10}")
    print("-" * 90)

    for key in sorted(health_data.keys(), key=lambda k: status_order.get(health_data[k]["status"], 9)):
        entry = health_data[key]
        status = entry["status"]

        # Status indicator
        if status == "critical":
            marker = "!!"
        elif status == "warning":
            marker = "! "
        else:
            marker = "  "

        print(
            f"{marker}{key:<38} {entry['current_value']:>10.2f} "
            f"{entry['current_z_score']:>8.2f} {entry['data_points']:>7} "
            f"{entry['recent_anomalies_24h']:>9} {status:<10}"
        )

    # Summary counts
    statuses = [e["status"] for e in health_data.values()]
    print(f"\n  Healthy: {statuses.count('healthy')}  |  "
          f"Warning: {statuses.count('warning')}  |  "
          f"Critical: {statuses.count('critical')}\n")


def _cmd_thresholds(args: argparse.Namespace) -> None:
    """Show configured thresholds."""
    detector = get_detector()
    thresholds = detector.get_thresholds()

    if not thresholds:
        print("No thresholds configured.")
        return

    print(f"\n{'Metric Key':<40} {'Min':>10} {'Max':>10} {'Z-Thresh':>10} {'Updated':<25}")
    print("-" * 100)

    for key, config in sorted(thresholds.items()):
        min_val = config.get("min_val")
        max_val = config.get("max_val")
        z_thresh = config.get("z_score_threshold")
        updated = config.get("updated_at", "?")

        min_str = f"{min_val:.2f}" if min_val is not None else "-"
        max_str = f"{max_val:.2f}" if max_val is not None else "-"
        z_str = f"{z_thresh:.2f}" if z_thresh is not None else "-"

        print(f"  {key:<38} {min_str:>10} {max_str:>10} {z_str:>10} {updated:<25}")

    print(f"\n{len(thresholds)} thresholds configured.\n")


def _cmd_set_threshold(args: argparse.Namespace) -> None:
    """Set a threshold for a metric."""
    detector = get_detector()

    try:
        source = MetricSource(args.source)
    except ValueError:
        print(f"Invalid source: {args.source}")
        sys.exit(1)

    config = detector.set_threshold(
        source=source,
        metric_name=args.metric,
        min_val=args.min,
        max_val=args.max,
        z_score_threshold=args.zscore,
    )

    key = AnomalyDetector._make_key(source, args.metric)
    print(f"Threshold set for {key}:")
    for k, v in config.items():
        print(f"  {k}: {v}")


def _cmd_acknowledge(args: argparse.Namespace) -> None:
    """Acknowledge an anomaly by ID (or prefix)."""
    detector = get_detector()

    # Support prefix matching
    target_id = args.anomaly_id
    matched = None
    for a in detector._anomalies:
        if a.anomaly_id == target_id or a.anomaly_id.startswith(target_id):
            matched = a
            break

    if matched is None:
        print(f"Anomaly not found: {target_id}")
        sys.exit(1)

    if detector.acknowledge(matched.anomaly_id):
        print(f"Acknowledged: {matched.anomaly_id[:8]} ({matched.anomaly_type.value} on {matched.metric_name})")
    else:
        print("Failed to acknowledge anomaly.")


def _cmd_resolve(args: argparse.Namespace) -> None:
    """Resolve an anomaly by ID (or prefix)."""
    detector = get_detector()

    target_id = args.anomaly_id
    matched = None
    for a in detector._anomalies:
        if a.anomaly_id == target_id or a.anomaly_id.startswith(target_id):
            matched = a
            break

    if matched is None:
        print(f"Anomaly not found: {target_id}")
        sys.exit(1)

    if detector.resolve(matched.anomaly_id):
        print(f"Resolved: {matched.anomaly_id[:8]} ({matched.anomaly_type.value} on {matched.metric_name})")
    else:
        print("Failed to resolve anomaly.")


def _cmd_ingest(args: argparse.Namespace) -> None:
    """Manually ingest a data point from CLI."""
    detector = get_detector()

    try:
        source = MetricSource(args.source)
    except ValueError:
        print(f"Invalid source: {args.source}")
        sys.exit(1)

    anomaly = detector.ingest_sync(
        value=args.value,
        source=source,
        metric_name=args.metric,
        metadata={"cli_injected": True},
    )

    key = AnomalyDetector._make_key(source, args.metric)
    point_count = len(detector._data_points.get(key, []))
    print(f"Ingested: {args.source}:{args.metric} = {args.value} (total points: {point_count})")

    if anomaly:
        print(f"  ANOMALY DETECTED: [{anomaly.severity.value.upper()}] {anomaly.description}")
    else:
        print("  No anomaly detected.")


def _cmd_rebuild(args: argparse.Namespace) -> None:
    """Rebuild all metric profiles."""
    detector = get_detector()
    count = detector.rebuild_all_profiles()
    print(f"Rebuilt {count} metric profiles.")


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show aggregate statistics (alias for status with more detail)."""
    detector = get_detector()
    stats = detector.get_stats()

    print("\n=== Anomaly Detection Statistics ===\n")
    print(f"  Data points ingested:    {stats['total_ingested']:,}")
    print(f"  Anomalies detected:      {stats['total_anomalies_detected']:,}")
    print(f"  Active anomalies:        {stats['active_anomalies']:,}")
    print(f"  Unacknowledged:          {stats['unacknowledged']:,}")
    print(f"  Stored anomalies:        {stats['total_stored_anomalies']:,} (max {MAX_ANOMALIES})")
    print(f"  Tracked metric keys:     {stats['tracked_metrics']:,}")
    print(f"  Profiles built:          {stats['profiles_built']:,}")
    print(f"  Thresholds configured:   {stats['thresholds_configured']:,}")

    # Detection rate
    total_in = stats["total_ingested"]
    total_anom = stats["total_anomalies_detected"]
    if total_in > 0:
        rate = (total_anom / total_in) * 100
        print(f"  Detection rate:          {rate:.2f}%")

    # Per-metric data point counts
    if args.verbose:
        print("\n  Data Points Per Metric:")
        for key, points in sorted(
            detector._data_points.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        ):
            print(f"    {key:<40} {len(points):>6} points")

    print()


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def main() -> None:
    """CLI entry point for the anomaly detector module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="anomaly_detector",
        description="OpenClaw Empire Anomaly Detector -- Statistical anomaly detection CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- status ---
    p_status = subparsers.add_parser("status", help="Overview of all tracked metrics")
    p_status.set_defaults(func=_cmd_status)

    # --- anomalies ---
    p_anom = subparsers.add_parser("anomalies", help="List recent anomalies")
    p_anom.add_argument("--severity", choices=["info", "warning", "critical"],
                        help="Filter by severity")
    p_anom.add_argument("--source", help="Filter by source (e.g., revenue, traffic)")
    p_anom.add_argument("--type", help="Filter by anomaly type (e.g., spike, drop)")
    p_anom.add_argument("--acknowledged", type=bool, default=None,
                        help="Filter by acknowledged status")
    p_anom.add_argument("--limit", type=int, default=50, help="Max results (default: 50)")
    p_anom.add_argument("-v", "--verbose", action="store_true",
                        help="Show anomaly details and context")
    p_anom.set_defaults(func=_cmd_anomalies)

    # --- summary ---
    p_sum = subparsers.add_parser("summary", help="Anomaly summary for N days")
    p_sum.add_argument("--days", type=int, default=7, help="Number of days (default: 7)")
    p_sum.set_defaults(func=_cmd_summary)

    # --- profiles ---
    p_prof = subparsers.add_parser("profiles", help="Show metric profiles")
    p_prof.add_argument("-v", "--verbose", action="store_true",
                        help="Show day-of-week means")
    p_prof.set_defaults(func=_cmd_profiles)

    # --- health ---
    p_health = subparsers.add_parser("health", help="Metric health dashboard")
    p_health.set_defaults(func=_cmd_health)

    # --- thresholds ---
    p_thresh = subparsers.add_parser("thresholds", help="Show configured thresholds")
    p_thresh.set_defaults(func=_cmd_thresholds)

    # --- set-threshold ---
    p_set = subparsers.add_parser("set-threshold", help="Set a threshold for a metric")
    p_set.add_argument("--source", required=True, help="Metric source (e.g., revenue)")
    p_set.add_argument("--metric", required=True, help="Metric name (e.g., daily_ads)")
    p_set.add_argument("--min", type=float, default=None, help="Minimum value threshold")
    p_set.add_argument("--max", type=float, default=None, help="Maximum value threshold")
    p_set.add_argument("--zscore", type=float, default=None, help="Z-score threshold")
    p_set.set_defaults(func=_cmd_set_threshold)

    # --- acknowledge ---
    p_ack = subparsers.add_parser("acknowledge", help="Acknowledge an anomaly")
    p_ack.add_argument("anomaly_id", help="Anomaly ID (or prefix)")
    p_ack.set_defaults(func=_cmd_acknowledge)

    # --- resolve ---
    p_res = subparsers.add_parser("resolve", help="Resolve an anomaly")
    p_res.add_argument("anomaly_id", help="Anomaly ID (or prefix)")
    p_res.set_defaults(func=_cmd_resolve)

    # --- ingest ---
    p_ing = subparsers.add_parser("ingest", help="Manually ingest a data point")
    p_ing.add_argument("--source", required=True, help="Metric source")
    p_ing.add_argument("--metric", required=True, help="Metric name")
    p_ing.add_argument("--value", type=float, required=True, help="Metric value")
    p_ing.set_defaults(func=_cmd_ingest)

    # --- rebuild ---
    p_rebuild = subparsers.add_parser("rebuild", help="Rebuild all metric profiles")
    p_rebuild.set_defaults(func=_cmd_rebuild)

    # --- stats ---
    p_stats = subparsers.add_parser("stats", help="Aggregate detection statistics")
    p_stats.add_argument("-v", "--verbose", action="store_true",
                         help="Show per-metric data point counts")
    p_stats.set_defaults(func=_cmd_stats)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
