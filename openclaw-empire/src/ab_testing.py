"""
A/B Testing Engine -- OpenClaw Empire Edition
==============================================

Headline, CTA, layout, image, subject-line, send-time, content-length, and
pricing experiments with pure-Python statistical significance testing and
automatic conclusion for Nick Creighton's 16-site WordPress publishing empire.

Statistical methods (all pure Python, no scipy/numpy):
    TWO-PROPORTION Z-TEST  -- Compare conversion rates between variants
    NORMAL CDF             -- Abramowitz & Stegun approximation
    CONFIDENCE INTERVALS   -- Wilson score intervals for proportions
    LIFT CALCULATION       -- Relative improvement over control
    SAMPLE SIZE            -- Required observations for desired power

Features:
    - Create multi-variant experiments (A/B/C/n) with traffic weighting
    - Consistent visitor-to-variant assignment via deterministic hashing
    - Record impressions, conversions, and custom events per variant
    - Auto-conclude experiments when significance thresholds are met
    - Background checker with configurable polling interval
    - Full experiment lifecycle: draft -> running -> paused -> concluded
    - Site-level experiment listing and cross-site reporting
    - JSON-persisted state with atomic writes

All data persisted to: data/ab_testing/

Usage:
    from src.ab_testing import get_engine, ExperimentType, MetricType

    engine = get_engine()

    # Create an experiment
    exp = await engine.create_experiment(
        name="Homepage Headline Test",
        experiment_type=ExperimentType.HEADLINE,
        site_id="witchcraft",
        page_url="https://witchcraftforbeginners.com/",
        primary_metric=MetricType.CLICK_RATE,
        variants=[
            {"name": "Control", "content": "Begin Your Magical Journey",
             "is_control": True},
            {"name": "Variant B", "content": "Unlock Ancient Witchcraft Secrets"},
        ],
    )

    # Record events
    variant_id = await engine.record_impression(exp.experiment_id, "visitor-123")
    await engine.record_conversion(exp.experiment_id, "visitor-123", value=1.0)

    # Check results
    result = await engine.calculate_results(exp.experiment_id)
    print(f"Winner: {result.winner}, p={result.p_value:.4f}, lift={result.lift:.1f}%")

    # Synchronous
    engine.record_impression_sync(exp.experiment_id, "visitor-456")

CLI:
    python -m src.ab_testing create --name "CTA Test" --type cta --site witchcraft --metric click_rate
    python -m src.ab_testing start EXPERIMENT_ID
    python -m src.ab_testing pause EXPERIMENT_ID
    python -m src.ab_testing resume EXPERIMENT_ID
    python -m src.ab_testing conclude EXPERIMENT_ID
    python -m src.ab_testing cancel EXPERIMENT_ID
    python -m src.ab_testing record --experiment EXPERIMENT_ID --visitor VIS_ID --event impression
    python -m src.ab_testing record --experiment EXPERIMENT_ID --visitor VIS_ID --event conversion --value 1.0
    python -m src.ab_testing results EXPERIMENT_ID
    python -m src.ab_testing list [--site SITE_ID] [--status running]
    python -m src.ab_testing stats [--site SITE_ID]
    python -m src.ab_testing sample-size --baseline 0.05 --mde 0.02 [--alpha 0.05] [--power 0.80]
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import hashlib
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ab_testing")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "ab_testing"
EXPERIMENTS_FILE = DATA_DIR / "experiments.json"
EVENTS_FILE = DATA_DIR / "events.json"
ASSIGNMENTS_FILE = DATA_DIR / "assignments.json"
RESULTS_FILE = DATA_DIR / "results.json"
STATS_FILE = DATA_DIR / "stats.json"

# Ensure data directory exists on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EXPERIMENTS = 500
MAX_EVENTS_PER_EXPERIMENT = 50_000
MAX_ASSIGNMENTS_PER_EXPERIMENT = 100_000
MAX_VARIANTS_PER_EXPERIMENT = 10
MAX_EXPERIMENT_AGE_DAYS = 365

DEFAULT_MIN_SAMPLE_SIZE = 100
DEFAULT_MAX_DURATION_DAYS = 30
DEFAULT_CONFIDENCE_THRESHOLD = 0.95
DEFAULT_AUTO_CHECK_INTERVAL = 3600  # seconds

# Weight tolerance for traffic allocation
WEIGHT_TOLERANCE = 0.001

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
        return copy.deepcopy(default)


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
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


# ===========================================================================
# Enums
# ===========================================================================


class ExperimentType(str, Enum):
    """Type of A/B test experiment."""
    HEADLINE = "headline"
    CTA = "cta"
    LAYOUT = "layout"
    IMAGE = "image"
    SUBJECT_LINE = "subject_line"
    SEND_TIME = "send_time"
    CONTENT_LENGTH = "content_length"
    PRICING = "pricing"


class ExperimentStatus(str, Enum):
    """Lifecycle status of an experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    CONCLUDED = "concluded"
    CANCELLED = "cancelled"


class MetricType(str, Enum):
    """Primary or secondary metric to optimise."""
    CLICK_RATE = "click_rate"
    CONVERSION_RATE = "conversion_rate"
    BOUNCE_RATE = "bounce_rate"
    TIME_ON_PAGE = "time_on_page"
    REVENUE = "revenue"
    OPEN_RATE = "open_rate"
    ENGAGEMENT_SCORE = "engagement_score"
    SUBSCRIBE_RATE = "subscribe_rate"


class SignificanceLevel(str, Enum):
    """Qualitative significance buckets (p-value thresholds)."""
    LOW = "low"            # p < 0.10
    MEDIUM = "medium"      # p < 0.05
    HIGH = "high"          # p < 0.01
    VERY_HIGH = "very_high"  # p < 0.001


# Mapping from p-value to significance level
_P_THRESHOLDS: List[Tuple[float, SignificanceLevel]] = [
    (0.001, SignificanceLevel.VERY_HIGH),
    (0.01, SignificanceLevel.HIGH),
    (0.05, SignificanceLevel.MEDIUM),
    (0.10, SignificanceLevel.LOW),
]


def _p_to_significance(p: float) -> Optional[SignificanceLevel]:
    """Map a p-value to the strongest significance level it satisfies."""
    for threshold, level in _P_THRESHOLDS:
        if p < threshold:
            return level
    return None


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class Variant:
    """A single variant (arm) within an experiment."""
    variant_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    description: str = ""
    content: str = ""
    traffic_weight: float = 0.5
    impressions: int = 0
    conversions: int = 0
    total_value: float = 0.0
    is_control: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def conversion_rate(self) -> float:
        """Return conversion rate; 0.0 if no impressions."""
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions

    @property
    def avg_value(self) -> float:
        """Average value per conversion; 0.0 if no conversions."""
        if self.conversions == 0:
            return 0.0
        return self.total_value / self.conversions

    @property
    def revenue_per_impression(self) -> float:
        """Revenue per impression (RPM-style)."""
        if self.impressions == 0:
            return 0.0
        return self.total_value / self.impressions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Variant:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Experiment:
    """Full experiment definition, state, and outcomes."""
    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = ""
    description: str = ""
    type: ExperimentType = ExperimentType.HEADLINE
    status: ExperimentStatus = ExperimentStatus.DRAFT
    site_id: str = ""
    page_url: str = ""
    primary_metric: MetricType = MetricType.CLICK_RATE
    secondary_metrics: List[str] = field(default_factory=list)
    variants: List[Variant] = field(default_factory=list)
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE
    max_duration_days: int = DEFAULT_MAX_DURATION_DAYS
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    auto_conclude: bool = True
    winner_variant_id: Optional[str] = None
    significance_level: Optional[str] = None
    p_value: Optional[float] = None
    lift_percent: Optional[float] = None
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    paused_at: Optional[str] = None
    concluded_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    updated_at: str = field(default_factory=_now_iso)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_impressions(self) -> int:
        return sum(v.impressions for v in self.variants)

    @property
    def total_conversions(self) -> int:
        return sum(v.conversions for v in self.variants)

    @property
    def total_value(self) -> float:
        return sum(v.total_value for v in self.variants)

    @property
    def control(self) -> Optional[Variant]:
        """Return the control variant, or first variant if none explicitly marked."""
        for v in self.variants:
            if v.is_control:
                return v
        return self.variants[0] if self.variants else None

    @property
    def is_active(self) -> bool:
        return self.status == ExperimentStatus.RUNNING

    @property
    def duration_days(self) -> float:
        """Days since experiment started (0 if not started)."""
        if not self.started_at:
            return 0.0
        start = _parse_iso(self.started_at)
        if start is None:
            return 0.0
        delta = _now_utc() - start
        return delta.total_seconds() / 86400.0

    @property
    def has_sufficient_data(self) -> bool:
        """Check if all variants have at least min_sample_size impressions."""
        if not self.variants:
            return False
        return all(v.impressions >= self.min_sample_size for v in self.variants)

    @property
    def is_expired(self) -> bool:
        """Check if experiment has exceeded max duration."""
        return self.duration_days > self.max_duration_days

    def get_variant(self, variant_id: str) -> Optional[Variant]:
        """Look up a variant by ID."""
        for v in self.variants:
            if v.variant_id == variant_id:
                return v
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["status"] = self.status.value
        d["primary_metric"] = self.primary_metric.value
        d["variants"] = [v.to_dict() for v in self.variants]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Experiment:
        d = dict(d)
        # Parse enums
        if "type" in d and isinstance(d["type"], str):
            try:
                d["type"] = ExperimentType(d["type"])
            except ValueError:
                d["type"] = ExperimentType.HEADLINE
        if "status" in d and isinstance(d["status"], str):
            try:
                d["status"] = ExperimentStatus(d["status"])
            except ValueError:
                d["status"] = ExperimentStatus.DRAFT
        if "primary_metric" in d and isinstance(d["primary_metric"], str):
            try:
                d["primary_metric"] = MetricType(d["primary_metric"])
            except ValueError:
                d["primary_metric"] = MetricType.CLICK_RATE
        # Parse variants
        raw_variants = d.pop("variants", [])
        variants = []
        for rv in raw_variants:
            if isinstance(rv, dict):
                variants.append(Variant.from_dict(rv))
            elif isinstance(rv, Variant):
                variants.append(rv)
        d["variants"] = variants
        # Filter to known fields
        known = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class ExperimentEvent:
    """A single recorded event (impression, conversion, or custom)."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    experiment_id: str = ""
    variant_id: str = ""
    event_type: str = "impression"  # impression, conversion, click, bounce, engage, custom
    value: float = 0.0
    visitor_id: str = ""
    timestamp: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExperimentEvent:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class VariantResult:
    """Statistical result for a single variant within an experiment."""
    variant_id: str = ""
    name: str = ""
    is_control: bool = False
    impressions: int = 0
    conversions: int = 0
    conversion_rate: float = 0.0
    total_value: float = 0.0
    revenue_per_impression: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    lift_vs_control: float = 0.0
    z_score: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    significance_level: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence_interval"] = list(self.confidence_interval)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VariantResult:
        d = dict(d)
        if "confidence_interval" in d and isinstance(d["confidence_interval"], list):
            d["confidence_interval"] = tuple(d["confidence_interval"])
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ExperimentResult:
    """Aggregated statistical result for an entire experiment."""
    experiment_id: str = ""
    variant_results: List[VariantResult] = field(default_factory=list)
    winner: Optional[str] = None
    confidence: float = 0.0
    p_value: float = 1.0
    lift: float = 0.0
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    recommendation: str = ""
    calculated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["variant_results"] = [vr.to_dict() for vr in self.variant_results]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExperimentResult:
        d = dict(d)
        raw_vr = d.pop("variant_results", [])
        vr_list = []
        for rv in raw_vr:
            if isinstance(rv, dict):
                vr_list.append(VariantResult.from_dict(rv))
            elif isinstance(rv, VariantResult):
                vr_list.append(rv)
        d["variant_results"] = vr_list
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


# ===========================================================================
# Pure-Python Statistical Functions
# ===========================================================================


def _normal_cdf(x: float) -> float:
    """
    Cumulative distribution function for the standard normal distribution.

    Uses the Abramowitz & Stegun approximation (formula 26.2.17) which
    provides accuracy to ~1.5e-7. This avoids any dependency on scipy.
    """
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0

    # Handle negative x via symmetry
    sign = 1.0
    if x < 0:
        sign = -1.0
        x = -x

    # Constants for the rational approximation
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    p = 0.2316419

    t = 1.0 / (1.0 + p * x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    # Standard normal PDF at x
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    # Approximation of the CDF
    cdf = 1.0 - pdf * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5)

    if sign < 0:
        cdf = 1.0 - cdf

    return cdf


def _normal_ppf(p: float) -> float:
    """
    Percent-point function (inverse CDF) of the standard normal.

    Uses the Beasley-Springer-Moro algorithm for the rational approximation.
    Accurate to about 1e-9 for p in (0.00001, 0.99999).
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0

    # Coefficients for the rational approximation
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Rational approximation for lower region
        q = math.sqrt(-2.0 * math.log(p))
        result = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        # Rational approximation for central region
        q = p - 0.5
        r = q * q
        result = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        # Rational approximation for upper region (symmetry)
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        result = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                  ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    return result


def _z_test_proportions(
    p1: float, n1: int, p2: float, n2: int
) -> Tuple[float, float]:
    """
    Two-proportion Z-test.

    Compares two sample proportions (e.g., conversion rates) to determine
    if the difference is statistically significant.

    Args:
        p1: Proportion (conversion rate) for sample 1 (control).
        n1: Sample size for sample 1.
        p2: Proportion (conversion rate) for sample 2 (variant).
        n2: Sample size for sample 2.

    Returns:
        Tuple of (z_score, p_value).  The p_value is two-tailed.
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Pooled proportion under H0: p1 == p2
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)

    if pooled == 0.0 or pooled == 1.0:
        return 0.0, 1.0

    # Standard error of the difference
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))

    if se == 0.0:
        return 0.0, 1.0

    z = (p2 - p1) / se

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))

    return z, p_value


def _calculate_confidence_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.

    More accurate than the normal approximation for small samples or
    proportions near 0 or 1.

    Args:
        successes: Number of successes (conversions).
        total: Total number of trials (impressions).
        confidence: Confidence level (0.0 to 1.0), default 0.95.

    Returns:
        (lower_bound, upper_bound) of the confidence interval.
    """
    if total == 0:
        return 0.0, 0.0

    p_hat = successes / total
    alpha = 1.0 - confidence
    z = _normal_ppf(1.0 - alpha / 2.0)
    z2 = z * z

    denominator = 1.0 + z2 / total
    centre = p_hat + z2 / (2.0 * total)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total)

    lower = (centre - spread) / denominator
    upper = (centre + spread) / denominator

    return max(0.0, lower), min(1.0, upper)


def _calculate_lift(control_rate: float, variant_rate: float) -> float:
    """
    Calculate relative lift of variant over control.

    Returns percentage lift: positive means variant is better.
    """
    if control_rate == 0.0:
        if variant_rate == 0.0:
            return 0.0
        return float("inf")
    return ((variant_rate - control_rate) / control_rate) * 100.0


def _min_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculate minimum sample size per variant for a two-proportion test.

    Uses the formula for sample size based on desired significance level
    (alpha) and statistical power.

    Args:
        baseline_rate: Expected conversion rate of the control (0.0 to 1.0).
        mde: Minimum detectable effect (absolute difference).
        alpha: Significance level (default 0.05 for 95% confidence).
        power: Statistical power (default 0.80).

    Returns:
        Minimum number of observations per variant.
    """
    if mde <= 0.0 or baseline_rate <= 0.0 or baseline_rate >= 1.0:
        return DEFAULT_MIN_SAMPLE_SIZE

    p1 = baseline_rate
    p2 = baseline_rate + mde

    if p2 >= 1.0:
        p2 = min(baseline_rate + mde, 0.999)

    z_alpha = _normal_ppf(1.0 - alpha / 2.0)
    z_beta = _normal_ppf(power)

    # Pooled proportion
    p_avg = (p1 + p2) / 2.0

    numerator = (
        z_alpha * math.sqrt(2.0 * p_avg * (1.0 - p_avg))
        + z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2

    denominator = (p2 - p1) ** 2

    if denominator == 0.0:
        return DEFAULT_MIN_SAMPLE_SIZE

    n = math.ceil(numerator / denominator)
    return max(n, DEFAULT_MIN_SAMPLE_SIZE)


def _chi_square_test(observed: List[List[int]]) -> Tuple[float, float]:
    """
    Simple 2xK chi-square test for independence.

    Used when there are more than 2 variants. Each row is
    [conversions, non_conversions] for a variant.

    Returns:
        (chi2_statistic, p_value) using chi-square approximation.
    """
    if not observed or len(observed) < 2:
        return 0.0, 1.0

    k = len(observed)
    row_totals = [sum(row) for row in observed]
    col_totals = [0, 0]
    for row in observed:
        col_totals[0] += row[0]
        col_totals[1] += row[1]
    grand_total = sum(row_totals)

    if grand_total == 0:
        return 0.0, 1.0

    chi2 = 0.0
    for i in range(k):
        for j in range(2):
            expected = (row_totals[i] * col_totals[j]) / grand_total
            if expected > 0:
                chi2 += (observed[i][j] - expected) ** 2 / expected

    # Degrees of freedom: (k-1) * (2-1) = k-1
    df = k - 1

    # Approximate p-value using the Wilson-Hilferty transformation
    p_value = _chi2_survival(chi2, df)

    return chi2, p_value


def _chi2_survival(x: float, df: int) -> float:
    """
    Survival function (1 - CDF) of the chi-squared distribution.

    Uses the Wilson-Hilferty normal approximation which is accurate for
    df >= 1 and reasonably good for df < 30.
    """
    if x <= 0.0 or df <= 0:
        return 1.0

    # Wilson-Hilferty transformation to standard normal
    k = float(df)
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))

    return 1.0 - _normal_cdf(z)


# ===========================================================================
# Visitor Assignment (consistent hashing)
# ===========================================================================


def _hash_visitor_to_variant(
    experiment_id: str,
    visitor_id: str,
    variants: List[Variant],
) -> str:
    """
    Deterministically assign a visitor to a variant using consistent hashing.

    The same (experiment_id, visitor_id) pair always returns the same variant,
    respecting traffic weights. This ensures visitors see a consistent
    experience without storing every assignment.
    """
    if not variants:
        raise ValueError("No variants to assign")

    if len(variants) == 1:
        return variants[0].variant_id

    # Create a deterministic hash in [0, 1)
    hash_input = f"{experiment_id}:{visitor_id}"
    digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    hash_value = int(digest[:16], 16) / (16 ** 16)

    # Normalise weights
    total_weight = sum(v.traffic_weight for v in variants)
    if total_weight == 0:
        total_weight = len(variants)
        normalised = [1.0 / total_weight] * len(variants)
    else:
        normalised = [v.traffic_weight / total_weight for v in variants]

    # Walk cumulative distribution
    cumulative = 0.0
    for i, weight in enumerate(normalised):
        cumulative += weight
        if hash_value < cumulative:
            return variants[i].variant_id

    # Fallback (floating-point edge case)
    return variants[-1].variant_id


# ===========================================================================
# ABTestEngine (Singleton)
# ===========================================================================


class ABTestEngine:
    """
    Core A/B testing engine.

    Manages the full lifecycle of experiments: creation, traffic assignment,
    event recording, statistical analysis, auto-conclusion, and reporting.
    All state is JSON-persisted with atomic writes.
    """

    def __init__(self) -> None:
        self._experiments: Dict[str, Experiment] = {}
        self._events: Dict[str, List[Dict[str, Any]]] = {}
        self._assignments: Dict[str, Dict[str, str]] = {}  # exp_id -> {visitor_id: variant_id}
        self._results_cache: Dict[str, ExperimentResult] = {}
        self._auto_checker_task: Optional[asyncio.Task] = None
        self._auto_checker_running: bool = False
        self._loaded: bool = False
        self._load()

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def _load(self) -> None:
        """Load all state from disk."""
        raw_experiments = _load_json(EXPERIMENTS_FILE, {})
        self._experiments = {}
        for eid, edata in raw_experiments.items():
            try:
                self._experiments[eid] = Experiment.from_dict(edata)
            except Exception as exc:
                logger.warning("Failed to load experiment %s: %s", eid, exc)

        self._events = _load_json(EVENTS_FILE, {})
        self._assignments = _load_json(ASSIGNMENTS_FILE, {})

        raw_results = _load_json(RESULTS_FILE, {})
        self._results_cache = {}
        for rid, rdata in raw_results.items():
            try:
                self._results_cache[rid] = ExperimentResult.from_dict(rdata)
            except Exception:
                pass

        self._loaded = True
        logger.debug(
            "Loaded %d experiments, %d event streams, %d assignment maps",
            len(self._experiments),
            len(self._events),
            len(self._assignments),
        )

    def _save_experiments(self) -> None:
        """Persist all experiments to disk."""
        data = {eid: exp.to_dict() for eid, exp in self._experiments.items()}
        _save_json(EXPERIMENTS_FILE, data)

    def _save_events(self) -> None:
        """Persist all events to disk."""
        _save_json(EVENTS_FILE, self._events)

    def _save_assignments(self) -> None:
        """Persist all visitor assignments to disk."""
        _save_json(ASSIGNMENTS_FILE, self._assignments)

    def _save_results(self) -> None:
        """Persist cached results to disk."""
        data = {rid: res.to_dict() for rid, res in self._results_cache.items()}
        _save_json(RESULTS_FILE, data)

    def _save_all(self) -> None:
        """Persist everything."""
        self._save_experiments()
        self._save_events()
        self._save_assignments()
        self._save_results()

    # -------------------------------------------------------------------
    # Experiment Management
    # -------------------------------------------------------------------

    async def create_experiment(
        self,
        name: str,
        experiment_type: ExperimentType,
        site_id: str,
        page_url: str = "",
        primary_metric: MetricType = MetricType.CLICK_RATE,
        secondary_metrics: Optional[List[str]] = None,
        variants: Optional[List[Dict[str, Any]]] = None,
        min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
        max_duration_days: int = DEFAULT_MAX_DURATION_DAYS,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        auto_conclude: bool = True,
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """
        Create a new experiment in DRAFT status.

        Args:
            name: Human-readable experiment name.
            experiment_type: What is being tested (headline, CTA, etc.).
            site_id: Which empire site this experiment targets.
            page_url: Specific page URL (optional).
            primary_metric: The metric to optimise for significance.
            secondary_metrics: Additional metrics to track.
            variants: List of variant definitions (dicts with name, content, etc.).
            min_sample_size: Per-variant minimum observations before concluding.
            max_duration_days: Auto-expire after this many days.
            confidence_threshold: Required confidence to declare a winner (0-1).
            auto_conclude: Whether the engine should auto-conclude this experiment.
            description: Longer experiment description.
            tags: Categorisation tags.
            metadata: Arbitrary key-value data.

        Returns:
            The newly created Experiment.
        """
        if len(self._experiments) >= MAX_EXPERIMENTS:
            # Purge oldest concluded/cancelled experiments
            self._purge_old_experiments()
            if len(self._experiments) >= MAX_EXPERIMENTS:
                raise ValueError(
                    f"Maximum of {MAX_EXPERIMENTS} experiments reached. "
                    "Delete or archive old experiments first."
                )

        experiment = Experiment(
            name=name,
            description=description,
            type=experiment_type,
            status=ExperimentStatus.DRAFT,
            site_id=site_id,
            page_url=page_url,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics or [],
            min_sample_size=min_sample_size,
            max_duration_days=max_duration_days,
            confidence_threshold=confidence_threshold,
            auto_conclude=auto_conclude,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Build variants
        if variants:
            if len(variants) > MAX_VARIANTS_PER_EXPERIMENT:
                raise ValueError(
                    f"Maximum {MAX_VARIANTS_PER_EXPERIMENT} variants per experiment."
                )
            has_control = any(v.get("is_control", False) for v in variants)
            for i, vdef in enumerate(variants):
                variant = Variant(
                    name=vdef.get("name", f"Variant {chr(65 + i)}"),
                    description=vdef.get("description", ""),
                    content=vdef.get("content", ""),
                    traffic_weight=vdef.get("traffic_weight", 1.0 / len(variants)),
                    is_control=vdef.get("is_control", (i == 0 and not has_control)),
                    metadata=vdef.get("metadata", {}),
                )
                experiment.variants.append(variant)
        else:
            # Default A/B with equal weights
            experiment.variants = [
                Variant(name="Control", is_control=True, traffic_weight=0.5),
                Variant(name="Variant B", traffic_weight=0.5),
            ]

        # Normalise weights to sum to 1.0
        self._normalise_weights(experiment)

        self._experiments[experiment.experiment_id] = experiment
        self._events[experiment.experiment_id] = []
        self._assignments[experiment.experiment_id] = {}
        self._save_experiments()

        logger.info(
            "Created experiment '%s' (%s) for site '%s' with %d variants",
            name, experiment.experiment_id, site_id, len(experiment.variants),
        )
        return experiment

    async def start_experiment(self, experiment_id: str) -> Experiment:
        """Transition an experiment from DRAFT or PAUSED to RUNNING."""
        exp = self._get_experiment_or_raise(experiment_id)

        if exp.status not in (ExperimentStatus.DRAFT, ExperimentStatus.PAUSED):
            raise ValueError(
                f"Cannot start experiment in '{exp.status.value}' status. "
                "Must be 'draft' or 'paused'."
            )

        if len(exp.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants to start.")

        exp.status = ExperimentStatus.RUNNING
        if exp.started_at is None:
            exp.started_at = _now_iso()
        exp.updated_at = _now_iso()
        self._save_experiments()

        logger.info("Started experiment '%s' (%s)", exp.name, experiment_id)
        return exp

    async def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause a RUNNING experiment."""
        exp = self._get_experiment_or_raise(experiment_id)

        if exp.status != ExperimentStatus.RUNNING:
            raise ValueError(
                f"Cannot pause experiment in '{exp.status.value}' status. "
                "Must be 'running'."
            )

        exp.status = ExperimentStatus.PAUSED
        exp.paused_at = _now_iso()
        exp.updated_at = _now_iso()
        self._save_experiments()

        logger.info("Paused experiment '%s' (%s)", exp.name, experiment_id)
        return exp

    async def resume_experiment(self, experiment_id: str) -> Experiment:
        """Resume a PAUSED experiment."""
        exp = self._get_experiment_or_raise(experiment_id)

        if exp.status != ExperimentStatus.PAUSED:
            raise ValueError(
                f"Cannot resume experiment in '{exp.status.value}' status. "
                "Must be 'paused'."
            )

        exp.status = ExperimentStatus.RUNNING
        exp.paused_at = None
        exp.updated_at = _now_iso()
        self._save_experiments()

        logger.info("Resumed experiment '%s' (%s)", exp.name, experiment_id)
        return exp

    async def conclude_experiment(
        self,
        experiment_id: str,
        winner_variant_id: Optional[str] = None,
    ) -> Experiment:
        """
        Conclude an experiment, optionally specifying a winner.

        If no winner is provided, the engine calculates results and picks
        the statistically significant winner (if any).
        """
        exp = self._get_experiment_or_raise(experiment_id)

        if exp.status in (ExperimentStatus.CONCLUDED, ExperimentStatus.CANCELLED):
            raise ValueError(
                f"Experiment already in terminal status: '{exp.status.value}'."
            )

        # Calculate final results
        result = await self.calculate_results(experiment_id)

        if winner_variant_id:
            variant = exp.get_variant(winner_variant_id)
            if variant is None:
                raise ValueError(f"Variant '{winner_variant_id}' not found in experiment.")
            exp.winner_variant_id = winner_variant_id
        elif result.winner:
            exp.winner_variant_id = result.winner

        exp.status = ExperimentStatus.CONCLUDED
        exp.concluded_at = _now_iso()
        exp.p_value = result.p_value
        exp.lift_percent = result.lift
        exp.updated_at = _now_iso()

        sig = _p_to_significance(result.p_value)
        if sig is not None:
            exp.significance_level = sig.value

        self._save_experiments()
        self._save_results()

        logger.info(
            "Concluded experiment '%s' (%s) — winner=%s, p=%.4f, lift=%.1f%%",
            exp.name, experiment_id,
            exp.winner_variant_id or "none",
            result.p_value, result.lift,
        )
        return exp

    async def cancel_experiment(self, experiment_id: str) -> Experiment:
        """Cancel an experiment. Terminal state — cannot be restarted."""
        exp = self._get_experiment_or_raise(experiment_id)

        if exp.status == ExperimentStatus.CANCELLED:
            raise ValueError("Experiment is already cancelled.")
        if exp.status == ExperimentStatus.CONCLUDED:
            raise ValueError("Cannot cancel a concluded experiment.")

        exp.status = ExperimentStatus.CANCELLED
        exp.cancelled_at = _now_iso()
        exp.updated_at = _now_iso()
        self._save_experiments()

        logger.info("Cancelled experiment '%s' (%s)", exp.name, experiment_id)
        return exp

    async def delete_experiment(self, experiment_id: str) -> None:
        """Permanently delete an experiment and all its data."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found.")

        name = self._experiments[experiment_id].name
        del self._experiments[experiment_id]
        self._events.pop(experiment_id, None)
        self._assignments.pop(experiment_id, None)
        self._results_cache.pop(experiment_id, None)
        self._save_all()

        logger.info("Deleted experiment '%s' (%s)", name, experiment_id)

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve a single experiment by ID."""
        return self._experiments.get(experiment_id)

    async def list_experiments(
        self,
        site_id: Optional[str] = None,
        status: Optional[ExperimentStatus] = None,
        experiment_type: Optional[ExperimentType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Experiment]:
        """List experiments with optional filters."""
        results: List[Experiment] = []

        for exp in self._experiments.values():
            if site_id and exp.site_id != site_id:
                continue
            if status and exp.status != status:
                continue
            if experiment_type and exp.type != experiment_type:
                continue
            if tags and not all(t in exp.tags for t in tags):
                continue
            results.append(exp)

        # Sort by most recently created
        results.sort(key=lambda e: e.created_at, reverse=True)

        return results[offset: offset + limit]

    # -------------------------------------------------------------------
    # Event Recording
    # -------------------------------------------------------------------

    async def record_impression(
        self,
        experiment_id: str,
        visitor_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record an impression for a visitor.

        Assigns the visitor to a variant (deterministically) and increments
        the variant's impression count.

        Returns:
            The variant_id the visitor was assigned to.
        """
        exp = self._get_experiment_or_raise(experiment_id)

        if exp.status != ExperimentStatus.RUNNING:
            raise ValueError(
                f"Cannot record impression: experiment status is '{exp.status.value}'."
            )

        variant_id = self.get_variant_for_visitor(experiment_id, visitor_id)
        variant = exp.get_variant(variant_id)
        if variant is None:
            raise ValueError(f"Assigned variant '{variant_id}' not found.")

        # Increment impressions
        variant.impressions += 1

        # Store assignment
        if experiment_id not in self._assignments:
            self._assignments[experiment_id] = {}
        self._assignments[experiment_id][visitor_id] = variant_id

        # Record event
        event = ExperimentEvent(
            experiment_id=experiment_id,
            variant_id=variant_id,
            event_type="impression",
            visitor_id=visitor_id,
            metadata=metadata or {},
        )
        self._append_event(experiment_id, event)

        exp.updated_at = _now_iso()
        self._save_experiments()
        self._save_assignments()

        return variant_id

    def record_impression_sync(
        self,
        experiment_id: str,
        visitor_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Synchronous wrapper for record_impression."""
        return _run_sync(self.record_impression(experiment_id, visitor_id, metadata))

    async def record_event(
        self,
        experiment_id: str,
        visitor_id: str,
        event_type: str,
        value: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentEvent:
        """
        Record a custom event for a visitor.

        The visitor must have been previously assigned (via record_impression
        or get_variant_for_visitor).

        Args:
            experiment_id: Which experiment.
            visitor_id: Which visitor.
            event_type: Event type string (e.g., 'click', 'scroll', 'engage').
            value: Numeric value associated with the event.
            metadata: Arbitrary event metadata.

        Returns:
            The recorded ExperimentEvent.
        """
        exp = self._get_experiment_or_raise(experiment_id)

        if exp.status != ExperimentStatus.RUNNING:
            raise ValueError(
                f"Cannot record event: experiment status is '{exp.status.value}'."
            )

        variant_id = self._get_assignment(experiment_id, visitor_id)
        if variant_id is None:
            # Auto-assign on first event
            variant_id = self.get_variant_for_visitor(experiment_id, visitor_id)
            if experiment_id not in self._assignments:
                self._assignments[experiment_id] = {}
            self._assignments[experiment_id][visitor_id] = variant_id

        event = ExperimentEvent(
            experiment_id=experiment_id,
            variant_id=variant_id,
            event_type=event_type,
            value=value,
            visitor_id=visitor_id,
            metadata=metadata or {},
        )
        self._append_event(experiment_id, event)

        exp.updated_at = _now_iso()
        self._save_events()

        return event

    def record_event_sync(
        self,
        experiment_id: str,
        visitor_id: str,
        event_type: str,
        value: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentEvent:
        """Synchronous wrapper for record_event."""
        return _run_sync(
            self.record_event(experiment_id, visitor_id, event_type, value, metadata)
        )

    async def record_conversion(
        self,
        experiment_id: str,
        visitor_id: str,
        value: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentEvent:
        """
        Record a conversion for a visitor.

        Increments the variant's conversion count and accumulates value.
        """
        exp = self._get_experiment_or_raise(experiment_id)

        if exp.status != ExperimentStatus.RUNNING:
            raise ValueError(
                f"Cannot record conversion: experiment status is '{exp.status.value}'."
            )

        variant_id = self._get_assignment(experiment_id, visitor_id)
        if variant_id is None:
            variant_id = self.get_variant_for_visitor(experiment_id, visitor_id)
            if experiment_id not in self._assignments:
                self._assignments[experiment_id] = {}
            self._assignments[experiment_id][visitor_id] = variant_id

        variant = exp.get_variant(variant_id)
        if variant is None:
            raise ValueError(f"Variant '{variant_id}' not found.")

        variant.conversions += 1
        variant.total_value += value

        event = ExperimentEvent(
            experiment_id=experiment_id,
            variant_id=variant_id,
            event_type="conversion",
            value=value,
            visitor_id=visitor_id,
            metadata=metadata or {},
        )
        self._append_event(experiment_id, event)

        exp.updated_at = _now_iso()
        self._save_experiments()
        self._save_events()

        return event

    def record_conversion_sync(
        self,
        experiment_id: str,
        visitor_id: str,
        value: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentEvent:
        """Synchronous wrapper for record_conversion."""
        return _run_sync(
            self.record_conversion(experiment_id, visitor_id, value, metadata)
        )

    def get_variant_for_visitor(
        self,
        experiment_id: str,
        visitor_id: str,
    ) -> str:
        """
        Get the variant assignment for a visitor.

        Uses cached assignment if available, otherwise generates a
        deterministic assignment via consistent hashing.
        """
        # Check cached assignment first
        cached = self._get_assignment(experiment_id, visitor_id)
        if cached is not None:
            return cached

        exp = self._get_experiment_or_raise(experiment_id)
        if not exp.variants:
            raise ValueError("Experiment has no variants.")

        variant_id = _hash_visitor_to_variant(
            experiment_id, visitor_id, exp.variants
        )
        return variant_id

    # -------------------------------------------------------------------
    # Statistical Analysis
    # -------------------------------------------------------------------

    async def calculate_results(self, experiment_id: str) -> ExperimentResult:
        """
        Calculate full statistical results for an experiment.

        For 2-variant experiments, uses a two-proportion Z-test.
        For 3+ variant experiments, uses pairwise Z-tests against control
        and a chi-square overall test.

        Returns:
            ExperimentResult with per-variant statistics, winner, and recommendation.
        """
        exp = self._get_experiment_or_raise(experiment_id)

        if not exp.variants:
            return ExperimentResult(experiment_id=experiment_id)

        control = exp.control
        if control is None:
            return ExperimentResult(experiment_id=experiment_id)

        variant_results: List[VariantResult] = []
        sample_sizes: Dict[str, int] = {}
        best_variant_id: Optional[str] = None
        best_p_value: float = 1.0
        best_lift: float = 0.0
        overall_p_value: float = 1.0

        control_rate = control.conversion_rate
        control_ci = _calculate_confidence_interval(
            control.conversions, control.impressions, exp.confidence_threshold
        )

        # Build control result
        control_result = VariantResult(
            variant_id=control.variant_id,
            name=control.name,
            is_control=True,
            impressions=control.impressions,
            conversions=control.conversions,
            conversion_rate=control_rate,
            total_value=control.total_value,
            revenue_per_impression=control.revenue_per_impression,
            confidence_interval=control_ci,
            lift_vs_control=0.0,
            z_score=0.0,
            p_value=1.0,
            is_significant=False,
            significance_level=None,
        )
        variant_results.append(control_result)
        sample_sizes[control.variant_id] = control.impressions

        # Compare each non-control variant to control
        for variant in exp.variants:
            if variant.variant_id == control.variant_id:
                continue

            var_rate = variant.conversion_rate
            ci = _calculate_confidence_interval(
                variant.conversions, variant.impressions, exp.confidence_threshold
            )
            lift = _calculate_lift(control_rate, var_rate)

            z_score, p_value = _z_test_proportions(
                control_rate, control.impressions,
                var_rate, variant.impressions,
            )

            sig_level = _p_to_significance(p_value)
            is_sig = p_value < (1.0 - exp.confidence_threshold)

            vr = VariantResult(
                variant_id=variant.variant_id,
                name=variant.name,
                is_control=False,
                impressions=variant.impressions,
                conversions=variant.conversions,
                conversion_rate=var_rate,
                total_value=variant.total_value,
                revenue_per_impression=variant.revenue_per_impression,
                confidence_interval=ci,
                lift_vs_control=lift,
                z_score=z_score,
                p_value=p_value,
                is_significant=is_sig,
                significance_level=sig_level.value if sig_level else None,
            )
            variant_results.append(vr)
            sample_sizes[variant.variant_id] = variant.impressions

            # Track best performing variant (lowest p with positive lift)
            if is_sig and lift > 0 and p_value < best_p_value:
                best_p_value = p_value
                best_lift = lift
                best_variant_id = variant.variant_id

        # Overall test for 3+ variants
        if len(exp.variants) >= 3:
            observed = []
            for v in exp.variants:
                non_conv = max(0, v.impressions - v.conversions)
                observed.append([v.conversions, non_conv])
            _, overall_p_value = _chi_square_test(observed)
        elif len(exp.variants) == 2:
            # Use the pairwise p-value for 2 variants
            non_control = [vr for vr in variant_results if not vr.is_control]
            if non_control:
                overall_p_value = non_control[0].p_value

        # Generate recommendation
        recommendation = self._generate_recommendation(
            exp, variant_results, best_variant_id, best_p_value, best_lift
        )

        # Determine overall confidence
        confidence = 1.0 - overall_p_value

        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_results=variant_results,
            winner=best_variant_id,
            confidence=confidence,
            p_value=overall_p_value,
            lift=best_lift,
            sample_sizes=sample_sizes,
            recommendation=recommendation,
            calculated_at=_now_iso(),
        )

        self._results_cache[experiment_id] = result
        self._save_results()

        return result

    def calculate_results_sync(self, experiment_id: str) -> ExperimentResult:
        """Synchronous wrapper for calculate_results."""
        return _run_sync(self.calculate_results(experiment_id))

    async def check_significance(self, experiment_id: str) -> Dict[str, Any]:
        """
        Quick significance check without full result calculation.

        Returns a dict with:
            - is_significant: bool
            - p_value: float
            - confidence: float
            - winner: Optional[str]
            - lift: float
            - sufficient_data: bool
            - expired: bool
            - recommendation: str
        """
        exp = self._get_experiment_or_raise(experiment_id)

        sufficient = exp.has_sufficient_data
        expired = exp.is_expired

        result = await self.calculate_results(experiment_id)

        is_significant = result.p_value < (1.0 - exp.confidence_threshold)

        action = "continue"
        if is_significant and sufficient:
            action = "conclude_winner"
        elif expired and not is_significant:
            action = "conclude_no_winner"
        elif expired and is_significant:
            action = "conclude_winner"
        elif sufficient and not is_significant:
            action = "continue"

        return {
            "experiment_id": experiment_id,
            "experiment_name": exp.name,
            "is_significant": is_significant,
            "p_value": result.p_value,
            "confidence": result.confidence,
            "winner": result.winner,
            "lift": result.lift,
            "sufficient_data": sufficient,
            "expired": expired,
            "duration_days": round(exp.duration_days, 1),
            "total_impressions": exp.total_impressions,
            "action": action,
            "recommendation": result.recommendation,
        }

    def check_significance_sync(self, experiment_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for check_significance."""
        return _run_sync(self.check_significance(experiment_id))

    # -------------------------------------------------------------------
    # Auto-Conclude
    # -------------------------------------------------------------------

    async def auto_check_experiments(self) -> List[Dict[str, Any]]:
        """
        Check all running experiments and auto-conclude those that meet criteria.

        Criteria for auto-conclusion:
            1. All variants have at least min_sample_size impressions, AND
            2. Statistical significance exceeds the confidence threshold, OR
            3. Experiment has exceeded max_duration_days.

        Returns:
            List of dicts describing actions taken.
        """
        actions: List[Dict[str, Any]] = []

        running = [
            exp for exp in self._experiments.values()
            if exp.status == ExperimentStatus.RUNNING and exp.auto_conclude
        ]

        for exp in running:
            try:
                check = await self.check_significance(exp.experiment_id)

                if check["action"] == "conclude_winner":
                    await self.conclude_experiment(
                        exp.experiment_id,
                        winner_variant_id=check["winner"],
                    )
                    actions.append({
                        "experiment_id": exp.experiment_id,
                        "name": exp.name,
                        "action": "concluded_winner",
                        "winner": check["winner"],
                        "p_value": check["p_value"],
                        "lift": check["lift"],
                    })
                    logger.info(
                        "Auto-concluded '%s' with winner %s (p=%.4f, lift=%.1f%%)",
                        exp.name, check["winner"], check["p_value"], check["lift"],
                    )

                elif check["action"] == "conclude_no_winner":
                    await self.conclude_experiment(exp.experiment_id)
                    actions.append({
                        "experiment_id": exp.experiment_id,
                        "name": exp.name,
                        "action": "concluded_no_winner",
                        "reason": "max_duration_exceeded",
                        "p_value": check["p_value"],
                        "duration_days": check["duration_days"],
                    })
                    logger.info(
                        "Auto-concluded '%s' with no winner (expired after %.1f days, p=%.4f)",
                        exp.name, check["duration_days"], check["p_value"],
                    )

            except Exception as exc:
                logger.error(
                    "Error auto-checking experiment '%s': %s", exp.name, exc
                )
                actions.append({
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "action": "error",
                    "error": str(exc),
                })

        return actions

    async def start_auto_checker(
        self,
        interval: int = DEFAULT_AUTO_CHECK_INTERVAL,
    ) -> None:
        """
        Start the background auto-checker loop.

        Args:
            interval: Seconds between checks (default 3600 = 1 hour).
        """
        if self._auto_checker_running:
            logger.warning("Auto-checker is already running.")
            return

        self._auto_checker_running = True

        async def _loop() -> None:
            logger.info("Auto-checker started (interval=%ds)", interval)
            while self._auto_checker_running:
                try:
                    actions = await self.auto_check_experiments()
                    if actions:
                        logger.info(
                            "Auto-checker cycle: %d actions taken", len(actions)
                        )
                except Exception as exc:
                    logger.error("Auto-checker error: %s", exc)
                await asyncio.sleep(interval)
            logger.info("Auto-checker stopped.")

        self._auto_checker_task = asyncio.create_task(_loop())

    def stop_auto_checker(self) -> None:
        """Stop the background auto-checker loop."""
        self._auto_checker_running = False
        if self._auto_checker_task and not self._auto_checker_task.done():
            self._auto_checker_task.cancel()
            self._auto_checker_task = None
        logger.info("Auto-checker stop requested.")

    # -------------------------------------------------------------------
    # Reporting & Analytics
    # -------------------------------------------------------------------

    async def get_stats(
        self,
        site_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics across all experiments.

        Args:
            site_id: Optionally filter to a specific site.

        Returns:
            Dictionary of aggregate statistics.
        """
        experiments = list(self._experiments.values())
        if site_id:
            experiments = [e for e in experiments if e.site_id == site_id]

        total = len(experiments)
        by_status: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_site: Dict[str, int] = {}
        total_impressions = 0
        total_conversions = 0
        total_value = 0.0
        concluded_with_winner = 0
        concluded_without_winner = 0
        avg_duration_days: List[float] = []
        avg_lift: List[float] = []
        avg_p_value: List[float] = []

        for exp in experiments:
            by_status[exp.status.value] = by_status.get(exp.status.value, 0) + 1
            by_type[exp.type.value] = by_type.get(exp.type.value, 0) + 1
            by_site[exp.site_id] = by_site.get(exp.site_id, 0) + 1
            total_impressions += exp.total_impressions
            total_conversions += exp.total_conversions
            total_value += exp.total_value

            if exp.status == ExperimentStatus.CONCLUDED:
                if exp.winner_variant_id:
                    concluded_with_winner += 1
                else:
                    concluded_without_winner += 1
                if exp.duration_days > 0:
                    avg_duration_days.append(exp.duration_days)
                if exp.lift_percent is not None:
                    avg_lift.append(exp.lift_percent)
                if exp.p_value is not None:
                    avg_p_value.append(exp.p_value)

        stats = {
            "total_experiments": total,
            "by_status": by_status,
            "by_type": by_type,
            "by_site": by_site,
            "total_impressions": total_impressions,
            "total_conversions": total_conversions,
            "total_value": round(total_value, 2),
            "overall_conversion_rate": (
                round(total_conversions / total_impressions, 4)
                if total_impressions > 0 else 0.0
            ),
            "concluded_with_winner": concluded_with_winner,
            "concluded_without_winner": concluded_without_winner,
            "win_rate": (
                round(concluded_with_winner / (concluded_with_winner + concluded_without_winner), 2)
                if (concluded_with_winner + concluded_without_winner) > 0 else 0.0
            ),
            "avg_duration_days": (
                round(sum(avg_duration_days) / len(avg_duration_days), 1)
                if avg_duration_days else 0.0
            ),
            "avg_lift_percent": (
                round(sum(avg_lift) / len(avg_lift), 1)
                if avg_lift else 0.0
            ),
            "avg_p_value": (
                round(sum(avg_p_value) / len(avg_p_value), 4)
                if avg_p_value else 1.0
            ),
            "calculated_at": _now_iso(),
        }

        # Save stats
        _save_json(STATS_FILE, stats)

        return stats

    def get_stats_sync(
        self,
        site_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for get_stats."""
        return _run_sync(self.get_stats(site_id))

    async def get_site_experiments(
        self,
        site_id: str,
        include_concluded: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get a summary of all experiments for a specific site.

        Args:
            site_id: The empire site identifier.
            include_concluded: Whether to include concluded/cancelled experiments.

        Returns:
            List of experiment summary dicts.
        """
        summaries: List[Dict[str, Any]] = []

        for exp in self._experiments.values():
            if exp.site_id != site_id:
                continue
            if not include_concluded and exp.status in (
                ExperimentStatus.CONCLUDED, ExperimentStatus.CANCELLED
            ):
                continue

            summary = {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "type": exp.type.value,
                "status": exp.status.value,
                "primary_metric": exp.primary_metric.value,
                "variants": len(exp.variants),
                "total_impressions": exp.total_impressions,
                "total_conversions": exp.total_conversions,
                "duration_days": round(exp.duration_days, 1),
                "created_at": exp.created_at,
                "winner": exp.winner_variant_id,
            }

            # Include latest results if cached
            cached = self._results_cache.get(exp.experiment_id)
            if cached:
                summary["p_value"] = cached.p_value
                summary["lift"] = cached.lift
                summary["confidence"] = cached.confidence

            summaries.append(summary)

        summaries.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return summaries

    def get_site_experiments_sync(
        self,
        site_id: str,
        include_concluded: bool = False,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_site_experiments."""
        return _run_sync(self.get_site_experiments(site_id, include_concluded))

    async def export_results(
        self,
        experiment_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """
        Export full experiment data and results for archiving or external analysis.

        Args:
            experiment_id: Which experiment to export.
            format: Output format (currently only 'json').

        Returns:
            Complete experiment export as a dict.
        """
        exp = self._get_experiment_or_raise(experiment_id)
        result = await self.calculate_results(experiment_id)

        events = self._events.get(experiment_id, [])
        assignments = self._assignments.get(experiment_id, {})

        export = {
            "experiment": exp.to_dict(),
            "results": result.to_dict(),
            "events_count": len(events),
            "unique_visitors": len(assignments),
            "events_summary": self._summarise_events(events),
            "exported_at": _now_iso(),
            "format_version": "1.0",
        }

        return export

    def export_results_sync(
        self,
        experiment_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Synchronous wrapper for export_results."""
        return _run_sync(self.export_results(experiment_id, format))

    # -------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------

    def _get_experiment_or_raise(self, experiment_id: str) -> Experiment:
        """Look up an experiment, raising ValueError if not found."""
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_id}' not found.")
        return exp

    def _get_assignment(
        self, experiment_id: str, visitor_id: str
    ) -> Optional[str]:
        """Look up a cached visitor-to-variant assignment."""
        exp_map = self._assignments.get(experiment_id)
        if exp_map is None:
            return None
        return exp_map.get(visitor_id)

    def _append_event(
        self, experiment_id: str, event: ExperimentEvent
    ) -> None:
        """Append an event, respecting the per-experiment cap."""
        if experiment_id not in self._events:
            self._events[experiment_id] = []

        events = self._events[experiment_id]
        events.append(event.to_dict())

        # Trim to cap
        if len(events) > MAX_EVENTS_PER_EXPERIMENT:
            self._events[experiment_id] = events[-MAX_EVENTS_PER_EXPERIMENT:]

        self._save_events()

    @staticmethod
    def _normalise_weights(experiment: Experiment) -> None:
        """Normalise variant traffic weights to sum to 1.0."""
        if not experiment.variants:
            return
        total = sum(v.traffic_weight for v in experiment.variants)
        if total == 0:
            equal = 1.0 / len(experiment.variants)
            for v in experiment.variants:
                v.traffic_weight = equal
        elif abs(total - 1.0) > WEIGHT_TOLERANCE:
            for v in experiment.variants:
                v.traffic_weight /= total

    def _purge_old_experiments(self) -> None:
        """Remove the oldest concluded/cancelled experiments to make space."""
        terminable = [
            (eid, exp) for eid, exp in self._experiments.items()
            if exp.status in (ExperimentStatus.CONCLUDED, ExperimentStatus.CANCELLED)
        ]
        terminable.sort(key=lambda x: x[1].created_at)

        # Remove oldest quarter
        to_remove = max(1, len(terminable) // 4)
        for eid, _ in terminable[:to_remove]:
            self._experiments.pop(eid, None)
            self._events.pop(eid, None)
            self._assignments.pop(eid, None)
            self._results_cache.pop(eid, None)

        if to_remove > 0:
            logger.info("Purged %d old experiments.", to_remove)

    @staticmethod
    def _generate_recommendation(
        exp: Experiment,
        variant_results: List[VariantResult],
        winner_id: Optional[str],
        best_p: float,
        best_lift: float,
    ) -> str:
        """Generate a human-readable recommendation string."""
        if not exp.variants:
            return "No variants defined. Add at least 2 variants and start the experiment."

        total_imp = exp.total_impressions
        if total_imp == 0:
            return "No impressions recorded yet. The experiment needs traffic to produce results."

        sufficient = exp.has_sufficient_data
        expired = exp.is_expired

        if not sufficient and not expired:
            min_needed = exp.min_sample_size
            variant_progress = []
            for v in exp.variants:
                pct = min(100, int(v.impressions / max(1, min_needed) * 100))
                variant_progress.append(f"{v.name}: {v.impressions}/{min_needed} ({pct}%)")
            progress_str = ", ".join(variant_progress)
            return (
                f"Insufficient data. Need {min_needed} impressions per variant. "
                f"Progress: {progress_str}. Continue collecting data."
            )

        if winner_id and best_p < (1.0 - exp.confidence_threshold):
            winner_name = "Unknown"
            for vr in variant_results:
                if vr.variant_id == winner_id:
                    winner_name = vr.name
                    break

            sig = _p_to_significance(best_p)
            sig_str = sig.value if sig else "not significant"

            return (
                f"WINNER: '{winner_name}' outperforms control with "
                f"{best_lift:+.1f}% lift (p={best_p:.4f}, significance={sig_str}). "
                f"Recommend deploying '{winner_name}' as the new default."
            )

        if expired:
            return (
                f"Experiment reached maximum duration ({exp.max_duration_days} days) "
                f"without achieving statistical significance (p={best_p:.4f}). "
                f"Consider: (1) extending duration, (2) increasing traffic, or "
                f"(3) testing a bolder variation."
            )

        # Significant but all lifts negative (control wins)
        any_positive = any(
            vr.lift_vs_control > 0 and vr.is_significant
            for vr in variant_results
            if not vr.is_control
        )
        if not any_positive:
            neg_variants = [
                vr for vr in variant_results
                if not vr.is_control and vr.is_significant and vr.lift_vs_control < 0
            ]
            if neg_variants:
                return (
                    "Control is winning. Variant(s) perform significantly worse. "
                    "Consider concluding and keeping the current control."
                )

        return (
            f"Data collection in progress. Current best p-value: {best_p:.4f}. "
            f"Continue running to reach higher confidence."
        )

    @staticmethod
    def _summarise_events(events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarise event counts by type."""
        summary: Dict[str, int] = {}
        for evt in events:
            etype = evt.get("event_type", "unknown")
            summary[etype] = summary.get(etype, 0) + 1
        return summary


# ===========================================================================
# Singleton
# ===========================================================================

_engine: Optional[ABTestEngine] = None


def get_engine() -> ABTestEngine:
    """Return the global ABTestEngine singleton, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = ABTestEngine()
    return _engine


# ===========================================================================
# Convenience Functions
# ===========================================================================


def create_experiment(
    name: str,
    experiment_type: ExperimentType,
    site_id: str,
    **kwargs: Any,
) -> Experiment:
    """Convenience: create an experiment via the singleton (sync)."""
    return _run_sync(
        get_engine().create_experiment(
            name=name, experiment_type=experiment_type, site_id=site_id, **kwargs
        )
    )


def record_impression(experiment_id: str, visitor_id: str) -> str:
    """Convenience: record an impression via the singleton (sync)."""
    return get_engine().record_impression_sync(experiment_id, visitor_id)


def record_conversion(
    experiment_id: str, visitor_id: str, value: float = 1.0
) -> ExperimentEvent:
    """Convenience: record a conversion via the singleton (sync)."""
    return get_engine().record_conversion_sync(experiment_id, visitor_id, value)


def get_results(experiment_id: str) -> ExperimentResult:
    """Convenience: calculate results via the singleton (sync)."""
    return get_engine().calculate_results_sync(experiment_id)


def sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Convenience: calculate minimum sample size per variant."""
    return _min_sample_size(baseline_rate, mde, alpha, power)


# ===========================================================================
# CLI Command Handlers
# ===========================================================================


def _cmd_create(args: argparse.Namespace) -> None:
    """Create a new experiment."""
    engine = get_engine()

    try:
        exp_type = ExperimentType(args.type)
    except ValueError:
        print(f"Invalid experiment type: {args.type}")
        print(f"Valid types: {', '.join(t.value for t in ExperimentType)}")
        sys.exit(1)

    try:
        metric = MetricType(args.metric)
    except ValueError:
        print(f"Invalid metric type: {args.metric}")
        print(f"Valid metrics: {', '.join(m.value for m in MetricType)}")
        sys.exit(1)

    # Parse variant definitions from --variant flags
    variants = None
    if args.variant:
        variants = []
        for i, vstr in enumerate(args.variant):
            parts = vstr.split(":", 1)
            name = parts[0].strip()
            content = parts[1].strip() if len(parts) > 1 else ""
            variants.append({
                "name": name,
                "content": content,
                "is_control": (i == 0),
            })

    exp = _run_sync(engine.create_experiment(
        name=args.name,
        experiment_type=exp_type,
        site_id=args.site,
        page_url=args.url or "",
        primary_metric=metric,
        variants=variants,
        min_sample_size=args.min_sample or DEFAULT_MIN_SAMPLE_SIZE,
        max_duration_days=args.max_days or DEFAULT_MAX_DURATION_DAYS,
        confidence_threshold=args.confidence or DEFAULT_CONFIDENCE_THRESHOLD,
        auto_conclude=not args.no_auto,
        description=args.description or "",
        tags=args.tags.split(",") if args.tags else [],
    ))

    print(f"\nExperiment created successfully!")
    print(f"  ID:       {exp.experiment_id}")
    print(f"  Name:     {exp.name}")
    print(f"  Type:     {exp.type.value}")
    print(f"  Site:     {exp.site_id}")
    print(f"  Metric:   {exp.primary_metric.value}")
    print(f"  Variants: {len(exp.variants)}")
    for v in exp.variants:
        ctrl = " (control)" if v.is_control else ""
        print(f"    - {v.name}{ctrl}: weight={v.traffic_weight:.2f}")
    print(f"  Status:   {exp.status.value}")
    print(f"\nRun 'python -m src.ab_testing start {exp.experiment_id}' to begin.")


def _cmd_start(args: argparse.Namespace) -> None:
    """Start an experiment."""
    engine = get_engine()
    try:
        exp = _run_sync(engine.start_experiment(args.experiment_id))
        print(f"Experiment '{exp.name}' started.")
        print(f"  Status:   {exp.status.value}")
        print(f"  Started:  {exp.started_at}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_pause(args: argparse.Namespace) -> None:
    """Pause an experiment."""
    engine = get_engine()
    try:
        exp = _run_sync(engine.pause_experiment(args.experiment_id))
        print(f"Experiment '{exp.name}' paused.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_resume(args: argparse.Namespace) -> None:
    """Resume a paused experiment."""
    engine = get_engine()
    try:
        exp = _run_sync(engine.resume_experiment(args.experiment_id))
        print(f"Experiment '{exp.name}' resumed.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_conclude(args: argparse.Namespace) -> None:
    """Conclude an experiment."""
    engine = get_engine()
    try:
        exp = _run_sync(engine.conclude_experiment(
            args.experiment_id,
            winner_variant_id=args.winner,
        ))
        print(f"\nExperiment '{exp.name}' concluded.")
        print(f"  Winner:      {exp.winner_variant_id or 'No clear winner'}")
        print(f"  p-value:     {exp.p_value:.4f}" if exp.p_value is not None else "  p-value:     N/A")
        print(f"  Lift:        {exp.lift_percent:.1f}%" if exp.lift_percent is not None else "  Lift:        N/A")
        print(f"  Significance:{exp.significance_level or 'N/A'}")
        print(f"  Concluded:   {exp.concluded_at}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_cancel(args: argparse.Namespace) -> None:
    """Cancel an experiment."""
    engine = get_engine()
    try:
        exp = _run_sync(engine.cancel_experiment(args.experiment_id))
        print(f"Experiment '{exp.name}' cancelled.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_record(args: argparse.Namespace) -> None:
    """Record an event for an experiment."""
    engine = get_engine()

    try:
        if args.event == "impression":
            variant_id = _run_sync(engine.record_impression(
                args.experiment, args.visitor, metadata={}
            ))
            print(f"Impression recorded for visitor '{args.visitor}' -> variant '{variant_id}'")

        elif args.event == "conversion":
            value = args.value if args.value is not None else 1.0
            evt = _run_sync(engine.record_conversion(
                args.experiment, args.visitor, value=value
            ))
            print(f"Conversion recorded: visitor='{args.visitor}', variant='{evt.variant_id}', value={value}")

        else:
            value = args.value if args.value is not None else 0.0
            evt = _run_sync(engine.record_event(
                args.experiment, args.visitor, event_type=args.event, value=value
            ))
            print(f"Event '{args.event}' recorded: visitor='{args.visitor}', variant='{evt.variant_id}'")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_results(args: argparse.Namespace) -> None:
    """Show results for an experiment."""
    engine = get_engine()

    try:
        exp = _run_sync(engine.get_experiment(args.experiment_id))
        if exp is None:
            print(f"Experiment '{args.experiment_id}' not found.")
            sys.exit(1)

        result = _run_sync(engine.calculate_results(args.experiment_id))

        print(f"\n{'='*70}")
        print(f"  Experiment: {exp.name}")
        print(f"  ID:         {exp.experiment_id}")
        print(f"  Type:       {exp.type.value}")
        print(f"  Site:       {exp.site_id}")
        print(f"  Status:     {exp.status.value}")
        print(f"  Metric:     {exp.primary_metric.value}")
        print(f"  Duration:   {exp.duration_days:.1f} days")
        print(f"  Impressions:{exp.total_impressions:,}")
        print(f"  Conversions:{exp.total_conversions:,}")
        print(f"{'='*70}")

        print(f"\n{'Variant':<20} {'Impr':>8} {'Conv':>8} {'Rate':>8} {'CI 95%':>18} {'Lift':>8} {'p-value':>8} {'Sig':>6}")
        print("-" * 96)

        for vr in result.variant_results:
            ci_str = f"[{vr.confidence_interval[0]:.4f}, {vr.confidence_interval[1]:.4f}]"
            lift_str = f"{vr.lift_vs_control:+.1f}%" if not vr.is_control else "---"
            p_str = f"{vr.p_value:.4f}" if not vr.is_control else "---"
            sig_str = ""
            if vr.is_control:
                sig_str = "ctrl"
            elif vr.is_significant:
                sig_str = vr.significance_level or "yes"
            else:
                sig_str = "no"

            name = vr.name[:18]
            print(
                f"  {name:<18} {vr.impressions:>8,} {vr.conversions:>8,} "
                f"{vr.conversion_rate:>7.4f} {ci_str:>18} {lift_str:>8} "
                f"{p_str:>8} {sig_str:>6}"
            )

        print(f"\n  Overall p-value: {result.p_value:.4f}")
        print(f"  Confidence:      {result.confidence:.2%}")
        print(f"  Winner:          {result.winner or 'None yet'}")
        print(f"  Lift:            {result.lift:+.1f}%")
        print(f"\n  Recommendation:  {result.recommendation}")
        print()

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_list(args: argparse.Namespace) -> None:
    """List experiments."""
    engine = get_engine()

    status_filter = None
    if args.status:
        try:
            status_filter = ExperimentStatus(args.status)
        except ValueError:
            print(f"Invalid status: {args.status}")
            print(f"Valid: {', '.join(s.value for s in ExperimentStatus)}")
            sys.exit(1)

    type_filter = None
    if args.type:
        try:
            type_filter = ExperimentType(args.type)
        except ValueError:
            print(f"Invalid type: {args.type}")
            sys.exit(1)

    experiments = _run_sync(engine.list_experiments(
        site_id=args.site,
        status=status_filter,
        experiment_type=type_filter,
        limit=args.limit,
    ))

    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'ID':<18} {'Name':<28} {'Type':<14} {'Status':<12} {'Site':<16} {'Impr':>8} {'Conv':>6} {'Rate':>7}")
    print("-" * 115)

    for exp in experiments:
        rate = exp.total_conversions / exp.total_impressions if exp.total_impressions > 0 else 0.0
        name = exp.name[:26]
        site = exp.site_id[:14]
        print(
            f"  {exp.experiment_id:<16} {name:<28} {exp.type.value:<14} "
            f"{exp.status.value:<12} {site:<16} {exp.total_impressions:>8,} "
            f"{exp.total_conversions:>6,} {rate:>6.4f}"
        )

    print(f"\n  Total: {len(experiments)} experiment(s)")
    print()


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show aggregate statistics."""
    engine = get_engine()
    stats = _run_sync(engine.get_stats(site_id=args.site))

    print("\n=== A/B Testing Statistics ===\n")
    print(f"  Total Experiments:          {stats['total_experiments']}")
    print(f"  Total Impressions:          {stats['total_impressions']:,}")
    print(f"  Total Conversions:          {stats['total_conversions']:,}")
    print(f"  Overall Conversion Rate:    {stats['overall_conversion_rate']:.4f}")
    print(f"  Total Value:                ${stats['total_value']:,.2f}")
    print(f"  Concluded with Winner:      {stats['concluded_with_winner']}")
    print(f"  Concluded without Winner:   {stats['concluded_without_winner']}")
    print(f"  Win Rate:                   {stats['win_rate']:.0%}")
    print(f"  Avg Duration (days):        {stats['avg_duration_days']}")
    print(f"  Avg Lift (%):               {stats['avg_lift_percent']}")
    print(f"  Avg p-value:                {stats['avg_p_value']}")

    by_status = stats.get("by_status", {})
    if by_status:
        print("\n  By Status:")
        for s, count in sorted(by_status.items()):
            bar = "#" * min(count, 40)
            print(f"    {s:<14} {count:>4}  {bar}")

    by_type = stats.get("by_type", {})
    if by_type:
        print("\n  By Type:")
        for t, count in sorted(by_type.items()):
            bar = "#" * min(count, 40)
            print(f"    {t:<18} {count:>4}  {bar}")

    by_site = stats.get("by_site", {})
    if by_site:
        print("\n  By Site:")
        for site, count in sorted(by_site.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * min(count, 40)
            print(f"    {site:<20} {count:>4}  {bar}")

    print()


def _cmd_sample_size(args: argparse.Namespace) -> None:
    """Calculate required sample size."""
    baseline = args.baseline
    mde = args.mde
    alpha = args.alpha
    power = args.power

    if baseline <= 0 or baseline >= 1:
        print("Error: --baseline must be between 0 and 1 (exclusive).")
        sys.exit(1)
    if mde <= 0:
        print("Error: --mde must be positive.")
        sys.exit(1)
    if alpha <= 0 or alpha >= 1:
        print("Error: --alpha must be between 0 and 1 (exclusive).")
        sys.exit(1)
    if power <= 0 or power >= 1:
        print("Error: --power must be between 0 and 1 (exclusive).")
        sys.exit(1)

    n = _min_sample_size(baseline, mde, alpha, power)
    total_for_ab = n * 2

    target_rate = baseline + mde
    lift = _calculate_lift(baseline, target_rate)

    print(f"\n=== Sample Size Calculator ===\n")
    print(f"  Baseline Rate:     {baseline:.4f} ({baseline*100:.2f}%)")
    print(f"  Target Rate:       {target_rate:.4f} ({target_rate*100:.2f}%)")
    print(f"  Min Detectable:    {mde:.4f} ({mde*100:.2f}% absolute)")
    print(f"  Relative Lift:     {lift:.1f}%")
    print(f"  Significance:      {alpha} (alpha)")
    print(f"  Power:             {power} (1-beta)")
    print(f"")
    print(f"  Required per variant: {n:,}")
    print(f"  Total for A/B test:   {total_for_ab:,}")
    print()

    # Estimate days needed at various traffic levels
    print("  Estimated duration at different traffic levels:")
    for daily in [50, 100, 250, 500, 1000, 2500, 5000, 10000]:
        days = math.ceil(total_for_ab / daily)
        weeks = days / 7
        if days <= 365:
            print(f"    {daily:>6,} visitors/day -> {days:>4} days ({weeks:.1f} weeks)")

    print()


def _cmd_delete(args: argparse.Namespace) -> None:
    """Delete an experiment permanently."""
    engine = get_engine()

    try:
        exp = _run_sync(engine.get_experiment(args.experiment_id))
        if exp is None:
            print(f"Experiment '{args.experiment_id}' not found.")
            sys.exit(1)

        if not args.force:
            print(f"About to permanently delete experiment '{exp.name}' ({exp.experiment_id}).")
            print(f"  Status: {exp.status.value}")
            print(f"  Impressions: {exp.total_impressions:,}")
            confirm = input("Type 'yes' to confirm: ")
            if confirm.lower() != "yes":
                print("Aborted.")
                return

        _run_sync(engine.delete_experiment(args.experiment_id))
        print(f"Experiment '{exp.name}' deleted.")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_export(args: argparse.Namespace) -> None:
    """Export experiment data."""
    engine = get_engine()

    try:
        export = _run_sync(engine.export_results(args.experiment_id))
        output = json.dumps(export, indent=2, default=str)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Exported to {out_path}")
        else:
            print(output)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_auto_check(args: argparse.Namespace) -> None:
    """Run one cycle of auto-checking."""
    engine = get_engine()
    actions = _run_sync(engine.auto_check_experiments())

    if not actions:
        print("No experiments needed action.")
        return

    print(f"\n=== Auto-Check Results ===\n")
    for action in actions:
        act = action.get("action", "unknown")
        name = action.get("name", "?")
        eid = action.get("experiment_id", "?")

        if act == "concluded_winner":
            print(f"  CONCLUDED: '{name}' ({eid})")
            print(f"    Winner: {action.get('winner')}")
            print(f"    p-value: {action.get('p_value', 0):.4f}")
            print(f"    Lift: {action.get('lift', 0):.1f}%")
        elif act == "concluded_no_winner":
            print(f"  EXPIRED: '{name}' ({eid})")
            print(f"    No significant winner after {action.get('duration_days', 0):.1f} days")
        elif act == "error":
            print(f"  ERROR: '{name}' ({eid})")
            print(f"    {action.get('error', 'Unknown error')}")
        print()


# ===========================================================================
# Main / CLI Entry Point
# ===========================================================================


def main() -> None:
    """CLI entry point for the A/B Testing Engine."""
    parser = argparse.ArgumentParser(
        prog="ab_testing",
        description="A/B Testing Engine for the OpenClaw Empire",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- create ---
    p_create = subparsers.add_parser("create", help="Create a new experiment")
    p_create.add_argument("--name", required=True, help="Experiment name")
    p_create.add_argument("--type", required=True, help="Experiment type (headline, cta, layout, ...)")
    p_create.add_argument("--site", required=True, help="Site ID")
    p_create.add_argument("--metric", default="click_rate", help="Primary metric (default: click_rate)")
    p_create.add_argument("--url", default=None, help="Page URL")
    p_create.add_argument("--variant", action="append", help="Variant as 'Name:Content' (repeat for each)")
    p_create.add_argument("--min-sample", type=int, default=None, help="Min sample size per variant")
    p_create.add_argument("--max-days", type=int, default=None, help="Max experiment duration in days")
    p_create.add_argument("--confidence", type=float, default=None, help="Confidence threshold (0-1)")
    p_create.add_argument("--no-auto", action="store_true", help="Disable auto-conclusion")
    p_create.add_argument("--description", default=None, help="Experiment description")
    p_create.add_argument("--tags", default=None, help="Comma-separated tags")
    p_create.set_defaults(func=_cmd_create)

    # --- start ---
    p_start = subparsers.add_parser("start", help="Start an experiment")
    p_start.add_argument("experiment_id", help="Experiment ID")
    p_start.set_defaults(func=_cmd_start)

    # --- pause ---
    p_pause = subparsers.add_parser("pause", help="Pause a running experiment")
    p_pause.add_argument("experiment_id", help="Experiment ID")
    p_pause.set_defaults(func=_cmd_pause)

    # --- resume ---
    p_resume = subparsers.add_parser("resume", help="Resume a paused experiment")
    p_resume.add_argument("experiment_id", help="Experiment ID")
    p_resume.set_defaults(func=_cmd_resume)

    # --- conclude ---
    p_conclude = subparsers.add_parser("conclude", help="Conclude an experiment")
    p_conclude.add_argument("experiment_id", help="Experiment ID")
    p_conclude.add_argument("--winner", default=None, help="Force a specific variant as winner")
    p_conclude.set_defaults(func=_cmd_conclude)

    # --- cancel ---
    p_cancel = subparsers.add_parser("cancel", help="Cancel an experiment")
    p_cancel.add_argument("experiment_id", help="Experiment ID")
    p_cancel.set_defaults(func=_cmd_cancel)

    # --- record ---
    p_record = subparsers.add_parser("record", help="Record an event")
    p_record.add_argument("--experiment", required=True, help="Experiment ID")
    p_record.add_argument("--visitor", required=True, help="Visitor ID")
    p_record.add_argument("--event", required=True, help="Event type (impression, conversion, click, ...)")
    p_record.add_argument("--value", type=float, default=None, help="Event value")
    p_record.set_defaults(func=_cmd_record)

    # --- results ---
    p_results = subparsers.add_parser("results", help="Show experiment results")
    p_results.add_argument("experiment_id", help="Experiment ID")
    p_results.set_defaults(func=_cmd_results)

    # --- list ---
    p_list = subparsers.add_parser("list", help="List experiments")
    p_list.add_argument("--site", default=None, help="Filter by site ID")
    p_list.add_argument("--status", default=None, help="Filter by status")
    p_list.add_argument("--type", default=None, help="Filter by experiment type")
    p_list.add_argument("--limit", type=int, default=50, help="Max results")
    p_list.set_defaults(func=_cmd_list)

    # --- stats ---
    p_stats = subparsers.add_parser("stats", help="Show aggregate statistics")
    p_stats.add_argument("--site", default=None, help="Filter by site ID")
    p_stats.set_defaults(func=_cmd_stats)

    # --- sample-size ---
    p_ss = subparsers.add_parser("sample-size", help="Calculate required sample size")
    p_ss.add_argument("--baseline", type=float, required=True, help="Baseline conversion rate (0-1)")
    p_ss.add_argument("--mde", type=float, required=True, help="Minimum detectable effect (absolute)")
    p_ss.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05)")
    p_ss.add_argument("--power", type=float, default=0.80, help="Statistical power (default 0.80)")
    p_ss.set_defaults(func=_cmd_sample_size)

    # --- delete ---
    p_delete = subparsers.add_parser("delete", help="Delete an experiment permanently")
    p_delete.add_argument("experiment_id", help="Experiment ID")
    p_delete.add_argument("--force", action="store_true", help="Skip confirmation")
    p_delete.set_defaults(func=_cmd_delete)

    # --- export ---
    p_export = subparsers.add_parser("export", help="Export experiment data")
    p_export.add_argument("experiment_id", help="Experiment ID")
    p_export.add_argument("--output", "-o", default=None, help="Output file path")
    p_export.set_defaults(func=_cmd_export)

    # --- auto-check ---
    p_auto = subparsers.add_parser("auto-check", help="Run one auto-check cycle")
    p_auto.set_defaults(func=_cmd_auto_check)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
