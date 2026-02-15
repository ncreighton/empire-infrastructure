"""
Performance Benchmarker -- OpenClaw Empire Edition
===================================================

Module latency tracking, percentile reporting, and SLA monitoring for
Nick Creighton's 16-site WordPress publishing empire.  Tracks response
times, throughput, error rates, token usage, and costs across every
subsystem: WordPress API, Anthropic LLM calls, phone commands, content
generation, screenshot capture, and more.

Features:
    - Record arbitrary metrics (latency, throughput, error rate, tokens, cost)
    - Context-manager timer for automatic latency capture
    - Decorator for auto-benchmarking async and sync functions
    - Pure-Python percentile calculation (p50, p75, p90, p95, p99)
    - SLA definitions with configurable thresholds and time windows
    - Violation detection with warning/violated/met status
    - Daily aggregate roll-ups with 90-day retention
    - Trend analysis, slowest-operation ranking, error hotspots
    - Cost breakdown by module
    - CLI with subcommands: report, module, slas, violations, trend,
      slowest, errors, costs, stats

All data persisted to: data/performance/

Usage:
    from src.performance_benchmarker import get_benchmarker, benchmark

    bench = get_benchmarker()

    # Record a measurement
    await bench.record("wordpress_client", "publish_post", MetricType.LATENCY, 1230.5, "ms")

    # Time an operation (context manager)
    async with bench.time("content_generator", "generate_article") as timer:
        article = await generate_article(topic)

    # Decorator
    @benchmark("wordpress_client", "upload_media")
    async def upload_media(file_path: str):
        ...

    # Percentile report
    report = await bench.get_percentiles("wordpress_client", "publish_post",
                                          MetricType.LATENCY, period="1h")
    print(f"p95 latency: {report.p95} ms")

    # Check SLAs
    violations = await bench.get_violations(hours=24)
    for v in violations:
        print(f"VIOLATED: {v.sla.description} -- current={v.current_value}")

CLI:
    python -m src.performance_benchmarker report --period 24h
    python -m src.performance_benchmarker module --name wordpress_client --period 1h
    python -m src.performance_benchmarker slas
    python -m src.performance_benchmarker violations --hours 24
    python -m src.performance_benchmarker trend --module wordpress_client --operation publish_post --days 7
    python -m src.performance_benchmarker slowest --limit 10 --period 24h
    python -m src.performance_benchmarker errors --period 24h
    python -m src.performance_benchmarker costs --days 7
    python -m src.performance_benchmarker stats
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import json
import logging
import math
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("performance_benchmarker")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
PERF_DATA_DIR = BASE_DIR / "data" / "performance"
DAILY_DIR = PERF_DATA_DIR / "daily"
SLAS_FILE = PERF_DATA_DIR / "slas.json"
MEASUREMENTS_FILE = PERF_DATA_DIR / "measurements.json"
CONFIG_FILE = PERF_DATA_DIR / "config.json"

PERF_DATA_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MEASUREMENTS_PER_KEY = 10_000
MAX_DAILY_FILES = 90
DEFAULT_PERIOD = "24h"

# Period string to timedelta mapping
PERIOD_MAP: Dict[str, timedelta] = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "12h": timedelta(hours=12),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
}

# ---------------------------------------------------------------------------
# UTC helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _today_iso() -> str:
    return _now_utc().strftime("%Y-%m-%d")


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string to a timezone-aware datetime."""
    if s is None:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _period_to_timedelta(period: str) -> timedelta:
    """Convert a period string like '1h', '24h', '7d' to timedelta."""
    td = PERIOD_MAP.get(period)
    if td is not None:
        return td
    # Try parsing manually: Nd or Nh or Nm
    period_lower = period.lower().strip()
    if period_lower.endswith("d"):
        try:
            return timedelta(days=int(period_lower[:-1]))
        except ValueError:
            pass
    elif period_lower.endswith("h"):
        try:
            return timedelta(hours=int(period_lower[:-1]))
        except ValueError:
            pass
    elif period_lower.endswith("m"):
        try:
            return timedelta(minutes=int(period_lower[:-1]))
        except ValueError:
            pass
    logger.warning("Unknown period '%s', defaulting to 24h", period)
    return timedelta(hours=24)


def _gen_id(prefix: str = "perf") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Atomic JSON helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# Sync runner
# ---------------------------------------------------------------------------


def _run_sync(coro):
    """Run an async coroutine from a sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Pure-Python percentile math
# ---------------------------------------------------------------------------


def _percentile(sorted_values: List[float], p: float) -> float:
    """
    Calculate the p-th percentile from a pre-sorted list of values.

    Uses linear interpolation between closest ranks (same as numpy default).
    p should be between 0 and 100.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    # Rank (0-indexed fractional position)
    rank = (p / 100.0) * (n - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    # Linear interpolation
    frac = rank - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def _std_dev(values: List[float], mean: float) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


# ===================================================================
# Enums
# ===================================================================


class MetricType(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    QUEUE_DEPTH = "queue_depth"


class SLAStatus(str, Enum):
    MET = "met"
    WARNING = "warning"
    VIOLATED = "violated"
    NOT_CONFIGURED = "not_configured"


# Comparison operators for SLA evaluation
COMPARISON_OPS: Dict[str, Callable[[float, float], bool]] = {
    "lt": lambda current, threshold: current < threshold,
    "lte": lambda current, threshold: current <= threshold,
    "gt": lambda current, threshold: current > threshold,
    "gte": lambda current, threshold: current >= threshold,
}

# Warning is within 20% of violation -- invert the comparison direction
WARNING_FACTOR: Dict[str, float] = {
    "lt": 0.80,   # current > threshold * 0.80 means approaching
    "lte": 0.80,
    "gt": 1.20,   # current < threshold * 1.20 means approaching
    "gte": 1.20,
}


# ===================================================================
# Data classes
# ===================================================================


@dataclass
class Measurement:
    """A single recorded metric data point."""
    timestamp: str
    module: str
    operation: str
    metric_type: str  # MetricType value
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Measurement:
        return cls(
            timestamp=d.get("timestamp", ""),
            module=d.get("module", ""),
            operation=d.get("operation", ""),
            metric_type=d.get("metric_type", MetricType.LATENCY.value),
            value=float(d.get("value", 0.0)),
            unit=d.get("unit", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class SLADefinition:
    """Defines an SLA threshold for a module/operation/metric combination."""
    sla_id: str
    module: str
    operation: str
    metric_type: str
    threshold: float
    comparison: str  # "lt", "gt", "lte", "gte"
    window_minutes: int = 60
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SLADefinition:
        return cls(
            sla_id=d.get("sla_id", _gen_id("sla")),
            module=d.get("module", ""),
            operation=d.get("operation", ""),
            metric_type=d.get("metric_type", MetricType.LATENCY.value),
            threshold=float(d.get("threshold", 0)),
            comparison=d.get("comparison", "lt"),
            window_minutes=int(d.get("window_minutes", 60)),
            description=d.get("description", ""),
        )


@dataclass
class SLAReport:
    """Result of evaluating an SLA against current measurements."""
    sla: SLADefinition
    status: SLAStatus
    current_value: float
    samples: int
    window_start: str
    window_end: str
    violations_count: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class PercentileReport:
    """Statistical summary of a metric over a given period."""
    module: str
    operation: str
    metric_type: str
    count: int
    min: float
    max: float
    mean: float
    median: float
    p75: float
    p90: float
    p95: float
    p99: float
    std_dev: float
    period: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===================================================================
# OperationTimer -- context manager
# ===================================================================


class OperationTimer:
    """
    Context manager for timing operations.

    Usage:
        async with benchmarker.time("module", "operation") as timer:
            result = await do_something()
        # latency is automatically recorded on exit
    """

    def __init__(
        self,
        benchmarker: PerformanceBenchmarker,
        module: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._benchmarker = benchmarker
        self._module = module
        self._operation = operation
        self._metadata = metadata or {}
        self._start: float = 0.0
        self._elapsed_ms: float = 0.0
        self._success: bool = True
        self._error_type: Optional[str] = None

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_ms

    async def __aenter__(self) -> OperationTimer:
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if exc_type is not None:
            self._success = False
            self._error_type = exc_type.__name__
        meta = {
            **self._metadata,
            "success": self._success,
        }
        if self._error_type:
            meta["error_type"] = self._error_type
        await self._benchmarker.record(
            module=self._module,
            operation=self._operation,
            metric_type=MetricType.LATENCY,
            value=self._elapsed_ms,
            unit="ms",
            metadata=meta,
        )
        return None  # do not suppress exceptions

    def __enter__(self) -> OperationTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if exc_type is not None:
            self._success = False
            self._error_type = exc_type.__name__
        meta = {
            **self._metadata,
            "success": self._success,
        }
        if self._error_type:
            meta["error_type"] = self._error_type
        _run_sync(self._benchmarker.record(
            module=self._module,
            operation=self._operation,
            metric_type=MetricType.LATENCY,
            value=self._elapsed_ms,
            unit="ms",
            metadata=meta,
        ))
        return None


# ===================================================================
# PerformanceBenchmarker
# ===================================================================


class PerformanceBenchmarker:
    """
    Module performance tracking with percentile reporting and SLA monitoring.

    Stores measurements in memory (keyed by module:operation), persists daily
    aggregates to disk, and evaluates SLA definitions against rolling windows.
    """

    def __init__(self) -> None:
        self._measurements: Dict[str, List[Measurement]] = {}
        self._slas: Dict[str, SLADefinition] = {}
        self._daily_aggregates: Dict[str, Dict] = {}
        self._max_measurements_per_key: int = MAX_MEASUREMENTS_PER_KEY
        self._max_daily_files: int = MAX_DAILY_FILES
        self._initialized = False

    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------

    async def initialize(self) -> None:
        """Load persisted SLAs and register defaults."""
        if self._initialized:
            return
        self._load_slas()
        self._register_default_slas()
        self._load_measurements()
        self._initialized = True
        logger.info(
            "PerformanceBenchmarker initialized: %d SLAs, %d measurement keys",
            len(self._slas),
            len(self._measurements),
        )

    def initialize_sync(self) -> None:
        _run_sync(self.initialize())

    def _ensure_init(self) -> None:
        if not self._initialized:
            self.initialize_sync()

    # ---------------------------------------------------------------
    # SLA persistence
    # ---------------------------------------------------------------

    def _load_slas(self) -> None:
        data = _load_json(SLAS_FILE, default=[])
        if isinstance(data, list):
            for item in data:
                sla = SLADefinition.from_dict(item)
                self._slas[sla.sla_id] = sla
        elif isinstance(data, dict):
            for sla_id, item in data.items():
                sla = SLADefinition.from_dict(item)
                self._slas[sla.sla_id] = sla

    def _persist_slas(self) -> None:
        data = [sla.to_dict() for sla in self._slas.values()]
        _save_json(SLAS_FILE, data)

    # ---------------------------------------------------------------
    # Measurement persistence
    # ---------------------------------------------------------------

    def _load_measurements(self) -> None:
        """Load recent measurements from disk into memory."""
        data = _load_json(MEASUREMENTS_FILE, default={})
        for key, items in data.items():
            if isinstance(items, list):
                self._measurements[key] = [
                    Measurement.from_dict(m) for m in items[-self._max_measurements_per_key:]
                ]

    def _persist_measurements(self) -> None:
        """Save current in-memory measurements to disk."""
        data: Dict[str, List[Dict]] = {}
        for key, measurements in self._measurements.items():
            # Only persist the most recent entries per key
            trimmed = measurements[-self._max_measurements_per_key:]
            data[key] = [m.to_dict() for m in trimmed]
        _save_json(MEASUREMENTS_FILE, data)

    # ---------------------------------------------------------------
    # Default SLAs
    # ---------------------------------------------------------------

    def _register_default_slas(self) -> None:
        """Register built-in SLA definitions if they don't already exist."""
        defaults = [
            SLADefinition(
                sla_id="sla-wp-api-latency",
                module="wordpress_client",
                operation="*",
                metric_type=MetricType.LATENCY.value,
                threshold=5000.0,
                comparison="lt",
                window_minutes=60,
                description="WordPress API: latency < 5000ms",
            ),
            SLADefinition(
                sla_id="sla-anthropic-latency",
                module="content_generator",
                operation="anthropic_call",
                metric_type=MetricType.LATENCY.value,
                threshold=30000.0,
                comparison="lt",
                window_minutes=60,
                description="Anthropic API: latency < 30000ms",
            ),
            SLADefinition(
                sla_id="sla-phone-cmd-latency",
                module="phone_controller",
                operation="*",
                metric_type=MetricType.LATENCY.value,
                threshold=10000.0,
                comparison="lt",
                window_minutes=60,
                description="Phone commands: latency < 10000ms",
            ),
            SLADefinition(
                sla_id="sla-content-gen-latency",
                module="content_generator",
                operation="generate_article",
                metric_type=MetricType.LATENCY.value,
                threshold=120000.0,
                comparison="lt",
                window_minutes=60,
                description="Content generation: latency < 120000ms",
            ),
            SLADefinition(
                sla_id="sla-screenshot-latency",
                module="vision_agent",
                operation="screenshot",
                metric_type=MetricType.LATENCY.value,
                threshold=5000.0,
                comparison="lt",
                window_minutes=60,
                description="Screenshot: latency < 5000ms",
            ),
        ]
        for sla in defaults:
            if sla.sla_id not in self._slas:
                self._slas[sla.sla_id] = sla
        self._persist_slas()

    # ---------------------------------------------------------------
    # Key helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _make_key(module: str, operation: str) -> str:
        return f"{module}:{operation}"

    def _get_measurements_for_key(
        self, key: str, since: Optional[datetime] = None
    ) -> List[Measurement]:
        """Return measurements for a key, optionally filtered by time."""
        items = self._measurements.get(key, [])
        if since is None:
            return items
        result = []
        for m in items:
            ts = _parse_iso(m.timestamp)
            if ts and ts >= since:
                result.append(m)
        return result

    def _get_measurements_for_module(
        self, module: str, since: Optional[datetime] = None
    ) -> List[Measurement]:
        """Return all measurements for a module across all operations."""
        result: List[Measurement] = []
        prefix = f"{module}:"
        for key, items in self._measurements.items():
            if key.startswith(prefix):
                if since is None:
                    result.extend(items)
                else:
                    for m in items:
                        ts = _parse_iso(m.timestamp)
                        if ts and ts >= since:
                            result.append(m)
        return result

    def _get_all_measurements(
        self, since: Optional[datetime] = None
    ) -> List[Measurement]:
        """Return all measurements, optionally filtered by time."""
        result: List[Measurement] = []
        for items in self._measurements.values():
            if since is None:
                result.extend(items)
            else:
                for m in items:
                    ts = _parse_iso(m.timestamp)
                    if ts and ts >= since:
                        result.append(m)
        return result

    def _get_matching_measurements(
        self,
        module: str,
        operation: str,
        metric_type: str,
        since: Optional[datetime] = None,
    ) -> List[Measurement]:
        """
        Return measurements matching module/operation/metric_type.

        If operation is '*', match all operations for the module.
        """
        if operation == "*":
            items = self._get_measurements_for_module(module, since)
        else:
            key = self._make_key(module, operation)
            items = self._get_measurements_for_key(key, since)

        return [m for m in items if m.metric_type == metric_type]

    # ---------------------------------------------------------------
    # Recording
    # ---------------------------------------------------------------

    async def record(
        self,
        module: str,
        operation: str,
        metric_type: MetricType | str,
        value: float,
        unit: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Measurement:
        """Record a single measurement."""
        self._ensure_init()

        if isinstance(metric_type, MetricType):
            mt_value = metric_type.value
        else:
            mt_value = str(metric_type)

        # Auto-detect unit if not provided
        if not unit:
            unit = self._default_unit(mt_value)

        measurement = Measurement(
            timestamp=_now_iso(),
            module=module,
            operation=operation,
            metric_type=mt_value,
            value=value,
            unit=unit,
            metadata=metadata or {},
        )

        key = self._make_key(module, operation)
        if key not in self._measurements:
            self._measurements[key] = []
        self._measurements[key].append(measurement)

        # Trim if over limit
        if len(self._measurements[key]) > self._max_measurements_per_key:
            excess = len(self._measurements[key]) - self._max_measurements_per_key
            self._measurements[key] = self._measurements[key][excess:]

        logger.debug(
            "Recorded %s %s:%s = %.2f %s",
            mt_value, module, operation, value, unit,
        )
        return measurement

    def record_sync(
        self,
        module: str,
        operation: str,
        metric_type: MetricType | str,
        value: float,
        unit: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Measurement:
        """Synchronous wrapper for record()."""
        return _run_sync(self.record(
            module, operation, metric_type, value, unit, metadata,
        ))

    @staticmethod
    def _default_unit(metric_type: str) -> str:
        units = {
            MetricType.LATENCY.value: "ms",
            MetricType.THROUGHPUT.value: "req/s",
            MetricType.ERROR_RATE.value: "%",
            MetricType.TOKEN_USAGE.value: "tokens",
            MetricType.COST.value: "$",
            MetricType.QUEUE_DEPTH.value: "items",
        }
        return units.get(metric_type, "")

    # ---------------------------------------------------------------
    # Timer context manager
    # ---------------------------------------------------------------

    def time(
        self,
        module: str,
        operation: str,
        **metadata: Any,
    ) -> OperationTimer:
        """
        Return a context manager that records latency on exit.

        Supports both async and sync usage:
            async with bench.time("mod", "op"):
                ...
            with bench.time("mod", "op"):
                ...
        """
        self._ensure_init()
        return OperationTimer(self, module, operation, metadata)

    # ---------------------------------------------------------------
    # Percentile reporting
    # ---------------------------------------------------------------

    async def get_percentiles(
        self,
        module: str,
        operation: str,
        metric_type: MetricType | str = MetricType.LATENCY,
        period: str = "1h",
    ) -> PercentileReport:
        """Calculate percentile statistics for a metric over a time period."""
        self._ensure_init()

        if isinstance(metric_type, MetricType):
            mt_value = metric_type.value
        else:
            mt_value = str(metric_type)

        td = _period_to_timedelta(period)
        since = _now_utc() - td

        measurements = self._get_matching_measurements(
            module, operation, mt_value, since,
        )
        values = [m.value for m in measurements]

        if not values:
            return PercentileReport(
                module=module,
                operation=operation,
                metric_type=mt_value,
                count=0,
                min=0.0,
                max=0.0,
                mean=0.0,
                median=0.0,
                p75=0.0,
                p90=0.0,
                p95=0.0,
                p99=0.0,
                std_dev=0.0,
                period=period,
            )

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean_val = sum(sorted_vals) / n

        return PercentileReport(
            module=module,
            operation=operation,
            metric_type=mt_value,
            count=n,
            min=sorted_vals[0],
            max=sorted_vals[-1],
            mean=round(mean_val, 3),
            median=round(_percentile(sorted_vals, 50), 3),
            p75=round(_percentile(sorted_vals, 75), 3),
            p90=round(_percentile(sorted_vals, 90), 3),
            p95=round(_percentile(sorted_vals, 95), 3),
            p99=round(_percentile(sorted_vals, 99), 3),
            std_dev=round(_std_dev(sorted_vals, mean_val), 3),
            period=period,
        )

    def get_percentiles_sync(
        self,
        module: str,
        operation: str,
        metric_type: MetricType | str = MetricType.LATENCY,
        period: str = "1h",
    ) -> PercentileReport:
        return _run_sync(self.get_percentiles(module, operation, metric_type, period))

    # ---------------------------------------------------------------
    # Module reports
    # ---------------------------------------------------------------

    async def get_module_report(
        self,
        module: str,
        period: str = "24h",
    ) -> Dict[str, Any]:
        """
        Get a full report for a module: all operations with their percentile stats.
        """
        self._ensure_init()
        td = _period_to_timedelta(period)
        since = _now_utc() - td

        # Discover operations for this module
        prefix = f"{module}:"
        operations: Dict[str, Dict[str, List[float]]] = {}

        for key, items in self._measurements.items():
            if not key.startswith(prefix):
                continue
            op = key[len(prefix):]
            for m in items:
                ts = _parse_iso(m.timestamp)
                if ts and ts >= since:
                    if op not in operations:
                        operations[op] = {}
                    mt = m.metric_type
                    if mt not in operations[op]:
                        operations[op][mt] = []
                    operations[op][mt].append(m.value)

        report: Dict[str, Any] = {
            "module": module,
            "period": period,
            "generated_at": _now_iso(),
            "operations": {},
        }

        for op, metrics in sorted(operations.items()):
            op_report: Dict[str, Any] = {}
            for mt, values in metrics.items():
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                mean_val = sum(sorted_vals) / n if n else 0
                op_report[mt] = {
                    "count": n,
                    "min": round(sorted_vals[0], 3) if sorted_vals else 0,
                    "max": round(sorted_vals[-1], 3) if sorted_vals else 0,
                    "mean": round(mean_val, 3),
                    "p50": round(_percentile(sorted_vals, 50), 3),
                    "p95": round(_percentile(sorted_vals, 95), 3),
                    "p99": round(_percentile(sorted_vals, 99), 3),
                }
            report["operations"][op] = op_report

        return report

    async def get_all_modules_report(
        self,
        period: str = "24h",
    ) -> Dict[str, Any]:
        """
        Summary of all modules with measurement counts and key stats.
        """
        self._ensure_init()
        td = _period_to_timedelta(period)
        since = _now_utc() - td

        module_stats: Dict[str, Dict[str, Any]] = {}

        for key, items in self._measurements.items():
            parts = key.split(":", 1)
            if len(parts) != 2:
                continue
            module_name, op = parts

            for m in items:
                ts = _parse_iso(m.timestamp)
                if ts and ts >= since:
                    if module_name not in module_stats:
                        module_stats[module_name] = {
                            "total_measurements": 0,
                            "operations": set(),
                            "metric_types": set(),
                            "latency_values": [],
                            "error_count": 0,
                        }
                    stats = module_stats[module_name]
                    stats["total_measurements"] += 1
                    stats["operations"].add(op)
                    stats["metric_types"].add(m.metric_type)
                    if m.metric_type == MetricType.LATENCY.value:
                        stats["latency_values"].append(m.value)
                    if m.metadata.get("success") is False:
                        stats["error_count"] += 1

        report: Dict[str, Any] = {
            "period": period,
            "generated_at": _now_iso(),
            "total_modules": len(module_stats),
            "modules": {},
        }

        for mod, stats in sorted(module_stats.items()):
            lat_vals = sorted(stats["latency_values"])
            n_lat = len(lat_vals)
            mean_lat = sum(lat_vals) / n_lat if n_lat else 0

            report["modules"][mod] = {
                "total_measurements": stats["total_measurements"],
                "operations_count": len(stats["operations"]),
                "operations": sorted(stats["operations"]),
                "metric_types": sorted(stats["metric_types"]),
                "error_count": stats["error_count"],
                "latency": {
                    "count": n_lat,
                    "mean": round(mean_lat, 3),
                    "p50": round(_percentile(lat_vals, 50), 3) if n_lat else 0,
                    "p95": round(_percentile(lat_vals, 95), 3) if n_lat else 0,
                    "p99": round(_percentile(lat_vals, 99), 3) if n_lat else 0,
                } if n_lat else None,
            }

        return report

    async def compare_modules(
        self,
        modules: List[str],
        metric_type: MetricType | str = MetricType.LATENCY,
        period: str = "24h",
    ) -> Dict[str, Any]:
        """Side-by-side module comparison for a given metric type."""
        self._ensure_init()

        if isinstance(metric_type, MetricType):
            mt_value = metric_type.value
        else:
            mt_value = str(metric_type)

        td = _period_to_timedelta(period)
        since = _now_utc() - td

        comparison: Dict[str, Any] = {
            "metric_type": mt_value,
            "period": period,
            "generated_at": _now_iso(),
            "modules": {},
        }

        for module in modules:
            measurements = self._get_measurements_for_module(module, since)
            values = [m.value for m in measurements if m.metric_type == mt_value]

            if not values:
                comparison["modules"][module] = {"count": 0, "data": None}
                continue

            sorted_vals = sorted(values)
            n = len(sorted_vals)
            mean_val = sum(sorted_vals) / n

            comparison["modules"][module] = {
                "count": n,
                "min": round(sorted_vals[0], 3),
                "max": round(sorted_vals[-1], 3),
                "mean": round(mean_val, 3),
                "p50": round(_percentile(sorted_vals, 50), 3),
                "p95": round(_percentile(sorted_vals, 95), 3),
                "p99": round(_percentile(sorted_vals, 99), 3),
                "std_dev": round(_std_dev(sorted_vals, mean_val), 3),
            }

        return comparison

    # ---------------------------------------------------------------
    # SLA management
    # ---------------------------------------------------------------

    async def define_sla(
        self,
        module: str,
        operation: str,
        metric_type: MetricType | str,
        threshold: float,
        comparison: str = "lt",
        window_minutes: int = 60,
        description: str = "",
    ) -> SLADefinition:
        """Define or update an SLA."""
        self._ensure_init()

        if isinstance(metric_type, MetricType):
            mt_value = metric_type.value
        else:
            mt_value = str(metric_type)

        if comparison not in COMPARISON_OPS:
            raise ValueError(f"Invalid comparison '{comparison}'. Use: {list(COMPARISON_OPS)}")

        sla_id = f"sla-{module}-{operation}-{mt_value}".replace("*", "all")
        sla = SLADefinition(
            sla_id=sla_id,
            module=module,
            operation=operation,
            metric_type=mt_value,
            threshold=threshold,
            comparison=comparison,
            window_minutes=window_minutes,
            description=description or f"{module}:{operation} {mt_value} {comparison} {threshold}",
        )
        self._slas[sla_id] = sla
        self._persist_slas()
        logger.info("Defined SLA %s: %s", sla_id, sla.description)
        return sla

    async def check_sla(self, sla_id: str) -> SLAReport:
        """Evaluate a single SLA against current measurements."""
        self._ensure_init()

        sla = self._slas.get(sla_id)
        if sla is None:
            return SLAReport(
                sla=SLADefinition(
                    sla_id=sla_id, module="", operation="",
                    metric_type="", threshold=0, comparison="lt",
                    description="SLA not found",
                ),
                status=SLAStatus.NOT_CONFIGURED,
                current_value=0.0,
                samples=0,
                window_start="",
                window_end="",
                violations_count=0,
            )

        now = _now_utc()
        window_start = now - timedelta(minutes=sla.window_minutes)
        window_end = now

        measurements = self._get_matching_measurements(
            sla.module, sla.operation, sla.metric_type, window_start,
        )

        if not measurements:
            return SLAReport(
                sla=sla,
                status=SLAStatus.MET,
                current_value=0.0,
                samples=0,
                window_start=window_start.isoformat(),
                window_end=window_end.isoformat(),
                violations_count=0,
            )

        values = [m.value for m in measurements]
        # Use mean for aggregate SLA evaluation
        current_value = sum(values) / len(values)

        # Count individual violations
        compare_fn = COMPARISON_OPS.get(sla.comparison, COMPARISON_OPS["lt"])
        violations_count = sum(
            1 for v in values if not compare_fn(v, sla.threshold)
        )

        # Determine status
        if not compare_fn(current_value, sla.threshold):
            status = SLAStatus.VIOLATED
        else:
            # Check if within 20% of violation (warning zone)
            warning_factor = WARNING_FACTOR.get(sla.comparison, 0.80)
            warning_threshold = sla.threshold * warning_factor
            if sla.comparison in ("lt", "lte"):
                # For "less than" comparisons, warning when current > 80% of threshold
                if current_value > warning_threshold:
                    status = SLAStatus.WARNING
                else:
                    status = SLAStatus.MET
            else:
                # For "greater than" comparisons, warning when current < 120% of threshold
                if current_value < warning_threshold:
                    status = SLAStatus.WARNING
                else:
                    status = SLAStatus.MET

        return SLAReport(
            sla=sla,
            status=status,
            current_value=round(current_value, 3),
            samples=len(values),
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
            violations_count=violations_count,
        )

    async def check_all_slas(self) -> List[SLAReport]:
        """Evaluate all defined SLAs."""
        self._ensure_init()
        reports = []
        for sla_id in self._slas:
            report = await self.check_sla(sla_id)
            reports.append(report)
        return reports

    async def get_violations(self, hours: int = 24) -> List[SLAReport]:
        """Return all SLAs currently in VIOLATED status."""
        self._ensure_init()
        all_reports = await self.check_all_slas()
        return [r for r in all_reports if r.status == SLAStatus.VIOLATED]

    async def remove_sla(self, sla_id: str) -> bool:
        """Remove an SLA definition. Returns True if found and removed."""
        self._ensure_init()
        if sla_id in self._slas:
            del self._slas[sla_id]
            self._persist_slas()
            logger.info("Removed SLA %s", sla_id)
            return True
        return False

    # ---------------------------------------------------------------
    # Analytics
    # ---------------------------------------------------------------

    async def get_trend(
        self,
        module: str,
        operation: str,
        metric_type: MetricType | str = MetricType.LATENCY,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Return daily averages for trending.  One entry per day for the
        requested number of days.
        """
        self._ensure_init()

        if isinstance(metric_type, MetricType):
            mt_value = metric_type.value
        else:
            mt_value = str(metric_type)

        now = _now_utc()
        trend_data: List[Dict[str, Any]] = []

        for day_offset in range(days - 1, -1, -1):
            day_start = (now - timedelta(days=day_offset)).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            day_end = day_start + timedelta(days=1)
            date_str = day_start.strftime("%Y-%m-%d")

            measurements = self._get_matching_measurements(
                module, operation, mt_value, day_start,
            )
            # Filter to just this day (not beyond day_end)
            day_values = []
            for m in measurements:
                ts = _parse_iso(m.timestamp)
                if ts and ts < day_end:
                    day_values.append(m.value)

            # Also check persisted daily aggregates
            daily_file = DAILY_DIR / f"{date_str}.json"
            if daily_file.exists() and not day_values:
                daily_data = _load_json(daily_file, default={})
                key = self._make_key(module, operation)
                if key in daily_data:
                    agg = daily_data[key].get(mt_value, {})
                    if agg.get("count", 0) > 0:
                        trend_data.append({
                            "date": date_str,
                            "mean": agg.get("mean", 0),
                            "min": agg.get("min", 0),
                            "max": agg.get("max", 0),
                            "count": agg.get("count", 0),
                            "p95": agg.get("p95", 0),
                            "source": "daily_aggregate",
                        })
                        continue

            if day_values:
                sorted_vals = sorted(day_values)
                n = len(sorted_vals)
                mean_val = sum(sorted_vals) / n
                trend_data.append({
                    "date": date_str,
                    "mean": round(mean_val, 3),
                    "min": round(sorted_vals[0], 3),
                    "max": round(sorted_vals[-1], 3),
                    "count": n,
                    "p95": round(_percentile(sorted_vals, 95), 3),
                    "source": "live",
                })
            else:
                trend_data.append({
                    "date": date_str,
                    "mean": 0,
                    "min": 0,
                    "max": 0,
                    "count": 0,
                    "p95": 0,
                    "source": "none",
                })

        return trend_data

    async def get_slowest_operations(
        self,
        limit: int = 10,
        period: str = "24h",
    ) -> List[Dict[str, Any]]:
        """Return the slowest operations ranked by mean latency."""
        self._ensure_init()
        td = _period_to_timedelta(period)
        since = _now_utc() - td

        # Gather latency by module:operation
        op_latencies: Dict[str, List[float]] = {}
        for key, items in self._measurements.items():
            for m in items:
                if m.metric_type != MetricType.LATENCY.value:
                    continue
                ts = _parse_iso(m.timestamp)
                if ts and ts >= since:
                    if key not in op_latencies:
                        op_latencies[key] = []
                    op_latencies[key].append(m.value)

        results: List[Dict[str, Any]] = []
        for key, values in op_latencies.items():
            parts = key.split(":", 1)
            module = parts[0] if len(parts) > 0 else ""
            operation = parts[1] if len(parts) > 1 else ""
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            mean_val = sum(sorted_vals) / n
            results.append({
                "module": module,
                "operation": operation,
                "mean_ms": round(mean_val, 3),
                "p95_ms": round(_percentile(sorted_vals, 95), 3),
                "p99_ms": round(_percentile(sorted_vals, 99), 3),
                "max_ms": round(sorted_vals[-1], 3),
                "count": n,
            })

        results.sort(key=lambda x: x["mean_ms"], reverse=True)
        return results[:limit]

    async def get_error_hotspots(
        self,
        period: str = "24h",
    ) -> List[Dict[str, Any]]:
        """Return operations with the highest error rates."""
        self._ensure_init()
        td = _period_to_timedelta(period)
        since = _now_utc() - td

        op_counts: Dict[str, Dict[str, int]] = {}  # key -> {total, errors}

        for key, items in self._measurements.items():
            for m in items:
                ts = _parse_iso(m.timestamp)
                if ts and ts >= since:
                    if key not in op_counts:
                        op_counts[key] = {"total": 0, "errors": 0}
                    op_counts[key]["total"] += 1
                    if m.metadata.get("success") is False:
                        op_counts[key]["errors"] += 1

        results: List[Dict[str, Any]] = []
        for key, counts in op_counts.items():
            if counts["errors"] == 0:
                continue
            parts = key.split(":", 1)
            module = parts[0] if len(parts) > 0 else ""
            operation = parts[1] if len(parts) > 1 else ""
            error_rate = (counts["errors"] / counts["total"]) * 100 if counts["total"] else 0
            results.append({
                "module": module,
                "operation": operation,
                "total": counts["total"],
                "errors": counts["errors"],
                "error_rate_pct": round(error_rate, 2),
            })

        results.sort(key=lambda x: x["error_rate_pct"], reverse=True)
        return results

    async def get_cost_breakdown(
        self,
        days: int = 7,
    ) -> Dict[str, Any]:
        """Cost breakdown by module over the given number of days."""
        self._ensure_init()
        since = _now_utc() - timedelta(days=days)

        module_costs: Dict[str, float] = {}
        total_cost = 0.0

        for key, items in self._measurements.items():
            for m in items:
                if m.metric_type != MetricType.COST.value:
                    continue
                ts = _parse_iso(m.timestamp)
                if ts and ts >= since:
                    parts = key.split(":", 1)
                    module = parts[0] if len(parts) > 0 else "unknown"
                    module_costs[module] = module_costs.get(module, 0.0) + m.value
                    total_cost += m.value

        # Build per-module breakdown
        breakdown: Dict[str, Any] = {
            "days": days,
            "generated_at": _now_iso(),
            "total_cost": round(total_cost, 4),
            "modules": {},
        }

        for mod, cost in sorted(module_costs.items(), key=lambda x: x[1], reverse=True):
            pct = (cost / total_cost * 100) if total_cost > 0 else 0
            breakdown["modules"][mod] = {
                "cost": round(cost, 4),
                "percentage": round(pct, 2),
            }

        return breakdown

    # ---------------------------------------------------------------
    # Persistence — daily aggregates
    # ---------------------------------------------------------------

    async def save_daily_aggregate(self) -> str:
        """
        Roll up today's measurements into a daily aggregate file.

        Returns the path of the saved file.
        """
        self._ensure_init()
        today = _today_iso()
        today_start = _now_utc().replace(hour=0, minute=0, second=0, microsecond=0)

        aggregates: Dict[str, Dict[str, Any]] = {}

        for key, items in self._measurements.items():
            for m in items:
                ts = _parse_iso(m.timestamp)
                if ts is None or ts < today_start:
                    continue

                if key not in aggregates:
                    aggregates[key] = {}
                mt = m.metric_type
                if mt not in aggregates[key]:
                    aggregates[key][mt] = {
                        "values": [],
                        "module": m.module,
                        "operation": m.operation,
                        "unit": m.unit,
                    }
                aggregates[key][mt]["values"].append(m.value)

        # Compute stats for each group
        result: Dict[str, Dict[str, Any]] = {}
        for key, metrics in aggregates.items():
            result[key] = {}
            for mt, data in metrics.items():
                values = sorted(data["values"])
                n = len(values)
                mean_val = sum(values) / n if n else 0
                result[key][mt] = {
                    "module": data["module"],
                    "operation": data["operation"],
                    "unit": data["unit"],
                    "count": n,
                    "min": round(values[0], 3) if values else 0,
                    "max": round(values[-1], 3) if values else 0,
                    "mean": round(mean_val, 3),
                    "p50": round(_percentile(values, 50), 3),
                    "p75": round(_percentile(values, 75), 3),
                    "p90": round(_percentile(values, 90), 3),
                    "p95": round(_percentile(values, 95), 3),
                    "p99": round(_percentile(values, 99), 3),
                    "std_dev": round(_std_dev(values, mean_val), 3),
                    "sum": round(sum(values), 4),
                }

        daily_file = DAILY_DIR / f"{today}.json"
        _save_json(daily_file, {
            "date": today,
            "generated_at": _now_iso(),
            "keys": result,
        })

        self._daily_aggregates[today] = result
        logger.info("Saved daily aggregate: %s (%d keys)", daily_file, len(result))
        return str(daily_file)

    async def load_daily(self, date_str: str) -> Dict[str, Any]:
        """Load a daily aggregate file."""
        daily_file = DAILY_DIR / f"{date_str}.json"
        return _load_json(daily_file, default={})

    async def purge_old(self, days: int = 90) -> int:
        """
        Remove daily aggregate files older than the given number of days.
        Returns the count of files removed.
        """
        cutoff = _now_utc() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        removed = 0

        for daily_file in sorted(DAILY_DIR.glob("*.json")):
            date_part = daily_file.stem  # e.g. "2026-01-15"
            if date_part < cutoff_str:
                try:
                    daily_file.unlink()
                    removed += 1
                    logger.info("Purged old daily file: %s", daily_file)
                except OSError as exc:
                    logger.warning("Failed to purge %s: %s", daily_file, exc)

        # Also trim in-memory measurements older than cutoff
        for key in list(self._measurements.keys()):
            items = self._measurements[key]
            filtered = []
            for m in items:
                ts = _parse_iso(m.timestamp)
                if ts and ts >= cutoff:
                    filtered.append(m)
            self._measurements[key] = filtered
            if not filtered:
                del self._measurements[key]

        return removed

    # ---------------------------------------------------------------
    # Stats
    # ---------------------------------------------------------------

    async def get_stats(self) -> Dict[str, Any]:
        """Overview statistics about the benchmarker state."""
        self._ensure_init()

        total_measurements = sum(len(v) for v in self._measurements.values())
        modules = set()
        for key in self._measurements:
            parts = key.split(":", 1)
            if parts:
                modules.add(parts[0])

        # Disk usage
        disk_bytes = 0
        file_count = 0
        for f in PERF_DATA_DIR.rglob("*.json"):
            try:
                disk_bytes += f.stat().st_size
                file_count += 1
            except OSError:
                pass

        daily_files = list(DAILY_DIR.glob("*.json"))
        oldest_daily = None
        newest_daily = None
        if daily_files:
            sorted_files = sorted(daily_files, key=lambda p: p.stem)
            oldest_daily = sorted_files[0].stem
            newest_daily = sorted_files[-1].stem

        return {
            "generated_at": _now_iso(),
            "total_measurements_in_memory": total_measurements,
            "measurement_keys": len(self._measurements),
            "modules_tracked": sorted(modules),
            "modules_count": len(modules),
            "slas_defined": len(self._slas),
            "daily_aggregate_files": len(daily_files),
            "oldest_daily": oldest_daily,
            "newest_daily": newest_daily,
            "disk_usage_bytes": disk_bytes,
            "disk_usage_mb": round(disk_bytes / (1024 * 1024), 2),
            "json_files_on_disk": file_count,
            "max_measurements_per_key": self._max_measurements_per_key,
            "max_daily_files": self._max_daily_files,
        }

    # ---------------------------------------------------------------
    # Flush / save
    # ---------------------------------------------------------------

    async def flush(self) -> None:
        """Persist all in-memory state to disk."""
        self._persist_measurements()
        self._persist_slas()
        await self.save_daily_aggregate()
        logger.info("Flushed all benchmarker data to disk")

    def flush_sync(self) -> None:
        _run_sync(self.flush())


# ===================================================================
# Decorator
# ===================================================================


def benchmark(module: str, operation: str):
    """
    Decorator that auto-records latency for async or sync functions.

    Records:
        - MetricType.LATENCY (ms)
        - success/failure in metadata
        - error_type in metadata on failure

    Usage:
        @benchmark("wordpress_client", "upload_media")
        async def upload_media(file_path: str):
            ...

        @benchmark("seo_auditor", "run_audit")
        def run_audit(site_id: str):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                bench = get_benchmarker()
                start = time.perf_counter()
                success = True
                error_type = None
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    success = False
                    error_type = type(exc).__name__
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    meta = {"success": success, "function": fn.__qualname__}
                    if error_type:
                        meta["error_type"] = error_type
                    await bench.record(
                        module=module,
                        operation=operation,
                        metric_type=MetricType.LATENCY,
                        value=elapsed_ms,
                        unit="ms",
                        metadata=meta,
                    )
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                bench = get_benchmarker()
                start = time.perf_counter()
                success = True
                error_type = None
                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    success = False
                    error_type = type(exc).__name__
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    meta = {"success": success, "function": fn.__qualname__}
                    if error_type:
                        meta["error_type"] = error_type
                    bench.record_sync(
                        module=module,
                        operation=operation,
                        metric_type=MetricType.LATENCY,
                        value=elapsed_ms,
                        unit="ms",
                        metadata=meta,
                    )
            return sync_wrapper
    return decorator


# ===================================================================
# Singleton
# ===================================================================

_benchmarker: Optional[PerformanceBenchmarker] = None


def get_benchmarker() -> PerformanceBenchmarker:
    """Return the module-level singleton PerformanceBenchmarker."""
    global _benchmarker
    if _benchmarker is None:
        _benchmarker = PerformanceBenchmarker()
        _benchmarker.initialize_sync()
    return _benchmarker


# ===================================================================
# CLI formatting helpers
# ===================================================================


def _fmt_value(value: float, unit: str = "") -> str:
    """Format a numeric value with its unit for display."""
    if unit == "ms":
        if value >= 1000:
            return f"{value / 1000:.2f}s"
        return f"{value:.1f}ms"
    elif unit == "$":
        return f"${value:.4f}"
    elif unit == "%":
        return f"{value:.2f}%"
    elif unit == "tokens":
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return f"{int(value)}"
    else:
        return f"{value:.3f}" if isinstance(value, float) else str(value)


def _sla_status_label(status: SLAStatus) -> str:
    labels = {
        SLAStatus.MET: "OK",
        SLAStatus.WARNING: "WARN",
        SLAStatus.VIOLATED: "FAIL",
        SLAStatus.NOT_CONFIGURED: "N/A",
    }
    return labels.get(status, "???")


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print a simple aligned text table."""
    if not rows:
        print("  (no data)")
        return

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Header
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))

    # Rows
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            w = widths[i] if i < len(widths) else 0
            cells.append(str(cell).ljust(w))
        print("  ".join(cells))


# ===================================================================
# CLI entry point
# ===================================================================


def main() -> None:
    """CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog="performance_benchmarker",
        description="OpenClaw Empire -- Performance Benchmarker CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # report
    p_report = sub.add_parser("report", help="All-modules summary report")
    p_report.add_argument("--period", default="24h", help="Time period (1h, 24h, 7d)")

    # module
    p_module = sub.add_parser("module", help="Detailed report for a single module")
    p_module.add_argument("--name", required=True, help="Module name")
    p_module.add_argument("--period", default="24h", help="Time period")

    # slas
    sub.add_parser("slas", help="Check all SLA definitions")

    # violations
    p_violations = sub.add_parser("violations", help="Show violated SLAs")
    p_violations.add_argument("--hours", type=int, default=24, help="Lookback hours")

    # trend
    p_trend = sub.add_parser("trend", help="Show daily trend for an operation")
    p_trend.add_argument("--module", required=True, help="Module name")
    p_trend.add_argument("--operation", required=True, help="Operation name")
    p_trend.add_argument("--metric", default="latency", help="Metric type")
    p_trend.add_argument("--days", type=int, default=7, help="Number of days")

    # slowest
    p_slowest = sub.add_parser("slowest", help="Slowest operations by mean latency")
    p_slowest.add_argument("--limit", type=int, default=10, help="Max results")
    p_slowest.add_argument("--period", default="24h", help="Time period")

    # errors
    p_errors = sub.add_parser("errors", help="Error hotspots")
    p_errors.add_argument("--period", default="24h", help="Time period")

    # costs
    p_costs = sub.add_parser("costs", help="Cost breakdown by module")
    p_costs.add_argument("--days", type=int, default=7, help="Number of days")

    # stats
    sub.add_parser("stats", help="Benchmarker system statistics")

    # flush
    sub.add_parser("flush", help="Persist all data to disk")

    # purge
    p_purge = sub.add_parser("purge", help="Remove old daily aggregate files")
    p_purge.add_argument("--days", type=int, default=90, help="Keep files newer than N days")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    bench = get_benchmarker()

    # ------------------------------------------------------------------
    if args.command == "report":
        report = _run_sync(bench.get_all_modules_report(period=args.period))
        print(f"\n=== All Modules Report ({args.period}) ===")
        print(f"Generated: {report['generated_at']}")
        print(f"Modules tracked: {report['total_modules']}\n")

        headers = ["Module", "Ops", "Measurements", "Errors", "Mean Lat.", "p95 Lat.", "p99 Lat."]
        rows = []
        for mod_name, mod_data in report.get("modules", {}).items():
            lat = mod_data.get("latency") or {}
            rows.append([
                mod_name,
                str(mod_data.get("operations_count", 0)),
                str(mod_data.get("total_measurements", 0)),
                str(mod_data.get("error_count", 0)),
                _fmt_value(lat.get("mean", 0), "ms") if lat else "-",
                _fmt_value(lat.get("p95", 0), "ms") if lat else "-",
                _fmt_value(lat.get("p99", 0), "ms") if lat else "-",
            ])
        _print_table(headers, rows)

    # ------------------------------------------------------------------
    elif args.command == "module":
        report = _run_sync(bench.get_module_report(module=args.name, period=args.period))
        print(f"\n=== Module Report: {args.name} ({args.period}) ===")
        print(f"Generated: {report['generated_at']}\n")

        operations = report.get("operations", {})
        if not operations:
            print("  No data recorded for this module in the given period.")
        else:
            for op_name, metrics in operations.items():
                print(f"  Operation: {op_name}")
                for mt_name, stats in metrics.items():
                    unit = "ms" if mt_name == "latency" else ""
                    print(f"    {mt_name}:")
                    print(f"      count={stats['count']}  "
                          f"mean={_fmt_value(stats['mean'], unit)}  "
                          f"p50={_fmt_value(stats['p50'], unit)}  "
                          f"p95={_fmt_value(stats['p95'], unit)}  "
                          f"p99={_fmt_value(stats['p99'], unit)}  "
                          f"min={_fmt_value(stats['min'], unit)}  "
                          f"max={_fmt_value(stats['max'], unit)}")
                print()

    # ------------------------------------------------------------------
    elif args.command == "slas":
        reports = _run_sync(bench.check_all_slas())
        print(f"\n=== SLA Status ({len(reports)} definitions) ===\n")

        headers = ["SLA ID", "Status", "Current", "Threshold", "Samples", "Violations", "Description"]
        rows = []
        for r in reports:
            sla = r.sla
            rows.append([
                sla.sla_id,
                _sla_status_label(r.status),
                _fmt_value(r.current_value, _unit_for_metric(sla.metric_type)),
                f"{sla.comparison} {_fmt_value(sla.threshold, _unit_for_metric(sla.metric_type))}",
                str(r.samples),
                str(r.violations_count),
                sla.description[:50],
            ])
        _print_table(headers, rows)

    # ------------------------------------------------------------------
    elif args.command == "violations":
        violations = _run_sync(bench.get_violations(hours=args.hours))
        print(f"\n=== SLA Violations (last {args.hours}h) ===\n")

        if not violations:
            print("  No SLA violations detected.")
        else:
            headers = ["SLA ID", "Current", "Threshold", "Samples", "Description"]
            rows = []
            for v in violations:
                sla = v.sla
                rows.append([
                    sla.sla_id,
                    _fmt_value(v.current_value, _unit_for_metric(sla.metric_type)),
                    f"{sla.comparison} {_fmt_value(sla.threshold, _unit_for_metric(sla.metric_type))}",
                    str(v.samples),
                    sla.description[:60],
                ])
            _print_table(headers, rows)

    # ------------------------------------------------------------------
    elif args.command == "trend":
        trend = _run_sync(bench.get_trend(
            module=args.module,
            operation=args.operation,
            metric_type=args.metric,
            days=args.days,
        ))
        print(f"\n=== Trend: {args.module}:{args.operation} ({args.metric}, {args.days}d) ===\n")

        headers = ["Date", "Mean", "p95", "Min", "Max", "Count", "Source"]
        rows = []
        for entry in trend:
            rows.append([
                entry["date"],
                _fmt_value(entry["mean"], "ms" if args.metric == "latency" else ""),
                _fmt_value(entry["p95"], "ms" if args.metric == "latency" else ""),
                _fmt_value(entry["min"], "ms" if args.metric == "latency" else ""),
                _fmt_value(entry["max"], "ms" if args.metric == "latency" else ""),
                str(entry["count"]),
                entry["source"],
            ])
        _print_table(headers, rows)

    # ------------------------------------------------------------------
    elif args.command == "slowest":
        slowest = _run_sync(bench.get_slowest_operations(
            limit=args.limit, period=args.period,
        ))
        print(f"\n=== Slowest Operations (top {args.limit}, {args.period}) ===\n")

        headers = ["Module", "Operation", "Mean", "p95", "p99", "Max", "Count"]
        rows = []
        for s in slowest:
            rows.append([
                s["module"],
                s["operation"],
                _fmt_value(s["mean_ms"], "ms"),
                _fmt_value(s["p95_ms"], "ms"),
                _fmt_value(s["p99_ms"], "ms"),
                _fmt_value(s["max_ms"], "ms"),
                str(s["count"]),
            ])
        _print_table(headers, rows)

    # ------------------------------------------------------------------
    elif args.command == "errors":
        hotspots = _run_sync(bench.get_error_hotspots(period=args.period))
        print(f"\n=== Error Hotspots ({args.period}) ===\n")

        if not hotspots:
            print("  No errors detected in the given period.")
        else:
            headers = ["Module", "Operation", "Total", "Errors", "Error Rate"]
            rows = []
            for h in hotspots:
                rows.append([
                    h["module"],
                    h["operation"],
                    str(h["total"]),
                    str(h["errors"]),
                    f"{h['error_rate_pct']:.2f}%",
                ])
            _print_table(headers, rows)

    # ------------------------------------------------------------------
    elif args.command == "costs":
        breakdown = _run_sync(bench.get_cost_breakdown(days=args.days))
        print(f"\n=== Cost Breakdown ({args.days}d) ===")
        print(f"Total: ${breakdown['total_cost']:.4f}\n")

        headers = ["Module", "Cost", "Percentage"]
        rows = []
        for mod, data in breakdown.get("modules", {}).items():
            rows.append([
                mod,
                f"${data['cost']:.4f}",
                f"{data['percentage']:.1f}%",
            ])
        _print_table(headers, rows)

    # ------------------------------------------------------------------
    elif args.command == "stats":
        stats = _run_sync(bench.get_stats())
        print("\n=== Performance Benchmarker Stats ===\n")
        print(f"  Generated:              {stats['generated_at']}")
        print(f"  Measurements in memory: {stats['total_measurements_in_memory']}")
        print(f"  Measurement keys:       {stats['measurement_keys']}")
        print(f"  Modules tracked:        {stats['modules_count']} ({', '.join(stats['modules_tracked'])})")
        print(f"  SLAs defined:           {stats['slas_defined']}")
        print(f"  Daily aggregate files:  {stats['daily_aggregate_files']}")
        print(f"  Oldest daily:           {stats['oldest_daily'] or 'N/A'}")
        print(f"  Newest daily:           {stats['newest_daily'] or 'N/A'}")
        print(f"  Disk usage:             {stats['disk_usage_mb']} MB ({stats['json_files_on_disk']} files)")
        print(f"  Max per key:            {stats['max_measurements_per_key']}")
        print(f"  Max daily files:        {stats['max_daily_files']}")

    # ------------------------------------------------------------------
    elif args.command == "flush":
        _run_sync(bench.flush())
        print("All benchmarker data flushed to disk.")

    # ------------------------------------------------------------------
    elif args.command == "purge":
        removed = _run_sync(bench.purge_old(days=args.days))
        print(f"Purged {removed} daily aggregate file(s) older than {args.days} days.")

    else:
        parser.print_help()


def _unit_for_metric(metric_type: str) -> str:
    """Return the display unit for a metric type string."""
    units = {
        MetricType.LATENCY.value: "ms",
        MetricType.THROUGHPUT.value: "req/s",
        MetricType.ERROR_RATE.value: "%",
        MetricType.TOKEN_USAGE.value: "tokens",
        MetricType.COST.value: "$",
        MetricType.QUEUE_DEPTH.value: "items",
    }
    return units.get(metric_type, "")


if __name__ == "__main__":
    main()
