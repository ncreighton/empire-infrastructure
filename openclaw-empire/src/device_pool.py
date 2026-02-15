"""
Unified Device Pool -- OpenClaw Empire Fleet Management
=========================================================

Merges PhoneFarm (physical ADB devices) and GeeLark (cloud phone profiles)
into a single device fleet with intelligent task routing, niche affinity,
parallel distribution, health monitoring, and cost optimisation.

Architecture:
    DevicePool (singleton via get_device_pool())
      |
      +-- UnifiedDevice registry   -- normalised view across backends
      +-- PoolTask queue           -- priority-ordered, bounded at 5000
      +-- NicheAssignment map      -- niche -> device affinity
      +-- LoadBalanceStrategy      -- 5 selection algorithms
      +-- Watchdog                 -- background health loop
      |
      +-- PhoneFarm (lazy import)  -- physical / emulator discovery
      +-- GeeLarkClient (lazy)     -- cloud phone discovery
      +-- OpenClaw node (lazy)     -- OpenClaw Android node discovery

Data persisted to: data/device_pool/

Usage:
    from src.device_pool import get_device_pool

    pool = get_device_pool()

    # Discover all backends
    counts = await pool.discover_all()

    # Execute a task with niche affinity
    task = await pool.execute_task(
        "open Instagram and like 5 posts",
        niche="witchcraft",
        capabilities=["browser"],
        strategy=LoadBalanceStrategy.NICHE_AFFINITY,
    )

    # Parallel fan-out
    results = await pool.execute_parallel([
        {"description": "post to Facebook", "niche": "witchcraft"},
        {"description": "post to Pinterest", "niche": "smarthome"},
        {"description": "check analytics", "niche": "aiaction"},
    ], max_concurrent=3)

    # Fleet overview
    summary = pool.get_fleet_summary()

CLI:
    python -m src.device_pool discover
    python -m src.device_pool list [--status online] [--type geelark]
    python -m src.device_pool device DEVICE_ID
    python -m src.device_pool execute --desc "open Chrome" [--niche witchcraft]
    python -m src.device_pool parallel --file tasks.json --concurrency 5
    python -m src.device_pool assign-niche --device DEVICE_ID --niche witchcraft
    python -m src.device_pool niches
    python -m src.device_pool health [DEVICE_ID]
    python -m src.device_pool costs [--days 7]
    python -m src.device_pool optimize
    python -m src.device_pool fleet
    python -m src.device_pool queue
    python -m src.device_pool stats
    python -m src.device_pool watchdog [--interval 60]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("device_pool")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")

POOL_DATA_DIR = BASE_DIR / "data" / "device_pool"
POOL_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEVICES_FILE = POOL_DATA_DIR / "devices.json"
TASKS_FILE = POOL_DATA_DIR / "tasks.json"
NICHES_FILE = POOL_DATA_DIR / "niches.json"
HISTORY_FILE = POOL_DATA_DIR / "history.json"
COSTS_FILE = POOL_DATA_DIR / "costs.json"

# Limits
MAX_TASKS = 5000
MAX_HISTORY = 10000
MAX_PARALLEL_DEFAULT = 5

# Health thresholds
HEALTH_EXCELLENT = 90
HEALTH_GOOD = 70
HEALTH_DEGRADED = 50
HEALTH_CRITICAL = 25

# Watchdog
WATCHDOG_INTERVAL_DEFAULT = 60

# Cost defaults (USD per hour)
COST_PHYSICAL = 0.0
COST_EMULATOR = 0.0
COST_GEELARK = 0.05
COST_OPENCLAW = 0.0

UTC = timezone.utc

# Empire niches (from site-registry)
EMPIRE_NICHES = [
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
]

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None


def _gen_id(prefix: str = "dp") -> str:
    """Generate a short unique identifier with prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


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
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------


def _run_sync(coro):
    """Run an async coroutine from synchronous code."""
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
# ENUMS
# ===================================================================


class DeviceType(str, Enum):
    """Backend type for a unified device."""
    PHYSICAL = "physical"
    EMULATOR = "emulator"
    GEELARK = "geelark"
    OPENCLAW = "openclaw"


class DeviceStatus(str, Enum):
    """Current operational status of a device."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"
    WARMING = "warming"
    MAINTENANCE = "maintenance"
    COOLDOWN = "cooldown"


class LoadBalanceStrategy(str, Enum):
    """Algorithm for selecting which device receives a task."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    NICHE_AFFINITY = "niche_affinity"
    COST_OPTIMIZED = "cost_optimized"
    HEALTH_FIRST = "health_first"


class TaskPriority(str, Enum):
    """Task urgency levels, ordered from most to least urgent."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


# Priority ordering for sorting (lower number = higher priority)
_PRIORITY_ORDER: Dict[str, int] = {
    TaskPriority.CRITICAL: 0,
    TaskPriority.HIGH: 1,
    TaskPriority.NORMAL: 2,
    TaskPriority.LOW: 3,
    TaskPriority.BACKGROUND: 4,
}


# ===================================================================
# DATA CLASSES
# ===================================================================


@dataclass
class UnifiedDevice:
    """
    Normalised representation of a device from any backend.

    Provides a uniform interface regardless of whether the underlying
    device is a physical ADB phone, an Android emulator, a GeeLark
    cloud phone, or an OpenClaw Android node.
    """
    device_id: str = field(default_factory=lambda: _gen_id("dev"))
    device_type: DeviceType = DeviceType.PHYSICAL
    name: str = ""
    status: DeviceStatus = DeviceStatus.OFFLINE
    backend_id: str = ""
    ip_address: Optional[str] = None
    port: Optional[int] = None
    os_version: str = "Android 14"
    screen_resolution: str = "1080x2400"
    assigned_niche: Optional[str] = None
    assigned_accounts: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    health_score: float = 100.0
    battery_level: Optional[int] = None
    last_seen: str = field(default_factory=_now_iso)
    created_at: str = field(default_factory=_now_iso)
    cost_per_hour: float = 0.0
    total_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.device_type, str):
            self.device_type = DeviceType(self.device_type)
        if isinstance(self.status, str):
            self.status = DeviceStatus(self.status)

    @property
    def is_available(self) -> bool:
        """Whether this device can accept new tasks."""
        return (
            self.status == DeviceStatus.ONLINE
            and self.current_task is None
            and self.health_score >= HEALTH_CRITICAL
        )

    @property
    def success_rate(self) -> float:
        """Fraction of tasks completed successfully (0.0 - 1.0)."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total

    @property
    def load_score(self) -> float:
        """Combined load metric: lower is better (idle preferred)."""
        base = 0.0
        if self.current_task:
            base += 50.0
        if self.status == DeviceStatus.BUSY:
            base += 30.0
        # Penalise high failure rate
        base += (1.0 - self.success_rate) * 20.0
        return base

    def has_capabilities(self, required: List[str]) -> bool:
        """Check whether this device has all required capabilities."""
        if not required:
            return True
        device_caps = set(c.lower() for c in self.capabilities)
        return all(r.lower() in device_caps for r in required)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["device_type"] = self.device_type.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedDevice:
        data = dict(data)
        if "device_type" in data and isinstance(data["device_type"], str):
            try:
                data["device_type"] = DeviceType(data["device_type"])
            except ValueError:
                data["device_type"] = DeviceType.PHYSICAL
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = DeviceStatus(data["status"])
            except ValueError:
                data["status"] = DeviceStatus.OFFLINE
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class PoolTask:
    """
    A task queued or executed within the device pool.

    Tasks flow through: pending -> assigned -> running -> completed/failed.
    Failed tasks are retried up to ``max_retries`` times.
    """
    task_id: str = field(default_factory=lambda: _gen_id("task"))
    description: str = ""
    device_id: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    niche: Optional[str] = None
    required_capabilities: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.priority, str):
            try:
                self.priority = TaskPriority(self.priority)
            except ValueError:
                self.priority = TaskPriority.NORMAL

    @property
    def priority_order(self) -> int:
        return _PRIORITY_ORDER.get(self.priority, 2)

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            start = _parse_iso(self.started_at)
            end = _parse_iso(self.completed_at)
            if start and end:
                return (end - start).total_seconds() * 1000
        return 0.0

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed")

    @property
    def can_retry(self) -> bool:
        return self.status == "failed" and self.retry_count < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["priority"] = self.priority.value if isinstance(self.priority, TaskPriority) else self.priority
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PoolTask:
        data = dict(data)
        if "priority" in data and isinstance(data["priority"], str):
            try:
                data["priority"] = TaskPriority(data["priority"])
            except ValueError:
                data["priority"] = TaskPriority.NORMAL
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class NicheAssignment:
    """
    Binds a niche (e.g. 'witchcraft') to a set of devices, accounts,
    and platforms with daily task limits.
    """
    niche: str = ""
    device_ids: List[str] = field(default_factory=list)
    accounts: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    daily_task_limit: int = 100
    tasks_today: int = 0
    last_reset: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reset_daily_counter(self) -> None:
        """Reset the daily task counter if a new day has started."""
        last = _parse_iso(self.last_reset)
        now = _now_utc()
        if last is None or last.date() < now.date():
            self.tasks_today = 0
            self.last_reset = _now_iso()

    @property
    def has_capacity(self) -> bool:
        self.reset_daily_counter()
        return self.tasks_today < self.daily_task_limit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NicheAssignment:
        data = dict(data)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ===================================================================
# DEVICE POOL
# ===================================================================


class DevicePool:
    """
    Unified fleet manager that aggregates physical ADB devices,
    emulators, GeeLark cloud phones, and OpenClaw Android nodes
    into a single pool with intelligent task routing.

    Thread-safe singleton accessed via ``get_device_pool()``.
    """

    def __init__(self) -> None:
        # Core registries
        self._devices: Dict[str, UnifiedDevice] = {}
        self._tasks: Dict[str, PoolTask] = {}
        self._niche_assignments: Dict[str, NicheAssignment] = {}
        self._task_queue: List[str] = []  # task_ids sorted by priority

        # Round-robin state
        self._rr_index: int = 0

        # Watchdog state
        self._watchdog_running: bool = False
        self._watchdog_task: Optional[asyncio.Task] = None

        # Cost accumulation
        self._cost_events: List[Dict[str, Any]] = []

        # Load persisted state
        self._load_state()
        logger.info(
            "DevicePool initialised: %d devices, %d tasks, %d niches",
            len(self._devices), len(self._tasks), len(self._niche_assignments),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load all persisted state from disk."""
        # Devices
        raw_devices = _load_json(DEVICES_FILE, default={})
        for did, ddata in raw_devices.items():
            try:
                self._devices[did] = UnifiedDevice.from_dict(ddata)
            except Exception as exc:
                logger.warning("Skipping corrupt device %s: %s", did, exc)

        # Tasks (only keep non-terminal up to MAX_TASKS)
        raw_tasks = _load_json(TASKS_FILE, default={})
        count = 0
        for tid, tdata in raw_tasks.items():
            if count >= MAX_TASKS:
                break
            try:
                self._tasks[tid] = PoolTask.from_dict(tdata)
                count += 1
            except Exception as exc:
                logger.warning("Skipping corrupt task %s: %s", tid, exc)

        # Rebuild task queue from pending tasks
        self._rebuild_task_queue()

        # Niches
        raw_niches = _load_json(NICHES_FILE, default={})
        for nid, ndata in raw_niches.items():
            try:
                self._niche_assignments[nid] = NicheAssignment.from_dict(ndata)
            except Exception as exc:
                logger.warning("Skipping corrupt niche %s: %s", nid, exc)

        # Costs
        self._cost_events = _load_json(COSTS_FILE, default=[])
        if not isinstance(self._cost_events, list):
            self._cost_events = []

    def _save_devices(self) -> None:
        """Persist device registry to disk."""
        data = {did: dev.to_dict() for did, dev in self._devices.items()}
        _save_json(DEVICES_FILE, data)

    def _save_tasks(self) -> None:
        """Persist task registry to disk."""
        data = {tid: t.to_dict() for tid, t in self._tasks.items()}
        _save_json(TASKS_FILE, data)

    def _save_niches(self) -> None:
        """Persist niche assignments to disk."""
        data = {nid: n.to_dict() for nid, n in self._niche_assignments.items()}
        _save_json(NICHES_FILE, data)

    def _save_costs(self) -> None:
        """Persist cost events to disk."""
        _save_json(COSTS_FILE, self._cost_events[-MAX_HISTORY:])

    def _save_all(self) -> None:
        """Persist all state."""
        self._save_devices()
        self._save_tasks()
        self._save_niches()
        self._save_costs()

    # ------------------------------------------------------------------
    # Task queue management
    # ------------------------------------------------------------------

    def _rebuild_task_queue(self) -> None:
        """Rebuild the priority-sorted task queue from pending tasks."""
        pending = [
            tid for tid, t in self._tasks.items()
            if t.status == "pending"
        ]
        pending.sort(key=lambda tid: (
            self._tasks[tid].priority_order,
            self._tasks[tid].created_at,
        ))
        self._task_queue = pending

    def _enqueue_task(self, task: PoolTask) -> None:
        """Add a task to the registry and queue."""
        # Enforce bounded task store
        if len(self._tasks) >= MAX_TASKS:
            self._evict_old_tasks()

        self._tasks[task.task_id] = task

        if task.status == "pending":
            # Insert in priority order
            insert_idx = len(self._task_queue)
            for i, tid in enumerate(self._task_queue):
                existing = self._tasks.get(tid)
                if existing and task.priority_order < existing.priority_order:
                    insert_idx = i
                    break
                if (existing and task.priority_order == existing.priority_order
                        and task.created_at < existing.created_at):
                    insert_idx = i
                    break
            self._task_queue.insert(insert_idx, task.task_id)

        self._save_tasks()

    def _dequeue_task(self) -> Optional[PoolTask]:
        """Pop the highest-priority pending task from the queue."""
        while self._task_queue:
            tid = self._task_queue.pop(0)
            task = self._tasks.get(tid)
            if task and task.status == "pending":
                return task
        return None

    def _evict_old_tasks(self) -> None:
        """Remove the oldest terminal tasks to make room."""
        terminal = [
            (tid, t) for tid, t in self._tasks.items()
            if t.is_terminal
        ]
        terminal.sort(key=lambda x: x[1].created_at)
        to_remove = max(len(terminal) // 2, 100)
        for tid, _ in terminal[:to_remove]:
            self._tasks.pop(tid, None)
        logger.info("Evicted %d old tasks", min(to_remove, len(terminal)))

    # ==================================================================
    # DISCOVERY
    # ==================================================================

    async def discover_all(self) -> Dict[str, int]:
        """
        Discover devices from all backends (PhoneFarm, GeeLark, OpenClaw).
        Returns a dict of counts by device type.
        """
        logger.info("Starting full device discovery across all backends...")
        counts: Dict[str, int] = {}

        try:
            physical = await self.discover_physical()
            counts["physical"] = len(physical)
            logger.info("Discovered %d physical/emulator devices", len(physical))
        except Exception as exc:
            logger.warning("Physical device discovery failed: %s", exc)
            counts["physical"] = 0

        try:
            geelark = await self.discover_geelark()
            counts["geelark"] = len(geelark)
            logger.info("Discovered %d GeeLark cloud phones", len(geelark))
        except Exception as exc:
            logger.warning("GeeLark discovery failed: %s", exc)
            counts["geelark"] = 0

        try:
            openclaw = await self.discover_openclaw()
            counts["openclaw"] = len(openclaw)
            logger.info("Discovered %d OpenClaw nodes", len(openclaw))
        except Exception as exc:
            logger.warning("OpenClaw discovery failed: %s", exc)
            counts["openclaw"] = 0

        counts["total"] = sum(v for k, v in counts.items() if k != "total")
        self._save_devices()
        logger.info("Discovery complete: %s", counts)
        return counts

    def discover_all_sync(self) -> Dict[str, int]:
        """Synchronous wrapper for discover_all."""
        return _run_sync(self.discover_all())

    async def discover_physical(self) -> List[UnifiedDevice]:
        """
        Discover physical and emulator devices from the PhoneFarm module.

        Lazy-imports ``src.phone_farm`` and reads its device registry, converting
        each PhoneFarm DeviceInfo into a UnifiedDevice.
        """
        discovered: List[UnifiedDevice] = []

        try:
            from src.phone_farm import get_farm
            farm = get_farm()
            # Trigger ADB discovery if the farm supports it
            if hasattr(farm, "discover_devices"):
                try:
                    await farm.discover_devices()
                except Exception as exc:
                    logger.debug("PhoneFarm discover_devices() error: %s", exc)

            # Read current device list
            farm_devices = {}
            if hasattr(farm, "_devices"):
                farm_devices = farm._devices
            elif hasattr(farm, "list_devices"):
                farm_devices = {d.device_id: d for d in farm.list_devices()}

            for fid, fdev in farm_devices.items():
                existing = self._find_by_backend("physical", fid) or self._find_by_backend("emulator", fid)
                if existing:
                    # Update existing device
                    dev = self._devices[existing]
                    dev.last_seen = _now_iso()
                    if hasattr(fdev, "status"):
                        status_val = fdev.status.value if hasattr(fdev.status, "value") else str(fdev.status)
                        dev.status = self._map_farm_status(status_val)
                    if hasattr(fdev, "health_score"):
                        dev.health_score = float(fdev.health_score)
                    if hasattr(fdev, "capabilities") and hasattr(fdev.capabilities, "battery_level"):
                        dev.battery_level = fdev.capabilities.battery_level
                    discovered.append(dev)
                else:
                    # Create new unified device
                    dev_type = DeviceType.PHYSICAL
                    if hasattr(fdev, "type"):
                        ftype = fdev.type.value if hasattr(fdev.type, "value") else str(fdev.type)
                        if ftype == "emulator":
                            dev_type = DeviceType.EMULATOR

                    name = getattr(fdev, "name", "") or fid
                    conn = getattr(fdev, "connection_string", "")
                    ip_addr = None
                    port_num = None
                    if conn and ":" in conn:
                        parts = conn.rsplit(":", 1)
                        ip_addr = parts[0]
                        try:
                            port_num = int(parts[1])
                        except ValueError:
                            pass

                    caps = self._extract_farm_capabilities(fdev)
                    os_ver = ""
                    resolution = "1080x2400"
                    if hasattr(fdev, "capabilities"):
                        fc = fdev.capabilities
                        os_ver = getattr(fc, "android_version", "")
                        sw = getattr(fc, "screen_width", 1080)
                        sh = getattr(fc, "screen_height", 2400)
                        resolution = f"{sw}x{sh}"

                    status_val = "offline"
                    if hasattr(fdev, "status"):
                        status_val = fdev.status.value if hasattr(fdev.status, "value") else str(fdev.status)

                    health = float(getattr(fdev, "health_score", 100))
                    battery = None
                    if hasattr(fdev, "capabilities") and hasattr(fdev.capabilities, "battery_level"):
                        battery = fdev.capabilities.battery_level

                    unified = UnifiedDevice(
                        device_id=_gen_id("phy"),
                        device_type=dev_type,
                        name=name,
                        status=self._map_farm_status(status_val),
                        backend_id=fid,
                        ip_address=ip_addr,
                        port=port_num,
                        os_version=os_ver or "Android 14",
                        screen_resolution=resolution,
                        capabilities=caps,
                        health_score=health,
                        battery_level=battery,
                        last_seen=_now_iso(),
                        created_at=_now_iso(),
                        cost_per_hour=COST_PHYSICAL if dev_type == DeviceType.PHYSICAL else COST_EMULATOR,
                        metadata={"source": "phone_farm", "connection_string": conn},
                    )
                    self._devices[unified.device_id] = unified
                    discovered.append(unified)
                    logger.debug("Registered physical device: %s (%s)", unified.name, unified.device_id)

        except ImportError:
            logger.info("phone_farm module not available, skipping physical discovery")
        except Exception as exc:
            logger.error("Error during physical discovery: %s", exc)

        return discovered

    async def discover_geelark(self) -> List[UnifiedDevice]:
        """
        Discover GeeLark cloud phone profiles and convert to UnifiedDevice.

        Lazy-imports ``src.geelark_client`` and reads its profile registry.
        """
        discovered: List[UnifiedDevice] = []

        try:
            from src.geelark_client import get_client
            client = get_client()

            # Read profiles
            profiles = {}
            if hasattr(client, "_profiles"):
                profiles = client._profiles
            elif hasattr(client, "list_profiles"):
                result = client.list_profiles()
                if isinstance(result, list):
                    profiles = {p.profile_id: p for p in result}
                elif isinstance(result, dict):
                    profiles = result

            for pid, prof in profiles.items():
                existing = self._find_by_backend("geelark", pid)
                if existing:
                    dev = self._devices[existing]
                    dev.last_seen = _now_iso()
                    status_val = getattr(prof, "status", "created")
                    dev.status = self._map_geelark_status(status_val)
                    dev.assigned_accounts = list(getattr(prof, "assigned_accounts", []))
                    discovered.append(dev)
                else:
                    name = getattr(prof, "name", "") or pid
                    status_val = getattr(prof, "status", "created")
                    os_ver = getattr(prof, "os_version", "Android 14")
                    resolution = getattr(prof, "screen_resolution", "1080x2340")
                    adb_addr = getattr(prof, "adb_address", "")
                    ip_addr = None
                    port_num = None
                    if adb_addr and ":" in adb_addr:
                        parts = adb_addr.rsplit(":", 1)
                        ip_addr = parts[0]
                        try:
                            port_num = int(parts[1])
                        except ValueError:
                            pass

                    accounts = list(getattr(prof, "assigned_accounts", []))
                    installed = list(getattr(prof, "installed_apps", []))
                    caps = ["browser", "camera", "gps"]
                    if installed:
                        caps.append("apps")
                    if adb_addr:
                        caps.append("adb")

                    unified = UnifiedDevice(
                        device_id=_gen_id("gl"),
                        device_type=DeviceType.GEELARK,
                        name=name,
                        status=self._map_geelark_status(status_val),
                        backend_id=pid,
                        ip_address=ip_addr,
                        port=port_num,
                        os_version=os_ver,
                        screen_resolution=resolution,
                        assigned_accounts=accounts,
                        capabilities=caps,
                        health_score=100.0,
                        last_seen=_now_iso(),
                        created_at=_now_iso(),
                        cost_per_hour=COST_GEELARK,
                        metadata={
                            "source": "geelark",
                            "adb_address": adb_addr,
                            "installed_apps": installed,
                            "group": getattr(prof, "group", ""),
                        },
                    )
                    self._devices[unified.device_id] = unified
                    discovered.append(unified)
                    logger.debug("Registered GeeLark profile: %s (%s)", unified.name, unified.device_id)

        except ImportError:
            logger.info("geelark_client module not available, skipping GeeLark discovery")
        except Exception as exc:
            logger.error("Error during GeeLark discovery: %s", exc)

        return discovered

    async def discover_openclaw(self) -> List[UnifiedDevice]:
        """
        Discover OpenClaw Android nodes by querying the gateway API.

        Attempts to reach the OpenClaw gateway at the configured URL and
        list any registered Android nodes.
        """
        discovered: List[UnifiedDevice] = []
        gateway_url = os.getenv("OPENCLAW_NODE_URL", "http://localhost:18789")

        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Query nodes list
                async with session.get(f"{gateway_url}/api/nodes") as resp:
                    if resp.status != 200:
                        logger.debug("OpenClaw gateway returned %d", resp.status)
                        return discovered
                    nodes_data = await resp.json()

                nodes = nodes_data if isinstance(nodes_data, list) else nodes_data.get("nodes", [])

                for node in nodes:
                    node_id = node.get("id", node.get("name", ""))
                    if not node_id:
                        continue

                    existing = self._find_by_backend("openclaw", node_id)
                    if existing:
                        dev = self._devices[existing]
                        dev.last_seen = _now_iso()
                        dev.status = DeviceStatus.ONLINE if node.get("connected") else DeviceStatus.OFFLINE
                        discovered.append(dev)
                    else:
                        unified = UnifiedDevice(
                            device_id=_gen_id("oc"),
                            device_type=DeviceType.OPENCLAW,
                            name=node.get("name", node_id),
                            status=DeviceStatus.ONLINE if node.get("connected") else DeviceStatus.OFFLINE,
                            backend_id=node_id,
                            ip_address=node.get("ip"),
                            port=node.get("port"),
                            os_version=node.get("os_version", "Android 14"),
                            screen_resolution=node.get("resolution", "1080x2400"),
                            capabilities=node.get("capabilities", ["camera", "gps", "sms", "browser", "adb"]),
                            health_score=100.0,
                            last_seen=_now_iso(),
                            created_at=_now_iso(),
                            cost_per_hour=COST_OPENCLAW,
                            metadata={"source": "openclaw", "gateway_url": gateway_url},
                        )
                        self._devices[unified.device_id] = unified
                        discovered.append(unified)
                        logger.debug("Registered OpenClaw node: %s (%s)", unified.name, unified.device_id)

        except ImportError:
            logger.info("aiohttp not available, skipping OpenClaw discovery")
        except Exception as exc:
            logger.debug("OpenClaw discovery failed: %s", exc)

        return discovered

    def register_device(self, device: UnifiedDevice) -> None:
        """Manually register a device in the pool."""
        self._devices[device.device_id] = device
        self._save_devices()
        logger.info("Registered device: %s (%s)", device.name, device.device_id)

    def remove_device(self, device_id: str) -> bool:
        """Remove a device from the pool. Returns True if found and removed."""
        dev = self._devices.pop(device_id, None)
        if dev:
            # Remove from niche assignments
            for na in self._niche_assignments.values():
                if device_id in na.device_ids:
                    na.device_ids.remove(device_id)
            self._save_devices()
            self._save_niches()
            logger.info("Removed device: %s (%s)", dev.name, device_id)
            return True
        logger.warning("Device not found for removal: %s", device_id)
        return False

    # ------------------------------------------------------------------
    # Backend helpers
    # ------------------------------------------------------------------

    def _find_by_backend(self, device_type: str, backend_id: str) -> Optional[str]:
        """Find a device_id by its backend type and backend-specific ID."""
        for did, dev in self._devices.items():
            if dev.backend_id == backend_id and dev.device_type.value == device_type:
                return did
        return None

    @staticmethod
    def _map_farm_status(status_str: str) -> DeviceStatus:
        """Map a PhoneFarm status string to DeviceStatus."""
        mapping = {
            "online": DeviceStatus.ONLINE,
            "offline": DeviceStatus.OFFLINE,
            "busy": DeviceStatus.BUSY,
            "error": DeviceStatus.ERROR,
        }
        return mapping.get(status_str.lower(), DeviceStatus.OFFLINE)

    @staticmethod
    def _map_geelark_status(status_str: str) -> DeviceStatus:
        """Map a GeeLark profile status to DeviceStatus."""
        mapping = {
            "running": DeviceStatus.ONLINE,
            "stopped": DeviceStatus.OFFLINE,
            "created": DeviceStatus.OFFLINE,
            "suspended": DeviceStatus.MAINTENANCE,
        }
        return mapping.get(status_str.lower(), DeviceStatus.OFFLINE)

    @staticmethod
    def _extract_farm_capabilities(fdev: Any) -> List[str]:
        """Extract capability strings from a PhoneFarm DeviceInfo."""
        caps = ["adb", "browser"]
        if hasattr(fdev, "capabilities"):
            fc = fdev.capabilities
            if getattr(fc, "installed_apps", None):
                caps.append("apps")
        if hasattr(fdev, "tags"):
            tags = getattr(fdev, "tags", [])
            if "camera" in tags:
                caps.append("camera")
            if "gps" in tags:
                caps.append("gps")
            if "sms" in tags:
                caps.append("sms")
        return list(set(caps))
