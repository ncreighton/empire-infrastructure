"""
Phone Farm Orchestrator -- OpenClaw Empire Multi-Device Automation

Scales Android phone automation from a single device to a fleet of
physical phones, emulators, and cloud instances.  Discovers devices
via ADB, assigns tasks with intelligent load balancing, runs them in
parallel with concurrency limits, monitors health, and auto-recovers
stale devices.

Architecture:
    PhoneFarm (singleton via get_farm())
      |
      +-- DeviceRegistry      -- track & persist device metadata
      +-- TaskQueue            -- priority queue of pending work
      +-- LoadBalancer         -- strategy-based device selection
      +-- DeviceGroupManager   -- logical device groupings
      +-- HealthMonitor        -- periodic health checks & auto-recovery
      |
      v
    PhoneController(s)         -- one per device (from phone_controller.py)

Data stored under: data/phone_farm/

Usage:
    from src.phone_farm import get_farm

    farm = get_farm()
    await farm.discover_devices()
    task_id = await farm.submit_task("open Chrome and search for Python", app="chrome")
    await farm.execute_queue()

CLI:
    python -m src.phone_farm devices
    python -m src.phone_farm discover
    python -m src.phone_farm submit --description "open Chrome" --app chrome
    python -m src.phone_farm queue
    python -m src.phone_farm status
    python -m src.phone_farm health
    python -m src.phone_farm watchdog
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("phone_farm")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

FARM_DATA_DIR = BASE_DIR / "data" / "phone_farm"
DEVICES_FILE = FARM_DATA_DIR / "devices.json"
TASKS_FILE = FARM_DATA_DIR / "tasks.json"
GROUPS_FILE = FARM_DATA_DIR / "groups.json"
HISTORY_FILE = FARM_DATA_DIR / "history.json"
METRICS_FILE = FARM_DATA_DIR / "metrics.json"

FARM_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default OpenClaw node URL (each device may override)
DEFAULT_NODE_URL = os.getenv("OPENCLAW_NODE_URL", "http://localhost:18789")

# ADB binary
ADB_PATH = os.getenv("ADB_PATH", "adb")

# Health-check interval (seconds) for the watchdog loop
WATCHDOG_INTERVAL = 60

# Maximum tasks kept in history
MAX_HISTORY_ENTRIES = 2000

# Maximum parallel task executions
MAX_PARALLEL = 5

# Health score thresholds
HEALTH_SCORE_GOOD = 80
HEALTH_SCORE_DEGRADED = 50
HEALTH_SCORE_CRITICAL = 20

# Auto-recovery attempt limit per device per hour
MAX_RECOVERY_PER_HOUR = 3


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


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


# ===================================================================
# ENUMS
# ===================================================================

class DeviceType(str, Enum):
    """Type of Android device."""
    PHYSICAL = "physical"
    EMULATOR = "emulator"
    CLOUD = "cloud"


class DeviceStatus(str, Enum):
    """Current status of a device."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Status of a task assignment."""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BalancingStrategy(str, Enum):
    """Load balancing strategies for task assignment."""
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    BEST_FIT = "best_fit"
    AFFINITY = "affinity"


# ===================================================================
# DATA CLASSES
# ===================================================================

@dataclass
class DeviceCapabilities:
    """Hardware and software capabilities of a device."""
    screen_width: int = 1080
    screen_height: int = 2400
    android_version: str = ""
    sdk_version: int = 0
    installed_apps: List[str] = field(default_factory=list)
    total_storage_mb: int = 0
    available_storage_mb: int = 0
    battery_level: int = 100
    is_charging: bool = False
    ram_mb: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DeviceCapabilities:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class DeviceInfo:
    """Metadata for a single device in the farm."""
    device_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    type: DeviceType = DeviceType.PHYSICAL
    connection_string: str = ""       # ADB serial or IP:port
    status: DeviceStatus = DeviceStatus.OFFLINE
    current_task: Optional[str] = None
    capabilities: DeviceCapabilities = field(default_factory=DeviceCapabilities)
    last_seen: Optional[str] = None
    health_score: int = 100           # 0-100
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_url: str = DEFAULT_NODE_URL
    # Tracking fields
    created_at: str = field(default_factory=_now_iso)
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_task_duration_ms: float = 0.0
    last_error: Optional[str] = None
    recovery_attempts: int = 0
    last_recovery: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            self.type = DeviceType(self.type)
        if isinstance(self.status, str):
            self.status = DeviceStatus(self.status)
        if isinstance(self.capabilities, dict):
            self.capabilities = DeviceCapabilities.from_dict(self.capabilities)

    @property
    def success_rate(self) -> float:
        """Percentage of completed tasks (0.0 - 1.0)."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total

    @property
    def avg_task_duration_ms(self) -> float:
        """Average task duration in milliseconds."""
        if self.tasks_completed == 0:
            return 0.0
        return self.total_task_duration_ms / self.tasks_completed

    @property
    def is_available(self) -> bool:
        """Whether this device can accept new tasks."""
        return (
            self.status == DeviceStatus.ONLINE
            and self.current_task is None
            and self.health_score >= HEALTH_SCORE_CRITICAL
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        d["status"] = self.status.value
        d["capabilities"] = self.capabilities.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> DeviceInfo:
        data = dict(data)
        if "type" in data and isinstance(data["type"], str):
            data["type"] = DeviceType(data["type"])
        if "status" in data and isinstance(data["status"], str):
            data["status"] = DeviceStatus(data["status"])
        if "capabilities" in data and isinstance(data["capabilities"], dict):
            data["capabilities"] = DeviceCapabilities.from_dict(data["capabilities"])
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class TaskAssignment:
    """A task to be executed on a device."""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    device_id: Optional[str] = None
    task_description: str = ""
    app: str = ""                     # target app (for capability matching)
    priority: int = 3                 # 1 (highest) - 5 (lowest)
    status: TaskStatus = TaskStatus.QUEUED
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 2
    preferred_device: Optional[str] = None
    device_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds, or 0 if not yet complete."""
        if self.started_at and self.completed_at:
            start = _parse_iso(self.started_at)
            end = _parse_iso(self.completed_at)
            if start and end:
                return (end - start).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TaskAssignment:
        data = dict(data)
        if "status" in data and isinstance(data["status"], str):
            data["status"] = TaskStatus(data["status"])
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class DeviceGroup:
    """A logical grouping of devices."""
    group_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    device_ids: List[str] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DeviceGroup:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ===================================================================
# LOAD BALANCER
# ===================================================================

class LoadBalancer:
    """
    Selects the best device for a task based on a configurable strategy.

    Strategies:
        ROUND_ROBIN — simple rotation through available devices
        LEAST_BUSY  — device with fewest active tasks
        BEST_FIT    — highest health score + lowest error rate for the app
        AFFINITY    — prefer device that last ran this app successfully
    """

    def __init__(self) -> None:
        self._round_robin_index: int = 0
        self._affinity_map: Dict[str, str] = {}   # app -> last_device_id

    def select(
        self,
        strategy: BalancingStrategy,
        devices: List[DeviceInfo],
        task: TaskAssignment,
        active_tasks: Dict[str, List[str]],
    ) -> Optional[DeviceInfo]:
        """
        Select the best device for the given task.

        Args:
            strategy:      The balancing strategy to use
            devices:        List of all registered devices
            task:          The task being assigned
            active_tasks:  Map of device_id -> list of active task_ids

        Returns:
            The selected DeviceInfo, or None if no device is available.
        """
        available = self._filter_available(devices, task)
        if not available:
            return None

        if strategy == BalancingStrategy.ROUND_ROBIN:
            return self._round_robin(available)
        elif strategy == BalancingStrategy.LEAST_BUSY:
            return self._least_busy(available, active_tasks)
        elif strategy == BalancingStrategy.BEST_FIT:
            return self._best_fit(available, task)
        elif strategy == BalancingStrategy.AFFINITY:
            return self._affinity(available, task)

        # Fallback
        return available[0] if available else None

    def record_affinity(self, app: str, device_id: str) -> None:
        """Record that a device successfully ran an app (for affinity strategy)."""
        if app:
            self._affinity_map[app.lower()] = device_id

    def _filter_available(
        self, devices: List[DeviceInfo], task: TaskAssignment
    ) -> List[DeviceInfo]:
        """Filter devices to only those that can accept the task."""
        candidates: List[DeviceInfo] = []
        for dev in devices:
            if not dev.is_available:
                continue
            # Check preferred device
            if task.preferred_device and dev.device_id != task.preferred_device:
                continue
            # Check tag requirements
            if task.device_tags:
                if not all(tag in dev.tags for tag in task.device_tags):
                    continue
            # Check app capability (if task specifies an app)
            if task.app and dev.capabilities.installed_apps:
                app_lower = task.app.lower()
                has_app = any(
                    app_lower in a.lower()
                    for a in dev.capabilities.installed_apps
                )
                if not has_app:
                    # Still include but with lower priority; the app might
                    # be installed under a different name or installable
                    pass
            candidates.append(dev)

        # If preferred_device was set but not found, fall back to all
        if not candidates and task.preferred_device:
            task_no_pref = TaskAssignment(
                task_id=task.task_id,
                task_description=task.task_description,
                app=task.app,
                priority=task.priority,
                device_tags=task.device_tags,
            )
            return self._filter_available(devices, task_no_pref)

        return candidates

    def _round_robin(self, devices: List[DeviceInfo]) -> DeviceInfo:
        """Simple round-robin selection."""
        idx = self._round_robin_index % len(devices)
        self._round_robin_index += 1
        return devices[idx]

    def _least_busy(
        self,
        devices: List[DeviceInfo],
        active_tasks: Dict[str, List[str]],
    ) -> DeviceInfo:
        """Select the device with fewest active tasks."""
        def load(dev: DeviceInfo) -> int:
            return len(active_tasks.get(dev.device_id, []))

        return min(devices, key=load)

    def _best_fit(
        self, devices: List[DeviceInfo], task: TaskAssignment
    ) -> DeviceInfo:
        """
        Select the device with the best composite score:
            score = health_score * 0.5 + success_rate * 100 * 0.3 + (100 - error_penalty) * 0.2

        For tasks specifying an app, devices with the app installed get a bonus.
        """
        def score(dev: DeviceInfo) -> float:
            base = dev.health_score * 0.5
            base += dev.success_rate * 100 * 0.3
            error_penalty = min(dev.tasks_failed * 5, 100)
            base += (100 - error_penalty) * 0.2

            # App bonus
            if task.app and dev.capabilities.installed_apps:
                app_lower = task.app.lower()
                if any(app_lower in a.lower() for a in dev.capabilities.installed_apps):
                    base += 15

            return base

        return max(devices, key=score)

    def _affinity(
        self, devices: List[DeviceInfo], task: TaskAssignment
    ) -> DeviceInfo:
        """
        Prefer the device that last ran this app successfully.
        Falls back to best_fit if no affinity match exists.
        """
        if task.app:
            preferred_id = self._affinity_map.get(task.app.lower())
            if preferred_id:
                for dev in devices:
                    if dev.device_id == preferred_id:
                        return dev

        return self._best_fit(devices, task)


# ===================================================================
# PHONE FARM — Main orchestrator
# ===================================================================

class PhoneFarm:
    """
    Multi-device phone farm orchestrator.

    Manages device discovery, task distribution, parallel execution,
    device groups, monitoring, and auto-recovery.  Delegates individual
    device control to PhoneController instances.
    """

    def __init__(self) -> None:
        self._devices: Dict[str, DeviceInfo] = {}
        self._tasks: Dict[str, TaskAssignment] = {}
        self._groups: Dict[str, DeviceGroup] = {}
        self._history: List[dict] = []
        self._metrics: Dict[str, Any] = {}
        self._active_tasks: Dict[str, List[str]] = defaultdict(list)

        self._balancer = LoadBalancer()
        self._strategy: BalancingStrategy = BalancingStrategy.BEST_FIT
        self._semaphore = asyncio.Semaphore(MAX_PARALLEL)
        self._watchdog_task: Optional[asyncio.Task] = None
        self._watchdog_running: bool = False
        self._recovery_log: Dict[str, List[float]] = defaultdict(list)

        # Load persisted state
        self._load_all()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all persisted state from disk."""
        # Devices
        raw_devices = _load_json(DEVICES_FILE, {})
        for did, ddata in raw_devices.items():
            try:
                self._devices[did] = DeviceInfo.from_dict(ddata)
            except Exception as exc:
                logger.warning("Failed to load device %s: %s", did, exc)

        # Tasks (only non-terminal tasks)
        raw_tasks = _load_json(TASKS_FILE, {})
        for tid, tdata in raw_tasks.items():
            try:
                task = TaskAssignment.from_dict(tdata)
                if task.status in (TaskStatus.QUEUED, TaskStatus.ASSIGNED, TaskStatus.RUNNING):
                    self._tasks[tid] = task
            except Exception as exc:
                logger.warning("Failed to load task %s: %s", tid, exc)

        # Groups
        raw_groups = _load_json(GROUPS_FILE, {})
        for gid, gdata in raw_groups.items():
            try:
                self._groups[gid] = DeviceGroup.from_dict(gdata)
            except Exception as exc:
                logger.warning("Failed to load group %s: %s", gid, exc)

        # History
        raw_history = _load_json(HISTORY_FILE, [])
        if isinstance(raw_history, list):
            self._history = raw_history[-MAX_HISTORY_ENTRIES:]
        else:
            self._history = []

        # Metrics
        self._metrics = _load_json(METRICS_FILE, {})

        logger.info(
            "Farm loaded: %d devices, %d pending tasks, %d groups, %d history entries",
            len(self._devices), len(self._tasks), len(self._groups), len(self._history),
        )

    def _save_devices(self) -> None:
        data = {did: dev.to_dict() for did, dev in self._devices.items()}
        _save_json(DEVICES_FILE, data)

    def _save_tasks(self) -> None:
        data = {tid: task.to_dict() for tid, task in self._tasks.items()}
        _save_json(TASKS_FILE, data)

    def _save_groups(self) -> None:
        data = {gid: grp.to_dict() for gid, grp in self._groups.items()}
        _save_json(GROUPS_FILE, data)

    def _save_history(self) -> None:
        self._history = self._history[-MAX_HISTORY_ENTRIES:]
        _save_json(HISTORY_FILE, self._history)

    def _save_metrics(self) -> None:
        _save_json(METRICS_FILE, self._metrics)

    def _record_history(self, task: TaskAssignment) -> None:
        """Append a completed/failed task to history."""
        self._history.append(task.to_dict())
        self._save_history()

    # ==================================================================
    # DEVICE REGISTRY
    # ==================================================================

    async def discover_devices(self) -> List[DeviceInfo]:
        """
        Run ``adb devices`` to find connected Android devices and
        register any new ones in the farm.

        Returns the list of newly discovered devices.
        """
        discovered: List[DeviceInfo] = []
        try:
            proc = await asyncio.create_subprocess_exec(
                ADB_PATH, "devices", "-l",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            output = stdout.decode("utf-8", errors="replace")
        except FileNotFoundError:
            logger.error("ADB binary not found at '%s'. Install ADB or set ADB_PATH.", ADB_PATH)
            return []
        except asyncio.TimeoutError:
            logger.error("ADB devices command timed out.")
            return []
        except Exception as exc:
            logger.error("Failed to run adb devices: %s", exc)
            return []

        # Parse output lines like:
        #   emulator-5554    device  product:sdk_gphone64_x86_64 model:...
        #   192.168.1.100:5555  device  product:...
        #   R5CT123ABCD      device  usb:1-1 product:...
        for line in output.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            serial = parts[0]
            state = parts[1]

            if state != "device":
                logger.debug("Skipping non-ready device %s (state=%s)", serial, state)
                continue

            # Check if already registered
            existing = self._find_device_by_serial(serial)
            if existing:
                existing.status = DeviceStatus.ONLINE
                existing.last_seen = _now_iso()
                logger.debug("Known device back online: %s (%s)", existing.name, serial)
                continue

            # Determine device type
            if serial.startswith("emulator-"):
                dev_type = DeviceType.EMULATOR
            elif re.match(r"\d+\.\d+\.\d+\.\d+:\d+", serial):
                dev_type = DeviceType.CLOUD
            else:
                dev_type = DeviceType.PHYSICAL

            # Extract model from -l output
            model = ""
            for p in parts[2:]:
                if p.startswith("model:"):
                    model = p.split(":", 1)[1]
                    break

            name = model or serial
            dev = DeviceInfo(
                name=name,
                type=dev_type,
                connection_string=serial,
                status=DeviceStatus.ONLINE,
                last_seen=_now_iso(),
            )

            # Query capabilities
            caps = await self._query_device_capabilities(serial)
            if caps:
                dev.capabilities = caps

            self._devices[dev.device_id] = dev
            discovered.append(dev)
            logger.info(
                "Discovered new device: %s (id=%s, type=%s, serial=%s)",
                dev.name, dev.device_id, dev.type.value, serial,
            )

        self._save_devices()
        return discovered

    def _find_device_by_serial(self, serial: str) -> Optional[DeviceInfo]:
        """Look up a device by its ADB serial / connection string."""
        for dev in self._devices.values():
            if dev.connection_string == serial:
                return dev
        return None

    async def _query_device_capabilities(self, serial: str) -> Optional[DeviceCapabilities]:
        """Query device hardware/software details via ADB."""
        caps = DeviceCapabilities()
        try:
            # Screen resolution
            out = await self._adb_shell(serial, "wm size")
            match = re.search(r"(\d+)x(\d+)", out)
            if match:
                caps.screen_width = int(match.group(1))
                caps.screen_height = int(match.group(2))

            # Android version
            out = await self._adb_shell(serial, "getprop ro.build.version.release")
            caps.android_version = out.strip()

            # SDK version
            out = await self._adb_shell(serial, "getprop ro.build.version.sdk")
            try:
                caps.sdk_version = int(out.strip())
            except ValueError:
                pass

            # Battery
            out = await self._adb_shell(serial, "dumpsys battery")
            level_match = re.search(r"level:\s*(\d+)", out)
            if level_match:
                caps.battery_level = int(level_match.group(1))
            charging_match = re.search(r"AC powered:\s*(true|false)", out, re.IGNORECASE)
            usb_match = re.search(r"USB powered:\s*(true|false)", out, re.IGNORECASE)
            caps.is_charging = (
                (charging_match and charging_match.group(1).lower() == "true")
                or (usb_match and usb_match.group(1).lower() == "true")
            ) or False

            # Storage
            out = await self._adb_shell(serial, "df /data")
            df_lines = out.strip().splitlines()
            if len(df_lines) >= 2:
                df_parts = df_lines[1].split()
                if len(df_parts) >= 4:
                    try:
                        caps.total_storage_mb = int(df_parts[1]) // 1024
                        caps.available_storage_mb = int(df_parts[3]) // 1024
                    except (ValueError, IndexError):
                        pass

            # Installed packages (just names, limit to 200 for performance)
            out = await self._adb_shell(serial, "pm list packages -3")
            packages = []
            for pline in out.strip().splitlines()[:200]:
                pline = pline.strip()
                if pline.startswith("package:"):
                    packages.append(pline[8:])
            caps.installed_apps = packages

            # RAM
            out = await self._adb_shell(serial, "cat /proc/meminfo")
            mem_match = re.search(r"MemTotal:\s*(\d+)\s*kB", out)
            if mem_match:
                caps.ram_mb = int(mem_match.group(1)) // 1024

            return caps

        except Exception as exc:
            logger.warning("Failed to query capabilities for %s: %s", serial, exc)
            return None

    async def _adb_shell(self, serial: str, command: str, timeout: float = 10) -> str:
        """Run an ADB shell command against a specific device serial."""
        try:
            proc = await asyncio.create_subprocess_exec(
                ADB_PATH, "-s", serial, "shell", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            logger.debug("ADB shell timed out: %s on %s", command, serial)
            return ""
        except Exception as exc:
            logger.debug("ADB shell failed: %s on %s: %s", command, serial, exc)
            return ""

    def add_device(
        self,
        name: str,
        connection_string: str,
        device_type: DeviceType = DeviceType.PHYSICAL,
        tags: Optional[List[str]] = None,
        node_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeviceInfo:
        """Manually register a device in the farm."""
        dev = DeviceInfo(
            name=name,
            type=device_type,
            connection_string=connection_string,
            status=DeviceStatus.OFFLINE,
            tags=tags or [],
            node_url=node_url or DEFAULT_NODE_URL,
            metadata=metadata or {},
        )
        self._devices[dev.device_id] = dev
        self._save_devices()
        logger.info("Added device: %s (id=%s, serial=%s)", name, dev.device_id, connection_string)
        return dev

    def remove_device(self, device_id: str) -> bool:
        """Remove a device from the farm. Returns True if found and removed."""
        if device_id not in self._devices:
            logger.warning("Cannot remove device %s: not found.", device_id)
            return False
        dev = self._devices.pop(device_id)
        # Remove from any groups
        for group in self._groups.values():
            if device_id in group.device_ids:
                group.device_ids.remove(device_id)
        self._save_devices()
        self._save_groups()
        logger.info("Removed device: %s (id=%s)", dev.name, device_id)
        return True

    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """Get a device by ID."""
        return self._devices.get(device_id)

    def get_device_by_name(self, name: str) -> Optional[DeviceInfo]:
        """Look up a device by its human-readable name (case-insensitive)."""
        name_lower = name.lower()
        for dev in self._devices.values():
            if dev.name.lower() == name_lower:
                return dev
        return None

    def list_devices(
        self,
        status: Optional[DeviceStatus] = None,
        device_type: Optional[DeviceType] = None,
        tag: Optional[str] = None,
    ) -> List[DeviceInfo]:
        """List all devices, optionally filtered."""
        devices = list(self._devices.values())
        if status is not None:
            devices = [d for d in devices if d.status == status]
        if device_type is not None:
            devices = [d for d in devices if d.type == device_type]
        if tag is not None:
            devices = [d for d in devices if tag in d.tags]
        devices.sort(key=lambda d: d.name)
        return devices

    # ==================================================================
    # HEALTH CHECKS
    # ==================================================================

    async def health_check(self, device_id: str) -> Dict[str, Any]:
        """
        Run a comprehensive health check on a single device.

        Checks: ADB connectivity, battery, storage, screen responsiveness.
        Updates the device's health_score and status accordingly.

        Returns a dict with check results.
        """
        dev = self._devices.get(device_id)
        if dev is None:
            return {"device_id": device_id, "error": "Device not found", "health_score": 0}

        result: Dict[str, Any] = {
            "device_id": device_id,
            "name": dev.name,
            "checks": {},
            "timestamp": _now_iso(),
        }

        score = 100
        serial = dev.connection_string

        # Check 1: ADB connectivity
        try:
            out = await self._adb_shell(serial, "echo ping", timeout=5)
            adb_ok = "ping" in out
            result["checks"]["adb_connectivity"] = adb_ok
            if not adb_ok:
                score -= 50
        except Exception:
            result["checks"]["adb_connectivity"] = False
            score -= 50

        if not result["checks"].get("adb_connectivity", False):
            dev.status = DeviceStatus.OFFLINE
            dev.health_score = max(0, score)
            self._save_devices()
            result["health_score"] = dev.health_score
            result["status"] = dev.status.value
            return result

        # Check 2: Battery level
        caps = await self._query_device_capabilities(serial)
        if caps:
            dev.capabilities = caps
            result["checks"]["battery_level"] = caps.battery_level
            result["checks"]["is_charging"] = caps.is_charging

            if caps.battery_level < 10 and not caps.is_charging:
                score -= 30
                result["checks"]["battery_warning"] = "Critical: <10% and not charging"
            elif caps.battery_level < 20 and not caps.is_charging:
                score -= 15
                result["checks"]["battery_warning"] = "Low: <20% and not charging"

            # Check 3: Storage
            result["checks"]["storage_total_mb"] = caps.total_storage_mb
            result["checks"]["storage_available_mb"] = caps.available_storage_mb
            if caps.total_storage_mb > 0:
                usage_pct = 1.0 - (caps.available_storage_mb / caps.total_storage_mb)
                result["checks"]["storage_usage_pct"] = round(usage_pct * 100, 1)
                if usage_pct > 0.95:
                    score -= 25
                    result["checks"]["storage_warning"] = "Critical: >95% full"
                elif usage_pct > 0.90:
                    score -= 10
                    result["checks"]["storage_warning"] = "Warning: >90% full"

        # Check 4: Screen responsiveness (try to get window manager info)
        try:
            out = await self._adb_shell(serial, "dumpsys window | head -5", timeout=5)
            screen_ok = len(out.strip()) > 0
            result["checks"]["screen_responsive"] = screen_ok
            if not screen_ok:
                score -= 20
        except Exception:
            result["checks"]["screen_responsive"] = False
            score -= 20

        # Check 5: Error rate penalty
        if dev.tasks_completed + dev.tasks_failed > 5:
            if dev.success_rate < 0.5:
                score -= 20
                result["checks"]["error_rate_warning"] = f"High failure rate: {1 - dev.success_rate:.0%}"
            elif dev.success_rate < 0.8:
                score -= 10

        # Apply score
        dev.health_score = max(0, min(100, score))
        dev.last_seen = _now_iso()

        # Update status
        if dev.current_task:
            dev.status = DeviceStatus.BUSY
        elif dev.health_score >= HEALTH_SCORE_DEGRADED:
            dev.status = DeviceStatus.ONLINE
        else:
            dev.status = DeviceStatus.ERROR

        result["health_score"] = dev.health_score
        result["status"] = dev.status.value

        self._save_devices()
        logger.info(
            "Health check: %s — score=%d, status=%s",
            dev.name, dev.health_score, dev.status.value,
        )
        return result

    async def health_check_all(self) -> List[Dict[str, Any]]:
        """Run health checks on all registered devices in parallel."""
        if not self._devices:
            logger.info("No devices registered. Run 'discover' first.")
            return []

        tasks = [
            self.health_check(device_id)
            for device_id in list(self._devices.keys())
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        checked: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Health check error: %s", r)
            elif isinstance(r, dict):
                checked.append(r)

        logger.info(
            "Health check complete: %d devices checked, %d online",
            len(checked),
            sum(1 for c in checked if c.get("status") == DeviceStatus.ONLINE.value),
        )
        return checked

    # ==================================================================
    # TASK DISTRIBUTION
    # ==================================================================

    async def submit_task(
        self,
        description: str,
        app: str = "",
        priority: int = 3,
        preferred_device: Optional[str] = None,
        device_tags: Optional[List[str]] = None,
        max_retries: int = 2,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a new task to the queue.

        Returns the task_id.
        """
        task = TaskAssignment(
            task_description=description,
            app=app,
            priority=priority,
            preferred_device=preferred_device,
            device_tags=device_tags or [],
            max_retries=max_retries,
            metadata=metadata or {},
        )
        self._tasks[task.task_id] = task
        self._save_tasks()
        logger.info(
            "Task submitted: %s (id=%s, priority=%d, app=%s)",
            description[:60], task.task_id, priority, app,
        )
        return task.task_id

    async def batch_submit(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """
        Submit multiple tasks at once.

        Each dict should have at minimum: description
        Optional keys: app, priority, preferred_device, device_tags, max_retries

        Returns list of task_ids.
        """
        task_ids: List[str] = []
        for spec in tasks:
            tid = await self.submit_task(
                description=spec.get("description", ""),
                app=spec.get("app", ""),
                priority=spec.get("priority", 3),
                preferred_device=spec.get("preferred_device"),
                device_tags=spec.get("device_tags", []),
                max_retries=spec.get("max_retries", 2),
                metadata=spec.get("metadata", {}),
            )
            task_ids.append(tid)
        logger.info("Batch submitted %d tasks.", len(task_ids))
        return task_ids

    def assign_task(self, task_id: str) -> Optional[str]:
        """
        Assign a queued task to the best available device using the
        current load balancing strategy.

        Returns the device_id, or None if no device is available.
        """
        task = self._tasks.get(task_id)
        if task is None:
            logger.warning("Cannot assign task %s: not found.", task_id)
            return None
        if task.status != TaskStatus.QUEUED:
            logger.warning("Cannot assign task %s: status is %s.", task_id, task.status.value)
            return None

        devices = list(self._devices.values())
        selected = self._balancer.select(
            self._strategy, devices, task, dict(self._active_tasks),
        )
        if selected is None:
            logger.debug("No available device for task %s", task_id)
            return None

        task.device_id = selected.device_id
        task.status = TaskStatus.ASSIGNED
        selected.current_task = task_id
        selected.status = DeviceStatus.BUSY

        self._save_tasks()
        self._save_devices()

        logger.info(
            "Task %s assigned to device %s (%s)",
            task_id, selected.device_id, selected.name,
        )
        return selected.device_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a task."""
        task = self._tasks.get(task_id)
        if task is None:
            # Check history
            for h in reversed(self._history):
                if h.get("task_id") == task_id:
                    return h
            return None
        return task.to_dict()

    def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[TaskAssignment]:
        """List tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status_filter is not None:
            tasks = [t for t in tasks if t.status == status_filter]
        tasks.sort(key=lambda t: (t.priority, t.created_at))
        return tasks

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued or assigned task. Returns True if cancelled."""
        task = self._tasks.get(task_id)
        if task is None:
            logger.warning("Cannot cancel task %s: not found.", task_id)
            return False
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            logger.warning("Cannot cancel task %s: already terminal (%s).", task_id, task.status.value)
            return False

        # Release device if assigned
        if task.device_id:
            dev = self._devices.get(task.device_id)
            if dev and dev.current_task == task_id:
                dev.current_task = None
                if dev.status == DeviceStatus.BUSY:
                    dev.status = DeviceStatus.ONLINE

        task.status = TaskStatus.CANCELLED
        task.completed_at = _now_iso()
        self._record_history(task)

        del self._tasks[task_id]
        self._save_tasks()
        self._save_devices()

        logger.info("Task %s cancelled.", task_id)
        return True

    def _get_queued_tasks(self) -> List[TaskAssignment]:
        """Get all queued tasks sorted by priority (1=highest) then creation time."""
        queued = [t for t in self._tasks.values() if t.status == TaskStatus.QUEUED]
        queued.sort(key=lambda t: (t.priority, t.created_at))
        return queued

    # ==================================================================
    # PARALLEL EXECUTION
    # ==================================================================

    async def execute_queue(self) -> List[Dict[str, Any]]:
        """
        Process the task queue: assign queued tasks to available devices
        and execute them in parallel with semaphore limiting.

        Returns a list of result dicts for all processed tasks.
        """
        results: List[Dict[str, Any]] = []
        queued = self._get_queued_tasks()

        if not queued:
            logger.info("No queued tasks to execute.")
            return results

        logger.info("Processing %d queued tasks...", len(queued))

        # Assign tasks to devices
        assigned_tasks: List[TaskAssignment] = []
        for task in queued:
            device_id = self.assign_task(task.task_id)
            if device_id:
                assigned_tasks.append(task)

        if not assigned_tasks:
            logger.warning(
                "No devices available for %d queued tasks. Run 'discover' or check health.",
                len(queued),
            )
            return results

        logger.info("Assigned %d tasks. Executing...", len(assigned_tasks))

        # Execute in parallel
        coros = [self._execute_task_with_semaphore(task) for task in assigned_tasks]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        for r in raw_results:
            if isinstance(r, Exception):
                logger.error("Task execution error: %s", r)
                results.append({"error": str(r)})
            elif isinstance(r, dict):
                results.append(r)

        return results

    async def execute_on_device(
        self, device_id: str, task_description: str, app: str = ""
    ) -> Dict[str, Any]:
        """
        Run a single task on a specific device (bypasses the queue).

        Returns a result dict.
        """
        dev = self._devices.get(device_id)
        if dev is None:
            return {"error": f"Device {device_id} not found"}

        task = TaskAssignment(
            device_id=device_id,
            task_description=task_description,
            app=app,
            status=TaskStatus.ASSIGNED,
        )
        self._tasks[task.task_id] = task

        dev.current_task = task.task_id
        dev.status = DeviceStatus.BUSY
        self._save_devices()
        self._save_tasks()

        return await self._execute_task(task)

    async def execute_batch(
        self, tasks: List[Dict[str, Any]], max_parallel: int = MAX_PARALLEL
    ) -> List[Dict[str, Any]]:
        """
        Submit and execute multiple tasks in parallel, limiting concurrency
        with a semaphore.

        Each dict in tasks: {description, app, priority, ...}
        Returns list of result dicts.
        """
        # Submit all tasks
        task_ids = await self.batch_submit(tasks)

        # Assign tasks
        assigned: List[TaskAssignment] = []
        for tid in task_ids:
            device_id = self.assign_task(tid)
            if device_id:
                assigned.append(self._tasks[tid])

        if not assigned:
            logger.warning("No devices available for batch of %d tasks.", len(tasks))
            return [{"error": "No devices available"}]

        # Execute with custom semaphore
        sem = asyncio.Semaphore(max_parallel)

        async def _run(task: TaskAssignment) -> Dict[str, Any]:
            async with sem:
                return await self._execute_task(task)

        raw_results = await asyncio.gather(
            *[_run(t) for t in assigned], return_exceptions=True
        )

        results: List[Dict[str, Any]] = []
        for r in raw_results:
            if isinstance(r, Exception):
                results.append({"error": str(r)})
            elif isinstance(r, dict):
                results.append(r)

        return results

    async def wait_for_completion(
        self, task_ids: List[str], timeout: float = 300, poll_interval: float = 2.0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Wait for multiple tasks to reach a terminal state.

        Returns a dict of task_id -> status info.
        """
        deadline = time.monotonic() + timeout
        results: Dict[str, Dict[str, Any]] = {}

        while time.monotonic() < deadline:
            all_done = True
            for tid in task_ids:
                if tid in results:
                    continue
                status = self.get_task_status(tid)
                if status is None:
                    results[tid] = {"error": "Task not found"}
                    continue
                s = status.get("status", "")
                if s in (
                    TaskStatus.COMPLETED.value,
                    TaskStatus.FAILED.value,
                    TaskStatus.CANCELLED.value,
                ):
                    results[tid] = status
                else:
                    all_done = False

            if all_done:
                break
            await asyncio.sleep(poll_interval)

        # Mark remaining as timed-out
        for tid in task_ids:
            if tid not in results:
                results[tid] = {"status": "timeout", "task_id": tid}

        return results

    async def _execute_task_with_semaphore(self, task: TaskAssignment) -> Dict[str, Any]:
        """Execute a task within the global semaphore."""
        async with self._semaphore:
            return await self._execute_task(task)

    async def _execute_task(self, task: TaskAssignment) -> Dict[str, Any]:
        """
        Execute a single task on its assigned device.

        Delegates to PhoneController (imported lazily to avoid circular imports).
        Handles retries, result recording, and device release.
        """
        from src.phone_controller import PhoneController, TaskExecutor

        dev = self._devices.get(task.device_id or "")
        if dev is None:
            task.status = TaskStatus.FAILED
            task.error = f"Assigned device {task.device_id} not found"
            task.completed_at = _now_iso()
            self._finalize_task(task, dev)
            return task.to_dict()

        task.status = TaskStatus.RUNNING
        task.started_at = _now_iso()
        self._active_tasks[dev.device_id].append(task.task_id)
        self._save_tasks()

        logger.info(
            "Executing task %s on device %s (%s): %s",
            task.task_id, dev.device_id, dev.name, task.task_description[:60],
        )

        try:
            controller = PhoneController(
                node_url=dev.node_url,
                node_name=dev.connection_string,
            )

            connected = await controller.connect()
            if not connected:
                raise ConnectionError(
                    f"Cannot connect to device {dev.name} ({dev.connection_string})"
                )

            executor = TaskExecutor(controller)
            plan = await executor.execute(task.task_description)

            task.status = TaskStatus.COMPLETED if plan.status == "completed" else TaskStatus.FAILED
            task.result = json.dumps({
                "plan_status": plan.status,
                "steps_total": len(plan.steps),
                "steps_completed": sum(1 for s in plan.steps if s.completed),
            })

            if plan.status != "completed":
                failed_steps = [
                    s for s in plan.steps
                    if s.result and s.result.error
                ]
                task.error = "; ".join(
                    s.result.error for s in failed_steps[:3]
                ) if failed_steps else "Task plan did not complete"

            await controller.close()

        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = f"{type(exc).__name__}: {exc}"
            logger.error("Task %s failed: %s", task.task_id, task.error)

        task.completed_at = _now_iso()

        # Handle retries
        if task.status == TaskStatus.FAILED and task.retries < task.max_retries:
            task.retries += 1
            logger.info(
                "Retrying task %s (attempt %d/%d)",
                task.task_id, task.retries, task.max_retries,
            )
            # Release current device
            self._release_device(dev, task)
            # Requeue
            task.status = TaskStatus.QUEUED
            task.device_id = None
            task.started_at = None
            task.completed_at = None
            self._save_tasks()
            return {"task_id": task.task_id, "status": "requeued", "retry": task.retries}

        # Record affinity on success
        if task.status == TaskStatus.COMPLETED and task.app:
            self._balancer.record_affinity(task.app, dev.device_id)

        self._finalize_task(task, dev)
        return task.to_dict()

    def _release_device(self, dev: Optional[DeviceInfo], task: TaskAssignment) -> None:
        """Release a device from a task assignment."""
        if dev is None:
            return
        if dev.current_task == task.task_id:
            dev.current_task = None
        if dev.status == DeviceStatus.BUSY:
            dev.status = DeviceStatus.ONLINE
        if task.task_id in self._active_tasks.get(dev.device_id, []):
            self._active_tasks[dev.device_id].remove(task.task_id)

    def _finalize_task(
        self, task: TaskAssignment, dev: Optional[DeviceInfo]
    ) -> None:
        """Finalize a completed/failed task: update device stats, persist."""
        self._release_device(dev, task)

        if dev:
            if task.status == TaskStatus.COMPLETED:
                dev.tasks_completed += 1
                dev.total_task_duration_ms += task.duration_ms
                dev.last_error = None
            elif task.status == TaskStatus.FAILED:
                dev.tasks_failed += 1
                dev.last_error = task.error

        self._record_history(task)

        # Remove from active tasks dict
        if task.task_id in self._tasks:
            del self._tasks[task.task_id]

        self._save_tasks()
        self._save_devices()

    # ==================================================================
    # DEVICE GROUPS
    # ==================================================================

    def create_group(
        self,
        name: str,
        device_ids: Optional[List[str]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> DeviceGroup:
        """Create a new device group."""
        # Validate device IDs
        valid_ids = []
        for did in (device_ids or []):
            if did in self._devices:
                valid_ids.append(did)
            else:
                logger.warning("Device %s not found, skipping for group %s", did, name)

        group = DeviceGroup(
            name=name,
            device_ids=valid_ids,
            description=description,
            tags=tags or [],
        )
        self._groups[group.group_id] = group
        self._save_groups()
        logger.info("Created group: %s (id=%s, %d devices)", name, group.group_id, len(valid_ids))
        return group

    def add_to_group(self, group_id: str, device_id: str) -> bool:
        """Add a device to a group. Returns True if successful."""
        group = self._groups.get(group_id)
        if group is None:
            logger.warning("Group %s not found.", group_id)
            return False
        if device_id not in self._devices:
            logger.warning("Device %s not found.", device_id)
            return False
        if device_id in group.device_ids:
            logger.debug("Device %s already in group %s.", device_id, group.name)
            return True
        group.device_ids.append(device_id)
        self._save_groups()
        logger.info("Added device %s to group %s.", device_id, group.name)
        return True

    def remove_from_group(self, group_id: str, device_id: str) -> bool:
        """Remove a device from a group. Returns True if successful."""
        group = self._groups.get(group_id)
        if group is None:
            logger.warning("Group %s not found.", group_id)
            return False
        if device_id not in group.device_ids:
            logger.debug("Device %s not in group %s.", device_id, group.name)
            return False
        group.device_ids.remove(device_id)
        self._save_groups()
        logger.info("Removed device %s from group %s.", device_id, group.name)
        return True

    def delete_group(self, group_id: str) -> bool:
        """Delete a device group. Returns True if found and deleted."""
        if group_id not in self._groups:
            logger.warning("Group %s not found.", group_id)
            return False
        group = self._groups.pop(group_id)
        self._save_groups()
        logger.info("Deleted group: %s (id=%s)", group.name, group_id)
        return True

    def get_group(self, group_id: str) -> Optional[DeviceGroup]:
        """Get a group by ID."""
        return self._groups.get(group_id)

    def get_group_by_name(self, name: str) -> Optional[DeviceGroup]:
        """Look up a group by name (case-insensitive)."""
        name_lower = name.lower()
        for group in self._groups.values():
            if group.name.lower() == name_lower:
                return group
        return None

    def list_groups(self) -> List[DeviceGroup]:
        """List all device groups."""
        groups = list(self._groups.values())
        groups.sort(key=lambda g: g.name)
        return groups

    async def execute_on_group(
        self, group_id: str, task_description: str, app: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Execute the same task on all devices in a group (e.g., install
        an app on all phones).

        Returns a list of result dicts, one per device.
        """
        group = self._groups.get(group_id)
        if group is None:
            return [{"error": f"Group {group_id} not found"}]

        if not group.device_ids:
            return [{"error": f"Group {group.name} has no devices"}]

        logger.info(
            "Executing on group %s (%d devices): %s",
            group.name, len(group.device_ids), task_description[:60],
        )

        coros = [
            self.execute_on_device(did, task_description, app)
            for did in group.device_ids
            if did in self._devices
        ]

        if not coros:
            return [{"error": "No valid devices in group"}]

        raw_results = await asyncio.gather(*coros, return_exceptions=True)
        results: List[Dict[str, Any]] = []
        for r in raw_results:
            if isinstance(r, Exception):
                results.append({"error": str(r)})
            elif isinstance(r, dict):
                results.append(r)
        return results

    # ==================================================================
    # MONITORING & REPORTS
    # ==================================================================

    def farm_status(self) -> Dict[str, Any]:
        """
        Dashboard view of the entire farm: device summary, active tasks,
        queue depth, health overview.
        """
        devices = list(self._devices.values())
        total = len(devices)
        online = sum(1 for d in devices if d.status == DeviceStatus.ONLINE)
        busy = sum(1 for d in devices if d.status == DeviceStatus.BUSY)
        offline = sum(1 for d in devices if d.status == DeviceStatus.OFFLINE)
        error = sum(1 for d in devices if d.status == DeviceStatus.ERROR)

        queued = sum(1 for t in self._tasks.values() if t.status == TaskStatus.QUEUED)
        running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
        assigned = sum(1 for t in self._tasks.values() if t.status == TaskStatus.ASSIGNED)

        avg_health = (
            sum(d.health_score for d in devices) / total if total > 0 else 0
        )

        # Recent completions (last 24h)
        cutoff = (_now_utc() - timedelta(hours=24)).isoformat()
        recent_completed = sum(
            1 for h in self._history
            if h.get("status") == TaskStatus.COMPLETED.value
            and (h.get("completed_at", "") or "") >= cutoff
        )
        recent_failed = sum(
            1 for h in self._history
            if h.get("status") == TaskStatus.FAILED.value
            and (h.get("completed_at", "") or "") >= cutoff
        )

        return {
            "timestamp": _now_iso(),
            "devices": {
                "total": total,
                "online": online,
                "busy": busy,
                "offline": offline,
                "error": error,
                "avg_health_score": round(avg_health, 1),
            },
            "tasks": {
                "queued": queued,
                "assigned": assigned,
                "running": running,
                "total_pending": queued + assigned + running,
            },
            "last_24h": {
                "completed": recent_completed,
                "failed": recent_failed,
                "success_rate": (
                    round(recent_completed / (recent_completed + recent_failed) * 100, 1)
                    if (recent_completed + recent_failed) > 0 else 0.0
                ),
            },
            "strategy": self._strategy.value,
            "groups": len(self._groups),
            "history_size": len(self._history),
        }

    def device_utilization(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Report how busy each device has been in the last N hours.

        Returns a list of dicts per device with task counts, durations, etc.
        """
        cutoff = (_now_utc() - timedelta(hours=hours)).isoformat()
        device_stats: Dict[str, Dict[str, Any]] = {}

        for dev in self._devices.values():
            device_stats[dev.device_id] = {
                "device_id": dev.device_id,
                "name": dev.name,
                "status": dev.status.value,
                "health_score": dev.health_score,
                "tasks_completed": 0,
                "tasks_failed": 0,
                "total_duration_ms": 0.0,
                "avg_duration_ms": 0.0,
                "utilization_pct": 0.0,
            }

        for h in self._history:
            did = h.get("device_id")
            if not did or did not in device_stats:
                continue
            completed_at = h.get("completed_at", "")
            if completed_at < cutoff:
                continue

            stat = device_stats[did]
            if h.get("status") == TaskStatus.COMPLETED.value:
                stat["tasks_completed"] += 1
                # Calculate duration
                started = h.get("started_at")
                if started and completed_at:
                    s = _parse_iso(started)
                    e = _parse_iso(completed_at)
                    if s and e:
                        stat["total_duration_ms"] += (e - s).total_seconds() * 1000
            elif h.get("status") == TaskStatus.FAILED.value:
                stat["tasks_failed"] += 1

        # Compute averages and utilization
        total_period_ms = hours * 3600 * 1000
        for stat in device_stats.values():
            total_tasks = stat["tasks_completed"] + stat["tasks_failed"]
            if stat["tasks_completed"] > 0:
                stat["avg_duration_ms"] = round(
                    stat["total_duration_ms"] / stat["tasks_completed"], 1
                )
            stat["utilization_pct"] = round(
                (stat["total_duration_ms"] / total_period_ms) * 100, 2
            ) if total_period_ms > 0 else 0.0

        result = sorted(
            device_stats.values(),
            key=lambda x: x["tasks_completed"],
            reverse=True,
        )
        return result

    def failure_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Report failed tasks in the last N hours, grouped by device and
        error type.
        """
        cutoff = (_now_utc() - timedelta(hours=hours)).isoformat()

        by_device: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_error: Dict[str, int] = defaultdict(int)
        total_failures = 0

        for h in self._history:
            if h.get("status") != TaskStatus.FAILED.value:
                continue
            completed_at = h.get("completed_at", "")
            if completed_at < cutoff:
                continue

            total_failures += 1
            did = h.get("device_id", "unknown")
            error = h.get("error", "Unknown error")

            by_device[did].append({
                "task_id": h.get("task_id"),
                "description": h.get("task_description", "")[:60],
                "error": error,
                "timestamp": completed_at,
            })

            # Categorize error
            error_type = self._categorize_error(error)
            by_error[error_type] += 1

        # Map device IDs to names
        by_device_named: Dict[str, Any] = {}
        for did, failures in by_device.items():
            dev = self._devices.get(did)
            name = dev.name if dev else did
            by_device_named[name] = {
                "device_id": did,
                "count": len(failures),
                "failures": failures[-5:],  # last 5
            }

        return {
            "period_hours": hours,
            "total_failures": total_failures,
            "by_device": by_device_named,
            "by_error_type": dict(by_error),
            "timestamp": _now_iso(),
        }

    @staticmethod
    def _categorize_error(error: str) -> str:
        """Categorize an error string into a broad type."""
        error_lower = error.lower()
        if "timeout" in error_lower:
            return "timeout"
        if "connection" in error_lower or "connect" in error_lower:
            return "connection"
        if "not found" in error_lower:
            return "not_found"
        if "permission" in error_lower or "denied" in error_lower:
            return "permission"
        if "element" in error_lower:
            return "ui_element"
        if "vision" in error_lower:
            return "vision"
        return "other"

    def performance_report(self) -> Dict[str, Any]:
        """
        Average task duration by app and by device across all history.
        """
        by_app: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_ms": 0.0, "successes": 0, "failures": 0}
        )
        by_device: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_ms": 0.0, "successes": 0, "failures": 0}
        )

        for h in self._history:
            app = h.get("app", "unknown") or "unknown"
            did = h.get("device_id", "unknown") or "unknown"
            status = h.get("status", "")

            started = h.get("started_at")
            completed = h.get("completed_at")
            duration_ms = 0.0
            if started and completed:
                s = _parse_iso(started)
                e = _parse_iso(completed)
                if s and e:
                    duration_ms = (e - s).total_seconds() * 1000

            for group, key in [(by_app, app), (by_device, did)]:
                group[key]["count"] += 1
                group[key]["total_ms"] += duration_ms
                if status == TaskStatus.COMPLETED.value:
                    group[key]["successes"] += 1
                elif status == TaskStatus.FAILED.value:
                    group[key]["failures"] += 1

        # Compute averages
        def finalize(d: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            for v in d.values():
                if v["count"] > 0:
                    v["avg_ms"] = round(v["total_ms"] / v["count"], 1)
                    v["success_rate"] = round(
                        v["successes"] / v["count"] * 100, 1
                    ) if v["count"] > 0 else 0
                else:
                    v["avg_ms"] = 0.0
                    v["success_rate"] = 0.0
            return dict(d)

        # Replace device IDs with names
        by_device_named: Dict[str, Dict[str, Any]] = {}
        for did, stats in by_device.items():
            dev = self._devices.get(did)
            name = dev.name if dev else did
            stats_copy = dict(stats)
            stats_copy["device_id"] = did
            by_device_named[name] = stats_copy

        return {
            "by_app": finalize(by_app),
            "by_device": finalize(by_device_named),
            "total_tasks": len(self._history),
            "timestamp": _now_iso(),
        }

    # ==================================================================
    # AUTO-RECOVERY
    # ==================================================================

    async def auto_recover(self, device_id: str) -> Dict[str, Any]:
        """
        Detect and attempt to fix common device issues:
            1. Reconnect ADB
            2. Clear stuck foreground app
            3. Reboot device (last resort)

        Returns a result dict with what was attempted and whether it worked.
        """
        dev = self._devices.get(device_id)
        if dev is None:
            return {"device_id": device_id, "error": "Device not found"}

        # Rate-limit recovery attempts
        now = time.monotonic()
        self._recovery_log[device_id] = [
            t for t in self._recovery_log[device_id]
            if now - t < 3600  # keep last hour
        ]
        if len(self._recovery_log[device_id]) >= MAX_RECOVERY_PER_HOUR:
            logger.warning(
                "Recovery rate limit reached for device %s (%d attempts/hour)",
                dev.name, MAX_RECOVERY_PER_HOUR,
            )
            return {
                "device_id": device_id,
                "name": dev.name,
                "action": "rate_limited",
                "success": False,
                "message": f"Max {MAX_RECOVERY_PER_HOUR} recovery attempts per hour exceeded",
            }

        self._recovery_log[device_id].append(now)
        dev.recovery_attempts += 1
        dev.last_recovery = _now_iso()

        serial = dev.connection_string
        result: Dict[str, Any] = {
            "device_id": device_id,
            "name": dev.name,
            "steps": [],
            "success": False,
        }

        # Step 1: Try ADB reconnect
        logger.info("Recovery step 1: ADB reconnect for %s", dev.name)
        try:
            proc = await asyncio.create_subprocess_exec(
                ADB_PATH, "disconnect", serial,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=10)

            await asyncio.sleep(2)

            # Reconnect (for network devices)
            if ":" in serial:
                proc = await asyncio.create_subprocess_exec(
                    ADB_PATH, "connect", serial,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                connect_out = stdout.decode("utf-8", errors="replace")
                result["steps"].append({
                    "action": "adb_reconnect",
                    "output": connect_out.strip(),
                })

            # Test connection
            out = await self._adb_shell(serial, "echo recovery_ping", timeout=5)
            if "recovery_ping" in out:
                result["steps"].append({"action": "ping_test", "success": True})
                dev.status = DeviceStatus.ONLINE
                result["success"] = True
                self._save_devices()
                logger.info("Recovery succeeded at step 1 (ADB reconnect) for %s", dev.name)
                return result
            else:
                result["steps"].append({"action": "ping_test", "success": False})

        except Exception as exc:
            result["steps"].append({"action": "adb_reconnect", "error": str(exc)})

        # Step 2: Force-stop stuck app and go home
        logger.info("Recovery step 2: Clear foreground for %s", dev.name)
        try:
            # Press home
            await self._adb_shell(serial, "input keyevent 3", timeout=5)
            await asyncio.sleep(1)

            # Kill top activity
            out = await self._adb_shell(serial, "dumpsys activity activities | head -20", timeout=5)
            pkg_match = re.search(r"u0\s+(\S+)/", out)
            if pkg_match:
                pkg = pkg_match.group(1)
                await self._adb_shell(serial, f"am force-stop {pkg}", timeout=5)
                result["steps"].append({"action": "force_stop", "package": pkg})

            # Test
            out = await self._adb_shell(serial, "echo recovery_ping2", timeout=5)
            if "recovery_ping2" in out:
                dev.status = DeviceStatus.ONLINE
                result["success"] = True
                self._save_devices()
                logger.info("Recovery succeeded at step 2 (force stop) for %s", dev.name)
                return result

        except Exception as exc:
            result["steps"].append({"action": "clear_foreground", "error": str(exc)})

        # Step 3: Reboot (last resort)
        logger.info("Recovery step 3: Reboot for %s", dev.name)
        try:
            await self._adb_shell(serial, "reboot", timeout=5)
            result["steps"].append({"action": "reboot", "initiated": True})

            # Wait for device to come back
            logger.info("Waiting for %s to reboot...", dev.name)
            reboot_deadline = time.monotonic() + 120  # 2 minutes

            while time.monotonic() < reboot_deadline:
                await asyncio.sleep(10)
                out = await self._adb_shell(serial, "echo reboot_ping", timeout=5)
                if "reboot_ping" in out:
                    dev.status = DeviceStatus.ONLINE
                    dev.last_seen = _now_iso()
                    result["success"] = True
                    result["steps"].append({"action": "reboot_verify", "success": True})
                    self._save_devices()
                    logger.info("Recovery succeeded at step 3 (reboot) for %s", dev.name)

                    # Refresh capabilities after reboot
                    caps = await self._query_device_capabilities(serial)
                    if caps:
                        dev.capabilities = caps
                        self._save_devices()

                    return result

            result["steps"].append({"action": "reboot_verify", "success": False, "message": "Timed out waiting for reboot"})

        except Exception as exc:
            result["steps"].append({"action": "reboot", "error": str(exc)})

        dev.status = DeviceStatus.ERROR
        self._save_devices()
        logger.error("All recovery steps failed for device %s", dev.name)
        return result

    async def reassign_failed(self, task_id: str) -> Optional[str]:
        """
        Reassign a failed task to a different device and requeue it.

        Returns the new task_id (if reassigned), or None.
        """
        # Check history for the failed task
        failed_data = None
        for h in reversed(self._history):
            if h.get("task_id") == task_id and h.get("status") == TaskStatus.FAILED.value:
                failed_data = h
                break

        if failed_data is None:
            # Also check active tasks
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.FAILED:
                failed_data = task.to_dict()
            else:
                logger.warning("Failed task %s not found.", task_id)
                return None

        original_device = failed_data.get("device_id")

        # Create a new task with the same description but excluding the failed device
        new_task_id = await self.submit_task(
            description=failed_data.get("task_description", ""),
            app=failed_data.get("app", ""),
            priority=max(1, failed_data.get("priority", 3) - 1),  # boost priority
            metadata={
                "reassigned_from": task_id,
                "excluded_device": original_device,
            },
        )

        logger.info(
            "Reassigned failed task %s -> new task %s (excluding device %s)",
            task_id, new_task_id, original_device,
        )
        return new_task_id

    async def watchdog(self) -> None:
        """
        Background loop that periodically:
            1. Health-checks all devices
            2. Auto-recovers unhealthy/offline devices
            3. Reassigns tasks stuck on error devices
            4. Processes the queue if tasks are waiting

        Runs every WATCHDOG_INTERVAL seconds until stopped.
        """
        self._watchdog_running = True
        logger.info("Watchdog started. Interval: %ds", WATCHDOG_INTERVAL)

        while self._watchdog_running:
            try:
                # 1. Health check all devices
                await self.health_check_all()

                # 2. Auto-recover unhealthy devices
                for dev in self._devices.values():
                    if dev.status in (DeviceStatus.ERROR, DeviceStatus.OFFLINE):
                        if dev.health_score < HEALTH_SCORE_CRITICAL:
                            logger.info("Watchdog: auto-recovering device %s", dev.name)
                            await self.auto_recover(dev.device_id)

                # 3. Reassign tasks stuck on error devices
                for task in list(self._tasks.values()):
                    if task.status in (TaskStatus.ASSIGNED, TaskStatus.RUNNING):
                        if task.device_id:
                            dev = self._devices.get(task.device_id)
                            if dev and dev.status == DeviceStatus.ERROR:
                                logger.info(
                                    "Watchdog: requeuing task %s (device %s in error state)",
                                    task.task_id, dev.name,
                                )
                                self._release_device(dev, task)
                                task.status = TaskStatus.QUEUED
                                task.device_id = None
                                task.started_at = None

                # 4. Process queue if there are waiting tasks
                queued = self._get_queued_tasks()
                if queued:
                    available = [d for d in self._devices.values() if d.is_available]
                    if available:
                        logger.info(
                            "Watchdog: processing %d queued tasks (%d devices available)",
                            len(queued), len(available),
                        )
                        await self.execute_queue()

                self._save_tasks()
                self._save_devices()

            except Exception as exc:
                logger.error("Watchdog iteration error: %s", exc)

            await asyncio.sleep(WATCHDOG_INTERVAL)

        logger.info("Watchdog stopped.")

    async def start_watchdog(self) -> None:
        """Start the watchdog in the background as an asyncio task."""
        if self._watchdog_task is not None and not self._watchdog_task.done():
            logger.warning("Watchdog is already running.")
            return
        self._watchdog_task = asyncio.create_task(self.watchdog())
        logger.info("Watchdog task started.")

    async def stop_watchdog(self) -> None:
        """Stop the watchdog background task."""
        self._watchdog_running = False
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None
        logger.info("Watchdog task stopped.")

    # ==================================================================
    # CONFIGURATION
    # ==================================================================

    def set_strategy(self, strategy: BalancingStrategy) -> None:
        """Set the load balancing strategy."""
        self._strategy = strategy
        logger.info("Load balancing strategy set to: %s", strategy.value)

    def set_max_parallel(self, n: int) -> None:
        """Update the maximum parallel task execution limit."""
        self._semaphore = asyncio.Semaphore(n)
        logger.info("Max parallel tasks set to: %d", n)

    # ==================================================================
    # SYNC WRAPPERS
    # ==================================================================

    def discover_devices_sync(self) -> List[DeviceInfo]:
        """Synchronous wrapper for discover_devices."""
        return asyncio.run(self.discover_devices())

    def health_check_sync(self, device_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for health_check."""
        return asyncio.run(self.health_check(device_id))

    def health_check_all_sync(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper for health_check_all."""
        return asyncio.run(self.health_check_all())

    def submit_task_sync(self, description: str, **kwargs: Any) -> str:
        """Synchronous wrapper for submit_task."""
        return asyncio.run(self.submit_task(description, **kwargs))

    def execute_queue_sync(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper for execute_queue."""
        return asyncio.run(self.execute_queue())

    def auto_recover_sync(self, device_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for auto_recover."""
        return asyncio.run(self.auto_recover(device_id))


# ===================================================================
# SINGLETON
# ===================================================================

_farm_instance: Optional[PhoneFarm] = None


def get_farm() -> PhoneFarm:
    """
    Get the global PhoneFarm singleton.

    Creates the instance on first call, loading persisted state from disk.
    """
    global _farm_instance
    if _farm_instance is None:
        _farm_instance = PhoneFarm()
    return _farm_instance


# ===================================================================
# CLI HELPERS
# ===================================================================

def _format_table(headers: List[str], rows: List[List[str]], max_col_width: int = 40) -> str:
    """Format a simple ASCII table for CLI output."""
    if not rows:
        return "(no results)"

    truncated_rows = []
    for row in rows:
        truncated_rows.append([
            val[:max_col_width - 3] + "..." if len(val) > max_col_width else val
            for val in row
        ])

    col_widths = [len(h) for h in headers]
    for row in truncated_rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    lines = [fmt.format(*headers)]
    lines.append("  ".join("-" * w for w in col_widths))
    for row in truncated_rows:
        while len(row) < len(headers):
            row.append("")
        lines.append(fmt.format(*row))

    return "\n".join(lines)


# ===================================================================
# CLI COMMANDS
# ===================================================================

def _cmd_devices(args: argparse.Namespace) -> None:
    """List all registered devices."""
    farm = get_farm()
    status_filter = DeviceStatus(args.status) if args.status else None
    devices = farm.list_devices(status=status_filter)

    if not devices:
        print("No devices registered. Run 'discover' to find connected devices.")
        return

    headers = ["ID", "Name", "Type", "Status", "Health", "Serial", "Tasks OK", "Tasks Fail", "Tags"]
    rows = []
    for dev in devices:
        rows.append([
            dev.device_id[:8],
            dev.name[:20],
            dev.type.value,
            dev.status.value,
            str(dev.health_score),
            dev.connection_string[:20],
            str(dev.tasks_completed),
            str(dev.tasks_failed),
            ",".join(dev.tags)[:20],
        ])

    print(f"\n  Phone Farm  --  {len(devices)} device(s)\n")
    print(_format_table(headers, rows))
    print()


def _cmd_discover(args: argparse.Namespace) -> None:
    """Discover connected ADB devices."""
    farm = get_farm()
    print("Scanning for ADB devices...")
    discovered = asyncio.run(farm.discover_devices())

    if discovered:
        print(f"\nDiscovered {len(discovered)} new device(s):")
        for dev in discovered:
            print(f"  + {dev.name} ({dev.type.value}) — {dev.connection_string}")
            print(f"    Android {dev.capabilities.android_version}, "
                  f"{dev.capabilities.screen_width}x{dev.capabilities.screen_height}, "
                  f"Battery: {dev.capabilities.battery_level}%")
    else:
        print("No new devices found.")

    total = len(farm.list_devices())
    print(f"\nTotal registered devices: {total}")


def _cmd_health(args: argparse.Namespace) -> None:
    """Run health checks on all devices."""
    farm = get_farm()
    print("Running health checks...")
    results = asyncio.run(farm.health_check_all())

    if not results:
        print("No devices to check.")
        return

    headers = ["Name", "Status", "Health", "Battery", "Storage", "ADB"]
    rows = []
    for r in results:
        checks = r.get("checks", {})
        rows.append([
            r.get("name", "?"),
            r.get("status", "?"),
            str(r.get("health_score", 0)),
            f"{checks.get('battery_level', '?')}%",
            f"{checks.get('storage_usage_pct', '?')}%",
            "OK" if checks.get("adb_connectivity") else "FAIL",
        ])

    print(f"\n  Health Check Results\n")
    print(_format_table(headers, rows))
    print()


def _cmd_submit(args: argparse.Namespace) -> None:
    """Submit a task to the queue."""
    farm = get_farm()
    task_id = asyncio.run(farm.submit_task(
        description=args.description,
        app=args.app or "",
        priority=args.priority,
    ))
    print(f"Task submitted: {task_id}")
    print(f"  Description: {args.description}")
    print(f"  Priority: {args.priority}")
    if args.app:
        print(f"  App: {args.app}")

    queued = len(farm.list_tasks(TaskStatus.QUEUED))
    print(f"\nQueue depth: {queued} task(s)")


def _cmd_batch(args: argparse.Namespace) -> None:
    """Submit a batch of tasks from a JSON file."""
    farm = get_farm()
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Error loading batch file: {exc}")
        return

    if not isinstance(tasks, list):
        print("Batch file must contain a JSON array of task objects.")
        return

    task_ids = asyncio.run(farm.batch_submit(tasks))
    print(f"Submitted {len(task_ids)} tasks:")
    for tid in task_ids:
        print(f"  {tid}")


def _cmd_queue(args: argparse.Namespace) -> None:
    """Process the task queue."""
    farm = get_farm()
    queued = farm.list_tasks(TaskStatus.QUEUED)

    if not queued:
        print("No tasks in queue.")
        return

    print(f"Processing {len(queued)} queued tasks...")
    results = asyncio.run(farm.execute_queue())

    completed = sum(1 for r in results if r.get("status") == TaskStatus.COMPLETED.value)
    failed = sum(1 for r in results if r.get("status") == TaskStatus.FAILED.value)
    requeued = sum(1 for r in results if r.get("status") == "requeued")

    print(f"\nResults: {completed} completed, {failed} failed, {requeued} requeued")


def _cmd_status(args: argparse.Namespace) -> None:
    """Show farm-wide status dashboard."""
    farm = get_farm()
    status = farm.farm_status()

    d = status["devices"]
    t = status["tasks"]
    h = status["last_24h"]

    print(f"\n  Phone Farm Status")
    print(f"  {'=' * 50}")
    print(f"  Timestamp:    {status['timestamp'][:19]}")
    print(f"  Strategy:     {status['strategy']}")
    print()
    print(f"  DEVICES ({d['total']} total):")
    print(f"    Online:     {d['online']}")
    print(f"    Busy:       {d['busy']}")
    print(f"    Offline:    {d['offline']}")
    print(f"    Error:      {d['error']}")
    print(f"    Avg Health: {d['avg_health_score']}")
    print()
    print(f"  TASKS:")
    print(f"    Queued:     {t['queued']}")
    print(f"    Assigned:   {t['assigned']}")
    print(f"    Running:    {t['running']}")
    print()
    print(f"  LAST 24 HOURS:")
    print(f"    Completed:  {h['completed']}")
    print(f"    Failed:     {h['failed']}")
    print(f"    Success:    {h['success_rate']}%")
    print()
    print(f"  Groups: {status['groups']}  |  History: {status['history_size']}")
    print()


def _cmd_groups(args: argparse.Namespace) -> None:
    """Manage device groups."""
    farm = get_farm()

    if args.create:
        device_ids = args.device_ids.split(",") if args.device_ids else []
        group = farm.create_group(
            name=args.create,
            device_ids=device_ids,
            description=args.group_description or "",
        )
        print(f"Created group: {group.name} (id={group.group_id}, {len(group.device_ids)} devices)")
        return

    if args.add:
        gid, did = args.add.split(",", 1)
        group = farm.get_group(gid) or farm.get_group_by_name(gid)
        if group:
            farm.add_to_group(group.group_id, did)
            print(f"Added device {did} to group {group.name}")
        else:
            print(f"Group not found: {gid}")
        return

    if args.remove_device:
        gid, did = args.remove_device.split(",", 1)
        group = farm.get_group(gid) or farm.get_group_by_name(gid)
        if group:
            farm.remove_from_group(group.group_id, did)
            print(f"Removed device {did} from group {group.name}")
        else:
            print(f"Group not found: {gid}")
        return

    # List groups
    groups = farm.list_groups()
    if not groups:
        print("No device groups. Use --create NAME to create one.")
        return

    headers = ["ID", "Name", "Devices", "Description", "Tags"]
    rows = []
    for g in groups:
        rows.append([
            g.group_id,
            g.name,
            str(len(g.device_ids)),
            g.description[:30],
            ",".join(g.tags)[:20],
        ])

    print(f"\n  Device Groups  --  {len(groups)} group(s)\n")
    print(_format_table(headers, rows))
    print()


def _cmd_utilization(args: argparse.Namespace) -> None:
    """Show device utilization report."""
    farm = get_farm()
    report = farm.device_utilization(hours=args.hours)

    if not report:
        print("No utilization data.")
        return

    headers = ["Device", "Status", "Health", "Completed", "Failed", "Avg (ms)", "Util %"]
    rows = []
    for r in report:
        rows.append([
            r["name"][:20],
            r["status"],
            str(r["health_score"]),
            str(r["tasks_completed"]),
            str(r["tasks_failed"]),
            f"{r['avg_duration_ms']:.0f}",
            f"{r['utilization_pct']:.1f}",
        ])

    print(f"\n  Device Utilization (last {args.hours}h)\n")
    print(_format_table(headers, rows))
    print()


def _cmd_failures(args: argparse.Namespace) -> None:
    """Show failure report."""
    farm = get_farm()
    report = farm.failure_report(hours=args.hours)

    print(f"\n  Failure Report (last {args.hours}h)")
    print(f"  {'=' * 50}")
    print(f"  Total failures: {report['total_failures']}")
    print()

    if report["by_error_type"]:
        print("  By Error Type:")
        for etype, count in sorted(report["by_error_type"].items(), key=lambda x: x[1], reverse=True):
            print(f"    {etype:<20} {count}")
        print()

    if report["by_device"]:
        print("  By Device:")
        for name, info in report["by_device"].items():
            print(f"    {name}: {info['count']} failure(s)")
            for f in info["failures"][-3:]:
                print(f"      - {f['error'][:60]}")
    print()


def _cmd_recover(args: argparse.Namespace) -> None:
    """Auto-recover a device."""
    farm = get_farm()
    dev = farm.get_device(args.device_id) or farm.get_device_by_name(args.device_id)
    if dev is None:
        print(f"Device not found: {args.device_id}")
        return

    print(f"Recovering device: {dev.name} ({dev.device_id})")
    result = asyncio.run(farm.auto_recover(dev.device_id))

    if result.get("success"):
        print(f"  Recovery SUCCEEDED")
    else:
        print(f"  Recovery FAILED")

    for step in result.get("steps", []):
        action = step.get("action", "?")
        success = step.get("success", "?")
        error = step.get("error", "")
        print(f"    [{action}] {'OK' if success else 'FAIL'} {error}")
    print()


def _cmd_watchdog(args: argparse.Namespace) -> None:
    """Start the watchdog background loop."""
    farm = get_farm()

    print(f"Starting watchdog (interval: {WATCHDOG_INTERVAL}s)")
    print(f"Monitoring {len(farm.list_devices())} devices")
    print("Press Ctrl+C to stop.\n")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(farm.watchdog())
    except KeyboardInterrupt:
        print("\nStopping watchdog...")
        farm._watchdog_running = False
    finally:
        loop.close()
        print("Watchdog stopped.")


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def main() -> None:
    """CLI entry point for the phone farm orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="phone_farm",
        description="OpenClaw Empire Phone Farm Orchestrator -- Multi-Device Automation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # devices
    sp_devices = subparsers.add_parser("devices", help="List all registered devices")
    sp_devices.add_argument("--status", choices=["online", "offline", "busy", "error"],
                            default=None, help="Filter by status")
    sp_devices.set_defaults(func=_cmd_devices)

    # discover
    sp_discover = subparsers.add_parser("discover", help="Discover connected ADB devices")
    sp_discover.set_defaults(func=_cmd_discover)

    # health
    sp_health = subparsers.add_parser("health", help="Run health checks on all devices")
    sp_health.set_defaults(func=_cmd_health)

    # submit
    sp_submit = subparsers.add_parser("submit", help="Submit a task to the queue")
    sp_submit.add_argument("--description", "-d", required=True, help="Task description")
    sp_submit.add_argument("--app", help="Target app (for capability matching)")
    sp_submit.add_argument("--priority", type=int, default=3, choices=[1, 2, 3, 4, 5],
                           help="Priority 1 (highest) to 5 (lowest), default: 3")
    sp_submit.set_defaults(func=_cmd_submit)

    # batch
    sp_batch = subparsers.add_parser("batch", help="Submit a batch of tasks from a JSON file")
    sp_batch.add_argument("--file", "-f", required=True, help="Path to JSON file with task array")
    sp_batch.set_defaults(func=_cmd_batch)

    # queue
    sp_queue = subparsers.add_parser("queue", help="Process the task queue")
    sp_queue.set_defaults(func=_cmd_queue)

    # status
    sp_status = subparsers.add_parser("status", help="Show farm-wide status dashboard")
    sp_status.set_defaults(func=_cmd_status)

    # groups
    sp_groups = subparsers.add_parser("groups", help="Manage device groups")
    sp_groups.add_argument("--create", help="Create a group with this name")
    sp_groups.add_argument("--device-ids", help="Comma-separated device IDs for new group")
    sp_groups.add_argument("--group-description", help="Group description")
    sp_groups.add_argument("--add", help="Add device to group: GROUP_ID,DEVICE_ID")
    sp_groups.add_argument("--remove-device", help="Remove device from group: GROUP_ID,DEVICE_ID")
    sp_groups.set_defaults(func=_cmd_groups)

    # utilization
    sp_util = subparsers.add_parser("utilization", help="Show device utilization report")
    sp_util.add_argument("--hours", type=int, default=24, help="Lookback period in hours (default: 24)")
    sp_util.set_defaults(func=_cmd_utilization)

    # failures
    sp_fail = subparsers.add_parser("failures", help="Show failure report")
    sp_fail.add_argument("--hours", type=int, default=24, help="Lookback period in hours (default: 24)")
    sp_fail.set_defaults(func=_cmd_failures)

    # recover
    sp_recover = subparsers.add_parser("recover", help="Auto-recover a device")
    sp_recover.add_argument("device_id", help="Device ID or name")
    sp_recover.set_defaults(func=_cmd_recover)

    # watchdog
    sp_watchdog = subparsers.add_parser("watchdog", help="Start the watchdog background loop")
    sp_watchdog.set_defaults(func=_cmd_watchdog)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
