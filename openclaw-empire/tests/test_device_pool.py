"""Test device_pool -- OpenClaw Empire."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch data directories before imports
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_pool_dirs(tmp_path, monkeypatch):
    """Redirect all device pool I/O to temp directory."""
    pool_dir = tmp_path / "device_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.device_pool.POOL_DATA_DIR", pool_dir)
    monkeypatch.setattr("src.device_pool.DEVICES_FILE", pool_dir / "devices.json")
    monkeypatch.setattr("src.device_pool.TASKS_FILE", pool_dir / "tasks.json")
    monkeypatch.setattr("src.device_pool.NICHES_FILE", pool_dir / "niches.json")
    monkeypatch.setattr("src.device_pool.HISTORY_FILE", pool_dir / "history.json")
    monkeypatch.setattr("src.device_pool.COSTS_FILE", pool_dir / "costs.json")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.device_pool import (
    COST_GEELARK,
    COST_PHYSICAL,
    DevicePool,
    DeviceStatus,
    DeviceType,
    EMPIRE_NICHES,
    HEALTH_CRITICAL,
    HEALTH_DEGRADED,
    HEALTH_EXCELLENT,
    HEALTH_GOOD,
    LoadBalanceStrategy,
    MAX_TASKS,
    NicheAssignment,
    PoolTask,
    TaskPriority,
    UnifiedDevice,
    _PRIORITY_ORDER,
)


# ===========================================================================
# DEVICE TYPE ENUM
# ===========================================================================


class TestDeviceType:

    def test_four_device_types(self):
        assert len(list(DeviceType)) == 4

    def test_values(self):
        expected = {"physical", "emulator", "geelark", "openclaw"}
        assert {dt.value for dt in DeviceType} == expected


# ===========================================================================
# DEVICE STATUS ENUM
# ===========================================================================


class TestDeviceStatus:

    def test_seven_statuses(self):
        assert len(list(DeviceStatus)) == 7

    def test_includes_cooldown(self):
        assert DeviceStatus.COOLDOWN.value == "cooldown"


# ===========================================================================
# LOAD BALANCE STRATEGY ENUM
# ===========================================================================


class TestLoadBalanceStrategy:

    def test_five_strategies(self):
        assert len(list(LoadBalanceStrategy)) == 5

    def test_values(self):
        expected = {"round_robin", "least_loaded", "niche_affinity", "cost_optimized", "health_first"}
        assert {s.value for s in LoadBalanceStrategy} == expected


# ===========================================================================
# TASK PRIORITY ENUM
# ===========================================================================


class TestTaskPriority:

    def test_five_priorities(self):
        assert len(list(TaskPriority)) == 5

    def test_priority_ordering(self):
        assert _PRIORITY_ORDER[TaskPriority.CRITICAL] < _PRIORITY_ORDER[TaskPriority.NORMAL]
        assert _PRIORITY_ORDER[TaskPriority.NORMAL] < _PRIORITY_ORDER[TaskPriority.BACKGROUND]


# ===========================================================================
# UNIFIED DEVICE DATACLASS
# ===========================================================================


class TestUnifiedDevice:

    def test_defaults(self):
        dev = UnifiedDevice()
        assert dev.device_type == DeviceType.PHYSICAL
        assert dev.status == DeviceStatus.OFFLINE
        assert dev.health_score == 100.0

    def test_is_available_when_online(self):
        dev = UnifiedDevice(
            status=DeviceStatus.ONLINE,
            health_score=80.0,
            current_task=None,
        )
        assert dev.is_available is True

    def test_not_available_when_busy(self):
        dev = UnifiedDevice(
            status=DeviceStatus.ONLINE,
            health_score=80.0,
            current_task="some-task",
        )
        assert dev.is_available is False

    def test_not_available_when_offline(self):
        dev = UnifiedDevice(
            status=DeviceStatus.OFFLINE,
            health_score=80.0,
        )
        assert dev.is_available is False

    def test_not_available_when_health_critical(self):
        dev = UnifiedDevice(
            status=DeviceStatus.ONLINE,
            health_score=HEALTH_CRITICAL - 1,  # Below critical threshold
        )
        assert dev.is_available is False

    def test_success_rate_default(self):
        dev = UnifiedDevice()
        assert dev.success_rate == 1.0  # No tasks yet

    def test_success_rate_computed(self):
        dev = UnifiedDevice(tasks_completed=8, tasks_failed=2)
        assert abs(dev.success_rate - 0.8) < 0.01

    def test_load_score_idle(self):
        dev = UnifiedDevice(
            status=DeviceStatus.ONLINE,
            current_task=None,
            tasks_completed=10,
            tasks_failed=0,
        )
        assert dev.load_score < 10.0

    def test_load_score_busy(self):
        dev = UnifiedDevice(
            status=DeviceStatus.BUSY,
            current_task="task-123",
        )
        assert dev.load_score >= 50.0

    def test_has_capabilities_empty_required(self):
        dev = UnifiedDevice(capabilities=["browser", "camera"])
        assert dev.has_capabilities([]) is True

    def test_has_capabilities_match(self):
        dev = UnifiedDevice(capabilities=["browser", "camera", "gps"])
        assert dev.has_capabilities(["browser", "camera"]) is True

    def test_has_capabilities_missing(self):
        dev = UnifiedDevice(capabilities=["browser"])
        assert dev.has_capabilities(["browser", "camera"]) is False

    def test_has_capabilities_case_insensitive(self):
        dev = UnifiedDevice(capabilities=["Browser", "GPS"])
        assert dev.has_capabilities(["browser", "gps"]) is True

    def test_to_dict_roundtrip(self):
        dev = UnifiedDevice(
            device_id="dev-test",
            device_type=DeviceType.GEELARK,
            name="Cloud Phone 1",
            status=DeviceStatus.ONLINE,
            assigned_niche="witchcraft",
            health_score=95.0,
        )
        d = dev.to_dict()
        restored = UnifiedDevice.from_dict(d)
        assert restored.device_id == "dev-test"
        assert restored.device_type == DeviceType.GEELARK
        assert restored.assigned_niche == "witchcraft"

    def test_from_dict_handles_string_enums(self):
        data = {
            "device_id": "test",
            "device_type": "geelark",
            "status": "online",
        }
        dev = UnifiedDevice.from_dict(data)
        assert dev.device_type == DeviceType.GEELARK
        assert dev.status == DeviceStatus.ONLINE

    def test_post_init_converts_strings(self):
        dev = UnifiedDevice(device_type="emulator", status="busy")
        assert dev.device_type == DeviceType.EMULATOR
        assert dev.status == DeviceStatus.BUSY


# ===========================================================================
# POOL TASK DATACLASS
# ===========================================================================


class TestPoolTask:

    def test_defaults(self):
        task = PoolTask(description="Open Chrome")
        assert task.status == "pending"
        assert task.priority == TaskPriority.NORMAL
        assert task.max_retries == 3

    def test_is_terminal(self):
        task = PoolTask(status="completed")
        assert task.is_terminal is True

        task2 = PoolTask(status="pending")
        assert task2.is_terminal is False

    def test_can_retry(self):
        task = PoolTask(status="failed", retry_count=1, max_retries=3)
        assert task.can_retry is True

        task2 = PoolTask(status="failed", retry_count=3, max_retries=3)
        assert task2.can_retry is False

    def test_priority_order(self):
        critical = PoolTask(priority=TaskPriority.CRITICAL)
        normal = PoolTask(priority=TaskPriority.NORMAL)
        assert critical.priority_order < normal.priority_order

    def test_to_dict_roundtrip(self):
        task = PoolTask(
            task_id="task-abc",
            description="Post to Instagram",
            niche="witchcraft",
            priority=TaskPriority.HIGH,
            required_capabilities=["browser", "camera"],
        )
        d = task.to_dict()
        restored = PoolTask.from_dict(d)
        assert restored.task_id == "task-abc"
        assert restored.priority == TaskPriority.HIGH
        assert restored.niche == "witchcraft"

    def test_post_init_converts_priority(self):
        task = PoolTask(priority="critical")
        assert task.priority == TaskPriority.CRITICAL


# ===========================================================================
# NICHE ASSIGNMENT DATACLASS
# ===========================================================================


class TestNicheAssignment:

    def test_defaults(self):
        na = NicheAssignment(niche="witchcraft")
        assert na.daily_task_limit == 100
        assert na.tasks_today == 0

    def test_has_capacity(self):
        na = NicheAssignment(niche="witchcraft", daily_task_limit=10, tasks_today=5)
        assert na.has_capacity is True

    def test_no_capacity_at_limit(self):
        na = NicheAssignment(niche="witchcraft", daily_task_limit=10, tasks_today=10)
        assert na.has_capacity is False

    def test_to_dict_roundtrip(self):
        na = NicheAssignment(
            niche="smarthome",
            device_ids=["dev-1", "dev-2"],
            platforms=["instagram", "pinterest"],
        )
        d = na.to_dict()
        restored = NicheAssignment.from_dict(d)
        assert restored.niche == "smarthome"
        assert len(restored.device_ids) == 2


# ===========================================================================
# DEVICE POOL CLASS
# ===========================================================================


class TestDevicePool:

    @pytest.fixture
    def pool(self):
        return DevicePool()

    # ------- Initialization -------

    def test_init_creates_empty_registries(self, pool):
        assert isinstance(pool._devices, dict)
        assert isinstance(pool._tasks, dict)
        assert isinstance(pool._niche_assignments, dict)

    # ------- Device Registration -------

    def test_register_device(self, pool):
        dev = UnifiedDevice(
            device_id="test-dev-001",
            name="Test Phone",
            device_type=DeviceType.PHYSICAL,
            status=DeviceStatus.ONLINE,
        )
        pool.register_device(dev)
        assert "test-dev-001" in pool._devices

    def test_remove_device(self, pool):
        dev = UnifiedDevice(device_id="test-dev-002", name="Removable")
        pool.register_device(dev)
        removed = pool.remove_device("test-dev-002")
        assert removed is True
        assert "test-dev-002" not in pool._devices

    def test_remove_nonexistent_device(self, pool):
        removed = pool.remove_device("ghost-device")
        assert removed is False

    # ------- Device Listing -------

    def test_list_devices_empty(self, pool):
        assert len(pool._devices) == 0

    def test_list_devices_after_registration(self, pool):
        for i in range(3):
            pool.register_device(UnifiedDevice(
                device_id=f"dev-{i}",
                name=f"Device {i}",
                device_type=DeviceType.PHYSICAL,
            ))
        assert len(pool._devices) == 3

    # ------- Task Queue -------

    def test_enqueue_task(self, pool):
        task = PoolTask(task_id="task-001", description="Test task")
        pool._enqueue_task(task)
        assert "task-001" in pool._tasks
        assert "task-001" in pool._task_queue

    def test_dequeue_task(self, pool):
        task = PoolTask(task_id="task-dq", description="Dequeue me")
        pool._enqueue_task(task)
        dequeued = pool._dequeue_task()
        assert dequeued is not None
        assert dequeued.task_id == "task-dq"

    def test_dequeue_empty_returns_none(self, pool):
        assert pool._dequeue_task() is None

    def test_priority_ordering_in_queue(self, pool):
        pool._enqueue_task(PoolTask(task_id="low", priority=TaskPriority.LOW))
        pool._enqueue_task(PoolTask(task_id="critical", priority=TaskPriority.CRITICAL))
        pool._enqueue_task(PoolTask(task_id="normal", priority=TaskPriority.NORMAL))

        first = pool._dequeue_task()
        assert first.task_id == "critical"

    # ------- Backend Helpers -------

    def test_find_by_backend(self, pool):
        dev = UnifiedDevice(
            device_id="dev-fb-test",
            device_type=DeviceType.GEELARK,
            backend_id="gl-profile-123",
        )
        pool.register_device(dev)
        found = pool._find_by_backend("geelark", "gl-profile-123")
        assert found == "dev-fb-test"

    def test_find_by_backend_not_found(self, pool):
        found = pool._find_by_backend("geelark", "nonexistent")
        assert found is None

    def test_map_farm_status(self):
        assert DevicePool._map_farm_status("online") == DeviceStatus.ONLINE
        assert DevicePool._map_farm_status("offline") == DeviceStatus.OFFLINE
        assert DevicePool._map_farm_status("busy") == DeviceStatus.BUSY
        assert DevicePool._map_farm_status("unknown") == DeviceStatus.OFFLINE

    def test_map_geelark_status(self):
        assert DevicePool._map_geelark_status("running") == DeviceStatus.ONLINE
        assert DevicePool._map_geelark_status("stopped") == DeviceStatus.OFFLINE
        assert DevicePool._map_geelark_status("suspended") == DeviceStatus.MAINTENANCE


# ===========================================================================
# DEVICE DISCOVERY (mocked backends)
# ===========================================================================


class TestDeviceDiscovery:

    @pytest.fixture
    def pool(self):
        return DevicePool()

    @pytest.mark.asyncio
    async def test_discover_all_with_no_backends(self, pool):
        """When no backend modules are available, discovery should still succeed."""
        with patch("src.device_pool.DevicePool.discover_physical", new_callable=AsyncMock, return_value=[]):
            with patch("src.device_pool.DevicePool.discover_geelark", new_callable=AsyncMock, return_value=[]):
                with patch("src.device_pool.DevicePool.discover_openclaw", new_callable=AsyncMock, return_value=[]):
                    counts = await pool.discover_all()

        assert counts["total"] == 0
        assert "physical" in counts
        assert "geelark" in counts

    @pytest.mark.asyncio
    async def test_discover_all_with_devices(self, pool):
        mock_devices = [
            UnifiedDevice(device_id="phy-1", device_type=DeviceType.PHYSICAL, name="Pixel"),
            UnifiedDevice(device_id="phy-2", device_type=DeviceType.EMULATOR, name="Emu"),
        ]

        with patch.object(pool, "discover_physical", new_callable=AsyncMock, return_value=mock_devices):
            with patch.object(pool, "discover_geelark", new_callable=AsyncMock, return_value=[]):
                with patch.object(pool, "discover_openclaw", new_callable=AsyncMock, return_value=[]):
                    counts = await pool.discover_all()

        assert counts["physical"] == 2
        assert counts["total"] == 2

    @pytest.mark.asyncio
    async def test_discover_handles_backend_failure(self, pool):
        """Discovery should not fail if one backend raises an exception."""
        with patch.object(pool, "discover_physical", new_callable=AsyncMock, side_effect=Exception("ADB failed")):
            with patch.object(pool, "discover_geelark", new_callable=AsyncMock, return_value=[]):
                with patch.object(pool, "discover_openclaw", new_callable=AsyncMock, return_value=[]):
                    counts = await pool.discover_all()

        assert counts["physical"] == 0
        assert "total" in counts


# ===========================================================================
# NICHE AFFINITY
# ===========================================================================


class TestNicheAffinity:

    @pytest.fixture
    def pool(self):
        return DevicePool()

    def test_assign_niche(self, pool):
        dev = UnifiedDevice(device_id="dev-niche", assigned_niche="witchcraft")
        pool.register_device(dev)
        assert pool._devices["dev-niche"].assigned_niche == "witchcraft"

    def test_niche_assignment_capacity(self):
        na = NicheAssignment(niche="witchcraft", daily_task_limit=50, tasks_today=49)
        assert na.has_capacity is True

        na.tasks_today = 50
        assert na.has_capacity is False


# ===========================================================================
# HEALTH MONITORING
# ===========================================================================


class TestHealthMonitoring:

    def test_health_thresholds(self):
        assert HEALTH_EXCELLENT > HEALTH_GOOD > HEALTH_DEGRADED > HEALTH_CRITICAL

    def test_device_health_affects_availability(self):
        dev = UnifiedDevice(
            status=DeviceStatus.ONLINE,
            health_score=HEALTH_CRITICAL - 1,
        )
        assert dev.is_available is False

        dev2 = UnifiedDevice(
            status=DeviceStatus.ONLINE,
            health_score=HEALTH_EXCELLENT,
        )
        assert dev2.is_available is True


# ===========================================================================
# EMPIRE NICHES
# ===========================================================================


class TestEmpireNiches:

    def test_sixteen_niches(self):
        assert len(EMPIRE_NICHES) == 16

    def test_known_niches(self):
        for n in ("witchcraft", "smarthome", "aiaction", "family"):
            assert n in EMPIRE_NICHES


# ===========================================================================
# COSTS
# ===========================================================================


class TestCosts:

    def test_physical_device_free(self):
        assert COST_PHYSICAL == 0.0

    def test_geelark_has_cost(self):
        assert COST_GEELARK > 0.0
