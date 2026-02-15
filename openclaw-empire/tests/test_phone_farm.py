"""
Tests for the Phone Farm module.

Tests device registration, load balancing, health monitoring, task
distribution, and device group management. All ADB/network calls mocked.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.phone_farm import (
        BalancingStrategy,
        DeviceCapabilities,
        DeviceGroup,
        DeviceInfo,
        DeviceStatus,
        DeviceType,
        LoadBalancer,
        PhoneFarm,
        TaskAssignment,
        TaskStatus,
        get_farm,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="phone_farm module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def farm_dir(tmp_path):
    """Isolated data directory for phone farm state."""
    d = tmp_path / "phone_farm"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sample_device():
    """Create a sample DeviceInfo."""
    return DeviceInfo(
        device_id="dev_001",
        name="Pixel 6 Pro",
        type=DeviceType.PHYSICAL,
        connection_string="R5CT123ABCD",
        status=DeviceStatus.ONLINE,
        capabilities=DeviceCapabilities(
            screen_width=1080,
            screen_height=2400,
            android_version="14",
            ram_mb=8192,
        ),
        health_score=100,
    )


@pytest.fixture
def sample_devices():
    """Multiple devices for balancing tests."""
    devices = []
    for i in range(4):
        d = DeviceInfo(
            device_id=f"dev_{i:03d}",
            name=f"Device {i}",
            type=DeviceType.PHYSICAL if i < 2 else DeviceType.CLOUD,
            connection_string=f"serial_{i}",
            status=DeviceStatus.ONLINE if i != 2 else DeviceStatus.BUSY,
            capabilities=DeviceCapabilities(
                screen_width=1080,
                screen_height=2400,
                android_version="14",
                ram_mb=8192,
            ),
            health_score=100,
        )
        devices.append(d)
    return devices


@pytest.fixture
def balancer():
    """LoadBalancer instance."""
    return LoadBalancer()


@pytest.fixture
def farm(farm_dir):
    """PhoneFarm with temp data dir."""
    with patch.object(PhoneFarm, "__init__", lambda self: None):
        f = PhoneFarm.__new__(PhoneFarm)
        f._devices = {}
        f._tasks = {}
        f._groups = {}
        f._history = []
        f._metrics = {}
        f._active_tasks = {}
        f._balancer = LoadBalancer()
        f._strategy = BalancingStrategy.BEST_FIT
        import asyncio
        f._semaphore = asyncio.Semaphore(5)
        f._watchdog_task = None
        f._watchdog_running = False
        from collections import defaultdict
        f._active_tasks = defaultdict(list)
        f._recovery_log = defaultdict(list)
        return f


# ===================================================================
# Enum Tests
# ===================================================================

class TestEnums:
    """Verify enum members."""

    def test_device_type(self):
        assert DeviceType.PHYSICAL is not None
        assert DeviceType.CLOUD is not None
        assert DeviceType.EMULATOR is not None

    def test_device_status(self):
        assert DeviceStatus.ONLINE is not None
        assert DeviceStatus.BUSY is not None
        assert DeviceStatus.OFFLINE is not None
        assert DeviceStatus.ERROR is not None

    def test_task_status(self):
        assert TaskStatus.QUEUED is not None
        assert TaskStatus.RUNNING is not None
        assert TaskStatus.COMPLETED is not None
        assert TaskStatus.FAILED is not None
        assert TaskStatus.ASSIGNED is not None
        assert TaskStatus.CANCELLED is not None

    def test_balancing_strategy(self):
        assert BalancingStrategy.ROUND_ROBIN is not None
        assert BalancingStrategy.LEAST_BUSY is not None
        assert BalancingStrategy.BEST_FIT is not None
        assert BalancingStrategy.AFFINITY is not None


# ===================================================================
# DeviceInfo Tests
# ===================================================================

class TestDeviceInfo:
    """Test DeviceInfo dataclass."""

    def test_create_device(self, sample_device):
        assert sample_device.device_id == "dev_001"
        assert sample_device.type == DeviceType.PHYSICAL

    def test_is_available(self, sample_device):
        assert sample_device.is_available is True

    def test_busy_device_not_available(self):
        d = DeviceInfo(
            device_id="dev_busy",
            name="Busy Phone",
            type=DeviceType.PHYSICAL,
            connection_string="serial_busy",
            status=DeviceStatus.BUSY,
            capabilities=DeviceCapabilities(
                screen_width=1080, screen_height=2400,
                android_version="14", ram_mb=4096,
            ),
            health_score=100,
        )
        assert d.is_available is False


# ===================================================================
# TaskAssignment Tests
# ===================================================================

class TestTaskAssignment:
    """Test task assignment tracking."""

    def test_create_assignment(self):
        ta = TaskAssignment(
            task_id="task_001",
            device_id="dev_001",
            task_description="Open Chrome and navigate",
            status=TaskStatus.QUEUED,
        )
        assert ta.task_id == "task_001"
        assert ta.status == TaskStatus.QUEUED

    def test_duration_ms(self):
        ta = TaskAssignment(
            task_id="task_002",
            device_id="dev_001",
            task_description="Quick tap",
            status=TaskStatus.COMPLETED,
            started_at="2026-01-15T10:00:00+00:00",
            completed_at="2026-01-15T10:00:05+00:00",
        )
        assert ta.duration_ms >= 4000  # ~5000ms


# ===================================================================
# DeviceGroup Tests
# ===================================================================

class TestDeviceGroup:
    """Test device grouping."""

    def test_create_group(self):
        group = DeviceGroup(
            group_id="grp_witchcraft",
            name="Witchcraft Devices",
            device_ids=["dev_001", "dev_002"],
        )
        assert group.group_id == "grp_witchcraft"
        assert len(group.device_ids) == 2


# ===================================================================
# LoadBalancer Tests
# ===================================================================

class TestLoadBalancer:
    """Test device selection strategies."""

    def test_round_robin_select(self, balancer, sample_devices):
        task = TaskAssignment(task_description="test task")
        selected = balancer.select(
            BalancingStrategy.ROUND_ROBIN, sample_devices, task, {}
        )
        assert selected is not None
        assert isinstance(selected, DeviceInfo)

    def test_round_robin_distributes(self, balancer, sample_devices):
        selections = set()
        task = TaskAssignment(task_description="test task")
        for _ in range(10):
            s = balancer.select(
                BalancingStrategy.ROUND_ROBIN, sample_devices, task, {}
            )
            if s:
                selections.add(s.device_id)
        # Should select from multiple available devices
        assert len(selections) >= 2

    def test_least_busy_strategy(self, sample_devices):
        lb = LoadBalancer()
        task = TaskAssignment(task_description="test task")
        selected = lb.select(
            BalancingStrategy.LEAST_BUSY, sample_devices, task, {}
        )
        assert selected is not None

    def test_best_fit_strategy(self, sample_devices):
        lb = LoadBalancer()
        task = TaskAssignment(task_description="test task", app="chrome")
        selected = lb.select(
            BalancingStrategy.BEST_FIT, sample_devices, task, {}
        )
        assert selected is not None

    def test_affinity_strategy(self, sample_devices):
        lb = LoadBalancer()
        lb.record_affinity("chrome", "dev_000")
        task = TaskAssignment(task_description="test task", app="chrome")
        selected = lb.select(
            BalancingStrategy.AFFINITY, sample_devices, task, {}
        )
        assert selected is not None

    def test_select_skips_busy(self, balancer, sample_devices):
        # dev_002 is BUSY in our fixture
        selections = set()
        task = TaskAssignment(task_description="test task")
        for _ in range(20):
            s = balancer.select(
                BalancingStrategy.ROUND_ROBIN, sample_devices, task, {}
            )
            if s:
                selections.add(s.device_id)
        assert "dev_002" not in selections


# ===================================================================
# PhoneFarm Core Tests
# ===================================================================

class TestPhoneFarm:
    """Test main PhoneFarm operations."""

    @pytest.mark.asyncio
    async def test_discover_devices(self, farm):
        if not hasattr(farm, "discover_devices"):
            pytest.skip("discover_devices not implemented")
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(
                b"List of devices attached\ndev_001\tdevice\ndev_002\tdevice\n",
                b"",
            ))
            mock_exec.return_value = mock_proc
            with patch.object(farm, "_query_device_capabilities", new_callable=AsyncMock, return_value=None):
                with patch.object(farm, "_save_devices"):
                    devices = await farm.discover_devices()
                    assert isinstance(devices, list)

    @pytest.mark.asyncio
    async def test_submit_task(self, farm, sample_device):
        farm._devices[sample_device.device_id] = sample_device
        with patch.object(farm, "_save_tasks"):
            task_id = await farm.submit_task(
                description="Open Chrome",
                app="chrome",
            )
            assert task_id is not None

    @pytest.mark.asyncio
    async def test_execute_queue(self, farm, sample_device):
        if not hasattr(farm, "execute_queue"):
            pytest.skip("execute_queue not implemented")
        farm._devices[sample_device.device_id] = sample_device
        task = TaskAssignment(
            task_id="q_001",
            device_id=sample_device.device_id,
            task_description="Test task",
            status=TaskStatus.QUEUED,
        )
        farm._tasks["q_001"] = task
        with patch.object(farm, "_execute_task_with_semaphore", new_callable=AsyncMock,
                         return_value={"status": "completed"}):
            with patch.object(farm, "assign_task", return_value=sample_device.device_id):
                with patch.object(farm, "_get_queued_tasks", return_value=[task]):
                    results = await farm.execute_queue()
                    assert isinstance(results, list)


# ===================================================================
# Health Monitoring Tests
# ===================================================================

class TestHealthMonitoring:
    """Test device health tracking."""

    def test_device_success_rate(self):
        d = DeviceInfo(
            device_id="dev_rate",
            name="Rate Test",
            type=DeviceType.PHYSICAL,
            connection_string="serial_rate",
            status=DeviceStatus.ONLINE,
            capabilities=DeviceCapabilities(
                screen_width=1080, screen_height=2400,
                android_version="14", ram_mb=4096,
            ),
            tasks_completed=9,
            tasks_failed=1,
            health_score=100,
        )
        rate = d.success_rate
        assert 0.85 <= rate <= 0.95


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test factory function."""

    def test_get_farm_returns_instance(self):
        with patch.object(PhoneFarm, "__init__", lambda self: None):
            f = get_farm()
            assert isinstance(f, PhoneFarm)
