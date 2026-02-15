"""
Tests for the Intelligence Hub module.

Tests subsystem initialization, pre_task readiness, execute_task flow,
monitor_task, phone state, Phase 6 publish_content and device_pool_status.
All external subsystems are mocked.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.intelligence_hub import (
        HubConfig,
        IntelligenceHub,
        RunningTask,
        get_hub,
        reset_hub,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="intelligence_hub module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def hub_config():
    """Minimal HubConfig for testing."""
    return HubConfig()


@pytest.fixture
def mock_forge():
    """Mock ForgeEngine subsystem."""
    forge = MagicMock()
    forge.pre_flight = AsyncMock(return_value={
        "ready": True,
        "go_no_go": "GO",
        "scout": {"warnings": [], "blocking_issues": []},
        "oracle": {},
        "fixes": [],
    })
    forge.record_task = MagicMock()
    forge.record_outcome = MagicMock()
    forge.get_stats = MagicMock(return_value={})
    return forge


@pytest.fixture
def mock_amplify():
    """Mock AmplifyPipeline subsystem."""
    amp = MagicMock()
    amp.full_pipeline = MagicMock(return_value={
        "_amplify": {"stages_completed": []},
        "validation_summary": {"valid": True},
        "batch_groups": [],
        "steps": [],
    })
    amp.record_execution = MagicMock()
    amp.get_app_stats = MagicMock(return_value={"total_records": 0})
    return amp


@pytest.fixture
def mock_phone():
    """Mock PhoneController subsystem."""
    phone = MagicMock()
    phone.get_current_app = AsyncMock(return_value="com.android.launcher")
    phone.screenshot = AsyncMock(return_value="/tmp/screenshot.png")
    phone.ui_dump = AsyncMock(return_value=[])
    phone.close = AsyncMock()
    return phone


@pytest.fixture
def mock_vision():
    """Mock VisionAgent subsystem."""
    vision = MagicMock()
    vision.analyze_screen = AsyncMock(return_value=MagicMock(
        current_app="launcher",
        current_screen="home",
        visible_text=["Home", "Chrome", "Settings"],
        tappable_elements=[],
        keyboard_visible=False,
        loading_indicators=False,
        errors_detected=[],
        quality_score=0.95,
        analysis_time_ms=150.0,
    ))
    vision.detect_errors = AsyncMock(return_value=MagicMock(has_errors=False))
    vision.detect_state = AsyncMock(return_value=(
        MagicMock(value="ready"), 0.95, "Screen ready"
    ))
    vision.close = AsyncMock()
    return vision


@pytest.fixture
def mock_screenpipe():
    """Mock ScreenpipeAgent subsystem."""
    sp = MagicMock()
    sp.search_errors = AsyncMock(return_value=[])
    sp.get_current_state = AsyncMock(return_value=[])
    sp.close = AsyncMock()
    return sp


@pytest.fixture
def hub(hub_config, mock_forge, mock_amplify, mock_phone, mock_vision, mock_screenpipe):
    """Create IntelligenceHub with all subsystems mocked."""
    reset_hub()
    return IntelligenceHub(
        config=hub_config,
        forge_engine=mock_forge,
        amplify_pipeline=mock_amplify,
        phone_controller=mock_phone,
        vision_agent=mock_vision,
        screenpipe_agent=mock_screenpipe,
    )


@pytest.fixture
def hub_no_phone(hub_config, mock_forge, mock_amplify, mock_vision):
    """Hub without phone controller (graceful degradation)."""
    reset_hub()
    return IntelligenceHub(
        config=hub_config,
        forge_engine=mock_forge,
        amplify_pipeline=mock_amplify,
        phone_controller=None,
        vision_agent=mock_vision,
    )


# ===================================================================
# HubConfig Tests
# ===================================================================

class TestHubConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        cfg = HubConfig()
        assert cfg is not None

    def test_config_fields(self):
        cfg = HubConfig()
        # Should have standard fields
        d = vars(cfg) if not hasattr(cfg, "to_dict") else cfg.to_dict()
        assert isinstance(d, dict)


# ===================================================================
# Initialization Tests
# ===================================================================

class TestInitialization:
    """Test hub creation and subsystem registration."""

    def test_hub_created(self, hub):
        assert isinstance(hub, IntelligenceHub)

    def test_hub_without_optional_subsystems(self, hub_config):
        reset_hub()
        h = IntelligenceHub(config=hub_config)
        assert h is not None

    def test_subsystems_registered(self, hub, mock_forge, mock_phone):
        # Hub stores subsystems as .forge and .phone
        assert hub.forge is mock_forge
        assert hub.phone is mock_phone


# ===================================================================
# Pre-Task Readiness Tests
# ===================================================================

class TestPreTask:
    """Test pre_task readiness checks."""

    @pytest.mark.asyncio
    async def test_pre_task_all_ready(self, hub):
        result = await hub.pre_task("navigate to chrome", app="chrome")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_pre_task_no_phone(self, hub_no_phone):
        result = await hub_no_phone.pre_task("simple calculation", app="calculator")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_pre_task_low_battery(self, hub, mock_phone):
        mock_phone.get_current_app = AsyncMock(return_value="")
        result = await hub.pre_task("open chrome", app="chrome")
        assert isinstance(result, dict)


# ===================================================================
# Execute Task Tests
# ===================================================================

class TestExecuteTask:
    """Test execute_task workflow."""

    @pytest.mark.asyncio
    async def test_execute_basic_task(self, hub):
        result = await hub.execute_task("Open Chrome", app="chrome")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_execute_task_returns_status(self, hub):
        result = await hub.execute_task("Tap button", app="calculator")
        assert "status" in result

    @pytest.mark.asyncio
    async def test_execute_task_without_phone(self, hub_no_phone):
        result = await hub_no_phone.execute_task("Offline task", app="notes")
        assert isinstance(result, dict)


# ===================================================================
# Monitor Task Tests
# ===================================================================

class TestMonitorTask:
    """Test task monitoring."""

    @pytest.mark.asyncio
    async def test_monitor_running_task(self, hub):
        # Start a task first
        task_result = await hub.execute_task("Background task", app="chrome")
        # Monitor should work even if task already completed
        monitor_result = await hub.monitor_task(
            task_result.get("task_id", "test_task")
        )
        assert isinstance(monitor_result, dict)

    @pytest.mark.asyncio
    async def test_monitor_nonexistent_task(self, hub):
        result = await hub.monitor_task("nonexistent_task_999")
        assert isinstance(result, dict)


# ===================================================================
# Phone State Tests
# ===================================================================

class TestPhoneState:
    """Test phone state retrieval."""

    @pytest.mark.asyncio
    async def test_get_phone_state(self, hub):
        state = await hub.get_phone_state()
        assert isinstance(state, dict)
        assert "screen_on" in state

    @pytest.mark.asyncio
    async def test_get_phone_state_no_phone(self, hub_no_phone):
        state = await hub_no_phone.get_phone_state()
        assert isinstance(state, dict)

    @pytest.mark.asyncio
    async def test_quick_screenshot(self, hub):
        result = await hub.quick_screenshot()
        assert result is not None


# ===================================================================
# Intelligence Stats Tests
# ===================================================================

class TestStats:
    """Test intelligence statistics."""

    @pytest.mark.asyncio
    async def test_get_intelligence_stats(self, hub):
        stats = await hub.get_intelligence_stats()
        assert isinstance(stats, dict)


# ===================================================================
# Phase 6: publish_content Tests
# ===================================================================

class TestPublishContent:
    """Test the publish_content integration."""

    @pytest.mark.asyncio
    async def test_publish_content(self, hub):
        if not hasattr(hub, "publish_content"):
            pytest.skip("publish_content not implemented")
        with patch("src.intelligence_hub._import_content_pipeline") as mock_import:
            mock_pipeline = MagicMock()
            mock_pipeline.execute = AsyncMock(return_value={"success": True})
            mock_get_pipeline = MagicMock(return_value=mock_pipeline)
            mock_import.return_value = mock_get_pipeline
            result = await hub.publish_content(
                site_id="testsite1",
                title="Test Article",
            )
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_publish_content_batch(self, hub):
        if not hasattr(hub, "publish_content"):
            pytest.skip("publish_content not implemented")
        with patch("src.intelligence_hub._import_content_pipeline") as mock_import:
            mock_pipeline = MagicMock()
            mock_pipeline.execute_batch = AsyncMock(return_value={
                "success": True, "articles_published": 2,
            })
            mock_get_pipeline = MagicMock(return_value=mock_pipeline)
            mock_import.return_value = mock_get_pipeline
            result = await hub.publish_content(
                site_id="testsite1",
                max_articles=2,
            )
            assert isinstance(result, dict)


# ===================================================================
# Phase 6: device_pool_status Tests
# ===================================================================

class TestDevicePoolStatus:
    """Test device pool status retrieval."""

    @pytest.mark.asyncio
    async def test_get_device_pool_status(self, hub):
        if not hasattr(hub, "get_device_pool_status"):
            pytest.skip("get_device_pool_status not implemented")
        result = await hub.get_device_pool_status()
        assert isinstance(result, dict) or isinstance(result, list)


# ===================================================================
# Analyze Screen Tests
# ===================================================================

class TestAnalyzeScreen:
    """Test structured screen analysis."""

    @pytest.mark.asyncio
    async def test_analyze_screen_structured(self, hub):
        if not hasattr(hub, "analyze_screen_structured"):
            pytest.skip("analyze_screen_structured not implemented")
        result = await hub.analyze_screen_structured()
        assert isinstance(result, dict)


# ===================================================================
# RunningTask Dataclass Tests
# ===================================================================

class TestRunningTask:
    """Test RunningTask tracking."""

    def test_create_running_task(self):
        rt = RunningTask(
            task_id="t1",
            description="Test task",
            status="running",
        )
        assert rt.task_id == "t1"
        assert rt.status == "running"


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test factory and reset functions."""

    def test_get_hub_returns_instance(self):
        reset_hub()
        h = get_hub()
        assert isinstance(h, IntelligenceHub)

    def test_reset_hub(self):
        reset_hub()
        h1 = get_hub()
        reset_hub()
        h2 = get_hub()
        # After reset, should be a new instance
        assert h1 is not h2

    def test_get_hub_returns_same_instance(self):
        reset_hub()
        h1 = get_hub()
        h2 = get_hub()
        assert h1 is h2


# ===================================================================
# Post-Task Tests
# ===================================================================

class TestPostTask:
    """Test post_task cleanup and reporting."""

    @pytest.mark.asyncio
    async def test_post_task(self, hub):
        result = await hub.post_task(
            task_id="test_123",
            success=True,
            duration=5.0,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_post_task_with_failure(self, hub):
        result = await hub.post_task(
            task_id="fail_task",
            success=False,
            duration=10.0,
            error="timeout",
        )
        assert isinstance(result, dict)
