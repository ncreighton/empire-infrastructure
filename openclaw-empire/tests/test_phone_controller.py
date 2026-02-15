"""Test phone_controller — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.phone_controller import (
        PhoneController,
        DeviceAction,
        ActionResult,
        UIElement,
        VisionAnalysis,
        TaskStep,
        TaskPlan,
        ActionType,
        DEFAULT_COMMAND_TIMEOUT,
        POST_ACTION_DELAY,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="phone_controller not available")


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def mock_session():
    """Mock aiohttp session."""
    session = MagicMock()
    session.closed = False
    return session


@pytest.fixture
def controller():
    """PhoneController with default settings."""
    return PhoneController(
        node_url="http://test-node:18789",
        node_name="test-android",
    )


# ===================================================================
# Enum / Data Class Tests
# ===================================================================


class TestActionType:
    def test_all_action_types(self):
        expected = {
            "tap", "long_press", "type_text", "swipe", "scroll_up",
            "scroll_down", "key_event", "launch_app", "screenshot",
            "ui_dump", "wait", "back", "home", "recents",
        }
        actual = {at.value for at in ActionType}
        assert expected == actual


class TestDeviceAction:
    def test_defaults(self):
        da = DeviceAction(action_type=ActionType.TAP, params={"x": 100, "y": 200})
        assert da.timeout == DEFAULT_COMMAND_TIMEOUT
        assert da.verify_after is True
        assert da.description == ""

    def test_repr(self):
        da = DeviceAction(
            action_type=ActionType.TAP,
            params={"x": 540, "y": 960},
            description="Tap center",
        )
        r = repr(da)
        assert "tap" in r
        assert "540" in r


class TestActionResult:
    def test_success_result(self):
        action = DeviceAction(action_type=ActionType.TAP)
        result = ActionResult(success=True, action=action, output="done")
        assert result.success is True
        assert result.error is None
        assert result.output == "done"

    def test_failure_result(self):
        action = DeviceAction(action_type=ActionType.LAUNCH_APP)
        result = ActionResult(success=False, action=action, error="App not found")
        assert result.success is False
        assert "App not found" in result.error


class TestUIElement:
    def test_center(self):
        el = UIElement(text="OK", bounds=(100, 200, 300, 400))
        assert el.center == (200, 300)

    def test_dimensions(self):
        el = UIElement(bounds=(0, 0, 100, 200))
        assert el.width == 100
        assert el.height == 200

    def test_defaults(self):
        el = UIElement()
        assert el.text == ""
        assert el.clickable is False
        assert el.enabled is True
        assert el.focused is False


class TestVisionAnalysis:
    def test_defaults(self):
        va = VisionAnalysis()
        assert va.description == ""
        assert va.confidence == 0.0
        assert va.elements_detected == []

    def test_with_data(self):
        va = VisionAnalysis(
            description="Login screen",
            current_app="com.instagram.android",
            confidence=0.92,
        )
        assert va.confidence == 0.92
        assert "instagram" in va.current_app


class TestTaskStep:
    def test_defaults(self):
        action = DeviceAction(action_type=ActionType.TAP)
        ts = TaskStep(step_number=1, description="Tap login", action=action)
        assert ts.completed is False
        assert ts.max_retries == 2
        assert ts.result is None


class TestTaskPlan:
    def test_defaults(self):
        tp = TaskPlan(task_description="Login to Instagram")
        assert tp.task_id != ""
        assert tp.steps == []
        assert tp.status == "pending"


# ===================================================================
# PhoneController Tests
# ===================================================================


class TestPhoneControllerInit:
    def test_default_init(self, controller):
        assert "test-node" in controller.node_url
        assert controller.node_name == "test-android"
        assert controller.command_timeout == DEFAULT_COMMAND_TIMEOUT
        assert controller.is_connected is False

    def test_url_trailing_slash_stripped(self):
        pc = PhoneController(node_url="http://test:18789/")
        assert not pc.node_url.endswith("/")

    def test_default_resolution(self, controller):
        assert controller.resolution == (1080, 2400)


class TestPhoneControllerSession:
    @pytest.mark.asyncio
    async def test_ensure_session_creates_session(self, controller):
        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value = MagicMock(closed=False)
            session = await controller._ensure_session()
            assert session is not None

    @pytest.mark.asyncio
    async def test_close_session(self, controller):
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        controller._session = mock_session

        await controller.close()
        mock_session.close.assert_called_once()
        assert controller._session is None


class TestPhoneControllerConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self, controller):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"connected": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(),
        ))
        mock_session.closed = False

        controller._session = mock_session

        with patch.object(controller, "_invoke_node", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = [
                {"connected": True},    # device.status
                {"stdout": "Physical size: 1080x2400"},  # wm size
            ]
            result = await controller.connect()
            assert result is True
            assert controller.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, controller):
        with patch.object(controller, "_invoke_node", side_effect=ConnectionError("timeout")):
            result = await controller.connect()
            assert result is False
            assert controller.is_connected is False


class TestPhoneControllerActions:
    @pytest.mark.asyncio
    async def test_tap(self, controller):
        with patch.object(controller, "_adb_shell", new_callable=AsyncMock, return_value=""):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await controller.tap(540, 960)
                assert result.success is True
                controller._adb_shell.assert_called_once_with("input tap 540 960")

    @pytest.mark.asyncio
    async def test_tap_failure(self, controller):
        with patch.object(controller, "_adb_shell", side_effect=Exception("ADB error")):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await controller.tap(100, 200)
                assert result.success is False
                assert "ADB error" in result.error

    @pytest.mark.asyncio
    async def test_long_press(self, controller):
        with patch.object(controller, "_adb_shell", new_callable=AsyncMock, return_value=""):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await controller.long_press(300, 400, duration_ms=2000)
                assert result.success is True

    @pytest.mark.asyncio
    async def test_type_text(self, controller):
        with patch.object(controller, "_adb_shell", new_callable=AsyncMock, return_value=""):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await controller.type_text("Hello World")
                assert result.success is True

    @pytest.mark.asyncio
    async def test_type_text_escapes_spaces(self, controller):
        with patch.object(controller, "_adb_shell", new_callable=AsyncMock, return_value="") as mock_adb:
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await controller.type_text("a b")
                call_args = mock_adb.call_args[0][0]
                assert "%s" in call_args

    @pytest.mark.asyncio
    async def test_swipe(self, controller):
        with patch.object(controller, "_adb_shell", new_callable=AsyncMock, return_value=""):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await controller.swipe(100, 800, 100, 200, duration_ms=500)
                assert result.success is True


class TestPhoneControllerNodeCommunication:
    @pytest.mark.asyncio
    async def test_invoke_node_sends_payload(self, controller):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"stdout": "ok"})
        mock_resp.text = AsyncMock(return_value="ok")

        context_manager = AsyncMock()
        context_manager.__aenter__ = AsyncMock(return_value=mock_resp)
        context_manager.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=context_manager)
        mock_session.closed = False

        controller._session = mock_session

        result = await controller._invoke_node("device.status")
        assert result == {"stdout": "ok"}

    @pytest.mark.asyncio
    async def test_adb_shell_returns_stdout(self, controller):
        with patch.object(controller, "_invoke_node", new_callable=AsyncMock, return_value={"stdout": "test_output"}):
            result = await controller._adb_shell("echo test")
            assert result == "test_output"
