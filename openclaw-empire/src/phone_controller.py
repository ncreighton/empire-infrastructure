"""
Phone Controller — OpenClaw Empire Android Automation

Remote Android phone control via ADB commands executed through the OpenClaw
Android node (Termux + Shizuku). Provides vision-guided automation that
screenshots the device, analyzes the screen with an AI vision model, decides
on the next action, executes it, and verifies the result.

Architecture:
    Claude/OpenClaw  -->  HTTP/WS  -->  Android Node (Termux)
                                          |
                                          v
                                     ADB shell commands
                                     (input, am, screencap, uiautomator)

Four core classes:
    PhoneController  — Low-level ADB command execution
    VisionLoop       — Screenshot -> analyze -> act -> verify cycle
    TaskExecutor     — Breaks high-level tasks into executable step sequences
    AppNavigator     — App-specific navigation patterns from app-registry.json

Integration points:
    FORGE  — Pre-task analysis (Scout audits, Oracle predictions, Codex learning)
    AMPLIFY — Task enhancement (Enrich, Expand, Fortify, Anticipate, Optimize, Validate)

Usage:
    from src.phone_controller import PhoneController, TaskExecutor

    controller = PhoneController(node_url="http://localhost:18789")
    executor = TaskExecutor(controller)

    # Low-level
    await controller.tap(540, 960)
    await controller.launch_app("com.facebook.katana")
    screenshot = await controller.screenshot()

    # High-level
    result = await executor.execute("open Facebook and create a post saying Hello World")
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("phone_controller")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DEFAULT_NODE_URL = os.getenv("OPENCLAW_NODE_URL", "http://localhost:18789")
DEFAULT_NODE_NAME = os.getenv("OPENCLAW_ANDROID_NODE", "android")
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://localhost:8002")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SCREENSHOT_DIR = Path(__file__).parent.parent / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

UI_DUMP_DIR = Path(__file__).parent.parent / "data" / "ui_dumps"
UI_DUMP_DIR.mkdir(parents=True, exist_ok=True)

APP_REGISTRY_PATH = Path(__file__).parent.parent / "configs" / "app-registry.json"

# Timeouts (seconds)
DEFAULT_COMMAND_TIMEOUT = 30
DEFAULT_WAIT_TIMEOUT = 15
SCREENSHOT_SETTLE_DELAY = 0.5
POST_ACTION_DELAY = 0.8


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ActionType(Enum):
    """Atomic actions the controller can perform on the device."""
    TAP = "tap"
    LONG_PRESS = "long_press"
    TYPE_TEXT = "type_text"
    SWIPE = "swipe"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    KEY_EVENT = "key_event"
    LAUNCH_APP = "launch_app"
    SCREENSHOT = "screenshot"
    UI_DUMP = "ui_dump"
    WAIT = "wait"
    BACK = "back"
    HOME = "home"
    RECENTS = "recents"


@dataclass
class DeviceAction:
    """A single action to be performed on the Android device."""
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    timeout: float = DEFAULT_COMMAND_TIMEOUT
    verify_after: bool = True

    def __repr__(self) -> str:
        return f"DeviceAction({self.action_type.value}, {self.params}, desc={self.description!r})"


@dataclass
class ActionResult:
    """Result of executing a single device action."""
    success: bool
    action: DeviceAction
    output: Optional[str] = None
    screenshot_path: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class UIElement:
    """A UI element found on screen via uiautomator or vision analysis."""
    text: str = ""
    resource_id: str = ""
    class_name: str = ""
    content_desc: str = ""
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # left, top, right, bottom
    clickable: bool = False
    enabled: bool = True
    focused: bool = False

    @property
    def center(self) -> Tuple[int, int]:
        """Return the center coordinates of this element."""
        left, top, right, bottom = self.bounds
        return ((left + right) // 2, (top + bottom) // 2)

    @property
    def width(self) -> int:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> int:
        return self.bounds[3] - self.bounds[1]


@dataclass
class VisionAnalysis:
    """Result from the vision AI analyzing a screenshot."""
    description: str = ""
    current_app: str = ""
    current_screen: str = ""
    elements_detected: List[Dict[str, Any]] = field(default_factory=list)
    suggested_action: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    raw_response: str = ""


@dataclass
class TaskStep:
    """A single step within a multi-step task plan."""
    step_number: int
    description: str
    action: DeviceAction
    expected_result: str = ""
    fallback_actions: List[DeviceAction] = field(default_factory=list)
    max_retries: int = 2
    completed: bool = False
    result: Optional[ActionResult] = None


@dataclass
class TaskPlan:
    """A full execution plan for a high-level task."""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_description: str = ""
    steps: List[TaskStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "pending"  # pending, running, completed, failed
    forge_analysis: Optional[Dict[str, Any]] = None
    amplify_enhancements: Optional[Dict[str, Any]] = None


# ===================================================================
# PhoneController — Core ADB command execution
# ===================================================================

class PhoneController:
    """
    Low-level Android device controller that executes ADB commands via
    the OpenClaw Android node's HTTP API.

    All commands are sent as HTTP POST requests to the node, which runs
    them through Termux/Shizuku on the paired Android device. The node
    translates each command into the appropriate ``adb shell`` invocation.

    Supports: tap, long press, type text, swipe, scroll, key events,
    app launching, screenshots, UI hierarchy dumps, and element search.

    Example:
        controller = PhoneController(node_url="http://192.168.1.50:18789")
        await controller.connect()
        await controller.launch_app("com.twitter.android")
        await controller.tap(540, 300)
        await controller.type_text("Hello from OpenClaw!")
        screenshot = await controller.screenshot()
    """

    def __init__(
        self,
        node_url: str = DEFAULT_NODE_URL,
        node_name: str = DEFAULT_NODE_NAME,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
    ) -> None:
        self.node_url = node_url.rstrip("/")
        self.node_name = node_name
        self.command_timeout = command_timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected: bool = False
        self._device_resolution: Tuple[int, int] = (1080, 2400)  # default, updated on connect
        self._screenshot_counter: int = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create and return the HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.command_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ----- Node communication -----

    async def _invoke_node(self, command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a command to the OpenClaw Android node and return the response.

        The node API expects POST /api/nodes/invoke with:
            { "node": "<name>", "command": "<cmd>", "params": {...} }
        """
        session = await self._ensure_session()
        payload = {
            "node": self.node_name,
            "command": command,
            "params": params or {},
        }
        url = f"{self.node_url}/api/nodes/invoke"

        logger.debug("Node invoke: %s %s", command, params or {})

        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ConnectionError(
                        f"Node returned HTTP {resp.status}: {body[:500]}"
                    )
                data = await resp.json()
                if data.get("error"):
                    raise RuntimeError(f"Node error: {data['error']}")
                return data
        except aiohttp.ClientError as exc:
            raise ConnectionError(f"Failed to reach node at {url}: {exc}") from exc

    async def _adb_shell(self, cmd: str, timeout: Optional[float] = None) -> str:
        """
        Execute a raw ADB shell command on the device via the node.

        Returns the stdout output as a string.
        """
        result = await self._invoke_node("adb.shell", {"command": cmd, "timeout": timeout or self.command_timeout})
        return result.get("stdout", "")

    # ----- Connection & device info -----

    async def connect(self) -> bool:
        """
        Verify the Android node is reachable and the device is connected.
        Updates device resolution from actual screen dimensions.
        """
        try:
            result = await self._invoke_node("device.status")
            self._connected = result.get("connected", False)

            if self._connected:
                # Query actual screen resolution
                wm_output = await self._adb_shell("wm size")
                match = re.search(r"(\d+)x(\d+)", wm_output)
                if match:
                    self._device_resolution = (int(match.group(1)), int(match.group(2)))
                    logger.info(
                        "Connected to device. Resolution: %dx%d",
                        self._device_resolution[0],
                        self._device_resolution[1],
                    )
            else:
                logger.warning("Node reachable but no device connected")

            return self._connected
        except Exception as exc:
            logger.error("Connection failed: %s", exc)
            self._connected = False
            return False

    @property
    def resolution(self) -> Tuple[int, int]:
        """Device screen resolution (width, height)."""
        return self._device_resolution

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ----- Input actions -----

    async def tap(self, x: int, y: int) -> ActionResult:
        """Tap a point on the screen."""
        action = DeviceAction(ActionType.TAP, {"x": x, "y": y}, f"Tap ({x}, {y})")
        start = time.monotonic()
        try:
            await self._adb_shell(f"input tap {x} {y}")
            await asyncio.sleep(POST_ACTION_DELAY)
            return ActionResult(success=True, action=action, duration_ms=(time.monotonic() - start) * 1000)
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc), duration_ms=(time.monotonic() - start) * 1000)

    async def long_press(self, x: int, y: int, duration_ms: int = 1000) -> ActionResult:
        """Long press at a point on the screen."""
        action = DeviceAction(ActionType.LONG_PRESS, {"x": x, "y": y, "duration_ms": duration_ms})
        start = time.monotonic()
        try:
            await self._adb_shell(f"input swipe {x} {y} {x} {y} {duration_ms}")
            await asyncio.sleep(POST_ACTION_DELAY)
            return ActionResult(success=True, action=action, duration_ms=(time.monotonic() - start) * 1000)
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc), duration_ms=(time.monotonic() - start) * 1000)

    async def type_text(self, text: str) -> ActionResult:
        """
        Type text into the currently focused field.

        Special characters are escaped for ADB input. Spaces are sent as
        ``%s`` per ADB conventions.
        """
        action = DeviceAction(ActionType.TYPE_TEXT, {"text": text}, f"Type: {text[:40]}...")
        start = time.monotonic()
        try:
            # ADB input text requires escaping of special characters
            escaped = text.replace("\\", "\\\\")
            escaped = escaped.replace(" ", "%s")
            escaped = escaped.replace("'", "\\'")
            escaped = escaped.replace('"', '\\"')
            escaped = escaped.replace("&", "\\&")
            escaped = escaped.replace("<", "\\<")
            escaped = escaped.replace(">", "\\>")
            escaped = escaped.replace("|", "\\|")
            escaped = escaped.replace(";", "\\;")
            escaped = escaped.replace("(", "\\(")
            escaped = escaped.replace(")", "\\)")

            await self._adb_shell(f"input text '{escaped}'")
            await asyncio.sleep(POST_ACTION_DELAY)
            return ActionResult(success=True, action=action, duration_ms=(time.monotonic() - start) * 1000)
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc), duration_ms=(time.monotonic() - start) * 1000)

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> ActionResult:
        """Swipe from (x1, y1) to (x2, y2) over the given duration."""
        action = DeviceAction(
            ActionType.SWIPE,
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration_ms": duration_ms},
            f"Swipe ({x1},{y1})->({x2},{y2})",
        )
        start = time.monotonic()
        try:
            await self._adb_shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")
            await asyncio.sleep(POST_ACTION_DELAY)
            return ActionResult(success=True, action=action, duration_ms=(time.monotonic() - start) * 1000)
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc), duration_ms=(time.monotonic() - start) * 1000)

    async def scroll_down(self, distance: int = 500) -> ActionResult:
        """Scroll down by swiping upward from center screen."""
        cx = self._device_resolution[0] // 2
        cy = self._device_resolution[1] // 2
        return await self.swipe(cx, cy + distance // 2, cx, cy - distance // 2, 400)

    async def scroll_up(self, distance: int = 500) -> ActionResult:
        """Scroll up by swiping downward from center screen."""
        cx = self._device_resolution[0] // 2
        cy = self._device_resolution[1] // 2
        return await self.swipe(cx, cy - distance // 2, cx, cy + distance // 2, 400)

    async def press_key(self, keycode: Union[int, str]) -> ActionResult:
        """Send a keycode event (e.g., KEYCODE_BACK = 4, KEYCODE_HOME = 3)."""
        action = DeviceAction(ActionType.KEY_EVENT, {"keycode": keycode}, f"Key: {keycode}")
        start = time.monotonic()
        try:
            await self._adb_shell(f"input keyevent {keycode}")
            await asyncio.sleep(POST_ACTION_DELAY)
            return ActionResult(success=True, action=action, duration_ms=(time.monotonic() - start) * 1000)
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc), duration_ms=(time.monotonic() - start) * 1000)

    async def press_back(self) -> ActionResult:
        """Press the Back button."""
        return await self.press_key(4)

    async def press_home(self) -> ActionResult:
        """Press the Home button."""
        return await self.press_key(3)

    async def press_recents(self) -> ActionResult:
        """Press the Recent Apps button."""
        return await self.press_key(187)

    async def press_enter(self) -> ActionResult:
        """Press the Enter key."""
        return await self.press_key(66)

    # ----- App management -----

    async def launch_app(self, package_name: str, activity: Optional[str] = None) -> ActionResult:
        """
        Launch an app by its package name.

        If ``activity`` is provided, launches that specific activity directly.
        Otherwise uses ``monkey`` to launch the default activity.
        """
        action = DeviceAction(
            ActionType.LAUNCH_APP,
            {"package": package_name, "activity": activity},
            f"Launch {package_name}",
        )
        start = time.monotonic()
        try:
            if activity:
                await self._adb_shell(
                    f"am start -n {package_name}/{activity}"
                )
            else:
                await self._adb_shell(
                    f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
                )
            # Give app time to load
            await asyncio.sleep(2.0)
            return ActionResult(success=True, action=action, duration_ms=(time.monotonic() - start) * 1000)
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc), duration_ms=(time.monotonic() - start) * 1000)

    async def force_stop_app(self, package_name: str) -> ActionResult:
        """Force stop an app."""
        action = DeviceAction(ActionType.KEY_EVENT, {"package": package_name}, f"Force stop {package_name}")
        start = time.monotonic()
        try:
            await self._adb_shell(f"am force-stop {package_name}")
            return ActionResult(success=True, action=action, duration_ms=(time.monotonic() - start) * 1000)
        except Exception as exc:
            return ActionResult(success=False, action=action, error=str(exc), duration_ms=(time.monotonic() - start) * 1000)

    async def get_current_app(self) -> str:
        """Return the package name of the currently focused app."""
        output = await self._adb_shell(
            "dumpsys activity activities | grep mResumedActivity"
        )
        match = re.search(r"u0\s+(\S+)/", output)
        return match.group(1) if match else ""

    # ----- Screen capture & UI analysis -----

    async def screenshot(self, save_local: bool = True) -> str:
        """
        Capture a screenshot and return the local file path.

        The screenshot is taken on-device with ``screencap``, pulled as
        base64 PNG via the node, and saved locally for vision analysis.
        """
        self._screenshot_counter += 1
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"screen_{ts}_{self._screenshot_counter:04d}.png"
        local_path = str(SCREENSHOT_DIR / filename)

        await asyncio.sleep(SCREENSHOT_SETTLE_DELAY)

        result = await self._invoke_node("screen.capture", {"format": "png"})
        image_data = result.get("image_base64", "")

        if not image_data:
            # Fallback: capture via adb shell and read bytes
            device_path = "/sdcard/openclaw_screen.png"
            await self._adb_shell(f"screencap -p {device_path}")
            pull_result = await self._invoke_node("file.read", {"path": device_path, "encoding": "base64"})
            image_data = pull_result.get("data", "")

        if image_data and save_local:
            raw = base64.b64decode(image_data)
            with open(local_path, "wb") as f:
                f.write(raw)
            logger.info("Screenshot saved: %s (%d bytes)", local_path, len(raw))
            return local_path

        if image_data:
            return image_data  # Return raw base64 if not saving locally

        raise RuntimeError("Failed to capture screenshot: no image data returned")

    async def ui_dump(self) -> List[UIElement]:
        """
        Dump the current UI hierarchy via ``uiautomator dump`` and parse
        it into a list of UIElement objects.
        """
        device_path = "/sdcard/openclaw_ui.xml"
        await self._adb_shell(f"uiautomator dump {device_path}")
        result = await self._invoke_node("file.read", {"path": device_path})
        xml_content = result.get("data", "")

        if not xml_content:
            logger.warning("UI dump returned empty content")
            return []

        # Save locally for debugging
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        local_path = UI_DUMP_DIR / f"ui_{ts}.xml"
        local_path.write_text(xml_content, encoding="utf-8")

        return self._parse_ui_xml(xml_content)

    @staticmethod
    def _parse_ui_xml(xml_content: str) -> List[UIElement]:
        """Parse uiautomator XML dump into UIElement objects."""
        elements: List[UIElement] = []
        # Parse node elements with regex (avoids xml.etree dependency issues with
        # malformed uiautomator output)
        pattern = re.compile(
            r'<node\s+'
            r'.*?text="(?P<text>[^"]*)"'
            r'.*?resource-id="(?P<rid>[^"]*)"'
            r'.*?class="(?P<cls>[^"]*)"'
            r'.*?content-desc="(?P<desc>[^"]*)"'
            r'.*?clickable="(?P<click>[^"]*)"'
            r'.*?enabled="(?P<enabled>[^"]*)"'
            r'.*?focused="(?P<focused>[^"]*)"'
            r'.*?bounds="\[(?P<x1>\d+),(?P<y1>\d+)\]\[(?P<x2>\d+),(?P<y2>\d+)\]"',
            re.DOTALL,
        )
        for m in pattern.finditer(xml_content):
            elements.append(UIElement(
                text=m.group("text"),
                resource_id=m.group("rid"),
                class_name=m.group("cls"),
                content_desc=m.group("desc"),
                bounds=(
                    int(m.group("x1")), int(m.group("y1")),
                    int(m.group("x2")), int(m.group("y2")),
                ),
                clickable=m.group("click") == "true",
                enabled=m.group("enabled") == "true",
                focused=m.group("focused") == "true",
            ))
        logger.debug("UI dump parsed %d elements", len(elements))
        return elements

    # ----- Element finding -----

    async def find_element(
        self,
        text: Optional[str] = None,
        resource_id: Optional[str] = None,
        class_name: Optional[str] = None,
        content_desc: Optional[str] = None,
        partial_match: bool = True,
    ) -> Optional[UIElement]:
        """
        Find a single UI element matching the given criteria.

        Searches by text, resource-id, class name, or content description.
        Returns the first match or None.
        """
        elements = await self.ui_dump()
        return self._match_element(elements, text, resource_id, class_name, content_desc, partial_match)

    async def find_elements(
        self,
        text: Optional[str] = None,
        resource_id: Optional[str] = None,
        class_name: Optional[str] = None,
        content_desc: Optional[str] = None,
        partial_match: bool = True,
    ) -> List[UIElement]:
        """Find all UI elements matching the given criteria."""
        elements = await self.ui_dump()
        return [
            el for el in elements
            if self._element_matches(el, text, resource_id, class_name, content_desc, partial_match)
        ]

    @staticmethod
    def _element_matches(
        el: UIElement,
        text: Optional[str],
        resource_id: Optional[str],
        class_name: Optional[str],
        content_desc: Optional[str],
        partial: bool,
    ) -> bool:
        """Check whether a UIElement matches the given search criteria."""
        def _matches(value: str, query: str) -> bool:
            if partial:
                return query.lower() in value.lower()
            return value.lower() == query.lower()

        if text and not _matches(el.text, text):
            return False
        if resource_id and not _matches(el.resource_id, resource_id):
            return False
        if class_name and not _matches(el.class_name, class_name):
            return False
        if content_desc and not _matches(el.content_desc, content_desc):
            return False
        return True

    @staticmethod
    def _match_element(
        elements: List[UIElement],
        text: Optional[str],
        resource_id: Optional[str],
        class_name: Optional[str],
        content_desc: Optional[str],
        partial: bool,
    ) -> Optional[UIElement]:
        """Return the first element matching the criteria, or None."""
        for el in elements:
            if PhoneController._element_matches(el, text, resource_id, class_name, content_desc, partial):
                return el
        return None

    async def wait_for_element(
        self,
        text: Optional[str] = None,
        resource_id: Optional[str] = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = 1.0,
    ) -> Optional[UIElement]:
        """
        Poll the UI hierarchy until an element matching the criteria appears,
        or until the timeout is reached. Returns the element or None.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            el = await self.find_element(text=text, resource_id=resource_id)
            if el is not None:
                logger.debug("Found element: text=%r, id=%r", el.text, el.resource_id)
                return el
            await asyncio.sleep(poll_interval)
        logger.warning("Timed out waiting for element (text=%r, id=%r)", text, resource_id)
        return None

    async def tap_element(self, element: UIElement) -> ActionResult:
        """Tap the center of a UIElement."""
        cx, cy = element.center
        logger.info("Tapping element '%s' at (%d, %d)", element.text or element.resource_id, cx, cy)
        return await self.tap(cx, cy)

    async def find_and_tap(
        self,
        text: Optional[str] = None,
        resource_id: Optional[str] = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
    ) -> ActionResult:
        """Wait for an element, then tap it. Combines wait_for_element + tap_element."""
        el = await self.wait_for_element(text=text, resource_id=resource_id, timeout=timeout)
        if el is None:
            return ActionResult(
                success=False,
                action=DeviceAction(ActionType.TAP, {"text": text, "resource_id": resource_id}),
                error=f"Element not found: text={text!r}, id={resource_id!r}",
            )
        return await self.tap_element(el)


# ===================================================================
# VisionLoop — Screenshot -> Analyze -> Act -> Verify cycle
# ===================================================================

class VisionLoop:
    """
    Implements a vision-guided automation loop: capture the screen, send
    it to an AI vision model for analysis, determine the next action,
    execute it, then verify the result with another screenshot.

    The loop drives the TaskExecutor when element-based navigation fails
    or when the task requires visual understanding (e.g., identifying
    buttons by their appearance rather than accessibility labels).

    Supports two vision backends:
        1. Local Vision Service (default) — Anthropic Haiku via the empire
           vision service running on localhost:8002
        2. Direct Anthropic API — Claude Haiku with vision, called directly

    Example:
        vision = VisionLoop(controller)
        analysis = await vision.analyze_screen("What app is open? What elements are visible?")
        action = await vision.decide_action("I need to tap the compose button")
        await controller.tap(action["x"], action["y"])
    """

    def __init__(
        self,
        controller: PhoneController,
        vision_url: str = VISION_SERVICE_URL,
        anthropic_key: str = ANTHROPIC_API_KEY,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.controller = controller
        self.vision_url = vision_url.rstrip("/")
        self.anthropic_key = anthropic_key
        self.model = model
        self._loop_count: int = 0
        self._max_loops: int = 25  # Safety limit per task

    async def analyze_screen(self, prompt: str, screenshot_path: Optional[str] = None) -> VisionAnalysis:
        """
        Capture (or reuse) a screenshot and send it to the vision model
        for analysis. Returns a structured VisionAnalysis.
        """
        if screenshot_path is None:
            screenshot_path = await self.controller.screenshot()

        image_b64 = self._load_image_b64(screenshot_path)
        raw_response = await self._call_vision(image_b64, prompt)

        return self._parse_vision_response(raw_response)

    async def decide_action(
        self,
        goal: str,
        screenshot_path: Optional[str] = None,
        context: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Given a goal description, analyze the current screen and return
        the recommended next action as a dictionary.

        Returns a dict with keys: action_type, params, reasoning
        or None if the vision model determines no action is needed.
        """
        prompt = (
            f"You are an Android automation assistant. The user's goal is: {goal}\n"
            f"Additional context: {context}\n\n"
            "Analyze the screenshot and determine the SINGLE next action to take.\n"
            "Respond in JSON with these fields:\n"
            '  "action_type": one of "tap", "type_text", "swipe", "scroll_down", "scroll_up", '
            '"press_back", "press_home", "wait", "done"\n'
            '  "params": action-specific parameters (x, y for tap; text for type_text; etc.)\n'
            '  "reasoning": brief explanation of why this action\n'
            '  "confidence": 0.0 to 1.0\n\n'
            "If the goal appears to already be achieved, use action_type=done.\n"
            "Respond with ONLY valid JSON, no markdown fences."
        )
        if screenshot_path is None:
            screenshot_path = await self.controller.screenshot()

        image_b64 = self._load_image_b64(screenshot_path)
        raw = await self._call_vision(image_b64, prompt)

        try:
            # Strip potential markdown fencing
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```\w*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Vision response was not valid JSON: %s", raw[:200])
            return None

    async def vision_guided_step(
        self,
        goal: str,
        context: str = "",
    ) -> ActionResult:
        """
        Execute one full vision loop iteration: screenshot, analyze, act.
        Returns the result of the executed action.
        """
        self._loop_count += 1
        if self._loop_count > self._max_loops:
            return ActionResult(
                success=False,
                action=DeviceAction(ActionType.WAIT, {}, "Vision loop safety limit"),
                error=f"Vision loop exceeded {self._max_loops} iterations",
            )

        decision = await self.decide_action(goal, context=context)
        if decision is None:
            return ActionResult(
                success=False,
                action=DeviceAction(ActionType.WAIT, {}, "Vision returned no action"),
                error="Vision model returned unparseable response",
            )

        action_type = decision.get("action_type", "wait")
        params = decision.get("params", {})
        logger.info(
            "Vision decided: %s %s (confidence=%.2f, reason=%s)",
            action_type, params, decision.get("confidence", 0), decision.get("reasoning", ""),
        )

        if action_type == "done":
            return ActionResult(
                success=True,
                action=DeviceAction(ActionType.WAIT, {}, "Goal achieved"),
                output="Vision determined goal is achieved",
            )

        return await self._execute_vision_action(action_type, params)

    async def _execute_vision_action(self, action_type: str, params: Dict[str, Any]) -> ActionResult:
        """Translate a vision decision into a PhoneController action."""
        ctrl = self.controller
        if action_type == "tap":
            return await ctrl.tap(int(params.get("x", 0)), int(params.get("y", 0)))
        elif action_type == "type_text":
            return await ctrl.type_text(params.get("text", ""))
        elif action_type == "swipe":
            return await ctrl.swipe(
                int(params.get("x1", 0)), int(params.get("y1", 0)),
                int(params.get("x2", 0)), int(params.get("y2", 0)),
                int(params.get("duration_ms", 300)),
            )
        elif action_type == "scroll_down":
            return await ctrl.scroll_down(int(params.get("distance", 500)))
        elif action_type == "scroll_up":
            return await ctrl.scroll_up(int(params.get("distance", 500)))
        elif action_type == "press_back":
            return await ctrl.press_back()
        elif action_type == "press_home":
            return await ctrl.press_home()
        elif action_type == "wait":
            await asyncio.sleep(float(params.get("seconds", 1.0)))
            return ActionResult(success=True, action=DeviceAction(ActionType.WAIT, params, "Wait"))
        else:
            return ActionResult(
                success=False,
                action=DeviceAction(ActionType.WAIT, {}),
                error=f"Unknown vision action type: {action_type}",
            )

    def reset_loop_count(self) -> None:
        """Reset the loop counter (call at the start of a new task)."""
        self._loop_count = 0

    # ----- Vision backend calls -----

    async def _call_vision(self, image_b64: str, prompt: str) -> str:
        """
        Send an image + prompt to the vision backend and return the text response.

        Tries the local vision service first, falls back to direct Anthropic API.
        """
        try:
            return await self._call_local_vision(image_b64, prompt)
        except Exception as local_err:
            logger.debug("Local vision service unavailable (%s), falling back to Anthropic API", local_err)

        if not self.anthropic_key:
            raise RuntimeError(
                "No vision backend available: local service down and ANTHROPIC_API_KEY not set"
            )
        return await self._call_anthropic_vision(image_b64, prompt)

    async def _call_local_vision(self, image_b64: str, prompt: str) -> str:
        """Call the local empire vision service at /vision/analyze."""
        session = await self.controller._ensure_session()
        payload = {"image": image_b64, "prompt": prompt}
        async with session.post(f"{self.vision_url}/vision/analyze", json=payload) as resp:
            if resp.status != 200:
                raise ConnectionError(f"Vision service returned {resp.status}")
            data = await resp.json()
            return data.get("analysis", data.get("text", ""))

    async def _call_anthropic_vision(self, image_b64: str, prompt: str) -> str:
        """Call Anthropic's Messages API directly with a vision request."""
        session = await self.controller._ensure_session()
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        async with session.post(
            "https://api.anthropic.com/v1/messages", json=payload, headers=headers
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Anthropic API error {resp.status}: {body[:300]}")
            data = await resp.json()
            content_blocks = data.get("content", [])
            return content_blocks[0].get("text", "") if content_blocks else ""

    @staticmethod
    def _load_image_b64(path: str) -> str:
        """Read an image file and return its base64 encoding."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def _parse_vision_response(self, raw: str) -> VisionAnalysis:
        """Parse a free-form vision response into a VisionAnalysis structure."""
        analysis = VisionAnalysis(raw_response=raw)

        # Try JSON parse first
        try:
            data = json.loads(raw)
            analysis.description = data.get("description", raw)
            analysis.current_app = data.get("current_app", "")
            analysis.current_screen = data.get("current_screen", "")
            analysis.elements_detected = data.get("elements", [])
            analysis.suggested_action = data.get("suggested_action")
            analysis.confidence = float(data.get("confidence", 0.5))
            return analysis
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to treating the whole response as description
        analysis.description = raw
        analysis.confidence = 0.5
        return analysis


# ===================================================================
# AppNavigator — App-specific navigation patterns
# ===================================================================

class AppNavigator:
    """
    Loads app-specific navigation patterns from ``configs/app-registry.json``
    and provides structured navigation helpers for known apps.

    The registry maps package names to navigation flows (e.g., how to create
    a post on Facebook, how to send a tweet, how to open Instagram DMs).

    If the registry file does not exist, the navigator operates in
    vision-only mode, relying entirely on VisionLoop for navigation.

    Registry format:
        {
            "apps": {
                "com.facebook.katana": {
                    "name": "Facebook",
                    "launch_activity": ".LoginActivity",
                    "flows": {
                        "create_post": [
                            {"action": "tap", "target": "What's on your mind?"},
                            {"action": "type_text", "param": "$TEXT"},
                            {"action": "tap", "target": "Post"}
                        ]
                    }
                }
            }
        }
    """

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        self.registry_path = registry_path or APP_REGISTRY_PATH
        self._apps: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load the app registry from disk, if it exists."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._apps = data.get("apps", {})
                logger.info("Loaded app registry with %d apps", len(self._apps))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load app registry: %s", exc)
        else:
            logger.info("No app registry found at %s — operating in vision-only mode", self.registry_path)

    def get_app_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Look up an app's metadata and navigation flows."""
        return self._apps.get(package_name)

    def get_flow(self, package_name: str, flow_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get a specific navigation flow for an app."""
        app = self._apps.get(package_name)
        if app:
            return app.get("flows", {}).get(flow_name)
        return None

    def get_launch_activity(self, package_name: str) -> Optional[str]:
        """Get the launch activity for an app, if known."""
        app = self._apps.get(package_name)
        if app:
            return app.get("launch_activity")
        return None

    def resolve_package(self, app_name: str) -> Optional[str]:
        """
        Resolve a human-readable app name to its package name.

        Searches the registry for matching names (case-insensitive).
        Also checks a built-in fallback map of common apps.
        """
        name_lower = app_name.lower().strip()

        # Search registry
        for pkg, info in self._apps.items():
            if info.get("name", "").lower() == name_lower:
                return pkg

        # Built-in fallback for common apps
        common_apps = {
            "facebook": "com.facebook.katana",
            "instagram": "com.instagram.android",
            "twitter": "com.twitter.android",
            "x": "com.twitter.android",
            "whatsapp": "com.whatsapp",
            "telegram": "org.telegram.messenger",
            "youtube": "com.google.android.youtube",
            "chrome": "com.android.chrome",
            "gmail": "com.google.android.gm",
            "camera": "com.android.camera2",
            "settings": "com.android.settings",
            "tiktok": "com.zhiliaoapp.musically",
            "spotify": "com.spotify.music",
            "discord": "com.discord",
            "pinterest": "com.pinterest",
            "reddit": "com.reddit.frontpage",
            "snapchat": "com.snapchat.android",
            "linkedin": "com.linkedin.android",
            "canva": "com.canva.editor",
            "etsy": "com.etsy.android",
            "amazon": "com.amazon.mShop.android.shopping",
        }
        return common_apps.get(name_lower)

    def list_known_flows(self, package_name: str) -> List[str]:
        """List available navigation flows for an app."""
        app = self._apps.get(package_name)
        if app:
            return list(app.get("flows", {}).keys())
        return []

    async def execute_flow(
        self,
        controller: PhoneController,
        package_name: str,
        flow_name: str,
        variables: Optional[Dict[str, str]] = None,
    ) -> List[ActionResult]:
        """
        Execute a predefined navigation flow for an app.

        Variables in flow steps (prefixed with ``$``) are replaced with
        values from the ``variables`` dict.
        """
        flow = self.get_flow(package_name, flow_name)
        if not flow:
            raise ValueError(f"No flow '{flow_name}' found for {package_name}")

        variables = variables or {}
        results: List[ActionResult] = []

        for step in flow:
            action = step.get("action", "")
            target = step.get("target", "")
            param = step.get("param", "")

            # Resolve variables
            for var_name, var_value in variables.items():
                target = target.replace(f"${var_name}", var_value)
                param = param.replace(f"${var_name}", var_value)

            if action == "tap":
                result = await controller.find_and_tap(text=target)
            elif action == "type_text":
                result = await controller.type_text(param)
            elif action == "wait":
                await asyncio.sleep(float(param or "1"))
                result = ActionResult(
                    success=True,
                    action=DeviceAction(ActionType.WAIT, {"seconds": param}),
                )
            elif action == "scroll_down":
                result = await controller.scroll_down()
            elif action == "back":
                result = await controller.press_back()
            else:
                logger.warning("Unknown flow action: %s", action)
                continue

            results.append(result)
            if not result.success:
                logger.error("Flow step failed: %s (target=%s)", action, target)
                break

        return results


# ===================================================================
# TaskExecutor — High-level task decomposition and execution
# ===================================================================

class TaskExecutor:
    """
    Breaks high-level natural-language commands into executable step
    sequences, runs them with verification and retry logic, and
    integrates with FORGE (pre-task analysis) and AMPLIFY (enhancement).

    The executor uses a combination of:
        1. AppNavigator flows — for known app patterns
        2. VisionLoop — for dynamic/unknown screens
        3. PhoneController — for direct element-based actions

    FORGE integration (pre-task):
        - Scout: audits the task plan for potential issues
        - Oracle: predicts likely failure points
        - Codex: recalls outcomes of similar past tasks

    AMPLIFY integration (enhancement):
        - Enrich: adds context-specific details to each step
        - Fortify: adds retry logic and fallback actions
        - Anticipate: predicts UI state transitions
        - Validate: pre-execution sanity checks

    Example:
        executor = TaskExecutor(controller)
        result = await executor.execute("open Instagram and like the first post")
    """

    def __init__(
        self,
        controller: PhoneController,
        vision: Optional[VisionLoop] = None,
        navigator: Optional[AppNavigator] = None,
        forge_engine: Optional[Any] = None,
        amplify_pipeline: Optional[Any] = None,
    ) -> None:
        self.controller = controller
        self.vision = vision or VisionLoop(controller)
        self.navigator = navigator or AppNavigator()
        self.forge = forge_engine
        self.amplify = amplify_pipeline
        self._task_history: List[TaskPlan] = []

    async def execute(self, task_description: str) -> TaskPlan:
        """
        Execute a high-level task from a natural-language description.

        Steps:
            1. Plan — decompose the task into steps
            2. FORGE analysis — pre-task audit (if available)
            3. AMPLIFY enhancement — enrich and fortify steps (if available)
            4. Execute — run each step with verification
            5. Record — save outcome for future learning
        """
        logger.info("=== Task: %s ===", task_description)

        # Step 1: Plan
        plan = await self._plan_task(task_description)
        logger.info("Plan created: %d steps", len(plan.steps))

        # Step 2: FORGE pre-analysis
        if self.forge is not None:
            plan.forge_analysis = await self._run_forge_analysis(plan)
            logger.info("FORGE analysis: %s", plan.forge_analysis)

        # Step 3: AMPLIFY enhancement
        if self.amplify is not None:
            plan.amplify_enhancements = await self._run_amplify(plan)
            logger.info("AMPLIFY enhancements applied")

        # Step 4: Execute
        plan.status = "running"
        self.vision.reset_loop_count()

        for step in plan.steps:
            logger.info("Step %d/%d: %s", step.step_number, len(plan.steps), step.description)
            result = await self._execute_step(step)
            step.result = result
            step.completed = result.success

            if not result.success:
                logger.warning("Step %d failed: %s", step.step_number, result.error)
                # Try fallback actions
                recovered = await self._try_fallbacks(step)
                if not recovered:
                    plan.status = "failed"
                    logger.error("Task failed at step %d", step.step_number)
                    break
                step.completed = True

        if plan.status != "failed":
            plan.status = "completed"
            logger.info("=== Task completed successfully ===")

        # Step 5: Record
        self._task_history.append(plan)
        if self.forge is not None:
            await self._record_to_forge(plan)

        return plan

    async def _plan_task(self, description: str) -> TaskPlan:
        """
        Decompose a task description into a TaskPlan with steps.

        Uses the vision model to analyze the task and generate a step plan,
        or falls back to heuristic decomposition for common patterns.
        """
        plan = TaskPlan(task_description=description)

        # Check for known app patterns first
        app_match = self._extract_app_intent(description)
        if app_match:
            app_name, intent, params = app_match
            package = self.navigator.resolve_package(app_name)
            if package:
                # Start with launching the app
                plan.steps.append(TaskStep(
                    step_number=1,
                    description=f"Launch {app_name}",
                    action=DeviceAction(
                        ActionType.LAUNCH_APP,
                        {"package": package},
                        f"Launch {app_name}",
                    ),
                    expected_result=f"{app_name} main screen visible",
                ))

                # Check for known flows
                flow = self.navigator.get_flow(package, intent)
                if flow:
                    for i, step_def in enumerate(flow, start=2):
                        action_type = ActionType.TAP  # default
                        step_params: Dict[str, Any] = {}

                        if step_def.get("action") == "type_text":
                            action_type = ActionType.TYPE_TEXT
                            text = step_def.get("param", "")
                            for key, val in params.items():
                                text = text.replace(f"${key}", val)
                            step_params = {"text": text}
                        elif step_def.get("action") == "tap":
                            step_params = {"text": step_def.get("target", "")}
                        elif step_def.get("action") == "scroll_down":
                            action_type = ActionType.SCROLL_DOWN
                        elif step_def.get("action") == "wait":
                            action_type = ActionType.WAIT
                            step_params = {"seconds": step_def.get("param", "2")}

                        plan.steps.append(TaskStep(
                            step_number=i,
                            description=f"{step_def.get('action', 'tap')} {step_def.get('target', step_def.get('param', ''))}",
                            action=DeviceAction(action_type, step_params),
                        ))
                    return plan

                # No known flow — use vision-based steps after launch
                plan.steps.append(TaskStep(
                    step_number=2,
                    description=f"Navigate to complete: {intent}",
                    action=DeviceAction(
                        ActionType.WAIT,
                        {"goal": intent, "params": params},
                        "Vision-guided navigation",
                    ),
                    expected_result=f"{intent} completed",
                ))
                return plan

        # Fully unknown task — create a single vision-guided step
        plan.steps.append(TaskStep(
            step_number=1,
            description=description,
            action=DeviceAction(
                ActionType.WAIT,
                {"goal": description},
                "Vision-guided full task",
            ),
            expected_result="Task completed",
        ))
        return plan

    def _extract_app_intent(self, description: str) -> Optional[Tuple[str, str, Dict[str, str]]]:
        """
        Parse a natural-language task description into (app_name, intent, params).

        Handles patterns like:
            "open Facebook and create a post saying Hello"
            "launch Instagram and follow user @example"
            "go to Twitter and search for AI news"
        """
        desc_lower = description.lower()

        # Pattern: "open/launch/go to <APP> and <ACTION>"
        pattern = re.compile(
            r"(?:open|launch|go\s+to|start)\s+"
            r"(?P<app>\w+(?:\s+\w+)?)\s+"
            r"(?:and|then|to)\s+"
            r"(?P<intent>.+)",
            re.IGNORECASE,
        )
        match = pattern.match(description)
        if match:
            app_name = match.group("app").strip()
            intent_raw = match.group("intent").strip()

            # Extract quoted parameters
            params: Dict[str, str] = {}
            text_match = re.search(r'(?:saying|with text|with message|posting)\s+"?([^"]+)"?', intent_raw, re.IGNORECASE)
            if text_match:
                params["TEXT"] = text_match.group(1).strip().rstrip('"')

            user_match = re.search(r'(?:user|@)(\w+)', intent_raw)
            if user_match:
                params["USER"] = user_match.group(1)

            query_match = re.search(r'(?:search(?:\s+for)?|find)\s+"?([^"]+)"?', intent_raw, re.IGNORECASE)
            if query_match:
                params["QUERY"] = query_match.group(1).strip().rstrip('"')

            # Normalize intent
            intent = re.sub(r'\s+saying\s+.+', '', intent_raw, flags=re.IGNORECASE)
            intent = re.sub(r'\s+with\s+text\s+.+', '', intent, flags=re.IGNORECASE)
            intent = intent.strip().replace(" ", "_")

            return (app_name, intent, params)

        return None

    async def _execute_step(self, step: TaskStep) -> ActionResult:
        """
        Execute a single task step, dispatching to the appropriate method.

        Vision-guided steps (those with a 'goal' param) are handled by
        the VisionLoop. Direct steps go through the PhoneController.
        """
        action = step.action

        # Vision-guided steps
        if action.params.get("goal"):
            goal = action.params["goal"]
            max_iterations = 10
            for i in range(max_iterations):
                result = await self.vision.vision_guided_step(goal)
                if result.success and result.output and "goal is achieved" in result.output.lower():
                    return result
                if not result.success:
                    return result
                # Continue the loop for more steps
            return ActionResult(
                success=False,
                action=action,
                error=f"Vision loop did not achieve goal in {max_iterations} iterations",
            )

        # Element-based tap (find by text)
        if action.action_type == ActionType.TAP and action.params.get("text"):
            return await self.controller.find_and_tap(text=action.params["text"])

        # Direct coordinate tap
        if action.action_type == ActionType.TAP and "x" in action.params:
            return await self.controller.tap(action.params["x"], action.params["y"])

        # Type text
        if action.action_type == ActionType.TYPE_TEXT:
            return await self.controller.type_text(action.params.get("text", ""))

        # Launch app
        if action.action_type == ActionType.LAUNCH_APP:
            return await self.controller.launch_app(action.params["package"])

        # Scroll
        if action.action_type == ActionType.SCROLL_DOWN:
            return await self.controller.scroll_down()
        if action.action_type == ActionType.SCROLL_UP:
            return await self.controller.scroll_up()

        # Key events
        if action.action_type == ActionType.KEY_EVENT:
            return await self.controller.press_key(action.params.get("keycode", 4))
        if action.action_type == ActionType.BACK:
            return await self.controller.press_back()
        if action.action_type == ActionType.HOME:
            return await self.controller.press_home()

        # Wait
        if action.action_type == ActionType.WAIT:
            seconds = float(action.params.get("seconds", 1))
            await asyncio.sleep(seconds)
            return ActionResult(success=True, action=action, output=f"Waited {seconds}s")

        # Screenshot
        if action.action_type == ActionType.SCREENSHOT:
            path = await self.controller.screenshot()
            return ActionResult(success=True, action=action, screenshot_path=path)

        # UI dump
        if action.action_type == ActionType.UI_DUMP:
            elements = await self.controller.ui_dump()
            return ActionResult(
                success=True, action=action,
                output=f"Found {len(elements)} elements",
            )

        return ActionResult(success=False, action=action, error=f"Unhandled action type: {action.action_type}")

    async def _try_fallbacks(self, step: TaskStep) -> bool:
        """
        Attempt fallback actions for a failed step.
        Returns True if any fallback succeeded.
        """
        for i, fallback in enumerate(step.fallback_actions):
            logger.info("Trying fallback %d/%d for step %d", i + 1, len(step.fallback_actions), step.step_number)
            result = await self._execute_step(TaskStep(
                step_number=step.step_number,
                description=f"Fallback {i + 1}: {fallback.description}",
                action=fallback,
            ))
            if result.success:
                step.result = result
                return True

        # Last resort: try vision-guided recovery
        logger.info("All fallbacks exhausted, attempting vision-guided recovery")
        recovery_result = await self.vision.vision_guided_step(
            goal=step.description,
            context=f"Previous attempt failed with error: {step.result.error if step.result else 'unknown'}",
        )
        if recovery_result.success:
            step.result = recovery_result
            return True

        return False

    # ----- FORGE integration -----

    async def _run_forge_analysis(self, plan: TaskPlan) -> Dict[str, Any]:
        """
        Run FORGE pre-task analysis on the task plan.

        Integrates with the FORGE engine's Scout, Oracle, and Codex modules
        to audit the plan, predict failures, and recall past outcomes.
        """
        analysis: Dict[str, Any] = {"scout": {}, "oracle": {}, "codex": {}}

        if not self.forge:
            return analysis

        try:
            # Scout: audit the task plan for potential issues
            if hasattr(self.forge, "scout"):
                scout_result = self.forge.scout.audit_config({
                    "task": plan.task_description,
                    "steps": len(plan.steps),
                    "step_descriptions": [s.description for s in plan.steps],
                })
                analysis["scout"] = {"issues": scout_result} if scout_result else {"issues": []}

            # Oracle: predict failure points
            if hasattr(self.forge, "oracle"):
                prediction = self.forge.oracle.predict_failure({
                    "task": plan.task_description,
                    "step_count": len(plan.steps),
                })
                analysis["oracle"] = prediction if isinstance(prediction, dict) else {"prediction": str(prediction)}

            # Codex: recall similar past tasks
            if hasattr(self.forge, "codex"):
                similar = self.forge.codex.recall(plan.task_description)
                analysis["codex"] = {"similar_tasks": similar} if similar else {"similar_tasks": []}

        except Exception as exc:
            logger.warning("FORGE analysis error (non-fatal): %s", exc)
            analysis["error"] = str(exc)

        return analysis

    async def _run_amplify(self, plan: TaskPlan) -> Dict[str, Any]:
        """
        Run AMPLIFY enhancement pipeline on the task plan.

        Integrates with the AMPLIFY pipeline's stages to enrich,
        fortify, and validate each step before execution.
        """
        enhancements: Dict[str, Any] = {"applied": []}

        if not self.amplify:
            return enhancements

        try:
            # Process each step through AMPLIFY
            if hasattr(self.amplify, "process_action"):
                for step in plan.steps:
                    enhanced = self.amplify.process_action(
                        step.description,
                        {"action_type": step.action.action_type.value, "params": step.action.params},
                    )
                    if enhanced and isinstance(enhanced, dict):
                        # Apply retry enhancements
                        if "max_retries" in enhanced:
                            step.max_retries = enhanced["max_retries"]
                        # Apply fallback actions
                        if "fallbacks" in enhanced:
                            for fb in enhanced["fallbacks"]:
                                step.fallback_actions.append(DeviceAction(
                                    action_type=ActionType(fb.get("action_type", "wait")),
                                    params=fb.get("params", {}),
                                    description=fb.get("description", "AMPLIFY fallback"),
                                ))
                        enhancements["applied"].append(step.step_number)

            # Full pipeline validation
            if hasattr(self.amplify, "validate"):
                validation = self.amplify.validate({
                    "task": plan.task_description,
                    "steps": [s.description for s in plan.steps],
                })
                enhancements["validation"] = validation

        except Exception as exc:
            logger.warning("AMPLIFY enhancement error (non-fatal): %s", exc)
            enhancements["error"] = str(exc)

        return enhancements

    async def _record_to_forge(self, plan: TaskPlan) -> None:
        """Record task outcome to FORGE Codex for future learning."""
        if not self.forge or not hasattr(self.forge, "codex"):
            return

        try:
            self.forge.codex.record_outcome(plan.task_id, {
                "task": plan.task_description,
                "status": plan.status,
                "steps_total": len(plan.steps),
                "steps_completed": sum(1 for s in plan.steps if s.completed),
                "duration_ms": sum(
                    s.result.duration_ms for s in plan.steps if s.result
                ),
                "timestamp": datetime.utcnow().isoformat(),
            })
        except Exception as exc:
            logger.debug("Failed to record to FORGE Codex: %s", exc)

    # ----- Convenience task methods -----

    async def open_app(self, app_name: str) -> ActionResult:
        """Launch an app by its human-readable name."""
        package = self.navigator.resolve_package(app_name)
        if not package:
            return ActionResult(
                success=False,
                action=DeviceAction(ActionType.LAUNCH_APP, {"app_name": app_name}),
                error=f"Unknown app: {app_name}. Could not resolve package name.",
            )
        activity = self.navigator.get_launch_activity(package)
        return await self.controller.launch_app(package, activity)

    async def screenshot_and_describe(self) -> Tuple[str, VisionAnalysis]:
        """Take a screenshot and return both the file path and vision analysis."""
        path = await self.controller.screenshot()
        analysis = await self.vision.analyze_screen(
            "Describe what is currently shown on the Android screen. "
            "Identify the app, the current screen/view, and all interactive elements visible.",
            screenshot_path=path,
        )
        return path, analysis

    async def find_and_interact(
        self,
        target_text: str,
        interaction: str = "tap",
        type_text: Optional[str] = None,
    ) -> ActionResult:
        """
        Find a UI element by text and interact with it.

        Supports tap, long_press, and type (tap then type text).
        Falls back to vision-guided search if element-based search fails.
        """
        el = await self.controller.wait_for_element(text=target_text, timeout=10)

        if el is None:
            # Fallback: try vision to find the element
            logger.info("Element '%s' not found via UI dump, trying vision", target_text)
            decision = await self.vision.decide_action(
                f"Find and {interaction} the element with text '{target_text}'",
            )
            if decision and decision.get("action_type") == "tap":
                return await self.controller.tap(
                    int(decision["params"].get("x", 0)),
                    int(decision["params"].get("y", 0)),
                )
            return ActionResult(
                success=False,
                action=DeviceAction(ActionType.TAP, {"text": target_text}),
                error=f"Could not find element: {target_text}",
            )

        if interaction == "long_press":
            cx, cy = el.center
            return await self.controller.long_press(cx, cy)
        elif interaction == "type" and type_text:
            await self.controller.tap_element(el)
            await asyncio.sleep(0.3)
            return await self.controller.type_text(type_text)
        else:
            return await self.controller.tap_element(el)

    def get_task_history(self) -> List[Dict[str, Any]]:
        """Return a summary of all executed tasks."""
        return [
            {
                "task_id": p.task_id,
                "description": p.task_description,
                "status": p.status,
                "steps": len(p.steps),
                "completed_steps": sum(1 for s in p.steps if s.completed),
                "created_at": p.created_at,
            }
            for p in self._task_history
        ]


# ===================================================================
# Module-level convenience functions
# ===================================================================

async def quick_task(
    task: str,
    node_url: str = DEFAULT_NODE_URL,
    node_name: str = DEFAULT_NODE_NAME,
) -> TaskPlan:
    """
    One-shot convenience function: connect to the Android node and
    execute a task.

    Example:
        result = await quick_task("open Chrome and search for Python tutorials")
    """
    controller = PhoneController(node_url=node_url, node_name=node_name)
    try:
        connected = await controller.connect()
        if not connected:
            plan = TaskPlan(task_description=task, status="failed")
            logger.error("Could not connect to Android node at %s", node_url)
            return plan

        executor = TaskExecutor(controller)
        return await executor.execute(task)
    finally:
        await controller.close()


def run_task(task: str, **kwargs: Any) -> TaskPlan:
    """
    Synchronous wrapper around ``quick_task`` for non-async callers.

    Example:
        result = run_task("open Settings and enable WiFi")
    """
    return asyncio.run(quick_task(task, **kwargs))


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenClaw Empire Phone Controller — Android automation via ADB",
    )
    parser.add_argument("task", nargs="?", help="Task to execute (natural language)")
    parser.add_argument("--node-url", default=DEFAULT_NODE_URL, help="OpenClaw node URL")
    parser.add_argument("--node-name", default=DEFAULT_NODE_NAME, help="Node name")
    parser.add_argument("--screenshot", action="store_true", help="Just take a screenshot")
    parser.add_argument("--describe", action="store_true", help="Screenshot + vision description")
    parser.add_argument("--ui-dump", action="store_true", help="Dump UI hierarchy")
    parser.add_argument("--launch", type=str, help="Launch app by name")
    parser.add_argument("--tap", type=str, help="Tap coordinates (x,y)")
    parser.add_argument("--type", type=str, dest="type_text", help="Type text")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    async def _main() -> None:
        controller = PhoneController(node_url=args.node_url, node_name=args.node_name)
        try:
            connected = await controller.connect()
            if not connected:
                print(f"ERROR: Could not connect to Android node at {args.node_url}")
                return

            print(f"Connected. Device resolution: {controller.resolution[0]}x{controller.resolution[1]}")

            if args.screenshot:
                path = await controller.screenshot()
                print(f"Screenshot saved: {path}")

            elif args.describe:
                executor = TaskExecutor(controller)
                path, analysis = await executor.screenshot_and_describe()
                print(f"Screenshot: {path}")
                print(f"App: {analysis.current_app}")
                print(f"Screen: {analysis.current_screen}")
                print(f"Description: {analysis.description}")

            elif args.ui_dump:
                elements = await controller.ui_dump()
                for el in elements:
                    if el.text or el.content_desc:
                        print(
                            f"  [{el.class_name.split('.')[-1]}] "
                            f"text={el.text!r} desc={el.content_desc!r} "
                            f"id={el.resource_id} bounds={el.bounds} "
                            f"click={el.clickable}"
                        )
                print(f"\nTotal elements: {len(elements)}")

            elif args.launch:
                executor = TaskExecutor(controller)
                result = await executor.open_app(args.launch)
                print(f"Launch {'OK' if result.success else 'FAILED'}: {result.error or ''}")

            elif args.tap:
                x, y = map(int, args.tap.split(","))
                result = await controller.tap(x, y)
                print(f"Tap {'OK' if result.success else 'FAILED'}")

            elif args.type_text:
                result = await controller.type_text(args.type_text)
                print(f"Type {'OK' if result.success else 'FAILED'}")

            elif args.task:
                executor = TaskExecutor(controller)
                plan = await executor.execute(args.task)
                print(f"\nTask: {plan.task_description}")
                print(f"Status: {plan.status}")
                print(f"Steps: {sum(1 for s in plan.steps if s.completed)}/{len(plan.steps)} completed")
                for step in plan.steps:
                    status_mark = "[OK]" if step.completed else "[FAIL]"
                    print(f"  {status_mark} Step {step.step_number}: {step.description}")
                    if step.result and step.result.error:
                        print(f"         Error: {step.result.error}")

            else:
                parser.print_help()

        finally:
            await controller.close()

    asyncio.run(_main())
