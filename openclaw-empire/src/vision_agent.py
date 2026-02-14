"""
Vision Agent for OpenClaw Empire
Analyzes Android phone screenshots via an AI vision service.

Supports:
- General phone screen analysis (identify app, read text, find elements)
- UI element location by description (returns tap coordinates)
- App state detection (loading, error, logged out, etc.)
- Error/crash/permission dialog detection
- Before/after screenshot comparison
- FORGE Sentinel integration for optimized prompt routing and quality scoring

Usage:
    from src.vision_agent import VisionAgent

    agent = VisionAgent()
    result = await agent.analyze_screen(screenshot_path="/tmp/phone.png")
    element = await agent.find_element("the login button", image_b64="...")
    state = await agent.detect_state(screenshot_path="/tmp/phone.png")
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class AppState(Enum):
    """Detected phone application states."""
    READY = "ready"
    LOADING = "loading"
    ERROR = "error"
    LOGGED_OUT = "logged_out"
    PERMISSION_REQUEST = "permission_request"
    CRASH = "crash"
    KEYBOARD_VISIBLE = "keyboard_visible"
    DIALOG_OPEN = "dialog_open"
    UNKNOWN = "unknown"


@dataclass
class ElementLocation:
    """A located UI element with bounding box and confidence."""
    description: str
    x: int
    y: int
    width: int = 0
    height: int = 0
    confidence: float = 0.0
    tappable: bool = True
    text: Optional[str] = None

    @property
    def center(self) -> Tuple[int, int]:
        """Tap target: center of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class ScreenAnalysis:
    """Full analysis of a phone screenshot."""
    current_app: str = ""
    current_screen: str = ""
    visible_text: List[str] = field(default_factory=list)
    tappable_elements: List[ElementLocation] = field(default_factory=list)
    navigation_state: Dict[str, Any] = field(default_factory=dict)
    keyboard_visible: bool = False
    loading_indicators: bool = False
    errors_detected: List[str] = field(default_factory=list)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    analysis_time_ms: float = 0.0


@dataclass
class ComparisonResult:
    """Result of comparing two phone screenshots."""
    changed: bool = False
    changes: List[str] = field(default_factory=list)
    before_state: str = ""
    after_state: str = ""
    progress_detected: bool = False
    error_appeared: bool = False
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorDetection:
    """Detected errors, crashes, or permission requests on phone screen."""
    has_errors: bool = False
    error_type: str = ""  # "dialog", "crash", "permission", "toast", "banner"
    error_message: str = ""
    dismissable: bool = False
    dismiss_button: Optional[ElementLocation] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FORGE Sentinel integration
# ---------------------------------------------------------------------------

class FORGESentinel:
    """
    Integrates with FORGE Sentinel for optimized prompt routing.

    The Sentinel provides domain-specific prompt templates for phone screen
    analysis and collects quality scores to refine future prompts.
    """

    # Optimized prompt templates for phone screen analysis
    PROMPTS = {
        "analyze": (
            "Analyze this Android phone screenshot. Identify:\n"
            "1. Current app name and screen/activity\n"
            "2. All visible text (read OCR-style)\n"
            "3. All tappable UI elements with approximate coordinates (x, y, width, height)\n"
            "4. Navigation state: active tab, screen depth, back button available\n"
            "5. Keyboard visibility\n"
            "6. Loading spinners or progress bars\n"
            "7. Any error messages or dialogs\n"
            "Respond in JSON with keys: app, screen, visible_text[], "
            "tappable_elements[], nav_state{}, keyboard_visible, loading, errors[]"
        ),
        "find_element": (
            "Find the UI element described as: '{description}'\n"
            "Return its bounding box as JSON: {{x, y, width, height, confidence, "
            "tappable, text}}\n"
            "Coordinates should be in pixels from top-left. If the element is not "
            "found, return {{\"found\": false, \"reason\": \"...\"}}."
        ),
        "detect_state": (
            "Determine the current state of this phone screen. Classify as one of:\n"
            "- ready: App is interactive, no overlays\n"
            "- loading: Spinner, skeleton, progress bar visible\n"
            "- error: Error dialog, crash report, or error message\n"
            "- logged_out: Login screen, sign-in prompt\n"
            "- permission_request: System permission dialog\n"
            "- crash: App has crashed or ANR dialog\n"
            "- keyboard_visible: On-screen keyboard is showing\n"
            "- dialog_open: Modal dialog or bottom sheet open\n"
            "- unknown: Cannot determine\n"
            "Respond in JSON: {{state, confidence, details}}"
        ),
        "detect_errors": (
            "Scan this phone screenshot for any errors, crashes, or permission "
            "requests. Look for:\n"
            "- Error dialogs or toasts\n"
            "- App crash / ANR dialogs\n"
            "- Permission request popups\n"
            "- Network error banners\n"
            "- Form validation errors\n"
            "Respond in JSON: {{has_errors, error_type, error_message, "
            "dismissable, dismiss_button_coords}}"
        ),
        "compare": (
            "Compare these two phone screenshots (before and after).\n"
            "Identify:\n"
            "1. What changed between the screenshots\n"
            "2. Whether progress was made (e.g., screen advanced, form submitted)\n"
            "3. Whether any new errors appeared\n"
            "Respond in JSON: {{changed, changes[], before_state, after_state, "
            "progress_detected, error_appeared}}"
        ),
        "read_region": (
            "Read the text in the specified region of this phone screenshot.\n"
            "Region: x={x}, y={y}, width={w}, height={h}\n"
            "Return the text exactly as displayed, preserving line breaks."
        ),
    }

    def __init__(self):
        self._quality_scores: List[Dict[str, Any]] = []

    def get_prompt(self, task: str, **kwargs) -> str:
        """Get the optimized prompt template for a given task."""
        template = self.PROMPTS.get(task, "")
        if kwargs:
            template = template.format(**kwargs)
        return template

    def report_quality(self, task: str, score: float, metadata: Optional[Dict] = None):
        """Report a quality score back to Sentinel for future optimization."""
        entry = {
            "task": task,
            "score": score,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._quality_scores.append(entry)
        logger.debug("FORGE Sentinel quality report: %s -> %.2f", task, score)

    def get_quality_history(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve quality score history, optionally filtered by task."""
        if task:
            return [s for s in self._quality_scores if s["task"] == task]
        return list(self._quality_scores)

    def average_quality(self, task: Optional[str] = None) -> float:
        """Get the average quality score, optionally filtered by task."""
        scores = self.get_quality_history(task)
        if not scores:
            return 0.0
        return sum(s["score"] for s in scores) / len(scores)


# ---------------------------------------------------------------------------
# Vision Agent
# ---------------------------------------------------------------------------

class VisionAgent:
    """
    Analyzes Android phone screenshots via an AI vision service.

    The vision service (default http://localhost:8002) provides endpoints for
    general analysis, element finding, state detection, error detection, and
    screenshot comparison. This agent wraps those endpoints with typed
    responses, retry logic, FORGE Sentinel integration, and both async and
    sync interfaces.

    Args:
        base_url: Vision service base URL. Default ``http://localhost:8002``.
        timeout: Request timeout in seconds. Default 30.
        max_retries: Number of retry attempts on transient failures. Default 3.
        retry_delay: Base delay between retries in seconds (exponential backoff). Default 1.0.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sentinel = FORGESentinel()
        self._session: Optional[aiohttp.ClientSession] = None

    # -- session management --------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and return an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- image encoding ------------------------------------------------------

    @staticmethod
    def _load_image_b64(image_path: Optional[str] = None, image_b64: Optional[str] = None) -> str:
        """
        Return a base64-encoded image string.

        Accepts either a file path or a pre-encoded base64 string.
        Raises ValueError if neither is provided.
        """
        if image_b64:
            return image_b64
        if image_path:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Screenshot not found: {image_path}")
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        raise ValueError("Provide either image_path or image_b64")

    # -- HTTP transport with retries -----------------------------------------

    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST to the vision service with retry + exponential backoff.

        Returns the parsed JSON response body. Raises on non-recoverable errors.
        """
        url = f"{self.base_url}{endpoint}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    body = await resp.text()
                    if resp.status >= 500:
                        # Server error -- retriable
                        logger.warning(
                            "Vision service %s returned %d (attempt %d/%d): %s",
                            endpoint, resp.status, attempt, self.max_retries, body[:200],
                        )
                        last_error = RuntimeError(
                            f"Vision service {resp.status}: {body[:200]}"
                        )
                    else:
                        # Client error -- not retriable
                        raise RuntimeError(
                            f"Vision service {resp.status}: {body[:200]}"
                        )
            except aiohttp.ClientError as exc:
                logger.warning(
                    "Vision service connection error on %s (attempt %d/%d): %s",
                    endpoint, attempt, self.max_retries, exc,
                )
                last_error = exc
            except asyncio.TimeoutError:
                logger.warning(
                    "Vision service timeout on %s (attempt %d/%d)",
                    endpoint, attempt, self.max_retries,
                )
                last_error = TimeoutError(f"Timeout calling {endpoint}")

            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        raise last_error or RuntimeError("Vision service request failed")

    # -- core analysis methods -----------------------------------------------

    async def analyze_screen(
        self,
        image_path: Optional[str] = None,
        image_b64: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> ScreenAnalysis:
        """
        Perform general analysis of a phone screenshot.

        Identifies the current app, visible text, tappable elements,
        navigation state, keyboard visibility, loading indicators, and errors.

        Args:
            image_path: Path to the screenshot file.
            image_b64: Base64-encoded screenshot.
            custom_prompt: Override the default analysis prompt.

        Returns:
            A ScreenAnalysis dataclass with all extracted information.
        """
        start = time.monotonic()
        img = self._load_image_b64(image_path, image_b64)
        prompt = custom_prompt or self.sentinel.get_prompt("analyze")

        payload = {"image": img, "prompt": prompt}
        raw = await self._post("/vision/analyze", payload)

        elapsed = (time.monotonic() - start) * 1000

        # Parse structured response
        data = raw if isinstance(raw, dict) else {}
        tappable = []
        for elem in data.get("tappable_elements", []):
            tappable.append(ElementLocation(
                description=elem.get("description", elem.get("text", "")),
                x=elem.get("x", 0),
                y=elem.get("y", 0),
                width=elem.get("width", 0),
                height=elem.get("height", 0),
                confidence=elem.get("confidence", 0.0),
                tappable=elem.get("tappable", True),
                text=elem.get("text"),
            ))

        result = ScreenAnalysis(
            current_app=data.get("app", ""),
            current_screen=data.get("screen", ""),
            visible_text=data.get("visible_text", []),
            tappable_elements=tappable,
            navigation_state=data.get("nav_state", {}),
            keyboard_visible=data.get("keyboard_visible", False),
            loading_indicators=data.get("loading", False),
            errors_detected=data.get("errors", []),
            raw_response=raw,
            quality_score=data.get("confidence", 0.0),
            analysis_time_ms=elapsed,
        )

        self.sentinel.report_quality("analyze", result.quality_score, {
            "app": result.current_app,
            "time_ms": elapsed,
        })
        logger.info(
            "Screen analysis complete: app=%s, screen=%s, elements=%d (%.0fms)",
            result.current_app, result.current_screen,
            len(result.tappable_elements), elapsed,
        )
        return result

    async def find_element(
        self,
        description: str,
        image_path: Optional[str] = None,
        image_b64: Optional[str] = None,
    ) -> Optional[ElementLocation]:
        """
        Find a UI element on the phone screen by its description.

        Args:
            description: Natural-language description of the element (e.g.
                "the blue Login button", "hamburger menu icon").
            image_path: Path to the screenshot file.
            image_b64: Base64-encoded screenshot.

        Returns:
            An ElementLocation with coordinates, or None if not found.
        """
        img = self._load_image_b64(image_path, image_b64)
        prompt = self.sentinel.get_prompt("find_element", description=description)

        payload = {"image": img, "prompt": prompt, "description": description}
        raw = await self._post("/vision/find-element", payload)

        data = raw if isinstance(raw, dict) else {}
        if data.get("found") is False:
            logger.info("Element not found: '%s' -- reason: %s",
                        description, data.get("reason", "unknown"))
            self.sentinel.report_quality("find_element", 0.0, {
                "description": description, "found": False,
            })
            return None

        elem = ElementLocation(
            description=description,
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            confidence=data.get("confidence", 0.0),
            tappable=data.get("tappable", True),
            text=data.get("text"),
        )
        self.sentinel.report_quality("find_element", elem.confidence, {
            "description": description, "found": True,
        })
        logger.info(
            "Found element '%s' at (%d, %d) conf=%.2f",
            description, elem.x, elem.y, elem.confidence,
        )
        return elem

    async def detect_state(
        self,
        image_path: Optional[str] = None,
        image_b64: Optional[str] = None,
    ) -> Tuple[AppState, float, str]:
        """
        Detect the current application state of the phone screen.

        Returns:
            Tuple of (AppState enum, confidence 0-1, detail string).
        """
        img = self._load_image_b64(image_path, image_b64)
        prompt = self.sentinel.get_prompt("detect_state")

        payload = {"image": img, "prompt": prompt}
        raw = await self._post("/vision/detect-state", payload)

        data = raw if isinstance(raw, dict) else {}
        state_str = data.get("state", "unknown").upper()
        try:
            state = AppState(state_str.lower())
        except ValueError:
            state = AppState.UNKNOWN

        confidence = float(data.get("confidence", 0.0))
        details = data.get("details", "")

        self.sentinel.report_quality("detect_state", confidence, {
            "state": state.value,
        })
        logger.info("Phone state: %s (conf=%.2f) %s", state.value, confidence, details)
        return state, confidence, details

    async def detect_errors(
        self,
        image_path: Optional[str] = None,
        image_b64: Optional[str] = None,
    ) -> ErrorDetection:
        """
        Scan the phone screen for error dialogs, crashes, and permission requests.

        Returns:
            ErrorDetection with type, message, and dismiss info.
        """
        img = self._load_image_b64(image_path, image_b64)
        prompt = self.sentinel.get_prompt("detect_errors")

        payload = {"image": img, "prompt": prompt}
        raw = await self._post("/vision/detect-errors", payload)

        data = raw if isinstance(raw, dict) else {}
        dismiss_btn = None
        coords = data.get("dismiss_button_coords")
        if coords and isinstance(coords, dict):
            dismiss_btn = ElementLocation(
                description="dismiss button",
                x=coords.get("x", 0),
                y=coords.get("y", 0),
                width=coords.get("width", 0),
                height=coords.get("height", 0),
            )

        result = ErrorDetection(
            has_errors=data.get("has_errors", False),
            error_type=data.get("error_type", ""),
            error_message=data.get("error_message", ""),
            dismissable=data.get("dismissable", False),
            dismiss_button=dismiss_btn,
            raw_response=raw,
        )

        score = 1.0 if result.has_errors else 0.0
        self.sentinel.report_quality("detect_errors", score, {
            "error_type": result.error_type,
        })
        if result.has_errors:
            logger.warning("Error detected on phone: [%s] %s",
                           result.error_type, result.error_message)
        else:
            logger.info("No errors detected on phone screen")
        return result

    async def compare_screenshots(
        self,
        before_path: Optional[str] = None,
        before_b64: Optional[str] = None,
        after_path: Optional[str] = None,
        after_b64: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Compare two phone screenshots to detect changes.

        Args:
            before_path: File path to the before screenshot.
            before_b64: Base64-encoded before screenshot.
            after_path: File path to the after screenshot.
            after_b64: Base64-encoded after screenshot.

        Returns:
            ComparisonResult with change details and progress indicators.
        """
        before_img = self._load_image_b64(before_path, before_b64)
        after_img = self._load_image_b64(after_path, after_b64)
        prompt = self.sentinel.get_prompt("compare")

        payload = {"before": before_img, "after": after_img, "prompt": prompt}
        raw = await self._post("/vision/compare", payload)

        data = raw if isinstance(raw, dict) else {}
        result = ComparisonResult(
            changed=data.get("changed", False),
            changes=data.get("changes", []),
            before_state=data.get("before_state", ""),
            after_state=data.get("after_state", ""),
            progress_detected=data.get("progress_detected", False),
            error_appeared=data.get("error_appeared", False),
            raw_response=raw,
        )

        score = 1.0 if result.changed else 0.5
        self.sentinel.report_quality("compare", score, {
            "progress": result.progress_detected,
        })
        logger.info(
            "Screenshot comparison: changed=%s, progress=%s, errors=%s",
            result.changed, result.progress_detected, result.error_appeared,
        )
        return result

    # -- phone-specific helpers -----------------------------------------------

    async def read_text_region(
        self,
        x: int, y: int, w: int, h: int,
        image_path: Optional[str] = None,
        image_b64: Optional[str] = None,
    ) -> str:
        """
        Read text from a specific region of the phone screen.

        Args:
            x, y: Top-left corner of the region in pixels.
            w, h: Width and height of the region in pixels.
            image_path: Path to the screenshot file.
            image_b64: Base64-encoded screenshot.

        Returns:
            Extracted text from the region.
        """
        img = self._load_image_b64(image_path, image_b64)
        prompt = self.sentinel.get_prompt("read_region", x=x, y=y, w=w, h=h)

        payload = {"image": img, "prompt": prompt}
        raw = await self._post("/vision/analyze", payload)

        text = ""
        if isinstance(raw, dict):
            text = raw.get("text", raw.get("result", str(raw)))
        elif isinstance(raw, str):
            text = raw

        logger.debug("Read text from region (%d,%d,%d,%d): %s", x, y, w, h, text[:100])
        return text

    async def get_navigation_depth(
        self,
        image_path: Optional[str] = None,
        image_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Determine navigation depth and active tab on the phone screen.

        Returns a dict with: active_tab, screen_depth, back_available, breadcrumbs.
        """
        analysis = await self.analyze_screen(image_path=image_path, image_b64=image_b64)
        nav = analysis.navigation_state
        return {
            "active_tab": nav.get("active_tab", ""),
            "screen_depth": nav.get("screen_depth", 0),
            "back_available": nav.get("back_available", False),
            "breadcrumbs": nav.get("breadcrumbs", []),
            "current_app": analysis.current_app,
            "current_screen": analysis.current_screen,
        }

    async def is_keyboard_visible(
        self,
        image_path: Optional[str] = None,
        image_b64: Optional[str] = None,
    ) -> bool:
        """Check whether the on-screen keyboard is visible."""
        state, confidence, _ = await self.detect_state(
            image_path=image_path, image_b64=image_b64,
        )
        if state == AppState.KEYBOARD_VISIBLE:
            return True
        # Fall back to full analysis if state detection is ambiguous
        if confidence < 0.5:
            analysis = await self.analyze_screen(
                image_path=image_path, image_b64=image_b64,
            )
            return analysis.keyboard_visible
        return False

    async def is_loading(
        self,
        image_path: Optional[str] = None,
        image_b64: Optional[str] = None,
    ) -> bool:
        """Check whether the phone screen shows a loading indicator."""
        state, confidence, _ = await self.detect_state(
            image_path=image_path, image_b64=image_b64,
        )
        if state == AppState.LOADING and confidence > 0.5:
            return True
        if confidence < 0.5:
            analysis = await self.analyze_screen(
                image_path=image_path, image_b64=image_b64,
            )
            return analysis.loading_indicators
        return False

    async def wait_for_state(
        self,
        target_state: AppState,
        capture_fn: Callable[[], str],
        timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """
        Poll the phone screen until a target state is reached.

        Args:
            target_state: The AppState to wait for.
            capture_fn: A callable that returns the path to a fresh screenshot.
            timeout: Maximum wait time in seconds.
            poll_interval: Seconds between each poll.

        Returns:
            True if the target state was reached, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            screenshot = capture_fn()
            state, confidence, _ = await self.detect_state(image_path=screenshot)
            if state == target_state and confidence > 0.5:
                logger.info("Target state %s reached", target_state.value)
                return True
            await asyncio.sleep(poll_interval)
        logger.warning("Timed out waiting for state %s after %.0fs",
                        target_state.value, timeout)
        return False

    # -- sync wrappers --------------------------------------------------------

    def analyze_screen_sync(self, **kwargs) -> ScreenAnalysis:
        """Synchronous wrapper for analyze_screen."""
        return self._run_sync(self.analyze_screen(**kwargs))

    def find_element_sync(self, description: str, **kwargs) -> Optional[ElementLocation]:
        """Synchronous wrapper for find_element."""
        return self._run_sync(self.find_element(description, **kwargs))

    def detect_state_sync(self, **kwargs) -> Tuple[AppState, float, str]:
        """Synchronous wrapper for detect_state."""
        return self._run_sync(self.detect_state(**kwargs))

    def detect_errors_sync(self, **kwargs) -> ErrorDetection:
        """Synchronous wrapper for detect_errors."""
        return self._run_sync(self.detect_errors(**kwargs))

    def compare_screenshots_sync(self, **kwargs) -> ComparisonResult:
        """Synchronous wrapper for compare_screenshots."""
        return self._run_sync(self.compare_screenshots(**kwargs))

    def read_text_region_sync(self, x: int, y: int, w: int, h: int, **kwargs) -> str:
        """Synchronous wrapper for read_text_region."""
        return self._run_sync(self.read_text_region(x, y, w, h, **kwargs))

    @staticmethod
    def _run_sync(coro):
        """Run an async coroutine in a sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context -- create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    # -- context manager ------------------------------------------------------

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_analyze(image_path: str, base_url: str = "http://localhost:8002") -> ScreenAnalysis:
    """One-shot synchronous phone screen analysis."""
    agent = VisionAgent(base_url=base_url)
    try:
        return agent.analyze_screen_sync(image_path=image_path)
    finally:
        asyncio.run(agent.close())


def quick_find(
    description: str, image_path: str, base_url: str = "http://localhost:8002"
) -> Optional[ElementLocation]:
    """One-shot synchronous element search."""
    agent = VisionAgent(base_url=base_url)
    try:
        return agent.find_element_sync(description, image_path=image_path)
    finally:
        asyncio.run(agent.close())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Vision Agent -- phone screenshot analysis")
    parser.add_argument("image", help="Path to phone screenshot")
    parser.add_argument("--url", default="http://localhost:8002", help="Vision service URL")
    parser.add_argument("--find", help="Find a UI element by description")
    parser.add_argument("--state", action="store_true", help="Detect app state only")
    parser.add_argument("--errors", action="store_true", help="Detect errors only")
    parser.add_argument("--compare", help="Path to second screenshot for comparison")
    args = parser.parse_args()

    agent = VisionAgent(base_url=args.url)

    if args.find:
        elem = agent.find_element_sync(args.find, image_path=args.image)
        if elem:
            print(json.dumps({
                "found": True,
                "x": elem.x, "y": elem.y,
                "width": elem.width, "height": elem.height,
                "center": elem.center,
                "confidence": elem.confidence,
                "text": elem.text,
            }, indent=2))
        else:
            print(json.dumps({"found": False}))

    elif args.state:
        state, conf, details = agent.detect_state_sync(image_path=args.image)
        print(json.dumps({
            "state": state.value,
            "confidence": conf,
            "details": details,
        }, indent=2))

    elif args.errors:
        err = agent.detect_errors_sync(image_path=args.image)
        print(json.dumps({
            "has_errors": err.has_errors,
            "error_type": err.error_type,
            "error_message": err.error_message,
            "dismissable": err.dismissable,
        }, indent=2))

    elif args.compare:
        result = agent.compare_screenshots_sync(
            before_path=args.image, after_path=args.compare,
        )
        print(json.dumps({
            "changed": result.changed,
            "changes": result.changes,
            "before_state": result.before_state,
            "after_state": result.after_state,
            "progress_detected": result.progress_detected,
            "error_appeared": result.error_appeared,
        }, indent=2))

    else:
        analysis = agent.analyze_screen_sync(image_path=args.image)
        print(json.dumps({
            "app": analysis.current_app,
            "screen": analysis.current_screen,
            "visible_text": analysis.visible_text,
            "keyboard_visible": analysis.keyboard_visible,
            "loading": analysis.loading_indicators,
            "errors": analysis.errors_detected,
            "elements_count": len(analysis.tappable_elements),
            "quality_score": analysis.quality_score,
            "analysis_time_ms": round(analysis.analysis_time_ms, 1),
        }, indent=2))

    asyncio.run(agent.close())
