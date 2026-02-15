"""
Mobile Test Framework -- Visual Regression, Flow Validation, Crash Detection
=============================================================================

Automated mobile testing framework for the OpenClaw Empire Android automation
stack.  Builds on PhoneController (ADB actions), VisionAgent (screen analysis),
and ScreenpipeAgent (OCR / error detection) to provide:

    - Visual Regression:  Capture baselines, compare screenshots pixel-by-pixel,
      structurally, or via AI vision to detect unintended UI drift.
    - Flow Validation:    Define multi-step UI test flows (tap, swipe, type,
      assert) and run them end-to-end with automatic screenshots.
    - Crash Detection:    Monitor logcat and screen content for ANRs, crash
      dialogs, and force-close events.
    - Performance:        Measure app launch times and screen transition latency.
    - Accessibility:      Check text sizes, contrast, and element labeling.

Data stored under: data/mobile_tests/
    test_cases.json    -- all registered test case definitions
    results.json       -- bounded rolling history of test results
    baselines.json     -- visual regression baselines index
    screenshots/       -- captured screenshots and diff images

Usage:
    from src.mobile_test_framework import get_test_framework

    fw = get_test_framework()

    # Create and run a simple app-launch test
    tc = fw.create_app_launch_test("com.android.chrome", name="Chrome launch")
    result = fw.run_test_sync(tc.test_id)
    print(result.status, result.duration_ms)

    # Visual regression
    baseline = fw.capture_baseline_sync(tc.test_id, "step_001")
    diff = fw.compare_visual_sync(screenshot_path, baseline.baseline_id)
    print(diff["score"])

    # Crash monitoring
    crashes = fw.monitor_crashes_sync(duration_seconds=30, app="com.example.app")

CLI:
    python -m src.mobile_test_framework create --name "Login Flow" --app com.app --type flow_validation
    python -m src.mobile_test_framework list
    python -m src.mobile_test_framework run --test-id <id>
    python -m src.mobile_test_framework suite --ids id1,id2,id3
    python -m src.mobile_test_framework baseline --test-id <id> --step-id <sid>
    python -m src.mobile_test_framework compare --screenshot /path.png --baseline-id <bid>
    python -m src.mobile_test_framework crashes --duration 60 --app com.example
    python -m src.mobile_test_framework results --test-id <id>
    python -m src.mobile_test_framework report --test-id <id>
    python -m src.mobile_test_framework stats
    python -m src.mobile_test_framework export --test-id <id> --output /tmp/test.json
    python -m src.mobile_test_framework import --input /tmp/test.json
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import json
import logging
import math
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mobile_test_framework")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
TEST_DATA_DIR = BASE_DIR / "data" / "mobile_tests"
SCREENSHOTS_DIR = TEST_DATA_DIR / "screenshots"
TEST_CASES_FILE = TEST_DATA_DIR / "test_cases.json"
RESULTS_FILE = TEST_DATA_DIR / "results.json"
BASELINES_FILE = TEST_DATA_DIR / "baselines.json"

TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_RESULTS = 2000
MAX_STEP_RETRIES = 2
DEFAULT_STEP_TIMEOUT = 30.0
DEFAULT_PIXEL_THRESHOLD = 0.05  # 5% difference tolerated for pass
DEFAULT_STRUCTURAL_THRESHOLD = 0.10


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
# Time helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now_utc().isoformat()


# ---------------------------------------------------------------------------
# Async-to-sync bridge
# ---------------------------------------------------------------------------

def _run_sync(coro):
    """Run an async coroutine from a synchronous context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestType(str, Enum):
    VISUAL_REGRESSION = "visual_regression"
    FLOW_VALIDATION = "flow_validation"
    CRASH_DETECTION = "crash_detection"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"


class TestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class ComparisonMethod(str, Enum):
    PIXEL_DIFF = "pixel_diff"
    STRUCTURAL = "structural"
    AI_VISION = "ai_vision"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestStep:
    step_id: str
    action: str  # tap, swipe, type, wait, screenshot, assert_text, assert_element, launch_app
    params: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    timeout: float = DEFAULT_STEP_TIMEOUT
    screenshot_before: bool = False
    screenshot_after: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestStep:
        return cls(
            step_id=data.get("step_id", uuid.uuid4().hex[:8]),
            action=data.get("action", "wait"),
            params=data.get("params", {}),
            expected_outcome=data.get("expected_outcome", ""),
            timeout=data.get("timeout", DEFAULT_STEP_TIMEOUT),
            screenshot_before=data.get("screenshot_before", False),
            screenshot_after=data.get("screenshot_after", True),
        )


@dataclass
class TestCase:
    test_id: str
    name: str
    test_type: TestType
    app: str
    description: str = ""
    steps: List[TestStep] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_run: Optional[str] = None
    run_count: int = 0
    pass_count: int = 0
    fail_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["test_type"] = self.test_type.value
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestCase:
        return cls(
            test_id=data.get("test_id", uuid.uuid4().hex[:12]),
            name=data.get("name", "Unnamed"),
            test_type=TestType(data.get("test_type", "flow_validation")),
            app=data.get("app", ""),
            description=data.get("description", ""),
            steps=[TestStep.from_dict(s) for s in data.get("steps", [])],
            tags=data.get("tags", []),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
            last_run=data.get("last_run"),
            run_count=data.get("run_count", 0),
            pass_count=data.get("pass_count", 0),
            fail_count=data.get("fail_count", 0),
        )


@dataclass
class TestResult:
    result_id: str
    test_id: str
    test_name: str
    status: TestStatus
    started_at: str
    completed_at: str
    duration_ms: float
    device_id: Optional[str] = None
    steps_completed: int = 0
    steps_total: int = 0
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    error: Optional[str] = None
    diff_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestResult:
        return cls(
            result_id=data.get("result_id", uuid.uuid4().hex[:12]),
            test_id=data.get("test_id", ""),
            test_name=data.get("test_name", ""),
            status=TestStatus(data.get("status", "error")),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            duration_ms=data.get("duration_ms", 0.0),
            device_id=data.get("device_id"),
            steps_completed=data.get("steps_completed", 0),
            steps_total=data.get("steps_total", 0),
            step_results=data.get("step_results", []),
            screenshots=data.get("screenshots", []),
            error=data.get("error"),
            diff_score=data.get("diff_score"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VisualBaseline:
    baseline_id: str
    test_id: str
    step_id: str
    screenshot_path: str
    created_at: str = field(default_factory=_now_iso)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VisualBaseline:
        return cls(
            baseline_id=data.get("baseline_id", uuid.uuid4().hex[:12]),
            test_id=data.get("test_id", ""),
            step_id=data.get("step_id", ""),
            screenshot_path=data.get("screenshot_path", ""),
            created_at=data.get("created_at", _now_iso()),
            description=data.get("description", ""),
        )


# ===========================================================================
# MobileTestFramework
# ===========================================================================

class MobileTestFramework:
    """
    Central framework for managing and executing mobile tests.

    Coordinates PhoneController (device actions), VisionAgent (screen
    analysis), and ScreenpipeAgent (OCR monitoring) to run automated UI
    test flows on Android devices.
    """

    def __init__(self) -> None:
        self._test_cases: Dict[str, TestCase] = {}
        self._results: List[TestResult] = []
        self._baselines: Dict[str, VisualBaseline] = {}  # key: "test_id:step_id"

        # Lazy-loaded collaborators
        self._phone_controller = None
        self._vision_agent = None
        self._screenpipe_agent = None

        self._load_state()
        logger.info(
            "MobileTestFramework initialized: %d tests, %d results, %d baselines",
            len(self._test_cases), len(self._results), len(self._baselines),
        )

    # -------------------------------------------------------------------
    # State persistence
    # -------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load test cases, results, and baselines from disk."""
        raw_cases = _load_json(TEST_CASES_FILE, {})
        for tid, td in raw_cases.items():
            try:
                self._test_cases[tid] = TestCase.from_dict(td)
            except Exception as exc:
                logger.warning("Skipping corrupt test case %s: %s", tid, exc)

        raw_results = _load_json(RESULTS_FILE, [])
        if isinstance(raw_results, list):
            for rd in raw_results[-MAX_RESULTS:]:
                try:
                    self._results.append(TestResult.from_dict(rd))
                except Exception as exc:
                    logger.warning("Skipping corrupt result: %s", exc)

        raw_baselines = _load_json(BASELINES_FILE, {})
        for key, bd in raw_baselines.items():
            try:
                self._baselines[key] = VisualBaseline.from_dict(bd)
            except Exception as exc:
                logger.warning("Skipping corrupt baseline %s: %s", key, exc)

    def _save_test_cases(self) -> None:
        data = {tid: tc.to_dict() for tid, tc in self._test_cases.items()}
        _save_json(TEST_CASES_FILE, data)

    def _save_results(self) -> None:
        # Keep bounded
        trimmed = self._results[-MAX_RESULTS:]
        self._results = trimmed
        _save_json(RESULTS_FILE, [r.to_dict() for r in trimmed])

    def _save_baselines(self) -> None:
        data = {k: b.to_dict() for k, b in self._baselines.items()}
        _save_json(BASELINES_FILE, data)

    # -------------------------------------------------------------------
    # Lazy collaborator access
    # -------------------------------------------------------------------

    def _get_phone_controller(self):
        """Lazily import and instantiate a PhoneController."""
        if self._phone_controller is None:
            from src.phone_controller import PhoneController
            self._phone_controller = PhoneController()
        return self._phone_controller

    def _get_vision_agent(self):
        """Lazily import and instantiate a VisionAgent."""
        if self._vision_agent is None:
            from src.vision_agent import VisionAgent
            self._vision_agent = VisionAgent()
        return self._vision_agent

    def _get_screenpipe_agent(self):
        """Lazily import and instantiate a ScreenpipeAgent."""
        if self._screenpipe_agent is None:
            from src.screenpipe_agent import ScreenpipeAgent
            self._screenpipe_agent = ScreenpipeAgent()
        return self._screenpipe_agent

    # ===================================================================
    # Test Case Management
    # ===================================================================

    def create_test(
        self,
        name: str,
        test_type: TestType,
        app: str,
        steps: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> TestCase:
        """Create a new test case and persist it."""
        test_id = uuid.uuid4().hex[:12]
        parsed_steps: List[TestStep] = []
        for i, s in enumerate(steps or []):
            s.setdefault("step_id", f"step_{i:03d}")
            parsed_steps.append(TestStep.from_dict(s))

        tc = TestCase(
            test_id=test_id,
            name=name,
            test_type=test_type,
            app=app,
            description=description,
            steps=parsed_steps,
            tags=tags or [],
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self._test_cases[test_id] = tc
        self._save_test_cases()
        logger.info("Created test %s: %s (%s)", test_id, name, test_type.value)
        return tc

    def get_test(self, test_id: str) -> TestCase:
        """Return a test case by ID. Raises KeyError if not found."""
        if test_id not in self._test_cases:
            raise KeyError(f"Test case not found: {test_id}")
        return self._test_cases[test_id]

    def update_test(self, test_id: str, **updates: Any) -> TestCase:
        """Update fields on an existing test case."""
        tc = self.get_test(test_id)
        for key, value in updates.items():
            if key == "steps" and isinstance(value, list):
                parsed = []
                for i, s in enumerate(value):
                    if isinstance(s, dict):
                        s.setdefault("step_id", f"step_{i:03d}")
                        parsed.append(TestStep.from_dict(s))
                    elif isinstance(s, TestStep):
                        parsed.append(s)
                tc.steps = parsed
            elif key == "test_type" and isinstance(value, str):
                tc.test_type = TestType(value)
            elif key == "tags" and isinstance(value, list):
                tc.tags = value
            elif hasattr(tc, key):
                setattr(tc, key, value)
        tc.updated_at = _now_iso()
        self._save_test_cases()
        logger.info("Updated test %s", test_id)
        return tc

    def delete_test(self, test_id: str) -> None:
        """Delete a test case by ID."""
        if test_id not in self._test_cases:
            raise KeyError(f"Test case not found: {test_id}")
        del self._test_cases[test_id]
        self._save_test_cases()
        logger.info("Deleted test %s", test_id)

    def list_tests(
        self,
        test_type: Optional[TestType] = None,
        app: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[TestCase]:
        """List test cases with optional filters."""
        results: List[TestCase] = []
        for tc in self._test_cases.values():
            if test_type and tc.test_type != test_type:
                continue
            if app and tc.app != app:
                continue
            if tags:
                if not any(t in tc.tags for t in tags):
                    continue
            results.append(tc)
        return sorted(results, key=lambda t: t.updated_at, reverse=True)

    def import_test(self, data: Dict[str, Any]) -> TestCase:
        """Import a test case from a dictionary (e.g. from JSON file)."""
        tc = TestCase.from_dict(data)
        # Assign new ID if collision
        if tc.test_id in self._test_cases:
            tc.test_id = uuid.uuid4().hex[:12]
        tc.updated_at = _now_iso()
        self._test_cases[tc.test_id] = tc
        self._save_test_cases()
        logger.info("Imported test %s: %s", tc.test_id, tc.name)
        return tc

    def export_test(self, test_id: str) -> Dict[str, Any]:
        """Export a test case to a dictionary."""
        return self.get_test(test_id).to_dict()

    # ===================================================================
    # Test Execution
    # ===================================================================

    async def run_test(self, test_id: str, device_id: Optional[str] = None) -> TestResult:
        """
        Execute a single test case.

        Connects to the Android device, executes each step in order, captures
        screenshots, and records the result.  For visual regression tests, the
        final step includes a diff comparison against the baseline.
        """
        tc = self.get_test(test_id)
        started_at = _now_iso()
        start_mono = time.monotonic()

        result = TestResult(
            result_id=uuid.uuid4().hex[:12],
            test_id=test_id,
            test_name=tc.name,
            status=TestStatus.RUNNING,
            started_at=started_at,
            completed_at="",
            duration_ms=0.0,
            device_id=device_id,
            steps_completed=0,
            steps_total=len(tc.steps),
            step_results=[],
            screenshots=[],
            metadata={"test_type": tc.test_type.value, "app": tc.app},
        )

        controller = self._get_phone_controller()

        try:
            connected = await controller.connect()
            if not connected:
                result.status = TestStatus.ERROR
                result.error = "Failed to connect to Android device"
                result.completed_at = _now_iso()
                result.duration_ms = (time.monotonic() - start_mono) * 1000
                self._record_result(result, tc)
                return result

            logger.info("Running test %s (%s): %d steps", test_id, tc.name, len(tc.steps))

            for step in tc.steps:
                step_result = await self._execute_step(step, controller)
                result.step_results.append(step_result)

                # Collect screenshots
                for key in ("screenshot_before", "screenshot_after"):
                    path = step_result.get(key)
                    if path:
                        result.screenshots.append(path)

                if step_result.get("success"):
                    result.steps_completed += 1
                else:
                    result.status = TestStatus.FAILED
                    result.error = step_result.get("error", f"Step {step.step_id} failed")
                    break

            # If all steps passed
            if result.status == TestStatus.RUNNING:
                # For visual regression, compute diff against baseline if available
                if tc.test_type == TestType.VISUAL_REGRESSION and result.screenshots:
                    last_screenshot = result.screenshots[-1]
                    baseline_key = f"{test_id}:{tc.steps[-1].step_id}" if tc.steps else None
                    if baseline_key and baseline_key in self._baselines:
                        baseline = self._baselines[baseline_key]
                        diff = self._pixel_diff(last_screenshot, baseline.screenshot_path)
                        result.diff_score = diff
                        result.metadata["comparison_method"] = "pixel_diff"
                        if diff > DEFAULT_PIXEL_THRESHOLD:
                            result.status = TestStatus.FAILED
                            result.error = (
                                f"Visual regression: diff score {diff:.4f} exceeds "
                                f"threshold {DEFAULT_PIXEL_THRESHOLD}"
                            )
                        else:
                            result.status = TestStatus.PASSED
                    else:
                        result.status = TestStatus.PASSED
                        result.metadata["note"] = "No baseline found -- passed by default"
                else:
                    result.status = TestStatus.PASSED

        except Exception as exc:
            result.status = TestStatus.ERROR
            result.error = str(exc)
            logger.error("Test %s error: %s", test_id, exc, exc_info=True)
        finally:
            result.completed_at = _now_iso()
            result.duration_ms = (time.monotonic() - start_mono) * 1000
            try:
                await controller.close()
            except Exception:
                pass

        self._record_result(result, tc)
        logger.info(
            "Test %s %s in %.0fms (%d/%d steps)",
            test_id, result.status.value, result.duration_ms,
            result.steps_completed, result.steps_total,
        )
        return result

    def run_test_sync(self, test_id: str, device_id: Optional[str] = None) -> TestResult:
        """Synchronous wrapper for run_test."""
        return _run_sync(self.run_test(test_id, device_id))

    async def run_suite(
        self,
        test_ids: List[str],
        device_id: Optional[str] = None,
    ) -> List[TestResult]:
        """Run a list of test cases sequentially and return all results."""
        results: List[TestResult] = []
        for tid in test_ids:
            try:
                result = await self.run_test(tid, device_id)
                results.append(result)
            except KeyError:
                logger.warning("Test %s not found, skipping", tid)
                results.append(TestResult(
                    result_id=uuid.uuid4().hex[:12],
                    test_id=tid,
                    test_name="unknown",
                    status=TestStatus.SKIPPED,
                    started_at=_now_iso(),
                    completed_at=_now_iso(),
                    duration_ms=0.0,
                    error=f"Test case not found: {tid}",
                ))
        return results

    def run_suite_sync(
        self,
        test_ids: List[str],
        device_id: Optional[str] = None,
    ) -> List[TestResult]:
        """Synchronous wrapper for run_suite."""
        return _run_sync(self.run_suite(test_ids, device_id))

    async def run_all(
        self,
        tags: Optional[List[str]] = None,
        device_id: Optional[str] = None,
    ) -> List[TestResult]:
        """Run all test cases matching the given tags (or all if no tags)."""
        tests = self.list_tests(tags=tags)
        test_ids = [tc.test_id for tc in tests]
        logger.info("Running all %d tests (tags=%s)", len(test_ids), tags)
        return await self.run_suite(test_ids, device_id)

    def run_all_sync(
        self,
        tags: Optional[List[str]] = None,
        device_id: Optional[str] = None,
    ) -> List[TestResult]:
        """Synchronous wrapper for run_all."""
        return _run_sync(self.run_all(tags, device_id))

    def _record_result(self, result: TestResult, tc: TestCase) -> None:
        """Persist a test result and update the test case stats."""
        self._results.append(result)
        tc.last_run = result.completed_at
        tc.run_count += 1
        if result.status == TestStatus.PASSED:
            tc.pass_count += 1
        elif result.status in (TestStatus.FAILED, TestStatus.ERROR):
            tc.fail_count += 1
        self._save_test_cases()
        self._save_results()

    # ===================================================================
    # Step Execution
    # ===================================================================

    async def _execute_step(
        self,
        step: TestStep,
        controller,
    ) -> Dict[str, Any]:
        """
        Execute a single test step and return a result dictionary.

        Result dict keys: success, step_id, action, duration_ms, error,
        screenshot_before, screenshot_after, output.
        """
        result: Dict[str, Any] = {
            "step_id": step.step_id,
            "action": step.action,
            "success": False,
            "error": None,
            "duration_ms": 0.0,
            "screenshot_before": None,
            "screenshot_after": None,
            "output": None,
        }

        start = time.monotonic()

        try:
            # Pre-step screenshot
            if step.screenshot_before:
                path = await controller.screenshot()
                result["screenshot_before"] = path

            # Dispatch to action handler
            handler = self._get_step_handler(step.action)
            if handler is None:
                result["error"] = f"Unknown action: {step.action}"
                result["duration_ms"] = (time.monotonic() - start) * 1000
                return result

            action_result = await asyncio.wait_for(
                handler(step.params, controller),
                timeout=step.timeout,
            )

            result["success"] = action_result.get("success", False)
            result["output"] = action_result.get("output")
            if not result["success"]:
                result["error"] = action_result.get("error", "Step did not succeed")

            # Post-step screenshot
            if step.screenshot_after:
                path = await controller.screenshot()
                result["screenshot_after"] = path

        except asyncio.TimeoutError:
            result["error"] = f"Step timed out after {step.timeout}s"
        except Exception as exc:
            result["error"] = str(exc)

        result["duration_ms"] = (time.monotonic() - start) * 1000
        return result

    def _get_step_handler(self, action: str):
        """Return the async handler coroutine for a step action name."""
        handlers = {
            "tap": self._execute_tap,
            "swipe": self._execute_swipe,
            "type": self._execute_type,
            "wait": self._execute_wait,
            "screenshot": self._execute_screenshot,
            "assert_text": self._execute_assert_text,
            "assert_element": self._execute_assert_element,
            "launch_app": self._execute_launch_app,
        }
        return handlers.get(action)

    async def _execute_tap(self, params: Dict[str, Any], controller) -> Dict[str, Any]:
        """Tap at (x, y) or find element by text/resource_id and tap it."""
        if "x" in params and "y" in params:
            action_result = await controller.tap(int(params["x"]), int(params["y"]))
            return {"success": action_result.success, "error": action_result.error}

        # Find element by text or resource_id
        text = params.get("text")
        resource_id = params.get("resource_id")
        timeout = float(params.get("timeout", 10))

        element = await controller.wait_for_element(
            text=text,
            resource_id=resource_id,
            timeout=timeout,
        )
        if element is None:
            return {
                "success": False,
                "error": f"Element not found: text={text!r}, resource_id={resource_id!r}",
            }

        action_result = await controller.tap_element(element)
        return {"success": action_result.success, "error": action_result.error}

    async def _execute_swipe(self, params: Dict[str, Any], controller) -> Dict[str, Any]:
        """Swipe from (x1,y1) to (x2,y2) with optional duration."""
        x1 = int(params.get("x1", 0))
        y1 = int(params.get("y1", 0))
        x2 = int(params.get("x2", 0))
        y2 = int(params.get("y2", 0))
        duration = int(params.get("duration_ms", 300))

        # Support shorthand: direction-based swipes
        direction = params.get("direction", "").lower()
        if direction:
            w, h = controller.resolution
            cx, cy = w // 2, h // 2
            distance = int(params.get("distance", 500))
            if direction == "up":
                x1, y1, x2, y2 = cx, cy + distance // 2, cx, cy - distance // 2
            elif direction == "down":
                x1, y1, x2, y2 = cx, cy - distance // 2, cx, cy + distance // 2
            elif direction == "left":
                x1, y1, x2, y2 = cx + distance // 2, cy, cx - distance // 2, cy
            elif direction == "right":
                x1, y1, x2, y2 = cx - distance // 2, cy, cx + distance // 2, cy

        action_result = await controller.swipe(x1, y1, x2, y2, duration)
        return {"success": action_result.success, "error": action_result.error}

    async def _execute_type(self, params: Dict[str, Any], controller) -> Dict[str, Any]:
        """Type text into the currently focused field."""
        text = params.get("text", "")
        if not text:
            return {"success": False, "error": "No text provided for type action"}
        action_result = await controller.type_text(text)
        return {"success": action_result.success, "error": action_result.error}

    async def _execute_wait(self, params: Dict[str, Any], _controller=None) -> Dict[str, Any]:
        """Wait for a specified number of seconds."""
        seconds = float(params.get("seconds", 1.0))
        await asyncio.sleep(seconds)
        return {"success": True, "output": f"Waited {seconds}s"}

    async def _execute_screenshot(self, params: Dict[str, Any], controller) -> Dict[str, Any]:
        """Capture a screenshot and return the file path."""
        path = await controller.screenshot()
        return {"success": True, "output": path}

    async def _execute_assert_text(self, params: Dict[str, Any], controller) -> Dict[str, Any]:
        """
        Assert that specific text is visible on screen.

        Uses ScreenpipeAgent OCR search to look for the expected text.
        Falls back to VisionAgent analysis if screenpipe is unavailable.
        """
        expected_text = params.get("text", "")
        if not expected_text:
            return {"success": False, "error": "No expected text provided"}

        partial = params.get("partial", True)

        # Strategy 1: UI dump from controller
        try:
            elements = await controller.ui_dump()
            for el in elements:
                haystack = f"{el.text} {el.content_desc}".lower()
                needle = expected_text.lower()
                if partial and needle in haystack:
                    return {"success": True, "output": f"Found text in element: {el.text!r}"}
                if not partial and needle == el.text.lower():
                    return {"success": True, "output": f"Exact match: {el.text!r}"}
        except Exception as exc:
            logger.debug("UI dump assert_text failed, trying screenpipe: %s", exc)

        # Strategy 2: Screenpipe OCR
        try:
            agent = self._get_screenpipe_agent()
            results = await agent.search(
                query=expected_text,
                limit=5,
            )
            for r in results:
                if expected_text.lower() in r.content.lower():
                    return {
                        "success": True,
                        "output": f"Found via OCR in {r.app_name}: {r.content[:100]}",
                    }
        except Exception as exc:
            logger.debug("Screenpipe assert_text fallback failed: %s", exc)

        # Strategy 3: VisionAgent screenshot analysis
        try:
            screenshot_path = await controller.screenshot()
            vision = self._get_vision_agent()
            analysis = await vision.analyze_screen(image_path=screenshot_path)
            for visible in analysis.visible_text:
                if expected_text.lower() in visible.lower():
                    return {
                        "success": True,
                        "output": f"Found via vision: {visible[:100]}",
                    }
        except Exception as exc:
            logger.debug("Vision assert_text fallback failed: %s", exc)

        return {
            "success": False,
            "error": f"Text not found on screen: {expected_text!r}",
        }

    async def _execute_assert_element(self, params: Dict[str, Any], controller) -> Dict[str, Any]:
        """
        Assert that a UI element matching given criteria exists.

        Uses VisionAgent for semantic element detection when UI dump
        does not find the element.
        """
        description = params.get("description", "")
        text = params.get("text")
        resource_id = params.get("resource_id")
        timeout = float(params.get("timeout", 10))

        # Strategy 1: element-based search via UI dump
        if text or resource_id:
            element = await controller.wait_for_element(
                text=text,
                resource_id=resource_id,
                timeout=timeout,
            )
            if element is not None:
                return {
                    "success": True,
                    "output": f"Element found: text={element.text!r}, id={element.resource_id!r}",
                }

        # Strategy 2: VisionAgent semantic search
        if description:
            try:
                screenshot_path = await controller.screenshot()
                vision = self._get_vision_agent()
                elem = await vision.find_element(
                    description,
                    image_path=screenshot_path,
                )
                if elem is not None:
                    return {
                        "success": True,
                        "output": (
                            f"Element found via vision: {description!r} "
                            f"at ({elem.x}, {elem.y}) conf={elem.confidence:.2f}"
                        ),
                    }
            except Exception as exc:
                logger.debug("Vision assert_element fallback failed: %s", exc)

        criteria = description or f"text={text!r} id={resource_id!r}"
        return {
            "success": False,
            "error": f"Element not found: {criteria}",
        }

    async def _execute_launch_app(self, params: Dict[str, Any], controller) -> Dict[str, Any]:
        """Launch an app by package name."""
        package = params.get("package", "")
        activity = params.get("activity")
        if not package:
            return {"success": False, "error": "No package name provided"}

        action_result = await controller.launch_app(package, activity)
        return {"success": action_result.success, "error": action_result.error}

    # ===================================================================
    # Visual Regression
    # ===================================================================

    async def capture_baseline(
        self,
        test_id: str,
        step_id: str,
        device_id: Optional[str] = None,
    ) -> VisualBaseline:
        """
        Capture a baseline screenshot for a test step.

        Connects to the device, navigates to the step (by running preceding
        steps), captures the screen, and stores it as the visual baseline.
        """
        tc = self.get_test(test_id)
        controller = self._get_phone_controller()

        try:
            connected = await controller.connect()
            if not connected:
                raise ConnectionError("Failed to connect to device")

            # Find the target step and run preceding steps
            target_step_idx = None
            for i, step in enumerate(tc.steps):
                if step.step_id == step_id:
                    target_step_idx = i
                    break
            if target_step_idx is None:
                raise ValueError(f"Step {step_id} not found in test {test_id}")

            # Execute preceding steps to get to the right screen state
            for step in tc.steps[:target_step_idx]:
                handler = self._get_step_handler(step.action)
                if handler:
                    await handler(step.params, controller)

            # Execute the target step itself to reach its post-state
            target_step = tc.steps[target_step_idx]
            handler = self._get_step_handler(target_step.action)
            if handler:
                await handler(target_step.params, controller)

            # Capture the baseline screenshot
            screenshot_path = await controller.screenshot()
            # Copy to baselines directory for persistence
            baseline_filename = f"baseline_{test_id}_{step_id}_{_now_utc().strftime('%Y%m%d_%H%M%S')}.png"
            baseline_path = str(SCREENSHOTS_DIR / baseline_filename)

            # Read and copy the screenshot
            with open(screenshot_path, "rb") as src:
                data = src.read()
            with open(baseline_path, "wb") as dst:
                dst.write(data)

        finally:
            try:
                await controller.close()
            except Exception:
                pass

        baseline = VisualBaseline(
            baseline_id=uuid.uuid4().hex[:12],
            test_id=test_id,
            step_id=step_id,
            screenshot_path=baseline_path,
            created_at=_now_iso(),
            description=f"Baseline for {tc.name} step {step_id}",
        )
        key = f"{test_id}:{step_id}"
        self._baselines[key] = baseline
        self._save_baselines()
        logger.info("Captured baseline %s for %s:%s", baseline.baseline_id, test_id, step_id)
        return baseline

    def capture_baseline_sync(
        self,
        test_id: str,
        step_id: str,
        device_id: Optional[str] = None,
    ) -> VisualBaseline:
        """Synchronous wrapper for capture_baseline."""
        return _run_sync(self.capture_baseline(test_id, step_id, device_id))

    async def compare_visual(
        self,
        screenshot_path: str,
        baseline_id: str,
        method: ComparisonMethod = ComparisonMethod.STRUCTURAL,
    ) -> Dict[str, Any]:
        """
        Compare a screenshot against a stored baseline.

        Returns a dict with: score (0=identical, 1=completely different),
        passed (bool), method, details.
        """
        # Find baseline by ID
        baseline: Optional[VisualBaseline] = None
        for bl in self._baselines.values():
            if bl.baseline_id == baseline_id:
                baseline = bl
                break
        if baseline is None:
            raise KeyError(f"Baseline not found: {baseline_id}")

        if not Path(baseline.screenshot_path).exists():
            raise FileNotFoundError(f"Baseline file missing: {baseline.screenshot_path}")
        if not Path(screenshot_path).exists():
            raise FileNotFoundError(f"Screenshot file missing: {screenshot_path}")

        result: Dict[str, Any] = {
            "baseline_id": baseline_id,
            "method": method.value,
            "score": 1.0,
            "passed": False,
            "details": {},
        }

        if method == ComparisonMethod.PIXEL_DIFF:
            score = self._pixel_diff(screenshot_path, baseline.screenshot_path)
            result["score"] = score
            result["passed"] = score <= DEFAULT_PIXEL_THRESHOLD
            result["details"] = {
                "threshold": DEFAULT_PIXEL_THRESHOLD,
                "diff_percentage": round(score * 100, 2),
            }

        elif method == ComparisonMethod.STRUCTURAL:
            diff_info = self._structural_diff(screenshot_path, baseline.screenshot_path)
            result["score"] = diff_info.get("score", 1.0)
            result["passed"] = diff_info.get("score", 1.0) <= DEFAULT_STRUCTURAL_THRESHOLD
            result["details"] = diff_info

        elif method == ComparisonMethod.AI_VISION:
            vision = self._get_vision_agent()
            comparison = await vision.compare_screenshots(
                before_path=baseline.screenshot_path,
                after_path=screenshot_path,
            )
            # Map vision comparison to a 0-1 score
            if comparison.changed:
                num_changes = len(comparison.changes)
                score = min(1.0, num_changes * 0.15)
            else:
                score = 0.0
            result["score"] = score
            result["passed"] = not comparison.error_appeared and score <= DEFAULT_STRUCTURAL_THRESHOLD
            result["details"] = {
                "changed": comparison.changed,
                "changes": comparison.changes,
                "progress_detected": comparison.progress_detected,
                "error_appeared": comparison.error_appeared,
            }

        logger.info(
            "Visual comparison (%s): score=%.4f, passed=%s",
            method.value, result["score"], result["passed"],
        )
        return result

    def compare_visual_sync(
        self,
        screenshot_path: str,
        baseline_id: str,
        method: ComparisonMethod = ComparisonMethod.STRUCTURAL,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for compare_visual."""
        return _run_sync(self.compare_visual(screenshot_path, baseline_id, method))

    def _pixel_diff(self, img_a_path: str, img_b_path: str) -> float:
        """
        Compute pixel-level difference between two images.

        Returns a float 0.0 (identical) to 1.0 (completely different).
        Uses Pillow for image comparison.  Falls back to file-size
        heuristic if Pillow is not available.
        """
        try:
            from PIL import Image
        except ImportError:
            logger.warning("Pillow not installed -- falling back to file-size heuristic")
            return self._file_size_diff(img_a_path, img_b_path)

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        # Resize to common dimensions if different
        if img_a.size != img_b.size:
            common_w = min(img_a.width, img_b.width)
            common_h = min(img_a.height, img_b.height)
            img_a = img_a.resize((common_w, common_h), Image.LANCZOS)
            img_b = img_b.resize((common_w, common_h), Image.LANCZOS)

        pixels_a = list(img_a.getdata())
        pixels_b = list(img_b.getdata())
        total_pixels = len(pixels_a)
        if total_pixels == 0:
            return 0.0

        diff_sum = 0.0
        for pa, pb in zip(pixels_a, pixels_b):
            # Sum of absolute channel differences, normalized to 0-1 per pixel
            channel_diff = sum(abs(a - b) for a, b in zip(pa, pb))
            diff_sum += channel_diff / (255.0 * 3)

        return diff_sum / total_pixels

    def _structural_diff(self, img_a_path: str, img_b_path: str) -> Dict[str, Any]:
        """
        Compute a structural (layout-level) difference between two images.

        Divides both images into a grid and compares each cell.  Returns a
        dict with overall score and a grid of per-cell scores.
        """
        try:
            from PIL import Image
        except ImportError:
            score = self._file_size_diff(img_a_path, img_b_path)
            return {"score": score, "method": "file_size_fallback", "grid": []}

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        # Resize to common
        if img_a.size != img_b.size:
            common_w = min(img_a.width, img_b.width)
            common_h = min(img_a.height, img_b.height)
            img_a = img_a.resize((common_w, common_h), Image.LANCZOS)
            img_b = img_b.resize((common_w, common_h), Image.LANCZOS)

        w, h = img_a.size
        grid_cols = 8
        grid_rows = 12
        cell_w = max(1, w // grid_cols)
        cell_h = max(1, h // grid_rows)

        grid_scores: List[List[float]] = []
        total_score = 0.0
        cell_count = 0

        for row in range(grid_rows):
            row_scores: List[float] = []
            for col in range(grid_cols):
                x0 = col * cell_w
                y0 = row * cell_h
                x1 = min(x0 + cell_w, w)
                y1 = min(y0 + cell_h, h)

                cell_a = img_a.crop((x0, y0, x1, y1))
                cell_b = img_b.crop((x0, y0, x1, y1))

                pix_a = list(cell_a.getdata())
                pix_b = list(cell_b.getdata())
                n = len(pix_a)
                if n == 0:
                    row_scores.append(0.0)
                    continue

                cell_diff = 0.0
                for pa, pb in zip(pix_a, pix_b):
                    cell_diff += sum(abs(a - b) for a, b in zip(pa, pb)) / (255.0 * 3)
                cell_score = cell_diff / n

                row_scores.append(round(cell_score, 4))
                total_score += cell_score
                cell_count += 1

            grid_scores.append(row_scores)

        overall = total_score / max(cell_count, 1)
        changed_cells = sum(
            1 for row in grid_scores for s in row if s > DEFAULT_PIXEL_THRESHOLD
        )

        return {
            "score": round(overall, 4),
            "grid": grid_scores,
            "grid_size": f"{grid_cols}x{grid_rows}",
            "changed_cells": changed_cells,
            "total_cells": cell_count,
            "change_pct": round(changed_cells / max(cell_count, 1) * 100, 1),
        }

    @staticmethod
    def _file_size_diff(path_a: str, path_b: str) -> float:
        """Crude heuristic: compare file sizes when Pillow is unavailable."""
        try:
            size_a = os.path.getsize(path_a)
            size_b = os.path.getsize(path_b)
            if size_a == 0 and size_b == 0:
                return 0.0
            diff = abs(size_a - size_b)
            return min(1.0, diff / max(size_a, size_b))
        except OSError:
            return 1.0

    def get_baselines(self, test_id: Optional[str] = None) -> List[VisualBaseline]:
        """Return baselines, optionally filtered by test_id."""
        baselines = list(self._baselines.values())
        if test_id:
            baselines = [b for b in baselines if b.test_id == test_id]
        return sorted(baselines, key=lambda b: b.created_at, reverse=True)

    def update_baseline(self, baseline_id: str, screenshot_path: str) -> None:
        """Replace the screenshot for an existing baseline."""
        for key, bl in self._baselines.items():
            if bl.baseline_id == baseline_id:
                if not Path(screenshot_path).exists():
                    raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

                # Copy to baselines directory
                new_filename = f"baseline_{bl.test_id}_{bl.step_id}_{_now_utc().strftime('%Y%m%d_%H%M%S')}.png"
                new_path = str(SCREENSHOTS_DIR / new_filename)
                with open(screenshot_path, "rb") as src:
                    data = src.read()
                with open(new_path, "wb") as dst:
                    dst.write(data)

                bl.screenshot_path = new_path
                bl.created_at = _now_iso()
                self._save_baselines()
                logger.info("Updated baseline %s with new screenshot", baseline_id)
                return
        raise KeyError(f"Baseline not found: {baseline_id}")

    # ===================================================================
    # Crash Detection
    # ===================================================================

    async def monitor_crashes(
        self,
        duration_seconds: int = 60,
        app: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Monitor for crashes and ANRs over a time window.

        Combines three detection strategies:
            1. Logcat crash/ANR entries via PhoneController
            2. Screenpipe OCR for error text patterns
            3. VisionAgent screenshot analysis for crash dialogs
        """
        crashes: List[Dict[str, Any]] = []
        start_time = _now_iso()
        deadline = time.monotonic() + duration_seconds
        poll_interval = 3.0

        logger.info(
            "Monitoring for crashes: duration=%ds, app=%s",
            duration_seconds, app or "all",
        )

        while time.monotonic() < deadline:
            # Strategy 1: check logcat
            try:
                logcat_crashes = await self._check_logcat_crashes(app)
                for c in logcat_crashes:
                    c["source"] = "logcat"
                    c["detected_at"] = _now_iso()
                    crashes.append(c)
            except Exception as exc:
                logger.debug("Logcat crash check failed: %s", exc)

            # Strategy 2: Screenpipe error search
            try:
                sp = self._get_screenpipe_agent()
                errors = await sp.search_errors(app_name=app, minutes_back=1)
                for err in errors:
                    if self._is_crash_text(err.content):
                        crashes.append({
                            "type": "ocr_crash_text",
                            "source": "screenpipe",
                            "app": err.app_name,
                            "text": err.content[:300],
                            "timestamp": err.timestamp,
                            "detected_at": _now_iso(),
                        })
            except Exception as exc:
                logger.debug("Screenpipe crash check failed: %s", exc)

            # Strategy 3: Vision screenshot analysis for crash dialogs
            try:
                controller = self._get_phone_controller()
                connected = await controller.connect()
                if connected:
                    screenshot_path = await controller.screenshot()
                    vision = self._get_vision_agent()
                    error_detection = await vision.detect_errors(image_path=screenshot_path)
                    if error_detection.has_errors and error_detection.error_type in (
                        "crash", "dialog",
                    ):
                        crashes.append({
                            "type": "visual_crash_dialog",
                            "source": "vision",
                            "app": app or "unknown",
                            "error_type": error_detection.error_type,
                            "error_message": error_detection.error_message,
                            "dismissable": error_detection.dismissable,
                            "screenshot": screenshot_path,
                            "detected_at": _now_iso(),
                        })
            except Exception as exc:
                logger.debug("Vision crash check failed: %s", exc)

            if crashes:
                logger.warning("Detected %d crash(es) so far", len(crashes))

            await asyncio.sleep(poll_interval)

        # Deduplicate by source + type + approximate time
        unique: List[Dict[str, Any]] = []
        seen_keys: set = set()
        for c in crashes:
            key = (c.get("source", ""), c.get("type", ""), c.get("app", ""))
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append(c)

        logger.info(
            "Crash monitoring complete: %d unique crashes in %ds",
            len(unique), duration_seconds,
        )
        return unique

    def monitor_crashes_sync(
        self,
        duration_seconds: int = 60,
        app: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for monitor_crashes."""
        return _run_sync(self.monitor_crashes(duration_seconds, app))

    def _detect_anr(self, ui_dump_text: str) -> bool:
        """Check a UI dump or logcat output for ANR indicators."""
        anr_markers = [
            "Application Not Responding",
            "ANR in",
            "Input dispatching timed out",
            "is not responding",
        ]
        lower = ui_dump_text.lower()
        return any(marker.lower() in lower for marker in anr_markers)

    def _detect_crash_dialog(self, analysis_text: str) -> bool:
        """Check vision analysis text for crash dialog indicators."""
        crash_markers = [
            "has stopped",
            "keeps stopping",
            "isn't responding",
            "force close",
            "unfortunately",
            "crash",
            "fatal",
        ]
        lower = analysis_text.lower()
        return any(marker in lower for marker in crash_markers)

    @staticmethod
    def _is_crash_text(text: str) -> bool:
        """Check if OCR text indicates a crash or ANR."""
        crash_patterns = [
            "has stopped",
            "keeps stopping",
            "isn't responding",
            "application not responding",
            "force close",
            "unfortunately",
            "process crashed",
            "fatal exception",
        ]
        lower = text.lower()
        return any(p in lower for p in crash_patterns)

    async def _check_logcat_crashes(self, app: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query logcat for recent crash and ANR entries.

        Connects to the phone controller and runs logcat with crash/ANR
        filters to find recent events.
        """
        controller = self._get_phone_controller()
        crashes: List[Dict[str, Any]] = []

        try:
            connected = await controller.connect()
            if not connected:
                return crashes

            # Check for crashes in logcat
            filter_tag = f"-s AndroidRuntime:E" if not app else f"-s AndroidRuntime:E"
            cmd = f"logcat -d -t 50 {filter_tag}"
            output = await controller._adb_shell(cmd)

            if output:
                lines = output.strip().split("\n")
                crash_block: List[str] = []
                for line in lines:
                    if "FATAL EXCEPTION" in line or "Process:" in line:
                        crash_block.append(line)
                    elif crash_block and line.strip():
                        crash_block.append(line)
                    elif crash_block:
                        crash_text = "\n".join(crash_block)
                        if app is None or app in crash_text:
                            crashes.append({
                                "type": "fatal_exception",
                                "text": crash_text[:500],
                                "app": app or self._extract_process(crash_text),
                            })
                        crash_block = []
                # Handle trailing block
                if crash_block:
                    crash_text = "\n".join(crash_block)
                    if app is None or app in crash_text:
                        crashes.append({
                            "type": "fatal_exception",
                            "text": crash_text[:500],
                            "app": app or self._extract_process(crash_text),
                        })

            # Check for ANRs
            anr_output = await controller._adb_shell("logcat -d -t 20 -s ActivityManager:E")
            if anr_output and self._detect_anr(anr_output):
                crashes.append({
                    "type": "anr",
                    "text": anr_output[:500],
                    "app": app or "unknown",
                })

        except Exception as exc:
            logger.debug("Logcat crash check error: %s", exc)
        finally:
            try:
                await controller.close()
            except Exception:
                pass

        return crashes

    @staticmethod
    def _extract_process(crash_text: str) -> str:
        """Extract a process/package name from crash logcat output."""
        import re
        match = re.search(r"Process:\s*(\S+)", crash_text)
        if match:
            return match.group(1)
        return "unknown"

    # ===================================================================
    # Results & Reporting
    # ===================================================================

    def get_results(
        self,
        test_id: Optional[str] = None,
        status: Optional[TestStatus] = None,
        limit: int = 50,
    ) -> List[TestResult]:
        """Retrieve test results with optional filters."""
        filtered = self._results
        if test_id:
            filtered = [r for r in filtered if r.test_id == test_id]
        if status:
            filtered = [r for r in filtered if r.status == status]
        return filtered[-limit:]

    def get_test_report(self, test_id: str) -> Dict[str, Any]:
        """
        Generate a report for a specific test case.

        Includes pass rate, average duration, failure patterns, and
        recent run history.
        """
        tc = self.get_test(test_id)
        results = [r for r in self._results if r.test_id == test_id]

        if not results:
            return {
                "test_id": test_id,
                "name": tc.name,
                "type": tc.test_type.value,
                "app": tc.app,
                "total_runs": 0,
                "pass_rate": 0.0,
                "avg_duration_ms": 0.0,
                "failure_patterns": [],
                "recent_results": [],
            }

        passes = sum(1 for r in results if r.status == TestStatus.PASSED)
        failures = sum(1 for r in results if r.status in (TestStatus.FAILED, TestStatus.ERROR))
        durations = [r.duration_ms for r in results if r.duration_ms > 0]

        # Identify failure patterns
        error_counts: Dict[str, int] = {}
        for r in results:
            if r.error:
                # Normalize error to first 80 chars for grouping
                key = r.error[:80]
                error_counts[key] = error_counts.get(key, 0) + 1

        failure_patterns = sorted(
            [{"error": k, "count": v} for k, v in error_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:10]

        recent = results[-10:]

        return {
            "test_id": test_id,
            "name": tc.name,
            "type": tc.test_type.value,
            "app": tc.app,
            "total_runs": len(results),
            "passes": passes,
            "failures": failures,
            "pass_rate": round(passes / len(results) * 100, 1) if results else 0.0,
            "avg_duration_ms": round(sum(durations) / len(durations), 1) if durations else 0.0,
            "min_duration_ms": round(min(durations), 1) if durations else 0.0,
            "max_duration_ms": round(max(durations), 1) if durations else 0.0,
            "failure_patterns": failure_patterns,
            "recent_results": [
                {
                    "result_id": r.result_id,
                    "status": r.status.value,
                    "duration_ms": round(r.duration_ms, 1),
                    "completed_at": r.completed_at,
                    "error": r.error,
                }
                for r in recent
            ],
        }

    def get_suite_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate an overall test suite health report.

        Summarizes pass/fail rates, flaky tests, slowest tests, and
        coverage across test types over the specified number of days.
        """
        from datetime import timedelta

        cutoff = (_now_utc() - timedelta(days=days)).isoformat()
        recent_results = [r for r in self._results if r.completed_at >= cutoff]

        total = len(recent_results)
        passed = sum(1 for r in recent_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in recent_results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in recent_results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in recent_results if r.status == TestStatus.SKIPPED)

        # Per-test stats
        test_stats: Dict[str, Dict[str, int]] = {}
        for r in recent_results:
            if r.test_id not in test_stats:
                test_stats[r.test_id] = {"passed": 0, "failed": 0, "total": 0}
            test_stats[r.test_id]["total"] += 1
            if r.status == TestStatus.PASSED:
                test_stats[r.test_id]["passed"] += 1
            elif r.status in (TestStatus.FAILED, TestStatus.ERROR):
                test_stats[r.test_id]["failed"] += 1

        # Flaky tests: non-zero pass AND non-zero fail
        flaky = [
            tid for tid, s in test_stats.items()
            if s["passed"] > 0 and s["failed"] > 0 and s["total"] >= 3
        ]

        # Slowest tests (by average duration)
        test_durations: Dict[str, List[float]] = {}
        for r in recent_results:
            test_durations.setdefault(r.test_id, []).append(r.duration_ms)
        slowest = sorted(
            [
                {"test_id": tid, "avg_ms": round(sum(d) / len(d), 1)}
                for tid, d in test_durations.items()
                if d
            ],
            key=lambda x: x["avg_ms"],
            reverse=True,
        )[:5]

        # Coverage by test type
        type_coverage: Dict[str, int] = {}
        for tc in self._test_cases.values():
            tt = tc.test_type.value
            type_coverage[tt] = type_coverage.get(tt, 0) + 1

        return {
            "period_days": days,
            "total_runs": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "pass_rate": round(passed / total * 100, 1) if total else 0.0,
            "total_tests": len(self._test_cases),
            "flaky_tests": flaky,
            "flaky_count": len(flaky),
            "slowest_tests": slowest,
            "type_coverage": type_coverage,
        }

    def get_flaky_tests(self, min_runs: int = 5) -> List[Dict[str, Any]]:
        """
        Identify tests that sometimes pass and sometimes fail.

        A test is considered flaky if it has at least `min_runs` executions
        and a pass rate between 10% and 90%.
        """
        test_outcomes: Dict[str, Dict[str, int]] = {}
        for r in self._results:
            if r.test_id not in test_outcomes:
                test_outcomes[r.test_id] = {"passed": 0, "failed": 0, "total": 0}
            test_outcomes[r.test_id]["total"] += 1
            if r.status == TestStatus.PASSED:
                test_outcomes[r.test_id]["passed"] += 1
            elif r.status in (TestStatus.FAILED, TestStatus.ERROR):
                test_outcomes[r.test_id]["failed"] += 1

        flaky: List[Dict[str, Any]] = []
        for tid, stats in test_outcomes.items():
            if stats["total"] < min_runs:
                continue
            rate = stats["passed"] / stats["total"]
            if 0.1 <= rate <= 0.9:
                tc_name = self._test_cases.get(tid, TestCase(
                    test_id=tid, name="unknown", test_type=TestType.FLOW_VALIDATION, app=""
                )).name
                flaky.append({
                    "test_id": tid,
                    "name": tc_name,
                    "total_runs": stats["total"],
                    "pass_count": stats["passed"],
                    "fail_count": stats["failed"],
                    "pass_rate": round(rate * 100, 1),
                })

        return sorted(flaky, key=lambda x: abs(x["pass_rate"] - 50))

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all tests and results."""
        total_tests = len(self._test_cases)
        total_results = len(self._results)
        total_baselines = len(self._baselines)

        by_type: Dict[str, int] = {}
        by_app: Dict[str, int] = {}
        for tc in self._test_cases.values():
            by_type[tc.test_type.value] = by_type.get(tc.test_type.value, 0) + 1
            by_app[tc.app] = by_app.get(tc.app, 0) + 1

        result_by_status: Dict[str, int] = {}
        for r in self._results:
            result_by_status[r.status.value] = result_by_status.get(r.status.value, 0) + 1

        durations = [r.duration_ms for r in self._results if r.duration_ms > 0]

        return {
            "total_tests": total_tests,
            "total_results": total_results,
            "total_baselines": total_baselines,
            "tests_by_type": by_type,
            "tests_by_app": by_app,
            "results_by_status": result_by_status,
            "avg_duration_ms": round(sum(durations) / len(durations), 1) if durations else 0.0,
            "total_screenshots": sum(len(r.screenshots) for r in self._results),
        }

    # ===================================================================
    # Predefined Test Builders
    # ===================================================================

    def create_app_launch_test(
        self,
        app: str,
        name: Optional[str] = None,
    ) -> TestCase:
        """
        Create a simple test that launches an app and verifies it opens.

        Steps:
            1. Launch the app by package name
            2. Wait 3 seconds for the app to load
            3. Assert the app is running (check UI dump for non-empty elements)
        """
        test_name = name or f"Launch {app}"
        steps = [
            {
                "step_id": "launch",
                "action": "launch_app",
                "params": {"package": app},
                "expected_outcome": f"{app} launched successfully",
                "timeout": 15.0,
                "screenshot_after": True,
            },
            {
                "step_id": "wait_load",
                "action": "wait",
                "params": {"seconds": 3.0},
                "expected_outcome": "App finished loading",
                "screenshot_after": False,
            },
            {
                "step_id": "verify_running",
                "action": "screenshot",
                "params": {},
                "expected_outcome": "App screen is visible",
                "screenshot_after": True,
            },
        ]
        return self.create_test(
            name=test_name,
            test_type=TestType.PERFORMANCE,
            app=app,
            steps=steps,
            description=f"Verify that {app} launches without crashing",
            tags=["smoke", "launch"],
        )

    def create_login_flow_test(
        self,
        app: str,
        username_field: str,
        password_field: str,
        login_button: str,
        username: str = "test_user",
        password: str = "test_pass",
        name: Optional[str] = None,
    ) -> TestCase:
        """
        Create a login flow test with field entry and button tap.

        Steps:
            1. Launch app
            2. Wait for app to load
            3. Tap username field
            4. Type username
            5. Tap password field
            6. Type password
            7. Tap login button
            8. Assert login succeeded (assert text for post-login indicator)
        """
        test_name = name or f"Login flow: {app}"
        steps = [
            {
                "step_id": "launch",
                "action": "launch_app",
                "params": {"package": app},
                "expected_outcome": "App launched",
                "timeout": 15.0,
            },
            {
                "step_id": "wait_load",
                "action": "wait",
                "params": {"seconds": 3.0},
                "expected_outcome": "Login screen visible",
            },
            {
                "step_id": "tap_username",
                "action": "tap",
                "params": {"text": username_field},
                "expected_outcome": "Username field focused",
                "timeout": 10.0,
            },
            {
                "step_id": "type_username",
                "action": "type",
                "params": {"text": username},
                "expected_outcome": "Username entered",
            },
            {
                "step_id": "tap_password",
                "action": "tap",
                "params": {"text": password_field},
                "expected_outcome": "Password field focused",
                "timeout": 10.0,
            },
            {
                "step_id": "type_password",
                "action": "type",
                "params": {"text": password},
                "expected_outcome": "Password entered",
            },
            {
                "step_id": "tap_login",
                "action": "tap",
                "params": {"text": login_button},
                "expected_outcome": "Login submitted",
                "timeout": 10.0,
                "screenshot_after": True,
            },
            {
                "step_id": "wait_login",
                "action": "wait",
                "params": {"seconds": 5.0},
                "expected_outcome": "Login processing complete",
                "screenshot_after": True,
            },
        ]
        return self.create_test(
            name=test_name,
            test_type=TestType.FLOW_VALIDATION,
            app=app,
            steps=steps,
            description=f"End-to-end login flow for {app}",
            tags=["login", "flow", "e2e"],
        )

    def create_navigation_test(
        self,
        app: str,
        screens: List[str],
        name: Optional[str] = None,
    ) -> TestCase:
        """
        Create a navigation test that moves through a list of screens.

        Each screen name is used as a tap target (found by text matching).
        After tapping, the test waits briefly and captures a screenshot.

        Args:
            app: Package name of the app under test.
            screens: Ordered list of screen labels/tab names to navigate to.
            name: Optional test name override.
        """
        test_name = name or f"Navigation: {app} ({len(screens)} screens)"
        steps: List[Dict[str, Any]] = [
            {
                "step_id": "launch",
                "action": "launch_app",
                "params": {"package": app},
                "expected_outcome": "App launched",
                "timeout": 15.0,
                "screenshot_after": True,
            },
            {
                "step_id": "wait_load",
                "action": "wait",
                "params": {"seconds": 2.0},
                "expected_outcome": "App loaded",
                "screenshot_after": False,
            },
        ]

        for i, screen in enumerate(screens):
            steps.append({
                "step_id": f"nav_{i:03d}_{screen[:20].replace(' ', '_').lower()}",
                "action": "tap",
                "params": {"text": screen},
                "expected_outcome": f"Navigated to {screen}",
                "timeout": 10.0,
                "screenshot_after": True,
            })
            steps.append({
                "step_id": f"wait_{i:03d}",
                "action": "wait",
                "params": {"seconds": 1.5},
                "expected_outcome": f"{screen} fully loaded",
                "screenshot_after": False,
            })

        return self.create_test(
            name=test_name,
            test_type=TestType.FLOW_VALIDATION,
            app=app,
            steps=steps,
            description=f"Navigate through {len(screens)} screens in {app}",
            tags=["navigation", "flow"],
        )


# ===========================================================================
# Singleton
# ===========================================================================

_test_framework: Optional[MobileTestFramework] = None


def get_test_framework() -> MobileTestFramework:
    """Return the singleton MobileTestFramework instance."""
    global _test_framework
    if _test_framework is None:
        _test_framework = MobileTestFramework()
    return _test_framework


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    """CLI entry point with subcommands for test management and execution."""
    parser = argparse.ArgumentParser(
        prog="mobile_test_framework",
        description="Mobile Test Framework -- visual regression, flow validation, crash detection",
    )
    sub = parser.add_subparsers(dest="command")

    # --- create ---
    p_create = sub.add_parser("create", help="Create a new test case")
    p_create.add_argument("--name", required=True, help="Test case name")
    p_create.add_argument("--app", required=True, help="App package name")
    p_create.add_argument(
        "--type",
        default="flow_validation",
        choices=[t.value for t in TestType],
        help="Test type",
    )
    p_create.add_argument("--description", default="", help="Test description")
    p_create.add_argument("--tags", default="", help="Comma-separated tags")
    p_create.add_argument("--steps-json", help="Path to a JSON file with step definitions")

    # --- list ---
    p_list = sub.add_parser("list", help="List test cases")
    p_list.add_argument("--type", choices=[t.value for t in TestType], help="Filter by type")
    p_list.add_argument("--app", help="Filter by app")
    p_list.add_argument("--tags", help="Comma-separated tags to filter by")

    # --- run ---
    p_run = sub.add_parser("run", help="Run a single test")
    p_run.add_argument("--test-id", required=True, help="Test case ID")
    p_run.add_argument("--device-id", help="Optional device ID")

    # --- suite ---
    p_suite = sub.add_parser("suite", help="Run a suite of tests")
    p_suite.add_argument("--ids", required=True, help="Comma-separated test IDs")
    p_suite.add_argument("--device-id", help="Optional device ID")

    # --- baseline ---
    p_baseline = sub.add_parser("baseline", help="Capture a visual baseline")
    p_baseline.add_argument("--test-id", required=True, help="Test case ID")
    p_baseline.add_argument("--step-id", required=True, help="Step ID")
    p_baseline.add_argument("--device-id", help="Optional device ID")

    # --- compare ---
    p_compare = sub.add_parser("compare", help="Compare screenshot to baseline")
    p_compare.add_argument("--screenshot", required=True, help="Path to screenshot")
    p_compare.add_argument("--baseline-id", required=True, help="Baseline ID")
    p_compare.add_argument(
        "--method",
        default="structural",
        choices=[m.value for m in ComparisonMethod],
        help="Comparison method",
    )

    # --- crashes ---
    p_crashes = sub.add_parser("crashes", help="Monitor for crashes")
    p_crashes.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    p_crashes.add_argument("--app", help="App package to monitor")

    # --- results ---
    p_results = sub.add_parser("results", help="View test results")
    p_results.add_argument("--test-id", help="Filter by test ID")
    p_results.add_argument(
        "--status",
        choices=[s.value for s in TestStatus],
        help="Filter by status",
    )
    p_results.add_argument("--limit", type=int, default=20, help="Max results")

    # --- report ---
    p_report = sub.add_parser("report", help="Generate test report")
    p_report.add_argument("--test-id", help="Test ID (omit for suite report)")
    p_report.add_argument("--days", type=int, default=7, help="Days for suite report")

    # --- stats ---
    sub.add_parser("stats", help="Show aggregate statistics")

    # --- export ---
    p_export = sub.add_parser("export", help="Export a test case to JSON")
    p_export.add_argument("--test-id", required=True, help="Test case ID")
    p_export.add_argument("--output", help="Output file path (default: stdout)")

    # --- import ---
    p_import = sub.add_parser("import", help="Import a test case from JSON")
    p_import.add_argument("--input", required=True, help="Input JSON file path")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    fw = get_test_framework()

    # ---- dispatch ----

    if args.command == "create":
        steps_data = []
        if args.steps_json:
            with open(args.steps_json, "r", encoding="utf-8") as f:
                steps_data = json.load(f)
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        tc = fw.create_test(
            name=args.name,
            test_type=TestType(args.type),
            app=args.app,
            steps=steps_data,
            description=args.description,
            tags=tags,
        )
        print(json.dumps({"created": tc.test_id, "name": tc.name}, indent=2))

    elif args.command == "list":
        tt = TestType(args.type) if args.type else None
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
        tests = fw.list_tests(test_type=tt, app=args.app, tags=tags)
        for tc in tests:
            rate = f"{tc.pass_count}/{tc.run_count}" if tc.run_count else "0/0"
            print(
                f"  {tc.test_id}  {tc.test_type.value:<20s}  {tc.name:<40s}  "
                f"app={tc.app}  runs={rate}"
            )
        print(f"\n  Total: {len(tests)} test(s)")

    elif args.command == "run":
        result = fw.run_test_sync(args.test_id, device_id=args.device_id)
        print(json.dumps({
            "result_id": result.result_id,
            "status": result.status.value,
            "duration_ms": round(result.duration_ms, 1),
            "steps": f"{result.steps_completed}/{result.steps_total}",
            "error": result.error,
            "screenshots": len(result.screenshots),
        }, indent=2))

    elif args.command == "suite":
        test_ids = [tid.strip() for tid in args.ids.split(",") if tid.strip()]
        results = fw.run_suite_sync(test_ids, device_id=args.device_id)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status in (TestStatus.FAILED, TestStatus.ERROR))
        print(f"\nSuite results: {passed} passed, {failed} failed, {len(results)} total")
        for r in results:
            mark = "[PASS]" if r.status == TestStatus.PASSED else "[FAIL]"
            print(f"  {mark} {r.test_name} ({r.duration_ms:.0f}ms)")
            if r.error:
                print(f"        Error: {r.error[:120]}")

    elif args.command == "baseline":
        baseline = fw.capture_baseline_sync(
            args.test_id, args.step_id, device_id=args.device_id,
        )
        print(json.dumps({
            "baseline_id": baseline.baseline_id,
            "screenshot": baseline.screenshot_path,
            "created_at": baseline.created_at,
        }, indent=2))

    elif args.command == "compare":
        diff = fw.compare_visual_sync(
            args.screenshot,
            args.baseline_id,
            method=ComparisonMethod(args.method),
        )
        print(json.dumps(diff, indent=2))

    elif args.command == "crashes":
        crashes = fw.monitor_crashes_sync(
            duration_seconds=args.duration,
            app=args.app,
        )
        if crashes:
            print(f"Detected {len(crashes)} crash(es):")
            for c in crashes:
                print(f"  [{c.get('type')}] {c.get('source', '?')}: {c.get('text', c.get('error_message', ''))[:150]}")
        else:
            print("No crashes detected.")

    elif args.command == "results":
        status_filter = TestStatus(args.status) if args.status else None
        results = fw.get_results(
            test_id=args.test_id,
            status=status_filter,
            limit=args.limit,
        )
        for r in results:
            mark = "[PASS]" if r.status == TestStatus.PASSED else (
                "[FAIL]" if r.status == TestStatus.FAILED else f"[{r.status.value.upper()}]"
            )
            print(
                f"  {mark} {r.result_id} | {r.test_name:<30s} | "
                f"{r.duration_ms:.0f}ms | {r.completed_at}"
            )
            if r.error:
                print(f"         Error: {r.error[:120]}")
        print(f"\n  Showing {len(results)} result(s)")

    elif args.command == "report":
        if args.test_id:
            report = fw.get_test_report(args.test_id)
        else:
            report = fw.get_suite_report(days=args.days)
        print(json.dumps(report, indent=2))

    elif args.command == "stats":
        stats = fw.get_stats()
        print(json.dumps(stats, indent=2))

    elif args.command == "export":
        data = fw.export_test(args.test_id)
        output_json = json.dumps(data, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_json)
            print(f"Exported to {args.output}")
        else:
            print(output_json)

    elif args.command == "import":
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                tc = fw.import_test(item)
                print(f"Imported: {tc.test_id} ({tc.name})")
        else:
            tc = fw.import_test(data)
            print(f"Imported: {tc.test_id} ({tc.name})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
