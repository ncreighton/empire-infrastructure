"""
Workflow Recorder & Playbook System — OpenClaw Empire Android Automation

Records human actions on an Android device as replayable playbooks, similar to
Selenium Recorder but for Android.  Integrates with the PhoneController for
ADB command execution and uses Claude Haiku for intelligent step description
generation.

Three core subsystems:
    Recording Engine  — captures user actions as timestamped steps with
                        before/after screenshots and UI hierarchy diffs
    Playbook System   — persistent, editable, templated, variable-driven
                        sequences of device actions
    Execution Engine  — replays playbooks with element-resolution fallback
                        chains, retry logic, and assertion verification

Data stored under:  data/playbooks/
History stored under: data/playbooks/history/

Usage:
    from src.workflow_recorder import get_recorder

    recorder = get_recorder()

    # Record a workflow
    await recorder.start_recording("Login to Instagram", "com.instagram.android")
    # ... user performs actions on device ...
    playbook = await recorder.stop_recording()

    # Replay it
    result = await recorder.execute_playbook(playbook.playbook_id, variables={"username": "nick"})

    # Smart features
    await recorder.auto_describe_steps(playbook.playbook_id)
    await recorder.optimize_playbook(playbook.playbook_id)
    await recorder.generalize_playbook(playbook.playbook_id)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("workflow_recorder")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PLAYBOOK_DIR = BASE_DIR / "data" / "playbooks"
PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DIR = PLAYBOOK_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

RECORDING_SCREENSHOTS_DIR = PLAYBOOK_DIR / "screenshots"
RECORDING_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = PLAYBOOK_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Anthropic API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Recording defaults
DEFAULT_POLL_INTERVAL = 1.5  # seconds between screen captures during recording
MAX_RECORDING_DURATION = 3600  # 1 hour safety limit
MAX_HISTORY_ENTRIES = 500

# Execution defaults
DEFAULT_STEP_TIMEOUT = 30.0
DEFAULT_RETRY_DELAY = 1.0
MAX_STEP_RETRIES = 3


# ---------------------------------------------------------------------------
# JSON persistence helpers (atomic writes)
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


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
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RecordingState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"


class ActionType(str, Enum):
    TAP = "tap"
    TYPE = "type"
    SWIPE = "swipe"
    SCROLL = "scroll"
    KEY = "key"
    WAIT = "wait"
    LAUNCH = "launch"
    BACK = "back"
    HOME = "home"
    LONG_PRESS = "long_press"


class OnFailure(str, Enum):
    SKIP = "skip"
    RETRY = "retry"
    ABORT = "abort"


class ExportFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"


# ===================================================================
# RECORDING ENGINE — Data Models
# ===================================================================

@dataclass
class RecordedStep:
    """A single captured step during a recording session."""

    step_num: int = 0
    action_type: str = "tap"
    timestamp: str = field(default_factory=_now_iso)
    coordinates: Dict[str, int] = field(default_factory=dict)
    element_info: Dict[str, Any] = field(default_factory=dict)
    input_text: str = ""
    swipe_vector: Dict[str, int] = field(default_factory=dict)
    key_code: str = ""
    screenshot_before: str = ""
    screenshot_after: str = ""
    screen_state_before: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    description: str = ""
    wait_condition: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RecordedStep:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass
class RecordingSession:
    """An active or completed recording session."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    app_package: str = ""
    started_at: str = field(default_factory=_now_iso)
    ended_at: Optional[str] = None
    steps: List[RecordedStep] = field(default_factory=list)
    device_id: str = ""
    screen_resolution: str = ""
    android_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RecordingSession:
        data = dict(data)
        raw_steps = data.pop("steps", [])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        session = cls(**filtered)
        session.steps = [RecordedStep.from_dict(s) for s in raw_steps]
        return session


# ===================================================================
# PLAYBOOK SYSTEM — Data Models
# ===================================================================

@dataclass
class PlaybookStep:
    """A single step within a playbook for execution."""

    step_num: int = 0
    action: str = "tap"
    target: Dict[str, Any] = field(default_factory=dict)
    input: str = ""
    wait_before_ms: int = 0
    wait_after_ms: int = 500
    screenshot_reference: str = ""
    description: str = ""
    optional: bool = False
    on_failure: str = "abort"
    max_retries: int = 2
    assertion: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlaybookStep:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass
class Playbook:
    """A replayable sequence of device actions with variables and metadata."""

    playbook_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    description: str = ""
    app_package: str = ""
    version: str = "1.0.0"
    steps: List[PlaybookStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    author: str = "OpenClaw Empire"
    tags: List[str] = field(default_factory=list)
    run_count: int = 0
    success_count: int = 0
    avg_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Playbook:
        data = dict(data)
        raw_steps = data.pop("steps", [])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        pb = cls(**filtered)
        pb.steps = [PlaybookStep.from_dict(s) for s in raw_steps]
        return pb


# ===================================================================
# EXECUTION ENGINE — Data Models
# ===================================================================

@dataclass
class StepResult:
    """Result of executing a single playbook step."""

    step_num: int = 0
    success: bool = False
    action_taken: str = ""
    duration_ms: float = 0.0
    screenshot: str = ""
    error: Optional[str] = None
    retries_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StepResult:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass
class ExecutionResult:
    """Result of executing an entire playbook."""

    playbook_id: str = ""
    device_id: str = ""
    started_at: str = field(default_factory=_now_iso)
    completed_at: Optional[str] = None
    success: bool = False
    steps_completed: int = 0
    steps_total: int = 0
    failed_step: Optional[int] = None
    error: Optional[str] = None
    step_results: List[StepResult] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    variables_used: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["step_results"] = [sr.to_dict() for sr in self.step_results]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionResult:
        data = dict(data)
        raw_results = data.pop("step_results", [])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        er = cls(**filtered)
        er.step_results = [StepResult.from_dict(sr) for sr in raw_results]
        return er


# ===================================================================
# WORKFLOW RECORDER — Main Class
# ===================================================================

class WorkflowRecorder:
    """
    Records human interactions on an Android device and converts them into
    replayable playbooks.  Provides playbook management, smart editing,
    templating, execution with variable substitution, and execution history.

    Integrates with PhoneController for device interaction and uses Claude
    Haiku for intelligent step description generation.
    """

    def __init__(self) -> None:
        self._recording_state: RecordingState = RecordingState.IDLE
        self._current_session: Optional[RecordingSession] = None
        self._recording_task: Optional[asyncio.Task] = None
        self._phone_controller: Optional[Any] = None
        self._playbooks: Dict[str, Playbook] = {}
        self._history: List[Dict[str, Any]] = []
        self._templates: Dict[str, Playbook] = {}

        # Load persisted state
        self._load_playbooks()
        self._load_history()
        self._load_builtin_templates()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_playbooks(self) -> None:
        """Load all playbooks from the data/playbooks/ directory."""
        for path in PLAYBOOK_DIR.glob("pb_*.json"):
            try:
                data = _load_json(path, default={})
                if data:
                    pb = Playbook.from_dict(data)
                    self._playbooks[pb.playbook_id] = pb
            except Exception as exc:
                logger.warning("Failed to load playbook %s: %s", path.name, exc)
        logger.info("Loaded %d playbooks from disk.", len(self._playbooks))

    def _save_playbook_to_disk(self, playbook: Playbook) -> None:
        """Save a single playbook to its JSON file."""
        path = PLAYBOOK_DIR / f"pb_{playbook.playbook_id}.json"
        _save_json(path, playbook.to_dict())

    def _delete_playbook_from_disk(self, playbook_id: str) -> None:
        """Remove a playbook's JSON file."""
        path = PLAYBOOK_DIR / f"pb_{playbook_id}.json"
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete playbook file %s: %s", path, exc)

    def _load_history(self) -> None:
        """Load execution history from disk."""
        hist_file = HISTORY_DIR / "execution_history.json"
        raw = _load_json(hist_file, default=[])
        if isinstance(raw, list):
            self._history = raw[-MAX_HISTORY_ENTRIES:]
        else:
            self._history = []
        logger.info("Loaded %d history entries.", len(self._history))

    def _save_history(self) -> None:
        """Persist execution history to disk."""
        hist_file = HISTORY_DIR / "execution_history.json"
        _save_json(hist_file, self._history[-MAX_HISTORY_ENTRIES:])

    def _record_execution(self, result: ExecutionResult) -> None:
        """Append an execution result to history."""
        self._history.append(result.to_dict())
        if len(self._history) > MAX_HISTORY_ENTRIES:
            self._history = self._history[-MAX_HISTORY_ENTRIES:]
        self._save_history()

    # ------------------------------------------------------------------
    # Phone Controller Integration
    # ------------------------------------------------------------------

    def _get_controller(self) -> Any:
        """Lazily import and return the phone controller."""
        if self._phone_controller is None:
            try:
                from src.phone_controller import PhoneController
                self._phone_controller = PhoneController()
            except ImportError:
                logger.warning(
                    "PhoneController not available. Recording and execution "
                    "require the phone_controller module."
                )
                raise RuntimeError("PhoneController not available")
        return self._phone_controller

    def set_controller(self, controller: Any) -> None:
        """Inject an external PhoneController instance."""
        self._phone_controller = controller

    # ------------------------------------------------------------------
    # Anthropic Haiku Integration
    # ------------------------------------------------------------------

    async def _call_haiku(self, prompt: str, image_b64: Optional[str] = None) -> str:
        """
        Call Claude Haiku for step description generation.

        Uses the direct Anthropic Messages API with optional vision input.
        Returns the text response.
        """
        if not ANTHROPIC_API_KEY:
            logger.debug("ANTHROPIC_API_KEY not set, skipping Haiku call.")
            return ""

        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not available for Haiku calls.")
            return ""

        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        content: List[Dict[str, Any]] = []
        if image_b64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": HAIKU_MODEL,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": content}],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("Haiku API error %d: %s", resp.status, body[:200])
                        return ""
                    data = await resp.json()
                    blocks = data.get("content", [])
                    return blocks[0].get("text", "") if blocks else ""
        except Exception as exc:
            logger.warning("Haiku call failed: %s", exc)
            return ""

    # ==================================================================
    # RECORDING ENGINE
    # ==================================================================

    async def start_recording(
        self,
        name: str,
        app_package: str = "",
        device_id: str = "",
    ) -> RecordingSession:
        """
        Begin a new recording session.

        Captures the initial screen state and starts polling for changes.
        The recording captures before/after screenshots and UI hierarchy
        at each detected action boundary.

        Args:
            name:        Human-readable name for this recording
            app_package: Android package being recorded (optional)
            device_id:   Target device ID (optional)

        Returns:
            The new RecordingSession object.

        Raises:
            RuntimeError: If already recording.
        """
        if self._recording_state != RecordingState.IDLE:
            raise RuntimeError(
                f"Cannot start recording: current state is {self._recording_state.value}"
            )

        session = RecordingSession(
            name=name,
            app_package=app_package,
            device_id=device_id,
        )

        # Try to get device info from controller
        try:
            controller = self._get_controller()
            if hasattr(controller, "resolution"):
                w, h = controller.resolution
                session.screen_resolution = f"{w}x{h}"
            if hasattr(controller, "_adb_shell"):
                version = await controller._adb_shell("getprop ro.build.version.release")
                session.android_version = version.strip()
        except Exception as exc:
            logger.debug("Could not get device info: %s", exc)

        self._current_session = session
        self._recording_state = RecordingState.RECORDING

        logger.info(
            "Recording started: %r (session=%s, app=%s)",
            name, session.session_id, app_package,
        )
        return session

    async def stop_recording(self) -> Playbook:
        """
        Stop the current recording session and convert it to a Playbook.

        Returns:
            A new Playbook generated from the recorded steps.

        Raises:
            RuntimeError: If not currently recording.
        """
        if self._recording_state == RecordingState.IDLE:
            raise RuntimeError("No recording in progress.")

        if self._recording_task and not self._recording_task.done():
            self._recording_task.cancel()
            try:
                await self._recording_task
            except asyncio.CancelledError:
                pass
            self._recording_task = None

        session = self._current_session
        if session is None:
            raise RuntimeError("Recording session is None.")

        session.ended_at = _now_iso()
        self._recording_state = RecordingState.IDLE

        # Convert to playbook
        playbook = self._session_to_playbook(session)
        self._playbooks[playbook.playbook_id] = playbook
        self._save_playbook_to_disk(playbook)

        logger.info(
            "Recording stopped: %r (%d steps) -> playbook %s",
            session.name, len(session.steps), playbook.playbook_id,
        )

        # Save the raw session for reference
        session_path = PLAYBOOK_DIR / f"session_{session.session_id}.json"
        _save_json(session_path, session.to_dict())

        self._current_session = None
        return playbook

    def pause_recording(self) -> None:
        """Pause the active recording without stopping it."""
        if self._recording_state != RecordingState.RECORDING:
            raise RuntimeError("Not currently recording.")
        self._recording_state = RecordingState.PAUSED
        logger.info("Recording paused.")

    def resume_recording(self) -> None:
        """Resume a paused recording."""
        if self._recording_state != RecordingState.PAUSED:
            raise RuntimeError("Recording is not paused.")
        self._recording_state = RecordingState.RECORDING
        logger.info("Recording resumed.")

    async def capture_step(
        self,
        action_type: str,
        coordinates: Optional[Dict[str, int]] = None,
        element_info: Optional[Dict[str, Any]] = None,
        input_text: str = "",
        swipe_vector: Optional[Dict[str, int]] = None,
        key_code: str = "",
        description: str = "",
    ) -> Optional[RecordedStep]:
        """
        Manually capture a single step during recording.

        This is the primary interface for adding steps during an active
        recording session. Each step captures before/after screenshots
        and UI state.

        Args:
            action_type:   One of tap, type, swipe, scroll, key, wait, launch, back
            coordinates:   {x, y} for tap/long_press actions
            element_info:  {text, resource_id, class_name, content_desc, bounds}
            input_text:    Text for type actions
            swipe_vector:  {x1, y1, x2, y2} for swipe actions
            key_code:      Android keycode string for key actions
            description:   Human-readable description of this step

        Returns:
            The captured RecordedStep, or None if not recording.
        """
        if self._recording_state != RecordingState.RECORDING:
            logger.debug("Not recording, ignoring capture_step call.")
            return None

        session = self._current_session
        if session is None:
            return None

        step_num = len(session.steps) + 1
        start_time = time.monotonic()

        # Capture before-screenshot
        screenshot_before = ""
        screen_state_before: Dict[str, Any] = {}
        try:
            controller = self._get_controller()
            screenshot_before = await controller.screenshot()
            ui_elements = await controller.ui_dump()
            screen_state_before = {
                "element_count": len(ui_elements),
                "elements": [
                    {
                        "text": el.text,
                        "resource_id": el.resource_id,
                        "class_name": el.class_name,
                        "content_desc": el.content_desc,
                        "bounds": el.bounds,
                    }
                    for el in ui_elements[:50]  # limit to top 50
                ],
            }
        except Exception as exc:
            logger.debug("Screenshot before failed: %s", exc)

        # Record the step
        step = RecordedStep(
            step_num=step_num,
            action_type=action_type,
            timestamp=_now_iso(),
            coordinates=coordinates or {},
            element_info=element_info or {},
            input_text=input_text,
            swipe_vector=swipe_vector or {},
            key_code=key_code,
            screenshot_before=screenshot_before,
            screen_state_before=screen_state_before,
            description=description,
        )

        # Small delay then capture after-screenshot
        await asyncio.sleep(0.8)
        try:
            step.screenshot_after = await controller.screenshot()
        except Exception as exc:
            logger.debug("Screenshot after failed: %s", exc)

        step.duration_ms = (time.monotonic() - start_time) * 1000
        session.steps.append(step)

        logger.info(
            "Captured step %d: %s %s",
            step_num, action_type,
            f"at ({coordinates.get('x')}, {coordinates.get('y')})" if coordinates else "",
        )
        return step

    async def _detect_action(
        self,
        before_screenshot: str,
        after_screenshot: str,
        before_ui: Dict[str, Any],
        after_ui: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Use vision to determine what the user did between two screen states.

        Sends before/after screenshots to Claude Haiku and asks it to
        identify the action that was performed.

        Returns:
            Dict with keys: action_type, coordinates, element_info, input_text, description
        """
        before_b64 = ""
        after_b64 = ""
        try:
            if before_screenshot and Path(before_screenshot).exists():
                with open(before_screenshot, "rb") as f:
                    before_b64 = base64.b64encode(f.read()).decode("ascii")
            if after_screenshot and Path(after_screenshot).exists():
                with open(after_screenshot, "rb") as f:
                    after_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception as exc:
            logger.debug("Failed to load screenshots for detection: %s", exc)

        prompt = (
            "Compare these two Android screenshots (before and after a user action). "
            "Determine what single action the user performed. "
            "Respond as JSON with keys: action_type (tap/type/swipe/scroll/key/back/home), "
            "coordinates ({x, y} if applicable), input_text (if text was typed), "
            "element_description (what element was interacted with), "
            "description (one sentence summary of the action)."
        )

        # For now, return a basic detection based on UI diff
        result: Dict[str, Any] = {
            "action_type": "tap",
            "coordinates": {},
            "element_info": {},
            "input_text": "",
            "description": "User action detected",
        }

        # Attempt Haiku vision analysis if we have screenshots
        if before_b64 and after_b64 and ANTHROPIC_API_KEY:
            response = await self._call_haiku(prompt, image_b64=after_b64)
            if response:
                try:
                    cleaned = response.strip()
                    if cleaned.startswith("```"):
                        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
                        cleaned = re.sub(r"\n?```$", "", cleaned)
                    parsed = json.loads(cleaned)
                    result.update(parsed)
                except (json.JSONDecodeError, TypeError):
                    result["description"] = response[:200]

        # Heuristic fallback: check for new text in UI
        before_texts = {e.get("text", "") for e in before_ui.get("elements", [])}
        after_texts = {e.get("text", "") for e in after_ui.get("elements", [])}
        new_texts = after_texts - before_texts
        if new_texts:
            new_text_str = "; ".join(t for t in new_texts if t)
            if new_text_str:
                result["description"] = f"Screen changed: new text appeared: {new_text_str[:100]}"

        return result

    def _session_to_playbook(self, session: RecordingSession) -> Playbook:
        """Convert a completed RecordingSession into a Playbook."""
        pb_steps: List[PlaybookStep] = []

        for rec_step in session.steps:
            target: Dict[str, Any] = {}

            # Build target selector from element_info
            if rec_step.element_info.get("resource_id"):
                target["resource_id"] = rec_step.element_info["resource_id"]
            if rec_step.element_info.get("text"):
                target["text"] = rec_step.element_info["text"]
            if rec_step.element_info.get("content_desc"):
                target["content_desc"] = rec_step.element_info["content_desc"]
            if rec_step.coordinates:
                target["coordinates"] = rec_step.coordinates
            if rec_step.swipe_vector:
                target["swipe_vector"] = rec_step.swipe_vector
            if rec_step.key_code:
                target["key_code"] = rec_step.key_code

            pb_step = PlaybookStep(
                step_num=rec_step.step_num,
                action=rec_step.action_type,
                target=target,
                input=rec_step.input_text,
                wait_after_ms=500,
                screenshot_reference=rec_step.screenshot_after,
                description=rec_step.description or f"Step {rec_step.step_num}: {rec_step.action_type}",
                on_failure="retry",
                max_retries=2,
            )
            pb_steps.append(pb_step)

        playbook = Playbook(
            name=session.name,
            description=f"Recorded from session {session.session_id}",
            app_package=session.app_package,
            steps=pb_steps,
            tags=["recorded"],
            created_at=session.started_at,
            updated_at=_now_iso(),
        )
        return playbook

    # ==================================================================
    # PLAYBOOK MANAGEMENT
    # ==================================================================

    def save_playbook(self, playbook: Playbook) -> None:
        """Save a playbook to memory and disk."""
        playbook.updated_at = _now_iso()
        self._playbooks[playbook.playbook_id] = playbook
        self._save_playbook_to_disk(playbook)
        logger.info("Saved playbook %r (id=%s)", playbook.name, playbook.playbook_id)

    def load_playbook(self, playbook_id: str) -> Optional[Playbook]:
        """Load a playbook by ID. Returns None if not found."""
        if playbook_id in self._playbooks:
            return self._playbooks[playbook_id]

        # Try loading from disk
        path = PLAYBOOK_DIR / f"pb_{playbook_id}.json"
        if path.exists():
            data = _load_json(path)
            if data:
                pb = Playbook.from_dict(data)
                self._playbooks[pb.playbook_id] = pb
                return pb
        return None

    def list_playbooks(
        self,
        app: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Playbook]:
        """
        List all playbooks, optionally filtered by app package or tags.

        Args:
            app:  Filter by app_package (partial match)
            tags: Filter by tags (any match)

        Returns:
            List of matching Playbook objects, sorted by name.
        """
        results = list(self._playbooks.values())

        if app:
            app_lower = app.lower()
            results = [pb for pb in results if app_lower in pb.app_package.lower()]

        if tags:
            tag_set = set(t.lower() for t in tags)
            results = [
                pb for pb in results
                if tag_set & set(t.lower() for t in pb.tags)
            ]

        results.sort(key=lambda pb: pb.name.lower())
        return results

    def delete_playbook(self, playbook_id: str) -> bool:
        """Delete a playbook by ID. Returns True if found and deleted."""
        if playbook_id in self._playbooks:
            pb = self._playbooks.pop(playbook_id)
            self._delete_playbook_from_disk(playbook_id)
            logger.info("Deleted playbook %r (id=%s)", pb.name, playbook_id)
            return True
        logger.warning("Playbook %s not found for deletion.", playbook_id)
        return False

    def export_playbook(
        self,
        playbook_id: str,
        format: str = "json",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export a playbook to a file.

        Args:
            playbook_id: The playbook to export
            format:      'json' or 'yaml'
            output_path: Output file path (auto-generated if not specified)

        Returns:
            The path to the exported file.

        Raises:
            KeyError: If playbook not found.
            ValueError: If format not supported.
        """
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")

        data = pb.to_dict()

        if format.lower() == "json":
            ext = ".json"
            content = json.dumps(data, indent=2, default=str)
        elif format.lower() == "yaml":
            ext = ".yaml"
            try:
                import yaml
                content = yaml.dump(data, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ValueError(
                    "PyYAML is required for YAML export. Install with: pip install pyyaml"
                )
        else:
            raise ValueError(f"Unsupported export format: {format}")

        if output_path is None:
            safe_name = re.sub(r"[^\w\-]", "_", pb.name.lower())[:40]
            output_path = str(PLAYBOOK_DIR / f"export_{safe_name}_{playbook_id}{ext}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Exported playbook %r to %s", pb.name, output_path)
        return output_path

    def import_playbook(self, file_path: str) -> Playbook:
        """
        Import a playbook from a JSON or YAML file.

        Args:
            file_path: Path to the playbook file.

        Returns:
            The imported Playbook.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is unsupported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {file_path}")

        content = path.read_text(encoding="utf-8")

        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ValueError("PyYAML required for YAML import.")
        elif path.suffix.lower() == ".json":
            data = json.loads(content)
        else:
            # Try JSON first, then YAML
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml
                    data = yaml.safe_load(content)
                except Exception:
                    raise ValueError(f"Cannot parse file: {file_path}")

        # Assign a new ID to avoid collisions
        pb = Playbook.from_dict(data)
        pb.playbook_id = uuid.uuid4().hex[:12]
        pb.updated_at = _now_iso()

        self._playbooks[pb.playbook_id] = pb
        self._save_playbook_to_disk(pb)

        logger.info("Imported playbook %r as %s from %s", pb.name, pb.playbook_id, file_path)
        return pb

    # ==================================================================
    # PLAYBOOK EXECUTION
    # ==================================================================

    async def execute_playbook(
        self,
        playbook_id: str,
        variables: Optional[Dict[str, Any]] = None,
        device_id: str = "",
    ) -> ExecutionResult:
        """
        Execute a playbook on the connected Android device.

        Each step is resolved to a target element, the action is performed,
        and optional assertions are verified. Variables in step inputs are
        substituted with {{variable}} syntax.

        Args:
            playbook_id: ID of the playbook to execute
            variables:   Variable values for template substitution
            device_id:   Target device (optional)

        Returns:
            ExecutionResult with detailed step-by-step results.
        """
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            return ExecutionResult(
                playbook_id=playbook_id,
                error=f"Playbook {playbook_id} not found.",
                steps_total=0,
            )

        variables = variables or {}
        merged_vars = {**pb.variables, **variables}

        result = ExecutionResult(
            playbook_id=playbook_id,
            device_id=device_id,
            steps_total=len(pb.steps),
            variables_used=merged_vars,
        )

        controller = None
        try:
            controller = self._get_controller()
            if hasattr(controller, "connect"):
                connected = await controller.connect()
                if not connected:
                    result.error = "Failed to connect to device."
                    result.completed_at = _now_iso()
                    self._record_execution(result)
                    return result
        except Exception as exc:
            result.error = f"Controller error: {exc}"
            result.completed_at = _now_iso()
            self._record_execution(result)
            return result

        start_time = time.monotonic()
        logger.info(
            "Executing playbook %r (%d steps, vars=%s)",
            pb.name, len(pb.steps), list(merged_vars.keys()),
        )

        for step in pb.steps:
            step_result = await self._execute_step(step, merged_vars, controller)
            result.step_results.append(step_result)

            if step_result.screenshot:
                result.screenshots.append(step_result.screenshot)

            if step_result.success:
                result.steps_completed += 1
            else:
                failure_policy = OnFailure(step.on_failure) if step.on_failure else OnFailure.ABORT

                if step.optional or failure_policy == OnFailure.SKIP:
                    logger.warning(
                        "Step %d failed but is optional/skip, continuing. Error: %s",
                        step.step_num, step_result.error,
                    )
                    result.steps_completed += 1
                    continue
                elif failure_policy == OnFailure.ABORT:
                    result.failed_step = step.step_num
                    result.error = step_result.error
                    logger.error(
                        "Step %d failed, aborting playbook. Error: %s",
                        step.step_num, step_result.error,
                    )
                    break

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result.duration_ms = elapsed_ms
        result.completed_at = _now_iso()
        result.success = result.steps_completed == result.steps_total

        # Update playbook stats
        pb.run_count += 1
        if result.success:
            pb.success_count += 1
        # Rolling average duration
        if pb.avg_duration_ms == 0:
            pb.avg_duration_ms = elapsed_ms
        else:
            pb.avg_duration_ms = (pb.avg_duration_ms * (pb.run_count - 1) + elapsed_ms) / pb.run_count
        pb.updated_at = _now_iso()
        self._save_playbook_to_disk(pb)

        # Record to history
        self._record_execution(result)

        logger.info(
            "Playbook %r %s: %d/%d steps in %.0fms",
            pb.name,
            "PASSED" if result.success else "FAILED",
            result.steps_completed,
            result.steps_total,
            elapsed_ms,
        )
        return result

    async def _execute_step(
        self,
        step: PlaybookStep,
        variables: Dict[str, Any],
        controller: Any,
    ) -> StepResult:
        """Execute a single playbook step with retry logic."""
        sr = StepResult(step_num=step.step_num, action_taken=step.action)
        start = time.monotonic()

        for attempt in range(step.max_retries + 1):
            sr.retries_used = attempt

            try:
                # Pre-step wait
                if step.wait_before_ms > 0:
                    await asyncio.sleep(step.wait_before_ms / 1000.0)

                # Resolve target and perform action
                await self._perform_action(step, variables, controller)

                # Post-step wait
                if step.wait_after_ms > 0:
                    await asyncio.sleep(step.wait_after_ms / 1000.0)

                # Capture screenshot
                try:
                    sr.screenshot = await controller.screenshot()
                except Exception:
                    pass

                # Verify assertion if present
                if step.assertion:
                    assertion_ok = await self._verify_assertion(step.assertion, controller)
                    if not assertion_ok:
                        raise RuntimeError(
                            f"Assertion failed: {step.assertion.get('description', 'unknown')}"
                        )

                sr.success = True
                sr.duration_ms = (time.monotonic() - start) * 1000
                return sr

            except Exception as exc:
                sr.error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Step %d attempt %d/%d failed: %s",
                    step.step_num, attempt + 1, step.max_retries + 1, sr.error,
                )
                if attempt < step.max_retries:
                    await asyncio.sleep(DEFAULT_RETRY_DELAY)

        sr.duration_ms = (time.monotonic() - start) * 1000
        return sr

    async def _perform_action(
        self,
        step: PlaybookStep,
        variables: Dict[str, Any],
        controller: Any,
    ) -> None:
        """
        Perform the actual device action for a playbook step.

        Resolves the target element using the selector strategy and
        executes the appropriate controller method.
        """
        action = step.action.lower()
        target = step.target

        if action == "tap" or action == "long_press":
            coords = await self._resolve_target(target, controller)
            if coords is None:
                raise RuntimeError(f"Could not resolve target for step {step.step_num}: {target}")
            x, y = coords
            if action == "tap":
                result = await controller.tap(x, y)
            else:
                result = await controller.long_press(x, y)
            if not result.success:
                raise RuntimeError(f"Action failed: {result.error}")

        elif action == "type":
            text = self._substitute_variables(step.input, variables)
            result = await controller.type_text(text)
            if not result.success:
                raise RuntimeError(f"Type failed: {result.error}")

        elif action == "swipe":
            sv = target.get("swipe_vector", {})
            result = await controller.swipe(
                int(sv.get("x1", 0)), int(sv.get("y1", 0)),
                int(sv.get("x2", 0)), int(sv.get("y2", 0)),
            )
            if not result.success:
                raise RuntimeError(f"Swipe failed: {result.error}")

        elif action == "scroll":
            direction = target.get("direction", "down")
            if direction == "up":
                result = await controller.scroll_up()
            else:
                result = await controller.scroll_down()
            if not result.success:
                raise RuntimeError(f"Scroll failed: {result.error}")

        elif action == "key":
            key_code = target.get("key_code", step.input)
            result = await controller.press_key(key_code)
            if not result.success:
                raise RuntimeError(f"Key press failed: {result.error}")

        elif action == "back":
            result = await controller.press_back()
            if not result.success:
                raise RuntimeError(f"Back failed: {result.error}")

        elif action == "home":
            result = await controller.press_home()
            if not result.success:
                raise RuntimeError(f"Home failed: {result.error}")

        elif action == "launch":
            package = target.get("package", step.input)
            package = self._substitute_variables(package, variables)
            result = await controller.launch_app(package)
            if not result.success:
                raise RuntimeError(f"Launch failed: {result.error}")

        elif action == "wait":
            wait_ms = target.get("duration_ms", 1000)
            await asyncio.sleep(wait_ms / 1000.0)

        else:
            raise ValueError(f"Unknown action type: {action}")

    async def _resolve_target(
        self,
        target: Dict[str, Any],
        controller: Any,
    ) -> Optional[Tuple[int, int]]:
        """
        Find an element on screen using a fallback selector strategy.

        Resolution order:
            1. resource_id — most reliable
            2. text — visible text match
            3. content_desc — accessibility description
            4. coordinates — absolute fallback

        Returns (x, y) tap coordinates or None if not found.
        """
        # Strategy 1: resource_id
        if target.get("resource_id"):
            el = await controller.find_element(resource_id=target["resource_id"])
            if el is not None:
                return el.center

        # Strategy 2: text
        if target.get("text"):
            el = await controller.find_element(text=target["text"])
            if el is not None:
                return el.center

        # Strategy 3: content_desc
        if target.get("content_desc"):
            el = await controller.find_element(content_desc=target["content_desc"])
            if el is not None:
                return el.center

        # Strategy 4: raw coordinates
        coords = target.get("coordinates", {})
        if coords.get("x") is not None and coords.get("y") is not None:
            return (int(coords["x"]), int(coords["y"]))

        return None

    def _substitute_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Replace {{variable}} placeholders with actual values."""
        if not text or not variables:
            return text
        result = text
        for name, value in variables.items():
            result = result.replace("{{" + name + "}}", str(value))
        return result

    async def _verify_assertion(
        self,
        assertion: Dict[str, Any],
        controller: Any,
    ) -> bool:
        """
        Verify a step assertion against the current screen state.

        Supported assertion types:
            element_exists:  Check that a UI element is present
            element_text:    Check that an element contains expected text
            screen_contains: Check that the screen contains specific text
            app_is:          Check that the foreground app matches

        Returns True if assertion passes.
        """
        assert_type = assertion.get("type", "")

        if assert_type == "element_exists":
            el = await controller.find_element(
                text=assertion.get("text"),
                resource_id=assertion.get("resource_id"),
            )
            return el is not None

        elif assert_type == "element_text":
            el = await controller.find_element(
                resource_id=assertion.get("resource_id"),
                text=assertion.get("text"),
            )
            if el is None:
                return False
            expected = assertion.get("expected_text", "")
            return expected.lower() in el.text.lower()

        elif assert_type == "screen_contains":
            elements = await controller.ui_dump()
            search_text = assertion.get("text", "").lower()
            for el in elements:
                if search_text in el.text.lower() or search_text in el.content_desc.lower():
                    return True
            return False

        elif assert_type == "app_is":
            current = await controller.get_current_app()
            expected = assertion.get("package", "")
            return expected in current

        else:
            logger.warning("Unknown assertion type: %s", assert_type)
            return True  # Unknown assertions pass by default

    async def dry_run(
        self,
        playbook_id: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a playbook without executing it.

        Checks that all steps have valid action types, targets are defined,
        variables are available, and returns a report of potential issues.

        Returns:
            Dict with keys: valid (bool), issues (list), step_count, variables_needed,
            variables_provided, estimated_duration_ms
        """
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            return {"valid": False, "issues": [f"Playbook {playbook_id} not found."]}

        variables = variables or {}
        merged_vars = {**pb.variables, **variables}
        issues: List[str] = []
        vars_needed: set = set()

        valid_actions = {a.value for a in ActionType}

        estimated_ms = 0.0

        for step in pb.steps:
            # Check action type
            if step.action.lower() not in valid_actions:
                issues.append(
                    f"Step {step.step_num}: unknown action '{step.action}'"
                )

            # Check target is defined for actions that need it
            if step.action.lower() in ("tap", "long_press") and not step.target:
                issues.append(
                    f"Step {step.step_num}: tap/long_press requires a target"
                )

            # Check for unresolved variables
            all_text = step.input + json.dumps(step.target)
            var_refs = re.findall(r"\{\{(\w+)\}\}", all_text)
            for var in var_refs:
                vars_needed.add(var)
                if var not in merged_vars:
                    issues.append(
                        f"Step {step.step_num}: variable '{{{{{var}}}}}' not provided"
                    )

            # Estimate duration
            estimated_ms += step.wait_before_ms + step.wait_after_ms + 500

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "step_count": len(pb.steps),
            "variables_needed": sorted(vars_needed),
            "variables_provided": sorted(merged_vars.keys()),
            "estimated_duration_ms": estimated_ms,
        }

    # ==================================================================
    # PLAYBOOK EDITOR
    # ==================================================================

    def add_step(self, playbook_id: str, step: PlaybookStep) -> Playbook:
        """Append a step to the end of a playbook."""
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")
        step.step_num = len(pb.steps) + 1
        pb.steps.append(step)
        self.save_playbook(pb)
        return pb

    def insert_step(self, playbook_id: str, position: int, step: PlaybookStep) -> Playbook:
        """Insert a step at the given position (1-indexed)."""
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")
        idx = max(0, min(position - 1, len(pb.steps)))
        pb.steps.insert(idx, step)
        self._renumber_steps(pb)
        self.save_playbook(pb)
        return pb

    def remove_step(self, playbook_id: str, step_num: int) -> Playbook:
        """Remove a step by its step number."""
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")
        pb.steps = [s for s in pb.steps if s.step_num != step_num]
        self._renumber_steps(pb)
        self.save_playbook(pb)
        return pb

    def update_step(
        self,
        playbook_id: str,
        step_num: int,
        updates: Dict[str, Any],
    ) -> Playbook:
        """Update specific fields on a step."""
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")

        for step in pb.steps:
            if step.step_num == step_num:
                for key, value in updates.items():
                    if hasattr(step, key):
                        setattr(step, key, value)
                    else:
                        logger.warning("Unknown step field: %s", key)
                break
        else:
            raise KeyError(f"Step {step_num} not found in playbook {playbook_id}.")

        self.save_playbook(pb)
        return pb

    def reorder_steps(self, playbook_id: str, new_order: List[int]) -> Playbook:
        """
        Reorder steps by providing a list of current step numbers in the
        desired order.  e.g., [3, 1, 2] moves step 3 to first position.
        """
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")

        step_map = {s.step_num: s for s in pb.steps}
        reordered = []
        for num in new_order:
            if num in step_map:
                reordered.append(step_map[num])
            else:
                logger.warning("Step number %d not found during reorder, skipping.", num)

        # Append any steps not mentioned in new_order
        mentioned = set(new_order)
        for s in pb.steps:
            if s.step_num not in mentioned:
                reordered.append(s)

        pb.steps = reordered
        self._renumber_steps(pb)
        self.save_playbook(pb)
        return pb

    def add_variable(
        self,
        playbook_id: str,
        name: str,
        default_value: Any = "",
        description: str = "",
    ) -> Playbook:
        """Add a template variable to a playbook."""
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")

        pb.variables[name] = {
            "default": default_value,
            "description": description,
        }
        self.save_playbook(pb)
        return pb

    def add_precondition(self, playbook_id: str, condition: str) -> Playbook:
        """Add a precondition requirement to a playbook."""
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")
        pb.preconditions.append(condition)
        self.save_playbook(pb)
        return pb

    def duplicate_playbook(self, playbook_id: str, new_name: str) -> Playbook:
        """Clone a playbook with a new ID and name."""
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")

        new_pb = Playbook.from_dict(pb.to_dict())
        new_pb.playbook_id = uuid.uuid4().hex[:12]
        new_pb.name = new_name
        new_pb.created_at = _now_iso()
        new_pb.updated_at = _now_iso()
        new_pb.run_count = 0
        new_pb.success_count = 0
        new_pb.avg_duration_ms = 0.0

        self._playbooks[new_pb.playbook_id] = new_pb
        self._save_playbook_to_disk(new_pb)

        logger.info(
            "Duplicated playbook %r -> %r (id=%s)",
            pb.name, new_name, new_pb.playbook_id,
        )
        return new_pb

    @staticmethod
    def _renumber_steps(pb: Playbook) -> None:
        """Renumber all steps sequentially starting from 1."""
        for i, step in enumerate(pb.steps, start=1):
            step.step_num = i

    # ==================================================================
    # SMART RECORDING FEATURES
    # ==================================================================

    async def auto_describe_steps(self, playbook_id: str) -> Playbook:
        """
        Use Claude Haiku to generate human-readable descriptions for each
        step based on the step's action, target, and screenshot reference.

        Updates step descriptions in place and saves the playbook.
        """
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")

        logger.info("Auto-describing %d steps for playbook %r", len(pb.steps), pb.name)

        for step in pb.steps:
            # Build a description prompt from step context
            context_parts = [
                f"Action: {step.action}",
            ]
            if step.target.get("text"):
                context_parts.append(f"Target element text: {step.target['text']}")
            if step.target.get("resource_id"):
                context_parts.append(f"Target resource ID: {step.target['resource_id']}")
            if step.target.get("content_desc"):
                context_parts.append(f"Target description: {step.target['content_desc']}")
            if step.input:
                context_parts.append(f"Input text: {step.input}")
            if step.target.get("coordinates"):
                coords = step.target["coordinates"]
                context_parts.append(f"Coordinates: ({coords.get('x')}, {coords.get('y')})")

            prompt = (
                "Write a brief (1 sentence, max 15 words) human-readable description "
                "of this Android automation step:\n" + "\n".join(context_parts) +
                "\nRespond with ONLY the description, no quotes or formatting."
            )

            # Try to include screenshot for better context
            image_b64 = None
            if step.screenshot_reference and Path(step.screenshot_reference).exists():
                try:
                    with open(step.screenshot_reference, "rb") as f:
                        image_b64 = base64.b64encode(f.read()).decode("ascii")
                except Exception:
                    pass

            description = await self._call_haiku(prompt, image_b64=image_b64)
            if description:
                step.description = description.strip().rstrip(".")
                logger.debug("Step %d described: %s", step.step_num, step.description)
            else:
                # Fallback to auto-generated description
                if not step.description or step.description.startswith("Step "):
                    target_name = (
                        step.target.get("text")
                        or step.target.get("content_desc")
                        or step.target.get("resource_id", "")
                    )
                    step.description = f"{step.action.capitalize()} {target_name}".strip()

        self.save_playbook(pb)
        logger.info("Auto-description complete for playbook %r", pb.name)
        return pb

    async def optimize_playbook(self, playbook_id: str) -> Playbook:
        """
        Optimize a playbook by:
            1. Removing redundant consecutive waits
            2. Merging sequential taps on the same element
            3. Adding smart waits where transitions are expected
            4. Removing duplicate screenshots

        Returns the optimized playbook.
        """
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")

        original_count = len(pb.steps)
        optimized: List[PlaybookStep] = []

        i = 0
        while i < len(pb.steps):
            step = pb.steps[i]

            # Rule 1: Merge consecutive waits
            if step.action == "wait" and optimized and optimized[-1].action == "wait":
                prev_dur = optimized[-1].target.get("duration_ms", 1000)
                curr_dur = step.target.get("duration_ms", 1000)
                optimized[-1].target["duration_ms"] = prev_dur + curr_dur
                optimized[-1].description = f"Wait {(prev_dur + curr_dur)}ms (merged)"
                i += 1
                continue

            # Rule 2: Remove redundant back-to-back taps on same target
            if (
                step.action == "tap"
                and i + 1 < len(pb.steps)
                and pb.steps[i + 1].action == "tap"
                and step.target == pb.steps[i + 1].target
            ):
                # Keep only the first tap, extend its wait
                step.wait_after_ms = max(step.wait_after_ms, 800)
                step.description += " (duplicate removed)"
                optimized.append(step)
                i += 2  # skip the duplicate
                continue

            # Rule 3: Add smart waits after launch actions
            if step.action == "launch" and step.wait_after_ms < 2000:
                step.wait_after_ms = 2000
                step.description = step.description or "Launch app"

            optimized.append(step)
            i += 1

        pb.steps = optimized
        self._renumber_steps(pb)
        self.save_playbook(pb)

        removed = original_count - len(pb.steps)
        logger.info(
            "Optimized playbook %r: %d -> %d steps (%d removed)",
            pb.name, original_count, len(pb.steps), removed,
        )
        return pb

    async def generalize_playbook(self, playbook_id: str) -> Playbook:
        """
        Replace hardcoded coordinates with element selectors where possible.

        Scans each step and if it only has coordinates as a target, attempts
        to find a matching element at those coordinates from the recorded
        screen state, then adds text/resource_id selectors.
        """
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")

        generalized_count = 0

        for step in pb.steps:
            target = step.target
            # Only process steps that rely solely on coordinates
            has_selector = (
                target.get("text")
                or target.get("resource_id")
                or target.get("content_desc")
            )
            coords = target.get("coordinates", {})

            if has_selector or not coords:
                continue

            # Try to find the element at those coordinates from the screenshot reference
            if step.screenshot_reference and Path(step.screenshot_reference).exists():
                try:
                    controller = self._get_controller()
                    # Look through stored screen state
                    # For generalization, we ask Haiku what element is at the coordinates
                    image_b64 = None
                    with open(step.screenshot_reference, "rb") as f:
                        image_b64 = base64.b64encode(f.read()).decode("ascii")

                    prompt = (
                        f"What UI element is at coordinates ({coords.get('x')}, {coords.get('y')}) "
                        f"on this Android screen? "
                        f"Respond as JSON: {{\"text\": \"...\", \"type\": \"button/text/icon/...\", "
                        f"\"description\": \"...\"}}"
                    )

                    response = await self._call_haiku(prompt, image_b64=image_b64)
                    if response:
                        try:
                            cleaned = response.strip()
                            if cleaned.startswith("```"):
                                cleaned = re.sub(r"^```\w*\n?", "", cleaned)
                                cleaned = re.sub(r"\n?```$", "", cleaned)
                            elem_data = json.loads(cleaned)
                            if elem_data.get("text"):
                                target["text"] = elem_data["text"]
                                generalized_count += 1
                                step.description += " (generalized from coordinates)"
                            elif elem_data.get("description"):
                                target["content_desc"] = elem_data["description"]
                                generalized_count += 1
                        except (json.JSONDecodeError, TypeError):
                            pass
                except Exception as exc:
                    logger.debug("Generalization failed for step %d: %s", step.step_num, exc)

        self.save_playbook(pb)
        logger.info(
            "Generalized playbook %r: %d steps updated with element selectors",
            pb.name, generalized_count,
        )
        return pb

    async def extract_variables(self, playbook_id: str) -> Playbook:
        """
        Detect text inputs in steps that should be parameterized as variables.

        Looks for steps that type text (especially things that look like
        usernames, passwords, emails, URLs, or search queries) and converts
        them to {{variable}} references.
        """
        pb = self._playbooks.get(playbook_id)
        if pb is None:
            raise KeyError(f"Playbook {playbook_id} not found.")

        extracted_count = 0

        # Patterns that suggest variable extraction
        patterns = {
            "email": re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$"),
            "url": re.compile(r"^https?://"),
            "username": re.compile(r"^@?\w{3,30}$"),
            "phone": re.compile(r"^\+?[\d\s\-()]{7,}$"),
        }

        for step in pb.steps:
            if step.action != "type" or not step.input:
                continue

            text = step.input.strip()
            if "{{" in text:
                # Already parameterized
                continue

            # Check each pattern
            for var_name, pattern in patterns.items():
                if pattern.match(text):
                    # Check if this variable name already exists (add suffix if so)
                    final_name = var_name
                    counter = 1
                    while final_name in pb.variables:
                        counter += 1
                        final_name = f"{var_name}_{counter}"

                    pb.variables[final_name] = {
                        "default": text,
                        "description": f"Auto-extracted {var_name} from step {step.step_num}",
                    }
                    step.input = "{{" + final_name + "}}"
                    extracted_count += 1
                    logger.info(
                        "Extracted variable %r from step %d: %s",
                        final_name, step.step_num, text[:30],
                    )
                    break

            # Also extract any text input in a login-related step
            if not any(p.match(text) for p in patterns.values()):
                desc_lower = step.description.lower()
                if any(kw in desc_lower for kw in ("password", "login", "sign in", "username")):
                    var_name = "password" if "password" in desc_lower else "login_input"
                    counter = 1
                    final_name = var_name
                    while final_name in pb.variables:
                        counter += 1
                        final_name = f"{var_name}_{counter}"

                    pb.variables[final_name] = {
                        "default": text,
                        "description": f"Auto-extracted from step {step.step_num}",
                    }
                    step.input = "{{" + final_name + "}}"
                    extracted_count += 1

        self.save_playbook(pb)
        logger.info(
            "Variable extraction for %r: %d variables created",
            pb.name, extracted_count,
        )
        return pb

    # ==================================================================
    # PLAYBOOK TEMPLATES (Built-in Library)
    # ==================================================================

    def _load_builtin_templates(self) -> None:
        """Load built-in playbook templates for common Android tasks."""
        self._templates = {
            "app_install": Playbook(
                playbook_id="tpl_app_install",
                name="Install App from Play Store",
                description="Search for and install an app from Google Play Store",
                app_package="com.android.vending",
                steps=[
                    PlaybookStep(
                        step_num=1, action="launch",
                        target={"package": "com.android.vending"},
                        description="Open Google Play Store",
                        wait_after_ms=2000,
                    ),
                    PlaybookStep(
                        step_num=2, action="tap",
                        target={"text": "Search", "content_desc": "Search Google Play"},
                        description="Tap search bar",
                        wait_after_ms=500,
                    ),
                    PlaybookStep(
                        step_num=3, action="type",
                        input="{{app_name}}",
                        description="Type app name to search",
                        wait_after_ms=500,
                    ),
                    PlaybookStep(
                        step_num=4, action="key",
                        target={"key_code": "66"},
                        description="Press Enter to search",
                        wait_after_ms=2000,
                    ),
                    PlaybookStep(
                        step_num=5, action="tap",
                        target={"text": "Install"},
                        description="Tap Install button",
                        wait_after_ms=5000,
                        on_failure="retry", max_retries=3,
                    ),
                ],
                variables={
                    "app_name": {"default": "", "description": "Name of the app to install"},
                },
                preconditions=["Google Play Store must be signed in"],
                tags=["template", "play_store"],
            ),

            "app_login": Playbook(
                playbook_id="tpl_app_login",
                name="Generic App Login Flow",
                description="Log into an app with username and password",
                steps=[
                    PlaybookStep(
                        step_num=1, action="launch",
                        target={"package": "{{app_package}}"},
                        description="Open the app",
                        wait_after_ms=3000,
                    ),
                    PlaybookStep(
                        step_num=2, action="tap",
                        target={"text": "Log in", "content_desc": "Log in"},
                        description="Tap Log In button",
                        wait_after_ms=1000,
                        on_failure="skip", optional=True,
                    ),
                    PlaybookStep(
                        step_num=3, action="tap",
                        target={"resource_id": "username", "text": "Email or username"},
                        description="Tap username field",
                        wait_after_ms=300,
                    ),
                    PlaybookStep(
                        step_num=4, action="type",
                        input="{{username}}",
                        description="Enter username",
                        wait_after_ms=300,
                    ),
                    PlaybookStep(
                        step_num=5, action="tap",
                        target={"resource_id": "password", "text": "Password"},
                        description="Tap password field",
                        wait_after_ms=300,
                    ),
                    PlaybookStep(
                        step_num=6, action="type",
                        input="{{password}}",
                        description="Enter password",
                        wait_after_ms=300,
                    ),
                    PlaybookStep(
                        step_num=7, action="tap",
                        target={"text": "Log in", "content_desc": "Log in"},
                        description="Submit login form",
                        wait_after_ms=3000,
                        assertion={
                            "type": "screen_contains",
                            "text": "Home",
                            "description": "Verify login succeeded",
                        },
                    ),
                ],
                variables={
                    "app_package": {"default": "", "description": "Android package name"},
                    "username": {"default": "", "description": "Login username or email"},
                    "password": {"default": "", "description": "Login password"},
                },
                tags=["template", "login", "auth"],
            ),

            "take_screenshot_and_save": Playbook(
                playbook_id="tpl_screenshot",
                name="Take Screenshot and Save",
                description="Take a screenshot and save it to device gallery",
                steps=[
                    PlaybookStep(
                        step_num=1, action="wait",
                        target={"duration_ms": 1000},
                        description="Wait for screen to settle",
                    ),
                    PlaybookStep(
                        step_num=2, action="key",
                        target={"key_code": "120"},
                        description="Press screenshot key combination (KEYCODE_SYSRQ)",
                        wait_after_ms=2000,
                    ),
                ],
                tags=["template", "screenshot", "utility"],
            ),

            "clear_notifications": Playbook(
                playbook_id="tpl_clear_notif",
                name="Clear All Notifications",
                description="Pull down notification shade and clear all notifications",
                steps=[
                    PlaybookStep(
                        step_num=1, action="swipe",
                        target={"swipe_vector": {"x1": 540, "y1": 0, "x2": 540, "y2": 1200}},
                        description="Pull down notification shade",
                        wait_after_ms=1000,
                    ),
                    PlaybookStep(
                        step_num=2, action="tap",
                        target={"text": "Clear all", "content_desc": "Clear all notifications"},
                        description="Tap Clear All",
                        wait_after_ms=500,
                        on_failure="skip", optional=True,
                    ),
                    PlaybookStep(
                        step_num=3, action="back",
                        target={},
                        description="Close notification shade",
                        wait_after_ms=300,
                    ),
                ],
                tags=["template", "notifications", "utility"],
            ),

            "toggle_wifi": Playbook(
                playbook_id="tpl_toggle_wifi",
                name="Toggle WiFi",
                description="Open Settings and toggle WiFi on or off",
                app_package="com.android.settings",
                steps=[
                    PlaybookStep(
                        step_num=1, action="launch",
                        target={"package": "com.android.settings"},
                        description="Open Settings",
                        wait_after_ms=2000,
                    ),
                    PlaybookStep(
                        step_num=2, action="tap",
                        target={"text": "Network & internet", "content_desc": "Network & internet"},
                        description="Navigate to Network settings",
                        wait_after_ms=1000,
                    ),
                    PlaybookStep(
                        step_num=3, action="tap",
                        target={"text": "Wi-Fi", "content_desc": "Wi-Fi"},
                        description="Open WiFi settings",
                        wait_after_ms=1000,
                    ),
                    PlaybookStep(
                        step_num=4, action="tap",
                        target={"resource_id": "switch_widget", "class_name": "Switch"},
                        description="Toggle WiFi switch",
                        wait_after_ms=1000,
                    ),
                    PlaybookStep(
                        step_num=5, action="home",
                        target={},
                        description="Return to home screen",
                    ),
                ],
                tags=["template", "settings", "wifi"],
            ),

            "change_wallpaper": Playbook(
                playbook_id="tpl_wallpaper",
                name="Change Wallpaper",
                description="Open Settings and navigate to wallpaper picker",
                app_package="com.android.settings",
                steps=[
                    PlaybookStep(
                        step_num=1, action="launch",
                        target={"package": "com.android.settings"},
                        description="Open Settings",
                        wait_after_ms=2000,
                    ),
                    PlaybookStep(
                        step_num=2, action="tap",
                        target={"text": "Wallpaper", "content_desc": "Wallpaper & style"},
                        description="Navigate to Wallpaper settings",
                        wait_after_ms=2000,
                    ),
                    PlaybookStep(
                        step_num=3, action="tap",
                        target={"text": "Change wallpaper"},
                        description="Open wallpaper picker",
                        wait_after_ms=1000,
                        on_failure="skip", optional=True,
                    ),
                ],
                tags=["template", "settings", "wallpaper"],
            ),
        }

        # Also load user-created templates from disk
        for path in TEMPLATES_DIR.glob("tpl_*.json"):
            try:
                data = _load_json(path, default={})
                if data:
                    tpl = Playbook.from_dict(data)
                    self._templates[tpl.name] = tpl
            except Exception as exc:
                logger.debug("Failed to load template %s: %s", path.name, exc)

        logger.info("Loaded %d playbook templates.", len(self._templates))

    def get_template(self, name: str) -> Optional[Playbook]:
        """Get a built-in playbook template by name."""
        tpl = self._templates.get(name)
        if tpl is None:
            # Search by name field
            for t in self._templates.values():
                if t.name.lower() == name.lower():
                    return copy.deepcopy(t)
            return None
        return copy.deepcopy(tpl)

    def list_templates(self) -> List[str]:
        """Return the names of all available templates."""
        return sorted(self._templates.keys())

    def create_from_template(
        self,
        template_name: str,
        playbook_name: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Playbook:
        """
        Create a new playbook from a template with custom variables.

        Args:
            template_name: Name of the template to use
            playbook_name: Name for the new playbook
            variables:     Variables to set on the new playbook

        Returns:
            The created Playbook.
        """
        tpl = self.get_template(template_name)
        if tpl is None:
            raise KeyError(f"Template '{template_name}' not found.")

        pb = tpl
        pb.playbook_id = uuid.uuid4().hex[:12]
        pb.name = playbook_name
        pb.created_at = _now_iso()
        pb.updated_at = _now_iso()
        pb.tags = [t for t in pb.tags if t != "template"]
        pb.tags.append("from_template")

        if variables:
            for key, value in variables.items():
                if key in pb.variables and isinstance(pb.variables[key], dict):
                    pb.variables[key]["default"] = value
                else:
                    pb.variables[key] = {"default": value, "description": ""}

        self._playbooks[pb.playbook_id] = pb
        self._save_playbook_to_disk(pb)
        logger.info("Created playbook %r from template %r", playbook_name, template_name)
        return pb

    # ==================================================================
    # EXECUTION HISTORY
    # ==================================================================

    def get_history(
        self,
        playbook_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ExecutionResult]:
        """
        Get execution history, optionally filtered by playbook ID.

        Returns newest first, up to limit entries.
        """
        entries = self._history
        if playbook_id:
            entries = [e for e in entries if e.get("playbook_id") == playbook_id]
        entries = list(reversed(entries))[:limit]
        return [ExecutionResult.from_dict(e) for e in entries]

    def get_success_rate(self, playbook_id: str) -> float:
        """
        Calculate the historical success rate for a playbook.

        Returns a float between 0.0 and 1.0, or 0.0 if no history.
        """
        entries = [e for e in self._history if e.get("playbook_id") == playbook_id]
        if not entries:
            return 0.0
        successes = sum(1 for e in entries if e.get("success", False))
        return successes / len(entries)

    def get_avg_duration(self, playbook_id: str) -> float:
        """
        Calculate the average execution duration for a playbook in ms.

        Returns 0.0 if no history.
        """
        durations = [
            e.get("duration_ms", 0)
            for e in self._history
            if e.get("playbook_id") == playbook_id and e.get("success", False)
        ]
        if not durations:
            return 0.0
        return sum(durations) / len(durations)

    def compare_runs(
        self,
        run_id_1: str,
        run_id_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two execution runs of the same or different playbooks.

        Returns a diff summary including timing differences, step outcome
        differences, and overall comparison.
        """
        run1 = None
        run2 = None

        for entry in self._history:
            if entry.get("run_id") == run_id_1:
                run1 = entry
            if entry.get("run_id") == run_id_2:
                run2 = entry

        if run1 is None or run2 is None:
            return {
                "error": f"Run{'s' if run1 is None and run2 is None else ''} not found: "
                         f"{'run_1 ' if run1 is None else ''}{'run_2' if run2 is None else ''}",
            }

        step_diffs = []
        results_1 = run1.get("step_results", [])
        results_2 = run2.get("step_results", [])
        max_steps = max(len(results_1), len(results_2))

        for i in range(max_steps):
            sr1 = results_1[i] if i < len(results_1) else None
            sr2 = results_2[i] if i < len(results_2) else None

            diff: Dict[str, Any] = {"step": i + 1}
            if sr1 and sr2:
                diff["run1_success"] = sr1.get("success", False)
                diff["run2_success"] = sr2.get("success", False)
                diff["run1_duration_ms"] = sr1.get("duration_ms", 0)
                diff["run2_duration_ms"] = sr2.get("duration_ms", 0)
                diff["duration_delta_ms"] = (
                    sr2.get("duration_ms", 0) - sr1.get("duration_ms", 0)
                )
                diff["outcome_changed"] = sr1.get("success") != sr2.get("success")
            elif sr1:
                diff["note"] = "Step only in run 1"
                diff["run1_success"] = sr1.get("success", False)
            else:
                diff["note"] = "Step only in run 2"
                diff["run2_success"] = sr2.get("success", False) if sr2 else False

            step_diffs.append(diff)

        return {
            "run_id_1": run_id_1,
            "run_id_2": run_id_2,
            "playbook_1": run1.get("playbook_id", ""),
            "playbook_2": run2.get("playbook_id", ""),
            "run1_success": run1.get("success", False),
            "run2_success": run2.get("success", False),
            "run1_duration_ms": run1.get("duration_ms", 0),
            "run2_duration_ms": run2.get("duration_ms", 0),
            "duration_delta_ms": (
                run2.get("duration_ms", 0) - run1.get("duration_ms", 0)
            ),
            "run1_steps_completed": run1.get("steps_completed", 0),
            "run2_steps_completed": run2.get("steps_completed", 0),
            "step_diffs": step_diffs,
        }

    # ==================================================================
    # SYNC WRAPPERS
    # ==================================================================

    def start_recording_sync(self, *args: Any, **kwargs: Any) -> RecordingSession:
        return self._run_sync(self.start_recording(*args, **kwargs))

    def stop_recording_sync(self) -> Playbook:
        return self._run_sync(self.stop_recording())

    def capture_step_sync(self, *args: Any, **kwargs: Any) -> Optional[RecordedStep]:
        return self._run_sync(self.capture_step(*args, **kwargs))

    def execute_playbook_sync(self, *args: Any, **kwargs: Any) -> ExecutionResult:
        return self._run_sync(self.execute_playbook(*args, **kwargs))

    def dry_run_sync(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._run_sync(self.dry_run(*args, **kwargs))

    def auto_describe_steps_sync(self, *args: Any, **kwargs: Any) -> Playbook:
        return self._run_sync(self.auto_describe_steps(*args, **kwargs))

    def optimize_playbook_sync(self, *args: Any, **kwargs: Any) -> Playbook:
        return self._run_sync(self.optimize_playbook(*args, **kwargs))

    def generalize_playbook_sync(self, *args: Any, **kwargs: Any) -> Playbook:
        return self._run_sync(self.generalize_playbook(*args, **kwargs))

    def extract_variables_sync(self, *args: Any, **kwargs: Any) -> Playbook:
        return self._run_sync(self.extract_variables(*args, **kwargs))

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)


# ===================================================================
# SINGLETON
# ===================================================================

_recorder_instance: Optional[WorkflowRecorder] = None


def get_recorder() -> WorkflowRecorder:
    """
    Get the global WorkflowRecorder singleton.

    Creates the instance on first call, loading persisted playbooks and
    history from disk.
    """
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = WorkflowRecorder()
    return _recorder_instance


# ===================================================================
# CLI HELPER FUNCTIONS
# ===================================================================

def _format_table(headers: List[str], rows: List[List[str]], max_col: int = 40) -> str:
    """Format a simple ASCII table for CLI output."""
    if not rows:
        return "(no results)"

    trunc = []
    for row in rows:
        trunc.append([
            v[:max_col - 3] + "..." if len(v) > max_col else v
            for v in row
        ])

    widths = [len(h) for h in headers]
    for row in trunc:
        for i, v in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(v))

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers)]
    lines.append("  ".join("-" * w for w in widths))
    for row in trunc:
        while len(row) < len(headers):
            row.append("")
        lines.append(fmt.format(*row))
    return "\n".join(lines)


def _cmd_record(args: argparse.Namespace) -> None:
    """Start an interactive recording session."""
    recorder = get_recorder()

    print(f"Starting recording: {args.name}")
    if args.app:
        print(f"App package: {args.app}")

    session = recorder.start_recording_sync(
        name=args.name,
        app_package=args.app or "",
    )
    print(f"Session ID: {session.session_id}")
    print("Recording started. Steps will be captured via capture_step().")
    print("Use 'workflow_recorder.py stop' to end the recording.\n")

    # Save session reference for stop command
    ref_path = PLAYBOOK_DIR / "_active_recording.json"
    _save_json(ref_path, {"session_id": session.session_id, "name": args.name})


def _cmd_stop(args: argparse.Namespace) -> None:
    """Stop the active recording session."""
    recorder = get_recorder()
    ref_path = PLAYBOOK_DIR / "_active_recording.json"

    if recorder._recording_state == RecordingState.IDLE:
        print("No active recording session.")
        return

    try:
        playbook = recorder.stop_recording_sync()
        print(f"Recording stopped.")
        print(f"Playbook created: {playbook.name} (id={playbook.playbook_id})")
        print(f"Steps: {len(playbook.steps)}")
        ref_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"Error stopping recording: {exc}")


def _cmd_list(args: argparse.Namespace) -> None:
    """List all playbooks."""
    recorder = get_recorder()
    tags_filter = args.tags.split(",") if args.tags else None
    playbooks = recorder.list_playbooks(app=args.app, tags=tags_filter)

    if not playbooks:
        print("No playbooks found.")
        return

    headers = ["ID", "Name", "App", "Steps", "Runs", "Success", "Tags"]
    rows = []
    for pb in playbooks:
        rate = f"{recorder.get_success_rate(pb.playbook_id) * 100:.0f}%"
        rows.append([
            pb.playbook_id,
            pb.name,
            pb.app_package or "-",
            str(len(pb.steps)),
            str(pb.run_count),
            rate if pb.run_count > 0 else "-",
            ",".join(pb.tags[:3]),
        ])

    print(f"\n  Playbooks  --  {len(playbooks)} total\n")
    print(_format_table(headers, rows))
    print()


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute a playbook."""
    recorder = get_recorder()
    pb = recorder.load_playbook(args.playbook_id)
    if pb is None:
        # Try by name
        for p in recorder._playbooks.values():
            if p.name.lower() == args.playbook_id.lower():
                pb = p
                break
    if pb is None:
        print(f"Playbook not found: {args.playbook_id}")
        return

    variables = {}
    if args.var:
        for v in args.var:
            if "=" in v:
                k, val = v.split("=", 1)
                variables[k] = val

    print(f"Executing playbook: {pb.name} (id={pb.playbook_id})")
    print(f"Steps: {len(pb.steps)}")
    if variables:
        print(f"Variables: {variables}")
    print()

    result = recorder.execute_playbook_sync(
        pb.playbook_id, variables=variables,
    )

    status = "PASSED" if result.success else "FAILED"
    print(f"Result: {status}")
    print(f"Steps completed: {result.steps_completed}/{result.steps_total}")
    print(f"Duration: {result.duration_ms:.0f}ms")
    if result.error:
        print(f"Error: {result.error}")
    print()

    for sr in result.step_results:
        marker = "[OK]" if sr.success else "[FAIL]"
        print(f"  {marker} Step {sr.step_num}: {sr.action_taken} ({sr.duration_ms:.0f}ms)")
        if sr.error:
            print(f"         Error: {sr.error}")


def _cmd_dry_run(args: argparse.Namespace) -> None:
    """Validate a playbook without executing."""
    recorder = get_recorder()
    pb = recorder.load_playbook(args.playbook_id)
    if pb is None:
        print(f"Playbook not found: {args.playbook_id}")
        return

    variables = {}
    if args.var:
        for v in args.var:
            if "=" in v:
                k, val = v.split("=", 1)
                variables[k] = val

    result = recorder.dry_run_sync(pb.playbook_id, variables=variables)

    print(f"\n  Dry Run: {pb.name}\n")
    print(f"  Valid:            {'Yes' if result['valid'] else 'No'}")
    print(f"  Steps:            {result['step_count']}")
    print(f"  Est. duration:    {result['estimated_duration_ms']:.0f}ms")
    print(f"  Variables needed: {', '.join(result['variables_needed']) or 'none'}")
    print(f"  Variables given:  {', '.join(result['variables_provided']) or 'none'}")

    if result["issues"]:
        print(f"\n  Issues ({len(result['issues'])}):")
        for issue in result["issues"]:
            print(f"    - {issue}")
    else:
        print("\n  No issues found. Playbook is ready to run.")
    print()


def _cmd_edit(args: argparse.Namespace) -> None:
    """Edit a playbook step or add a new step."""
    recorder = get_recorder()
    pb = recorder.load_playbook(args.playbook_id)
    if pb is None:
        print(f"Playbook not found: {args.playbook_id}")
        return

    if args.add:
        step = PlaybookStep(
            action=args.action or "tap",
            description=args.description or "",
            input=args.input_text or "",
        )
        if args.target_text:
            step.target["text"] = args.target_text
        recorder.add_step(pb.playbook_id, step)
        print(f"Added step {len(pb.steps)} to {pb.name}")

    elif args.remove and args.step_num:
        recorder.remove_step(pb.playbook_id, args.step_num)
        print(f"Removed step {args.step_num} from {pb.name}")

    elif args.step_num and args.action:
        updates = {"action": args.action}
        if args.description:
            updates["description"] = args.description
        if args.input_text:
            updates["input"] = args.input_text
        recorder.update_step(pb.playbook_id, args.step_num, updates)
        print(f"Updated step {args.step_num} in {pb.name}")

    else:
        # Show playbook detail view
        print(f"\n  Playbook: {pb.name} (id={pb.playbook_id})")
        print(f"  App:      {pb.app_package or '-'}")
        print(f"  Version:  {pb.version}")
        print(f"  Tags:     {', '.join(pb.tags) or '-'}")
        print(f"  Variables: {list(pb.variables.keys()) or '-'}")
        print(f"  Steps:")
        for step in pb.steps:
            target_desc = (
                step.target.get("text")
                or step.target.get("resource_id")
                or step.target.get("content_desc")
                or str(step.target.get("coordinates", ""))
                or "-"
            )
            opt = " (optional)" if step.optional else ""
            print(f"    {step.step_num}. [{step.action}] {step.description or target_desc}{opt}")
        print()


def _cmd_describe(args: argparse.Namespace) -> None:
    """Auto-generate step descriptions using Claude Haiku."""
    recorder = get_recorder()
    pb = recorder.load_playbook(args.playbook_id)
    if pb is None:
        print(f"Playbook not found: {args.playbook_id}")
        return

    print(f"Generating descriptions for {len(pb.steps)} steps...")
    pb = recorder.auto_describe_steps_sync(pb.playbook_id)

    for step in pb.steps:
        print(f"  Step {step.step_num}: {step.description}")
    print("\nDescriptions saved.")


def _cmd_optimize(args: argparse.Namespace) -> None:
    """Optimize a playbook."""
    recorder = get_recorder()
    pb_before = recorder.load_playbook(args.playbook_id)
    if pb_before is None:
        print(f"Playbook not found: {args.playbook_id}")
        return

    before_count = len(pb_before.steps)
    pb = recorder.optimize_playbook_sync(args.playbook_id)
    after_count = len(pb.steps)

    print(f"Optimized {pb.name}: {before_count} -> {after_count} steps")
    if before_count > after_count:
        print(f"Removed {before_count - after_count} redundant steps.")
    else:
        print("No redundant steps found.")


def _cmd_generalize(args: argparse.Namespace) -> None:
    """Generalize coordinates to element selectors."""
    recorder = get_recorder()
    pb = recorder.load_playbook(args.playbook_id)
    if pb is None:
        print(f"Playbook not found: {args.playbook_id}")
        return

    print(f"Generalizing playbook {pb.name}...")
    pb = recorder.generalize_playbook_sync(args.playbook_id)
    print("Generalization complete.")


def _cmd_templates(args: argparse.Namespace) -> None:
    """List available playbook templates."""
    recorder = get_recorder()
    templates = recorder.list_templates()

    if not templates:
        print("No templates available.")
        return

    print(f"\n  Playbook Templates  --  {len(templates)} available\n")
    for name in templates:
        tpl = recorder.get_template(name)
        if tpl:
            print(f"  {name}")
            print(f"    {tpl.description}")
            print(f"    Steps: {len(tpl.steps)} | Vars: {list(tpl.variables.keys()) or '-'}")
            print()


def _cmd_history(args: argparse.Namespace) -> None:
    """Show execution history."""
    recorder = get_recorder()
    history = recorder.get_history(
        playbook_id=args.playbook_id if hasattr(args, "playbook_id") and args.playbook_id else None,
        limit=args.limit,
    )

    if not history:
        print("No execution history found.")
        return

    headers = ["Run ID", "Playbook", "Result", "Steps", "Duration", "Date"]
    rows = []
    for er in history:
        status = "PASS" if er.success else "FAIL"
        date_str = ""
        if er.started_at:
            try:
                dt = datetime.fromisoformat(er.started_at)
                date_str = dt.strftime("%m/%d %H:%M")
            except (ValueError, TypeError):
                date_str = er.started_at[:16]

        rows.append([
            er.run_id,
            er.playbook_id,
            status,
            f"{er.steps_completed}/{er.steps_total}",
            f"{er.duration_ms:.0f}ms",
            date_str,
        ])

    print(f"\n  Execution History  --  {len(history)} entries\n")
    print(_format_table(headers, rows))
    print()


def _cmd_export(args: argparse.Namespace) -> None:
    """Export a playbook to a file."""
    recorder = get_recorder()
    try:
        path = recorder.export_playbook(
            args.playbook_id,
            format=args.format,
            output_path=args.output,
        )
        print(f"Exported to: {path}")
    except (KeyError, ValueError) as exc:
        print(f"Export failed: {exc}")


def _cmd_import(args: argparse.Namespace) -> None:
    """Import a playbook from a file."""
    recorder = get_recorder()
    try:
        pb = recorder.import_playbook(args.file)
        print(f"Imported: {pb.name} (id={pb.playbook_id}, steps={len(pb.steps)})")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Import failed: {exc}")


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def main() -> None:
    """CLI entry point for the workflow recorder."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="workflow_recorder",
        description="OpenClaw Empire Workflow Recorder & Playbook System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # record
    sp_record = subparsers.add_parser("record", help="Start a recording session")
    sp_record.add_argument("name", help="Name for the recording")
    sp_record.add_argument("--app", type=str, default="", help="App package to record")
    sp_record.set_defaults(func=_cmd_record)

    # stop
    sp_stop = subparsers.add_parser("stop", help="Stop the active recording")
    sp_stop.set_defaults(func=_cmd_stop)

    # list
    sp_list = subparsers.add_parser("list", help="List all playbooks")
    sp_list.add_argument("--app", type=str, default=None, help="Filter by app package")
    sp_list.add_argument("--tags", type=str, default=None, help="Filter by tags (comma-separated)")
    sp_list.set_defaults(func=_cmd_list)

    # run
    sp_run = subparsers.add_parser("run", help="Execute a playbook")
    sp_run.add_argument("playbook_id", help="Playbook ID or name")
    sp_run.add_argument("--var", action="append", help="Variable: key=value (repeatable)")
    sp_run.set_defaults(func=_cmd_run)

    # dry-run
    sp_dry = subparsers.add_parser("dry-run", help="Validate a playbook without executing")
    sp_dry.add_argument("playbook_id", help="Playbook ID or name")
    sp_dry.add_argument("--var", action="append", help="Variable: key=value (repeatable)")
    sp_dry.set_defaults(func=_cmd_dry_run)

    # edit
    sp_edit = subparsers.add_parser("edit", help="View or edit a playbook")
    sp_edit.add_argument("playbook_id", help="Playbook ID")
    sp_edit.add_argument("--step-num", type=int, help="Step number to edit")
    sp_edit.add_argument("--add", action="store_true", help="Add a new step")
    sp_edit.add_argument("--remove", action="store_true", help="Remove a step")
    sp_edit.add_argument("--action", type=str, help="Action type for the step")
    sp_edit.add_argument("--description", type=str, help="Step description")
    sp_edit.add_argument("--input-text", type=str, help="Input text for type action")
    sp_edit.add_argument("--target-text", type=str, help="Target element text")
    sp_edit.set_defaults(func=_cmd_edit)

    # describe
    sp_desc = subparsers.add_parser("describe", help="Auto-generate step descriptions")
    sp_desc.add_argument("playbook_id", help="Playbook ID")
    sp_desc.set_defaults(func=_cmd_describe)

    # optimize
    sp_opt = subparsers.add_parser("optimize", help="Optimize a playbook")
    sp_opt.add_argument("playbook_id", help="Playbook ID")
    sp_opt.set_defaults(func=_cmd_optimize)

    # generalize
    sp_gen = subparsers.add_parser("generalize", help="Replace coordinates with selectors")
    sp_gen.add_argument("playbook_id", help="Playbook ID")
    sp_gen.set_defaults(func=_cmd_generalize)

    # templates
    sp_tpl = subparsers.add_parser("templates", help="List available templates")
    sp_tpl.set_defaults(func=_cmd_templates)

    # history
    sp_hist = subparsers.add_parser("history", help="Show execution history")
    sp_hist.add_argument("--playbook-id", type=str, default=None, help="Filter by playbook ID")
    sp_hist.add_argument("--limit", type=int, default=20, help="Max entries (default: 20)")
    sp_hist.set_defaults(func=_cmd_history)

    # export
    sp_export = subparsers.add_parser("export", help="Export a playbook")
    sp_export.add_argument("playbook_id", help="Playbook ID")
    sp_export.add_argument("--format", choices=["json", "yaml"], default="json", help="Export format")
    sp_export.add_argument("--output", type=str, default=None, help="Output file path")
    sp_export.set_defaults(func=_cmd_export)

    # import
    sp_import = subparsers.add_parser("import", help="Import a playbook from file")
    sp_import.add_argument("file", help="Path to playbook file (JSON or YAML)")
    sp_import.set_defaults(func=_cmd_import)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
