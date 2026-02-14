"""
OpenClaw Empire API Server
===========================

FastAPI server exposing the OpenClaw Empire intelligence system as HTTP
endpoints. Sits between the OpenClaw gateway and all intelligence subsystems
(PhoneController, FORGE, AMPLIFY, Screenpipe, Vision).

Run directly:
    python -m src.api
    uvicorn src.api:app --host 0.0.0.0 --port 8765

Port configurable via OPENCLAW_API_PORT environment variable (default 8765).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.amplify_pipeline import AmplifyPipeline
from src.forge_engine import ForgeEngine
from src.phone_controller import AppNavigator, PhoneController, TaskExecutor, VisionLoop
from src.screenpipe_agent import ScreenpipeAgent
from src.vision_agent import VisionAgent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("openclaw.api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    ))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_PORT = int(os.getenv("OPENCLAW_API_PORT", "8765"))
NODE_URL = os.getenv("OPENCLAW_NODE_URL", "http://localhost:18789")
NODE_NAME = os.getenv("OPENCLAW_ANDROID_NODE", "android")
VISION_URL = os.getenv("VISION_SERVICE_URL", "http://localhost:8002")
SCREENPIPE_URL = os.getenv("SCREENPIPE_URL", "http://localhost:3030")

# ---------------------------------------------------------------------------
# Pydantic Models -- Requests
# ---------------------------------------------------------------------------


class TapRequest(BaseModel):
    x: int
    y: int


class TypeRequest(BaseModel):
    text: str


class SwipeRequest(BaseModel):
    direction: Optional[str] = Field(None, description="up, down, left, right")
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None


class LaunchRequest(BaseModel):
    app_name: str


class TaskExecuteRequest(BaseModel):
    task_description: str
    app: Optional[str] = None
    confirm_irreversible: bool = False


class TaskAnalyzeRequest(BaseModel):
    task_description: str
    app: Optional[str] = None


class TaskCompleteRequest(BaseModel):
    success: bool
    duration: Optional[float] = None
    error: Optional[str] = None


class ForgePreFlightRequest(BaseModel):
    phone_state: Dict[str, Any]
    task: Dict[str, Any]


class ForgeLearnRequest(BaseModel):
    task_id: str
    outcome: str
    duration: Optional[float] = None
    error: Optional[str] = None


class VisionPromptRequest(BaseModel):
    template: str
    context: Dict[str, Any] = Field(default_factory=dict)


class AmplifyRecordRequest(BaseModel):
    action_type: str
    app_name: str
    duration: float
    success: bool


class MonitorRequest(BaseModel):
    pattern: str
    timeout: Optional[float] = 60.0
    app: Optional[str] = None


class VisionRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None


class FindElementRequest(BaseModel):
    description: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None


# ---------------------------------------------------------------------------
# Pydantic Models -- Responses
# ---------------------------------------------------------------------------


class StatusResponse(BaseModel):
    status: str
    timestamp: str
    subsystems: Dict[str, str] = Field(default_factory=dict)
    version: str = "1.0.0"


class ActionResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


class TaskResponse(BaseModel):
    task_id: str
    status: str
    description: str
    steps_total: int = 0
    steps_completed: int = 0
    data: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------


class AppState:
    """Holds references to all subsystem singletons."""

    def __init__(self) -> None:
        self.controller: Optional[PhoneController] = None
        self.executor: Optional[TaskExecutor] = None
        self.navigator: Optional[AppNavigator] = None
        self.vision_loop: Optional[VisionLoop] = None
        self.forge: Optional[ForgeEngine] = None
        self.amplify: Optional[AmplifyPipeline] = None
        self.screenpipe: Optional[ScreenpipeAgent] = None
        self.vision: Optional[VisionAgent] = None
        self.running_tasks: Dict[str, Dict[str, Any]] = {}
        self.start_time: float = 0.0


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize subsystems on startup, tear down on shutdown."""
    logger.info("Starting OpenClaw Empire API on port %d", API_PORT)
    state.start_time = time.monotonic()

    # Phone controller
    state.controller = PhoneController(node_url=NODE_URL, node_name=NODE_NAME)
    state.navigator = AppNavigator()
    state.vision_loop = VisionLoop(state.controller)

    # Intelligence engines
    state.forge = ForgeEngine()
    state.amplify = AmplifyPipeline()

    # Task executor with FORGE + AMPLIFY integration
    state.executor = TaskExecutor(
        controller=state.controller,
        vision=state.vision_loop,
        navigator=state.navigator,
        forge_engine=state.forge,
        amplify_pipeline=state.amplify,
    )

    # Passive monitoring + vision
    state.screenpipe = ScreenpipeAgent(base_url=SCREENPIPE_URL)
    state.vision = VisionAgent(base_url=VISION_URL)

    # Attempt phone connection (non-blocking, failures are OK at startup)
    try:
        connected = await state.controller.connect()
        logger.info("Phone connection: %s", "OK" if connected else "not available")
    except Exception as exc:
        logger.warning("Phone connection failed at startup (non-fatal): %s", exc)

    logger.info("All subsystems initialized")
    yield

    # Shutdown
    logger.info("Shutting down OpenClaw Empire API")
    if state.controller:
        await state.controller.close()
    if state.screenpipe:
        await state.screenpipe.close()
    if state.vision:
        await state.vision.close()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenClaw Empire API",
    description="Intelligence gateway for Android phone automation across 16 WordPress sites.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===================================================================
# Health
# ===================================================================


@app.get("/health", response_model=StatusResponse, tags=["Health"])
async def health():
    """Server health check with subsystem status."""
    subs: Dict[str, str] = {}
    subs["phone"] = "connected" if (state.controller and state.controller.is_connected) else "disconnected"
    subs["forge"] = "ready" if state.forge else "unavailable"
    subs["amplify"] = "ready" if state.amplify else "unavailable"
    subs["screenpipe"] = "configured" if state.screenpipe else "unavailable"
    subs["vision"] = "configured" if state.vision else "unavailable"
    uptime = time.monotonic() - state.start_time if state.start_time else 0
    subs["uptime_seconds"] = f"{uptime:.0f}"
    return StatusResponse(status="ok", timestamp=_now_iso(), subsystems=subs)


@app.get("/stats", tags=["Health"])
async def stats():
    """Full intelligence stats from all subsystems."""
    result: Dict[str, Any] = {"timestamp": _now_iso()}
    if state.forge:
        result["forge"] = state.forge.get_stats()
    if state.executor:
        result["task_history"] = state.executor.get_task_history()
    result["running_tasks"] = len(state.running_tasks)
    return result


# ===================================================================
# Phone Control
# ===================================================================


def _require_controller() -> PhoneController:
    if state.controller is None:
        raise HTTPException(503, "Phone controller not initialized")
    return state.controller


@app.post("/phone/screenshot", tags=["Phone Control"])
async def phone_screenshot():
    """Take a screenshot and return the file path with optional analysis."""
    ctrl = _require_controller()
    start = time.monotonic()
    try:
        path = await ctrl.screenshot()
        analysis = None
        if state.vision:
            try:
                result = await state.vision.analyze_screen(image_path=path)
                analysis = result.raw_response
            except Exception as exc:
                logger.warning("Vision analysis failed: %s", exc)
        elapsed = (time.monotonic() - start) * 1000
        return ActionResponse(
            success=True,
            message=f"Screenshot saved: {path}",
            data={"path": path, "analysis": analysis},
            duration_ms=round(elapsed, 1),
        )
    except Exception as exc:
        raise HTTPException(500, f"Screenshot failed: {exc}")


@app.get("/phone/state", tags=["Phone Control"])
async def phone_state():
    """Get comprehensive phone state (app, resolution, connection)."""
    ctrl = _require_controller()
    try:
        current_app = await ctrl.get_current_app()
        return {
            "connected": ctrl.is_connected,
            "resolution": list(ctrl.resolution),
            "current_app": current_app,
        }
    except Exception as exc:
        raise HTTPException(500, f"Failed to get phone state: {exc}")


@app.post("/phone/tap", response_model=ActionResponse, tags=["Phone Control"])
async def phone_tap(req: TapRequest):
    """Tap at the specified coordinates."""
    ctrl = _require_controller()
    result = await ctrl.tap(req.x, req.y)
    return ActionResponse(
        success=result.success,
        message=result.error or f"Tapped ({req.x}, {req.y})",
        duration_ms=result.duration_ms,
    )


@app.post("/phone/type", response_model=ActionResponse, tags=["Phone Control"])
async def phone_type(req: TypeRequest):
    """Type text into the currently focused field."""
    ctrl = _require_controller()
    result = await ctrl.type_text(req.text)
    return ActionResponse(
        success=result.success,
        message=result.error or f"Typed {len(req.text)} characters",
        duration_ms=result.duration_ms,
    )


@app.post("/phone/swipe", response_model=ActionResponse, tags=["Phone Control"])
async def phone_swipe(req: SwipeRequest):
    """Swipe by direction or explicit coordinates."""
    ctrl = _require_controller()
    if req.direction:
        w, h = ctrl.resolution
        cx, cy = w // 2, h // 2
        dist = h // 3
        directions = {
            "up": (cx, cy + dist, cx, cy - dist),
            "down": (cx, cy - dist, cx, cy + dist),
            "left": (cx + dist, cy, cx - dist, cy),
            "right": (cx - dist, cy, cx + dist, cy),
        }
        coords = directions.get(req.direction.lower())
        if not coords:
            raise HTTPException(400, f"Invalid direction: {req.direction}. Use up/down/left/right.")
        result = await ctrl.swipe(*coords)
    elif req.x1 is not None and req.y1 is not None and req.x2 is not None and req.y2 is not None:
        result = await ctrl.swipe(req.x1, req.y1, req.x2, req.y2)
    else:
        raise HTTPException(400, "Provide 'direction' or all of x1, y1, x2, y2.")
    return ActionResponse(
        success=result.success,
        message=result.error or "Swipe completed",
        duration_ms=result.duration_ms,
    )


@app.post("/phone/launch", response_model=ActionResponse, tags=["Phone Control"])
async def phone_launch(req: LaunchRequest):
    """Launch an app by name."""
    if state.navigator is None:
        raise HTTPException(503, "Navigator not initialized")
    ctrl = _require_controller()
    package = state.navigator.resolve_package(req.app_name)
    if not package:
        raise HTTPException(404, f"Unknown app: {req.app_name}")
    activity = state.navigator.get_launch_activity(package)
    result = await ctrl.launch_app(package, activity)
    return ActionResponse(
        success=result.success,
        message=result.error or f"Launched {req.app_name} ({package})",
        duration_ms=result.duration_ms,
    )


@app.post("/phone/back", response_model=ActionResponse, tags=["Phone Control"])
async def phone_back():
    """Press the back button."""
    result = await _require_controller().press_back()
    return ActionResponse(success=result.success, message=result.error or "Back pressed", duration_ms=result.duration_ms)


@app.post("/phone/home", response_model=ActionResponse, tags=["Phone Control"])
async def phone_home():
    """Press the home button."""
    result = await _require_controller().press_home()
    return ActionResponse(success=result.success, message=result.error or "Home pressed", duration_ms=result.duration_ms)


# ===================================================================
# Task Execution
# ===================================================================


@app.post("/task/execute", response_model=TaskResponse, tags=["Task Execution"])
async def task_execute(req: TaskExecuteRequest):
    """Execute a natural language task on the phone."""
    if state.executor is None:
        raise HTTPException(503, "Task executor not initialized")

    task_id = uuid.uuid4().hex[:12]
    state.running_tasks[task_id] = {
        "description": req.task_description,
        "status": "running",
        "started_at": _now_iso(),
    }

    try:
        plan = await state.executor.execute(req.task_description)
        state.running_tasks[task_id]["status"] = plan.status
        state.running_tasks[task_id]["completed_at"] = _now_iso()
        return TaskResponse(
            task_id=plan.task_id,
            status=plan.status,
            description=plan.task_description,
            steps_total=len(plan.steps),
            steps_completed=sum(1 for s in plan.steps if s.completed),
            data={
                "forge_analysis": plan.forge_analysis,
                "amplify_enhancements": plan.amplify_enhancements,
            },
        )
    except Exception as exc:
        state.running_tasks[task_id]["status"] = "error"
        state.running_tasks[task_id]["error"] = str(exc)
        raise HTTPException(500, f"Task execution failed: {exc}")


@app.post("/task/analyze", response_model=TaskResponse, tags=["Task Execution"])
async def task_analyze(req: TaskAnalyzeRequest):
    """Pre-flight analysis only -- does not execute the task."""
    if state.forge is None:
        raise HTTPException(503, "FORGE engine not initialized")

    task_spec: Dict[str, Any] = {"task_description": req.task_description}
    if req.app:
        task_spec["app"] = req.app

    # Run FORGE pre-flight with mock phone state
    phone_state: Dict[str, Any] = {}
    try:
        ctrl = _require_controller()
        current_app = await ctrl.get_current_app()
        phone_state = {
            "screen_on": True,
            "locked": False,
            "active_app": current_app,
            "wifi_connected": True,
            "battery_percent": 100,
        }
    except Exception:
        phone_state = {"screen_on": True, "locked": False, "wifi_connected": True, "battery_percent": 100}

    result = await state.forge.pre_flight(phone_state, task_spec)
    return TaskResponse(
        task_id=uuid.uuid4().hex[:12],
        status="analyzed",
        description=req.task_description,
        data=result,
    )


@app.get("/task/{task_id}/status", tags=["Task Execution"])
async def task_status(task_id: str):
    """Get the status of a running or completed task."""
    info = state.running_tasks.get(task_id)
    if info is None:
        raise HTTPException(404, f"Task {task_id} not found")
    return {"task_id": task_id, **info}


@app.post("/task/{task_id}/complete", response_model=ActionResponse, tags=["Task Execution"])
async def task_complete(task_id: str, req: TaskCompleteRequest):
    """Record external task completion."""
    if state.forge:
        outcome = "success" if req.success else "failure"
        state.forge.record_outcome(task_id, outcome, req.duration, req.error)

    info = state.running_tasks.get(task_id, {})
    info["status"] = "completed" if req.success else "failed"
    info["completed_at"] = _now_iso()
    state.running_tasks[task_id] = info

    return ActionResponse(success=True, message=f"Task {task_id} marked as {'completed' if req.success else 'failed'}")


# ===================================================================
# FORGE Intelligence
# ===================================================================


def _require_forge() -> ForgeEngine:
    if state.forge is None:
        raise HTTPException(503, "FORGE engine not initialized")
    return state.forge


@app.post("/forge/pre-flight", tags=["FORGE Intelligence"])
async def forge_preflight(req: ForgePreFlightRequest):
    """Run full FORGE pre-flight: Scout scan + Oracle prediction + Smith fixes."""
    forge = _require_forge()
    result = await forge.pre_flight(req.phone_state, req.task)
    return result


@app.get("/forge/stats", tags=["FORGE Intelligence"])
async def forge_stats():
    """FORGE engine statistics."""
    return _require_forge().get_stats()


@app.get("/forge/codex/app/{app_name}", tags=["FORGE Intelligence"])
async def forge_codex_app(app_name: str):
    """Get Codex knowledge about a specific app."""
    forge = _require_forge()
    return forge.codex.get_app_history(app_name)


@app.get("/forge/codex/patterns/{app_name}", tags=["FORGE Intelligence"])
async def forge_codex_patterns(app_name: str):
    """Get failure patterns recorded for an app."""
    forge = _require_forge()
    return {
        "app": app_name,
        "failure_patterns": forge.codex.get_failure_patterns(app_name),
        "common_errors": forge.codex.get_common_errors(app_name),
    }


@app.post("/forge/codex/learn", response_model=ActionResponse, tags=["FORGE Intelligence"])
async def forge_codex_learn(req: ForgeLearnRequest):
    """Feed learning data into the Codex."""
    forge = _require_forge()
    result = forge.record_outcome(req.task_id, req.outcome, req.duration, req.error)
    return ActionResponse(
        success=result is not None,
        message=f"Recorded outcome '{req.outcome}' for task {req.task_id}",
    )


@app.post("/forge/vision-prompt", tags=["FORGE Intelligence"])
async def forge_vision_prompt(req: VisionPromptRequest):
    """Get a SENTINEL-optimized vision prompt for a given template and context."""
    forge = _require_forge()
    try:
        prompt = forge.vision_prompt(req.template, req.context)
        return {"template": req.template, "prompt": prompt}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ===================================================================
# AMPLIFY Pipeline
# ===================================================================


def _require_amplify() -> AmplifyPipeline:
    if state.amplify is None:
        raise HTTPException(503, "AMPLIFY pipeline not initialized")
    return state.amplify


@app.post("/amplify/process", tags=["AMPLIFY Pipeline"])
async def amplify_process(task_config: Dict[str, Any]):
    """Run the full 6-stage AMPLIFY pipeline on a task configuration."""
    pipeline = _require_amplify()
    try:
        enhanced = pipeline.full_pipeline(task_config)
        return enhanced
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/amplify/stats/{app_name}", tags=["AMPLIFY Pipeline"])
async def amplify_stats(app_name: str):
    """Get AMPLIFY performance stats for an app."""
    return _require_amplify().get_app_stats(app_name)


@app.post("/amplify/record", response_model=ActionResponse, tags=["AMPLIFY Pipeline"])
async def amplify_record(req: AmplifyRecordRequest):
    """Record execution timing for the OPTIMIZE learning stage."""
    pipeline = _require_amplify()
    pipeline.record_execution(req.action_type, req.app_name, req.duration, req.success)
    return ActionResponse(
        success=True,
        message=f"Recorded {req.action_type} for {req.app_name}: {req.duration:.2f}s ({'ok' if req.success else 'fail'})",
    )


# ===================================================================
# Screenpipe
# ===================================================================


def _require_screenpipe() -> ScreenpipeAgent:
    if state.screenpipe is None:
        raise HTTPException(503, "Screenpipe agent not initialized")
    return state.screenpipe


@app.get("/screenpipe/state", tags=["Screenpipe"])
async def screenpipe_state():
    """Current screen state from Screenpipe OCR."""
    agent = _require_screenpipe()
    try:
        results = await agent.get_current_state()
        return {
            "results": [
                {"content": r.content, "app": r.app_name, "window": r.window_name, "timestamp": r.timestamp}
                for r in results
            ],
            "count": len(results),
        }
    except Exception as exc:
        raise HTTPException(502, f"Screenpipe unavailable: {exc}")


@app.get("/screenpipe/errors", tags=["Screenpipe"])
async def screenpipe_errors(app: Optional[str] = None, minutes: int = 10):
    """Recent errors detected on screen via OCR."""
    agent = _require_screenpipe()
    try:
        results = await agent.search_errors(app_name=app, minutes_back=minutes)
        return {
            "errors": [
                {"content": r.content, "app": r.app_name, "window": r.window_name, "timestamp": r.timestamp}
                for r in results
            ],
            "count": len(results),
        }
    except Exception as exc:
        raise HTTPException(502, f"Screenpipe error search failed: {exc}")


@app.get("/screenpipe/timeline", tags=["Screenpipe"])
async def screenpipe_timeline(minutes: int = 30, app: Optional[str] = None):
    """Activity timeline from Screenpipe recordings."""
    agent = _require_screenpipe()
    try:
        entries = await agent.get_activity_timeline(minutes_back=minutes, app_name=app)
        return {
            "timeline": [
                {
                    "app": e.app_name,
                    "window": e.window_name,
                    "start": e.start_time,
                    "end": e.end_time,
                    "duration_seconds": e.duration_seconds,
                    "snippets": e.text_snippets,
                }
                for e in entries
            ],
            "count": len(entries),
        }
    except Exception as exc:
        raise HTTPException(502, f"Screenpipe timeline failed: {exc}")


@app.post("/screenpipe/monitor", tags=["Screenpipe"])
async def screenpipe_monitor(req: MonitorRequest):
    """Start monitoring for a text pattern on screen."""
    agent = _require_screenpipe()
    try:
        match = await agent.monitor_for_pattern(
            pattern=req.pattern,
            timeout=req.timeout or 60.0,
            app_name=req.app,
        )
        if match:
            return {
                "found": True,
                "pattern": match.pattern,
                "matched_text": match.matched_text,
                "app": match.app_name,
                "window": match.window_name,
                "timestamp": match.timestamp,
                "context": match.context,
            }
        return {"found": False, "pattern": req.pattern, "timeout": req.timeout}
    except Exception as exc:
        raise HTTPException(502, f"Screenpipe monitor failed: {exc}")


# ===================================================================
# Vision
# ===================================================================


def _require_vision() -> VisionAgent:
    if state.vision is None:
        raise HTTPException(503, "Vision agent not initialized")
    return state.vision


@app.post("/vision/analyze", tags=["Vision"])
async def vision_analyze(req: VisionRequest):
    """Analyze a phone screenshot with the vision AI."""
    agent = _require_vision()
    try:
        result = await agent.analyze_screen(image_path=req.image_path, image_b64=req.image_base64)
        return {
            "app": result.current_app,
            "screen": result.current_screen,
            "visible_text": result.visible_text,
            "elements_count": len(result.tappable_elements),
            "keyboard_visible": result.keyboard_visible,
            "loading": result.loading_indicators,
            "errors": result.errors_detected,
            "quality_score": result.quality_score,
            "analysis_time_ms": round(result.analysis_time_ms, 1),
        }
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Vision service error: {exc}")


@app.post("/vision/find-element", tags=["Vision"])
async def vision_find_element(req: FindElementRequest):
    """Find a UI element on a screenshot by its description."""
    agent = _require_vision()
    try:
        elem = await agent.find_element(
            req.description, image_path=req.image_path, image_b64=req.image_base64,
        )
        if elem is None:
            return {"found": False, "description": req.description}
        return {
            "found": True,
            "description": req.description,
            "x": elem.x,
            "y": elem.y,
            "width": elem.width,
            "height": elem.height,
            "center": list(elem.center),
            "confidence": elem.confidence,
            "tappable": elem.tappable,
            "text": elem.text,
        }
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Vision service error: {exc}")


@app.post("/vision/detect-state", tags=["Vision"])
async def vision_detect_state(req: VisionRequest):
    """Detect the current app state from a screenshot."""
    agent = _require_vision()
    try:
        app_state, confidence, details = await agent.detect_state(
            image_path=req.image_path, image_b64=req.image_base64,
        )
        return {"state": app_state.value, "confidence": confidence, "details": details}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Vision service error: {exc}")


@app.post("/vision/detect-errors", tags=["Vision"])
async def vision_detect_errors(req: VisionRequest):
    """Detect errors, crashes, or permission dialogs on a screenshot."""
    agent = _require_vision()
    try:
        result = await agent.detect_errors(
            image_path=req.image_path, image_b64=req.image_base64,
        )
        resp: Dict[str, Any] = {
            "has_errors": result.has_errors,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "dismissable": result.dismissable,
        }
        if result.dismiss_button:
            resp["dismiss_button"] = {
                "x": result.dismiss_button.x,
                "y": result.dismiss_button.y,
                "width": result.dismiss_button.width,
                "height": result.dismiss_button.height,
            }
        return resp
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Vision service error: {exc}")


# ===================================================================
# Entry Point
# ===================================================================

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=False,
        log_level="info",
    )
