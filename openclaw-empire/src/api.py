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

import asyncio
import base64
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.amplify_pipeline import AmplifyPipeline
from src.auth import (
    SecurityMiddleware,
    TokenAuth,
    TokenInfo,
    RateLimiter,
    WebhookSecurity,
    init_auth,
    rate_limit,
    rate_limit_phone,
    rate_limit_strict,
    rate_limit_task,
    require_auth,
    require_scope,
)
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

ALLOWED_ORIGINS = os.getenv(
    "OPENCLAW_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8765",
).split(",")

AUTH_DISABLED = os.getenv("OPENCLAW_AUTH_DISABLED", "").lower() in ("true", "1", "yes")

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


# -- New request models for auth + empire endpoints -------------------------


class TokenGenerateRequest(BaseModel):
    name: str = "default"
    scopes: List[str] = Field(default_factory=lambda: ["admin"])
    expires_days: Optional[int] = None


class ContentGenerateRequest(BaseModel):
    site_id: str
    title: str
    content_type: str = "blog_post"
    voice_profile: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class ContentOutlineRequest(BaseModel):
    site_id: str
    topic: str
    depth: int = 3


class WordPressPublishRequest(BaseModel):
    site_id: str
    title: str
    content: str
    status: str = "draft"
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    featured_image_id: Optional[int] = None


class SchedulerJobRequest(BaseModel):
    name: str
    schedule: str  # cron expression
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class VoiceScoreRequest(BaseModel):
    site_id: str
    text: str


# -- Dashboard v2 request models -------------------------------------------


class PipelineTriggerRequest(BaseModel):
    site_id: str
    title: str


class MissionExecuteRequest(BaseModel):
    mission_type: str = Field(..., description="e.g. CONTENT_PUBLISH, SEO_AUDIT, SOCIAL_BLAST")
    params: Dict[str, Any] = Field(default_factory=dict)


class PhoneCommandRequest(BaseModel):
    device_id: str
    command: str = Field(..., description="tap, swipe, home, back, screenshot, type")
    params: Dict[str, Any] = Field(default_factory=dict)


class PhoneMirrorConfig(BaseModel):
    device_id: str
    fps: float = 1.0


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


# -- Dashboard v2 response models ------------------------------------------


class EmpireStatsResponse(BaseModel):
    total_articles: int = 0
    total_sites: int = 16
    active_devices: int = 0
    active_pipelines: int = 0
    uptime_seconds: float = 0.0
    revenue_today: float = 0.0
    timestamp: str = ""


class DeviceInfo(BaseModel):
    device_id: str
    name: str = ""
    status: str = "unknown"
    platform: str = ""
    last_seen: Optional[str] = None
    current_app: Optional[str] = None


class PipelineInfo(BaseModel):
    pipeline_id: str
    site_id: str
    title: str
    status: str = "pending"
    current_stage: str = ""
    stage_index: int = 0
    total_stages: int = 14
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class CalendarEntry(BaseModel):
    site_id: str
    title: str
    scheduled_date: str
    status: str = "scheduled"
    content_type: str = "blog_post"


class WorkflowTemplate(BaseModel):
    workflow_id: str
    name: str
    description: str = ""
    steps: List[str] = Field(default_factory=list)
    last_run: Optional[str] = None


class ABTestInfo(BaseModel):
    test_id: str
    name: str
    site_id: str
    status: str = "running"
    variant_a: str = ""
    variant_b: str = ""
    winner: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class RevenueDataPoint(BaseModel):
    date: str
    revenue: float
    source: str = ""
    site_id: Optional[str] = None


# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages WebSocket connections organized by named channels."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            if channel not in self.active_connections:
                self.active_connections[channel] = []
            self.active_connections[channel].append(websocket)
        logger.info("WS connect: channel=%s, total=%d", channel, len(self.active_connections[channel]))

    async def disconnect(self, channel: str, websocket: WebSocket) -> None:
        async with self._lock:
            if channel in self.active_connections:
                try:
                    self.active_connections[channel].remove(websocket)
                except ValueError:
                    pass
                if not self.active_connections[channel]:
                    del self.active_connections[channel]
        logger.info("WS disconnect: channel=%s", channel)

    async def broadcast(self, channel: str, data: dict) -> None:
        """Send a JSON message to all connections on a channel."""
        connections = self.active_connections.get(channel, [])
        dead: List[WebSocket] = []
        for conn in connections:
            try:
                await conn.send_json(data)
            except Exception:
                dead.append(conn)
        # Clean up dead connections
        if dead:
            async with self._lock:
                for conn in dead:
                    try:
                        self.active_connections.get(channel, []).remove(conn)
                    except ValueError:
                        pass

    def channel_count(self, channel: str) -> int:
        return len(self.active_connections.get(channel, []))

    @property
    def total_connections(self) -> int:
        return sum(len(v) for v in self.active_connections.values())


ws_manager = ConnectionManager()


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
        # Auth subsystems
        self.auth: Optional[TokenAuth] = None
        self.rate_limiter: Optional[RateLimiter] = None
        # Dashboard v2 state
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        self.activity_feed: List[Dict[str, Any]] = []  # last N events
        self.activity_feed_max: int = 200


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize subsystems on startup, tear down on shutdown."""
    logger.info("Starting OpenClaw Empire API on port %d", API_PORT)
    state.start_time = time.monotonic()

    # Auth subsystems
    state.auth = TokenAuth()
    state.rate_limiter = RateLimiter()
    init_auth(token_auth=state.auth, rate_limiter=state.rate_limiter)
    logger.info(
        "Auth initialized (disabled=%s, tokens_loaded=%d)",
        AUTH_DISABLED,
        len(state.auth.list_tokens()),
    )

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
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SecurityMiddleware)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===================================================================
# Health (public — no auth required, rate-limited only)
# ===================================================================


@app.get("/health", response_model=StatusResponse, tags=["Health"])
async def health(_rl=Depends(rate_limit)):
    """Server health check with subsystem status."""
    subs: Dict[str, str] = {}
    subs["phone"] = "connected" if (state.controller and state.controller.is_connected) else "disconnected"
    subs["forge"] = "ready" if state.forge else "unavailable"
    subs["amplify"] = "ready" if state.amplify else "unavailable"
    subs["screenpipe"] = "configured" if state.screenpipe else "unavailable"
    subs["vision"] = "configured" if state.vision else "unavailable"
    subs["auth"] = "enabled" if not AUTH_DISABLED else "disabled"
    uptime = time.monotonic() - state.start_time if state.start_time else 0
    subs["uptime_seconds"] = f"{uptime:.0f}"
    return StatusResponse(status="ok", timestamp=_now_iso(), subsystems=subs)


@app.get("/stats", tags=["Health"])
async def stats(_rl=Depends(rate_limit)):
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
async def phone_screenshot(
    _auth=Depends(require_scope("phone:read")),
    _rl=Depends(rate_limit),
):
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
async def phone_state(
    _auth=Depends(require_scope("phone:read")),
    _rl=Depends(rate_limit),
):
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
async def phone_tap(
    req: TapRequest,
    _auth=Depends(require_scope("phone:control")),
    _rl=Depends(rate_limit_phone),
):
    """Tap at the specified coordinates."""
    ctrl = _require_controller()
    result = await ctrl.tap(req.x, req.y)
    return ActionResponse(
        success=result.success,
        message=result.error or f"Tapped ({req.x}, {req.y})",
        duration_ms=result.duration_ms,
    )


@app.post("/phone/type", response_model=ActionResponse, tags=["Phone Control"])
async def phone_type(
    req: TypeRequest,
    _auth=Depends(require_scope("phone:control")),
    _rl=Depends(rate_limit_phone),
):
    """Type text into the currently focused field."""
    ctrl = _require_controller()
    result = await ctrl.type_text(req.text)
    return ActionResponse(
        success=result.success,
        message=result.error or f"Typed {len(req.text)} characters",
        duration_ms=result.duration_ms,
    )


@app.post("/phone/swipe", response_model=ActionResponse, tags=["Phone Control"])
async def phone_swipe(
    req: SwipeRequest,
    _auth=Depends(require_scope("phone:control")),
    _rl=Depends(rate_limit_phone),
):
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
async def phone_launch(
    req: LaunchRequest,
    _auth=Depends(require_scope("phone:control")),
    _rl=Depends(rate_limit_phone),
):
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
async def phone_back(
    _auth=Depends(require_scope("phone:control")),
    _rl=Depends(rate_limit_phone),
):
    """Press the back button."""
    result = await _require_controller().press_back()
    return ActionResponse(success=result.success, message=result.error or "Back pressed", duration_ms=result.duration_ms)


@app.post("/phone/home", response_model=ActionResponse, tags=["Phone Control"])
async def phone_home(
    _auth=Depends(require_scope("phone:control")),
    _rl=Depends(rate_limit_phone),
):
    """Press the home button."""
    result = await _require_controller().press_home()
    return ActionResponse(success=result.success, message=result.error or "Home pressed", duration_ms=result.duration_ms)


# ===================================================================
# Task Execution
# ===================================================================


@app.post("/task/execute", response_model=TaskResponse, tags=["Task Execution"])
async def task_execute(
    req: TaskExecuteRequest,
    _auth=Depends(require_scope("task:execute")),
    _rl=Depends(rate_limit_task),
):
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
async def task_analyze(
    req: TaskAnalyzeRequest,
    _auth=Depends(require_scope("task:execute")),
    _rl=Depends(rate_limit),
):
    """Pre-flight analysis only -- does not execute the task."""
    if state.forge is None:
        raise HTTPException(503, "FORGE engine not initialized")

    task_spec: Dict[str, Any] = {"task_description": req.task_description}
    if req.app:
        task_spec["app"] = req.app

    # Run FORGE pre-flight with mock phone state
    ps: Dict[str, Any] = {}
    try:
        ctrl = _require_controller()
        current_app = await ctrl.get_current_app()
        ps = {
            "screen_on": True,
            "locked": False,
            "active_app": current_app,
            "wifi_connected": True,
            "battery_percent": 100,
        }
    except Exception:
        ps = {"screen_on": True, "locked": False, "wifi_connected": True, "battery_percent": 100}

    result = await state.forge.pre_flight(ps, task_spec)
    return TaskResponse(
        task_id=uuid.uuid4().hex[:12],
        status="analyzed",
        description=req.task_description,
        data=result,
    )


@app.get("/task/{task_id}/status", tags=["Task Execution"])
async def task_status(
    task_id: str,
    _auth=Depends(require_scope("task:execute")),
    _rl=Depends(rate_limit),
):
    """Get the status of a running or completed task."""
    info = state.running_tasks.get(task_id)
    if info is None:
        raise HTTPException(404, f"Task {task_id} not found")
    return {"task_id": task_id, **info}


@app.post("/task/{task_id}/complete", response_model=ActionResponse, tags=["Task Execution"])
async def task_complete(
    task_id: str,
    req: TaskCompleteRequest,
    _auth=Depends(require_scope("task:execute")),
    _rl=Depends(rate_limit_strict),
):
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
async def forge_preflight(
    req: ForgePreFlightRequest,
    _auth=Depends(require_scope("forge:read")),
    _rl=Depends(rate_limit),
):
    """Run full FORGE pre-flight: Scout scan + Oracle prediction + Smith fixes."""
    forge = _require_forge()
    result = await forge.pre_flight(req.phone_state, req.task)
    return result


@app.get("/forge/stats", tags=["FORGE Intelligence"])
async def forge_stats(
    _auth=Depends(require_scope("forge:read")),
    _rl=Depends(rate_limit),
):
    """FORGE engine statistics."""
    return _require_forge().get_stats()


@app.get("/forge/codex/app/{app_name}", tags=["FORGE Intelligence"])
async def forge_codex_app(
    app_name: str,
    _auth=Depends(require_scope("forge:read")),
    _rl=Depends(rate_limit),
):
    """Get Codex knowledge about a specific app."""
    forge = _require_forge()
    return forge.codex.get_app_history(app_name)


@app.get("/forge/codex/patterns/{app_name}", tags=["FORGE Intelligence"])
async def forge_codex_patterns(
    app_name: str,
    _auth=Depends(require_scope("forge:read")),
    _rl=Depends(rate_limit),
):
    """Get failure patterns recorded for an app."""
    forge = _require_forge()
    return {
        "app": app_name,
        "failure_patterns": forge.codex.get_failure_patterns(app_name),
        "common_errors": forge.codex.get_common_errors(app_name),
    }


@app.post("/forge/codex/learn", response_model=ActionResponse, tags=["FORGE Intelligence"])
async def forge_codex_learn(
    req: ForgeLearnRequest,
    _auth=Depends(require_scope("forge:write")),
    _rl=Depends(rate_limit_strict),
):
    """Feed learning data into the Codex."""
    forge = _require_forge()
    result = forge.record_outcome(req.task_id, req.outcome, req.duration, req.error)
    return ActionResponse(
        success=result is not None,
        message=f"Recorded outcome '{req.outcome}' for task {req.task_id}",
    )


@app.post("/forge/vision-prompt", tags=["FORGE Intelligence"])
async def forge_vision_prompt(
    req: VisionPromptRequest,
    _auth=Depends(require_scope("forge:read")),
    _rl=Depends(rate_limit),
):
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
async def amplify_process(
    task_config: Dict[str, Any],
    _auth=Depends(require_scope("amplify:write")),
    _rl=Depends(rate_limit_strict),
):
    """Run the full 6-stage AMPLIFY pipeline on a task configuration."""
    pipeline = _require_amplify()
    try:
        enhanced = pipeline.full_pipeline(task_config)
        return enhanced
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/amplify/stats/{app_name}", tags=["AMPLIFY Pipeline"])
async def amplify_stats(
    app_name: str,
    _auth=Depends(require_scope("amplify:read")),
    _rl=Depends(rate_limit),
):
    """Get AMPLIFY performance stats for an app."""
    return _require_amplify().get_app_stats(app_name)


@app.post("/amplify/record", response_model=ActionResponse, tags=["AMPLIFY Pipeline"])
async def amplify_record(
    req: AmplifyRecordRequest,
    _auth=Depends(require_scope("amplify:write")),
    _rl=Depends(rate_limit_strict),
):
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
async def screenpipe_state(
    _auth=Depends(require_scope("screenpipe:read")),
    _rl=Depends(rate_limit),
):
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
async def screenpipe_errors(
    app: Optional[str] = None,
    minutes: int = 10,
    _auth=Depends(require_scope("screenpipe:read")),
    _rl=Depends(rate_limit),
):
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
async def screenpipe_timeline(
    minutes: int = 30,
    app: Optional[str] = None,
    _auth=Depends(require_scope("screenpipe:read")),
    _rl=Depends(rate_limit),
):
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
async def screenpipe_monitor(
    req: MonitorRequest,
    _auth=Depends(require_scope("screenpipe:read")),
    _rl=Depends(rate_limit),
):
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
async def vision_analyze(
    req: VisionRequest,
    _auth=Depends(require_scope("vision:read")),
    _rl=Depends(rate_limit),
):
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
async def vision_find_element(
    req: FindElementRequest,
    _auth=Depends(require_scope("vision:read")),
    _rl=Depends(rate_limit),
):
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
async def vision_detect_state(
    req: VisionRequest,
    _auth=Depends(require_scope("vision:read")),
    _rl=Depends(rate_limit),
):
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
async def vision_detect_errors(
    req: VisionRequest,
    _auth=Depends(require_scope("vision:read")),
    _rl=Depends(rate_limit),
):
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
# Auth Management (admin only)
# ===================================================================


@app.post("/auth/token", tags=["Auth Management"])
async def auth_generate_token(
    req: TokenGenerateRequest,
    _auth=Depends(require_scope("admin")),
    _rl=Depends(rate_limit_strict),
):
    """Generate a new API token. Admin only."""
    if state.auth is None:
        raise HTTPException(503, "Auth subsystem not initialized")

    from src.auth import ALL_SCOPES as valid_scopes

    invalid = [s for s in req.scopes if s not in valid_scopes]
    if invalid:
        raise HTTPException(400, f"Unknown scope(s): {', '.join(invalid)}")

    raw_token = state.auth.generate_token(
        name=req.name,
        expires_days=req.expires_days,
        scopes=req.scopes,
    )
    return {
        "token": raw_token,
        "name": req.name,
        "scopes": req.scopes,
        "expires_days": req.expires_days,
        "message": "Save this token now. It will not be shown again.",
    }


@app.get("/auth/tokens", tags=["Auth Management"])
async def auth_list_tokens(
    _auth=Depends(require_scope("admin")),
    _rl=Depends(rate_limit),
):
    """List all valid tokens (names and metadata only). Admin only."""
    if state.auth is None:
        raise HTTPException(503, "Auth subsystem not initialized")
    tokens = state.auth.list_tokens()
    return {
        "tokens": [asdict(t) for t in tokens],
        "count": len(tokens),
    }


@app.delete("/auth/token/{name}", tags=["Auth Management"])
async def auth_revoke_token(
    name: str,
    _auth=Depends(require_scope("admin")),
    _rl=Depends(rate_limit_strict),
):
    """Revoke a token by name. Admin only."""
    if state.auth is None:
        raise HTTPException(503, "Auth subsystem not initialized")
    revoked = state.auth.revoke_token(name=name)
    if not revoked:
        raise HTTPException(404, f"No token found with name '{name}'")
    return {"revoked": True, "name": name}


# ===================================================================
# WordPress Endpoints (placeholder — lazily imports subsystem)
# ===================================================================


def _wordpress_unavailable() -> None:
    """Raise 503 if the WordPress module is not yet available."""
    try:
        from src import wordpress_manager  # noqa: F401
    except ImportError:
        raise HTTPException(503, "WordPress subsystem not yet available. Module pending implementation.")


@app.get("/wordpress/health", tags=["WordPress"])
async def wordpress_health(
    _auth=Depends(require_scope("wordpress:read")),
    _rl=Depends(rate_limit),
):
    """Health check across all configured WordPress sites."""
    try:
        from src.wordpress_manager import get_sites_health
        return await get_sites_health()
    except ImportError:
        raise HTTPException(503, "WordPress subsystem not yet available")
    except Exception as exc:
        raise HTTPException(500, f"WordPress health check failed: {exc}")


@app.get("/wordpress/sites", tags=["WordPress"])
async def wordpress_sites(
    _auth=Depends(require_scope("wordpress:read")),
    _rl=Depends(rate_limit),
):
    """List all configured WordPress sites with status."""
    try:
        from src.wordpress_manager import list_sites
        return await list_sites()
    except ImportError:
        raise HTTPException(503, "WordPress subsystem not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Failed to list sites: {exc}")


@app.post("/wordpress/publish", tags=["WordPress"])
async def wordpress_publish(
    req: WordPressPublishRequest,
    _auth=Depends(require_scope("wordpress:write")),
    _rl=Depends(rate_limit_strict),
):
    """Publish or draft a post to a WordPress site."""
    try:
        from src.wordpress_manager import publish_post
        result = await publish_post(
            site_id=req.site_id,
            title=req.title,
            content=req.content,
            status=req.status,
            categories=req.categories,
            tags=req.tags,
            featured_image_id=req.featured_image_id,
        )
        return result
    except ImportError:
        raise HTTPException(503, "WordPress subsystem not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Publish failed: {exc}")


@app.get("/wordpress/dashboard", tags=["WordPress"])
async def wordpress_dashboard(
    _auth=Depends(require_scope("wordpress:read")),
    _rl=Depends(rate_limit),
):
    """Aggregated dashboard: posts, traffic, health across all sites."""
    try:
        from src.wordpress_manager import get_dashboard
        return await get_dashboard()
    except ImportError:
        raise HTTPException(503, "WordPress subsystem not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Dashboard fetch failed: {exc}")


# ===================================================================
# Content Endpoints (placeholder — lazily imports subsystem)
# ===================================================================


@app.post("/content/generate", tags=["Content"])
async def content_generate(
    req: ContentGenerateRequest,
    _auth=Depends(require_scope("wordpress:write")),
    _rl=Depends(rate_limit_strict),
):
    """Generate content for a specific site using its brand voice."""
    try:
        from src.content_engine import generate_content
        result = await generate_content(
            site_id=req.site_id,
            title=req.title,
            content_type=req.content_type,
            voice_profile=req.voice_profile,
            keywords=req.keywords,
        )
        return result
    except ImportError:
        raise HTTPException(503, "Content engine not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Content generation failed: {exc}")


@app.post("/content/outline", tags=["Content"])
async def content_outline(
    req: ContentOutlineRequest,
    _auth=Depends(require_scope("wordpress:write")),
    _rl=Depends(rate_limit_strict),
):
    """Generate a content outline for a topic."""
    try:
        from src.content_engine import generate_outline
        return await generate_outline(
            site_id=req.site_id,
            topic=req.topic,
            depth=req.depth,
        )
    except ImportError:
        raise HTTPException(503, "Content engine not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Outline generation failed: {exc}")


@app.get("/content/calendar", tags=["Content"])
async def content_calendar(
    site_id: Optional[str] = None,
    days: int = 30,
    _auth=Depends(require_scope("wordpress:read")),
    _rl=Depends(rate_limit),
):
    """Get the content calendar for one or all sites."""
    try:
        from src.content_engine import get_calendar
        return await get_calendar(site_id=site_id, days=days)
    except ImportError:
        raise HTTPException(503, "Content engine not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Calendar fetch failed: {exc}")


# ===================================================================
# n8n Integration Endpoints (placeholder — lazily imports subsystem)
# ===================================================================


@app.post("/n8n/trigger/{workflow}", tags=["n8n"])
async def n8n_trigger(
    workflow: str,
    payload: Dict[str, Any] = None,
    _auth=Depends(require_scope("task:execute")),
    _rl=Depends(rate_limit_strict),
):
    """Trigger an n8n workflow by name via webhook."""
    try:
        from src.n8n_client import trigger_workflow
        result = await trigger_workflow(workflow, payload or {})
        return result
    except ImportError:
        raise HTTPException(503, "n8n client not yet available")
    except Exception as exc:
        raise HTTPException(500, f"n8n trigger failed: {exc}")


@app.get("/n8n/workflows", tags=["n8n"])
async def n8n_workflows(
    _auth=Depends(require_scope("task:execute")),
    _rl=Depends(rate_limit),
):
    """List available n8n workflows."""
    try:
        from src.n8n_client import list_workflows
        return await list_workflows()
    except ImportError:
        raise HTTPException(503, "n8n client not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Failed to list workflows: {exc}")


@app.get("/n8n/executions", tags=["n8n"])
async def n8n_executions(
    workflow: Optional[str] = None,
    limit: int = 20,
    _auth=Depends(require_scope("task:execute")),
    _rl=Depends(rate_limit),
):
    """Get recent n8n workflow executions."""
    try:
        from src.n8n_client import list_executions
        return await list_executions(workflow=workflow, limit=limit)
    except ImportError:
        raise HTTPException(503, "n8n client not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Failed to list executions: {exc}")


# ===================================================================
# Scheduler Endpoints (placeholder — lazily imports subsystem)
# ===================================================================


@app.get("/scheduler/jobs", tags=["Scheduler"])
async def scheduler_list_jobs(
    _auth=Depends(require_scope("scheduler:read")),
    _rl=Depends(rate_limit),
):
    """List all scheduled jobs."""
    try:
        from src.scheduler import list_jobs
        return await list_jobs()
    except ImportError:
        raise HTTPException(503, "Scheduler not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Failed to list jobs: {exc}")


@app.post("/scheduler/jobs", tags=["Scheduler"])
async def scheduler_create_job(
    req: SchedulerJobRequest,
    _auth=Depends(require_scope("scheduler:write")),
    _rl=Depends(rate_limit_strict),
):
    """Create a new scheduled job."""
    try:
        from src.scheduler import create_job
        return await create_job(
            name=req.name,
            schedule=req.schedule,
            action=req.action,
            params=req.params,
            enabled=req.enabled,
        )
    except ImportError:
        raise HTTPException(503, "Scheduler not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Failed to create job: {exc}")


@app.post("/scheduler/run/{job_id}", tags=["Scheduler"])
async def scheduler_run_job(
    job_id: str,
    _auth=Depends(require_scope("scheduler:write")),
    _rl=Depends(rate_limit_strict),
):
    """Manually trigger a scheduled job."""
    try:
        from src.scheduler import run_job
        return await run_job(job_id)
    except ImportError:
        raise HTTPException(503, "Scheduler not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Failed to run job: {exc}")


@app.get("/scheduler/upcoming", tags=["Scheduler"])
async def scheduler_upcoming(
    hours: int = 24,
    _auth=Depends(require_scope("scheduler:read")),
    _rl=Depends(rate_limit),
):
    """Get upcoming scheduled events."""
    try:
        from src.scheduler import get_upcoming
        return await get_upcoming(hours=hours)
    except ImportError:
        raise HTTPException(503, "Scheduler not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Failed to get upcoming jobs: {exc}")


# ===================================================================
# Revenue Endpoints (placeholder — lazily imports subsystem)
# ===================================================================


@app.get("/revenue/today", tags=["Revenue"])
async def revenue_today(
    _auth=Depends(require_scope("revenue:read")),
    _rl=Depends(rate_limit),
):
    """Today's revenue across all channels."""
    try:
        from src.revenue_tracker import get_today
        return await get_today()
    except ImportError:
        raise HTTPException(503, "Revenue tracker not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Revenue fetch failed: {exc}")


@app.get("/revenue/report", tags=["Revenue"])
async def revenue_report(
    days: int = 30,
    _auth=Depends(require_scope("revenue:read")),
    _rl=Depends(rate_limit),
):
    """Revenue report for the specified period."""
    try:
        from src.revenue_tracker import get_report
        return await get_report(days=days)
    except ImportError:
        raise HTTPException(503, "Revenue tracker not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Revenue report failed: {exc}")


@app.get("/revenue/sites/{site_id}", tags=["Revenue"])
async def revenue_by_site(
    site_id: str,
    days: int = 30,
    _auth=Depends(require_scope("revenue:read")),
    _rl=Depends(rate_limit),
):
    """Revenue breakdown for a specific site."""
    try:
        from src.revenue_tracker import get_site_revenue
        return await get_site_revenue(site_id=site_id, days=days)
    except ImportError:
        raise HTTPException(503, "Revenue tracker not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Site revenue fetch failed: {exc}")


# ===================================================================
# Voice Profile Endpoints (placeholder — lazily imports subsystem)
# ===================================================================


@app.get("/voice/profiles", tags=["Voice"])
async def voice_profiles(
    _auth=Depends(require_scope("wordpress:read")),
    _rl=Depends(rate_limit),
):
    """List all brand voice profiles."""
    try:
        from src.voice_engine import list_profiles
        return await list_profiles()
    except ImportError:
        raise HTTPException(503, "Voice engine not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Failed to list profiles: {exc}")


@app.get("/voice/profile/{site_id}", tags=["Voice"])
async def voice_profile(
    site_id: str,
    _auth=Depends(require_scope("wordpress:read")),
    _rl=Depends(rate_limit),
):
    """Get the brand voice profile for a specific site."""
    try:
        from src.voice_engine import get_profile
        return await get_profile(site_id=site_id)
    except ImportError:
        raise HTTPException(503, "Voice engine not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Failed to get profile: {exc}")


@app.post("/voice/score", tags=["Voice"])
async def voice_score(
    req: VoiceScoreRequest,
    _auth=Depends(require_scope("wordpress:read")),
    _rl=Depends(rate_limit),
):
    """Score how well text matches a site's brand voice."""
    try:
        from src.voice_engine import score_text
        return await score_text(site_id=req.site_id, text=req.text)
    except ImportError:
        raise HTTPException(503, "Voice engine not yet available")
    except Exception as exc:
        raise HTTPException(500, f"Voice scoring failed: {exc}")


# ===================================================================
# Dashboard v2 — Helper Utilities
# ===================================================================


async def _emit_activity(
    event_type: str,
    message: str,
    module: str = "system",
    severity: str = "info",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Record an activity event and broadcast to connected WS clients."""
    event = {
        "event_type": event_type,
        "message": message,
        "module": module,
        "timestamp": _now_iso(),
        "severity": severity,
    }
    if extra:
        event.update(extra)
    state.activity_feed.insert(0, event)
    # Trim to max
    if len(state.activity_feed) > state.activity_feed_max:
        state.activity_feed = state.activity_feed[: state.activity_feed_max]
    await ws_manager.broadcast("activity-feed", {"type": "activity", "data": event})


# ===================================================================
# Dashboard v2 — WebSocket Endpoints
# ===================================================================


@app.websocket("/ws/agent-status")
async def ws_agent_status(websocket: WebSocket):
    """Real-time agent mission updates.

    On connect: sends current agent/mission status.
    Broadcasts: mission start/stop, step progress, goal completion.
    """
    await ws_manager.connect("agent-status", websocket)
    try:
        # Send initial status snapshot
        initial: Dict[str, Any] = {
            "type": "agent_status",
            "data": {
                "mission_id": None,
                "status": "idle",
                "step": None,
                "progress_percent": 0.0,
                "running_tasks": len(state.running_tasks),
            },
        }
        # Check if any autonomous agent module is available
        try:
            from src.autonomous_agent import get_current_mission  # type: ignore[import]
            mission = get_current_mission()
            if mission:
                initial["data"].update({
                    "mission_id": mission.get("mission_id"),
                    "status": mission.get("status", "running"),
                    "step": mission.get("current_step"),
                    "progress_percent": mission.get("progress_percent", 0.0),
                })
        except (ImportError, Exception):
            pass

        await websocket.send_json(initial)

        # Keep connection alive, listen for pings / close
        while True:
            data = await websocket.receive_text()
            # Client can send "ping" to keep alive
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect("agent-status", websocket)


@app.websocket("/ws/pipeline-progress")
async def ws_pipeline_progress(websocket: WebSocket):
    """Content pipeline stage updates.

    On connect: sends all active pipelines.
    Broadcasts: stage start/complete, pipeline done.
    """
    await ws_manager.connect("pipeline-progress", websocket)
    try:
        # Send active pipelines snapshot
        pipelines_snapshot = []
        for pid, pdata in state.active_pipelines.items():
            pipelines_snapshot.append({
                "pipeline_id": pid,
                "site_id": pdata.get("site_id", ""),
                "title": pdata.get("title", ""),
                "stage": pdata.get("stage", ""),
                "stage_index": pdata.get("stage_index", 0),
                "total_stages": pdata.get("total_stages", 14),
                "status": pdata.get("status", "running"),
            })
        await websocket.send_json({
            "type": "pipeline_progress",
            "data": {
                "active_pipelines": pipelines_snapshot,
                "count": len(pipelines_snapshot),
            },
        })

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect("pipeline-progress", websocket)


@app.websocket("/ws/activity-feed")
async def ws_activity_feed(websocket: WebSocket):
    """General activity/event stream.

    On connect: sends last 20 events.
    Broadcasts: any module activity (publishes, errors, revenue events, device events).
    """
    await ws_manager.connect("activity-feed", websocket)
    try:
        # Send last 20 events
        recent = state.activity_feed[:20]
        await websocket.send_json({
            "type": "activity_backlog",
            "data": {
                "events": recent,
                "count": len(recent),
            },
        })

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect("activity-feed", websocket)


@app.websocket("/ws/phone-mirror")
async def ws_phone_mirror(websocket: WebSocket):
    """Phone screenshot stream.

    Client sends: { "device_id": "...", "fps": 1 }
    Server streams: { "type": "phone_frame", "data": { "device_id": "...", "screenshot_base64": "...", "timestamp": "..." } }
    """
    await ws_manager.connect("phone-mirror", websocket)
    streaming_task: Optional[asyncio.Task] = None
    try:
        while True:
            raw = await websocket.receive_json()
            config = PhoneMirrorConfig(**raw)

            # Cancel any existing streaming task
            if streaming_task and not streaming_task.done():
                streaming_task.cancel()

            async def _stream_frames(device_id: str, fps: float) -> None:
                interval = 1.0 / max(fps, 0.1)
                while True:
                    try:
                        # Try to get screenshot from device pool or phone controller
                        screenshot_b64 = None
                        try:
                            from src.device_pool import DevicePool  # type: ignore[import]
                            pool = DevicePool.instance()
                            device = pool.get_device(device_id)
                            if device:
                                screenshot_b64 = await device.screenshot_base64()
                        except (ImportError, Exception):
                            pass

                        # Fallback to default phone controller
                        if screenshot_b64 is None and state.controller and state.controller.is_connected:
                            try:
                                path = await state.controller.screenshot()
                                if path:
                                    import aiofiles  # type: ignore[import]
                                    async with aiofiles.open(path, "rb") as f:
                                        raw_bytes = await f.read()
                                    screenshot_b64 = base64.b64encode(raw_bytes).decode("ascii")
                            except Exception:
                                # If aiofiles not available, use sync read
                                try:
                                    path = await state.controller.screenshot()
                                    if path:
                                        with open(path, "rb") as f:
                                            raw_bytes = f.read()
                                        screenshot_b64 = base64.b64encode(raw_bytes).decode("ascii")
                                except Exception:
                                    pass

                        if screenshot_b64:
                            await websocket.send_json({
                                "type": "phone_frame",
                                "data": {
                                    "device_id": device_id,
                                    "screenshot_base64": screenshot_b64,
                                    "timestamp": _now_iso(),
                                },
                            })
                        await asyncio.sleep(interval)
                    except asyncio.CancelledError:
                        return
                    except Exception as exc:
                        logger.warning("phone-mirror frame error: %s", exc)
                        await asyncio.sleep(interval)

            streaming_task = asyncio.create_task(_stream_frames(config.device_id, config.fps))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("phone-mirror error: %s", exc)
    finally:
        if streaming_task and not streaming_task.done():
            streaming_task.cancel()
        await ws_manager.disconnect("phone-mirror", websocket)


# ===================================================================
# Dashboard v2 — New REST Endpoints
# ===================================================================


@app.get("/api/stats", response_model=EmpireStatsResponse, tags=["Dashboard v2"])
async def api_empire_stats(
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit),
):
    """Quick empire stats: total articles, revenue, active devices, uptime."""
    uptime = time.monotonic() - state.start_time if state.start_time else 0.0
    total_articles = 0
    revenue_today = 0.0
    active_devices = 0

    # Articles count from WordPress manager
    try:
        from src.wordpress_manager import get_total_article_count  # type: ignore[import]
        total_articles = await get_total_article_count()
    except (ImportError, Exception):
        pass

    # Revenue from tracker
    try:
        from src.revenue_tracker import get_today  # type: ignore[import]
        rev = await get_today()
        revenue_today = rev.get("total", 0.0) if isinstance(rev, dict) else 0.0
    except (ImportError, Exception):
        pass

    # Active devices from device pool
    try:
        from src.device_pool import DevicePool  # type: ignore[import]
        pool = DevicePool.instance()
        active_devices = len([d for d in pool.list_devices() if d.get("status") == "online"])
    except (ImportError, Exception):
        # Check if the default phone controller is connected
        if state.controller and state.controller.is_connected:
            active_devices = 1

    return EmpireStatsResponse(
        total_articles=total_articles,
        total_sites=16,
        active_devices=active_devices,
        active_pipelines=len(state.active_pipelines),
        uptime_seconds=round(uptime, 1),
        revenue_today=revenue_today,
        timestamp=_now_iso(),
    )


@app.get("/api/devices", tags=["Dashboard v2"])
async def api_list_devices(
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit),
):
    """List all devices from the DevicePool."""
    devices: List[Dict[str, Any]] = []
    try:
        from src.device_pool import DevicePool  # type: ignore[import]
        pool = DevicePool.instance()
        for d in pool.list_devices():
            devices.append({
                "device_id": d.get("device_id", ""),
                "name": d.get("name", ""),
                "status": d.get("status", "unknown"),
                "platform": d.get("platform", ""),
                "last_seen": d.get("last_seen"),
                "current_app": d.get("current_app"),
            })
    except (ImportError, Exception):
        # Fallback: expose default phone controller as a device
        if state.controller:
            devices.append({
                "device_id": "default",
                "name": NODE_NAME,
                "status": "connected" if state.controller.is_connected else "disconnected",
                "platform": "android",
                "last_seen": _now_iso() if state.controller.is_connected else None,
                "current_app": None,
            })
    return {"devices": devices, "count": len(devices)}


@app.get("/api/pipelines", tags=["Dashboard v2"])
async def api_list_pipelines(
    status: Optional[str] = Query(None, description="Filter by status: running, completed, failed"),
    limit: int = Query(20, ge=1, le=100),
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit),
):
    """List active and recent content pipeline runs."""
    pipelines: List[Dict[str, Any]] = []

    # Active pipelines from state
    for pid, pdata in state.active_pipelines.items():
        if status and pdata.get("status") != status:
            continue
        pipelines.append({"pipeline_id": pid, **pdata})

    # Try to get historical pipelines from pipeline manager
    try:
        from src.pipeline_manager import get_recent_pipelines  # type: ignore[import]
        historical = await get_recent_pipelines(limit=limit, status=status)
        for p in historical:
            if p.get("pipeline_id") not in state.active_pipelines:
                pipelines.append(p)
    except (ImportError, Exception):
        pass

    pipelines = pipelines[:limit]
    return {"pipelines": pipelines, "count": len(pipelines)}


@app.post("/api/pipelines/trigger", tags=["Dashboard v2"])
async def api_trigger_pipeline(
    req: PipelineTriggerRequest,
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit_strict),
):
    """Trigger a content pipeline for a site."""
    pipeline_id = uuid.uuid4().hex[:12]

    # Register pipeline in active state
    pipeline_data = {
        "site_id": req.site_id,
        "title": req.title,
        "status": "starting",
        "stage": "init",
        "stage_index": 0,
        "total_stages": 14,
        "started_at": _now_iso(),
        "completed_at": None,
    }
    state.active_pipelines[pipeline_id] = pipeline_data

    # Broadcast pipeline start
    await ws_manager.broadcast("pipeline-progress", {
        "type": "pipeline_progress",
        "data": {
            "pipeline_id": pipeline_id,
            "stage": "init",
            "stage_index": 0,
            "total_stages": 14,
            "success": True,
        },
    })
    await _emit_activity(
        event_type="pipeline_started",
        message=f"Content pipeline started for {req.site_id}: {req.title}",
        module="pipeline",
        severity="info",
    )

    # Try to actually start the pipeline via the pipeline manager
    try:
        from src.pipeline_manager import start_pipeline  # type: ignore[import]
        result = await start_pipeline(
            pipeline_id=pipeline_id,
            site_id=req.site_id,
            title=req.title,
        )
        pipeline_data["status"] = "running"
        return {
            "pipeline_id": pipeline_id,
            "status": "running",
            "site_id": req.site_id,
            "title": req.title,
            "message": "Pipeline started successfully",
            "data": result if isinstance(result, dict) else {},
        }
    except ImportError:
        # Pipeline manager not available -- keep the pipeline in "starting" as a placeholder
        pipeline_data["status"] = "pending_module"
        return {
            "pipeline_id": pipeline_id,
            "status": "pending_module",
            "site_id": req.site_id,
            "title": req.title,
            "message": "Pipeline registered but pipeline_manager module is not yet available",
        }
    except Exception as exc:
        pipeline_data["status"] = "error"
        pipeline_data["error"] = str(exc)
        await _emit_activity(
            event_type="pipeline_error",
            message=f"Pipeline failed for {req.site_id}: {exc}",
            module="pipeline",
            severity="error",
        )
        raise HTTPException(500, f"Pipeline trigger failed: {exc}")


@app.get("/api/content/calendar", tags=["Dashboard v2"])
async def api_content_calendar(
    site: Optional[str] = Query(None, description="Filter by site ID"),
    days_ahead: int = Query(30, ge=1, le=365),
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit),
):
    """Get content calendar entries for the dashboard."""
    entries: List[Dict[str, Any]] = []
    try:
        from src.content_engine import get_calendar  # type: ignore[import]
        result = await get_calendar(site_id=site, days=days_ahead)
        if isinstance(result, dict):
            entries = result.get("entries", result.get("calendar", []))
        elif isinstance(result, list):
            entries = result
    except (ImportError, Exception):
        pass

    # Fallback: try content_calendar skill
    if not entries:
        try:
            from src.content_calendar import get_upcoming_entries  # type: ignore[import]
            entries = await get_upcoming_entries(site_id=site, days_ahead=days_ahead)
        except (ImportError, Exception):
            pass

    return {"entries": entries, "count": len(entries), "days_ahead": days_ahead, "site_filter": site}


@app.post("/api/missions/execute", tags=["Dashboard v2"])
async def api_execute_mission(
    req: MissionExecuteRequest,
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit_strict),
):
    """Execute a workflow mission (e.g. CONTENT_PUBLISH, SEO_AUDIT, SOCIAL_BLAST)."""
    mission_id = uuid.uuid4().hex[:12]

    # Broadcast mission start
    await ws_manager.broadcast("agent-status", {
        "type": "agent_status",
        "data": {
            "mission_id": mission_id,
            "status": "starting",
            "step": "init",
            "progress_percent": 0.0,
        },
    })
    await _emit_activity(
        event_type="mission_started",
        message=f"Mission {req.mission_type} started (id={mission_id})",
        module="agent",
        severity="info",
    )

    try:
        from src.autonomous_agent import execute_mission  # type: ignore[import]
        result = await execute_mission(
            mission_id=mission_id,
            mission_type=req.mission_type,
            params=req.params,
        )
        await ws_manager.broadcast("agent-status", {
            "type": "agent_status",
            "data": {
                "mission_id": mission_id,
                "status": "completed",
                "step": "done",
                "progress_percent": 100.0,
            },
        })
        return {
            "mission_id": mission_id,
            "mission_type": req.mission_type,
            "status": "completed",
            "result": result if isinstance(result, dict) else {"raw": str(result)},
        }
    except ImportError:
        return {
            "mission_id": mission_id,
            "mission_type": req.mission_type,
            "status": "pending_module",
            "message": "autonomous_agent module not yet available. Mission registered.",
        }
    except Exception as exc:
        await ws_manager.broadcast("agent-status", {
            "type": "agent_status",
            "data": {
                "mission_id": mission_id,
                "status": "error",
                "step": "failed",
                "progress_percent": 0.0,
            },
        })
        await _emit_activity(
            event_type="mission_error",
            message=f"Mission {req.mission_type} failed: {exc}",
            module="agent",
            severity="error",
        )
        raise HTTPException(500, f"Mission execution failed: {exc}")


@app.get("/api/workflows", tags=["Dashboard v2"])
async def api_list_workflows(
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit),
):
    """List saved workflow templates."""
    workflows: List[Dict[str, Any]] = []
    try:
        from src.workflow_registry import list_workflows  # type: ignore[import]
        workflows = await list_workflows()
        if not isinstance(workflows, list):
            workflows = workflows.get("workflows", []) if isinstance(workflows, dict) else []
    except (ImportError, Exception):
        pass

    # Fallback: list built-in mission types
    if not workflows:
        workflows = [
            {
                "workflow_id": "content_publish",
                "name": "Content Publish",
                "description": "Generate, optimize, and publish content to a WordPress site",
                "steps": ["research", "outline", "draft", "seo_optimize", "voice_check", "publish", "social_amplify"],
                "last_run": None,
            },
            {
                "workflow_id": "seo_audit",
                "name": "SEO Audit",
                "description": "Run full SEO audit across all sites",
                "steps": ["crawl", "analyze", "report", "fix_critical"],
                "last_run": None,
            },
            {
                "workflow_id": "social_blast",
                "name": "Social Blast",
                "description": "Cross-platform social media campaign",
                "steps": ["generate_captions", "create_images", "schedule_posts", "monitor_engagement"],
                "last_run": None,
            },
            {
                "workflow_id": "revenue_report",
                "name": "Revenue Report",
                "description": "Generate comprehensive revenue report across all channels",
                "steps": ["collect_adsense", "collect_affiliate", "collect_kdp", "collect_etsy", "compile_report"],
                "last_run": None,
            },
        ]
    return {"workflows": workflows, "count": len(workflows)}


@app.get("/api/ab-tests", tags=["Dashboard v2"])
async def api_list_ab_tests(
    site_id: Optional[str] = Query(None, description="Filter by site ID"),
    status: Optional[str] = Query(None, description="Filter by status: running, completed, paused"),
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit),
):
    """List A/B test experiments."""
    tests: List[Dict[str, Any]] = []
    try:
        from src.ab_testing import list_experiments  # type: ignore[import]
        tests = await list_experiments(site_id=site_id, status=status)
        if not isinstance(tests, list):
            tests = tests.get("experiments", []) if isinstance(tests, dict) else []
    except (ImportError, Exception):
        pass
    return {"experiments": tests, "count": len(tests)}


@app.get("/api/revenue", tags=["Dashboard v2"])
async def api_revenue_chart(
    days: int = Query(30, ge=1, le=365),
    site_id: Optional[str] = Query(None, description="Filter by site ID"),
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit),
):
    """Revenue data formatted for dashboard charts."""
    data_points: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"total": 0.0, "average_daily": 0.0, "trend": "flat"}

    try:
        from src.revenue_tracker import get_report  # type: ignore[import]
        report = await get_report(days=days)
        if isinstance(report, dict):
            data_points = report.get("daily", report.get("data_points", []))
            summary["total"] = report.get("total", 0.0)
            summary["average_daily"] = report.get("average_daily", 0.0)
            summary["trend"] = report.get("trend", "flat")
    except (ImportError, Exception):
        pass

    # If site filter requested but not handled by tracker, try site-specific
    if site_id and not data_points:
        try:
            from src.revenue_tracker import get_site_revenue  # type: ignore[import]
            site_report = await get_site_revenue(site_id=site_id, days=days)
            if isinstance(site_report, dict):
                data_points = site_report.get("daily", [])
                summary["total"] = site_report.get("total", 0.0)
        except (ImportError, Exception):
            pass

    return {
        "data_points": data_points,
        "summary": summary,
        "days": days,
        "site_filter": site_id,
        "count": len(data_points),
    }


@app.post("/api/phone/command", tags=["Dashboard v2"])
async def api_phone_command(
    req: PhoneCommandRequest,
    _auth=Depends(require_auth),
    _rl=Depends(rate_limit_phone),
):
    """Send a command to a phone device."""
    # Try device pool first for multi-device support
    try:
        from src.device_pool import DevicePool  # type: ignore[import]
        pool = DevicePool.instance()
        device = pool.get_device(req.device_id)
        if device:
            result = await device.execute_command(req.command, req.params)
            await _emit_activity(
                event_type="phone_command",
                message=f"Command '{req.command}' sent to device {req.device_id}",
                module="phone",
                severity="info",
            )
            return ActionResponse(
                success=True,
                message=f"Command '{req.command}' executed on {req.device_id}",
                data=result if isinstance(result, dict) else {"raw": str(result)},
            )
    except (ImportError, Exception):
        pass

    # Fallback to default phone controller (device_id == "default" or any)
    ctrl = _require_controller()
    command = req.command.lower()
    params = req.params

    try:
        if command == "tap":
            x = params.get("x", 0)
            y = params.get("y", 0)
            result = await ctrl.tap(int(x), int(y))
        elif command == "swipe":
            direction = params.get("direction")
            if direction:
                w, h = ctrl.resolution
                cx, cy = w // 2, h // 2
                dist = h // 3
                directions = {
                    "up": (cx, cy + dist, cx, cy - dist),
                    "down": (cx, cy - dist, cx, cy + dist),
                    "left": (cx + dist, cy, cx - dist, cy),
                    "right": (cx - dist, cy, cx + dist, cy),
                }
                coords = directions.get(direction.lower())
                if not coords:
                    raise HTTPException(400, f"Invalid swipe direction: {direction}")
                result = await ctrl.swipe(*coords)
            else:
                result = await ctrl.swipe(
                    int(params.get("x1", 0)),
                    int(params.get("y1", 0)),
                    int(params.get("x2", 0)),
                    int(params.get("y2", 0)),
                )
        elif command == "home":
            result = await ctrl.press_home()
        elif command == "back":
            result = await ctrl.press_back()
        elif command == "screenshot":
            path = await ctrl.screenshot()
            await _emit_activity(
                event_type="phone_command",
                message=f"Screenshot taken on device {req.device_id}",
                module="phone",
                severity="info",
            )
            return ActionResponse(
                success=True,
                message=f"Screenshot saved: {path}",
                data={"path": path},
            )
        elif command == "type":
            text = params.get("text", "")
            result = await ctrl.type_text(text)
        else:
            raise HTTPException(400, f"Unknown command: {command}. Supported: tap, swipe, home, back, screenshot, type")

        await _emit_activity(
            event_type="phone_command",
            message=f"Command '{command}' executed on device {req.device_id}",
            module="phone",
            severity="info",
        )
        return ActionResponse(
            success=result.success,
            message=result.error or f"Command '{command}' executed",
            duration_ms=result.duration_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Phone command failed: {exc}")


# ===================================================================
# Dashboard v2 — Broadcast Helpers (for use by other modules)
# ===================================================================


async def broadcast_agent_status(
    mission_id: str,
    status: str,
    step: str,
    progress_percent: float,
) -> None:
    """Broadcast an agent status update to all connected WS clients.

    Call this from other modules (e.g. autonomous_agent) to push real-time
    updates to the dashboard.
    """
    await ws_manager.broadcast("agent-status", {
        "type": "agent_status",
        "data": {
            "mission_id": mission_id,
            "status": status,
            "step": step,
            "progress_percent": progress_percent,
        },
    })


async def broadcast_pipeline_progress(
    pipeline_id: str,
    stage: str,
    stage_index: int,
    total_stages: int,
    success: bool,
) -> None:
    """Broadcast a pipeline stage update to all connected WS clients.

    Call this from the pipeline_manager to push stage progress.
    """
    # Update active pipeline state
    if pipeline_id in state.active_pipelines:
        state.active_pipelines[pipeline_id]["stage"] = stage
        state.active_pipelines[pipeline_id]["stage_index"] = stage_index
        if stage_index >= total_stages:
            state.active_pipelines[pipeline_id]["status"] = "completed"
            state.active_pipelines[pipeline_id]["completed_at"] = _now_iso()

    await ws_manager.broadcast("pipeline-progress", {
        "type": "pipeline_progress",
        "data": {
            "pipeline_id": pipeline_id,
            "stage": stage,
            "stage_index": stage_index,
            "total_stages": total_stages,
            "success": success,
        },
    })


async def emit_activity_event(
    event_type: str,
    message: str,
    module: str = "system",
    severity: str = "info",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Public API for other modules to emit activity feed events."""
    await _emit_activity(event_type, message, module, severity, extra)


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
