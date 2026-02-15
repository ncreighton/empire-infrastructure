"""
Unified Orchestrator — Brain Merger for OpenClaw Empire
=======================================================

Merges AutonomousAgent + IntelligenceHub into a single unified orchestrator
with mission templates, conversation mode, and circuit-breaker dispatch.

Central command brain for Nick Creighton's 16-site WordPress publishing empire.
Every empire operation — content publishing, social growth, phone automation,
account creation, app exploration, revenue tracking, site maintenance, device
management, Substack daily — routes through this module.

Architecture:
    - Mission Templates: predefined step sequences for every MissionType
    - Circuit-Breaker Dispatch: fault-tolerant module invocation with retry,
      exponential backoff, and circuit state tracking
    - RAG-Enhanced Conversation: Sonnet-powered conversational interface with
      semantic memory retrieval for context-aware responses
    - Lazy Module Registry: importlib-based deferred loading of ~25 modules
    - Singleton Pattern: get_orchestrator() returns the global instance

Data persisted to: data/orchestrator/
    missions.json          -- all missions (bounded at 2000)
    conversations.json     -- conversation sessions (bounded at 500)
    dispatch_log.json      -- dispatch audit trail (bounded at 10000)
    stats.json             -- aggregate statistics

Usage:
    from src.unified_orchestrator import get_orchestrator

    orch = get_orchestrator()

    # Execute a content publish mission
    mission = await orch.execute_mission(MissionType.CONTENT_PUBLISH,
        {"site_id": "witchcraft", "title": "Moon Water Ritual"})

    # Dispatch to any module directly
    result = await orch.dispatch("content_generator", "generate_full_article",
        site_id="witchcraft", title="Moon Water Guide")

    # Conversational interface
    reply = await orch.converse("What's our revenue looking like this week?")

    # Phone automation with FORGE + AMPLIFY
    result = await orch.execute_phone_task("Post to Instagram for witchcraft niche")

CLI:
    python -m src.unified_orchestrator mission --type CONTENT_PUBLISH --site witchcraft --title "Moon Water"
    python -m src.unified_orchestrator dispatch --module content_generator --method generate_full_article
    python -m src.unified_orchestrator publish --site witchcraft --title "Moon Water Guide"
    python -m src.unified_orchestrator phone --task "Post to Instagram"
    python -m src.unified_orchestrator social --site witchcraft --platforms pinterest,instagram
    python -m src.unified_orchestrator revenue --sites witchcraft,smarthome
    python -m src.unified_orchestrator maintain --site witchcraft
    python -m src.unified_orchestrator devices
    python -m src.unified_orchestrator converse --message "How are our sites doing?"
    python -m src.unified_orchestrator missions --status completed --limit 20
    python -m src.unified_orchestrator stats
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import importlib
import json
import logging
import os
import random
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("unified_orchestrator")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s.%(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "orchestrator"
MISSIONS_FILE = DATA_DIR / "missions.json"
CONVERSATIONS_FILE = DATA_DIR / "conversations.json"
DISPATCH_LOG_FILE = DATA_DIR / "dispatch_log.json"
STATS_FILE = DATA_DIR / "stats.json"
SITE_REGISTRY_PATH = BASE_DIR / "configs" / "site-registry.json"

# Ensure data directory exists on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_MISSIONS = 2000
MAX_CONVERSATIONS = 500
MAX_DISPATCH_LOG = 10000
MAX_MISSION_STEPS = 50
MAX_CONVERSATION_MESSAGES = 200
MAX_RETRIES_DEFAULT = 3
BASE_DELAY_DEFAULT = 1.0
MAX_DELAY_DEFAULT = 30.0
CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_RECOVERY_TIMEOUT = 60.0
DISPATCH_TIMEOUT_DEFAULT = 120.0

# Anthropic model strings
SONNET_MODEL = "claude-sonnet-4-20250514"
HAIKU_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(str(tmp), str(path))


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _run_sync(coro):
    """Run an async coroutine from sync context, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _gen_id(prefix: str = "") -> str:
    """Generate a short unique identifier with optional prefix."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MissionType(str, Enum):
    """Types of missions the orchestrator can execute."""
    CONTENT_PUBLISH = "content_publish"
    SOCIAL_GROWTH = "social_growth"
    ACCOUNT_CREATION = "account_creation"
    APP_EXPLORATION = "app_exploration"
    MONETIZATION = "monetization"
    SITE_MAINTENANCE = "site_maintenance"
    REVENUE_CHECK = "revenue_check"
    DEVICE_MAINTENANCE = "device_maintenance"
    SUBSTACK_DAILY = "substack_daily"


class MissionStatus(str, Enum):
    """Lifecycle states for a mission."""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Lifecycle states for a mission step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DispatchResult(str, Enum):
    """Outcome of a dispatch call through the circuit breaker."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    RETRY_EXHAUSTED = "retry_exhausted"


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MissionStep:
    """A single step within a mission, targeting a specific module method."""
    step_id: str = field(default_factory=lambda: _gen_id("step-"))
    module: str = ""
    method: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    retries: int = 0
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> MissionStep:
        data = dict(data)
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = StepStatus(data["status"])
            except ValueError:
                data["status"] = StepStatus.PENDING
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Mission:
    """A structured multi-step operation with lifecycle management."""
    mission_id: str = field(default_factory=lambda: _gen_id("msn-"))
    mission_type: MissionType = MissionType.CONTENT_PUBLISH
    description: str = ""
    status: MissionStatus = MissionStatus.PENDING
    steps: List[MissionStep] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    timestamps: Dict[str, str] = field(default_factory=dict)
    total_duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "mission_type": self.mission_type.value,
            "description": self.description,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "params": self.params,
            "result": self.result,
            "timestamps": self.timestamps,
            "total_duration": self.total_duration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Mission:
        data = dict(data)
        if "mission_type" in data and isinstance(data["mission_type"], str):
            try:
                data["mission_type"] = MissionType(data["mission_type"])
            except ValueError:
                data["mission_type"] = MissionType.CONTENT_PUBLISH
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = MissionStatus(data["status"])
            except ValueError:
                data["status"] = MissionStatus.PENDING
        steps_raw = data.pop("steps", [])
        steps = [MissionStep.from_dict(s) if isinstance(s, dict) else s for s in steps_raw]
        obj = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        obj.steps = steps
        return obj


@dataclass
class ConversationMessage:
    """A single message in a conversation session."""
    role: str = "user"  # user | assistant | system
    content: str = ""
    timestamp: str = field(default_factory=_now_iso)
    context_used: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ConversationMessage:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationSession:
    """A multi-turn conversation session with message history."""
    session_id: str = field(default_factory=lambda: _gen_id("conv-"))
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    last_active: str = field(default_factory=_now_iso)
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "last_active": self.last_active,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConversationSession:
        data = dict(data)
        msgs_raw = data.pop("messages", [])
        msgs = [ConversationMessage.from_dict(m) if isinstance(m, dict) else m for m in msgs_raw]
        obj = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        obj.messages = msgs
        return obj


@dataclass
class DispatchLog:
    """Audit record for a single dispatch call."""
    timestamp: str = field(default_factory=_now_iso)
    module: str = ""
    method: str = ""
    success: bool = False
    duration: float = 0.0
    error: str = ""
    circuit_state: str = CircuitState.CLOSED.value
    retries: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DispatchLog:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Internal Circuit Breaker (per-module, lightweight)
# ---------------------------------------------------------------------------

class _ModuleCircuit:
    """
    Per-module circuit breaker with failure counting, open/half-open/closed
    state machine, and recovery timeout.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
        recovery_timeout: float = CIRCUIT_RECOVERY_TIMEOUT,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float = 0.0
        self.last_success_time: float = 0.0
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0

    def can_execute(self) -> bool:
        """Check if the circuit allows execution."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit %s transitioned OPEN -> HALF_OPEN after %.1fs",
                    self.name, elapsed,
                )
                return True
            return False
        # HALF_OPEN: allow one probe request
        return True

    def record_success(self) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.total_successes += 1
        self.success_count += 1
        self.last_success_time = time.monotonic()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info("Circuit %s recovered -> CLOSED", self.name)
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success in closed state
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.state == CircuitState.HALF_OPEN:
            # Probe failed; re-open
            self.state = CircuitState.OPEN
            logger.warning("Circuit %s probe failed -> OPEN", self.name)
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    "Circuit %s tripped -> OPEN after %d failures",
                    self.name, self.failure_count,
                )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
        }


# ---------------------------------------------------------------------------
# Module Registry
# ---------------------------------------------------------------------------

# Mapping: module_name -> (import_path, factory_function_name)
# Factory functions are called with no args to get a singleton/instance.
MODULE_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Phase 1-2: Core empire modules
    "wordpress_client": ("src.wordpress_client", "get_empire_manager"),
    "content_generator": ("src.content_generator", "get_generator"),
    "content_calendar": ("src.content_calendar", "get_calendar"),
    "brand_voice_engine": ("src.brand_voice_engine", "get_engine"),
    "revenue_tracker": ("src.revenue_tracker", "get_tracker"),
    "social_publisher": ("src.social_publisher", "get_publisher"),
    "seo_auditor": ("src.seo_auditor", "get_auditor"),
    "internal_linker": ("src.internal_linker", "get_linker"),
    "content_repurposer": ("src.content_repurposer", "get_repurposer"),
    "notification_hub": ("src.notification_hub", "get_hub"),
    "affiliate_manager": ("src.affiliate_manager", "get_manager"),
    "etsy_manager": ("src.etsy_manager", "get_manager"),
    "kdp_publisher": ("src.kdp_publisher", "get_publisher"),
    "n8n_client": ("src.n8n_client", "get_empire_integration"),
    "task_scheduler": ("src.task_scheduler", "get_scheduler"),
    # Phase 3-4: Phone automation
    "phone_controller": ("src.phone_controller", "PhoneController"),
    "forge_engine": ("src.forge_engine", "get_engine"),
    "amplify_pipeline": ("src.amplify_pipeline", "AmplifyPipeline"),
    "vision_agent": ("src.vision_agent", "VisionAgent"),
    "screenpipe_agent": ("src.screenpipe_agent", "ScreenpipeAgent"),
    "intelligence_hub": ("src.intelligence_hub", "get_hub"),
    # Phase 5: Autonomous phone agent
    "phone_os_agent": ("src.phone_os_agent", "get_phone_os_agent"),
    "browser_controller": ("src.browser_controller", "get_browser"),
    "identity_manager": ("src.identity_manager", "get_identity_manager"),
    "app_learner": ("src.app_learner", "get_app_learner"),
    "app_discovery": ("src.app_discovery", "get_app_discovery"),
    "email_agent": ("src.email_agent", "get_email_agent"),
    "account_factory": ("src.account_factory", "get_account_factory"),
    "social_media_agent": ("src.social_media_agent", "get_social_agent"),
    "agent_memory": ("src.agent_memory", "get_memory"),
    "social_automation": ("src.social_automation", "get_social_bot"),
    "account_manager": ("src.account_manager", "get_account_manager"),
    # Phase 6: Reliability & observability
    "circuit_breaker": ("src.circuit_breaker", "get_breaker_registry"),
    "audit_logger": ("src.audit_logger", "get_audit_logger"),
    "rag_memory": ("src.rag_memory", "get_rag_memory"),
    "backup_manager": ("src.backup_manager", "get_backup_manager"),
    "anomaly_detector": ("src.anomaly_detector", "get_detector"),
    "prompt_library": ("src.prompt_library", "get_prompt_library"),
    "content_quality_scorer": ("src.content_quality_scorer", "get_scorer"),
    "performance_benchmarker": ("src.performance_benchmarker", "get_benchmarker"),
    "phone_farm": ("src.phone_farm", "get_farm"),
    "device_pool": ("src.device_pool", "DevicePool"),
    "geelark_client": ("src.geelark_client", "get_client"),
}

# Modules whose factory function is a class constructor (not a get_ singleton)
# These need to be instantiated with () instead of called as a getter
_CLASS_CONSTRUCTORS = {
    "phone_controller", "amplify_pipeline", "vision_agent",
    "screenpipe_agent", "device_pool",
}


# ---------------------------------------------------------------------------
# Site registry loader
# ---------------------------------------------------------------------------

_site_registry_cache: Optional[Dict[str, Any]] = None


def _load_site_registry() -> Dict[str, Any]:
    """Load the site registry, caching after first read."""
    global _site_registry_cache
    if _site_registry_cache is None:
        _site_registry_cache = _load_json(SITE_REGISTRY_PATH, {})
    return _site_registry_cache


def _get_all_site_ids() -> List[str]:
    """Return all site IDs from the registry."""
    registry = _load_site_registry()
    sites = registry.get("sites", registry)
    return list(sites.keys())


# ---------------------------------------------------------------------------
# Mission Templates
# ---------------------------------------------------------------------------

def _content_publish_steps(params: Dict[str, Any]) -> List[MissionStep]:
    """Generate steps for a CONTENT_PUBLISH mission."""
    site_id = params.get("site_id", "")
    title = params.get("title", "")
    topic = params.get("topic", title)
    keywords = params.get("keywords", [])
    post_id = params.get("post_id")

    steps = [
        MissionStep(
            step_id=_gen_id("step-"),
            module="brand_voice_engine",
            method="get_voice_profile",
            kwargs={"site_id": site_id},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="content_generator",
            method="research_topic",
            kwargs={"site_id": site_id, "topic": topic},
            depends_on=[],
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="content_generator",
            method="generate_outline",
            kwargs={"site_id": site_id, "title": title, "keywords": keywords},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="content_generator",
            method="generate_full_article",
            kwargs={"site_id": site_id, "title": title, "keywords": keywords},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="content_quality_scorer",
            method="score_article",
            kwargs={"site_id": site_id},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="seo_auditor",
            method="audit_post_content",
            kwargs={"site_id": site_id},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="internal_linker",
            method="suggest_links",
            kwargs={"site_id": site_id},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="wordpress_client",
            method="create_post",
            kwargs={"site_id": site_id, "title": title, "status": "draft"},
        ),
    ]

    # Optional: set featured image if post_id is provided
    if post_id:
        steps.append(
            MissionStep(
                step_id=_gen_id("step-"),
                module="wordpress_client",
                method="set_featured_image",
                kwargs={"site_id": site_id, "post_id": post_id},
            )
        )

    # Schedule social media posts
    steps.append(
        MissionStep(
            step_id=_gen_id("step-"),
            module="social_publisher",
            method="create_campaign",
            kwargs={"site_id": site_id, "title": title},
        )
    )

    # Update content calendar
    steps.append(
        MissionStep(
            step_id=_gen_id("step-"),
            module="content_calendar",
            method="mark_published",
            kwargs={"site_id": site_id, "title": title},
        )
    )

    return steps


def _social_growth_steps(params: Dict[str, Any]) -> List[MissionStep]:
    """Generate steps for a SOCIAL_GROWTH mission."""
    site_id = params.get("site_id", "")
    platforms = params.get("platforms", ["pinterest", "instagram", "facebook"])

    steps = [
        MissionStep(
            step_id=_gen_id("step-"),
            module="brand_voice_engine",
            method="get_voice_profile",
            kwargs={"site_id": site_id},
        ),
    ]

    for platform in platforms:
        steps.extend([
            MissionStep(
                step_id=_gen_id("step-"),
                module="social_publisher",
                method="generate_content",
                kwargs={"site_id": site_id, "platform": platform},
            ),
            MissionStep(
                step_id=_gen_id("step-"),
                module="social_publisher",
                method="schedule_post",
                kwargs={"site_id": site_id, "platform": platform},
            ),
        ])

    # Analytics check
    steps.append(
        MissionStep(
            step_id=_gen_id("step-"),
            module="social_publisher",
            method="get_stats",
            kwargs={"site_id": site_id},
        )
    )

    return steps


def _account_creation_steps(params: Dict[str, Any]) -> List[MissionStep]:
    """Generate steps for an ACCOUNT_CREATION mission."""
    platform = params.get("platform", "")
    niche = params.get("niche", "")
    details = params.get("details", {})

    return [
        MissionStep(
            step_id=_gen_id("step-"),
            module="identity_manager",
            method="generate_persona",
            kwargs={"niche": niche, **details},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="email_agent",
            method="create_email_account",
            kwargs={"niche": niche},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="account_factory",
            method="create_account",
            kwargs={"platform": platform, "niche": niche},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="account_manager",
            method="register_account",
            kwargs={"platform": platform, "niche": niche},
        ),
    ]


def _app_exploration_steps(params: Dict[str, Any]) -> List[MissionStep]:
    """Generate steps for an APP_EXPLORATION mission."""
    app_name = params.get("app_name", "")
    goals = params.get("goals", [])

    return [
        MissionStep(
            step_id=_gen_id("step-"),
            module="app_discovery",
            method="search_play_store",
            kwargs={"query": app_name},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="phone_os_agent",
            method="launch_app",
            kwargs={"package_name": app_name},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="app_learner",
            method="explore_app",
            kwargs={"app_name": app_name, "goals": goals},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="app_learner",
            method="generate_playbook",
            kwargs={"app_name": app_name},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="agent_memory",
            method="store",
            kwargs={"category": "app_exploration", "app": app_name},
        ),
    ]


def _monetization_steps(params: Dict[str, Any]) -> List[MissionStep]:
    """Generate steps for a MONETIZATION mission."""
    site_id = params.get("site_id", "")
    stream = params.get("stream", "all")

    steps = [
        MissionStep(
            step_id=_gen_id("step-"),
            module="revenue_tracker",
            method="get_daily_summary",
            kwargs={},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="affiliate_manager",
            method="check_opportunities",
            kwargs={"site_id": site_id},
        ),
    ]

    if stream in ("all", "kdp"):
        steps.append(
            MissionStep(
                step_id=_gen_id("step-"),
                module="kdp_publisher",
                method="get_pipeline_status",
                kwargs={},
            )
        )

    if stream in ("all", "etsy"):
        steps.append(
            MissionStep(
                step_id=_gen_id("step-"),
                module="etsy_manager",
                method="monthly_report",
                kwargs={},
            )
        )

    steps.append(
        MissionStep(
            step_id=_gen_id("step-"),
            module="notification_hub",
            method="send",
            kwargs={
                "title": "Monetization Report",
                "body": f"Monetization check completed for {site_id or 'all sites'}",
            },
        )
    )

    return steps


def _site_maintenance_steps(params: Dict[str, Any]) -> List[MissionStep]:
    """Generate steps for a SITE_MAINTENANCE mission."""
    site_id = params.get("site_id", "")

    return [
        MissionStep(
            step_id=_gen_id("step-"),
            module="wordpress_client",
            method="health_check",
            kwargs={"site_id": site_id},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="seo_auditor",
            method="quick_audit",
            kwargs={"site_id": site_id},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="internal_linker",
            method="audit_links",
            kwargs={"site_id": site_id},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="content_quality_scorer",
            method="scan_site",
            kwargs={"site_id": site_id},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="backup_manager",
            method="create_selective_backup",
            kwargs={"directories": ["content", "calendar"]},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="anomaly_detector",
            method="check_site",
            kwargs={"site_id": site_id},
        ),
    ]


def _revenue_check_steps(params: Dict[str, Any]) -> List[MissionStep]:
    """Generate steps for a REVENUE_CHECK mission."""
    site_ids = params.get("site_ids", [])

    steps = [
        MissionStep(
            step_id=_gen_id("step-"),
            module="revenue_tracker",
            method="get_daily_summary",
            kwargs={},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="revenue_tracker",
            method="get_weekly_trend",
            kwargs={},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="revenue_tracker",
            method="check_goals",
            kwargs={},
        ),
    ]

    for sid in site_ids:
        steps.append(
            MissionStep(
                step_id=_gen_id("step-"),
                module="revenue_tracker",
                method="get_site_revenue",
                kwargs={"site_id": sid},
            )
        )

    steps.append(
        MissionStep(
            step_id=_gen_id("step-"),
            module="revenue_tracker",
            method="format_daily_summary",
            kwargs={},
        )
    )

    return steps


def _device_maintenance_steps(params: Dict[str, Any]) -> List[MissionStep]:
    """Generate steps for a DEVICE_MAINTENANCE mission."""
    return [
        MissionStep(
            step_id=_gen_id("step-"),
            module="phone_farm",
            method="health_check_all",
            kwargs={},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="device_pool",
            method="status",
            kwargs={},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="geelark_client",
            method="list_profiles",
            kwargs={},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="phone_os_agent",
            method="get_device_info",
            kwargs={},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="phone_farm",
            method="cleanup_stale_sessions",
            kwargs={},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="backup_manager",
            method="create_selective_backup",
            kwargs={"directories": ["forge", "amplify", "vision", "memory"]},
        ),
    ]


def _substack_daily_steps(params: Dict[str, Any]) -> List[MissionStep]:
    """Generate steps for a SUBSTACK_DAILY mission."""
    account_id = params.get("account_id", "witchcraft")

    return [
        MissionStep(
            step_id=_gen_id("step-"),
            module="brand_voice_engine",
            method="get_voice_profile",
            kwargs={"site_id": account_id},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="content_calendar",
            method="get_next_scheduled",
            kwargs={"site_id": account_id, "channel": "substack"},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="content_generator",
            method="generate_newsletter_content",
            kwargs={"site_id": account_id},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="content_quality_scorer",
            method="score_article",
            kwargs={"site_id": account_id, "format": "newsletter"},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="social_publisher",
            method="create_campaign",
            kwargs={"site_id": account_id, "channel": "substack"},
        ),
        MissionStep(
            step_id=_gen_id("step-"),
            module="content_calendar",
            method="mark_published",
            kwargs={"site_id": account_id, "channel": "substack"},
        ),
    ]


# Template registry: MissionType -> step generator function
MISSION_TEMPLATES: Dict[MissionType, Callable[[Dict[str, Any]], List[MissionStep]]] = {
    MissionType.CONTENT_PUBLISH: _content_publish_steps,
    MissionType.SOCIAL_GROWTH: _social_growth_steps,
    MissionType.ACCOUNT_CREATION: _account_creation_steps,
    MissionType.APP_EXPLORATION: _app_exploration_steps,
    MissionType.MONETIZATION: _monetization_steps,
    MissionType.SITE_MAINTENANCE: _site_maintenance_steps,
    MissionType.REVENUE_CHECK: _revenue_check_steps,
    MissionType.DEVICE_MAINTENANCE: _device_maintenance_steps,
    MissionType.SUBSTACK_DAILY: _substack_daily_steps,
}


# ===================================================================
# UnifiedOrchestrator
# ===================================================================

class UnifiedOrchestrator:
    """
    The unified brain of the OpenClaw Empire.

    Merges the goal-driven autonomy of AutonomousAgent with the subsystem
    coordination of IntelligenceHub. Provides:

    - Mission-based execution with step-level fault tolerance
    - Circuit-breaker-protected dispatch to any of ~25 empire modules
    - RAG-enhanced conversational interface (Claude Sonnet)
    - FORGE + AMPLIFY integration for phone automation tasks
    - Revenue, social, maintenance, and device management workflows
    - Full audit trail with dispatch logging and mission history

    All state is persisted atomically to data/orchestrator/.
    """

    _instance: Optional[UnifiedOrchestrator] = None

    def __init__(self) -> None:
        # Module instance cache (lazy-loaded)
        self._module_cache: Dict[str, Any] = {}

        # Per-module circuit breakers
        self._circuits: Dict[str, _ModuleCircuit] = {}

        # Mission storage
        self._missions: Dict[str, Mission] = {}
        self._load_missions()

        # Conversation sessions
        self._sessions: Dict[str, ConversationSession] = {}
        self._load_conversations()

        # Dispatch log (in-memory ring buffer, periodically flushed)
        self._dispatch_log: List[DispatchLog] = []
        self._load_dispatch_log()

        # Stats
        self._stats: Dict[str, Any] = _load_json(STATS_FILE, {
            "total_missions": 0,
            "completed_missions": 0,
            "failed_missions": 0,
            "total_dispatches": 0,
            "successful_dispatches": 0,
            "failed_dispatches": 0,
            "total_conversations": 0,
            "total_messages": 0,
            "circuit_opens": 0,
            "retries_total": 0,
            "modules_loaded": 0,
            "uptime_started": _now_iso(),
        })

        logger.info(
            "UnifiedOrchestrator initialized: %d missions, %d sessions, %d dispatch entries",
            len(self._missions),
            len(self._sessions),
            len(self._dispatch_log),
        )

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def _load_missions(self) -> None:
        """Load missions from disk."""
        data = _load_json(MISSIONS_FILE, [])
        if isinstance(data, list):
            for m in data:
                if isinstance(m, dict):
                    mission = Mission.from_dict(m)
                    self._missions[mission.mission_id] = mission
        elif isinstance(data, dict):
            for mid, m in data.items():
                if isinstance(m, dict):
                    mission = Mission.from_dict(m)
                    self._missions[mission.mission_id] = mission

    def _save_missions(self) -> None:
        """Persist missions to disk, bounded at MAX_MISSIONS."""
        # Keep most recent missions
        all_missions = sorted(
            self._missions.values(),
            key=lambda m: m.timestamps.get("created", ""),
            reverse=True,
        )[:MAX_MISSIONS]
        self._missions = {m.mission_id: m for m in all_missions}
        _save_json(MISSIONS_FILE, [m.to_dict() for m in all_missions])

    def _load_conversations(self) -> None:
        """Load conversation sessions from disk."""
        data = _load_json(CONVERSATIONS_FILE, [])
        if isinstance(data, list):
            for s in data:
                if isinstance(s, dict):
                    session = ConversationSession.from_dict(s)
                    self._sessions[session.session_id] = session

    def _save_conversations(self) -> None:
        """Persist conversation sessions to disk, bounded."""
        all_sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.last_active,
            reverse=True,
        )[:MAX_CONVERSATIONS]
        self._sessions = {s.session_id: s for s in all_sessions}
        _save_json(CONVERSATIONS_FILE, [s.to_dict() for s in all_sessions])

    def _load_dispatch_log(self) -> None:
        """Load dispatch log from disk."""
        data = _load_json(DISPATCH_LOG_FILE, [])
        if isinstance(data, list):
            self._dispatch_log = [
                DispatchLog.from_dict(d) if isinstance(d, dict) else d
                for d in data[-MAX_DISPATCH_LOG:]
            ]

    def _save_dispatch_log(self) -> None:
        """Persist dispatch log to disk, bounded."""
        self._dispatch_log = self._dispatch_log[-MAX_DISPATCH_LOG:]
        _save_json(DISPATCH_LOG_FILE, [d.to_dict() for d in self._dispatch_log])

    def _save_stats(self) -> None:
        """Persist stats to disk."""
        _save_json(STATS_FILE, self._stats)

    def _persist_all(self) -> None:
        """Flush all state to disk."""
        self._save_missions()
        self._save_conversations()
        self._save_dispatch_log()
        self._save_stats()

    # -------------------------------------------------------------------
    # Module loading (lazy importlib)
    # -------------------------------------------------------------------

    def _get_module_instance(self, module_name: str) -> Any:
        """
        Lazily import and instantiate a module from MODULE_REGISTRY.

        Returns the cached instance if already loaded. Uses importlib to
        avoid circular imports and allow partial availability.
        """
        if module_name in self._module_cache:
            return self._module_cache[module_name]

        if module_name not in MODULE_REGISTRY:
            raise ValueError(
                f"Unknown module '{module_name}'. "
                f"Available: {', '.join(sorted(MODULE_REGISTRY.keys()))}"
            )

        import_path, factory_name = MODULE_REGISTRY[module_name]

        try:
            mod = importlib.import_module(import_path)
            factory = getattr(mod, factory_name)

            # Class constructors need to be called with ()
            # Singleton getters are already called with ()
            instance = factory()

            self._module_cache[module_name] = instance
            self._stats["modules_loaded"] = len(self._module_cache)
            logger.info("Loaded module: %s via %s.%s", module_name, import_path, factory_name)
            return instance

        except Exception as exc:
            logger.error(
                "Failed to load module '%s' from %s.%s: %s",
                module_name, import_path, factory_name, exc,
            )
            raise RuntimeError(f"Module '{module_name}' unavailable: {exc}") from exc

    def _get_circuit(self, module_name: str) -> _ModuleCircuit:
        """Get or create the circuit breaker for a module."""
        if module_name not in self._circuits:
            self._circuits[module_name] = _ModuleCircuit(module_name)
        return self._circuits[module_name]

    # -------------------------------------------------------------------
    # Core Dispatch — circuit breaker + retry + audit logging
    # -------------------------------------------------------------------

    async def dispatch(
        self,
        module: str,
        method: str,
        *,
        max_retries: int = MAX_RETRIES_DEFAULT,
        base_delay: float = BASE_DELAY_DEFAULT,
        timeout: float = DISPATCH_TIMEOUT_DEFAULT,
        **kwargs: Any,
    ) -> dict:
        """
        Dispatch a method call to any empire module with circuit breaker
        protection, retry with exponential backoff, and full audit logging.

        Args:
            module: Module name from MODULE_REGISTRY (e.g., "content_generator")
            method: Method name to call on the module instance
            max_retries: Maximum retry attempts on transient failures
            base_delay: Initial backoff delay in seconds
            timeout: Total timeout for the dispatch (seconds)
            **kwargs: Arguments passed to the target method

        Returns:
            dict with keys: status (DispatchResult), result, duration,
            retries, module, method, circuit_state
        """
        circuit = self._get_circuit(module)
        start_time = time.monotonic()
        last_error = ""
        retries_used = 0

        # Check circuit state
        if not circuit.can_execute():
            log_entry = DispatchLog(
                module=module,
                method=method,
                success=False,
                duration=0.0,
                error="Circuit breaker open",
                circuit_state=circuit.state.value,
                retries=0,
            )
            self._dispatch_log.append(log_entry)
            self._stats["total_dispatches"] = self._stats.get("total_dispatches", 0) + 1
            self._stats["failed_dispatches"] = self._stats.get("failed_dispatches", 0) + 1
            self._stats["circuit_opens"] = self._stats.get("circuit_opens", 0) + 1

            logger.warning(
                "DISPATCH BLOCKED: %s.%s — circuit %s",
                module, method, circuit.state.value,
            )

            return {
                "status": DispatchResult.CIRCUIT_OPEN,
                "result": None,
                "duration": 0.0,
                "retries": 0,
                "module": module,
                "method": method,
                "circuit_state": circuit.state.value,
                "error": "Circuit breaker is open; service is degraded",
            }

        # Retry loop with exponential backoff
        for attempt in range(max_retries + 1):
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                last_error = f"Dispatch timeout after {elapsed:.1f}s"
                break

            try:
                instance = self._get_module_instance(module)
                target = getattr(instance, method)

                # Call the method (handle both sync and async)
                if asyncio.iscoroutinefunction(target):
                    result = await asyncio.wait_for(
                        target(**kwargs),
                        timeout=max(1.0, timeout - elapsed),
                    )
                else:
                    result = target(**kwargs)

                duration = time.monotonic() - start_time
                circuit.record_success()

                # Log success
                log_entry = DispatchLog(
                    module=module,
                    method=method,
                    success=True,
                    duration=duration,
                    circuit_state=circuit.state.value,
                    retries=retries_used,
                )
                self._dispatch_log.append(log_entry)
                self._stats["total_dispatches"] = self._stats.get("total_dispatches", 0) + 1
                self._stats["successful_dispatches"] = self._stats.get("successful_dispatches", 0) + 1
                self._stats["retries_total"] = self._stats.get("retries_total", 0) + retries_used

                logger.info(
                    "DISPATCH OK: %s.%s in %.2fs (retries=%d)",
                    module, method, duration, retries_used,
                )

                return {
                    "status": DispatchResult.SUCCESS,
                    "result": result,
                    "duration": duration,
                    "retries": retries_used,
                    "module": module,
                    "method": method,
                    "circuit_state": circuit.state.value,
                    "error": "",
                }

            except asyncio.TimeoutError:
                last_error = f"Method {module}.{method} timed out"
                retries_used += 1
                circuit.record_failure()
                logger.warning(
                    "DISPATCH TIMEOUT: %s.%s attempt %d/%d",
                    module, method, attempt + 1, max_retries + 1,
                )

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                retries_used += 1
                circuit.record_failure()
                logger.warning(
                    "DISPATCH ERROR: %s.%s attempt %d/%d — %s",
                    module, method, attempt + 1, max_retries + 1, last_error,
                )

            # Exponential backoff with jitter
            if attempt < max_retries:
                delay = min(
                    base_delay * (2 ** attempt) + random.uniform(0, base_delay),
                    MAX_DELAY_DEFAULT,
                )
                remaining = timeout - (time.monotonic() - start_time)
                if remaining <= delay:
                    break
                await asyncio.sleep(delay)

        # All retries exhausted
        duration = time.monotonic() - start_time
        log_entry = DispatchLog(
            module=module,
            method=method,
            success=False,
            duration=duration,
            error=last_error,
            circuit_state=circuit.state.value,
            retries=retries_used,
        )
        self._dispatch_log.append(log_entry)
        self._stats["total_dispatches"] = self._stats.get("total_dispatches", 0) + 1
        self._stats["failed_dispatches"] = self._stats.get("failed_dispatches", 0) + 1
        self._stats["retries_total"] = self._stats.get("retries_total", 0) + retries_used

        logger.error(
            "DISPATCH FAILED: %s.%s after %d retries in %.2fs — %s",
            module, method, retries_used, duration, last_error,
        )

        # Determine if it was a timeout or retry exhaustion
        status = DispatchResult.TIMEOUT if "timed out" in last_error.lower() else DispatchResult.RETRY_EXHAUSTED

        return {
            "status": status,
            "result": None,
            "duration": duration,
            "retries": retries_used,
            "module": module,
            "method": method,
            "circuit_state": circuit.state.value,
            "error": last_error,
        }

    def dispatch_sync(self, module: str, method: str, **kwargs: Any) -> dict:
        """Synchronous wrapper for dispatch()."""
        return _run_sync(self.dispatch(module, method, **kwargs))

    # -------------------------------------------------------------------
    # Mission Planning & Execution
    # -------------------------------------------------------------------

    async def plan_mission(
        self,
        mission_type: MissionType,
        params: Dict[str, Any],
        description: str = "",
    ) -> Mission:
        """
        Plan a mission without executing it. Creates the mission with
        all steps in PENDING status and saves it to disk.

        Args:
            mission_type: The type of mission to plan
            params: Parameters for the mission template
            description: Human-readable description

        Returns:
            The planned Mission object
        """
        if mission_type not in MISSION_TEMPLATES:
            raise ValueError(
                f"No template for mission type '{mission_type.value}'. "
                f"Available: {[mt.value for mt in MISSION_TEMPLATES]}"
            )

        template_fn = MISSION_TEMPLATES[mission_type]
        steps = template_fn(params)

        if not description:
            description = f"{mission_type.value} mission: {json.dumps(params, default=str)[:200]}"

        mission = Mission(
            mission_type=mission_type,
            description=description,
            status=MissionStatus.PLANNING,
            steps=steps,
            params=params,
            timestamps={"created": _now_iso(), "planned": _now_iso()},
        )

        self._missions[mission.mission_id] = mission
        self._stats["total_missions"] = self._stats.get("total_missions", 0) + 1
        self._save_missions()

        logger.info(
            "Mission planned: %s (%s) with %d steps",
            mission.mission_id, mission_type.value, len(steps),
        )

        return mission

    def plan_mission_sync(self, mission_type: MissionType, params: Dict[str, Any], **kw) -> Mission:
        """Synchronous wrapper for plan_mission()."""
        return _run_sync(self.plan_mission(mission_type, params, **kw))

    async def execute_planned(self, mission_id: str) -> Mission:
        """
        Execute a previously planned mission by ID.

        Args:
            mission_id: The ID of the mission to execute

        Returns:
            The completed Mission object
        """
        mission = self._missions.get(mission_id)
        if not mission:
            raise ValueError(f"Mission '{mission_id}' not found")

        if mission.status not in (MissionStatus.PENDING, MissionStatus.PLANNING):
            raise ValueError(
                f"Mission '{mission_id}' is in status '{mission.status.value}', "
                f"expected PENDING or PLANNING"
            )

        return await self._execute_mission_steps(mission)

    def execute_planned_sync(self, mission_id: str) -> Mission:
        """Synchronous wrapper for execute_planned()."""
        return _run_sync(self.execute_planned(mission_id))

    async def execute_mission(
        self,
        mission_type: MissionType,
        params: Dict[str, Any],
        description: str = "",
    ) -> Mission:
        """
        Plan and immediately execute a mission.

        Args:
            mission_type: The type of mission
            params: Parameters for the template
            description: Human-readable description

        Returns:
            The completed Mission object
        """
        mission = await self.plan_mission(mission_type, params, description)
        return await self._execute_mission_steps(mission)

    def execute_mission_sync(self, mission_type: MissionType, params: Dict[str, Any], **kw) -> Mission:
        """Synchronous wrapper for execute_mission()."""
        return _run_sync(self.execute_mission(mission_type, params, **kw))

    async def _execute_mission_steps(self, mission: Mission) -> Mission:
        """
        Internal: execute all steps of a mission sequentially, respecting
        dependencies and recording per-step timing and results.
        """
        mission.status = MissionStatus.EXECUTING
        mission.timestamps["started"] = _now_iso()
        mission_start = time.monotonic()
        completed_step_ids: Set[str] = set()
        all_results: Dict[str, Any] = {}

        logger.info(
            "Executing mission %s (%s): %d steps",
            mission.mission_id, mission.mission_type.value, len(mission.steps),
        )

        for step in mission.steps:
            # Check dependencies
            unmet = [dep for dep in step.depends_on if dep not in completed_step_ids]
            if unmet:
                # Check if all dependencies actually failed (not just pending)
                dep_failed = any(
                    self._get_step_by_id(mission, dep_id) is not None
                    and self._get_step_by_id(mission, dep_id).status == StepStatus.FAILED
                    for dep_id in unmet
                )
                if dep_failed:
                    step.status = StepStatus.SKIPPED
                    step.error = f"Skipped: dependency failed ({', '.join(unmet)})"
                    logger.info(
                        "Step %s skipped: dependency failed", step.step_id,
                    )
                    continue
                # Dependencies not yet completed — this shouldn't happen in
                # sequential execution, but handle it gracefully
                logger.warning(
                    "Step %s has unmet dependencies: %s — attempting anyway",
                    step.step_id, unmet,
                )

            # Execute the step via dispatch
            step.status = StepStatus.RUNNING
            step.started_at = _now_iso()
            step_start = time.monotonic()

            try:
                # Inject results from prior steps into kwargs if needed
                enriched_kwargs = dict(step.kwargs)

                # Forward-chain: if previous step produced content/article,
                # inject it into dependent steps
                if step.module == "wordpress_client" and step.method == "create_post":
                    # Look for generated article content from content_generator
                    for prev_id, prev_result in all_results.items():
                        if isinstance(prev_result, dict):
                            if "html" in prev_result:
                                enriched_kwargs.setdefault("content", prev_result["html"])
                            elif "content" in prev_result:
                                enriched_kwargs.setdefault("content", prev_result["content"])

                dispatch_result = await self.dispatch(
                    step.module,
                    step.method,
                    **enriched_kwargs,
                )

                step.duration_seconds = time.monotonic() - step_start
                step.completed_at = _now_iso()
                step.retries = dispatch_result.get("retries", 0)

                if dispatch_result["status"] == DispatchResult.SUCCESS:
                    step.status = StepStatus.COMPLETED
                    step.result = dispatch_result.get("result")
                    all_results[step.step_id] = step.result
                    completed_step_ids.add(step.step_id)
                    logger.info(
                        "Step %s completed: %s.%s in %.2fs",
                        step.step_id, step.module, step.method, step.duration_seconds,
                    )
                else:
                    step.status = StepStatus.FAILED
                    step.error = dispatch_result.get("error", "Unknown dispatch failure")
                    logger.warning(
                        "Step %s failed: %s.%s — %s",
                        step.step_id, step.module, step.method, step.error,
                    )

            except Exception as exc:
                step.status = StepStatus.FAILED
                step.error = f"{type(exc).__name__}: {exc}"
                step.duration_seconds = time.monotonic() - step_start
                step.completed_at = _now_iso()
                logger.error(
                    "Step %s exception: %s.%s — %s",
                    step.step_id, step.module, step.method, step.error,
                )

        # Determine overall mission status
        mission.total_duration = time.monotonic() - mission_start
        mission.timestamps["completed"] = _now_iso()

        completed = sum(1 for s in mission.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in mission.steps if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in mission.steps if s.status == StepStatus.SKIPPED)
        total = len(mission.steps)

        if failed == 0 and completed == total:
            mission.status = MissionStatus.COMPLETED
            self._stats["completed_missions"] = self._stats.get("completed_missions", 0) + 1
        elif completed > 0:
            # Partial success — still mark completed but with partial result
            mission.status = MissionStatus.COMPLETED
            self._stats["completed_missions"] = self._stats.get("completed_missions", 0) + 1
        else:
            mission.status = MissionStatus.FAILED
            self._stats["failed_missions"] = self._stats.get("failed_missions", 0) + 1

        mission.result = {
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "total": total,
            "all_results": {k: str(v)[:500] if v else None for k, v in all_results.items()},
        }

        self._save_missions()
        self._save_dispatch_log()
        self._save_stats()

        logger.info(
            "Mission %s %s: %d/%d steps completed, %d failed, %d skipped in %.2fs",
            mission.mission_id, mission.status.value,
            completed, total, failed, skipped, mission.total_duration,
        )

        return mission

    @staticmethod
    def _get_step_by_id(mission: Mission, step_id: str) -> Optional[MissionStep]:
        """Find a step within a mission by its ID."""
        for step in mission.steps:
            if step.step_id == step_id:
                return step
        return None

    # -------------------------------------------------------------------
    # Mission queries
    # -------------------------------------------------------------------

    def get_mission(self, mission_id: str) -> Optional[Mission]:
        """Get a mission by ID."""
        return self._missions.get(mission_id)

    def list_missions(
        self,
        status: Optional[MissionStatus] = None,
        mission_type: Optional[MissionType] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Mission]:
        """
        List missions with optional filtering by status and type.

        Args:
            status: Filter by mission status
            mission_type: Filter by mission type
            limit: Maximum number of results
            offset: Skip this many results

        Returns:
            List of matching Mission objects
        """
        missions = list(self._missions.values())

        if status is not None:
            missions = [m for m in missions if m.status == status]

        if mission_type is not None:
            missions = [m for m in missions if m.mission_type == mission_type]

        # Sort by creation time descending
        missions.sort(key=lambda m: m.timestamps.get("created", ""), reverse=True)

        return missions[offset:offset + limit]

    async def cancel_mission(self, mission_id: str) -> Mission:
        """Cancel a pending or executing mission."""
        mission = self._missions.get(mission_id)
        if not mission:
            raise ValueError(f"Mission '{mission_id}' not found")

        if mission.status in (MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.CANCELLED):
            raise ValueError(
                f"Mission '{mission_id}' already in terminal status '{mission.status.value}'"
            )

        mission.status = MissionStatus.CANCELLED
        mission.timestamps["cancelled"] = _now_iso()

        # Cancel any pending/running steps
        for step in mission.steps:
            if step.status in (StepStatus.PENDING, StepStatus.RUNNING):
                step.status = StepStatus.SKIPPED
                step.error = "Mission cancelled"

        self._save_missions()
        logger.info("Mission %s cancelled", mission_id)
        return mission

    def cancel_mission_sync(self, mission_id: str) -> Mission:
        """Synchronous wrapper for cancel_mission()."""
        return _run_sync(self.cancel_mission(mission_id))

    def get_mission_stats(self) -> Dict[str, Any]:
        """Get aggregate mission statistics."""
        missions = list(self._missions.values())
        by_type: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        durations: List[float] = []

        for m in missions:
            by_type[m.mission_type.value] = by_type.get(m.mission_type.value, 0) + 1
            by_status[m.status.value] = by_status.get(m.status.value, 0) + 1
            if m.total_duration > 0:
                durations.append(m.total_duration)

        avg_duration = sum(durations) / len(durations) if durations else 0
        total_steps = sum(len(m.steps) for m in missions)
        completed_steps = sum(
            sum(1 for s in m.steps if s.status == StepStatus.COMPLETED)
            for m in missions
        )

        return {
            "total_missions": len(missions),
            "by_type": by_type,
            "by_status": by_status,
            "average_duration_seconds": round(avg_duration, 2),
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "step_success_rate": round(completed_steps / total_steps * 100, 1) if total_steps else 0,
            "modules_loaded": len(self._module_cache),
            "circuits": {name: c.to_dict() for name, c in self._circuits.items()},
            "global_stats": dict(self._stats),
        }

    # -------------------------------------------------------------------
    # High-level operations (convenience wrappers around missions)
    # -------------------------------------------------------------------

    async def publish_content(
        self,
        site_id: str,
        title: Optional[str] = None,
        topic: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        post_id: Optional[int] = None,
    ) -> Mission:
        """
        Publish content to a WordPress site via the content pipeline.

        Delegates to a CONTENT_PUBLISH mission with full voice matching,
        SEO optimization, quality scoring, and social media scheduling.

        Args:
            site_id: Target site ID (e.g., "witchcraft")
            title: Article title (generated from topic if not provided)
            topic: Topic for research (defaults to title)
            keywords: SEO keywords
            post_id: Existing post ID to update

        Returns:
            Completed Mission object
        """
        params = {
            "site_id": site_id,
            "title": title or topic or "Untitled",
            "topic": topic or title or "Untitled",
            "keywords": keywords or [],
        }
        if post_id:
            params["post_id"] = post_id

        return await self.execute_mission(
            MissionType.CONTENT_PUBLISH,
            params,
            description=f"Publish '{params['title']}' to {site_id}",
        )

    def publish_content_sync(self, site_id: str, **kw) -> Mission:
        """Synchronous wrapper for publish_content()."""
        return _run_sync(self.publish_content(site_id, **kw))

    async def execute_phone_task(
        self,
        description: str,
        niche: Optional[str] = None,
        app: Optional[str] = None,
    ) -> dict:
        """
        Execute a phone automation task with FORGE pre-analysis and
        AMPLIFY enhancement.

        Flow: FORGE scout -> AMPLIFY enhance -> IntelligenceHub execute -> record

        Args:
            description: Natural language task description
            niche: Optional niche context for persona selection
            app: Optional target app name

        Returns:
            dict with task results and learning data
        """
        result = {
            "task": description,
            "niche": niche,
            "app": app,
            "forge_analysis": None,
            "amplify_enhanced": False,
            "execution_result": None,
            "recorded": False,
            "error": None,
        }

        # Step 1: FORGE pre-analysis
        try:
            forge_result = await self.dispatch(
                "forge_engine", "full_pre_task_analysis",
                task_description=description,
                app=app or "unknown",
            )
            if forge_result["status"] == DispatchResult.SUCCESS:
                result["forge_analysis"] = forge_result["result"]
                logger.info("FORGE analysis complete for phone task")
            else:
                logger.warning("FORGE analysis failed: %s", forge_result.get("error"))
        except Exception as exc:
            logger.warning("FORGE pre-analysis failed: %s", exc)

        # Step 2: AMPLIFY enhancement
        try:
            amplify_result = await self.dispatch(
                "amplify_pipeline", "full_pipeline",
                task_config={
                    "description": description,
                    "app": app,
                    "niche": niche,
                    "forge_analysis": result["forge_analysis"],
                },
            )
            if amplify_result["status"] == DispatchResult.SUCCESS:
                result["amplify_enhanced"] = True
                logger.info("AMPLIFY enhancement complete")
        except Exception as exc:
            logger.warning("AMPLIFY enhancement failed: %s", exc)

        # Step 3: Execute via IntelligenceHub
        try:
            exec_result = await self.dispatch(
                "intelligence_hub", "execute_task",
                description=description,
                app=app,
            )
            result["execution_result"] = exec_result.get("result")
            if exec_result["status"] == DispatchResult.SUCCESS:
                logger.info("Phone task executed successfully")
            else:
                result["error"] = exec_result.get("error")
                logger.warning("Phone task execution failed: %s", result["error"])
        except Exception as exc:
            result["error"] = str(exc)
            logger.error("Phone task execution exception: %s", exc)

        # Step 4: Record to agent memory
        try:
            await self.dispatch(
                "agent_memory", "store",
                category="phone_task",
                data={
                    "description": description,
                    "niche": niche,
                    "app": app,
                    "success": result["error"] is None,
                    "timestamp": _now_iso(),
                },
            )
            result["recorded"] = True
        except Exception as exc:
            logger.warning("Failed to record phone task to memory: %s", exc)

        return result

    def execute_phone_task_sync(self, description: str, **kw) -> dict:
        """Synchronous wrapper for execute_phone_task()."""
        return _run_sync(self.execute_phone_task(description, **kw))

    async def grow_social(
        self,
        site_id: str,
        platforms: Optional[List[str]] = None,
    ) -> Mission:
        """
        Execute a social growth mission for a site across specified platforms.

        Args:
            site_id: Target site ID
            platforms: List of platform names (default: pinterest, instagram, facebook)

        Returns:
            Completed Mission object
        """
        params = {
            "site_id": site_id,
            "platforms": platforms or ["pinterest", "instagram", "facebook"],
        }
        return await self.execute_mission(
            MissionType.SOCIAL_GROWTH,
            params,
            description=f"Social growth for {site_id}: {', '.join(params['platforms'])}",
        )

    def grow_social_sync(self, site_id: str, **kw) -> Mission:
        """Synchronous wrapper for grow_social()."""
        return _run_sync(self.grow_social(site_id, **kw))

    async def check_revenue(
        self,
        site_ids: Optional[List[str]] = None,
    ) -> Mission:
        """
        Execute a revenue check mission across specified (or all) sites.

        Args:
            site_ids: List of site IDs to check (default: all)

        Returns:
            Completed Mission object
        """
        if not site_ids:
            site_ids = _get_all_site_ids()

        params = {"site_ids": site_ids}
        return await self.execute_mission(
            MissionType.REVENUE_CHECK,
            params,
            description=f"Revenue check for {len(site_ids)} sites",
        )

    def check_revenue_sync(self, **kw) -> Mission:
        """Synchronous wrapper for check_revenue()."""
        return _run_sync(self.check_revenue(**kw))

    async def maintain_site(self, site_id: str) -> Mission:
        """
        Execute a site maintenance mission: health check, SEO audit,
        link audit, quality scan, backup, anomaly detection.

        Args:
            site_id: Target site ID

        Returns:
            Completed Mission object
        """
        params = {"site_id": site_id}
        return await self.execute_mission(
            MissionType.SITE_MAINTENANCE,
            params,
            description=f"Maintenance for {site_id}",
        )

    def maintain_site_sync(self, site_id: str) -> Mission:
        """Synchronous wrapper for maintain_site()."""
        return _run_sync(self.maintain_site(site_id))

    async def maintain_devices(self) -> Mission:
        """
        Execute a device maintenance mission: phone farm health,
        device pool status, GeeLark profiles, cleanup stale sessions.

        Returns:
            Completed Mission object
        """
        return await self.execute_mission(
            MissionType.DEVICE_MAINTENANCE,
            {},
            description="Device fleet maintenance",
        )

    def maintain_devices_sync(self) -> Mission:
        """Synchronous wrapper for maintain_devices()."""
        return _run_sync(self.maintain_devices())

    async def create_account(
        self,
        platform: str,
        niche: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Mission:
        """
        Execute an account creation mission: generate persona, create
        email, register on platform, track in account manager.

        Args:
            platform: Platform name (instagram, tiktok, etc.)
            niche: Niche for the persona
            details: Additional persona details

        Returns:
            Completed Mission object
        """
        params = {
            "platform": platform,
            "niche": niche,
            "details": details or {},
        }
        return await self.execute_mission(
            MissionType.ACCOUNT_CREATION,
            params,
            description=f"Create {platform} account for {niche} niche",
        )

    def create_account_sync(self, platform: str, niche: str, **kw) -> Mission:
        """Synchronous wrapper for create_account()."""
        return _run_sync(self.create_account(platform, niche, **kw))

    async def explore_app(
        self,
        app_name: str,
        goals: Optional[List[str]] = None,
    ) -> Mission:
        """
        Execute an app exploration mission: search, launch, explore UI,
        generate playbook, store learnings.

        Args:
            app_name: App name or package name
            goals: Exploration goals

        Returns:
            Completed Mission object
        """
        params = {
            "app_name": app_name,
            "goals": goals or ["learn UI", "find key features"],
        }
        return await self.execute_mission(
            MissionType.APP_EXPLORATION,
            params,
            description=f"Explore app: {app_name}",
        )

    def explore_app_sync(self, app_name: str, **kw) -> Mission:
        """Synchronous wrapper for explore_app()."""
        return _run_sync(self.explore_app(app_name, **kw))

    async def substack_daily(self, account_id: str = "witchcraft") -> Mission:
        """
        Execute a daily Substack newsletter mission: voice, calendar,
        generate content, score quality, schedule social, mark published.

        Args:
            account_id: Site/account ID for the Substack newsletter

        Returns:
            Completed Mission object
        """
        params = {"account_id": account_id}
        return await self.execute_mission(
            MissionType.SUBSTACK_DAILY,
            params,
            description=f"Substack daily for {account_id}",
        )

    def substack_daily_sync(self, account_id: str = "witchcraft") -> Mission:
        """Synchronous wrapper for substack_daily()."""
        return _run_sync(self.substack_daily(account_id))

    # -------------------------------------------------------------------
    # RAG-Enhanced Conversation
    # -------------------------------------------------------------------

    async def converse(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        RAG-enhanced conversation with the orchestrator brain.

        Uses the RAG memory module to retrieve relevant context from
        agent memory, then calls Claude Sonnet for a contextual response.
        Maintains multi-turn conversation sessions.

        Args:
            message: User message
            session_id: Existing session ID for multi-turn (creates new if None)

        Returns:
            Assistant response string
        """
        # Get or create session
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
        else:
            session = ConversationSession()
            self._sessions[session.session_id] = session
            self._stats["total_conversations"] = self._stats.get("total_conversations", 0) + 1

        # Add user message
        user_msg = ConversationMessage(role="user", content=message)

        # RAG context retrieval
        rag_context = ""
        context_sources: List[str] = []
        try:
            rag_result = await self.dispatch(
                "rag_memory", "build_context",
                query=message,
                max_tokens=2000,
                max_retries=1,
                timeout=15.0,
            )
            if rag_result["status"] == DispatchResult.SUCCESS and rag_result.get("result"):
                ctx = rag_result["result"]
                if hasattr(ctx, "context_string"):
                    rag_context = ctx.context_string
                    context_sources = [f"rag:{s}" for s in getattr(ctx, "sources", [])]
                elif isinstance(ctx, dict):
                    rag_context = ctx.get("context_string", str(ctx))
                    context_sources = [f"rag:{s}" for s in ctx.get("sources", [])]
                elif isinstance(ctx, str):
                    rag_context = ctx
        except Exception as exc:
            logger.debug("RAG context retrieval failed: %s", exc)

        user_msg.context_used = context_sources
        session.messages.append(user_msg)

        # Trim conversation history to fit context window
        if len(session.messages) > MAX_CONVERSATION_MESSAGES:
            session.messages = session.messages[-MAX_CONVERSATION_MESSAGES:]

        # Build system prompt
        system_prompt = self._build_conversation_system_prompt(rag_context)

        # Build message history for the API
        api_messages = []
        for msg in session.messages:
            if msg.role in ("user", "assistant"):
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Call Claude Sonnet
        response_text = await self._call_claude(system_prompt, api_messages)

        # Add assistant response
        assistant_msg = ConversationMessage(
            role="assistant",
            content=response_text,
            context_used=context_sources,
        )
        session.messages.append(assistant_msg)
        session.last_active = _now_iso()
        self._stats["total_messages"] = self._stats.get("total_messages", 0) + 2

        # Estimate tokens (rough: 1 token ~= 4 chars)
        session.total_tokens += len(message) // 4 + len(response_text) // 4

        self._save_conversations()
        self._save_stats()

        return response_text

    def converse_sync(self, message: str, session_id: Optional[str] = None) -> str:
        """Synchronous wrapper for converse()."""
        return _run_sync(self.converse(message, session_id))

    def _build_conversation_system_prompt(self, rag_context: str = "") -> str:
        """Build the system prompt for conversation mode."""
        base = (
            "You are the OpenClaw Empire brain — the unified orchestrator for "
            "Nick Creighton's 16-site WordPress publishing empire. You manage "
            "content publishing, social media growth, phone automation, account "
            "creation, app exploration, revenue tracking, site maintenance, "
            "device management, and Substack newsletters.\n\n"
            "You have access to all empire modules and can execute any operation. "
            "When the user asks you to do something, explain what you would do "
            "and the steps involved. Be concise, actionable, and specific.\n\n"
            "Key facts:\n"
            "- 16 WordPress sites across witchcraft, tech, AI, family, and niche topics\n"
            "- Revenue streams: ads, affiliate, KDP, Etsy POD, Substack, YouTube\n"
            "- All sites use Blocksy theme (except Family Flourish: Astra)\n"
            "- SEO: RankMath Pro | Cache: LiteSpeed | Security: Wordfence\n"
            "- Phone automation via FORGE + AMPLIFY + IntelligenceHub\n"
            "- Content is voice-matched per site via BrandVoiceEngine\n"
        )

        if rag_context:
            base += (
                "\n--- RELEVANT CONTEXT FROM MEMORY ---\n"
                f"{rag_context}\n"
                "--- END CONTEXT ---\n\n"
                "Use the above context to inform your responses when relevant.\n"
            )

        return base

    async def _call_claude(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Call the Anthropic Claude API for conversation.

        Uses Sonnet model with prompt caching for cost efficiency.
        Falls back to a template response if API is unavailable.

        Args:
            system_prompt: System prompt with context
            messages: List of {"role": ..., "content": ...} dicts

        Returns:
            Model response text
        """
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed — using fallback response")
            return self._fallback_response(messages[-1]["content"] if messages else "")

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — using fallback response")
            return self._fallback_response(messages[-1]["content"] if messages else "")

        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)

            # Use prompt caching for the system prompt if it's large enough
            system_content: Any
            if len(system_prompt) > 2048:
                system_content = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                system_content = system_prompt

            response = await client.messages.create(
                model=SONNET_MODEL,
                max_tokens=2000,
                system=system_content,
                messages=messages,
            )

            if response.content and len(response.content) > 0:
                return response.content[0].text
            return "I processed your request but generated no output."

        except Exception as exc:
            logger.error("Claude API call failed: %s", exc)
            return self._fallback_response(
                messages[-1]["content"] if messages else "",
                error=str(exc),
            )

    @staticmethod
    def _fallback_response(user_message: str, error: str = "") -> str:
        """Generate a fallback response when the API is unavailable."""
        prefix = ""
        if error:
            prefix = f"[API unavailable: {error}]\n\n"

        msg_lower = user_message.lower()

        if any(w in msg_lower for w in ("revenue", "money", "income", "earning")):
            return (
                f"{prefix}To check revenue, I would run a REVENUE_CHECK mission "
                "that queries the revenue tracker for daily/weekly summaries "
                "across all income streams (ads, affiliate, KDP, Etsy, Substack). "
                "Use: orchestrator.check_revenue()"
            )
        if any(w in msg_lower for w in ("publish", "article", "content", "write")):
            return (
                f"{prefix}To publish content, I would run a CONTENT_PUBLISH mission "
                "that generates voice-matched, SEO-optimized articles with quality "
                "scoring, internal linking, and social scheduling. "
                "Use: orchestrator.publish_content(site_id, title=...)"
            )
        if any(w in msg_lower for w in ("social", "instagram", "pinterest", "grow")):
            return (
                f"{prefix}To grow social presence, I would run a SOCIAL_GROWTH mission "
                "generating platform-specific content with voice matching and scheduling. "
                "Use: orchestrator.grow_social(site_id, platforms=[...])"
            )
        if any(w in msg_lower for w in ("phone", "device", "android")):
            return (
                f"{prefix}For phone automation, I use FORGE pre-analysis and AMPLIFY "
                "enhancement before executing via IntelligenceHub. "
                "Use: orchestrator.execute_phone_task(description)"
            )
        if any(w in msg_lower for w in ("maintain", "health", "audit", "check")):
            return (
                f"{prefix}For site maintenance, I run health checks, SEO audits, "
                "link audits, quality scans, backups, and anomaly detection. "
                "Use: orchestrator.maintain_site(site_id)"
            )

        return (
            f"{prefix}I'm the OpenClaw Empire orchestrator. I can help with:\n"
            "- Content publishing (CONTENT_PUBLISH)\n"
            "- Social media growth (SOCIAL_GROWTH)\n"
            "- Account creation (ACCOUNT_CREATION)\n"
            "- App exploration (APP_EXPLORATION)\n"
            "- Revenue tracking (REVENUE_CHECK)\n"
            "- Site maintenance (SITE_MAINTENANCE)\n"
            "- Device management (DEVICE_MAINTENANCE)\n"
            "- Phone automation (execute_phone_task)\n"
            "- Substack newsletters (SUBSTACK_DAILY)\n\n"
            "What would you like to do?"
        )

    # -------------------------------------------------------------------
    # Conversation session queries
    # -------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ConversationSession]:
        """
        List conversation sessions ordered by last activity.

        Args:
            limit: Maximum results
            offset: Skip this many

        Returns:
            List of ConversationSession objects
        """
        sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.last_active,
            reverse=True,
        )
        return sessions[offset:offset + limit]

    # -------------------------------------------------------------------
    # Dispatch log queries
    # -------------------------------------------------------------------

    def get_dispatch_log(
        self,
        module: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100,
    ) -> List[DispatchLog]:
        """
        Query the dispatch log with optional filters.

        Args:
            module: Filter by module name
            success: Filter by success status
            limit: Maximum results

        Returns:
            List of DispatchLog entries (newest first)
        """
        entries = list(reversed(self._dispatch_log))

        if module:
            entries = [e for e in entries if e.module == module]
        if success is not None:
            entries = [e for e in entries if e.success == success]

        return entries[:limit]

    def get_dispatch_stats(self) -> Dict[str, Any]:
        """Get aggregate dispatch statistics by module."""
        by_module: Dict[str, Dict[str, Any]] = {}

        for entry in self._dispatch_log:
            if entry.module not in by_module:
                by_module[entry.module] = {
                    "calls": 0, "successes": 0, "failures": 0,
                    "total_duration": 0.0, "total_retries": 0,
                }
            stats = by_module[entry.module]
            stats["calls"] += 1
            if entry.success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            stats["total_duration"] += entry.duration
            stats["total_retries"] += entry.retries

        # Calculate averages
        for module_name, stats in by_module.items():
            if stats["calls"] > 0:
                stats["avg_duration"] = round(stats["total_duration"] / stats["calls"], 3)
                stats["success_rate"] = round(stats["successes"] / stats["calls"] * 100, 1)
            else:
                stats["avg_duration"] = 0.0
                stats["success_rate"] = 0.0

        return {
            "by_module": by_module,
            "total_entries": len(self._dispatch_log),
            "circuits": {name: c.to_dict() for name, c in self._circuits.items()},
        }

    # -------------------------------------------------------------------
    # Full status summary
    # -------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        mission_stats = self.get_mission_stats()
        dispatch_stats = self.get_dispatch_stats()

        return {
            "orchestrator": "UnifiedOrchestrator",
            "version": "1.0.0",
            "uptime_started": self._stats.get("uptime_started"),
            "modules_loaded": list(self._module_cache.keys()),
            "modules_available": list(MODULE_REGISTRY.keys()),
            "missions": {
                "total": mission_stats["total_missions"],
                "by_status": mission_stats["by_status"],
                "by_type": mission_stats["by_type"],
                "step_success_rate": mission_stats["step_success_rate"],
            },
            "dispatches": {
                "total": self._stats.get("total_dispatches", 0),
                "successful": self._stats.get("successful_dispatches", 0),
                "failed": self._stats.get("failed_dispatches", 0),
                "retries_total": self._stats.get("retries_total", 0),
            },
            "conversations": {
                "total_sessions": len(self._sessions),
                "total_messages": self._stats.get("total_messages", 0),
            },
            "circuits": {
                name: c.to_dict() for name, c in self._circuits.items()
            },
            "dispatch_by_module": dispatch_stats.get("by_module", {}),
        }

    # -------------------------------------------------------------------
    # Reset utilities
    # -------------------------------------------------------------------

    def reset_circuit(self, module_name: str) -> bool:
        """Reset a module's circuit breaker to CLOSED state."""
        if module_name in self._circuits:
            circuit = self._circuits[module_name]
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.success_count = 0
            logger.info("Circuit %s reset to CLOSED", module_name)
            return True
        return False

    def reset_all_circuits(self) -> int:
        """Reset all circuit breakers. Returns count of circuits reset."""
        count = 0
        for circuit in self._circuits.values():
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.success_count = 0
            count += 1
        logger.info("Reset %d circuit breakers", count)
        return count

    def clear_module_cache(self) -> int:
        """Clear the module instance cache, forcing re-import on next use."""
        count = len(self._module_cache)
        self._module_cache.clear()
        self._stats["modules_loaded"] = 0
        logger.info("Cleared %d cached module instances", count)
        return count

    def flush(self) -> None:
        """Flush all in-memory state to disk."""
        self._persist_all()
        logger.info("All state flushed to disk")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_orchestrator_instance: Optional[UnifiedOrchestrator] = None


def get_orchestrator() -> UnifiedOrchestrator:
    """
    Get the global UnifiedOrchestrator singleton.

    Creates the instance on first call. Thread-safe via GIL.

    Returns:
        The global UnifiedOrchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = UnifiedOrchestrator()
    return _orchestrator_instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_mission_summary(mission: Mission) -> str:
    """Format a mission as a human-readable summary string."""
    lines = [
        f"Mission: {mission.mission_id}",
        f"  Type:        {mission.mission_type.value}",
        f"  Status:      {mission.status.value}",
        f"  Description: {mission.description[:100]}",
        f"  Steps:       {len(mission.steps)}",
    ]
    if mission.total_duration > 0:
        lines.append(f"  Duration:    {mission.total_duration:.2f}s")
    if mission.result and isinstance(mission.result, dict):
        lines.append(
            f"  Results:     {mission.result.get('completed', 0)} completed, "
            f"{mission.result.get('failed', 0)} failed, "
            f"{mission.result.get('skipped', 0)} skipped"
        )
    for step in mission.steps:
        status_icon = {
            StepStatus.COMPLETED: "[OK]",
            StepStatus.FAILED: "[FAIL]",
            StepStatus.SKIPPED: "[SKIP]",
            StepStatus.RUNNING: "[RUN]",
            StepStatus.PENDING: "[...]",
        }.get(step.status, "[???]")
        lines.append(
            f"    {status_icon} {step.module}.{step.method}"
            f" ({step.duration_seconds:.2f}s)"
            + (f" — {step.error[:60]}" if step.error else "")
        )
    return "\n".join(lines)


def _format_stats(stats: Dict[str, Any]) -> str:
    """Format statistics as a human-readable summary."""
    lines = [
        "=== UnifiedOrchestrator Statistics ===",
        "",
        f"Modules loaded: {len(stats.get('modules_loaded', []))} / {len(stats.get('modules_available', []))}",
        f"Uptime since:   {stats.get('uptime_started', 'unknown')[:19]}",
        "",
        "--- Missions ---",
        f"  Total:    {stats['missions']['total']}",
    ]
    for status_name, count in sorted(stats["missions"].get("by_status", {}).items()):
        lines.append(f"    {status_name}: {count}")
    lines.append(f"  Step success rate: {stats['missions'].get('step_success_rate', 0)}%")

    lines.extend([
        "",
        "--- Dispatches ---",
        f"  Total:      {stats['dispatches']['total']}",
        f"  Successful: {stats['dispatches']['successful']}",
        f"  Failed:     {stats['dispatches']['failed']}",
        f"  Retries:    {stats['dispatches']['retries_total']}",
    ])

    lines.extend([
        "",
        "--- Conversations ---",
        f"  Sessions: {stats['conversations']['total_sessions']}",
        f"  Messages: {stats['conversations']['total_messages']}",
    ])

    if stats.get("circuits"):
        lines.append("")
        lines.append("--- Circuit Breakers ---")
        for name, circuit_data in sorted(stats["circuits"].items()):
            lines.append(
                f"  {name}: {circuit_data.get('state', '?')} "
                f"(calls={circuit_data.get('total_calls', 0)}, "
                f"failures={circuit_data.get('total_failures', 0)})"
            )

    if stats.get("dispatch_by_module"):
        lines.append("")
        lines.append("--- Dispatch by Module ---")
        for mod_name, mod_stats in sorted(stats["dispatch_by_module"].items()):
            lines.append(
                f"  {mod_name}: {mod_stats['calls']} calls, "
                f"{mod_stats.get('success_rate', 0)}% success, "
                f"avg {mod_stats.get('avg_duration', 0):.2f}s"
            )

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for the Unified Orchestrator."""
    parser = argparse.ArgumentParser(
        prog="unified_orchestrator",
        description="Unified Orchestrator — Brain Merger for OpenClaw Empire",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # --- mission ---
    p_mission = sub.add_parser("mission", help="Execute a mission by type")
    p_mission.add_argument("--type", required=True, help="Mission type (e.g., CONTENT_PUBLISH)")
    p_mission.add_argument("--site", default="", help="Site ID")
    p_mission.add_argument("--title", default="", help="Content title")
    p_mission.add_argument("--topic", default="", help="Content topic")
    p_mission.add_argument("--keywords", default="", help="Comma-separated keywords")
    p_mission.add_argument("--platform", default="", help="Platform name")
    p_mission.add_argument("--niche", default="", help="Niche")
    p_mission.add_argument("--app", default="", help="App name")
    p_mission.add_argument("--account-id", default="witchcraft", help="Account/site ID")
    p_mission.add_argument("--plan-only", action="store_true", help="Plan without executing")

    # --- dispatch ---
    p_dispatch = sub.add_parser("dispatch", help="Dispatch a method call to a module")
    p_dispatch.add_argument("--module", required=True, help="Module name")
    p_dispatch.add_argument("--method", required=True, help="Method name")
    p_dispatch.add_argument("--kwargs", default="{}", help="JSON kwargs")
    p_dispatch.add_argument("--retries", type=int, default=3, help="Max retries")
    p_dispatch.add_argument("--timeout", type=float, default=120.0, help="Timeout seconds")

    # --- publish ---
    p_publish = sub.add_parser("publish", help="Publish content to a site")
    p_publish.add_argument("--site", required=True, help="Site ID")
    p_publish.add_argument("--title", default="", help="Article title")
    p_publish.add_argument("--topic", default="", help="Topic")
    p_publish.add_argument("--keywords", default="", help="Comma-separated keywords")
    p_publish.add_argument("--post-id", type=int, default=None, help="Existing post ID")

    # --- phone ---
    p_phone = sub.add_parser("phone", help="Execute a phone automation task")
    p_phone.add_argument("--task", required=True, help="Task description")
    p_phone.add_argument("--niche", default="", help="Niche context")
    p_phone.add_argument("--app", default="", help="Target app")

    # --- social ---
    p_social = sub.add_parser("social", help="Run social growth for a site")
    p_social.add_argument("--site", required=True, help="Site ID")
    p_social.add_argument("--platforms", default="", help="Comma-separated platforms")

    # --- revenue ---
    p_revenue = sub.add_parser("revenue", help="Check revenue across sites")
    p_revenue.add_argument("--sites", default="", help="Comma-separated site IDs (all if empty)")

    # --- maintain ---
    p_maintain = sub.add_parser("maintain", help="Run site maintenance")
    p_maintain.add_argument("--site", required=True, help="Site ID")

    # --- devices ---
    sub.add_parser("devices", help="Run device fleet maintenance")

    # --- converse ---
    p_converse = sub.add_parser("converse", help="Chat with the orchestrator")
    p_converse.add_argument("--message", required=True, help="Message to send")
    p_converse.add_argument("--session", default=None, help="Session ID for multi-turn")

    # --- missions ---
    p_missions = sub.add_parser("missions", help="List missions")
    p_missions.add_argument("--status", default="", help="Filter by status")
    p_missions.add_argument("--type", default="", dest="mission_type", help="Filter by type")
    p_missions.add_argument("--limit", type=int, default=20, help="Max results")

    # --- stats ---
    sub.add_parser("stats", help="Show orchestrator statistics")

    # --- status ---
    sub.add_parser("status", help="Show full orchestrator status")

    # --- reset-circuit ---
    p_reset = sub.add_parser("reset-circuit", help="Reset a circuit breaker")
    p_reset.add_argument("--module", default="", help="Module name (all if empty)")

    # --- flush ---
    sub.add_parser("flush", help="Flush all state to disk")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    orch = get_orchestrator()

    # ── mission ──
    if args.command == "mission":
        try:
            mt = MissionType(args.type.lower().replace("-", "_"))
        except ValueError:
            print(f"Invalid mission type '{args.type}'. Available:")
            for t in MissionType:
                print(f"  {t.value}")
            sys.exit(1)

        params: Dict[str, Any] = {}
        if args.site:
            params["site_id"] = args.site
        if args.title:
            params["title"] = args.title
        if args.topic:
            params["topic"] = args.topic
        if args.keywords:
            params["keywords"] = [k.strip() for k in args.keywords.split(",")]
        if args.platform:
            params["platform"] = args.platform
        if args.niche:
            params["niche"] = args.niche
        if args.app:
            params["app_name"] = args.app
        if mt == MissionType.SUBSTACK_DAILY:
            params["account_id"] = args.account_id

        if args.plan_only:
            mission = _run_sync(orch.plan_mission(mt, params))
        else:
            mission = _run_sync(orch.execute_mission(mt, params))

        print(_format_mission_summary(mission))

    # ── dispatch ──
    elif args.command == "dispatch":
        kwargs = json.loads(args.kwargs)
        result = _run_sync(
            orch.dispatch(
                args.module, args.method,
                max_retries=args.retries,
                timeout=args.timeout,
                **kwargs,
            )
        )
        print(json.dumps(result, indent=2, default=str))

    # ── publish ──
    elif args.command == "publish":
        kw: Dict[str, Any] = {}
        if args.title:
            kw["title"] = args.title
        if args.topic:
            kw["topic"] = args.topic
        if args.keywords:
            kw["keywords"] = [k.strip() for k in args.keywords.split(",")]
        if args.post_id:
            kw["post_id"] = args.post_id
        mission = _run_sync(orch.publish_content(args.site, **kw))
        print(_format_mission_summary(mission))

    # ── phone ──
    elif args.command == "phone":
        kw = {}
        if args.niche:
            kw["niche"] = args.niche
        if args.app:
            kw["app"] = args.app
        result = _run_sync(orch.execute_phone_task(args.task, **kw))
        print(json.dumps(result, indent=2, default=str))

    # ── social ──
    elif args.command == "social":
        kw = {}
        if args.platforms:
            kw["platforms"] = [p.strip() for p in args.platforms.split(",")]
        mission = _run_sync(orch.grow_social(args.site, **kw))
        print(_format_mission_summary(mission))

    # ── revenue ──
    elif args.command == "revenue":
        kw = {}
        if args.sites:
            kw["site_ids"] = [s.strip() for s in args.sites.split(",")]
        mission = _run_sync(orch.check_revenue(**kw))
        print(_format_mission_summary(mission))

    # ── maintain ──
    elif args.command == "maintain":
        mission = _run_sync(orch.maintain_site(args.site))
        print(_format_mission_summary(mission))

    # ── devices ──
    elif args.command == "devices":
        mission = _run_sync(orch.maintain_devices())
        print(_format_mission_summary(mission))

    # ── converse ──
    elif args.command == "converse":
        reply = _run_sync(orch.converse(args.message, session_id=args.session))
        print(reply)

    # ── missions ──
    elif args.command == "missions":
        status_filter = None
        type_filter = None
        if args.status:
            try:
                status_filter = MissionStatus(args.status.lower())
            except ValueError:
                print(f"Invalid status '{args.status}'. Available:")
                for s in MissionStatus:
                    print(f"  {s.value}")
                sys.exit(1)
        if args.mission_type:
            try:
                type_filter = MissionType(args.mission_type.lower().replace("-", "_"))
            except ValueError:
                print(f"Invalid type '{args.mission_type}'. Available:")
                for t in MissionType:
                    print(f"  {t.value}")
                sys.exit(1)

        missions = orch.list_missions(
            status=status_filter,
            mission_type=type_filter,
            limit=args.limit,
        )
        if not missions:
            print("No missions found.")
        else:
            for m in missions:
                print(_format_mission_summary(m))
                print()

    # ── stats ──
    elif args.command == "stats":
        status_data = orch.status()
        print(_format_stats(status_data))

    # ── status ──
    elif args.command == "status":
        print(json.dumps(orch.status(), indent=2, default=str))

    # ── reset-circuit ──
    elif args.command == "reset-circuit":
        if args.module:
            ok = orch.reset_circuit(args.module)
            if ok:
                print(f"Circuit '{args.module}' reset to CLOSED")
            else:
                print(f"No circuit found for '{args.module}'")
        else:
            count = orch.reset_all_circuits()
            print(f"Reset {count} circuit breakers")

    # ── flush ──
    elif args.command == "flush":
        orch.flush()
        print("All state flushed to disk")


if __name__ == "__main__":
    main()
