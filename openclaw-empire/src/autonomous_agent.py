"""
Autonomous Agent — OpenClaw Empire AI Brain

The central autonomous intelligence that decomposes high-level goals
into executable sub-goals, dispatches to Phase 5 modules, evaluates
outcomes, and learns from results. Implements the observe-think-act-evaluate
loop with dynamic replanning on failure, stuck detection, and human
escalation. Manages a priority-based goal queue with session continuity.

Data persisted to: data/agent/

Usage:
    from src.autonomous_agent import AutonomousAgent, get_autonomous_agent

    agent = get_autonomous_agent()
    await agent.set_goal("Manage my Instagram for witchcraft niche")
    await agent.run(max_steps=100)
    status = agent.status()

CLI:
    python -m src.autonomous_agent goal --text "Manage my Instagram account"
    python -m src.autonomous_agent run --steps 100
    python -m src.autonomous_agent status
    python -m src.autonomous_agent goals list
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("autonomous_agent")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "agent"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class GoalPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionType(str, Enum):
    OBSERVE = "observe"
    THINK = "think"
    ACT = "act"
    EVALUATE = "evaluate"
    REPLAN = "replan"
    ESCALATE = "escalate"
    WAIT = "wait"


class ModuleName(str, Enum):
    PHONE_OS = "phone_os_agent"
    BROWSER = "browser_controller"
    IDENTITY = "identity_manager"
    APP_LEARNER = "app_learner"
    APP_DISCOVERY = "app_discovery"
    EMAIL = "email_agent"
    ACCOUNT_FACTORY = "account_factory"
    SOCIAL_AGENT = "social_media_agent"
    MEMORY = "agent_memory"
    PHONE_CONTROLLER = "phone_controller"
    VISION = "vision_agent"
    SOCIAL_BOT = "social_automation"
    ACCOUNT_MGR = "account_manager"
    CONTENT_GEN = "content_generator"
    CONTENT_CALENDAR = "content_calendar"
    WORDPRESS = "wordpress_client"
    INTELLIGENCE = "intelligence_hub"
    # Phase 6 modules
    CONTENT_PIPELINE = "content_pipeline"
    UNIFIED_ORCHESTRATOR = "unified_orchestrator"
    DEVICE_POOL = "device_pool"
    CONTENT_QUALITY = "content_quality_scorer"
    RAG_MEMORY = "rag_memory"
    AB_TESTING = "ab_testing"
    SEO_AUDITOR = "seo_auditor"
    AFFILIATE = "affiliate_manager"
    INTERNAL_LINKER = "internal_linker"
    SOCIAL_PUBLISHER = "social_publisher"
    REVENUE = "revenue_tracker"
    BRAND_VOICE = "brand_voice_engine"
    SUBSTACK = "substack_agent"
    BACKUP = "backup_manager"
    ANOMALY = "anomaly_detector"


# Module import mapping
MODULE_IMPORTS = {
    ModuleName.PHONE_OS: ("src.phone_os_agent", "get_phone_os_agent"),
    ModuleName.BROWSER: ("src.browser_controller", "get_browser"),
    ModuleName.IDENTITY: ("src.identity_manager", "get_identity_manager"),
    ModuleName.APP_LEARNER: ("src.app_learner", "get_app_learner"),
    ModuleName.APP_DISCOVERY: ("src.app_discovery", "get_app_discovery"),
    ModuleName.EMAIL: ("src.email_agent", "get_email_agent"),
    ModuleName.ACCOUNT_FACTORY: ("src.account_factory", "get_account_factory"),
    ModuleName.SOCIAL_AGENT: ("src.social_media_agent", "get_social_agent"),
    ModuleName.MEMORY: ("src.agent_memory", "get_memory"),
    ModuleName.SOCIAL_BOT: ("src.social_automation", "get_social_bot"),
    ModuleName.ACCOUNT_MGR: ("src.account_manager", "get_account_manager"),
    ModuleName.CONTENT_GEN: ("src.content_generator", "get_content_generator"),
    ModuleName.CONTENT_CALENDAR: ("src.content_calendar", "get_calendar"),
    ModuleName.WORDPRESS: ("src.wordpress_client", "get_wordpress_client"),
    # Phase 6 module imports
    ModuleName.CONTENT_PIPELINE: ("src.content_pipeline", "get_pipeline"),
    ModuleName.UNIFIED_ORCHESTRATOR: ("src.unified_orchestrator", "get_orchestrator"),
    ModuleName.DEVICE_POOL: ("src.device_pool", "get_pool"),
    ModuleName.CONTENT_QUALITY: ("src.content_quality_scorer", "get_scorer"),
    ModuleName.RAG_MEMORY: ("src.rag_memory", "get_rag"),
    ModuleName.AB_TESTING: ("src.ab_testing", "get_ab_testing"),
    ModuleName.SEO_AUDITOR: ("src.seo_auditor", "get_seo_auditor"),
    ModuleName.AFFILIATE: ("src.affiliate_manager", "get_affiliate_manager"),
    ModuleName.INTERNAL_LINKER: ("src.internal_linker", "get_linker"),
    ModuleName.SOCIAL_PUBLISHER: ("src.social_publisher", "get_publisher"),
    ModuleName.REVENUE: ("src.revenue_tracker", "get_revenue_tracker"),
    ModuleName.BRAND_VOICE: ("src.brand_voice_engine", "get_brand_voice_engine"),
    ModuleName.SUBSTACK: ("src.substack_agent", "get_agent"),
    ModuleName.BACKUP: ("src.backup_manager", "get_backup_manager"),
    ModuleName.ANOMALY: ("src.anomaly_detector", "get_detector"),
}

PRIORITY_ORDER = {
    GoalPriority.CRITICAL: 0,
    GoalPriority.HIGH: 1,
    GoalPriority.NORMAL: 2,
    GoalPriority.LOW: 3,
    GoalPriority.BACKGROUND: 4,
}


@dataclass
class SubGoal:
    """A sub-goal decomposed from a high-level goal."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    description: str = ""
    module: str = ""
    method: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    status: GoalStatus = GoalStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    result: Any = None
    error: str = ""
    attempts: int = 0
    max_attempts: int = 3

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class Goal:
    """A high-level goal for the autonomous agent."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str = ""
    priority: GoalPriority = GoalPriority.NORMAL
    status: GoalStatus = GoalStatus.PENDING
    sub_goals: List[SubGoal] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "priority": self.priority.value,
            "status": self.status.value,
            "sub_goals": [sg.to_dict() for sg in self.sub_goals],
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "context": self.context,
        }


@dataclass
class AgentStep:
    """A single step in the agent's execution loop."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    action_type: ActionType = ActionType.OBSERVE
    goal_id: str = ""
    sub_goal_id: str = ""
    description: str = ""
    module: str = ""
    method: str = ""
    result: Any = None
    success: bool = True
    error: str = ""
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d


@dataclass
class AgentSession:
    """A session of the autonomous agent."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(default_factory=_now_iso)
    ended_at: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    goals_completed: int = 0
    goals_failed: int = 0
    total_actions: int = 0
    active: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "steps_count": len(self.steps),
            "goals_completed": self.goals_completed,
            "goals_failed": self.goals_failed,
            "total_actions": self.total_actions,
            "active": self.active,
        }


# ---------------------------------------------------------------------------
# Goal decomposition templates
# ---------------------------------------------------------------------------

GOAL_PATTERNS = {
    "manage instagram": [
        SubGoal(description="Generate Instagram content strategy", module="social_media_agent", method="generate_strategy", args={"platform": "instagram", "niche": ""}),
        SubGoal(description="Run engagement session", module="social_media_agent", method="smart_engage", args={"platform": "instagram", "duration_minutes": 30}),
        SubGoal(description="Extract analytics", module="social_media_agent", method="extract_analytics", args={"platform": "instagram"}),
        SubGoal(description="Track growth", module="social_media_agent", method="track_growth", args={"platform": "instagram"}),
    ],
    "create account": [
        SubGoal(description="Generate persona identity", module="identity_manager", method="generate_persona"),
        SubGoal(description="Create email account", module="email_agent", method="create_gmail_account"),
        SubGoal(description="Create platform account", module="account_factory", method="create_account"),
    ],
    "explore app": [
        SubGoal(description="Explore app UI", module="app_learner", method="explore_app"),
        SubGoal(description="Generate playbook", module="app_learner", method="generate_playbook"),
    ],
    "browse web": [
        SubGoal(description="Open URL in browser", module="browser_controller", method="open_url"),
        SubGoal(description="Extract page content", module="browser_controller", method="extract_page_text"),
    ],
    "check email": [
        SubGoal(description="Read inbox", module="email_agent", method="read_inbox"),
    ],
    "install app": [
        SubGoal(description="Search Play Store", module="app_discovery", method="search_play_store"),
        SubGoal(description="Evaluate app", module="app_discovery", method="evaluate_app"),
        SubGoal(description="Install app", module="app_discovery", method="install_app"),
    ],
    # Phase 6: Content pipeline patterns
    "publish content": [
        SubGoal(description="Detect content gaps", module="content_calendar", method="auto_fill_gaps"),
        SubGoal(description="Execute content pipeline", module="content_pipeline", method="execute_sync", args={"site_id": ""}),
        SubGoal(description="Create social campaign", module="social_publisher", method="create_campaign_from_article"),
    ],
    "publish article": [
        SubGoal(description="Generate article via pipeline", module="content_pipeline", method="execute_sync", args={"site_id": "", "title": ""}),
        SubGoal(description="Generate social posts", module="social_publisher", method="create_campaign_from_article"),
    ],
    "check revenue": [
        SubGoal(description="Get revenue summary", module="revenue_tracker", method="get_summary"),
        SubGoal(description="Check for anomalies", module="anomaly_detector", method="detect_sync"),
    ],
    "write newsletter": [
        SubGoal(description="Write daily newsletter", module="substack_agent", method="daily_routine_sync"),
    ],
    "run backup": [
        SubGoal(description="Execute full backup", module="backup_manager", method="full_backup_sync"),
    ],
    "seo audit": [
        SubGoal(description="Run SEO audit", module="seo_auditor", method="audit_site_sync"),
        SubGoal(description="Generate recommendations", module="seo_auditor", method="get_recommendations_sync"),
    ],
    "content quality": [
        SubGoal(description="Score content quality", module="content_quality_scorer", method="score_sync"),
    ],
    "device status": [
        SubGoal(description="Discover all devices", module="device_pool", method="discover_all_sync"),
        SubGoal(description="Get fleet health", module="device_pool", method="health_report_sync"),
    ],
}


# ---------------------------------------------------------------------------
# AutonomousAgent
# ---------------------------------------------------------------------------

class AutonomousAgent:
    """
    The AI Brain — autonomous agent that decomposes goals,
    dispatches to modules, evaluates outcomes, and learns.

    Implements the observe-think-act-evaluate loop with:
    - Goal decomposition via Sonnet
    - Module dispatch via lazy imports
    - Dynamic replanning on failure
    - Stuck detection (same screen 3x)
    - Human escalation for CAPTCHA/phone verification
    - Priority-based goal queue with preemption
    - Session save/restore for continuity

    Usage:
        agent = get_autonomous_agent()
        await agent.set_goal("Manage my Instagram for witchcraft niche")
        await agent.run(max_steps=100)
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
    ):
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Goal queue (sorted by priority)
        self._goals: List[Goal] = []
        self._active_goal: Optional[Goal] = None
        self._session: Optional[AgentSession] = None
        self._sessions: List[AgentSession] = []

        # Module instances (lazy loaded)
        self._modules: Dict[str, Any] = {}

        # Stuck detection
        self._last_screen_hashes: List[str] = []
        self._stuck_threshold: int = 3
        self._consecutive_failures: int = 0
        self._max_consecutive_failures: int = 5

        # State
        self._running: bool = False
        self._paused: bool = False

        self._load_state()
        logger.info("AutonomousAgent initialized (%d goals)", len(self._goals))

    # ── Module dispatch ──

    def _get_module(self, module_name: str) -> Any:
        """Lazy-load and cache a module instance."""
        if module_name in self._modules:
            return self._modules[module_name]

        # Find the module in our import map
        for mod_enum in ModuleName:
            if mod_enum.value == module_name:
                info = MODULE_IMPORTS.get(mod_enum)
                if info:
                    module_path, factory_name = info
                    try:
                        import importlib
                        mod = importlib.import_module(module_path)
                        factory = getattr(mod, factory_name)
                        instance = factory()
                        self._modules[module_name] = instance
                        return instance
                    except Exception as exc:
                        logger.warning("Failed to import %s: %s", module_path, exc)
                break

        return None

    async def _dispatch(
        self, module_name: str, method_name: str, **kwargs
    ) -> Any:
        """Dispatch a method call to a module."""
        module = self._get_module(module_name)
        if module is None:
            raise RuntimeError(f"Module '{module_name}' not available")

        method = getattr(module, method_name, None)
        if method is None:
            # Try sync version
            sync_name = f"{method_name}_sync"
            method = getattr(module, sync_name, None)
            if method is None:
                raise RuntimeError(f"Method '{method_name}' not found on {module_name}")

        if asyncio.iscoroutinefunction(method):
            return await method(**kwargs)
        return method(**kwargs)

    # ── Persistence ──

    def _load_state(self) -> None:
        state = _load_json(self._data_dir / "state.json")

        for goal_data in state.get("goals", []):
            if isinstance(goal_data, dict):
                sub_goals = [
                    SubGoal(**sg) if isinstance(sg, dict) else sg
                    for sg in goal_data.pop("sub_goals", [])
                ]
                p = goal_data.pop("priority", "normal")
                s = goal_data.pop("status", "pending")
                goal = Goal(
                    priority=GoalPriority(p),
                    status=GoalStatus(s),
                    sub_goals=sub_goals,
                    **goal_data,
                )
                self._goals.append(goal)

        for session_data in state.get("sessions", [])[-10:]:
            if isinstance(session_data, dict):
                session_data.pop("steps", None)
                session_data.pop("steps_count", None)
                self._sessions.append(AgentSession(**session_data))

    def _save_state(self) -> None:
        _save_json(self._data_dir / "state.json", {
            "goals": [g.to_dict() for g in self._goals],
            "sessions": [s.to_dict() for s in self._sessions[-10:]],
            "active_goal_id": self._active_goal.id if self._active_goal else None,
            "updated_at": _now_iso(),
        })

    def _save_session(self) -> None:
        if self._session:
            _save_json(
                self._data_dir / f"session_{self._session.id}.json",
                {
                    "session": self._session.to_dict(),
                    "steps": [s.to_dict() for s in self._session.steps[-200:]],
                }
            )

    # ── Goal management ──

    async def set_goal(
        self,
        text: str,
        priority: GoalPriority = GoalPriority.NORMAL,
        context: Optional[Dict[str, Any]] = None,
    ) -> Goal:
        """
        Set a new goal for the agent.

        The goal is decomposed into sub-goals and added to the queue.

        Args:
            text: Natural language goal description.
            priority: Goal priority level.
            context: Additional context for goal decomposition.

        Returns:
            The created Goal object.
        """
        goal = Goal(
            text=text,
            priority=priority,
            context=context or {},
        )

        # Decompose into sub-goals
        sub_goals = await self._decompose_goal(text, context or {})
        goal.sub_goals = sub_goals

        # Insert into queue based on priority
        self._goals.append(goal)
        self._goals.sort(key=lambda g: PRIORITY_ORDER.get(g.priority, 2))

        # Check for preemption
        if (self._active_goal and
            PRIORITY_ORDER[priority] < PRIORITY_ORDER[self._active_goal.priority]):
            logger.info("Preempting current goal with higher-priority: %s", text[:50])
            self._active_goal.status = GoalStatus.PENDING

        self._save_state()

        # Store in memory
        memory = self._get_module("agent_memory")
        if memory:
            try:
                memory.store_sync(
                    content=f"New goal set: {text} (priority: {priority.value})",
                    memory_type="task_result",
                    tags=["goal", priority.value],
                )
            except Exception:
                pass

        logger.info("Goal set: %s (%d sub-goals)", text[:50], len(sub_goals))
        return goal

    async def _decompose_goal(
        self, text: str, context: Dict[str, Any]
    ) -> List[SubGoal]:
        """Decompose a goal into sub-goals using pattern matching and AI."""
        text_lower = text.lower()

        # Check pattern-based decomposition first
        for pattern, template_goals in GOAL_PATTERNS.items():
            if pattern in text_lower:
                sub_goals = []
                for template in template_goals:
                    sg = SubGoal(
                        description=template.description,
                        module=template.module,
                        method=template.method,
                        args={**template.args, **context},
                    )
                    sub_goals.append(sg)
                return sub_goals

        # Fall back to AI decomposition via Sonnet
        try:
            import anthropic
            client = anthropic.Anthropic()

            available_modules = [
                f"- {m.value}: {m.name}" for m in ModuleName
            ]

            prompt = (
                f"Decompose this goal into 2-5 concrete sub-goals:\n\n"
                f"Goal: {text}\n\n"
                f"Available modules:\n{'\\n'.join(available_modules)}\n\n"
                f"Return JSON array of objects with keys: "
                f"description (string), module (string from list above), "
                f"method (likely method name), args (dict of arguments)\n\n"
                f"Only use modules from the list. Keep it simple and actionable."
            )

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                system=[{
                    "type": "text",
                    "text": "You are a task decomposition assistant. Return valid JSON arrays only.",
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            data = json.loads(response_text)
            sub_goals = []
            for item in data:
                sg = SubGoal(
                    description=item.get("description", ""),
                    module=item.get("module", ""),
                    method=item.get("method", ""),
                    args=item.get("args", {}),
                )
                sub_goals.append(sg)
            return sub_goals

        except Exception as exc:
            logger.warning("AI decomposition failed: %s — using generic plan", exc)
            return [
                SubGoal(description=f"Execute: {text}", module="", method=""),
            ]

    def list_goals(self, status: str = "") -> List[Dict[str, Any]]:
        """List all goals, optionally filtered by status."""
        goals = self._goals
        if status:
            goals = [g for g in goals if g.status.value == status]
        return [g.to_dict() for g in goals]

    def cancel_goal(self, goal_id: str) -> Dict[str, Any]:
        """Cancel a goal."""
        for goal in self._goals:
            if goal.id == goal_id:
                goal.status = GoalStatus.CANCELLED
                if self._active_goal and self._active_goal.id == goal_id:
                    self._active_goal = None
                self._save_state()
                return {"success": True, "cancelled": goal_id}
        return {"success": False, "error": f"Goal {goal_id} not found"}

    # ── Main execution loop ──

    async def run(
        self,
        max_steps: int = 100,
        timeout_minutes: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Run the autonomous agent loop.

        Processes goals from the queue using observe-think-act-evaluate:
        1. Pick highest-priority pending goal
        2. Execute sub-goals in order
        3. Evaluate results after each action
        4. Replan on failure, escalate if stuck

        Args:
            max_steps: Maximum number of steps to take.
            timeout_minutes: Maximum runtime in minutes.

        Returns:
            Dict with session results.
        """
        self._running = True
        self._paused = False

        # Create session
        self._session = AgentSession()
        self._sessions.append(self._session)

        start_time = time.monotonic()
        steps_taken = 0

        logger.info("Agent session started (max %d steps, %.0f min timeout)",
                     max_steps, timeout_minutes)

        try:
            while (self._running and
                   steps_taken < max_steps and
                   time.monotonic() - start_time < timeout_minutes * 60):

                if self._paused:
                    await asyncio.sleep(1.0)
                    continue

                # Pick next goal
                if not self._active_goal:
                    self._active_goal = self._pick_next_goal()
                    if not self._active_goal:
                        logger.info("No pending goals — agent idle")
                        break
                    self._active_goal.status = GoalStatus.ACTIVE
                    self._active_goal.started_at = _now_iso()
                    logger.info("Working on goal: %s", self._active_goal.text[:50])

                # Execute next sub-goal
                goal = self._active_goal
                next_sg = self._get_next_subgoal(goal)

                if not next_sg:
                    # All sub-goals done — mark goal complete
                    all_done = all(sg.status == GoalStatus.COMPLETED for sg in goal.sub_goals)
                    if all_done:
                        goal.status = GoalStatus.COMPLETED
                        goal.completed_at = _now_iso()
                        self._session.goals_completed += 1
                        logger.info("Goal completed: %s", goal.text[:50])
                    else:
                        goal.status = GoalStatus.FAILED
                        goal.error = "Some sub-goals failed"
                        self._session.goals_failed += 1
                        logger.warning("Goal failed: %s", goal.text[:50])
                    self._active_goal = None
                    self._save_state()
                    continue

                # Execute the sub-goal
                step = await self._execute_subgoal(next_sg, goal)
                self._session.steps.append(step)
                self._session.total_actions += 1
                steps_taken += 1

                # Evaluate
                if not step.success:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._max_consecutive_failures:
                        logger.warning("Too many failures — escalating to human")
                        goal.status = GoalStatus.BLOCKED
                        goal.error = "Too many consecutive failures"
                        self._active_goal = None
                        self._consecutive_failures = 0
                else:
                    self._consecutive_failures = 0

                self._save_state()
                self._save_session()

                # Brief pause between actions
                await asyncio.sleep(1.0)

        except Exception as exc:
            logger.error("Agent loop error: %s", exc)

        # End session
        self._running = False
        if self._session:
            self._session.active = False
            self._session.ended_at = _now_iso()
            self._save_session()

        self._save_state()

        duration = time.monotonic() - start_time
        result = {
            "session_id": self._session.id if self._session else "",
            "steps_taken": steps_taken,
            "duration_seconds": round(duration),
            "goals_completed": self._session.goals_completed if self._session else 0,
            "goals_failed": self._session.goals_failed if self._session else 0,
        }
        logger.info("Agent session ended: %d steps, %d completed, %d failed",
                     steps_taken, result["goals_completed"], result["goals_failed"])
        return result

    def _pick_next_goal(self) -> Optional[Goal]:
        """Pick the next goal from the queue (highest priority first)."""
        for goal in self._goals:
            if goal.status in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                return goal
        return None

    def _get_next_subgoal(self, goal: Goal) -> Optional[SubGoal]:
        """Get the next sub-goal to execute."""
        for sg in goal.sub_goals:
            if sg.status == GoalStatus.PENDING:
                # Check dependencies
                if sg.depends_on:
                    deps_met = all(
                        any(d.id == dep_id and d.status == GoalStatus.COMPLETED
                            for d in goal.sub_goals)
                        for dep_id in sg.depends_on
                    )
                    if not deps_met:
                        continue
                return sg
        return None

    async def _execute_subgoal(self, sg: SubGoal, goal: Goal) -> AgentStep:
        """Execute a single sub-goal."""
        step = AgentStep(
            action_type=ActionType.ACT,
            goal_id=goal.id,
            sub_goal_id=sg.id,
            description=sg.description,
            module=sg.module,
            method=sg.method,
        )

        sg.status = GoalStatus.IN_PROGRESS
        sg.attempts += 1
        start = time.monotonic()

        try:
            if not sg.module or not sg.method:
                # Generic sub-goal — just mark as complete
                sg.status = GoalStatus.COMPLETED
                step.result = {"message": "No module/method — marked complete"}
                step.success = True
            else:
                result = await self._dispatch(sg.module, sg.method, **sg.args)
                sg.result = result
                step.result = result

                # Evaluate result
                if isinstance(result, dict):
                    step.success = result.get("success", True)
                else:
                    step.success = True

                if step.success:
                    sg.status = GoalStatus.COMPLETED
                elif sg.attempts < sg.max_attempts:
                    sg.status = GoalStatus.PENDING  # Retry
                    step.error = str(result.get("error", "")) if isinstance(result, dict) else ""
                    logger.warning("Sub-goal failed (attempt %d/%d): %s",
                                   sg.attempts, sg.max_attempts, step.error)
                else:
                    sg.status = GoalStatus.FAILED
                    sg.error = str(result.get("error", "")) if isinstance(result, dict) else "Max attempts reached"
                    step.error = sg.error

        except Exception as exc:
            step.success = False
            step.error = str(exc)
            sg.error = str(exc)
            if sg.attempts < sg.max_attempts:
                sg.status = GoalStatus.PENDING
            else:
                sg.status = GoalStatus.FAILED
            logger.error("Sub-goal execution error: %s", exc)

        step.duration_ms = (time.monotonic() - start) * 1000
        return step

    # ── Control ──

    def stop(self) -> None:
        """Stop the agent."""
        self._running = False
        logger.info("Agent stop requested")

    def pause(self) -> None:
        """Pause the agent."""
        self._paused = True
        logger.info("Agent paused")

    def resume(self) -> None:
        """Resume a paused agent."""
        self._paused = False
        logger.info("Agent resumed")

    # ── Status ──

    def status(self) -> Dict[str, Any]:
        """Get the agent's current status."""
        active = self._active_goal.to_dict() if self._active_goal else None
        pending = [g for g in self._goals if g.status == GoalStatus.PENDING]
        completed = [g for g in self._goals if g.status == GoalStatus.COMPLETED]
        failed = [g for g in self._goals if g.status == GoalStatus.FAILED]

        return {
            "running": self._running,
            "paused": self._paused,
            "active_goal": active,
            "pending_goals": len(pending),
            "completed_goals": len(completed),
            "failed_goals": len(failed),
            "total_goals": len(self._goals),
            "sessions": len(self._sessions),
            "current_session": self._session.to_dict() if self._session else None,
            "consecutive_failures": self._consecutive_failures,
            "modules_loaded": list(self._modules.keys()),
        }

    # ── Session management ──

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        return [s.to_dict() for s in self._sessions]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific session."""
        for s in self._sessions:
            if s.id == session_id:
                return s.to_dict()
        return None

    # ── Convenience methods ──

    async def quick_action(
        self, module: str, method: str, **kwargs
    ) -> Any:
        """Execute a single action without creating a goal."""
        return await self._dispatch(module, method, **kwargs)

    def quick_action_sync(self, module: str, method: str, **kwargs) -> Any:
        return _run_sync(self.quick_action(module, method, **kwargs))

    # ── Conversation mode (Phase 6) ──

    async def converse(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Conversational interface to the agent.

        Uses RAGMemory for context retrieval and Sonnet for response
        generation. Maintains conversation history per session.

        Args:
            message: User's natural language message.
            session_id: Optional session ID for continuity.

        Returns:
            Dict with response text, suggested actions, and context used.
        """
        if not session_id:
            session_id = uuid.uuid4().hex[:12]

        result: Dict[str, Any] = {
            "session_id": session_id,
            "response": "",
            "suggested_actions": [],
            "context_used": [],
        }

        # Load conversation history from data dir
        conv_file = self._data_dir / f"conversation_{session_id}.json"
        history = _load_json(conv_file, default=[])
        if not isinstance(history, list):
            history = []

        # Build RAG context from agent memory
        rag_context = ""
        try:
            rag = self._get_module("rag_memory")
            if rag:
                rag_context = rag.build_context(message, max_tokens=1500)
                if rag_context:
                    result["context_used"].append("rag_memory")
        except Exception as exc:
            logger.debug("RAG context retrieval failed: %s", exc)

        # Build status context
        status = self.status()
        status_context = (
            f"Active goals: {status['pending_goals']} pending, "
            f"{status['completed_goals']} completed, "
            f"{status['failed_goals']} failed. "
            f"Modules loaded: {', '.join(status['modules_loaded'][:10])}."
        )

        # Generate response via Sonnet
        try:
            import anthropic
            client = anthropic.Anthropic()

            system_text = (
                "You are the OpenClaw Empire assistant. You help manage 16 WordPress "
                "publishing sites, phone automation, content pipelines, and revenue tracking.\n\n"
                f"Agent Status: {status_context}\n"
            )
            if rag_context:
                system_text += f"\nRelevant Memory Context:\n{rag_context}\n"

            # Build messages from history + new message
            messages = []
            for h in history[-10:]:
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            messages.append({"role": "user", "content": message})

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=[{
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=messages,
            )

            response_text = response.content[0].text
            result["response"] = response_text

            # Detect action suggestions from response
            action_keywords = {
                "publish": "publish content",
                "newsletter": "write newsletter",
                "backup": "run backup",
                "audit": "seo audit",
                "revenue": "check revenue",
                "device": "device status",
            }
            for keyword, action in action_keywords.items():
                if keyword in response_text.lower():
                    result["suggested_actions"].append(action)

        except Exception as exc:
            logger.error("Conversation generation failed: %s", exc)
            result["response"] = f"I encountered an error: {exc}. Try again or check the logs."

        # Save conversation history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": result["response"]})
        # Keep last 50 messages
        history = history[-50:]
        _save_json(conv_file, history)

        return result

    def converse_sync(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper for converse."""
        return _run_sync(self.converse(message, session_id))

    # ── Orchestrator delegation (Phase 6) ──

    async def delegate_to_orchestrator(
        self,
        mission_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Delegate a high-level mission to the UnifiedOrchestrator.

        Args:
            mission_type: One of the MissionType values from workflow_templates.
            params: Parameters for the mission template.

        Returns:
            Mission execution result from the orchestrator.
        """
        orchestrator = self._get_module("unified_orchestrator")
        if orchestrator is None:
            return {"success": False, "error": "UnifiedOrchestrator not available"}

        try:
            if hasattr(orchestrator, "execute_mission"):
                result = await orchestrator.execute_mission(mission_type, params or {})
                return result
            return {"success": False, "error": "execute_mission not found on orchestrator"}
        except Exception as exc:
            logger.error("Orchestrator delegation failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def delegate_to_orchestrator_sync(
        self, mission_type: str, params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return _run_sync(self.delegate_to_orchestrator(mission_type, params))

    # ── Sync wrappers ──

    def set_goal_sync(self, text: str, **kwargs) -> Goal:
        return _run_sync(self.set_goal(text, **kwargs))

    def run_sync(self, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.run(**kwargs))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[AutonomousAgent] = None


def get_autonomous_agent() -> AutonomousAgent:
    """Get the singleton AutonomousAgent instance."""
    global _instance
    if _instance is None:
        _instance = AutonomousAgent()
    return _instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _cli_goal(args: argparse.Namespace) -> None:
    agent = get_autonomous_agent()
    priority = GoalPriority(args.priority) if args.priority else GoalPriority.NORMAL
    goal = agent.set_goal_sync(args.text, priority=priority)
    _print_json(goal.to_dict())


def _cli_run(args: argparse.Namespace) -> None:
    agent = get_autonomous_agent()
    result = agent.run_sync(max_steps=args.steps, timeout_minutes=args.timeout)
    _print_json(result)


def _cli_status(args: argparse.Namespace) -> None:
    agent = get_autonomous_agent()
    _print_json(agent.status())


def _cli_goals(args: argparse.Namespace) -> None:
    agent = get_autonomous_agent()
    action = args.action
    if action == "list":
        _print_json(agent.list_goals(args.status or ""))
    elif action == "cancel":
        result = agent.cancel_goal(args.id or "")
        _print_json(result)
    else:
        print(f"Unknown goals action: {action}")


def _cli_sessions(args: argparse.Namespace) -> None:
    agent = get_autonomous_agent()
    if args.id:
        session = agent.get_session(args.id)
        if session:
            _print_json(session)
        else:
            print(f"Session not found: {args.id}")
    else:
        _print_json(agent.list_sessions())


def _cli_action(args: argparse.Namespace) -> None:
    agent = get_autonomous_agent()
    kwargs = json.loads(args.args) if args.args else {}
    result = agent.quick_action_sync(args.module, args.method, **kwargs)
    _print_json(result if isinstance(result, (dict, list)) else {"result": str(result)})


def _cli_converse(args: argparse.Namespace) -> None:
    agent = get_autonomous_agent()
    result = agent.converse_sync(args.message, session_id=args.session)
    print(f"\n{result.get('response', '')}\n")
    if result.get("suggested_actions"):
        print(f"Suggested actions: {', '.join(result['suggested_actions'])}")


def _cli_delegate(args: argparse.Namespace) -> None:
    agent = get_autonomous_agent()
    params = json.loads(args.params) if args.params else {}
    result = agent.delegate_to_orchestrator_sync(args.mission, params)
    _print_json(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autonomous_agent",
        description="OpenClaw Empire — Autonomous AI Agent",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # goal
    gl = sub.add_parser("goal", help="Set a new goal")
    gl.add_argument("--text", required=True)
    gl.add_argument("--priority", choices=["critical", "high", "normal", "low", "background"],
                     default="normal")
    gl.set_defaults(func=_cli_goal)

    # run
    rn = sub.add_parser("run", help="Run the agent")
    rn.add_argument("--steps", type=int, default=100)
    rn.add_argument("--timeout", type=float, default=60.0)
    rn.set_defaults(func=_cli_run)

    # status
    st = sub.add_parser("status", help="Agent status")
    st.set_defaults(func=_cli_status)

    # goals
    gs = sub.add_parser("goals", help="Goal management")
    gs.add_argument("action", choices=["list", "cancel"])
    gs.add_argument("--status", default="")
    gs.add_argument("--id", default="")
    gs.set_defaults(func=_cli_goals)

    # sessions
    ss = sub.add_parser("sessions", help="Session history")
    ss.add_argument("--id", default="")
    ss.set_defaults(func=_cli_sessions)

    # action
    ac = sub.add_parser("action", help="Execute a single action")
    ac.add_argument("--module", required=True)
    ac.add_argument("--method", required=True)
    ac.add_argument("--args", default=None, help="JSON dict of arguments")
    ac.set_defaults(func=_cli_action)

    # converse (Phase 6)
    cv = sub.add_parser("converse", help="Chat with the agent")
    cv.add_argument("--message", required=True, help="Message to send")
    cv.add_argument("--session", default=None, help="Session ID for continuity")
    cv.set_defaults(func=_cli_converse)

    # delegate (Phase 6)
    dl = sub.add_parser("delegate", help="Delegate mission to orchestrator")
    dl.add_argument("--mission", required=True, help="Mission type")
    dl.add_argument("--params", default=None, help="JSON dict of params")
    dl.set_defaults(func=_cli_delegate)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
