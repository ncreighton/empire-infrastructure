"""
Intelligence Hub -- Central Coordinator for OpenClaw Empire
============================================================

Wires together FORGE, AMPLIFY, Vision Agent, Screenpipe Agent, and Phone
Controller into a single unified interface for Android phone automation.

Subsystems:
    FORGE           -- Pre-task intelligence (Scout, Sentinel, Oracle, Smith, Codex)
    AMPLIFY         -- 6-stage task enhancement pipeline
    VisionAgent     -- AI-powered phone screenshot analysis
    ScreenpipeAgent -- Passive OCR/audio/UI event monitoring
    PhoneController -- Low-level ADB command execution
    TaskExecutor    -- High-level task decomposition and execution

Usage:
    from src.intelligence_hub import IntelligenceHub, get_hub

    hub = IntelligenceHub()
    # or use singleton:
    hub = get_hub()

    # Full end-to-end
    result = await hub.execute_task("open WordPress and publish a post")

    # Pre-task analysis only
    readiness = await hub.pre_task("publish to witchcraftforbeginners", app="wordpress")

    # Quick phone check
    state = await hub.get_phone_state()
    screenshot = await hub.quick_screenshot()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("intelligence_hub")

# ---------------------------------------------------------------------------
# Lazy imports to avoid circular issues and allow partial subsystem availability
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).parent.parent


def _import_forge():
    from src.forge_engine import ForgeEngine
    return ForgeEngine


def _import_amplify():
    from src.amplify_pipeline import AmplifyPipeline
    return AmplifyPipeline


def _import_phone():
    from src.phone_controller import PhoneController, TaskExecutor
    return PhoneController, TaskExecutor


def _import_vision():
    from src.vision_agent import VisionAgent
    return VisionAgent


def _import_screenpipe():
    from src.screenpipe_agent import ScreenpipeAgent
    return ScreenpipeAgent


# ── Phase 5: Autonomous AI Phone Agent lazy imports ──

def _import_agent_memory():
    from src.agent_memory import get_memory
    return get_memory


def _import_phone_os():
    from src.phone_os_agent import get_phone_os_agent
    return get_phone_os_agent


def _import_browser():
    from src.browser_controller import get_browser
    return get_browser


def _import_identity():
    from src.identity_manager import get_identity_manager
    return get_identity_manager


def _import_app_learner():
    from src.app_learner import get_app_learner
    return get_app_learner


def _import_app_discovery():
    from src.app_discovery import get_app_discovery
    return get_app_discovery


def _import_email_agent():
    from src.email_agent import get_email_agent
    return get_email_agent


def _import_account_factory():
    from src.account_factory import get_account_factory
    return get_account_factory


def _import_social_agent():
    from src.social_media_agent import get_social_agent
    return get_social_agent


def _import_autonomous_agent():
    from src.autonomous_agent import get_autonomous_agent
    return get_autonomous_agent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HubConfig:
    """Configuration for IntelligenceHub subsystem connections."""
    node_url: str = os.getenv("OPENCLAW_NODE_URL", "http://localhost:18789")
    node_name: str = os.getenv("OPENCLAW_ANDROID_NODE", "android")
    vision_url: str = os.getenv("VISION_SERVICE_URL", "http://localhost:8002")
    screenpipe_url: str = os.getenv("SCREENPIPE_URL", "http://localhost:3030")
    vision_timeout: int = int(os.getenv("VISION_TIMEOUT", "30"))
    screenpipe_timeout: int = int(os.getenv("SCREENPIPE_TIMEOUT", "15"))
    command_timeout: float = float(os.getenv("COMMAND_TIMEOUT", "30"))
    confirm_irreversible: bool = True
    max_auto_fix_attempts: int = 3
    screenpipe_error_lookback_minutes: int = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_task_id() -> str:
    return f"task-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Task tracking
# ---------------------------------------------------------------------------

@dataclass
class RunningTask:
    """In-memory record of a task in progress."""
    task_id: str
    description: str
    app: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed, stalled
    started_at: str = ""
    completed_at: str = ""
    steps_completed: int = 0
    steps_total: int = 0
    errors: List[str] = field(default_factory=list)
    learnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    progress_percent: float = 0.0
    current_step: str = ""
    plan: Any = None  # TaskPlan from phone_controller


# ===================================================================
# IntelligenceHub
# ===================================================================

class IntelligenceHub:
    """
    Central coordinator for the OpenClaw Empire intelligence system.

    Creates instances of all subsystems (FORGE, AMPLIFY, Vision Agent,
    Screenpipe Agent, Phone Controller) and orchestrates them through
    unified pre_task, execute_task, monitor_task, and post_task flows.

    Every method is available in both async and sync variants. Errors in
    individual subsystems are caught and logged without crashing the hub.
    """

    def __init__(
        self,
        config: Optional[HubConfig] = None,
        forge_engine: Optional[Any] = None,
        amplify_pipeline: Optional[Any] = None,
        phone_controller: Optional[Any] = None,
        task_executor: Optional[Any] = None,
        vision_agent: Optional[Any] = None,
        screenpipe_agent: Optional[Any] = None,
    ) -> None:
        self.config = config or HubConfig()
        self._tasks: Dict[str, RunningTask] = {}

        # --- Subsystem initialization (graceful per-module) ---

        # FORGE
        self.forge = forge_engine
        if self.forge is None:
            try:
                ForgeEngine = _import_forge()
                self.forge = ForgeEngine()
                logger.info("FORGE engine initialized")
            except Exception as exc:
                logger.warning("FORGE engine unavailable: %s", exc)
                self.forge = None

        # AMPLIFY
        self.amplify = amplify_pipeline
        if self.amplify is None:
            try:
                AmplifyPipeline = _import_amplify()
                self.amplify = AmplifyPipeline()
                logger.info("AMPLIFY pipeline initialized")
            except Exception as exc:
                logger.warning("AMPLIFY pipeline unavailable: %s", exc)
                self.amplify = None

        # Phone Controller
        self.phone = phone_controller
        if self.phone is None:
            try:
                PhoneController, _ = _import_phone()
                self.phone = PhoneController(
                    node_url=self.config.node_url,
                    node_name=self.config.node_name,
                    command_timeout=self.config.command_timeout,
                )
                logger.info("Phone controller initialized (node=%s)", self.config.node_url)
            except Exception as exc:
                logger.warning("Phone controller unavailable: %s", exc)
                self.phone = None

        # Task Executor
        self.executor = task_executor
        if self.executor is None and self.phone is not None:
            try:
                _, TaskExecutor = _import_phone()
                self.executor = TaskExecutor(
                    controller=self.phone,
                    forge_engine=self.forge,
                    amplify_pipeline=self.amplify,
                )
                logger.info("Task executor initialized")
            except Exception as exc:
                logger.warning("Task executor unavailable: %s", exc)
                self.executor = None

        # Vision Agent
        self.vision = vision_agent
        if self.vision is None:
            try:
                VisionAgent = _import_vision()
                self.vision = VisionAgent(
                    base_url=self.config.vision_url,
                    timeout=self.config.vision_timeout,
                )
                logger.info("Vision agent initialized (url=%s)", self.config.vision_url)
            except Exception as exc:
                logger.warning("Vision agent unavailable: %s", exc)
                self.vision = None

        # Screenpipe Agent
        self.screenpipe = screenpipe_agent
        if self.screenpipe is None:
            try:
                ScreenpipeAgent = _import_screenpipe()
                self.screenpipe = ScreenpipeAgent(
                    base_url=self.config.screenpipe_url,
                    timeout=self.config.screenpipe_timeout,
                )
                logger.info("Screenpipe agent initialized (url=%s)", self.config.screenpipe_url)
            except Exception as exc:
                logger.warning("Screenpipe agent unavailable: %s", exc)
                self.screenpipe = None

        logger.info(
            "IntelligenceHub ready: forge=%s amplify=%s phone=%s vision=%s screenpipe=%s",
            self.forge is not None,
            self.amplify is not None,
            self.phone is not None,
            self.vision is not None,
            self.screenpipe is not None,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all subsystem sessions."""
        if self.phone is not None:
            try:
                await self.phone.close()
            except Exception:
                pass
        if self.vision is not None:
            try:
                await self.vision.close()
            except Exception:
                pass
        if self.screenpipe is not None:
            try:
                await self.screenpipe.close()
            except Exception:
                pass
        logger.info("IntelligenceHub closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ==================================================================
    # 1. pre_task -- Readiness analysis before execution
    # ==================================================================

    async def pre_task(self, task_description: str, app: Optional[str] = None) -> dict:
        """
        Analyze readiness before executing a task.

        Gathers phone state, runs FORGE pre-flight (Scout + Oracle),
        runs AMPLIFY full pipeline, and checks Screenpipe for recent errors.

        Returns a dict with: readiness, enhanced_config, risk_assessment,
        warnings, go_no_go decision.
        """
        logger.info("pre_task: %s (app=%s)", task_description, app)
        result: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "task_description": task_description,
            "app": app,
            "readiness": False,
            "enhanced_config": {},
            "risk_assessment": {},
            "warnings": [],
            "go_no_go": "UNKNOWN",
            "phone_state": {},
            "screenpipe_errors": [],
            "forge_report": {},
        }

        # --- 1. Get phone state ---
        phone_state = await self._safe_get_phone_state()
        result["phone_state"] = phone_state

        # --- 2. Parse task into structured format ---
        task_spec = self._parse_task(task_description, app, phone_state)

        # --- 3. FORGE pre-flight (Scout + Oracle + Smith) ---
        forge_report = {}
        if self.forge is not None:
            try:
                forge_report = await self.forge.pre_flight(phone_state, task_spec)
                result["forge_report"] = forge_report
                result["risk_assessment"] = forge_report.get("oracle", {})
                result["readiness"] = forge_report.get("ready", False)
                result["go_no_go"] = forge_report.get("go_no_go", "UNKNOWN")

                # Collect warnings from Scout
                scout = forge_report.get("scout", {})
                for w in scout.get("warnings", []):
                    result["warnings"].append(w.get("message", str(w)))
                for b in scout.get("blocking_issues", []):
                    result["warnings"].append(f"BLOCKING: {b.get('message', str(b))}")

            except Exception as exc:
                logger.warning("FORGE pre-flight failed: %s", exc)
                result["warnings"].append(f"FORGE analysis unavailable: {exc}")

        # --- 4. AMPLIFY full pipeline ---
        if self.amplify is not None:
            try:
                enhanced = self.amplify.full_pipeline(dict(task_spec))
                result["enhanced_config"] = {
                    "stages_completed": enhanced.get("_amplify", {}).get("stages_completed", []),
                    "validation_summary": enhanced.get("validation_summary", {}),
                    "batch_groups": enhanced.get("batch_groups", []),
                    "step_count": len(enhanced.get("steps", [])),
                }
                validation = enhanced.get("validation_summary", {})
                if not validation.get("valid", True):
                    result["warnings"].extend(validation.get("blocking_failures", []))
                    if result["go_no_go"] == "GO":
                        result["go_no_go"] = "CAUTION"
            except Exception as exc:
                logger.warning("AMPLIFY pipeline failed: %s", exc)
                result["warnings"].append(f"AMPLIFY enhancement unavailable: {exc}")

        # --- 5. Screenpipe recent errors ---
        if self.screenpipe is not None:
            try:
                errors = await self.screenpipe.search_errors(
                    app_name=app,
                    minutes_back=self.config.screenpipe_error_lookback_minutes,
                )
                error_summaries = [
                    {"text": e.content[:200], "app": e.app_name, "timestamp": e.timestamp}
                    for e in errors[:10]
                ]
                result["screenpipe_errors"] = error_summaries
                if error_summaries:
                    result["warnings"].append(
                        f"Screenpipe detected {len(error_summaries)} recent error(s) for {app or 'all apps'}"
                    )
            except Exception as exc:
                logger.debug("Screenpipe error check failed: %s", exc)

        # Final readiness determination if FORGE was not available
        if not forge_report:
            result["readiness"] = len([
                w for w in result["warnings"] if "BLOCKING" in w
            ]) == 0
            result["go_no_go"] = "GO" if result["readiness"] else "CAUTION"

        logger.info(
            "pre_task complete: go=%s, readiness=%s, warnings=%d",
            result["go_no_go"], result["readiness"], len(result["warnings"]),
        )
        return result

    def pre_task_sync(self, task_description: str, app: Optional[str] = None) -> dict:
        """Synchronous wrapper for pre_task."""
        return _run_sync(self.pre_task(task_description, app))

    # ==================================================================
    # 2. execute_task -- Full end-to-end execution
    # ==================================================================

    async def execute_task(
        self,
        task_description: str,
        app: Optional[str] = None,
        confirm_irreversible: Optional[bool] = None,
    ) -> dict:
        """
        Full end-to-end task execution with intelligence integration.

        Flow: pre_task analysis -> auto-fix blocking issues -> execute via
        TaskExecutor -> monitor via Screenpipe -> post-task learning.

        Returns a dict with: task_id, status, duration, steps_completed,
        steps_total, errors, learnings.
        """
        task_id = _gen_task_id()
        if confirm_irreversible is None:
            confirm_irreversible = self.config.confirm_irreversible
        start_time = time.monotonic()

        task_record = RunningTask(
            task_id=task_id,
            description=task_description,
            app=app,
            status="running",
            started_at=_now_iso(),
        )
        self._tasks[task_id] = task_record
        logger.info("execute_task [%s]: %s (app=%s)", task_id, task_description, app)

        # --- Phase 1: Pre-task analysis ---
        pre = await self.pre_task(task_description, app)

        # --- Phase 2: Auto-fix blocking issues ---
        if not pre.get("readiness", False) and self.forge is not None:
            fixes = pre.get("forge_report", {}).get("fixes", [])
            if fixes:
                logger.info("[%s] Attempting %d auto-fixes", task_id, len(fixes))
                for i, fix in enumerate(fixes[:self.config.max_auto_fix_attempts]):
                    task_record.learnings.append(
                        f"Auto-fix attempted: {fix.get('strategy', 'unknown')} for {fix.get('issue', 'unknown')}"
                    )
                    logger.info("  Fix %d: %s (%s)", i + 1, fix.get("strategy"), fix.get("issue"))

                # Re-check readiness after fixes
                pre = await self.pre_task(task_description, app)

        # --- Phase 3: Go/No-Go ---
        if pre.get("go_no_go") == "NO_GO":
            task_record.status = "failed"
            task_record.completed_at = _now_iso()
            task_record.duration_seconds = time.monotonic() - start_time
            task_record.errors.append("Pre-task check returned NO_GO")
            task_record.errors.extend(pre.get("warnings", []))
            logger.warning("[%s] Aborted: NO_GO", task_id)
            return self._task_result(task_record)

        # --- Phase 4: Execute via TaskExecutor ---
        plan = None
        if self.executor is not None:
            try:
                # Record in FORGE Codex before execution
                if self.forge is not None:
                    try:
                        self.forge.record_task(
                            task_id, app or "unknown",
                            [{"description": task_description}],
                            metadata={"pre_task": pre.get("go_no_go")},
                        )
                    except Exception:
                        pass

                plan = await self.executor.execute(task_description)
                task_record.plan = plan
                task_record.status = plan.status
                task_record.steps_total = len(plan.steps)
                task_record.steps_completed = sum(1 for s in plan.steps if s.completed)
                task_record.progress_percent = (
                    (task_record.steps_completed / task_record.steps_total * 100)
                    if task_record.steps_total > 0 else 0
                )

                # Collect errors from failed steps
                for step in plan.steps:
                    if step.result and not step.result.success and step.result.error:
                        task_record.errors.append(
                            f"Step {step.step_number}: {step.result.error}"
                        )

            except Exception as exc:
                logger.error("[%s] Execution failed: %s", task_id, exc)
                task_record.status = "failed"
                task_record.errors.append(f"Execution error: {exc}")
        else:
            task_record.status = "failed"
            task_record.errors.append("TaskExecutor not available (phone not connected)")

        # --- Phase 5: Post-task learning ---
        task_record.completed_at = _now_iso()
        task_record.duration_seconds = time.monotonic() - start_time

        post_result = await self.post_task(
            task_id,
            success=(task_record.status == "completed"),
            duration=task_record.duration_seconds,
            error=task_record.errors[0] if task_record.errors else None,
        )
        task_record.learnings.extend(post_result.get("learnings_recorded", []))

        # --- Phase 6: Screenpipe post-execution check ---
        if self.screenpipe is not None:
            try:
                post_errors = await self.screenpipe.search_errors(
                    app_name=app, minutes_back=2,
                )
                for e in post_errors[:3]:
                    task_record.learnings.append(
                        f"Post-execution error observed: {e.content[:100]}"
                    )
            except Exception:
                pass

        logger.info(
            "[%s] Complete: status=%s, steps=%d/%d, duration=%.1fs",
            task_id, task_record.status,
            task_record.steps_completed, task_record.steps_total,
            task_record.duration_seconds,
        )
        return self._task_result(task_record)

    def execute_task_sync(
        self,
        task_description: str,
        app: Optional[str] = None,
        confirm_irreversible: Optional[bool] = None,
    ) -> dict:
        """Synchronous wrapper for execute_task."""
        return _run_sync(self.execute_task(task_description, app, confirm_irreversible))

    # ==================================================================
    # 3. monitor_task -- Check running task progress
    # ==================================================================

    async def monitor_task(self, task_id: str) -> dict:
        """
        Check progress of a running or completed task.

        Uses internal task records plus Screenpipe and Vision for
        live progress detection on running tasks.
        """
        record = self._tasks.get(task_id)
        if record is None:
            return {
                "task_id": task_id,
                "status": "not_found",
                "error": f"No task with id '{task_id}' found",
            }

        result: Dict[str, Any] = {
            "task_id": task_id,
            "status": record.status,
            "progress_percent": record.progress_percent,
            "current_step": record.current_step,
            "steps_completed": record.steps_completed,
            "steps_total": record.steps_total,
            "errors": record.errors,
            "duration_seconds": record.duration_seconds,
        }

        # If still running, enrich with live data
        if record.status == "running":
            # Check Screenpipe for recent activity
            if self.screenpipe is not None:
                try:
                    current = await self.screenpipe.get_current_state(record.app)
                    if current:
                        result["live_screen_text"] = current[0].content[:300]
                except Exception:
                    pass

            # Check Vision for current screen state
            if self.vision is not None and self.phone is not None:
                try:
                    screenshot_path = await self.phone.screenshot()
                    state, confidence, details = await self.vision.detect_state(
                        image_path=screenshot_path,
                    )
                    result["live_state"] = {
                        "state": state.value,
                        "confidence": confidence,
                        "details": details,
                    }
                    # Detect if task is stalled
                    if state.value in ("error", "crash"):
                        result["status"] = "error"
                        record.status = "error"
                except Exception as exc:
                    logger.debug("Live state check failed: %s", exc)

            # Update duration for running tasks
            if record.started_at:
                try:
                    started = datetime.fromisoformat(record.started_at)
                    now = datetime.now(timezone.utc)
                    result["duration_seconds"] = (now - started).total_seconds()
                except (ValueError, TypeError):
                    pass

        return result

    def monitor_task_sync(self, task_id: str) -> dict:
        """Synchronous wrapper for monitor_task."""
        return _run_sync(self.monitor_task(task_id))

    # ==================================================================
    # 4. post_task -- Record outcomes for learning
    # ==================================================================

    async def post_task(
        self,
        task_id: str,
        success: bool,
        duration: float,
        error: Optional[str] = None,
    ) -> dict:
        """
        Record task outcome for FORGE Codex and AMPLIFY Optimize learning.

        Feeds Screenpipe observations into the Codex and returns a summary
        of what was learned.
        """
        result: Dict[str, Any] = {
            "task_id": task_id,
            "success": success,
            "learnings_recorded": [],
            "pattern_updates": [],
        }

        record = self._tasks.get(task_id)
        app = record.app if record else "unknown"
        outcome = "success" if success else "failure"

        # --- FORGE Codex learning ---
        if self.forge is not None:
            try:
                self.forge.record_outcome(
                    task_id,
                    outcome=outcome,
                    duration_seconds=duration,
                    error=error,
                )
                result["learnings_recorded"].append(f"FORGE Codex: recorded {outcome} for {app}")
            except Exception as exc:
                logger.debug("FORGE outcome recording failed: %s", exc)

        # --- AMPLIFY timing data ---
        if self.amplify is not None:
            try:
                action_type = "full_task"
                if record and record.plan and hasattr(record.plan, "steps"):
                    for step in record.plan.steps:
                        if step.result:
                            step_duration = step.result.duration_ms / 1000.0
                            step_action = step.action.action_type.value if step.action else "unknown"
                            self.amplify.record_execution(
                                action_type=step_action,
                                app_name=app,
                                duration=step_duration,
                                success=step.completed,
                            )
                            result["learnings_recorded"].append(
                                f"AMPLIFY Optimize: {step_action} on {app} -> {step_duration:.1f}s"
                            )
                else:
                    self.amplify.record_execution(
                        action_type=action_type,
                        app_name=app,
                        duration=duration,
                        success=success,
                    )
                    result["learnings_recorded"].append(
                        f"AMPLIFY Optimize: {action_type} on {app} -> {duration:.1f}s"
                    )
            except Exception as exc:
                logger.debug("AMPLIFY timing recording failed: %s", exc)

        # --- Screenpipe observations -> Codex ---
        if self.screenpipe is not None and self.forge is not None:
            try:
                observations = self.screenpipe.codex.get_observations(limit=10)
                for obs in observations:
                    if obs.get("category") == "error":
                        err_data = obs.get("data", {})
                        self.forge.codex.add_vision_tip(
                            app or "general",
                            f"Error seen in {err_data.get('app', 'unknown')}: {err_data.get('text', '')[:100]}",
                        )
                        result["pattern_updates"].append(
                            f"Error pattern recorded for {err_data.get('app', 'unknown')}"
                        )
            except Exception as exc:
                logger.debug("Screenpipe -> Codex sync failed: %s", exc)

        logger.info(
            "post_task [%s]: %d learnings, %d pattern updates",
            task_id, len(result["learnings_recorded"]), len(result["pattern_updates"]),
        )
        return result

    def post_task_sync(
        self,
        task_id: str,
        success: bool,
        duration: float,
        error: Optional[str] = None,
    ) -> dict:
        """Synchronous wrapper for post_task."""
        return _run_sync(self.post_task(task_id, success, duration, error))

    # ==================================================================
    # 5. quick_screenshot -- Capture and analyze
    # ==================================================================

    async def quick_screenshot(self) -> dict:
        """
        Take a screenshot and analyze it with VisionAgent.

        Returns: screenshot_path, analysis (current app, screen, elements).
        """
        result: Dict[str, Any] = {
            "screenshot_path": None,
            "analysis": {},
            "error": None,
        }

        if self.phone is None:
            result["error"] = "Phone controller not available"
            return result

        try:
            screenshot_path = await self.phone.screenshot()
            result["screenshot_path"] = screenshot_path
        except Exception as exc:
            result["error"] = f"Screenshot failed: {exc}"
            return result

        if self.vision is not None:
            try:
                analysis = await self.vision.analyze_screen(image_path=screenshot_path)
                result["analysis"] = {
                    "current_app": analysis.current_app,
                    "current_screen": analysis.current_screen,
                    "visible_text": analysis.visible_text[:20],
                    "elements_count": len(analysis.tappable_elements),
                    "keyboard_visible": analysis.keyboard_visible,
                    "loading": analysis.loading_indicators,
                    "errors": analysis.errors_detected,
                    "quality_score": analysis.quality_score,
                    "analysis_time_ms": round(analysis.analysis_time_ms, 1),
                }
            except Exception as exc:
                result["analysis"] = {"error": f"Vision analysis failed: {exc}"}

        return result

    def quick_screenshot_sync(self) -> dict:
        """Synchronous wrapper for quick_screenshot."""
        return _run_sync(self.quick_screenshot())

    # ==================================================================
    # 6. get_phone_state -- Comprehensive device state
    # ==================================================================

    async def get_phone_state(self) -> dict:
        """
        Get comprehensive phone state: screen on, battery, wifi, active app,
        notifications, storage. Uses PhoneController + Screenpipe data.
        """
        return await self._safe_get_phone_state()

    def get_phone_state_sync(self) -> dict:
        """Synchronous wrapper for get_phone_state."""
        return _run_sync(self.get_phone_state())

    # ==================================================================
    # 7. get_intelligence_stats -- Combined subsystem statistics
    # ==================================================================

    async def get_intelligence_stats(self) -> dict:
        """Combined statistics from all subsystems."""
        stats: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "subsystems": {
                "forge": self.forge is not None,
                "amplify": self.amplify is not None,
                "phone": self.phone is not None,
                "executor": self.executor is not None,
                "vision": self.vision is not None,
                "screenpipe": self.screenpipe is not None,
            },
            "active_tasks": len([t for t in self._tasks.values() if t.status == "running"]),
            "total_tasks": len(self._tasks),
            "task_history": [],
        }

        # FORGE stats
        if self.forge is not None:
            try:
                stats["forge"] = self.forge.get_stats()
            except Exception as exc:
                stats["forge"] = {"error": str(exc)}

        # AMPLIFY stats
        if self.amplify is not None:
            try:
                # Get stats for known empire apps
                app_stats = {}
                for app_key in ["wordpress", "chrome", "instagram", "pinterest", "etsy"]:
                    try:
                        app_data = self.amplify.get_app_stats(app_key)
                        if app_data.get("total_records", 0) > 0:
                            app_stats[app_key] = app_data
                    except Exception:
                        pass
                stats["amplify"] = {
                    "app_performance": app_stats,
                }
            except Exception as exc:
                stats["amplify"] = {"error": str(exc)}

        # Vision Agent stats
        if self.vision is not None:
            try:
                quality_history = self.vision.sentinel.get_quality_history()
                stats["vision"] = {
                    "quality_reports": len(quality_history),
                    "avg_quality": round(self.vision.sentinel.average_quality(), 3),
                    "by_task": {
                        task: round(self.vision.sentinel.average_quality(task), 3)
                        for task in ["analyze", "find_element", "detect_state", "detect_errors", "compare"]
                        if self.vision.sentinel.get_quality_history(task)
                    },
                }
            except Exception as exc:
                stats["vision"] = {"error": str(exc)}

        # Screenpipe Agent stats
        if self.screenpipe is not None:
            try:
                codex_summary = self.screenpipe.codex.summarize()
                stats["screenpipe"] = {
                    "codex_observations": codex_summary,
                    "total_observations": sum(codex_summary.values()),
                }
            except Exception as exc:
                stats["screenpipe"] = {"error": str(exc)}

        # Task history summary
        for task_id, record in list(self._tasks.items())[-20:]:
            stats["task_history"].append({
                "task_id": record.task_id,
                "description": record.description[:80],
                "status": record.status,
                "duration": round(record.duration_seconds, 1),
                "steps": f"{record.steps_completed}/{record.steps_total}",
            })

        return stats

    def get_intelligence_stats_sync(self) -> dict:
        """Synchronous wrapper for get_intelligence_stats."""
        return _run_sync(self.get_intelligence_stats())

    # ==================================================================
    # Internal helpers
    # ==================================================================

    async def _safe_get_phone_state(self) -> dict:
        """
        Build a phone_state dict compatible with FORGE Scout.

        Gathers data from PhoneController, VisionAgent, and Screenpipe,
        falling back to safe defaults when subsystems are unavailable.
        """
        state: Dict[str, Any] = {
            "screen_on": True,
            "locked": False,
            "battery_percent": 100,
            "battery_charging": False,
            "wifi_connected": True,
            "wifi_ssid": "",
            "storage_free_mb": 9999,
            "active_app": "",
            "active_window": "",
            "installed_apps": [],
            "notifications": [],
            "visible_dialogs": [],
        }

        if self.phone is None:
            state["_source"] = "defaults_only"
            return state

        # Try to get active app
        try:
            active_app = await self.phone.get_current_app()
            state["active_app"] = active_app
        except Exception as exc:
            logger.debug("Could not get active app: %s", exc)

        # Try screenshot + vision analysis for richer state
        if self.vision is not None:
            try:
                screenshot_path = await self.phone.screenshot()
                errors = await self.vision.detect_errors(image_path=screenshot_path)
                if errors.has_errors:
                    state["visible_dialogs"].append(errors.error_message)

                app_state, confidence, details = await self.vision.detect_state(
                    image_path=screenshot_path,
                )
                if app_state.value == "logged_out":
                    state["visible_dialogs"].append("App appears to be logged out")
                elif app_state.value in ("crash", "error"):
                    state["visible_dialogs"].append(f"App state: {app_state.value} - {details}")

            except Exception as exc:
                logger.debug("Vision-based state check failed: %s", exc)

        # Try UI dump for installed app detection
        try:
            elements = await self.phone.ui_dump()
            if elements:
                state["screen_on"] = True
                state["locked"] = False
            state["_ui_elements_count"] = len(elements)
        except Exception as exc:
            logger.debug("UI dump failed: %s", exc)

        # Screenpipe recent state
        if self.screenpipe is not None:
            try:
                recent = await self.screenpipe.get_current_state()
                if recent:
                    state["active_window"] = recent[0].window_name
                    if not state["active_app"] and recent[0].app_name:
                        state["active_app"] = recent[0].app_name
            except Exception as exc:
                logger.debug("Screenpipe state check failed: %s", exc)

        state["_source"] = "live"
        return state

    def _parse_task(self, description: str, app: Optional[str], phone_state: dict) -> dict:
        """Parse a task description into a structured dict for FORGE/AMPLIFY."""
        task: Dict[str, Any] = {
            "app": app or "",
            "task_description": description,
            "needs_network": True,
            "needs_auth": False,
            "is_irreversible": False,
            "time_sensitive": False,
            "steps": [],
        }

        desc_lower = description.lower()

        # Infer properties from description
        irreversible_keywords = ["publish", "post", "send", "delete", "purchase", "buy", "pay"]
        auth_keywords = ["login", "sign in", "authenticate", "credentials"]
        network_keywords = ["upload", "download", "publish", "browse", "search", "fetch"]

        task["is_irreversible"] = any(kw in desc_lower for kw in irreversible_keywords)
        task["needs_auth"] = any(kw in desc_lower for kw in auth_keywords)
        task["needs_network"] = any(kw in desc_lower for kw in network_keywords)

        # Estimate steps from description complexity
        word_count = len(description.split())
        if word_count <= 5:
            task["estimated_steps"] = 3
        elif word_count <= 15:
            task["estimated_steps"] = 6
        else:
            task["estimated_steps"] = 10

        return task

    def _task_result(self, record: RunningTask) -> dict:
        """Convert a RunningTask record into the public result dict."""
        return {
            "task_id": record.task_id,
            "status": record.status,
            "duration": round(record.duration_seconds, 2),
            "steps_completed": record.steps_completed,
            "steps_total": record.steps_total,
            "progress_percent": round(record.progress_percent, 1),
            "errors": record.errors,
            "learnings": record.learnings,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
        }


# ===================================================================
# Singleton access
# ===================================================================

_hub: Optional[IntelligenceHub] = None


def get_hub(config: Optional[HubConfig] = None) -> IntelligenceHub:
    """Get or create the singleton IntelligenceHub instance."""
    global _hub
    if _hub is None:
        _hub = IntelligenceHub(config=config)
    return _hub


def reset_hub() -> None:
    """Reset the singleton (useful for tests)."""
    global _hub
    _hub = None


# ===================================================================
# Sync helper
# ===================================================================

def _run_sync(coro):
    """Run an async coroutine from a synchronous context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    import argparse
    import json
    import pprint

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="OpenClaw Intelligence Hub -- unified automation coordinator",
    )
    sub = parser.add_subparsers(dest="command")

    # pre-task
    p_pre = sub.add_parser("pre-task", help="Run pre-task readiness analysis")
    p_pre.add_argument("description", help="Task description")
    p_pre.add_argument("--app", help="Target app name")

    # execute
    p_exec = sub.add_parser("execute", help="Execute a task end-to-end")
    p_exec.add_argument("description", help="Task description")
    p_exec.add_argument("--app", help="Target app name")

    # screenshot
    p_screen = sub.add_parser("screenshot", help="Take and analyze a screenshot")

    # phone-state
    p_state = sub.add_parser("phone-state", help="Get comprehensive phone state")

    # stats
    p_stats = sub.add_parser("stats", help="Show intelligence stats from all subsystems")

    # demo (offline, no connections required)
    p_demo = sub.add_parser("demo", help="Run offline demo with mock data")

    args = parser.parse_args()

    if args.command == "pre-task":
        hub = IntelligenceHub()
        result = _run_sync(hub.pre_task(args.description, app=args.app))
        pprint.pprint(result, width=100)
        _run_sync(hub.close())

    elif args.command == "execute":
        hub = IntelligenceHub()
        result = _run_sync(hub.execute_task(args.description, app=args.app))
        pprint.pprint(result, width=100)
        _run_sync(hub.close())

    elif args.command == "screenshot":
        hub = IntelligenceHub()
        result = _run_sync(hub.quick_screenshot())
        pprint.pprint(result, width=100)
        _run_sync(hub.close())

    elif args.command == "phone-state":
        hub = IntelligenceHub()
        result = _run_sync(hub.get_phone_state())
        pprint.pprint(result, width=100)
        _run_sync(hub.close())

    elif args.command == "stats":
        hub = IntelligenceHub()
        result = _run_sync(hub.get_intelligence_stats())
        print(json.dumps(result, indent=2, default=str))
        _run_sync(hub.close())

    elif args.command == "demo":
        print("=" * 60)
        print("Intelligence Hub -- Offline Demo")
        print("=" * 60)

        # Create hub -- subsystems that can't connect will be None
        hub = IntelligenceHub()

        print(f"\nSubsystems available:")
        for name in ["forge", "amplify", "phone", "executor", "vision", "screenpipe"]:
            available = getattr(hub, name) is not None
            status = "OK" if available else "N/A"
            print(f"  {name:>15}: {status}")

        # Show stats
        stats = _run_sync(hub.get_intelligence_stats())
        print(f"\nIntelligence Stats:")
        for key, val in stats.get("subsystems", {}).items():
            print(f"  {key}: {'available' if val else 'offline'}")

        if hub.forge is not None:
            forge_stats = stats.get("forge", {})
            codex = forge_stats.get("codex", {})
            print(f"\n  FORGE Codex:")
            print(f"    Tasks recorded: {codex.get('total_tasks', 0)}")
            print(f"    Apps known: {codex.get('apps_known', 0)}")
            print(f"    Success rate: {codex.get('overall_success_rate', 0):.1%}")
            print(f"    Failure patterns: {codex.get('total_failure_patterns', 0)}")
            print(f"    Vision tips: {codex.get('total_vision_tips', 0)}")

        print(f"\n  Active tasks: {stats.get('active_tasks', 0)}")
        print(f"  Total tasks: {stats.get('total_tasks', 0)}")

        # Demo pre_task with mock (FORGE-only, no phone needed)
        if hub.forge is not None:
            print("\n" + "-" * 60)
            print("Demo: FORGE pre-flight with mock phone state")
            print("-" * 60)

            mock_phone = {
                "screen_on": True,
                "locked": False,
                "battery_percent": 85,
                "battery_charging": False,
                "wifi_connected": True,
                "wifi_ssid": "HomeNetwork",
                "storage_free_mb": 4096,
                "active_app": "launcher",
                "active_window": "Home",
                "installed_apps": ["chrome", "wordpress", "gmail", "whatsapp"],
                "notifications": [],
                "visible_dialogs": [],
            }
            mock_task = {
                "app": "wordpress",
                "action_type": "publish",
                "needs_network": True,
                "needs_auth": True,
                "is_irreversible": True,
                "steps": [
                    {"type": "navigate", "target": "Posts > Add New"},
                    {"type": "type_text", "field": "title", "value": "Test Article"},
                    {"type": "simple_tap", "target": "Publish"},
                ],
            }

            report = _run_sync(hub.forge.pre_flight(mock_phone, mock_task))
            print(f"  Ready: {report['ready']}")
            print(f"  Go/No-Go: {report['go_no_go']}")
            oracle = report.get("oracle", {})
            print(f"  Risk: {oracle.get('risk_level', 'N/A')} ({oracle.get('risk_score', 0):.3f})")
            print(f"  Est. duration: {oracle.get('estimated_duration_seconds', 0):.1f}s")
            for action in oracle.get("preventive_actions", []):
                print(f"    - {action}")

        _run_sync(hub.close())
        print("\nDemo complete.")

    else:
        parser.print_help()
