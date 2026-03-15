"""VibeCoderEngine — master orchestrator for autonomous coding missions.

Pipeline: Scout → Plan → AMPLIFY → Execute → Review → Git → Deploy → Persist

Integrates with VPS infrastructure:
  - Docker deployments via SSH
  - n8n workflow deployment
  - EMPIRE-BRAIN learning push
  - Service health verification

Pattern: openclaw/openclaw_engine.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from openclaw.vibecoder.agents.vibe_executor_agent import VibeExecutorAgent
from openclaw.vibecoder.agents.vibe_planner_agent import VibePlannerAgent
from openclaw.vibecoder.agents.vibe_reviewer_agent import VibeReviewerAgent
from openclaw.vibecoder.amplify.code_amplify import CodeAmplify
from openclaw.vibecoder.forge.code_sentinel import CodeSentinel
from openclaw.vibecoder.forge.code_smith import CodeSmith
from openclaw.vibecoder.forge.mission_oracle import MissionOracle
from openclaw.vibecoder.forge.model_router import ModelRouter
from openclaw.vibecoder.forge.project_scout import ProjectScout
from openclaw.vibecoder.forge.vibe_codex import VibeCodex
from openclaw.vibecoder.models import (
    AmplifyResult,
    DeployTarget,
    Mission,
    MissionScope,
    MissionStatus,
    OracleEstimate,
    ProjectInfo,
    ReviewVerdict,
    VibeDashboard,
)

logger = logging.getLogger(__name__)

# VPS connection
_VPS_HOST = os.environ.get("VPS_HOST", "217.216.84.245")
_VPS_USER = os.environ.get("VPS_USER", "empire")
_VPS_BASE = os.environ.get("VPS_BASE", "/opt/empire")


class VibeCoderEngine:
    """Master orchestrator for the VibeCoder coding agent.

    Usage::

        engine = VibeCoderEngine()
        result = await engine.run_mission(
            project_id="openclaw-agent",
            title="Add docstring to models.py",
            description="Add module docstring to models.py",
        )
    """

    def __init__(
        self,
        db_path: str | None = None,
        monthly_budget: float = 100.0,
        notifier: Any = None,
    ):
        # FORGE modules
        self.scout = ProjectScout()
        self.sentinel = CodeSentinel()
        self.oracle = MissionOracle()
        self.smith = CodeSmith()
        self.codex = VibeCodex(db_path=db_path)
        self.model_router = ModelRouter(
            db_path=db_path, monthly_budget=monthly_budget,
        )

        # AMPLIFY
        self.amplify = CodeAmplify()

        # Agents
        self.planner = VibePlannerAgent()
        self.executor = VibeExecutorAgent(model_router=self.model_router)
        self.reviewer = VibeReviewerAgent()

        # Webhook notifier (optional, injected from OpenClawEngine)
        self._notifier = notifier

    # ─── Notifications ───────────────────────────────────────────────────

    def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        """Fire a webhook notification for a mission lifecycle event.

        Resolves the string event_type to the EventType enum and dispatches
        asynchronously (fire-and-forget) if an event loop is running.
        """
        if not self._notifier:
            return
        try:
            from openclaw.automation.webhook_notifier import EventType
            # Map string to enum member
            try:
                event = EventType(event_type)
            except ValueError:
                logger.debug(f"[VibeCoder] Unknown event type: {event_type}")
                return

            # Fire-and-forget async notification
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._notifier.notify(event, data))
            except RuntimeError:
                # No event loop — skip notification (non-critical)
                pass
        except Exception as e:
            logger.debug(f"[VibeCoder] Notification failed (non-critical): {e}")

    # ─── Mission Lifecycle ────────────────────────────────────────────────

    def submit_mission(
        self,
        project_id: str,
        title: str,
        description: str,
        priority: int = 5,
        auto_deploy: bool = False,
    ) -> Mission:
        """Submit a mission to the queue. Returns immediately."""
        mission = Mission(
            mission_id=str(uuid.uuid4())[:12],
            project_id=project_id,
            title=title,
            description=description,
            priority=priority,
            auto_deploy=auto_deploy,
            status=MissionStatus.QUEUED,
            created_at=datetime.now(),
        )
        # Classify scope upfront
        mission.scope = self.oracle.classify_scope(title, description)

        self.codex.create_mission(mission)
        logger.info(
            f"[VibeCoder] Mission queued: {mission.mission_id} "
            f"({mission.scope.value}) — {title}"
        )
        self._notify("mission_queued", {
            "mission_id": mission.mission_id,
            "project_id": project_id,
            "title": title,
            "scope": mission.scope.value,
        })
        return mission

    async def run_mission(
        self,
        project_id: str,
        title: str,
        description: str,
        priority: int = 5,
        auto_deploy: bool = False,
        immediate: bool = True,
    ) -> Mission:
        """Submit and immediately execute a mission."""
        mission = self.submit_mission(
            project_id, title, description, priority, auto_deploy,
        )
        if immediate:
            return await self.execute_mission(mission.mission_id)
        return mission

    async def execute_mission(self, mission_id: str) -> Mission:
        """Execute a queued mission through the full pipeline."""
        row = self.codex.get_mission(mission_id)
        if not row:
            raise ValueError(f"Mission not found: {mission_id}")

        mission = self._row_to_mission(row)
        mission.started_at = datetime.now()

        # Look up project
        project_info = self._get_project_info(mission.project_id)
        mission.project_info = project_info

        try:
            # Step 1: Scout project
            self.codex.update_mission_status(mission_id, MissionStatus.PLANNING)
            logger.info(f"[VibeCoder] Step 1: Scouting project {mission.project_id}...")
            if not project_info:
                root = self._resolve_project_path(mission.project_id)
                if root:
                    project_info = self.scout.analyze(mission.project_id, root)
                    self.codex.register_project(project_info)
                    mission.project_info = project_info

            # Step 2: Plan
            logger.info(f"[VibeCoder] Step 2: Planning steps...")
            mission.steps = self.planner.plan(mission, project_info)

            # Step 3: AMPLIFY
            self.codex.update_mission_status(mission_id, MissionStatus.AMPLIFYING)
            logger.info(f"[VibeCoder] Step 3: Running AMPLIFY pipeline...")
            amplify_result = self.amplify.amplify(mission, project_info)

            if not amplify_result.ready:
                mission.warnings.append(
                    f"AMPLIFY score {amplify_result.quality_score:.0f}/100 — "
                    "proceeding with caution"
                )

            # Step 4: Execute
            self.codex.update_mission_status(mission_id, MissionStatus.EXECUTING)
            logger.info(f"[VibeCoder] Step 4: Executing {len(mission.steps)} steps...")
            mission = await self.executor.execute_mission(mission, project_info)

            # Log each step
            for step in mission.steps:
                self.codex.log_step(mission_id, step)

            # Step 5: Review
            self.codex.update_mission_status(mission_id, MissionStatus.REVIEWING)
            logger.info(f"[VibeCoder] Step 5: Reviewing changes...")
            changes = self.executor.changes
            review = self.reviewer.review(changes, mission, project_info)
            mission.review = review

            # Log code changes
            for change in changes:
                self.codex.log_change(mission_id, change)

            if review.verdict == ReviewVerdict.REJECTED:
                mission.errors.append(f"Review rejected: {review.issues}")
                self._finalize_mission(mission, MissionStatus.FAILED)
                return mission

            if review.verdict == ReviewVerdict.NEEDS_CHANGES:
                # Try auto-fix
                fixes = self.reviewer.auto_fix(changes, project_info)
                if fixes:
                    mission.warnings.append(f"Auto-fixed: {', '.join(fixes)}")

            # Step 6: Sentinel quality gate
            sentinel_result = self.sentinel.score(changes, project_info)
            mission.sentinel_score = sentinel_result
            if sentinel_result.blockers:
                mission.errors.append(f"Sentinel blockers: {sentinel_result.blockers}")
                self._finalize_mission(mission, MissionStatus.FAILED)
                return mission

            # Step 7: Deploy (if auto_deploy and all gates passed)
            if mission.auto_deploy and review.verdict == ReviewVerdict.APPROVED:
                self.codex.update_mission_status(mission_id, MissionStatus.DEPLOYING)
                logger.info(f"[VibeCoder] Step 7: Deploying...")
                await self._deploy(mission, project_info)

            # Step 8: Push learning to EMPIRE-BRAIN
            await self._push_learning(mission)

            # Finalize
            self._finalize_mission(mission, MissionStatus.COMPLETED)

        except Exception as e:
            logger.error(f"[VibeCoder] Mission {mission_id} failed: {e}", exc_info=True)
            mission.errors.append(str(e))
            self._finalize_mission(mission, MissionStatus.FAILED)

        return mission

    # ─── Project Management ───────────────────────────────────────────────

    def register_project(
        self,
        project_id: str,
        root_path: str,
    ) -> ProjectInfo:
        """Scan and register a project."""
        info = self.scout.analyze(project_id, root_path)
        self.codex.register_project(info)
        logger.info(
            f"[VibeCoder] Registered project: {project_id} "
            f"({info.language}/{info.framework}, {info.total_files} files)"
        )
        return info

    def scout_project(self, project_id: str) -> ProjectInfo | None:
        """Re-scan a registered project."""
        row = self.codex.get_project(project_id)
        if not row:
            # Try to auto-discover
            root = self._resolve_project_path(project_id)
            if root:
                return self.register_project(project_id, root)
            return None

        info = self.scout.analyze(project_id, row["root_path"])
        self.codex.register_project(info)
        return info

    def estimate(
        self,
        project_id: str,
        title: str,
        description: str,
    ) -> OracleEstimate:
        """Estimate cost and time for a mission without executing."""
        mission = Mission(
            mission_id="estimate",
            project_id=project_id,
            title=title,
            description=description,
        )
        mission.scope = self.oracle.classify_scope(title, description)
        project_info = self._get_project_info(project_id)
        mission.steps = self.planner.plan(mission, project_info)
        return self.oracle.estimate_mission(mission)

    def get_dashboard(self) -> VibeDashboard:
        """Get aggregate dashboard statistics."""
        return self.codex.get_dashboard()

    # ─── Mission control ──────────────────────────────────────────────────

    def cancel_mission(self, mission_id: str) -> bool:
        """Cancel a queued or paused mission."""
        return self.codex.delete_mission(mission_id)

    def get_mission(self, mission_id: str) -> dict[str, Any] | None:
        """Get mission details including steps."""
        mission = self.codex.get_mission(mission_id)
        if mission:
            mission["steps"] = self.codex.get_steps(mission_id)
            mission["changes"] = self.codex.get_changes(mission_id)
        return mission

    def list_missions(
        self,
        status: str | None = None,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List missions with optional filters."""
        return self.codex.get_missions(status=status, project_id=project_id, limit=limit)

    def list_projects(self) -> list[dict[str, Any]]:
        """List all registered projects."""
        return self.codex.get_all_projects()

    async def retry_mission(self, mission_id: str) -> Mission:
        """Retry a failed mission."""
        self.codex.update_mission_status(mission_id, MissionStatus.QUEUED)
        return await self.execute_mission(mission_id)

    def pause_mission(self, mission_id: str) -> bool:
        """Pause a running or queued mission."""
        row = self.codex.get_mission(mission_id)
        if not row or row["status"] not in ("queued", "executing", "planning", "amplifying"):
            return False
        self.codex.update_mission_status(mission_id, MissionStatus.PAUSED)
        logger.info(f"[VibeCoder] Mission {mission_id} paused")
        return True

    def resume_mission(self, mission_id: str) -> bool:
        """Resume a paused mission back to queued."""
        row = self.codex.get_mission(mission_id)
        if not row or row["status"] != "paused":
            return False
        self.codex.update_mission_status(mission_id, MissionStatus.QUEUED)
        logger.info(f"[VibeCoder] Mission {mission_id} resumed → queued")
        return True

    def approve_mission(self, mission_id: str) -> bool:
        """Approve a mission waiting for approval."""
        row = self.codex.get_mission(mission_id)
        if not row or row["status"] != "needs_approval":
            return False
        self.codex.update_mission_status(mission_id, MissionStatus.QUEUED)
        logger.info(f"[VibeCoder] Mission {mission_id} approved → queued")
        return True

    async def deploy_mission(self, mission_id: str) -> str:
        """Manually deploy a completed mission."""
        row = self.codex.get_mission(mission_id)
        if not row:
            raise ValueError(f"Mission not found: {mission_id}")
        if row["status"] != "completed":
            raise ValueError(f"Cannot deploy mission in '{row['status']}' status")

        project_info = self._get_project_info(row["project_id"])
        if not project_info:
            return "No project info — cannot deploy"

        if project_info.deploy_target == DeployTarget.NONE:
            return "No deploy target configured for this project"

        mission = self._row_to_mission(row)
        self.codex.update_mission_status(mission_id, MissionStatus.DEPLOYING)
        try:
            await self._deploy(mission, project_info)
            self.codex.update_mission_status(mission_id, MissionStatus.COMPLETED)
            return f"Deployed to {project_info.deploy_target.value}"
        except Exception as e:
            self.codex.update_mission_status(mission_id, MissionStatus.COMPLETED)
            return f"Deploy failed: {e}"

    # ─── Deploy ───────────────────────────────────────────────────────────

    async def _deploy(
        self, mission: Mission, project_info: ProjectInfo | None
    ) -> None:
        """Deploy changes based on project deploy target."""
        if not project_info:
            mission.warnings.append("No project info — skipping deploy")
            return

        target = project_info.deploy_target
        if target == DeployTarget.NONE:
            mission.warnings.append("No deploy target configured")
            return

        if target == DeployTarget.VPS_DOCKER:
            result = await self.executor.deploy_to_vps(
                mission.project_id, project_info,
            )
            mission.warnings.append(f"VPS deploy: {result[:200]}")

            # Verify service health
            deploy_config = project_info.deploy_config
            port = deploy_config.get("port")
            if port:
                healthy = await self.executor.verify_vps_service(
                    mission.project_id, int(port),
                )
                if healthy:
                    logger.info(f"[VibeCoder] VPS service healthy on port {port}")
                else:
                    mission.warnings.append(
                        f"VPS service health check failed on port {port}"
                    )

        elif target == DeployTarget.GITHUB:
            # Push to remote
            if project_info.root_path:
                await self.executor._run_shell(
                    f"git push -u origin {mission.branch_name or 'HEAD'}",
                    cwd=project_info.root_path,
                )

    async def _push_learning(self, mission: Mission) -> None:
        """Push mission results to EMPIRE-BRAIN for learning."""
        try:
            import httpx
            brain_url = os.environ.get("EMPIRE_BRAIN_URL", "http://localhost:8200")
            payload = {
                "type": "vibecoder_mission",
                "project_id": mission.project_id,
                "scope": mission.scope.value,
                "status": mission.status.value,
                "duration_seconds": mission.duration_seconds,
                "total_cost_usd": mission.total_cost_usd,
                "total_tokens": mission.total_tokens,
                "steps_count": len(mission.steps),
                "errors": mission.errors[:5],
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{brain_url}/brain/learnings",
                    json=payload,
                )
        except Exception as e:
            logger.debug(f"[VibeCoder] Brain push failed (non-critical): {e}")

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _get_project_info(self, project_id: str) -> ProjectInfo | None:
        """Look up project info from the codex."""
        row = self.codex.get_project(project_id)
        if row:
            return self.codex.project_to_info(row)
        return None

    def _resolve_project_path(self, project_id: str) -> str | None:
        """Try to find the project root path."""
        # Common locations
        candidates = [
            os.path.join("D:/Claude Code Projects", project_id),
            os.path.join("/opt/empire", project_id),
            os.path.join(os.getcwd(), project_id),
            os.getcwd(),  # Current directory might be the project
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return None

    def _finalize_mission(self, mission: Mission, status: MissionStatus) -> None:
        """Finalize a mission with timing and persist."""
        mission.status = status
        mission.completed_at = datetime.now()
        if mission.started_at:
            mission.duration_seconds = (
                mission.completed_at - mission.started_at
            ).total_seconds()

        self.codex.update_mission_status(
            mission.mission_id, status,
            total_tokens=mission.total_tokens,
            total_cost_usd=mission.total_cost_usd,
            duration_seconds=mission.duration_seconds,
            errors=mission.errors,
            warnings=mission.warnings,
            branch_name=mission.branch_name,
            commit_hash=mission.commit_hash,
            pr_url=mission.pr_url,
        )

        logger.info(
            f"[VibeCoder] Mission {mission.mission_id} {status.value}: "
            f"cost=${mission.total_cost_usd:.4f}, "
            f"tokens={mission.total_tokens}, "
            f"duration={mission.duration_seconds:.1f}s"
        )

        event_type = (
            "mission_completed" if status == MissionStatus.COMPLETED
            else "mission_failed" if status == MissionStatus.FAILED
            else f"mission_{status.value}"
        )
        self._notify(event_type, {
            "mission_id": mission.mission_id,
            "project_id": mission.project_id,
            "title": mission.title,
            "status": status.value,
            "scope": mission.scope.value,
            "cost_usd": round(mission.total_cost_usd, 4),
            "tokens": mission.total_tokens,
            "duration_seconds": round(mission.duration_seconds, 1),
            "errors": mission.errors[:3],
            "commit_hash": mission.commit_hash,
            "pr_url": mission.pr_url,
        })

    @staticmethod
    def _row_to_mission(row: dict[str, Any]) -> Mission:
        """Convert a DB row to a Mission dataclass."""
        import json
        return Mission(
            mission_id=row["mission_id"],
            project_id=row["project_id"],
            title=row["title"],
            description=row.get("description", ""),
            scope=MissionScope(row.get("scope", "unknown")),
            status=MissionStatus(row.get("status", "queued")),
            priority=row.get("priority", 5),
            auto_deploy=bool(row.get("auto_deploy", 0)),
            branch_name=row.get("branch_name", ""),
            commit_hash=row.get("commit_hash", ""),
            pr_url=row.get("pr_url", ""),
            total_tokens=row.get("total_tokens", 0),
            total_cost_usd=row.get("total_cost_usd", 0.0),
            errors=json.loads(row.get("errors_json", "[]")),
            warnings=json.loads(row.get("warnings_json", "[]")),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row.get("started_at") else None,
        )
