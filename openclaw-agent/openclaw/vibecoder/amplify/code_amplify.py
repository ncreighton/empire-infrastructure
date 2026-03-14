"""CodeAmplify — 6-stage plan quality pipeline for VibeCoder missions.

Stages:
  1. Enrich   — add context from project scan + existing code
  2. Expand   — identify edge cases, add error handling steps
  3. Fortify  — add security checks, input validation steps
  4. Anticipate — predict failure modes, add rollback steps
  5. Optimize — remove redundant steps, optimize engine routing
  6. Validate — final quality gate, ensure plan is executable

Pattern: openclaw/amplify/amplify_pipeline.py
All logic is algorithmic — zero LLM cost.
"""

from __future__ import annotations

import logging
from typing import Any

from openclaw.vibecoder.models import (
    AmplifyResult,
    EngineType,
    Mission,
    MissionScope,
    MissionStep,
    ProjectInfo,
    StepType,
    StepStatus,
)
from openclaw.vibecoder.forge.mission_oracle import MissionOracle

logger = logging.getLogger(__name__)


class CodeAmplify:
    """6-stage plan quality pipeline.

    Usage::

        amplify = CodeAmplify()
        result = amplify.amplify(mission, project_info)
        if result.ready:
            # Proceed with execution
    """

    def __init__(self):
        self.oracle = MissionOracle()

    def amplify(
        self,
        mission: Mission,
        project_info: ProjectInfo | None = None,
    ) -> AmplifyResult:
        """Run all 6 AMPLIFY stages on the mission plan."""
        result = AmplifyResult(mission=mission)

        stages = [
            ("enrich", self._stage_enrich),
            ("expand", self._stage_expand),
            ("fortify", self._stage_fortify),
            ("anticipate", self._stage_anticipate),
            ("optimize", self._stage_optimize),
            ("validate", self._stage_validate),
        ]

        for name, stage_func in stages:
            try:
                detail = stage_func(mission, project_info)
                result.stage_details[name] = detail

                # Store in mission's AMPLIFY data
                if name == "enrich":
                    mission.enrichments = detail
                elif name == "expand":
                    mission.expansions = detail
                elif name == "fortify":
                    mission.fortifications = detail
                elif name == "anticipate":
                    mission.anticipations = detail
                elif name == "optimize":
                    mission.optimizations = detail
                elif name == "validate":
                    mission.validations = detail

                result.stages_completed += 1
                logger.debug(f"[AMPLIFY] Stage {name} complete")

            except Exception as e:
                result.issues.append(f"Stage {name} failed: {str(e)[:100]}")
                logger.error(f"[AMPLIFY] Stage {name} failed: {e}")

        # Calculate quality score
        result.quality_score = self._calculate_quality(result)
        result.ready = result.quality_score >= 60 and not any(
            "BLOCKER" in issue for issue in result.issues
        )

        logger.info(
            f"[AMPLIFY] Complete: score={result.quality_score:.0f}/100, "
            f"stages={result.stages_completed}/6, "
            f"ready={result.ready}"
        )
        return result

    # ─── Stage implementations ────────────────────────────────────────────

    def _stage_enrich(
        self, mission: Mission, project_info: ProjectInfo | None
    ) -> dict[str, Any]:
        """Stage 1: Enrich — add context from project scan."""
        detail: dict[str, Any] = {"added_context": []}

        if project_info:
            # Add test command to test steps
            for step in mission.steps:
                if step.step_type == StepType.RUN_TESTS and not step.command:
                    step.command = self._get_test_command(project_info)
                    detail["added_context"].append("Set test command")

            # Add install command
            for step in mission.steps:
                if step.step_type == StepType.INSTALL_DEPENDENCY and not step.command:
                    step.command = self._get_install_command(project_info)
                    detail["added_context"].append("Set install command")

            detail["language"] = project_info.language
            detail["framework"] = project_info.framework
            detail["has_tests"] = project_info.has_tests

        return detail

    def _stage_expand(
        self, mission: Mission, project_info: ProjectInfo | None
    ) -> dict[str, Any]:
        """Stage 2: Expand — identify edge cases, add error handling."""
        detail: dict[str, Any] = {"steps_added": 0}

        # Add a git branch step at the beginning if not present
        has_branch = any(
            s.step_type == StepType.GIT_OPERATION and "branch" in s.description.lower()
            for s in mission.steps
        )
        if not has_branch and project_info and project_info.has_git:
            branch_step = MissionStep(
                step_number=0,
                step_type=StepType.GIT_OPERATION,
                description="Create feature branch",
                engine=EngineType.ALGORITHMIC,
            )
            mission.steps.insert(0, branch_step)
            detail["steps_added"] += 1

        # Add test verification after code changes if not present
        has_code_change = any(
            s.step_type in (StepType.CREATE_FILE, StepType.EDIT_FILE, StepType.GENERATE_CODE)
            for s in mission.steps
        )
        has_test_run = any(s.step_type == StepType.RUN_TESTS for s in mission.steps)
        if has_code_change and not has_test_run and project_info and project_info.has_tests:
            test_step = MissionStep(
                step_number=0,
                step_type=StepType.RUN_TESTS,
                description="Run tests to verify changes",
                command=self._get_test_command(project_info) if project_info else "",
                engine=EngineType.ALGORITHMIC,
            )
            # Insert before git commit
            commit_idx = next(
                (i for i, s in enumerate(mission.steps)
                 if s.step_type == StepType.GIT_OPERATION and "commit" in s.description.lower()),
                len(mission.steps),
            )
            mission.steps.insert(commit_idx, test_step)
            detail["steps_added"] += 1

        # Renumber
        for i, step in enumerate(mission.steps):
            step.step_number = i + 1

        return detail

    def _stage_fortify(
        self, mission: Mission, project_info: ProjectInfo | None
    ) -> dict[str, Any]:
        """Stage 3: Fortify — add security considerations."""
        detail: dict[str, Any] = {"security_notes": []}

        # Check if mission involves sensitive files
        sensitive_patterns = [".env", "credentials", "secret", "password", "key"]
        for step in mission.steps:
            desc = f"{step.description} {step.target_file}".lower()
            for pattern in sensitive_patterns:
                if pattern in desc:
                    detail["security_notes"].append(
                        f"Step {step.step_number} touches sensitive content: {pattern}"
                    )

        # Ensure no .env files are committed
        for step in mission.steps:
            if step.step_type == StepType.GIT_OPERATION and "commit" in step.description.lower():
                detail["security_notes"].append(
                    "CodeSentinel will verify no secrets in committed files"
                )
                break

        return detail

    def _stage_anticipate(
        self, mission: Mission, project_info: ProjectInfo | None
    ) -> dict[str, Any]:
        """Stage 4: Anticipate — predict failure modes."""
        detail: dict[str, Any] = {"risks": [], "mitigations": []}

        # Network-dependent steps
        network_steps = [
            s for s in mission.steps
            if s.step_type in (StepType.INSTALL_DEPENDENCY,)
            or "push" in s.description.lower()
            or "deploy" in s.description.lower()
        ]
        if network_steps:
            detail["risks"].append("Network-dependent steps may fail")
            detail["mitigations"].append("Steps have retry logic (max 2)")

        # Complex LLM steps
        llm_steps = [
            s for s in mission.steps
            if s.engine in (EngineType.API_SONNET, EngineType.CLI_CLAUDE)
        ]
        if len(llm_steps) > 5:
            detail["risks"].append(f"High LLM step count ({len(llm_steps)}) — cost risk")
            estimate = self.oracle.estimate_mission(mission)
            detail["estimated_cost"] = estimate.estimated_cost_usd

        # File conflicts
        target_files = [s.target_file for s in mission.steps if s.target_file]
        if len(target_files) != len(set(target_files)):
            detail["risks"].append("Multiple steps target the same file — potential conflicts")

        return detail

    def _stage_optimize(
        self, mission: Mission, project_info: ProjectInfo | None
    ) -> dict[str, Any]:
        """Stage 5: Optimize — re-route engines for cost efficiency."""
        detail: dict[str, Any] = {"optimizations": [], "cost_saved": 0.0}

        for step in mission.steps:
            old_engine = step.engine

            # Downgrade CLI to Sonnet for simple steps
            if step.engine == EngineType.CLI_CLAUDE:
                if step.step_type in (StepType.EDIT_FILE, StepType.CREATE_FILE):
                    if len(step.description) < 100:
                        step.engine = EngineType.API_SONNET
                        detail["optimizations"].append(
                            f"Step {step.step_number}: CLI→Sonnet (simple edit)"
                        )

            # Downgrade Sonnet to Haiku for review/classification
            if step.engine == EngineType.API_SONNET:
                if step.step_type == StepType.REVIEW:
                    step.engine = EngineType.API_HAIKU
                    detail["optimizations"].append(
                        f"Step {step.step_number}: Sonnet→Haiku (review task)"
                    )

        return detail

    def _stage_validate(
        self, mission: Mission, project_info: ProjectInfo | None
    ) -> dict[str, Any]:
        """Stage 6: Validate — final quality gate."""
        detail: dict[str, Any] = {"valid": True, "issues": []}

        # Must have at least one step
        if not mission.steps:
            detail["valid"] = False
            detail["issues"].append("No steps in plan")

        # Must have a git commit step (unless scope is docs-only)
        has_commit = any(
            s.step_type == StepType.GIT_OPERATION and "commit" in s.description.lower()
            for s in mission.steps
        )
        if not has_commit and mission.scope != MissionScope.DOCS:
            detail["issues"].append("No git commit step — changes won't be persisted")

        # All steps should have descriptions
        empty_desc = [s for s in mission.steps if not s.description.strip()]
        if empty_desc:
            detail["issues"].append(f"{len(empty_desc)} step(s) missing descriptions")

        # Check step ordering: git operations should come last
        git_steps = [
            i for i, s in enumerate(mission.steps)
            if s.step_type == StepType.GIT_OPERATION
        ]
        code_steps = [
            i for i, s in enumerate(mission.steps)
            if s.step_type in (StepType.CREATE_FILE, StepType.EDIT_FILE, StepType.GENERATE_CODE)
        ]
        if git_steps and code_steps:
            # Branch step can be first, but commit should be after code
            commit_indices = [
                i for i in git_steps
                if "commit" in mission.steps[i].description.lower()
            ]
            for ci in commit_indices:
                if any(ci < cs for cs in code_steps):
                    detail["issues"].append(
                        "Git commit step is before code generation — reorder needed"
                    )

        if detail["issues"]:
            detail["valid"] = False

        return detail

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _calculate_quality(self, result: AmplifyResult) -> float:
        """Calculate overall plan quality score."""
        score = 0.0

        # Base score from stages completed (60 points)
        score += (result.stages_completed / 6) * 60

        # Bonus for enrichment (10 points)
        if result.stage_details.get("enrich", {}).get("added_context"):
            score += 10

        # Bonus for no security issues (10 points)
        sec_notes = result.stage_details.get("fortify", {}).get("security_notes", [])
        if not sec_notes:
            score += 10
        elif len(sec_notes) <= 2:
            score += 5

        # Bonus for validation passing (20 points)
        if result.stage_details.get("validate", {}).get("valid", False):
            score += 20

        # Penalty for issues
        score -= len(result.issues) * 5

        return max(0.0, min(100.0, score))

    @staticmethod
    def _get_test_command(project_info: ProjectInfo) -> str:
        if project_info.language == "python":
            return "python -m pytest tests/ -v --tb=short"
        if project_info.language in ("javascript", "typescript"):
            return "npm test"
        return ""

    @staticmethod
    def _get_install_command(project_info: ProjectInfo) -> str:
        if project_info.language == "python":
            return "pip install -r requirements.txt"
        if project_info.language in ("javascript", "typescript"):
            return f"{project_info.package_manager or 'npm'} install"
        return ""
