"""VibePlannerAgent — decomposes a mission into ordered MissionSteps.

Uses MissionOracle for scope classification + engine routing.
Uses ProjectScout context to inform step ordering.
Haiku API call for ambiguous missions; otherwise pure algorithmic.

Part of the VibeCoder agent system.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from openclaw.vibecoder.models import (
    EngineType,
    Mission,
    MissionScope,
    MissionStep,
    ProjectInfo,
    StepType,
)
from openclaw.vibecoder.forge.mission_oracle import MissionOracle

logger = logging.getLogger(__name__)


# ─── Step generation templates ────────────────────────────────────────────────

_SCOPE_STEPS: dict[MissionScope, list[dict[str, Any]]] = {
    MissionScope.BUGFIX: [
        {"type": StepType.EDIT_FILE, "desc": "Fix the bug in {target}"},
        {"type": StepType.RUN_TESTS, "desc": "Run tests to verify fix"},
        {"type": StepType.GIT_OPERATION, "desc": "Commit bug fix"},
    ],
    MissionScope.FEATURE: [
        {"type": StepType.CREATE_FILE, "desc": "Create feature implementation"},
        {"type": StepType.EDIT_FILE, "desc": "Wire feature into existing code"},
        {"type": StepType.CREATE_FILE, "desc": "Add tests for new feature"},
        {"type": StepType.RUN_TESTS, "desc": "Run tests"},
        {"type": StepType.GIT_OPERATION, "desc": "Commit new feature"},
    ],
    MissionScope.REFACTOR: [
        {"type": StepType.EDIT_FILE, "desc": "Refactor target code"},
        {"type": StepType.EDIT_FILE, "desc": "Update affected imports/references"},
        {"type": StepType.RUN_TESTS, "desc": "Run tests to ensure no regressions"},
        {"type": StepType.GIT_OPERATION, "desc": "Commit refactor"},
    ],
    MissionScope.DOCS: [
        {"type": StepType.EDIT_FILE, "desc": "Update documentation"},
        {"type": StepType.GIT_OPERATION, "desc": "Commit docs update"},
    ],
    MissionScope.TEST: [
        {"type": StepType.CREATE_FILE, "desc": "Create test file"},
        {"type": StepType.RUN_TESTS, "desc": "Run tests to verify"},
        {"type": StepType.GIT_OPERATION, "desc": "Commit tests"},
    ],
    MissionScope.CONFIG: [
        {"type": StepType.EDIT_FILE, "desc": "Update configuration"},
        {"type": StepType.RUN_TESTS, "desc": "Verify config changes"},
        {"type": StepType.GIT_OPERATION, "desc": "Commit config update"},
    ],
    MissionScope.DEPLOY: [
        {"type": StepType.EDIT_FILE, "desc": "Update deploy configuration"},
        {"type": StepType.RUN_COMMAND, "desc": "Build/deploy"},
        {"type": StepType.GIT_OPERATION, "desc": "Commit deploy changes"},
    ],
    MissionScope.NEW_PROJECT: [
        {"type": StepType.RUN_COMMAND, "desc": "Create project directory structure"},
        {"type": StepType.GENERATE_CODE, "desc": "Generate project scaffold"},
        {"type": StepType.CREATE_FILE, "desc": "Create configuration files"},
        {"type": StepType.INSTALL_DEPENDENCY, "desc": "Install dependencies"},
        {"type": StepType.CREATE_FILE, "desc": "Add initial tests"},
        {"type": StepType.RUN_TESTS, "desc": "Run initial tests"},
        {"type": StepType.GIT_OPERATION, "desc": "Initialize git and commit"},
    ],
    MissionScope.MULTI_FILE: [
        {"type": StepType.GENERATE_CODE, "desc": "Generate multi-file changes"},
        {"type": StepType.RUN_TESTS, "desc": "Run tests"},
        {"type": StepType.GIT_OPERATION, "desc": "Commit changes"},
    ],
}


class VibePlannerAgent:
    """Decompose a mission into an ordered list of MissionSteps.

    Usage::

        planner = VibePlannerAgent()
        steps = planner.plan(mission, project_info)
    """

    def __init__(self):
        self.oracle = MissionOracle()

    def plan(
        self,
        mission: Mission,
        project_info: ProjectInfo | None = None,
    ) -> list[MissionStep]:
        """Create an execution plan for the mission."""
        # 1. Classify scope if unknown
        if mission.scope == MissionScope.UNKNOWN:
            mission.scope = self.oracle.classify_scope(mission.title, mission.description)
            logger.info(f"[Planner] Classified scope: {mission.scope.value}")

        # 2. Generate base steps from scope template
        steps = self._generate_base_steps(mission)

        # 3. Enrich with project-specific context
        if project_info:
            steps = self._enrich_with_context(steps, mission, project_info)

        # 4. Insert dependency installation if needed
        steps = self._add_dependency_steps(steps, mission)

        # 5. Route each step to optimal engine
        for step in steps:
            step.engine = self.oracle.route_step(step, project_info)

        # 6. Renumber steps
        for i, step in enumerate(steps):
            step.step_number = i + 1

        logger.info(
            f"[Planner] Generated {len(steps)} steps for mission "
            f"'{mission.title}' (scope={mission.scope.value})"
        )
        return steps

    def _generate_base_steps(self, mission: Mission) -> list[MissionStep]:
        """Generate steps from the scope template."""
        template = _SCOPE_STEPS.get(mission.scope, _SCOPE_STEPS[MissionScope.FEATURE])
        steps = []
        # Extract target once from both title and description
        target = self._extract_target_file(
            f"{mission.title} {mission.description}"
        )
        for i, tmpl in enumerate(template):
            desc = tmpl["desc"]
            desc = desc.replace("{target}", target or "target file")

            steps.append(MissionStep(
                step_number=i + 1,
                step_type=tmpl["type"],
                description=desc,
                target_file=target or "",
            ))
        return steps

    def _enrich_with_context(
        self,
        steps: list[MissionStep],
        mission: Mission,
        project_info: ProjectInfo,
    ) -> list[MissionStep]:
        """Add project-specific context to steps."""
        enriched = []
        for step in steps:
            # Set test runner command based on project
            if step.step_type == StepType.RUN_TESTS:
                step.command = self._get_test_command(project_info)

            # Set install command
            if step.step_type == StepType.INSTALL_DEPENDENCY:
                step.command = self._get_install_command(project_info)

            enriched.append(step)

        return enriched

    def _add_dependency_steps(
        self,
        steps: list[MissionStep],
        mission: Mission,
    ) -> list[MissionStep]:
        """Insert dependency installation if description mentions new packages."""
        desc = mission.description.lower()
        dep_keywords = ["install", "add dependency", "pip install", "npm install", "require"]
        if any(kw in desc for kw in dep_keywords):
            # Check if install step already exists
            has_install = any(s.step_type == StepType.INSTALL_DEPENDENCY for s in steps)
            if not has_install:
                install_step = MissionStep(
                    step_number=0,
                    step_type=StepType.INSTALL_DEPENDENCY,
                    description="Install required dependencies",
                )
                # Insert before first code step
                steps.insert(0, install_step)
        return steps

    @staticmethod
    def _extract_target_file(description: str) -> str | None:
        """Extract a file path from the mission description.

        Preserves directory paths (e.g. 'openclaw/vibecoder/models.py')
        not just filenames ('models.py').
        """
        # Pattern 1: backtick-quoted paths (highest priority, most explicit)
        backtick = re.search(r'`([^`]+\.\w+)`', description)
        if backtick:
            path = backtick.group(1).replace("\\", "/")
            return path

        # Pattern 2: paths with directory separators (next priority)
        dir_path = re.search(
            r'(?:[\w.-]+[/\\])+[\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|rb|java|php|css|html|yml|yaml|json|toml|md)',
            description,
        )
        if dir_path:
            return dir_path.group(0).replace("\\", "/")

        # Pattern 3: bare filenames (last resort)
        bare = re.search(
            r'[\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|rb|java|php|css|html|yml|yaml|json|toml|md)',
            description,
        )
        if bare:
            return bare.group(0)

        return None

    @staticmethod
    def _get_test_command(project_info: ProjectInfo) -> str:
        """Get the appropriate test command for the project."""
        if project_info.language == "python":
            if "pytest" in project_info.dependencies:
                return "python -m pytest tests/ -v --tb=short"
            return "python -m pytest tests/ -v"
        if project_info.language in ("javascript", "typescript"):
            if project_info.package_manager == "bun":
                return "bun test"
            return "npm test"
        if project_info.language == "go":
            return "go test ./..."
        if project_info.language == "rust":
            return "cargo test"
        return "echo 'No test runner configured'"

    @staticmethod
    def _get_install_command(project_info: ProjectInfo) -> str:
        """Get the appropriate install command."""
        if project_info.language == "python":
            return "pip install -r requirements.txt"
        if project_info.language in ("javascript", "typescript"):
            pm = project_info.package_manager or "npm"
            return f"{pm} install"
        if project_info.language == "go":
            return "go mod tidy"
        if project_info.language == "rust":
            return "cargo build"
        return ""
