"""MissionOracle — engine routing + cost estimation.

Decides which execution engine to use for each mission step:
  - Algorithmic ($0)     → git commands, shell, templates, file ops
  - API Haiku ($0.80/M)  → classification, commit messages, scope detection
  - API Sonnet ($3/M)    → single-file edits, focused code gen
  - CLI Claude (turns)   → multi-file, refactors, new projects, complex bugs

All routing logic is rule-based — zero LLM cost.
Part of the VibeCoder FORGE intelligence layer.
"""

from __future__ import annotations

import re
from typing import Any

from openclaw.vibecoder.models import (
    EngineType,
    Mission,
    MissionScope,
    MissionStep,
    OracleEstimate,
    ProjectInfo,
    StepType,
)


# ─── Cost constants ──────────────────────────────────────────────────────────

_COST_PER_MILLION: dict[EngineType, tuple[float, float]] = {
    EngineType.ALGORITHMIC: (0.0, 0.0),
    EngineType.API_HAIKU: (0.80, 4.00),
    EngineType.API_SONNET: (3.00, 15.00),
    EngineType.CLI_CLAUDE: (3.00, 15.00),  # Sonnet default for CLI
}

# Average token usage per step type
_AVG_TOKENS: dict[StepType, dict[EngineType, int]] = {
    StepType.CREATE_FILE: {
        EngineType.API_SONNET: 3000,
        EngineType.CLI_CLAUDE: 5000,
    },
    StepType.EDIT_FILE: {
        EngineType.API_SONNET: 2000,
        EngineType.CLI_CLAUDE: 3000,
    },
    StepType.GENERATE_CODE: {
        EngineType.API_SONNET: 4000,
        EngineType.CLI_CLAUDE: 8000,
    },
    StepType.REFACTOR: {
        EngineType.CLI_CLAUDE: 10000,
    },
    StepType.REVIEW: {
        EngineType.API_HAIKU: 1000,
        EngineType.API_SONNET: 2000,
    },
    StepType.RUN_COMMAND: {
        EngineType.ALGORITHMIC: 0,
    },
    StepType.RUN_TESTS: {
        EngineType.ALGORITHMIC: 0,
    },
    StepType.GIT_OPERATION: {
        EngineType.ALGORITHMIC: 0,
    },
    StepType.DELETE_FILE: {
        EngineType.ALGORITHMIC: 0,
    },
    StepType.INSTALL_DEPENDENCY: {
        EngineType.ALGORITHMIC: 0,
    },
}

# Scope → typical complexity
_SCOPE_COMPLEXITY: dict[MissionScope, str] = {
    MissionScope.BUGFIX: "low",
    MissionScope.FEATURE: "medium",
    MissionScope.REFACTOR: "high",
    MissionScope.DOCS: "low",
    MissionScope.TEST: "low",
    MissionScope.CONFIG: "low",
    MissionScope.DEPLOY: "low",
    MissionScope.NEW_PROJECT: "high",
    MissionScope.MULTI_FILE: "high",
    MissionScope.UNKNOWN: "medium",
}


class MissionOracle:
    """Route missions to the optimal execution engine and estimate costs.

    Usage::

        oracle = MissionOracle()
        engine = oracle.route_step(step, project_info)
        estimate = oracle.estimate_mission(mission)
    """

    def classify_scope(self, title: str, description: str) -> MissionScope:
        """Classify mission scope from title + description (rule-based)."""
        text = f"{title} {description}".lower()

        if any(w in text for w in ["fix", "bug", "error", "broken", "crash", "issue"]):
            return MissionScope.BUGFIX
        if any(w in text for w in ["refactor", "restructure", "reorganize", "cleanup", "clean up"]):
            return MissionScope.REFACTOR
        if any(w in text for w in ["test", "spec", "coverage", "pytest", "jest", "unittest"]):
            return MissionScope.TEST
        if any(w in text for w in ["readme", "comment", "docstring", "documentation"]) or \
           (re.search(r'\bdocs?\b', text) and "docker" not in text):
            return MissionScope.DOCS
        if any(w in text for w in ["deploy", "docker", "ci/cd", "pipeline", "release"]):
            return MissionScope.DEPLOY
        if any(w in text for w in ["config", "setting", "env", "environment", "configure"]):
            return MissionScope.CONFIG
        if any(w in text for w in ["new project", "scaffold", "bootstrap", "init", "create project"]):
            return MissionScope.NEW_PROJECT
        if any(w in text for w in ["multiple files", "multi-file", "across", "several files"]):
            return MissionScope.MULTI_FILE
        if any(w in text for w in ["add", "feature", "implement", "create", "build", "new"]):
            return MissionScope.FEATURE
        return MissionScope.UNKNOWN

    def route_step(self, step: MissionStep, project_info: ProjectInfo | None = None) -> EngineType:
        """Decide which engine handles a step."""
        # Always algorithmic
        if step.step_type in (
            StepType.RUN_COMMAND, StepType.RUN_TESTS,
            StepType.GIT_OPERATION, StepType.DELETE_FILE,
            StepType.INSTALL_DEPENDENCY,
        ):
            return EngineType.ALGORITHMIC

        # File creation: API Sonnet for single file, CLI for scaffolding
        if step.step_type == StepType.CREATE_FILE:
            return EngineType.API_SONNET

        # Edits: API Sonnet for focused single-file
        if step.step_type == StepType.EDIT_FILE:
            return EngineType.API_SONNET

        # Full code generation
        if step.step_type == StepType.GENERATE_CODE:
            if self._is_complex_generation(step):
                return EngineType.CLI_CLAUDE
            return EngineType.API_SONNET

        # Refactoring is always complex
        if step.step_type == StepType.REFACTOR:
            return EngineType.CLI_CLAUDE

        # Review: Haiku for simple, Sonnet for complex
        if step.step_type == StepType.REVIEW:
            return EngineType.API_HAIKU

        return EngineType.API_SONNET

    def route_mission(self, mission: Mission) -> EngineType:
        """Decide the primary engine for an entire mission."""
        scope = mission.scope

        # New projects and multi-file work → CLI
        if scope in (MissionScope.NEW_PROJECT, MissionScope.MULTI_FILE, MissionScope.REFACTOR):
            return EngineType.CLI_CLAUDE

        # Simple scopes → API
        if scope in (MissionScope.DOCS, MissionScope.CONFIG, MissionScope.TEST):
            return EngineType.API_SONNET

        # Bugfix: depends on description complexity
        if scope == MissionScope.BUGFIX:
            if len(mission.description) > 500 or "multiple" in mission.description.lower():
                return EngineType.CLI_CLAUDE
            return EngineType.API_SONNET

        # Feature: medium complexity, could go either way
        if scope == MissionScope.FEATURE:
            if len(mission.description) > 300:
                return EngineType.CLI_CLAUDE
            return EngineType.API_SONNET

        return EngineType.API_SONNET

    def estimate_mission(self, mission: Mission) -> OracleEstimate:
        """Estimate cost and time for a mission."""
        primary_engine = self.route_mission(mission)
        total_tokens = 0
        total_cost = 0.0

        if mission.steps:
            for step in mission.steps:
                engine = self.route_step(step)
                tokens = self._estimate_step_tokens(step, engine)
                cost = self._tokens_to_cost(tokens, engine)
                total_tokens += tokens
                total_cost += cost
        else:
            # Estimate from scope
            total_tokens, total_cost = self._estimate_from_scope(mission.scope, primary_engine)

        complexity = _SCOPE_COMPLEXITY.get(mission.scope, "medium")
        minutes = {"low": 2, "medium": 5, "high": 15}.get(complexity, 5)

        confidence = 0.7
        if mission.steps:
            confidence = 0.85
        if mission.scope == MissionScope.UNKNOWN:
            confidence = 0.4

        return OracleEstimate(
            engine=primary_engine,
            estimated_tokens=total_tokens,
            estimated_cost_usd=round(total_cost, 4),
            estimated_minutes=minutes,
            reasoning=self._build_reasoning(mission, primary_engine, complexity),
            confidence=confidence,
        )

    def estimate_step(self, step: MissionStep) -> OracleEstimate:
        """Estimate cost for a single step."""
        engine = self.route_step(step)
        tokens = self._estimate_step_tokens(step, engine)
        cost = self._tokens_to_cost(tokens, engine)

        return OracleEstimate(
            engine=engine,
            estimated_tokens=tokens,
            estimated_cost_usd=round(cost, 4),
            estimated_minutes=1 if engine != EngineType.CLI_CLAUDE else 5,
            reasoning=f"{step.step_type.value} → {engine.value}",
            confidence=0.75,
        )

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _is_complex_generation(self, step: MissionStep) -> bool:
        """Determine if code generation is complex enough for CLI."""
        desc = step.description.lower()
        if any(w in desc for w in ["multi-file", "scaffold", "full module", "entire"]):
            return True
        if len(step.description) > 200:
            return True
        return False

    def _estimate_step_tokens(self, step: MissionStep, engine: EngineType) -> int:
        """Estimate token usage for a step."""
        type_tokens = _AVG_TOKENS.get(step.step_type, {})
        if engine in type_tokens:
            return type_tokens[engine]
        # Fallback
        if engine == EngineType.ALGORITHMIC:
            return 0
        if engine == EngineType.API_HAIKU:
            return 500
        if engine == EngineType.API_SONNET:
            return 2000
        return 5000  # CLI_CLAUDE

    def _estimate_from_scope(
        self, scope: MissionScope, engine: EngineType
    ) -> tuple[int, float]:
        """Estimate tokens + cost from scope alone."""
        base_tokens = {
            MissionScope.BUGFIX: 3000,
            MissionScope.FEATURE: 5000,
            MissionScope.REFACTOR: 10000,
            MissionScope.DOCS: 1500,
            MissionScope.TEST: 3000,
            MissionScope.CONFIG: 1000,
            MissionScope.DEPLOY: 2000,
            MissionScope.NEW_PROJECT: 15000,
            MissionScope.MULTI_FILE: 10000,
            MissionScope.UNKNOWN: 5000,
        }
        tokens = base_tokens.get(scope, 5000)
        cost = self._tokens_to_cost(tokens, engine)
        return tokens, cost

    def _tokens_to_cost(self, tokens: int, engine: EngineType) -> float:
        """Convert token count to USD cost estimate."""
        if engine == EngineType.ALGORITHMIC:
            return 0.0
        input_rate, output_rate = _COST_PER_MILLION.get(engine, (3.0, 15.0))
        # Assume 60% input, 40% output
        input_tokens = int(tokens * 0.6)
        output_tokens = int(tokens * 0.4)
        cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
        return cost

    def _build_reasoning(
        self, mission: Mission, engine: EngineType, complexity: str
    ) -> str:
        """Build human-readable reasoning for the routing decision."""
        parts = [
            f"Scope: {mission.scope.value} ({complexity} complexity)",
            f"Engine: {engine.value}",
        ]
        if mission.steps:
            parts.append(f"Steps: {len(mission.steps)}")
        if engine == EngineType.CLI_CLAUDE:
            parts.append("Reason: Complex/multi-file work requires full agent")
        elif engine == EngineType.API_SONNET:
            parts.append("Reason: Focused task suitable for single API call")
        elif engine == EngineType.API_HAIKU:
            parts.append("Reason: Simple classification/review task")
        return " | ".join(parts)
