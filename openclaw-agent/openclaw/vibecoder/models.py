"""VibeCoder data models — enums and dataclasses for autonomous coding missions.

Pattern: openclaw/models.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ─── Enums ────────────────────────────────────────────────────────────────────


class MissionStatus(str, Enum):
    """Lifecycle status of a coding mission."""
    QUEUED = "queued"
    PLANNING = "planning"
    AMPLIFYING = "amplifying"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMMITTING = "committing"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    NEEDS_APPROVAL = "needs_approval"


class MissionScope(str, Enum):
    """Scope classification of a mission."""
    BUGFIX = "bugfix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    DOCS = "docs"
    TEST = "test"
    CONFIG = "config"
    DEPLOY = "deploy"
    NEW_PROJECT = "new_project"
    MULTI_FILE = "multi_file"
    UNKNOWN = "unknown"


class EngineType(str, Enum):
    """Which execution engine handles a step."""
    ALGORITHMIC = "algorithmic"   # Git, shell, templates, file ops ($0)
    API_HAIKU = "api_haiku"       # Classification, commit msgs, scope detection
    API_SONNET = "api_sonnet"     # Single-file edits, focused code gen
    CLI_CLAUDE = "cli_claude"     # Multi-file, refactors, new projects, complex bugs


class StepType(str, Enum):
    """Types of mission steps."""
    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    DELETE_FILE = "delete_file"
    RUN_COMMAND = "run_command"
    RUN_TESTS = "run_tests"
    GIT_OPERATION = "git_operation"
    INSTALL_DEPENDENCY = "install_dependency"
    GENERATE_CODE = "generate_code"
    REFACTOR = "refactor"
    REVIEW = "review"


class StepStatus(str, Enum):
    """Execution status of a mission step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ReviewVerdict(str, Enum):
    """Outcome of code review."""
    APPROVED = "approved"
    NEEDS_CHANGES = "needs_changes"
    REJECTED = "rejected"


class QualityGrade(str, Enum):
    """Quality grade from CodeSentinel scoring."""
    S = "S"   # 95+
    A = "A"   # 85-94
    B = "B"   # 75-84
    C = "C"   # 60-74
    D = "D"   # 45-59
    F = "F"   # <45


class DeployTarget(str, Enum):
    """Where code gets deployed after approval."""
    NONE = "none"
    LOCAL = "local"
    VPS_DOCKER = "vps_docker"
    GITHUB = "github"
    NPM = "npm"
    PYPI = "pypi"


# ─── Scout / Analysis ────────────────────────────────────────────────────────


@dataclass
class ProjectInfo:
    """Result of ProjectScout analysis of a codebase."""
    project_id: str
    root_path: str
    language: str = "python"
    framework: str = ""
    package_manager: str = ""
    has_tests: bool = False
    has_ci: bool = False
    has_git: bool = False
    has_docker: bool = False
    entry_points: list[str] = field(default_factory=list)
    source_dirs: list[str] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    total_files: int = 0
    total_lines: int = 0
    dependencies: list[str] = field(default_factory=list)
    deploy_target: DeployTarget = DeployTarget.NONE
    deploy_config: dict[str, Any] = field(default_factory=dict)
    scanned_at: Optional[datetime] = None


# ─── Mission & Steps ─────────────────────────────────────────────────────────


@dataclass
class MissionStep:
    """A single step in a mission execution plan."""
    step_number: int
    step_type: StepType
    description: str
    target_file: str = ""
    engine: EngineType = EngineType.ALGORITHMIC
    command: str = ""
    code_snippet: str = ""
    status: StepStatus = StepStatus.PENDING
    error_message: str = ""
    output: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    retry_count: int = 0
    max_retries: int = 2
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Mission:
    """A coding mission submitted to the VibeCoder agent."""
    mission_id: str
    project_id: str
    title: str
    description: str
    scope: MissionScope = MissionScope.UNKNOWN
    status: MissionStatus = MissionStatus.QUEUED
    priority: int = 5  # 1=highest, 10=lowest
    auto_deploy: bool = False

    # Pipeline data
    steps: list[MissionStep] = field(default_factory=list)
    project_info: Optional[ProjectInfo] = None
    review: Optional[ReviewResult] = None
    sentinel_score: Optional[SentinelResult] = None

    # Git
    branch_name: str = ""
    commit_hash: str = ""
    pr_url: str = ""

    # Cost tracking
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Timing
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # AMPLIFY stage data
    enrichments: dict[str, Any] = field(default_factory=dict)
    expansions: dict[str, Any] = field(default_factory=dict)
    fortifications: dict[str, Any] = field(default_factory=dict)
    anticipations: dict[str, Any] = field(default_factory=dict)
    optimizations: dict[str, Any] = field(default_factory=dict)
    validations: dict[str, Any] = field(default_factory=dict)


# ─── FORGE Results ────────────────────────────────────────────────────────────


@dataclass
class SentinelResult:
    """Code quality score from CodeSentinel."""
    total_score: float = 0.0
    grade: QualityGrade = QualityGrade.F

    # 6 criteria (100 points total)
    lint_score: float = 0.0         # /20 — no lint errors
    security_score: float = 0.0     # /20 — no secrets, no injection
    test_score: float = 0.0         # /20 — tests exist and pass
    convention_score: float = 0.0   # /15 — follows project conventions
    complexity_score: float = 0.0   # /15 — reasonable cyclomatic complexity
    coverage_score: float = 0.0     # /10 — files touched have test coverage

    issues: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def calculate(self) -> None:
        """Calculate total score and assign grade."""
        self.total_score = (
            self.lint_score + self.security_score + self.test_score
            + self.convention_score + self.complexity_score + self.coverage_score
        )
        if self.total_score >= 95:
            self.grade = QualityGrade.S
        elif self.total_score >= 85:
            self.grade = QualityGrade.A
        elif self.total_score >= 75:
            self.grade = QualityGrade.B
        elif self.total_score >= 60:
            self.grade = QualityGrade.C
        elif self.total_score >= 45:
            self.grade = QualityGrade.D
        else:
            self.grade = QualityGrade.F


@dataclass
class OracleEstimate:
    """Cost and engine estimate from MissionOracle."""
    engine: EngineType
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0
    estimated_minutes: float = 0.0
    reasoning: str = ""
    confidence: float = 0.0  # 0-1


@dataclass
class ReviewResult:
    """Result from the code reviewer."""
    verdict: ReviewVerdict = ReviewVerdict.NEEDS_CHANGES
    score: float = 0.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    files_reviewed: int = 0
    lines_added: int = 0
    lines_removed: int = 0


@dataclass
class CodeChange:
    """A single file change tracked by the system."""
    file_path: str
    change_type: str = "modified"  # created, modified, deleted
    diff: str = ""
    lines_added: int = 0
    lines_removed: int = 0


# ─── AMPLIFY Result ───────────────────────────────────────────────────────────


@dataclass
class AmplifyResult:
    """Result from the code AMPLIFY pipeline."""
    mission: Mission
    stages_completed: int = 0
    quality_score: float = 0.0
    ready: bool = False
    stage_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


# ─── Dashboard ────────────────────────────────────────────────────────────────


@dataclass
class VibeDashboard:
    """Aggregate stats for the VibeCoder dashboard."""
    total_missions: int = 0
    completed_missions: int = 0
    failed_missions: int = 0
    queued_missions: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    avg_duration_seconds: float = 0.0
    missions_by_scope: dict[str, int] = field(default_factory=dict)
    missions_by_status: dict[str, int] = field(default_factory=dict)
    recent_missions: list[dict[str, Any]] = field(default_factory=list)
    registered_projects: int = 0
