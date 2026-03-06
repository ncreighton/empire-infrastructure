"""OpenClaw data models — enums and dataclasses for the entire pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ─── Enums ────────────────────────────────────────────────────────────────────


class PlatformCategory(str, Enum):
    """Categories of platforms OpenClaw can target."""
    AI_MARKETPLACE = "ai_marketplace"
    WORKFLOW_MARKETPLACE = "workflow_marketplace"
    DIGITAL_PRODUCT = "digital_product"
    EDUCATION = "education"
    PROMPT_MARKETPLACE = "prompt_marketplace"
    THREE_D_MODELS = "3d_models"
    CODE_REPOSITORY = "code_repository"
    SOCIAL_PLATFORM = "social_platform"


class AccountStatus(str, Enum):
    """Status of an account on a platform."""
    NOT_STARTED = "not_started"
    PLANNED = "planned"
    SIGNUP_IN_PROGRESS = "signup_in_progress"
    SIGNUP_FAILED = "signup_failed"
    EMAIL_VERIFICATION_PENDING = "email_verification_pending"
    PROFILE_INCOMPLETE = "profile_incomplete"
    PROFILE_COMPLETE = "profile_complete"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    WAITLISTED = "waitlisted"


class SignupComplexity(str, Enum):
    """How complex a signup flow is."""
    TRIVIAL = "trivial"         # Email + password only
    SIMPLE = "simple"           # + profile fields
    MODERATE = "moderate"       # + CAPTCHA or email verification
    COMPLEX = "complex"         # + phone verification or manual approval
    MANUAL_ONLY = "manual_only" # Requires human intervention


class CaptchaType(str, Enum):
    """Types of CAPTCHAs encountered."""
    NONE = "none"
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    TURNSTILE = "turnstile"
    FUNCAPTCHA = "funcaptcha"
    IMAGE_CHALLENGE = "image_challenge"
    UNKNOWN = "unknown"


class StepType(str, Enum):
    """Types of steps in a signup plan."""
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL_FIELD = "fill_field"
    FILL_TEXTAREA = "fill_textarea"
    SELECT_DROPDOWN = "select_dropdown"
    UPLOAD_FILE = "upload_file"
    SOLVE_CAPTCHA = "solve_captcha"
    SUBMIT_FORM = "submit_form"
    WAIT_FOR_ELEMENT = "wait_for_element"
    WAIT_FOR_NAVIGATION = "wait_for_navigation"
    VERIFY_EMAIL = "verify_email"
    ACCEPT_TERMS = "accept_terms"
    DISMISS_MODAL = "dismiss_modal"
    OAUTH_LOGIN = "oauth_login"
    SCREENSHOT = "screenshot"
    CUSTOM = "custom"


class StepStatus(str, Enum):
    """Execution status of a single step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_HUMAN = "needs_human"


class QualityGrade(str, Enum):
    """Quality grade from Sentinel scoring."""
    S = "S"   # 95+
    A = "A"   # 85-94
    B = "B"   # 75-84
    C = "C"   # 60-74
    D = "D"   # 45-59
    F = "F"   # <45


class OraclePriority(str, Enum):
    """Priority level from Market Oracle."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SKIP = "skip"


# ─── Platform Config ──────────────────────────────────────────────────────────


@dataclass
class FieldConfig:
    """Configuration for a single form field."""
    name: str
    selector: str = ""
    field_type: str = "text"  # text, textarea, select, checkbox, file
    required: bool = True
    max_length: int = 0
    placeholder: str = ""
    options: list[str] = field(default_factory=list)  # for select fields


@dataclass
class PlatformConfig:
    """Complete configuration for a target platform."""
    platform_id: str
    name: str
    category: PlatformCategory
    signup_url: str
    login_url: str = ""
    profile_url_template: str = ""  # e.g., "https://gumroad.com/{username}"

    # Signup flow
    fields: list[FieldConfig] = field(default_factory=list)
    captcha_type: CaptchaType = CaptchaType.NONE
    requires_email_verification: bool = False
    requires_phone_verification: bool = False
    has_oauth: bool = False
    oauth_providers: list[str] = field(default_factory=list)  # google, github, etc.
    has_waitlist: bool = False

    # Profile limits
    username_max_length: int = 30
    bio_max_length: int = 500
    tagline_max_length: int = 100
    description_max_length: int = 2000
    allows_avatar: bool = True
    allows_banner: bool = False
    allows_links: bool = True
    max_links: int = 5

    # Metadata
    complexity: SignupComplexity = SignupComplexity.SIMPLE
    monetization_potential: int = 5  # 1-10
    audience_size: int = 5           # 1-10
    seo_value: int = 5               # 1-10
    estimated_signup_minutes: int = 5
    known_quirks: list[str] = field(default_factory=list)
    notes: str = ""


# ─── Signup Steps & Plans ─────────────────────────────────────────────────────


@dataclass
class SignupStep:
    """A single step in a signup plan."""
    step_number: int
    step_type: StepType
    description: str
    target: str = ""          # URL, selector, or field name
    value: str = ""           # Value to fill, URL to navigate to
    is_sensitive: bool = False  # True for passwords, emails
    timeout_seconds: int = 30
    retry_count: int = 0
    max_retries: int = 2
    status: StepStatus = StepStatus.PENDING
    error_message: str = ""
    screenshot_path: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class SignupPlan:
    """Complete signup plan for a platform."""
    platform_id: str
    platform_name: str
    steps: list[SignupStep] = field(default_factory=list)
    profile_content: Optional[ProfileContent] = None

    # AMPLIFY stage data
    enrichments: dict[str, Any] = field(default_factory=dict)
    expansions: dict[str, Any] = field(default_factory=dict)
    fortifications: dict[str, Any] = field(default_factory=dict)
    anticipations: dict[str, Any] = field(default_factory=dict)
    optimizations: dict[str, Any] = field(default_factory=dict)
    validations: dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ─── Profile Content ──────────────────────────────────────────────────────────


@dataclass
class ProfileContent:
    """Generated profile content for a platform."""
    platform_id: str
    username: str = ""
    display_name: str = ""
    email: str = ""
    bio: str = ""
    tagline: str = ""
    description: str = ""
    website_url: str = ""
    avatar_path: str = ""
    banner_path: str = ""
    social_links: dict[str, str] = field(default_factory=dict)
    custom_fields: dict[str, str] = field(default_factory=dict)
    seo_keywords: list[str] = field(default_factory=list)
    generated_at: Optional[datetime] = None


# ─── FORGE Results ────────────────────────────────────────────────────────────


@dataclass
class ScoutResult:
    """Result from PlatformScout analysis."""
    platform_id: str
    complexity: SignupComplexity
    estimated_minutes: int
    captcha_type: CaptchaType
    required_fields: list[str] = field(default_factory=list)
    optional_fields: list[str] = field(default_factory=list)
    readiness_checklist: list[dict[str, Any]] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    completeness_score: float = 0.0


@dataclass
class SentinelScore:
    """Profile quality score from ProfileSentinel."""
    platform_id: str
    total_score: float = 0.0
    grade: QualityGrade = QualityGrade.F

    # 6 criteria (100 points total)
    completeness: float = 0.0       # /20 — all fields filled
    seo_quality: float = 0.0        # /20 — keywords, discoverability
    brand_consistency: float = 0.0  # /15 — matches brand identity
    link_presence: float = 0.0      # /15 — website, social links
    bio_quality: float = 0.0        # /15 — engaging, not generic
    avatar_quality: float = 0.0     # /15 — has avatar, correct size

    feedback: list[str] = field(default_factory=list)
    enhancements: list[str] = field(default_factory=list)

    def calculate(self) -> None:
        """Calculate total score and assign grade."""
        self.total_score = (
            self.completeness + self.seo_quality + self.brand_consistency
            + self.link_presence + self.bio_quality + self.avatar_quality
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
class OracleRecommendation:
    """Platform recommendation from MarketOracle."""
    platform_id: str
    platform_name: str
    category: PlatformCategory
    priority: OraclePriority
    score: float = 0.0  # 0-100 composite
    monetization_score: float = 0.0
    audience_score: float = 0.0
    effort_score: float = 0.0  # lower = easier
    seo_score: float = 0.0
    reasoning: str = ""
    recommended_order: int = 0


# ─── AMPLIFY Result ───────────────────────────────────────────────────────────


@dataclass
class AmplifyResult:
    """Result from the AMPLIFY pipeline."""
    plan: SignupPlan
    stages_completed: int = 0
    quality_score: float = 0.0
    ready: bool = False
    stage_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


# ─── Master Result ────────────────────────────────────────────────────────────


@dataclass
class OpenClawResult:
    """Final result from the OpenClaw engine."""
    platform_id: str
    platform_name: str
    success: bool = False
    status: AccountStatus = AccountStatus.NOT_STARTED

    # Pipeline outputs
    scout_result: Optional[ScoutResult] = None
    profile_content: Optional[ProfileContent] = None
    sentinel_score: Optional[SentinelScore] = None
    amplify_result: Optional[AmplifyResult] = None

    # Execution
    steps_completed: int = 0
    steps_total: int = 0
    screenshots: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Account info (populated on success)
    profile_url: str = ""
    username: str = ""


@dataclass
class DashboardStats:
    """Aggregate stats for the dashboard."""
    total_platforms: int = 0
    active_accounts: int = 0
    pending_signups: int = 0
    failed_signups: int = 0
    avg_profile_score: float = 0.0
    platforms_by_category: dict[str, int] = field(default_factory=dict)
    platforms_by_status: dict[str, int] = field(default_factory=dict)
    recent_activity: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CaptchaTask:
    """A CAPTCHA that needs solving."""
    task_id: str
    platform_id: str
    captcha_type: CaptchaType
    site_key: str = ""
    page_url: str = ""
    screenshot_path: str = ""
    solution: str = ""
    status: str = "pending"  # pending, solving, solved, failed
    created_at: Optional[datetime] = None
    solved_at: Optional[datetime] = None
