"""
Workflow Templates — Mission Type Definitions for the UnifiedOrchestrator
=========================================================================

Defines reusable workflow templates for Nick Creighton's 16-site WordPress
publishing empire. Each template describes a sequence of steps (dispatch,
condition, parallel, delay, notify, log) that the UnifiedOrchestrator can
execute as a "mission."

Nine built-in mission types are provided out of the box:
    CONTENT_PUBLISH, SOCIAL_GROWTH, ACCOUNT_CREATION, APP_EXPLORATION,
    MONETIZATION, SITE_MAINTENANCE, REVENUE_CHECK, DEVICE_MAINTENANCE,
    SUBSTACK_DAILY

Custom templates can be created, exported, and imported via the manager
or the CLI.

Data storage: data/workflow_templates/
    templates.json           — all workflow templates (built-in + custom)
    execution_records.json   — execution history

Usage:
    from src.workflow_templates import get_templates

    mgr = get_templates()
    template = mgr.get_template(MissionType.CONTENT_PUBLISH)
    errors = mgr.validate_params(template, {"site_id": "witchcraft", "title": "Moon Water"})
    rendered = mgr.render_steps(template, {"site_id": "witchcraft", "title": "Moon Water"})

CLI:
    python -m src.workflow_templates list
    python -m src.workflow_templates show --type content_publish
    python -m src.workflow_templates validate --type content_publish --param site_id=witchcraft
    python -m src.workflow_templates create --file custom_template.json
    python -m src.workflow_templates delete --id <template-id>
    python -m src.workflow_templates export --type content_publish --file out.json
    python -m src.workflow_templates import --file custom.json
    python -m src.workflow_templates stats
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("workflow_templates")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "workflow_templates"
TEMPLATES_FILE = DATA_DIR / "templates.json"
EXECUTION_FILE = DATA_DIR / "execution_records.json"

# Ensure data directory exists on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum execution records kept on disk
MAX_EXECUTION_RECORDS = 2000

# Parameter substitution pattern: {param.xxx}
PARAM_RE = re.compile(r"\{param\.([a-zA-Z_][a-zA-Z0-9_.]*)\}")

# Valid site IDs across the empire
VALID_SITE_IDS = (
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _run_sync(coro):
    """Run an async coroutine from synchronous code, handling nested loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# JSON persistence helpers (atomic writes)
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
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        # Atomic replace
        if os.name == "nt":
            os.replace(str(tmp), str(path))
        else:
            tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Enums (str, Enum for JSON serialization)
# ---------------------------------------------------------------------------


class MissionType(str, Enum):
    """Types of automated missions the orchestrator can execute."""
    CONTENT_PUBLISH = "content_publish"
    SOCIAL_GROWTH = "social_growth"
    ACCOUNT_CREATION = "account_creation"
    APP_EXPLORATION = "app_exploration"
    MONETIZATION = "monetization"
    SITE_MAINTENANCE = "site_maintenance"
    REVENUE_CHECK = "revenue_check"
    DEVICE_MAINTENANCE = "device_maintenance"
    SUBSTACK_DAILY = "substack_daily"


class StepType(str, Enum):
    """Types of workflow steps."""
    DISPATCH = "dispatch"       # Invoke a module/method
    CONDITION = "condition"     # Branch based on a runtime value
    PARALLEL = "parallel"       # Execute sub-steps concurrently
    DELAY = "delay"             # Wait before proceeding
    NOTIFY = "notify"           # Send a notification
    LOG = "log"                 # Log a message or metric


class ConditionOperator(str, Enum):
    """Operators for conditional step evaluation."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    EXISTS = "exists"
    TRUTHY = "truthy"


class ExecutionStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class StepDefinition:
    """A single step within a workflow template."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_type: str = StepType.DISPATCH.value
    name: str = ""
    description: str = ""
    module: str = ""
    method: str = ""
    kwargs_template: Dict[str, Any] = field(default_factory=dict)
    condition: Dict[str, Any] = field(default_factory=dict)
    parallel_steps: List[Dict[str, Any]] = field(default_factory=list)
    on_failure: str = "fail"         # "fail", "skip", "retry"
    max_retries: int = 0
    timeout_seconds: int = 300
    delay_seconds: int = 0
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StepDefinition:
        """Deserialize from dictionary."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class WorkflowTemplate:
    """A reusable workflow template describing a mission."""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mission_type: str = MissionType.CONTENT_PUBLISH.value
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    steps: List[Dict[str, Any]] = field(default_factory=list)
    required_params: List[str] = field(default_factory=list)
    optional_params: Dict[str, Any] = field(default_factory=dict)
    estimated_duration_seconds: int = 60
    created_at: str = field(default_factory=_now_iso)
    is_builtin: bool = False
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowTemplate:
        """Deserialize from dictionary."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def get_steps(self) -> List[StepDefinition]:
        """Parse steps list into StepDefinition objects."""
        return [StepDefinition.from_dict(s) for s in self.steps]


@dataclass
class ExecutionRecord:
    """Record of a single workflow execution."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    template_id: str = ""
    mission_type: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = ExecutionStatus.PENDING.value
    steps_completed: int = 0
    steps_total: int = 0
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionRecord:
        """Deserialize from dictionary."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ===========================================================================
# BUILT-IN WORKFLOW TEMPLATES
# ===========================================================================

def _builtin_content_publish() -> WorkflowTemplate:
    """CONTENT_PUBLISH: check_calendar -> run_pipeline -> notify"""
    return WorkflowTemplate(
        template_id="builtin-content-publish",
        mission_type=MissionType.CONTENT_PUBLISH.value,
        name="Content Publish Pipeline",
        description=(
            "Full content publishing workflow: check the editorial calendar for "
            "the next scheduled piece, run the content generation pipeline "
            "(outline, draft, SEO optimize, generate images), publish to "
            "WordPress, and notify via configured channels."
        ),
        version="1.0.0",
        steps=[
            StepDefinition(
                step_id="check_calendar",
                step_type=StepType.DISPATCH.value,
                name="Check Editorial Calendar",
                description="Look up the next scheduled entry for the target site.",
                module="src.content_calendar",
                method="get_next_scheduled",
                kwargs_template={
                    "site_id": "{param.site_id}",
                },
                on_failure="fail",
                timeout_seconds=30,
            ).to_dict(),
            StepDefinition(
                step_id="generate_content",
                step_type=StepType.DISPATCH.value,
                name="Run Content Pipeline",
                description="Generate article outline, draft, and SEO-optimize.",
                module="src.content_generator",
                method="generate_article",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "title": "{param.title}",
                    "keywords": "{param.keywords}",
                    "word_count": "{param.word_count}",
                },
                on_failure="fail",
                max_retries=2,
                timeout_seconds=180,
                depends_on=["check_calendar"],
            ).to_dict(),
            StepDefinition(
                step_id="generate_images",
                step_type=StepType.DISPATCH.value,
                name="Generate Featured Images",
                description="Create branded featured image and social media variants.",
                module="src.content_generator",
                method="generate_images",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "title": "{param.title}",
                    "image_types": "{param.image_types}",
                },
                on_failure="skip",
                max_retries=1,
                timeout_seconds=120,
                depends_on=["generate_content"],
            ).to_dict(),
            StepDefinition(
                step_id="publish_to_wp",
                step_type=StepType.DISPATCH.value,
                name="Publish to WordPress",
                description="Push final article and featured image to the WordPress site.",
                module="src.wordpress_client",
                method="create_post",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "title": "{param.title}",
                    "content": "{param.content}",
                    "status": "{param.publish_status}",
                },
                on_failure="fail",
                timeout_seconds=60,
                depends_on=["generate_images"],
            ).to_dict(),
            StepDefinition(
                step_id="notify_published",
                step_type=StepType.NOTIFY.value,
                name="Notify Completion",
                description="Send notification that article was published.",
                module="src.notification_hub",
                method="send",
                kwargs_template={
                    "channel": "{param.notify_channel}",
                    "message": "Published '{param.title}' to {param.site_id}",
                },
                on_failure="skip",
                timeout_seconds=15,
                depends_on=["publish_to_wp"],
            ).to_dict(),
        ],
        required_params=["site_id", "title"],
        optional_params={
            "keywords": [],
            "word_count": 2000,
            "image_types": ["blog_featured", "pinterest_pin"],
            "publish_status": "publish",
            "notify_channel": "discord",
            "content": "",
        },
        estimated_duration_seconds=300,
        is_builtin=True,
        tags=["content", "wordpress", "publishing"],
    )


def _builtin_social_growth() -> WorkflowTemplate:
    """SOCIAL_GROWTH: check_analytics -> repurpose -> schedule -> engage"""
    return WorkflowTemplate(
        template_id="builtin-social-growth",
        mission_type=MissionType.SOCIAL_GROWTH.value,
        name="Social Growth Campaign",
        description=(
            "Social media growth workflow: analyze recent analytics for top "
            "performing content, repurpose it into platform-specific formats, "
            "schedule posts across configured social channels, and queue "
            "engagement actions (likes, replies, follows)."
        ),
        version="1.0.0",
        steps=[
            StepDefinition(
                step_id="check_analytics",
                step_type=StepType.DISPATCH.value,
                name="Check Social Analytics",
                description="Pull engagement metrics from configured social platforms.",
                module="src.social_publisher",
                method="get_analytics",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "platform": "{param.platform}",
                    "days": "{param.lookback_days}",
                },
                on_failure="skip",
                timeout_seconds=60,
            ).to_dict(),
            StepDefinition(
                step_id="repurpose_content",
                step_type=StepType.DISPATCH.value,
                name="Repurpose Top Content",
                description="Transform top-performing content into platform-specific formats.",
                module="src.content_repurposer",
                method="repurpose",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "source_url": "{param.source_url}",
                    "target_platforms": "{param.target_platforms}",
                },
                on_failure="fail",
                max_retries=2,
                timeout_seconds=120,
                depends_on=["check_analytics"],
            ).to_dict(),
            StepDefinition(
                step_id="schedule_posts",
                step_type=StepType.DISPATCH.value,
                name="Schedule Social Posts",
                description="Schedule repurposed content across all target platforms.",
                module="src.social_publisher",
                method="schedule_batch",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "posts": "{param.posts}",
                    "schedule_times": "{param.schedule_times}",
                },
                on_failure="fail",
                timeout_seconds=90,
                depends_on=["repurpose_content"],
            ).to_dict(),
            StepDefinition(
                step_id="queue_engagement",
                step_type=StepType.DISPATCH.value,
                name="Queue Engagement Actions",
                description="Queue follow, like, and reply actions for organic growth.",
                module="src.social_automation",
                method="queue_engagement",
                kwargs_template={
                    "platform": "{param.platform}",
                    "persona_id": "{param.persona_id}",
                    "actions": "{param.engagement_actions}",
                },
                on_failure="skip",
                timeout_seconds=60,
                depends_on=["schedule_posts"],
            ).to_dict(),
        ],
        required_params=["site_id", "platform"],
        optional_params={
            "lookback_days": 7,
            "source_url": "",
            "target_platforms": ["pinterest", "instagram", "twitter"],
            "posts": [],
            "schedule_times": [],
            "persona_id": "",
            "engagement_actions": ["like", "follow"],
        },
        estimated_duration_seconds=240,
        is_builtin=True,
        tags=["social", "growth", "engagement"],
    )


def _builtin_account_creation() -> WorkflowTemplate:
    """ACCOUNT_CREATION: generate_identity -> select_device -> create -> verify -> configure"""
    return WorkflowTemplate(
        template_id="builtin-account-creation",
        mission_type=MissionType.ACCOUNT_CREATION.value,
        name="Account Creation Pipeline",
        description=(
            "End-to-end account creation workflow: generate a digital identity "
            "(persona), select an available cloud phone device from the pool, "
            "automate the sign-up flow on the target platform, verify the "
            "account via SMS/email, and apply initial configuration (bio, "
            "avatar, settings)."
        ),
        version="1.0.0",
        steps=[
            StepDefinition(
                step_id="generate_identity",
                step_type=StepType.DISPATCH.value,
                name="Generate Digital Identity",
                description="Create a realistic persona for the new account.",
                module="src.identity_manager",
                method="generate_persona",
                kwargs_template={
                    "niche": "{param.niche}",
                    "age_range": "{param.age_range}",
                    "gender": "{param.gender}",
                },
                on_failure="fail",
                timeout_seconds=60,
            ).to_dict(),
            StepDefinition(
                step_id="select_device",
                step_type=StepType.DISPATCH.value,
                name="Select Cloud Phone Device",
                description="Pick an available device from the phone pool for account creation.",
                module="src.device_pool",
                method="acquire_device",
                kwargs_template={
                    "purpose": "account_creation",
                    "platform": "{param.platform}",
                },
                on_failure="fail",
                timeout_seconds=30,
                depends_on=["generate_identity"],
            ).to_dict(),
            StepDefinition(
                step_id="create_account",
                step_type=StepType.DISPATCH.value,
                name="Create Account on Platform",
                description="Automate the sign-up flow on the target platform.",
                module="src.account_factory",
                method="create_account",
                kwargs_template={
                    "platform": "{param.platform}",
                    "persona_id": "{param.persona_id}",
                    "device_id": "{param.device_id}",
                    "proxy": "{param.proxy}",
                },
                on_failure="fail",
                max_retries=2,
                timeout_seconds=300,
                depends_on=["select_device"],
            ).to_dict(),
            StepDefinition(
                step_id="verify_account",
                step_type=StepType.DISPATCH.value,
                name="Verify Account (SMS/Email)",
                description="Complete SMS or email verification for the newly created account.",
                module="src.account_factory",
                method="verify_account",
                kwargs_template={
                    "account_id": "{param.account_id}",
                    "verification_method": "{param.verification_method}",
                },
                on_failure="retry",
                max_retries=3,
                timeout_seconds=120,
                depends_on=["create_account"],
            ).to_dict(),
            StepDefinition(
                step_id="configure_profile",
                step_type=StepType.DISPATCH.value,
                name="Configure Profile",
                description="Set bio, avatar, display name, and initial settings.",
                module="src.account_factory",
                method="configure_profile",
                kwargs_template={
                    "account_id": "{param.account_id}",
                    "persona_id": "{param.persona_id}",
                    "platform": "{param.platform}",
                },
                on_failure="skip",
                timeout_seconds=120,
                depends_on=["verify_account"],
            ).to_dict(),
        ],
        required_params=["platform", "niche"],
        optional_params={
            "age_range": "25-40",
            "gender": "any",
            "persona_id": "",
            "device_id": "",
            "proxy": "",
            "account_id": "",
            "verification_method": "sms",
        },
        estimated_duration_seconds=600,
        is_builtin=True,
        tags=["accounts", "identity", "phone-farm"],
    )


def _builtin_app_exploration() -> WorkflowTemplate:
    """APP_EXPLORATION: select_device -> install -> explore_ui -> record -> extract -> save"""
    return WorkflowTemplate(
        template_id="builtin-app-exploration",
        mission_type=MissionType.APP_EXPLORATION.value,
        name="App Exploration & Learning",
        description=(
            "Autonomous app exploration workflow: select a device, install "
            "the target app, systematically explore the UI to learn its "
            "structure, record screen interactions for replay, extract "
            "actionable element maps via OCR/accessibility, and save "
            "the learned workflow for future automation."
        ),
        version="1.0.0",
        steps=[
            StepDefinition(
                step_id="select_device",
                step_type=StepType.DISPATCH.value,
                name="Select Exploration Device",
                description="Acquire a device from the pool for app exploration.",
                module="src.device_pool",
                method="acquire_device",
                kwargs_template={
                    "purpose": "app_exploration",
                    "platform": "{param.device_platform}",
                },
                on_failure="fail",
                timeout_seconds=30,
            ).to_dict(),
            StepDefinition(
                step_id="install_app",
                step_type=StepType.DISPATCH.value,
                name="Install Target App",
                description="Install the app on the selected device.",
                module="src.app_discovery",
                method="install_app",
                kwargs_template={
                    "device_id": "{param.device_id}",
                    "package_name": "{param.package_name}",
                    "app_name": "{param.app_name}",
                },
                on_failure="fail",
                max_retries=2,
                timeout_seconds=120,
                depends_on=["select_device"],
            ).to_dict(),
            StepDefinition(
                step_id="explore_ui",
                step_type=StepType.DISPATCH.value,
                name="Explore UI Structure",
                description="Systematically navigate the app, cataloguing screens and elements.",
                module="src.app_learner",
                method="explore",
                kwargs_template={
                    "device_id": "{param.device_id}",
                    "package_name": "{param.package_name}",
                    "max_depth": "{param.max_depth}",
                    "max_screens": "{param.max_screens}",
                },
                on_failure="skip",
                timeout_seconds=600,
                depends_on=["install_app"],
            ).to_dict(),
            StepDefinition(
                step_id="record_interactions",
                step_type=StepType.DISPATCH.value,
                name="Record Screen Interactions",
                description="Record user interaction flows for future replay.",
                module="src.workflow_recorder",
                method="record_session",
                kwargs_template={
                    "device_id": "{param.device_id}",
                    "session_name": "{param.session_name}",
                    "duration_seconds": "{param.record_duration}",
                },
                on_failure="skip",
                timeout_seconds=300,
                depends_on=["explore_ui"],
            ).to_dict(),
            StepDefinition(
                step_id="extract_elements",
                step_type=StepType.DISPATCH.value,
                name="Extract Element Maps",
                description="Use OCR and accessibility APIs to build element maps.",
                module="src.ocr_extractor",
                method="extract_elements",
                kwargs_template={
                    "device_id": "{param.device_id}",
                    "package_name": "{param.package_name}",
                },
                on_failure="skip",
                timeout_seconds=120,
                depends_on=["explore_ui"],
            ).to_dict(),
            StepDefinition(
                step_id="save_workflow",
                step_type=StepType.LOG.value,
                name="Save Learned Workflow",
                description="Persist the exploration results and learned workflow patterns.",
                module="src.app_learner",
                method="save_workflow",
                kwargs_template={
                    "package_name": "{param.package_name}",
                    "app_name": "{param.app_name}",
                },
                on_failure="fail",
                timeout_seconds=30,
                depends_on=["record_interactions", "extract_elements"],
            ).to_dict(),
        ],
        required_params=["package_name", "app_name"],
        optional_params={
            "device_platform": "android",
            "device_id": "",
            "max_depth": 5,
            "max_screens": 30,
            "session_name": "",
            "record_duration": 120,
        },
        estimated_duration_seconds=900,
        is_builtin=True,
        tags=["exploration", "learning", "phone-farm", "ui"],
    )


def _builtin_monetization() -> WorkflowTemplate:
    """MONETIZATION: check_affiliate -> optimize_listings -> check_kdp -> check_etsy -> summary"""
    return WorkflowTemplate(
        template_id="builtin-monetization",
        mission_type=MissionType.MONETIZATION.value,
        name="Monetization Review",
        description=(
            "Cross-platform monetization review: check affiliate link health "
            "and conversion rates, optimize product listings, review KDP "
            "book performance, check Etsy POD sales, and compile a summary "
            "with actionable recommendations."
        ),
        version="1.0.0",
        steps=[
            StepDefinition(
                step_id="check_affiliate",
                step_type=StepType.DISPATCH.value,
                name="Check Affiliate Performance",
                description="Pull affiliate link stats and conversion rates across all sites.",
                module="src.affiliate_manager",
                method="check_performance",
                kwargs_template={
                    "site_ids": "{param.site_ids}",
                    "days": "{param.lookback_days}",
                },
                on_failure="skip",
                timeout_seconds=90,
            ).to_dict(),
            StepDefinition(
                step_id="optimize_listings",
                step_type=StepType.DISPATCH.value,
                name="Optimize Product Listings",
                description="Review and optimize affiliate product listings for better CTR.",
                module="src.affiliate_manager",
                method="optimize_listings",
                kwargs_template={
                    "site_ids": "{param.site_ids}",
                    "min_impressions": "{param.min_impressions}",
                },
                on_failure="skip",
                timeout_seconds=120,
                depends_on=["check_affiliate"],
            ).to_dict(),
            StepDefinition(
                step_id="check_kdp",
                step_type=StepType.DISPATCH.value,
                name="Check KDP Performance",
                description="Pull KDP book sales and royalty data.",
                module="src.kdp_publisher",
                method="get_performance",
                kwargs_template={
                    "days": "{param.lookback_days}",
                },
                on_failure="skip",
                timeout_seconds=60,
            ).to_dict(),
            StepDefinition(
                step_id="check_etsy",
                step_type=StepType.DISPATCH.value,
                name="Check Etsy POD Sales",
                description="Pull Etsy print-on-demand sales and listing analytics.",
                module="src.etsy_manager",
                method="get_performance",
                kwargs_template={
                    "days": "{param.lookback_days}",
                },
                on_failure="skip",
                timeout_seconds=60,
            ).to_dict(),
            StepDefinition(
                step_id="summary",
                step_type=StepType.NOTIFY.value,
                name="Compile Monetization Summary",
                description="Aggregate results and send a summary with recommendations.",
                module="src.notification_hub",
                method="send",
                kwargs_template={
                    "channel": "{param.notify_channel}",
                    "message": "Monetization review complete for {param.site_ids}",
                    "data": "{param.summary_data}",
                },
                on_failure="skip",
                timeout_seconds=30,
                depends_on=["optimize_listings", "check_kdp", "check_etsy"],
            ).to_dict(),
        ],
        required_params=["site_ids"],
        optional_params={
            "lookback_days": 30,
            "min_impressions": 100,
            "notify_channel": "discord",
            "summary_data": {},
        },
        estimated_duration_seconds=360,
        is_builtin=True,
        tags=["monetization", "affiliate", "kdp", "etsy", "revenue"],
    )


def _builtin_site_maintenance() -> WorkflowTemplate:
    """SITE_MAINTENANCE: seo_audit -> fix_issues -> check_links -> clear_cache -> health_report"""
    return WorkflowTemplate(
        template_id="builtin-site-maintenance",
        mission_type=MissionType.SITE_MAINTENANCE.value,
        name="Site Maintenance Sweep",
        description=(
            "Comprehensive site maintenance: run an SEO audit, auto-fix "
            "detected issues (broken images, missing alt text, thin meta), "
            "check for broken internal/external links, clear LiteSpeed "
            "cache, and generate a health report."
        ),
        version="1.0.0",
        steps=[
            StepDefinition(
                step_id="seo_audit",
                step_type=StepType.DISPATCH.value,
                name="Run SEO Audit",
                description="Comprehensive SEO audit: meta tags, headings, schema, speed.",
                module="src.seo_auditor",
                method="run_audit",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "depth": "{param.audit_depth}",
                },
                on_failure="fail",
                timeout_seconds=300,
            ).to_dict(),
            StepDefinition(
                step_id="fix_issues",
                step_type=StepType.DISPATCH.value,
                name="Auto-Fix SEO Issues",
                description="Automatically fix detected SEO issues where possible.",
                module="src.seo_auditor",
                method="auto_fix",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "fix_types": "{param.fix_types}",
                    "dry_run": "{param.dry_run}",
                },
                on_failure="skip",
                max_retries=1,
                timeout_seconds=180,
                depends_on=["seo_audit"],
            ).to_dict(),
            StepDefinition(
                step_id="check_links",
                step_type=StepType.DISPATCH.value,
                name="Check Broken Links",
                description="Crawl the site for broken internal and external links.",
                module="src.internal_linker",
                method="check_broken_links",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "max_pages": "{param.max_pages}",
                },
                on_failure="skip",
                timeout_seconds=300,
                depends_on=["seo_audit"],
            ).to_dict(),
            StepDefinition(
                step_id="clear_cache",
                step_type=StepType.DISPATCH.value,
                name="Clear LiteSpeed Cache",
                description="Purge the LiteSpeed cache to serve fresh content.",
                module="src.wordpress_client",
                method="clear_cache",
                kwargs_template={
                    "site_id": "{param.site_id}",
                },
                on_failure="skip",
                timeout_seconds=30,
                depends_on=["fix_issues"],
            ).to_dict(),
            StepDefinition(
                step_id="health_report",
                step_type=StepType.NOTIFY.value,
                name="Generate Health Report",
                description="Compile audit results into a health report and send notification.",
                module="src.notification_hub",
                method="send",
                kwargs_template={
                    "channel": "{param.notify_channel}",
                    "message": "Maintenance complete for {param.site_id}",
                    "data": "{param.report_data}",
                },
                on_failure="skip",
                timeout_seconds=30,
                depends_on=["check_links", "clear_cache"],
            ).to_dict(),
        ],
        required_params=["site_id"],
        optional_params={
            "audit_depth": "full",
            "fix_types": ["meta", "alt_text", "schema"],
            "dry_run": False,
            "max_pages": 500,
            "notify_channel": "discord",
            "report_data": {},
        },
        estimated_duration_seconds=600,
        is_builtin=True,
        tags=["maintenance", "seo", "health", "cache"],
    )


def _builtin_revenue_check() -> WorkflowTemplate:
    """REVENUE_CHECK: aggregate_revenue -> check_affiliate -> check_kdp -> check_etsy -> forecast -> alert"""
    return WorkflowTemplate(
        template_id="builtin-revenue-check",
        mission_type=MissionType.REVENUE_CHECK.value,
        name="Revenue Check & Forecast",
        description=(
            "Cross-platform revenue aggregation: pull revenue data from all "
            "sources (AdSense, affiliate networks, KDP, Etsy), compute "
            "trends and forecasts, and send an alert if targets are missed "
            "or anomalies detected."
        ),
        version="1.0.0",
        steps=[
            StepDefinition(
                step_id="aggregate_revenue",
                step_type=StepType.DISPATCH.value,
                name="Aggregate Revenue Data",
                description="Collect revenue figures across all channels and sites.",
                module="src.revenue_tracker",
                method="aggregate",
                kwargs_template={
                    "period": "{param.period}",
                    "site_ids": "{param.site_ids}",
                },
                on_failure="fail",
                timeout_seconds=120,
            ).to_dict(),
            StepDefinition(
                step_id="check_affiliate",
                step_type=StepType.DISPATCH.value,
                name="Check Affiliate Revenue",
                description="Pull affiliate network earnings (Amazon, Impact, etc.).",
                module="src.affiliate_manager",
                method="get_earnings",
                kwargs_template={
                    "period": "{param.period}",
                },
                on_failure="skip",
                timeout_seconds=60,
                depends_on=["aggregate_revenue"],
            ).to_dict(),
            StepDefinition(
                step_id="check_kdp",
                step_type=StepType.DISPATCH.value,
                name="Check KDP Royalties",
                description="Pull KDP royalty data for all published books.",
                module="src.kdp_publisher",
                method="get_royalties",
                kwargs_template={
                    "period": "{param.period}",
                },
                on_failure="skip",
                timeout_seconds=60,
                depends_on=["aggregate_revenue"],
            ).to_dict(),
            StepDefinition(
                step_id="check_etsy",
                step_type=StepType.DISPATCH.value,
                name="Check Etsy Revenue",
                description="Pull Etsy POD shop revenue and fees.",
                module="src.etsy_manager",
                method="get_revenue",
                kwargs_template={
                    "period": "{param.period}",
                },
                on_failure="skip",
                timeout_seconds=60,
                depends_on=["aggregate_revenue"],
            ).to_dict(),
            StepDefinition(
                step_id="forecast",
                step_type=StepType.DISPATCH.value,
                name="Generate Revenue Forecast",
                description="Compute trend lines and monthly/quarterly revenue forecast.",
                module="src.revenue_tracker",
                method="forecast",
                kwargs_template={
                    "period": "{param.period}",
                    "horizon_days": "{param.forecast_horizon}",
                },
                on_failure="skip",
                timeout_seconds=60,
                depends_on=["check_affiliate", "check_kdp", "check_etsy"],
            ).to_dict(),
            StepDefinition(
                step_id="alert",
                step_type=StepType.NOTIFY.value,
                name="Revenue Alert",
                description="Send revenue summary and alert if targets are missed.",
                module="src.notification_hub",
                method="send",
                kwargs_template={
                    "channel": "{param.notify_channel}",
                    "message": "Revenue check: {param.period}",
                    "priority": "{param.alert_priority}",
                },
                on_failure="skip",
                timeout_seconds=15,
                depends_on=["forecast"],
            ).to_dict(),
        ],
        required_params=["period"],
        optional_params={
            "site_ids": list(VALID_SITE_IDS),
            "forecast_horizon": 30,
            "notify_channel": "discord",
            "alert_priority": "normal",
        },
        estimated_duration_seconds=360,
        is_builtin=True,
        tags=["revenue", "forecast", "affiliate", "kdp", "etsy"],
    )


def _builtin_device_maintenance() -> WorkflowTemplate:
    """DEVICE_MAINTENANCE: health_check -> restart_unhealthy -> clear_storage -> update_apps -> cost_optimize"""
    return WorkflowTemplate(
        template_id="builtin-device-maintenance",
        mission_type=MissionType.DEVICE_MAINTENANCE.value,
        name="Device Pool Maintenance",
        description=(
            "Phone farm device maintenance: run health checks on all cloud "
            "phone devices, restart any that are unhealthy, clear cached "
            "data and old APKs, update installed apps to latest versions, "
            "and optimize costs by hibernating idle devices."
        ),
        version="1.0.0",
        steps=[
            StepDefinition(
                step_id="health_check",
                step_type=StepType.DISPATCH.value,
                name="Device Health Check",
                description="Check battery, connectivity, and responsiveness of all devices.",
                module="src.device_pool",
                method="health_check_all",
                kwargs_template={
                    "device_filter": "{param.device_filter}",
                },
                on_failure="fail",
                timeout_seconds=120,
            ).to_dict(),
            StepDefinition(
                step_id="restart_unhealthy",
                step_type=StepType.DISPATCH.value,
                name="Restart Unhealthy Devices",
                description="Reboot devices that failed the health check.",
                module="src.device_pool",
                method="restart_unhealthy",
                kwargs_template={
                    "max_restart": "{param.max_restart}",
                },
                on_failure="skip",
                max_retries=1,
                timeout_seconds=180,
                depends_on=["health_check"],
            ).to_dict(),
            StepDefinition(
                step_id="clear_storage",
                step_type=StepType.DISPATCH.value,
                name="Clear Device Storage",
                description="Delete cached data, old screenshots, and unused APKs.",
                module="src.phone_controller",
                method="clear_storage_all",
                kwargs_template={
                    "device_filter": "{param.device_filter}",
                    "min_free_mb": "{param.min_free_mb}",
                },
                on_failure="skip",
                timeout_seconds=120,
                depends_on=["restart_unhealthy"],
            ).to_dict(),
            StepDefinition(
                step_id="update_apps",
                step_type=StepType.DISPATCH.value,
                name="Update Installed Apps",
                description="Update key apps to their latest versions on all devices.",
                module="src.app_discovery",
                method="update_all_apps",
                kwargs_template={
                    "device_filter": "{param.device_filter}",
                    "app_list": "{param.app_list}",
                },
                on_failure="skip",
                timeout_seconds=300,
                depends_on=["clear_storage"],
            ).to_dict(),
            StepDefinition(
                step_id="cost_optimize",
                step_type=StepType.DISPATCH.value,
                name="Optimize Device Costs",
                description="Hibernate idle devices to reduce cloud phone costs.",
                module="src.device_pool",
                method="optimize_costs",
                kwargs_template={
                    "idle_threshold_hours": "{param.idle_threshold_hours}",
                },
                on_failure="skip",
                timeout_seconds=60,
                depends_on=["update_apps"],
            ).to_dict(),
        ],
        required_params=[],
        optional_params={
            "device_filter": "all",
            "max_restart": 5,
            "min_free_mb": 500,
            "app_list": [],
            "idle_threshold_hours": 24,
        },
        estimated_duration_seconds=720,
        is_builtin=True,
        tags=["device", "maintenance", "phone-farm", "cost"],
    )


def _builtin_substack_daily() -> WorkflowTemplate:
    """SUBSTACK_DAILY: check_calendar -> write_newsletter -> voice_validate -> publish -> cross_promote -> scrape_analytics"""
    return WorkflowTemplate(
        template_id="builtin-substack-daily",
        mission_type=MissionType.SUBSTACK_DAILY.value,
        name="Substack Daily Newsletter",
        description=(
            "Daily Substack newsletter workflow: check the editorial calendar "
            "for today's topic, generate the newsletter draft using the "
            "appropriate brand voice, validate voice consistency, publish "
            "to Substack, cross-promote across social channels, and scrape "
            "analytics from the published edition."
        ),
        version="1.0.0",
        steps=[
            StepDefinition(
                step_id="check_calendar",
                step_type=StepType.DISPATCH.value,
                name="Check Newsletter Calendar",
                description="Look up today's scheduled newsletter topic and parameters.",
                module="src.content_calendar",
                method="get_next_scheduled",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "content_type": "newsletter",
                },
                on_failure="fail",
                timeout_seconds=30,
            ).to_dict(),
            StepDefinition(
                step_id="write_newsletter",
                step_type=StepType.DISPATCH.value,
                name="Write Newsletter Draft",
                description="Generate the newsletter content matching the brand voice.",
                module="src.content_generator",
                method="generate_newsletter",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "topic": "{param.topic}",
                    "subtitle": "{param.subtitle}",
                    "tone": "{param.tone}",
                    "word_count": "{param.word_count}",
                },
                on_failure="fail",
                max_retries=2,
                timeout_seconds=180,
                depends_on=["check_calendar"],
            ).to_dict(),
            StepDefinition(
                step_id="voice_validate",
                step_type=StepType.DISPATCH.value,
                name="Validate Brand Voice",
                description="Score the draft against the site's brand voice profile.",
                module="src.brand_voice_engine",
                method="validate",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "content": "{param.content}",
                    "min_score": "{param.min_voice_score}",
                },
                on_failure="skip",
                timeout_seconds=60,
                depends_on=["write_newsletter"],
            ).to_dict(),
            StepDefinition(
                step_id="publish",
                step_type=StepType.DISPATCH.value,
                name="Publish to Substack",
                description="Publish the validated newsletter to Substack.",
                module="src.social_publisher",
                method="publish_substack",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "title": "{param.topic}",
                    "content": "{param.content}",
                    "subtitle": "{param.subtitle}",
                },
                on_failure="fail",
                timeout_seconds=60,
                depends_on=["voice_validate"],
            ).to_dict(),
            StepDefinition(
                step_id="cross_promote",
                step_type=StepType.PARALLEL.value,
                name="Cross-Promote on Social",
                description="Share the newsletter across configured social platforms.",
                module="src.social_publisher",
                method="cross_promote",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "platforms": "{param.cross_promote_platforms}",
                    "url": "{param.newsletter_url}",
                },
                parallel_steps=[
                    {
                        "step_id": "promote_twitter",
                        "module": "src.social_publisher",
                        "method": "post_twitter",
                        "kwargs_template": {
                            "site_id": "{param.site_id}",
                            "text": "{param.promo_text}",
                            "url": "{param.newsletter_url}",
                        },
                    },
                    {
                        "step_id": "promote_pinterest",
                        "module": "src.social_publisher",
                        "method": "post_pinterest",
                        "kwargs_template": {
                            "site_id": "{param.site_id}",
                            "description": "{param.promo_text}",
                            "url": "{param.newsletter_url}",
                        },
                    },
                ],
                on_failure="skip",
                timeout_seconds=90,
                depends_on=["publish"],
            ).to_dict(),
            StepDefinition(
                step_id="scrape_analytics",
                step_type=StepType.DISPATCH.value,
                name="Scrape Newsletter Analytics",
                description="Pull open rates, click rates, and subscriber growth.",
                module="src.social_publisher",
                method="get_substack_analytics",
                kwargs_template={
                    "site_id": "{param.site_id}",
                    "edition_url": "{param.newsletter_url}",
                },
                on_failure="skip",
                delay_seconds=300,
                timeout_seconds=60,
                depends_on=["cross_promote"],
            ).to_dict(),
        ],
        required_params=["site_id", "topic"],
        optional_params={
            "subtitle": "",
            "tone": "",
            "word_count": 1200,
            "content": "",
            "min_voice_score": 0.75,
            "cross_promote_platforms": ["twitter", "pinterest"],
            "newsletter_url": "",
            "promo_text": "",
        },
        estimated_duration_seconds=480,
        is_builtin=True,
        tags=["newsletter", "substack", "social", "voice"],
    )


# ---------------------------------------------------------------------------
# Builtin registry — maps MissionType -> builder function
# ---------------------------------------------------------------------------

_BUILTIN_BUILDERS: Dict[str, Any] = {
    MissionType.CONTENT_PUBLISH.value: _builtin_content_publish,
    MissionType.SOCIAL_GROWTH.value: _builtin_social_growth,
    MissionType.ACCOUNT_CREATION.value: _builtin_account_creation,
    MissionType.APP_EXPLORATION.value: _builtin_app_exploration,
    MissionType.MONETIZATION.value: _builtin_monetization,
    MissionType.SITE_MAINTENANCE.value: _builtin_site_maintenance,
    MissionType.REVENUE_CHECK.value: _builtin_revenue_check,
    MissionType.DEVICE_MAINTENANCE.value: _builtin_device_maintenance,
    MissionType.SUBSTACK_DAILY.value: _builtin_substack_daily,
}


# ===========================================================================
# WorkflowTemplateManager
# ===========================================================================


class WorkflowTemplateManager:
    """
    Singleton manager for workflow templates.

    Handles loading, saving, CRUD for custom templates, parameter
    validation, step rendering, and execution record-keeping.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._execution_records: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load templates and execution records from disk, seeding builtins if needed."""
        raw = _load_json(TEMPLATES_FILE, default={"templates": {}})
        if isinstance(raw, dict) and "templates" in raw:
            self._templates = raw["templates"]
        elif isinstance(raw, dict):
            self._templates = raw
        else:
            self._templates = {}

        # Seed built-in templates if missing
        for mission_val, builder in _BUILTIN_BUILDERS.items():
            tmpl = builder()
            if tmpl.template_id not in self._templates:
                self._templates[tmpl.template_id] = tmpl.to_dict()
            else:
                # Always overwrite builtins so they stay up-to-date
                existing = self._templates[tmpl.template_id]
                if existing.get("is_builtin", False):
                    self._templates[tmpl.template_id] = tmpl.to_dict()

        self._save_templates()

        # Load execution records
        raw_exec = _load_json(EXECUTION_FILE, default={"records": []})
        if isinstance(raw_exec, dict) and "records" in raw_exec:
            self._execution_records = raw_exec["records"]
        elif isinstance(raw_exec, list):
            self._execution_records = raw_exec
        else:
            self._execution_records = []

    def _save_templates(self) -> None:
        """Persist templates to disk atomically."""
        _save_json(TEMPLATES_FILE, {"templates": self._templates})

    def _save_executions(self) -> None:
        """Persist execution records to disk atomically, truncating old ones."""
        if len(self._execution_records) > MAX_EXECUTION_RECORDS:
            self._execution_records = self._execution_records[-MAX_EXECUTION_RECORDS:]
        _save_json(EXECUTION_FILE, {"records": self._execution_records})

    # ------------------------------------------------------------------
    # Template CRUD
    # ------------------------------------------------------------------

    def get_template(self, mission_type: MissionType | str) -> Optional[WorkflowTemplate]:
        """
        Get a workflow template by mission type.

        Looks up the built-in template ID for the given mission type.
        Returns None if no matching template is found.
        """
        if isinstance(mission_type, MissionType):
            mt_val = mission_type.value
        else:
            mt_val = str(mission_type)

        # Search by mission_type field
        for tid, tdata in self._templates.items():
            if tdata.get("mission_type") == mt_val:
                return WorkflowTemplate.from_dict(tdata)
        return None

    def get_template_by_id(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a template by its unique template_id."""
        tdata = self._templates.get(template_id)
        if tdata is not None:
            return WorkflowTemplate.from_dict(tdata)
        return None

    def list_templates(
        self,
        mission_type: Optional[str] = None,
        tag: Optional[str] = None,
        builtin_only: bool = False,
        custom_only: bool = False,
    ) -> List[WorkflowTemplate]:
        """
        List templates with optional filters.

        Args:
            mission_type: Filter by mission type value.
            tag: Filter by tag (must be present in the template's tags list).
            builtin_only: Only return built-in templates.
            custom_only: Only return custom (non-builtin) templates.

        Returns:
            List of matching WorkflowTemplate objects.
        """
        results: List[WorkflowTemplate] = []
        for tdata in self._templates.values():
            if mission_type and tdata.get("mission_type") != mission_type:
                continue
            if tag and tag not in tdata.get("tags", []):
                continue
            if builtin_only and not tdata.get("is_builtin", False):
                continue
            if custom_only and tdata.get("is_builtin", False):
                continue
            results.append(WorkflowTemplate.from_dict(tdata))
        return sorted(results, key=lambda t: t.name)

    def create_custom(
        self,
        mission_type: str,
        name: str,
        description: str = "",
        steps: Optional[List[Dict[str, Any]]] = None,
        required_params: Optional[List[str]] = None,
        optional_params: Optional[Dict[str, Any]] = None,
        estimated_duration_seconds: int = 60,
        tags: Optional[List[str]] = None,
    ) -> WorkflowTemplate:
        """
        Create a new custom workflow template.

        Args:
            mission_type: The mission type value (may be a custom string).
            name: Human-readable template name.
            description: Template description.
            steps: List of step definition dicts.
            required_params: Parameter names that must be supplied.
            optional_params: Parameter names with default values.
            estimated_duration_seconds: Expected run time.
            tags: Searchable tags.

        Returns:
            The newly created WorkflowTemplate.

        Raises:
            ValueError: If name is empty.
        """
        if not name.strip():
            raise ValueError("Template name must not be empty.")

        template = WorkflowTemplate(
            template_id=str(uuid.uuid4()),
            mission_type=mission_type,
            name=name.strip(),
            description=description.strip(),
            version="1.0.0",
            steps=steps or [],
            required_params=required_params or [],
            optional_params=optional_params or {},
            estimated_duration_seconds=estimated_duration_seconds,
            created_at=_now_iso(),
            is_builtin=False,
            tags=tags or [],
        )
        self._templates[template.template_id] = template.to_dict()
        self._save_templates()
        logger.info("Created custom template: %s (%s)", template.name, template.template_id)
        return template

    def update_custom(
        self,
        template_id: str,
        **updates: Any,
    ) -> WorkflowTemplate:
        """
        Update fields on a custom template.

        Built-in templates cannot be updated via this method.

        Args:
            template_id: The template to update.
            **updates: Field names and new values to apply.

        Returns:
            The updated WorkflowTemplate.

        Raises:
            KeyError: If template_id is not found.
            ValueError: If attempting to update a built-in template.
        """
        if template_id not in self._templates:
            raise KeyError(f"Template not found: {template_id}")

        tdata = self._templates[template_id]
        if tdata.get("is_builtin", False):
            raise ValueError("Cannot update a built-in template. Create a custom copy instead.")

        # Apply updates to known fields only
        known_fields = {f.name for f in WorkflowTemplate.__dataclass_fields__.values()}
        for key, value in updates.items():
            if key in known_fields and key not in ("template_id", "is_builtin", "created_at"):
                tdata[key] = value

        self._templates[template_id] = tdata
        self._save_templates()
        logger.info("Updated template: %s", template_id)
        return WorkflowTemplate.from_dict(tdata)

    def delete_custom(self, template_id: str) -> bool:
        """
        Delete a custom template.

        Built-in templates cannot be deleted.

        Args:
            template_id: The template to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValueError: If attempting to delete a built-in template.
        """
        if template_id not in self._templates:
            return False

        tdata = self._templates[template_id]
        if tdata.get("is_builtin", False):
            raise ValueError("Cannot delete a built-in template.")

        del self._templates[template_id]
        self._save_templates()
        logger.info("Deleted custom template: %s", template_id)
        return True

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_template(self, template_id: str) -> Dict[str, Any]:
        """
        Export a template as a standalone dictionary suitable for JSON serialization.

        Args:
            template_id: The template to export.

        Returns:
            Dictionary containing the full template data.

        Raises:
            KeyError: If template_id is not found.
        """
        if template_id not in self._templates:
            raise KeyError(f"Template not found: {template_id}")
        return copy.deepcopy(self._templates[template_id])

    def import_template(
        self,
        data: Dict[str, Any],
        overwrite: bool = False,
    ) -> WorkflowTemplate:
        """
        Import a template from a dictionary.

        If a template with the same ID already exists:
            - If overwrite=True and it's not built-in, replace it.
            - If overwrite=False, assign a new template_id.
            - Built-in templates are never overwritten by import.

        Args:
            data: Template data dictionary.
            overwrite: Whether to replace an existing template with the same ID.

        Returns:
            The imported WorkflowTemplate.
        """
        template = WorkflowTemplate.from_dict(data)
        template.is_builtin = False  # Imports are never built-in

        existing = self._templates.get(template.template_id)
        if existing:
            if existing.get("is_builtin", False):
                # Never overwrite builtins; assign new ID
                template.template_id = str(uuid.uuid4())
                logger.info(
                    "Import collides with built-in '%s'; assigned new ID: %s",
                    existing.get("name"),
                    template.template_id,
                )
            elif not overwrite:
                template.template_id = str(uuid.uuid4())
                logger.info(
                    "Template ID already exists; assigned new ID: %s",
                    template.template_id,
                )

        self._templates[template.template_id] = template.to_dict()
        self._save_templates()
        logger.info("Imported template: %s (%s)", template.name, template.template_id)
        return template

    # ------------------------------------------------------------------
    # Rendering & Validation
    # ------------------------------------------------------------------

    def render_steps(
        self,
        template: WorkflowTemplate,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Render a template's steps by substituting ``{param.xxx}`` placeholders
        in ``kwargs_template`` with actual parameter values.

        Also handles nested parameter references inside ``parallel_steps``.

        Args:
            template: The workflow template to render.
            params: Runtime parameters to substitute.

        Returns:
            A list of step dicts with placeholders replaced by actual values.
        """
        merged = dict(template.optional_params)
        merged.update(params)
        rendered: List[Dict[str, Any]] = []

        for step_data in template.steps:
            step = copy.deepcopy(step_data)
            # Render kwargs_template
            if "kwargs_template" in step:
                step["kwargs_template"] = self._substitute_params(
                    step["kwargs_template"], merged
                )
            # Render parallel_steps
            if "parallel_steps" in step:
                for ps in step["parallel_steps"]:
                    if "kwargs_template" in ps:
                        ps["kwargs_template"] = self._substitute_params(
                            ps["kwargs_template"], merged
                        )
            rendered.append(step)

        return rendered

    def _substitute_params(
        self,
        kwargs_template: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Replace ``{param.xxx}`` placeholders in a kwargs dictionary.

        Handles string values (full replacement and inline substitution),
        and recursively processes nested dicts and lists.
        """
        result: Dict[str, Any] = {}
        for key, value in kwargs_template.items():
            result[key] = self._substitute_value(value, params)
        return result

    def _substitute_value(self, value: Any, params: Dict[str, Any]) -> Any:
        """Recursively substitute parameter placeholders in a value."""
        if isinstance(value, str):
            # Check if the entire value is a single placeholder
            match = PARAM_RE.fullmatch(value)
            if match:
                param_name = match.group(1)
                return self._resolve_param(param_name, params)
            # Otherwise do inline substitutions (may be multiple)
            def _replacer(m: re.Match) -> str:
                pname = m.group(1)
                resolved = self._resolve_param(pname, params)
                return str(resolved)
            return PARAM_RE.sub(_replacer, value)
        elif isinstance(value, dict):
            return {k: self._substitute_value(v, params) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_value(item, params) for item in value]
        return value

    @staticmethod
    def _resolve_param(param_name: str, params: Dict[str, Any]) -> Any:
        """
        Resolve a dotted parameter name against the params dict.

        Supports nested access: ``site.domain`` looks up ``params["site"]["domain"]``.
        Returns the placeholder string ``{param.xxx}`` if not found.
        """
        parts = param_name.split(".")
        current: Any = params
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return f"{{param.{param_name}}}"
        return current

    def validate_params(
        self,
        template: WorkflowTemplate,
        params: Dict[str, Any],
    ) -> List[str]:
        """
        Validate that all required parameters are present.

        Args:
            template: The workflow template to validate against.
            params: The parameters to check.

        Returns:
            A list of error messages. Empty list means all valid.
        """
        errors: List[str] = []
        for rp in template.required_params:
            if rp not in params or params[rp] is None:
                errors.append(f"Missing required parameter: '{rp}'")
            elif isinstance(params[rp], str) and not params[rp].strip():
                errors.append(f"Required parameter '{rp}' must not be empty")

        # Validate known parameter types
        if "site_id" in params and params["site_id"]:
            sid = params["site_id"]
            if isinstance(sid, str) and sid not in VALID_SITE_IDS:
                errors.append(
                    f"Unknown site_id '{sid}'. Valid: {', '.join(VALID_SITE_IDS)}"
                )

        return errors

    # ------------------------------------------------------------------
    # Execution Tracking
    # ------------------------------------------------------------------

    def record_execution(
        self,
        template_id: str,
        mission_type: str,
        params: Dict[str, Any],
        status: str = ExecutionStatus.COMPLETED.value,
        steps_completed: int = 0,
        steps_total: int = 0,
        started_at: str = "",
        completed_at: str = "",
        duration_seconds: float = 0.0,
        result: Optional[Dict[str, Any]] = None,
        error: str = "",
    ) -> ExecutionRecord:
        """
        Record a workflow execution for analytics.

        Args:
            template_id: The executed template's ID.
            mission_type: The mission type.
            params: Parameters used for execution.
            status: Final status.
            steps_completed: Number of steps that completed successfully.
            steps_total: Total number of steps in the workflow.
            started_at: ISO timestamp when execution started.
            completed_at: ISO timestamp when execution ended.
            duration_seconds: Total wall-clock time.
            result: Arbitrary result data.
            error: Error message if failed.

        Returns:
            The created ExecutionRecord.
        """
        record = ExecutionRecord(
            record_id=str(uuid.uuid4()),
            template_id=template_id,
            mission_type=mission_type,
            params=params,
            status=status,
            steps_completed=steps_completed,
            steps_total=steps_total,
            started_at=started_at or _now_iso(),
            completed_at=completed_at or _now_iso(),
            duration_seconds=duration_seconds,
            result=result or {},
            error=error,
            created_at=_now_iso(),
        )
        self._execution_records.append(record.to_dict())
        self._save_executions()
        logger.info(
            "Recorded execution: %s [%s] %s (%.1fs)",
            record.record_id[:8],
            mission_type,
            status,
            duration_seconds,
        )
        return record

    def get_execution_history(
        self,
        template_id: Optional[str] = None,
        mission_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[ExecutionRecord]:
        """
        Get execution history with optional filters.

        Args:
            template_id: Filter by template.
            mission_type: Filter by mission type.
            status: Filter by execution status.
            limit: Maximum records to return (newest first).

        Returns:
            List of ExecutionRecord objects, newest first.
        """
        results: List[ExecutionRecord] = []
        for rdata in reversed(self._execution_records):
            if template_id and rdata.get("template_id") != template_id:
                continue
            if mission_type and rdata.get("mission_type") != mission_type:
                continue
            if status and rdata.get("status") != status:
                continue
            results.append(ExecutionRecord.from_dict(rdata))
            if len(results) >= limit:
                break
        return results

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Compute aggregate execution statistics.

        Returns:
            Dictionary with counts by mission_type, status, average duration,
            success rate, and recent activity.
        """
        total = len(self._execution_records)
        builtin_count = sum(
            1 for t in self._templates.values() if t.get("is_builtin", False)
        )
        custom_count = sum(
            1 for t in self._templates.values() if not t.get("is_builtin", False)
        )

        if total == 0:
            return {
                "total_executions": 0,
                "by_mission_type": {},
                "by_status": {},
                "avg_duration_seconds": 0.0,
                "success_rate": 0.0,
                "recent_failures": [],
                "templates_count": len(self._templates),
                "builtin_count": builtin_count,
                "custom_count": custom_count,
            }

        by_type: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        total_duration = 0.0
        success_count = 0
        recent_failures: List[Dict[str, Any]] = []

        for rdata in self._execution_records:
            mt = rdata.get("mission_type", "unknown")
            st = rdata.get("status", "unknown")
            by_type[mt] += 1
            by_status[st] += 1
            total_duration += rdata.get("duration_seconds", 0.0)
            if st == ExecutionStatus.COMPLETED.value:
                success_count += 1
            elif st == ExecutionStatus.FAILED.value:
                recent_failures.append({
                    "record_id": rdata.get("record_id", ""),
                    "mission_type": mt,
                    "error": rdata.get("error", ""),
                    "created_at": rdata.get("created_at", ""),
                })

        # Keep only 10 most recent failures
        recent_failures = recent_failures[-10:]

        return {
            "total_executions": total,
            "by_mission_type": dict(by_type),
            "by_status": dict(by_status),
            "avg_duration_seconds": round(total_duration / total, 2) if total else 0.0,
            "success_rate": round(success_count / total, 4) if total else 0.0,
            "recent_failures": recent_failures,
            "templates_count": len(self._templates),
            "builtin_count": builtin_count,
            "custom_count": custom_count,
        }


# ===========================================================================
# SINGLETON
# ===========================================================================

_template_manager: Optional[WorkflowTemplateManager] = None


def get_templates() -> WorkflowTemplateManager:
    """
    Get the global WorkflowTemplateManager singleton.

    Creates the instance on first call, loading persisted state from disk.
    """
    global _template_manager
    if _template_manager is None:
        _template_manager = WorkflowTemplateManager()
    return _template_manager


# ===========================================================================
# CONVENIENCE FUNCTIONS
# ===========================================================================


def get_template_for_mission(mission_type: MissionType | str) -> Optional[WorkflowTemplate]:
    """Convenience: get the template for a given mission type."""
    return get_templates().get_template(mission_type)


def validate_mission_params(
    mission_type: MissionType | str,
    params: Dict[str, Any],
) -> List[str]:
    """Convenience: validate params for a given mission type."""
    mgr = get_templates()
    tmpl = mgr.get_template(mission_type)
    if tmpl is None:
        return [f"No template found for mission type: {mission_type}"]
    return mgr.validate_params(tmpl, params)


def render_mission_steps(
    mission_type: MissionType | str,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convenience: render steps for a given mission type with params."""
    mgr = get_templates()
    tmpl = mgr.get_template(mission_type)
    if tmpl is None:
        raise ValueError(f"No template found for mission type: {mission_type}")
    return mgr.render_steps(tmpl, params)


def list_mission_types() -> List[str]:
    """Return all available mission type values."""
    return [mt.value for mt in MissionType]


# ===========================================================================
# CLI COMMAND HANDLERS
# ===========================================================================


def _cmd_list(args: argparse.Namespace) -> None:
    """Handle the 'list' command."""
    mgr = get_templates()
    templates = mgr.list_templates(
        mission_type=getattr(args, "type", None),
        tag=getattr(args, "tag", None),
        builtin_only=getattr(args, "builtin", False),
        custom_only=getattr(args, "custom", False),
    )

    if not templates:
        print("No templates found.")
        return

    print(f"\n{'ID':<40} {'Type':<22} {'Name':<35} {'Built-in':<10} {'Steps':<6}")
    print("-" * 115)
    for t in templates:
        print(
            f"{t.template_id:<40} {t.mission_type:<22} {t.name:<35} "
            f"{'Yes' if t.is_builtin else 'No':<10} {len(t.steps):<6}"
        )
    print(f"\nTotal: {len(templates)} template(s)")


def _cmd_show(args: argparse.Namespace) -> None:
    """Handle the 'show' command."""
    mgr = get_templates()

    # Try by mission type first, then by template_id
    mission_type_val = args.type
    tmpl = mgr.get_template(mission_type_val)
    if tmpl is None:
        tmpl = mgr.get_template_by_id(mission_type_val)
    if tmpl is None:
        print(f"Template not found: {mission_type_val}")
        return

    print(f"\nTemplate: {tmpl.name}")
    print(f"  ID:            {tmpl.template_id}")
    print(f"  Mission Type:  {tmpl.mission_type}")
    print(f"  Version:       {tmpl.version}")
    print(f"  Built-in:      {'Yes' if tmpl.is_builtin else 'No'}")
    print(f"  Created:       {tmpl.created_at}")
    print(f"  Est. Duration: {tmpl.estimated_duration_seconds}s")
    print(f"  Tags:          {', '.join(tmpl.tags) or '(none)'}")
    print(f"  Description:   {tmpl.description}")

    print(f"\n  Required Params: {', '.join(tmpl.required_params) or '(none)'}")
    if tmpl.optional_params:
        print("  Optional Params:")
        for k, v in tmpl.optional_params.items():
            print(f"    {k}: {v!r}")

    print(f"\n  Steps ({len(tmpl.steps)}):")
    for i, step in enumerate(tmpl.steps, 1):
        sname = step.get("name", step.get("step_id", "?"))
        stype = step.get("step_type", "?")
        smod = step.get("module", "")
        smethod = step.get("method", "")
        sfail = step.get("on_failure", "fail")
        sdeps = step.get("depends_on", [])
        print(f"    {i}. [{stype}] {sname}")
        if smod or smethod:
            print(f"       Module: {smod}.{smethod}()")
        if sdeps:
            print(f"       Depends on: {', '.join(sdeps)}")
        print(f"       On failure: {sfail}")


def _cmd_validate(args: argparse.Namespace) -> None:
    """Handle the 'validate' command."""
    mgr = get_templates()
    tmpl = mgr.get_template(args.type)
    if tmpl is None:
        tmpl = mgr.get_template_by_id(args.type)
    if tmpl is None:
        print(f"Template not found: {args.type}")
        return

    # Parse --param key=value pairs
    params: Dict[str, Any] = {}
    for pstr in (args.param or []):
        if "=" in pstr:
            k, v = pstr.split("=", 1)
            params[k.strip()] = v.strip()
        else:
            print(f"Warning: ignoring malformed param '{pstr}' (expected key=value)")

    errors = mgr.validate_params(tmpl, params)
    if errors:
        print(f"\nValidation FAILED for '{tmpl.name}':")
        for err in errors:
            print(f"  - {err}")
    else:
        print(f"\nValidation PASSED for '{tmpl.name}'")
        print(f"  Params: {params}")

    # Also show rendered steps
    rendered = mgr.render_steps(tmpl, params)
    print(f"\n  Rendered {len(rendered)} step(s):")
    for i, step in enumerate(rendered, 1):
        sname = step.get("name", step.get("step_id", "?"))
        kwargs = step.get("kwargs_template", {})
        unresolved = [
            v for v in kwargs.values()
            if isinstance(v, str) and "{param." in v
        ]
        status_icon = "?" if unresolved else "OK"
        print(f"    {i}. {sname} [{status_icon}]")
        if unresolved:
            print(f"       Unresolved: {unresolved}")


def _cmd_create(args: argparse.Namespace) -> None:
    """Handle the 'create' command — import a custom template from a JSON file."""
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}")
        return

    mgr = get_templates()

    # If it's a full template dict, import it
    if "template_id" in data or "mission_type" in data:
        tmpl = mgr.import_template(data, overwrite=False)
        print(f"Created template: {tmpl.name} ({tmpl.template_id})")
    else:
        # Minimal creation: requires at least name and mission_type
        name = data.get("name", "")
        mt = data.get("mission_type", "custom")
        if not name:
            print("Error: JSON must contain 'name' field.")
            return
        tmpl = mgr.create_custom(
            mission_type=mt,
            name=name,
            description=data.get("description", ""),
            steps=data.get("steps", []),
            required_params=data.get("required_params", []),
            optional_params=data.get("optional_params", {}),
            estimated_duration_seconds=data.get("estimated_duration_seconds", 60),
            tags=data.get("tags", []),
        )
        print(f"Created custom template: {tmpl.name} ({tmpl.template_id})")


def _cmd_delete(args: argparse.Namespace) -> None:
    """Handle the 'delete' command."""
    mgr = get_templates()

    # Resolve by mission type or template_id
    template_id = args.id
    tmpl = mgr.get_template_by_id(template_id)
    if tmpl is None:
        # Try mission type lookup
        tmpl = mgr.get_template(template_id)
        if tmpl is not None:
            template_id = tmpl.template_id

    if tmpl is None:
        print(f"Template not found: {args.id}")
        return

    if tmpl.is_builtin:
        print(f"Cannot delete built-in template: {tmpl.name}")
        return

    if not getattr(args, "force", False):
        response = input(f"Delete template '{tmpl.name}' ({template_id})? [y/N] ").strip()
        if response.lower() not in ("y", "yes"):
            print("Cancelled.")
            return

    deleted = mgr.delete_custom(template_id)
    if deleted:
        print(f"Deleted template: {template_id}")
    else:
        print(f"Could not delete template: {template_id}")


def _cmd_export(args: argparse.Namespace) -> None:
    """Handle the 'export' command."""
    mgr = get_templates()

    # Resolve by mission type first, then template_id
    tmpl = mgr.get_template(args.type)
    if tmpl is None:
        tmpl = mgr.get_template_by_id(args.type)
    if tmpl is None:
        print(f"Template not found: {args.type}")
        return

    data = mgr.export_template(tmpl.template_id)

    output_file = getattr(args, "file", None)
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        print(f"Exported to: {output_path}")
    else:
        print(json.dumps(data, indent=2, default=str))


def _cmd_import(args: argparse.Namespace) -> None:
    """Handle the 'import' command."""
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}")
        return

    mgr = get_templates()
    overwrite = getattr(args, "overwrite", False)
    tmpl = mgr.import_template(data, overwrite=overwrite)
    print(f"Imported template: {tmpl.name} ({tmpl.template_id})")


def _cmd_stats(args: argparse.Namespace) -> None:
    """Handle the 'stats' command."""
    mgr = get_templates()
    stats = mgr.get_execution_stats()

    print("\n=== Workflow Template Statistics ===")
    print(f"\n  Templates:        {stats['templates_count']}")
    print(f"    Built-in:       {stats['builtin_count']}")
    print(f"    Custom:         {stats['custom_count']}")
    print(f"\n  Total Executions: {stats['total_executions']}")
    print(f"  Success Rate:     {stats['success_rate']:.1%}")
    print(f"  Avg Duration:     {stats['avg_duration_seconds']:.1f}s")

    if stats["by_mission_type"]:
        print("\n  Executions by Mission Type:")
        for mt, count in sorted(stats["by_mission_type"].items()):
            print(f"    {mt:<25} {count}")

    if stats["by_status"]:
        print("\n  Executions by Status:")
        for st, count in sorted(stats["by_status"].items()):
            print(f"    {st:<15} {count}")

    if stats["recent_failures"]:
        print("\n  Recent Failures:")
        for fail in stats["recent_failures"][-5:]:
            print(
                f"    [{fail['created_at'][:19]}] {fail['mission_type']}: "
                f"{fail['error'][:60]}"
            )


# ===========================================================================
# MAIN / CLI ENTRY POINT
# ===========================================================================


def main() -> None:
    """CLI entry point for workflow templates management."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="workflow_templates",
        description="OpenClaw Empire Workflow Templates — mission type definitions for the UnifiedOrchestrator",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    sp_list = subparsers.add_parser("list", help="List all workflow templates")
    sp_list.add_argument("--type", type=str, default=None, help="Filter by mission type")
    sp_list.add_argument("--tag", type=str, default=None, help="Filter by tag")
    sp_list.add_argument("--builtin", action="store_true", help="Show built-in only")
    sp_list.add_argument("--custom", action="store_true", help="Show custom only")
    sp_list.set_defaults(func=_cmd_list)

    # show
    sp_show = subparsers.add_parser("show", help="Show template details")
    sp_show.add_argument("--type", type=str, required=True, help="Mission type or template ID")
    sp_show.set_defaults(func=_cmd_show)

    # validate
    sp_validate = subparsers.add_parser("validate", help="Validate parameters for a template")
    sp_validate.add_argument("--type", type=str, required=True, help="Mission type or template ID")
    sp_validate.add_argument("--param", type=str, nargs="*", help="Parameters as key=value pairs")
    sp_validate.set_defaults(func=_cmd_validate)

    # create
    sp_create = subparsers.add_parser("create", help="Create a custom template from JSON file")
    sp_create.add_argument("--file", type=str, required=True, help="JSON file with template definition")
    sp_create.set_defaults(func=_cmd_create)

    # delete
    sp_delete = subparsers.add_parser("delete", help="Delete a custom template")
    sp_delete.add_argument("--id", type=str, required=True, help="Template ID to delete")
    sp_delete.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    sp_delete.set_defaults(func=_cmd_delete)

    # export
    sp_export = subparsers.add_parser("export", help="Export a template to JSON")
    sp_export.add_argument("--type", type=str, required=True, help="Mission type or template ID")
    sp_export.add_argument("--file", type=str, default=None, help="Output file path")
    sp_export.set_defaults(func=_cmd_export)

    # import
    sp_import = subparsers.add_parser("import", help="Import a template from JSON file")
    sp_import.add_argument("--file", type=str, required=True, help="JSON file to import")
    sp_import.add_argument("--overwrite", action="store_true", help="Overwrite existing template")
    sp_import.set_defaults(func=_cmd_import)

    # stats
    sp_stats = subparsers.add_parser("stats", help="Show execution statistics")
    sp_stats.set_defaults(func=_cmd_stats)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


# ===========================================================================
# MODULE ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()
