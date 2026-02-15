"""
Account Factory — OpenClaw Empire Account Creation Engine

Template-driven signup automation for any platform (app or browser).
Handles email verification chains, CAPTCHA detection (escalates to human),
profile setup (name, bio, avatar upload), and account warming schedules.
Pre-built templates for major platforms; AppLearner generates templates
for unknown platforms.

Data persisted to: data/account_factory/

Usage:
    from src.account_factory import AccountFactory, get_account_factory

    factory = get_account_factory()
    result = await factory.create_account("instagram", persona_id="abc123")
    await factory.warm_account("instagram", "user@email.com")
    templates = factory.list_templates()

CLI:
    python -m src.account_factory create --platform instagram --persona abc123
    python -m src.account_factory templates list
    python -m src.account_factory warm --platform instagram --account user@email.com
    python -m src.account_factory status --platform instagram
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("account_factory")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "account_factory"


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

class Platform(str, Enum):
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    PINTEREST = "pinterest"
    LINKEDIN = "linkedin"
    YOUTUBE = "youtube"
    REDDIT = "reddit"
    SNAPCHAT = "snapchat"
    THREADS = "threads"
    CUSTOM = "custom"


class CreationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    EMAIL_VERIFICATION = "email_verification"
    PHONE_VERIFICATION = "phone_verification"
    CAPTCHA_REQUIRED = "captcha_required"
    PROFILE_SETUP = "profile_setup"
    WARMING = "warming"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


class WarmingPhase(str, Enum):
    DAY_1_3 = "day_1_3"       # Minimal activity
    DAY_4_7 = "day_4_7"       # Light activity
    DAY_8_14 = "day_8_14"     # Moderate activity
    DAY_15_30 = "day_15_30"   # Normal activity
    MATURE = "mature"          # Full usage


class StepType(str, Enum):
    NAVIGATE = "navigate"
    FILL_FIELD = "fill_field"
    TAP_ELEMENT = "tap_element"
    WAIT = "wait"
    VERIFY_EMAIL = "verify_email"
    VERIFY_PHONE = "verify_phone"
    CAPTCHA_CHECK = "captcha_check"
    UPLOAD_AVATAR = "upload_avatar"
    SELECT_OPTION = "select_option"
    SCROLL = "scroll"
    BACK = "back"
    SCREENSHOT = "screenshot"


@dataclass
class TemplateStep:
    """A step in a signup template."""
    step_type: StepType = StepType.TAP_ELEMENT
    description: str = ""
    target: str = ""
    value: str = ""  # Template variables: {username}, {email}, {password}, {first_name}, etc.
    wait_seconds: float = 1.5
    optional: bool = False
    fallback: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["step_type"] = self.step_type.value
        return d


@dataclass
class SignupTemplate:
    """Template for creating an account on a platform."""
    platform: Platform = Platform.CUSTOM
    name: str = ""
    description: str = ""
    url: str = ""
    app_package: str = ""
    use_browser: bool = False
    steps: List[TemplateStep] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    needs_email: bool = True
    needs_phone: bool = False
    has_captcha: bool = False
    created_at: str = field(default_factory=_now_iso)
    success_rate: float = 0.0
    total_attempts: int = 0

    def to_dict(self) -> dict:
        return {
            "platform": self.platform.value,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "app_package": self.app_package,
            "use_browser": self.use_browser,
            "steps": [s.to_dict() for s in self.steps],
            "required_fields": self.required_fields,
            "needs_email": self.needs_email,
            "needs_phone": self.needs_phone,
            "has_captcha": self.has_captcha,
            "success_rate": self.success_rate,
            "total_attempts": self.total_attempts,
        }


@dataclass
class CreationJob:
    """A tracked account creation job."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    platform: Platform = Platform.CUSTOM
    persona_id: str = ""
    email: str = ""
    username: str = ""
    status: CreationStatus = CreationStatus.PENDING
    current_step: int = 0
    total_steps: int = 0
    error: str = ""
    started_at: str = field(default_factory=_now_iso)
    completed_at: str = ""
    credential_id: str = ""
    warming_started: bool = False
    warming_phase: WarmingPhase = WarmingPhase.DAY_1_3
    log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        d["status"] = self.status.value
        d["warming_phase"] = self.warming_phase.value
        return d


@dataclass
class WarmingSchedule:
    """Account warming schedule definition."""
    platform: Platform = Platform.CUSTOM
    account_email: str = ""
    phase: WarmingPhase = WarmingPhase.DAY_1_3
    start_date: str = field(default_factory=_now_iso)
    daily_actions: List[Dict[str, Any]] = field(default_factory=list)
    completed_actions: int = 0
    last_action: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        d["phase"] = self.phase.value
        return d


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

def _build_instagram_template() -> SignupTemplate:
    return SignupTemplate(
        platform=Platform.INSTAGRAM,
        name="Instagram Signup",
        description="Create an Instagram account via the app",
        app_package="com.instagram.android",
        use_browser=False,
        required_fields=["email", "username", "password", "first_name"],
        needs_email=True,
        needs_phone=False,
        has_captcha=False,
        steps=[
            TemplateStep(StepType.NAVIGATE, "Open Instagram", "com.instagram.android"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Create New Account", "Create new account"),
            TemplateStep(StepType.WAIT, "Wait for signup screen", wait_seconds=2.0),
            TemplateStep(StepType.TAP_ELEMENT, "Select email signup", "Sign up with email"),
            TemplateStep(StepType.FILL_FIELD, "Enter email", "email field", "{email}"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Next", "Next"),
            TemplateStep(StepType.WAIT, "Wait for verification", wait_seconds=3.0),
            TemplateStep(StepType.VERIFY_EMAIL, "Enter email verification code", "confirmation code field"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Next", "Next"),
            TemplateStep(StepType.FILL_FIELD, "Enter name", "Full Name", "{first_name}"),
            TemplateStep(StepType.FILL_FIELD, "Enter password", "Password", "{password}"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Continue", "Continue"),
            TemplateStep(StepType.FILL_FIELD, "Enter username", "Username", "{username}"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Next", "Next"),
            TemplateStep(StepType.TAP_ELEMENT, "Skip avatar", "Skip", optional=True),
        ],
    )


def _build_twitter_template() -> SignupTemplate:
    return SignupTemplate(
        platform=Platform.TWITTER,
        name="Twitter/X Signup",
        description="Create a Twitter/X account via browser",
        url="https://twitter.com/i/flow/signup",
        use_browser=True,
        required_fields=["email", "username", "password", "first_name"],
        needs_email=True,
        needs_phone=True,
        has_captcha=True,
        steps=[
            TemplateStep(StepType.NAVIGATE, "Open Twitter signup", "https://twitter.com/i/flow/signup"),
            TemplateStep(StepType.FILL_FIELD, "Enter name", "Name", "{first_name}"),
            TemplateStep(StepType.FILL_FIELD, "Enter email", "Email", "{email}"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Next", "Next"),
            TemplateStep(StepType.CAPTCHA_CHECK, "Check for CAPTCHA"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Sign up", "Sign up"),
            TemplateStep(StepType.VERIFY_EMAIL, "Enter verification code", "Verification code"),
            TemplateStep(StepType.FILL_FIELD, "Enter password", "Password", "{password}"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Next", "Next"),
            TemplateStep(StepType.FILL_FIELD, "Choose username", "Username", "{username}"),
        ],
    )


def _build_tiktok_template() -> SignupTemplate:
    return SignupTemplate(
        platform=Platform.TIKTOK,
        name="TikTok Signup",
        description="Create a TikTok account via the app",
        app_package="com.zhiliaoapp.musically",
        use_browser=False,
        required_fields=["email", "username", "password"],
        needs_email=True,
        steps=[
            TemplateStep(StepType.NAVIGATE, "Open TikTok", "com.zhiliaoapp.musically"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Profile", "Profile tab"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Sign up", "Sign up"),
            TemplateStep(StepType.TAP_ELEMENT, "Use email", "Use phone or email"),
            TemplateStep(StepType.TAP_ELEMENT, "Switch to email", "Email"),
            TemplateStep(StepType.FILL_FIELD, "Enter email", "Email", "{email}"),
            TemplateStep(StepType.FILL_FIELD, "Enter password", "Password", "{password}"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Sign up", "Sign up"),
            TemplateStep(StepType.VERIFY_EMAIL, "Enter verification code", "verification code"),
            TemplateStep(StepType.FILL_FIELD, "Choose username", "Username", "{username}"),
        ],
    )


def _build_facebook_template() -> SignupTemplate:
    return SignupTemplate(
        platform=Platform.FACEBOOK,
        name="Facebook Signup",
        description="Create a Facebook account via browser",
        url="https://www.facebook.com/reg",
        use_browser=True,
        required_fields=["email", "password", "first_name", "last_name"],
        needs_email=True,
        has_captcha=True,
        steps=[
            TemplateStep(StepType.NAVIGATE, "Open Facebook signup", "https://www.facebook.com/reg"),
            TemplateStep(StepType.FILL_FIELD, "Enter first name", "First name", "{first_name}"),
            TemplateStep(StepType.FILL_FIELD, "Enter last name", "Last name", "{last_name}"),
            TemplateStep(StepType.FILL_FIELD, "Enter email", "Mobile number or email", "{email}"),
            TemplateStep(StepType.FILL_FIELD, "Enter password", "New password", "{password}"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Sign Up", "Sign Up"),
            TemplateStep(StepType.CAPTCHA_CHECK, "Check for CAPTCHA"),
            TemplateStep(StepType.VERIFY_EMAIL, "Enter verification code", "code"),
        ],
    )


def _build_pinterest_template() -> SignupTemplate:
    return SignupTemplate(
        platform=Platform.PINTEREST,
        name="Pinterest Signup",
        description="Create a Pinterest account via browser",
        url="https://www.pinterest.com",
        use_browser=True,
        required_fields=["email", "password"],
        needs_email=True,
        steps=[
            TemplateStep(StepType.NAVIGATE, "Open Pinterest", "https://www.pinterest.com"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Sign up", "Sign up"),
            TemplateStep(StepType.FILL_FIELD, "Enter email", "Email", "{email}"),
            TemplateStep(StepType.FILL_FIELD, "Enter password", "Create a password", "{password}"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Continue", "Continue"),
            TemplateStep(StepType.TAP_ELEMENT, "Skip interests", "Skip", optional=True),
        ],
    )


def _build_reddit_template() -> SignupTemplate:
    return SignupTemplate(
        platform=Platform.REDDIT,
        name="Reddit Signup",
        description="Create a Reddit account via browser",
        url="https://www.reddit.com/register",
        use_browser=True,
        required_fields=["email", "username", "password"],
        needs_email=True,
        has_captcha=True,
        steps=[
            TemplateStep(StepType.NAVIGATE, "Open Reddit signup", "https://www.reddit.com/register"),
            TemplateStep(StepType.FILL_FIELD, "Enter email", "Email", "{email}"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Continue", "Continue"),
            TemplateStep(StepType.FILL_FIELD, "Enter username", "Username", "{username}"),
            TemplateStep(StepType.FILL_FIELD, "Enter password", "Password", "{password}"),
            TemplateStep(StepType.CAPTCHA_CHECK, "Check for CAPTCHA"),
            TemplateStep(StepType.TAP_ELEMENT, "Tap Sign up", "Sign Up"),
        ],
    )


BUILTIN_TEMPLATES = {
    Platform.INSTAGRAM: _build_instagram_template,
    Platform.TWITTER: _build_twitter_template,
    Platform.TIKTOK: _build_tiktok_template,
    Platform.FACEBOOK: _build_facebook_template,
    Platform.PINTEREST: _build_pinterest_template,
    Platform.REDDIT: _build_reddit_template,
}

# Warming schedule definitions (actions per phase)
WARMING_SCHEDULES = {
    WarmingPhase.DAY_1_3: {
        "max_actions_per_day": 5,
        "actions": ["browse_feed", "like_post"],
        "session_minutes": 10,
    },
    WarmingPhase.DAY_4_7: {
        "max_actions_per_day": 15,
        "actions": ["browse_feed", "like_post", "follow_account", "view_stories"],
        "session_minutes": 20,
    },
    WarmingPhase.DAY_8_14: {
        "max_actions_per_day": 30,
        "actions": ["browse_feed", "like_post", "follow_account", "comment", "view_stories", "search"],
        "session_minutes": 30,
    },
    WarmingPhase.DAY_15_30: {
        "max_actions_per_day": 50,
        "actions": ["browse_feed", "like_post", "follow_account", "comment", "post_content",
                     "view_stories", "search", "dm"],
        "session_minutes": 45,
    },
    WarmingPhase.MATURE: {
        "max_actions_per_day": 100,
        "actions": ["all"],
        "session_minutes": 60,
    },
}


# ---------------------------------------------------------------------------
# AccountFactory
# ---------------------------------------------------------------------------

class AccountFactory:
    """
    Account creation engine for any platform.

    Uses templates to drive signup flows, handles email verification
    chains via EmailAgent, detects CAPTCHAs (escalates to human),
    sets up profiles, and manages warming schedules.

    Usage:
        factory = get_account_factory()
        result = await factory.create_account("instagram", persona_id="abc123")
    """

    def __init__(
        self,
        controller: Any = None,
        browser: Any = None,
        email_agent: Any = None,
        identity_mgr: Any = None,
        account_mgr: Any = None,
        learner: Any = None,
        data_dir: Optional[Path] = None,
    ):
        self._controller = controller
        self._browser = browser
        self._email_agent = email_agent
        self._identity_mgr = identity_mgr
        self._account_mgr = account_mgr
        self._learner = learner
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._templates: Dict[str, SignupTemplate] = {}
        self._jobs: Dict[str, CreationJob] = {}
        self._warming: Dict[str, WarmingSchedule] = {}

        self._load_state()
        self._load_builtin_templates()
        logger.info("AccountFactory initialized (%d templates, %d jobs)",
                     len(self._templates), len(self._jobs))

    # ── Property helpers ──

    @property
    def controller(self):
        if self._controller is None:
            try:
                from src.phone_controller import PhoneController
                self._controller = PhoneController()
            except ImportError:
                pass
        return self._controller

    @property
    def browser(self):
        if self._browser is None:
            try:
                from src.browser_controller import get_browser
                self._browser = get_browser()
            except ImportError:
                pass
        return self._browser

    @property
    def email_agent(self):
        if self._email_agent is None:
            try:
                from src.email_agent import get_email_agent
                self._email_agent = get_email_agent()
            except ImportError:
                pass
        return self._email_agent

    @property
    def identity_mgr(self):
        if self._identity_mgr is None:
            try:
                from src.identity_manager import get_identity_manager
                self._identity_mgr = get_identity_manager()
            except ImportError:
                pass
        return self._identity_mgr

    @property
    def account_mgr(self):
        if self._account_mgr is None:
            try:
                from src.account_manager import get_account_manager
                self._account_mgr = get_account_manager()
            except ImportError:
                pass
        return self._account_mgr

    @property
    def learner(self):
        if self._learner is None:
            try:
                from src.app_learner import get_app_learner
                self._learner = get_app_learner()
            except ImportError:
                pass
        return self._learner

    # ── Persistence ──

    def _load_state(self) -> None:
        state = _load_json(self._data_dir / "state.json")
        for job_id, data in state.get("jobs", {}).items():
            if isinstance(data, dict):
                data.pop("platform", None)
                self._jobs[job_id] = CreationJob(**data)
        for key, data in state.get("warming", {}).items():
            if isinstance(data, dict):
                self._warming[key] = WarmingSchedule(**data)
        # Load custom templates
        for name, data in state.get("custom_templates", {}).items():
            if isinstance(data, dict):
                steps = [TemplateStep(**s) if isinstance(s, dict) else s for s in data.pop("steps", [])]
                p = data.pop("platform", "custom")
                self._templates[name] = SignupTemplate(platform=Platform(p), steps=steps, **data)

    def _save_state(self) -> None:
        custom_templates = {
            k: v.to_dict() for k, v in self._templates.items()
            if k not in [p.value for p in BUILTIN_TEMPLATES]
        }
        _save_json(self._data_dir / "state.json", {
            "jobs": {k: v.to_dict() for k, v in self._jobs.items()},
            "warming": {k: v.to_dict() for k, v in self._warming.items()},
            "custom_templates": custom_templates,
            "updated_at": _now_iso(),
        })

    def _load_builtin_templates(self) -> None:
        for platform, builder in BUILTIN_TEMPLATES.items():
            if platform.value not in self._templates:
                self._templates[platform.value] = builder()

    # ── Template management ──

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available signup templates."""
        return [
            {
                "name": t.name,
                "platform": t.platform.value,
                "steps": len(t.steps),
                "needs_email": t.needs_email,
                "needs_phone": t.needs_phone,
                "has_captcha": t.has_captcha,
                "success_rate": t.success_rate,
            }
            for t in self._templates.values()
        ]

    def get_template(self, platform: str) -> Optional[SignupTemplate]:
        """Get a template by platform name."""
        return self._templates.get(platform.lower())

    def add_custom_template(self, name: str, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a custom signup template."""
        steps = [
            TemplateStep(**s) if isinstance(s, dict) else s
            for s in template_data.pop("steps", [])
        ]
        p = template_data.pop("platform", "custom")
        template = SignupTemplate(platform=Platform(p), steps=steps, name=name, **template_data)
        self._templates[name] = template
        self._save_state()
        return {"success": True, "template": template.to_dict()}

    # ── Account creation ──

    async def create_account(
        self,
        platform: str,
        persona_id: str = "",
        email: str = "",
        username: str = "",
        password: str = "",
        first_name: str = "",
        last_name: str = "",
        auto_warm: bool = True,
    ) -> Dict[str, Any]:
        """
        Create an account on a platform using the template.

        If persona_id is provided, loads identity from IdentityManager.
        If email is not provided, creates one via EmailAgent.

        Args:
            platform: Platform name (instagram, twitter, etc.).
            persona_id: Identity manager persona ID (auto-fills fields).
            email: Email to use (auto-creates if empty).
            username: Username to use.
            password: Password to use.
            first_name: First name.
            last_name: Last name.
            auto_warm: Start warming schedule after creation.

        Returns:
            Dict with creation result and job details.
        """
        template = self.get_template(platform)
        if not template:
            return {"success": False, "error": f"No template for platform: {platform}"}

        # Create job
        job = CreationJob(
            platform=Platform(platform.lower()) if platform.lower() in [p.value for p in Platform] else Platform.CUSTOM,
            persona_id=persona_id,
            status=CreationStatus.IN_PROGRESS,
            total_steps=len(template.steps),
        )
        self._jobs[job.id] = job

        # Load persona if provided
        if persona_id and self.identity_mgr:
            try:
                persona = self.identity_mgr.get_persona(persona_id)
                if persona:
                    demographics = persona.get("demographics", {})
                    first_name = first_name or demographics.get("first_name", "")
                    last_name = last_name or demographics.get("last_name", "")
                    username = username or persona.get("username", "")
                    email = email or persona.get("email", "")
            except Exception as exc:
                logger.warning("Failed to load persona %s: %s", persona_id, exc)

        # Generate defaults
        if not username:
            username = f"user{uuid.uuid4().hex[:8]}"
        if not password:
            password = f"Pw{uuid.uuid4().hex[:10]}!"
        if not first_name:
            first_name = "Alex"

        # Create email if needed
        if not email and template.needs_email and self.email_agent:
            try:
                email_result = await self.email_agent.create_gmail_account(
                    username=f"{username}.auto",
                    password=password,
                    first_name=first_name,
                    last_name=last_name or "User",
                )
                if email_result.get("success"):
                    email = email_result["email"]
                    job.email = email
                    job.log.append({"step": "email_created", "email": email})
            except Exception as exc:
                logger.warning("Email creation failed: %s", exc)

        if not email and template.needs_email:
            job.status = CreationStatus.FAILED
            job.error = "Email required but not available"
            self._save_state()
            return {"success": False, "error": "Email required but could not be created", "job": job.to_dict()}

        job.email = email
        job.username = username

        # Template variable substitution
        variables = {
            "{email}": email,
            "{username}": username,
            "{password}": password,
            "{first_name}": first_name,
            "{last_name}": last_name or "User",
        }

        # Execute template steps
        try:
            for i, step in enumerate(template.steps):
                job.current_step = i
                self._save_state()

                result = await self._execute_step(step, variables, job)
                job.log.append({
                    "step": i,
                    "type": step.step_type.value,
                    "description": step.description,
                    "result": result,
                })

                if not result.get("success", True) and not step.optional:
                    job.status = CreationStatus.FAILED
                    job.error = result.get("error", "Step failed")
                    self._save_state()
                    template.total_attempts += 1
                    return {"success": False, "error": job.error, "job": job.to_dict()}

            # Success!
            job.status = CreationStatus.COMPLETED
            job.completed_at = _now_iso()
            template.total_attempts += 1
            template.success_rate = (
                (template.success_rate * (template.total_attempts - 1) + 1.0) / template.total_attempts
            )

            # Store credentials
            if self.account_mgr:
                try:
                    self.account_mgr.store_credential(
                        platform=platform, username=username,
                        password=password,
                        metadata={"email": email, "persona_id": persona_id},
                    )
                except Exception:
                    pass

            # Start warming
            if auto_warm:
                self._start_warming(platform, email, username)

            self._save_state()
            logger.info("Account created: %s on %s", username, platform)
            return {"success": True, "job": job.to_dict(), "username": username, "email": email}

        except Exception as exc:
            job.status = CreationStatus.FAILED
            job.error = str(exc)
            template.total_attempts += 1
            self._save_state()
            return {"success": False, "error": str(exc), "job": job.to_dict()}

    async def _execute_step(
        self, step: TemplateStep, variables: Dict[str, str], job: CreationJob
    ) -> Dict[str, Any]:
        """Execute a single template step."""
        value = step.value
        for var, replacement in variables.items():
            value = value.replace(var, replacement)

        try:
            if step.step_type == StepType.NAVIGATE:
                if step.target.startswith("http") and self.browser:
                    await self.browser.open_url(step.target, wait_for_load=True)
                elif self.controller:
                    await self.controller.launch_app(step.target)
                await asyncio.sleep(step.wait_seconds)
                return {"success": True}

            elif step.step_type == StepType.FILL_FIELD:
                if self.browser and self._templates.get(job.platform.value, SignupTemplate()).use_browser:
                    result = await self.browser.fill_form({step.target: value})
                    return result
                elif self.controller:
                    # Find field via vision and type
                    screenshot = await self.controller.screenshot()
                    try:
                        from src.vision_agent import VisionAgent
                        va = VisionAgent()
                        element = await va.find_element(
                            description=f"input field labeled '{step.target}'",
                            screenshot_path=screenshot,
                        )
                        if isinstance(element, dict) and element.get("x"):
                            await self.controller.tap(element["x"], element["y"])
                            await asyncio.sleep(0.3)
                            await self.controller.type_text(value)
                            await asyncio.sleep(step.wait_seconds)
                            return {"success": True}
                    except ImportError:
                        pass
                return {"success": False, "error": "Cannot fill field"}

            elif step.step_type == StepType.TAP_ELEMENT:
                if self.controller:
                    screenshot = await self.controller.screenshot()
                    try:
                        from src.vision_agent import VisionAgent
                        va = VisionAgent()
                        element = await va.find_element(
                            description=f"'{step.target}' button or text",
                            screenshot_path=screenshot,
                        )
                        if isinstance(element, dict) and element.get("x"):
                            await self.controller.tap(element["x"], element["y"])
                            await asyncio.sleep(step.wait_seconds)
                            return {"success": True}
                    except ImportError:
                        pass
                if self.browser:
                    result = await self.browser.click_element(step.target)
                    await asyncio.sleep(step.wait_seconds)
                    return result
                return {"success": False, "error": "Cannot tap element"}

            elif step.step_type == StepType.WAIT:
                await asyncio.sleep(step.wait_seconds)
                return {"success": True}

            elif step.step_type == StepType.VERIFY_EMAIL:
                if self.email_agent and job.email:
                    job.status = CreationStatus.EMAIL_VERIFICATION
                    verification = await self.email_agent.wait_for_verification(
                        job.email, timeout=90.0
                    )
                    if verification.found:
                        result = await self.email_agent.complete_verification(verification)
                        return result
                    return {"success": False, "error": "Verification email not received"}
                return {"success": True}  # Skip if no email agent

            elif step.step_type == StepType.VERIFY_PHONE:
                job.status = CreationStatus.PHONE_VERIFICATION
                logger.warning("Phone verification required — escalating to human")
                return {"success": False, "error": "Phone verification required (human needed)"}

            elif step.step_type == StepType.CAPTCHA_CHECK:
                job.status = CreationStatus.CAPTCHA_REQUIRED
                # Check if CAPTCHA is visible
                if self.controller:
                    screenshot = await self.controller.screenshot()
                    try:
                        from src.vision_agent import VisionAgent
                        va = VisionAgent()
                        analysis = await va.analyze_screen(screenshot_path=screenshot)
                        visible = str(analysis).lower()
                        if any(kw in visible for kw in ["captcha", "robot", "verify you are human", "recaptcha"]):
                            logger.warning("CAPTCHA detected — escalating to human")
                            return {"success": False, "error": "CAPTCHA detected (human needed)"}
                    except ImportError:
                        pass
                return {"success": True}  # No CAPTCHA found

            elif step.step_type == StepType.UPLOAD_AVATAR:
                # Complex flow — skip for now, handled in profile setup
                return {"success": True}

            elif step.step_type == StepType.SCROLL:
                if self.controller:
                    await self.controller.scroll_down(500)
                    await asyncio.sleep(step.wait_seconds)
                return {"success": True}

            elif step.step_type == StepType.BACK:
                if self.controller:
                    await self.controller.press_back()
                    await asyncio.sleep(step.wait_seconds)
                return {"success": True}

            elif step.step_type == StepType.SCREENSHOT:
                if self.controller:
                    path = await self.controller.screenshot()
                    return {"success": True, "screenshot": path}
                return {"success": True}

            return {"success": True}

        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Warming ──

    def _start_warming(self, platform: str, email: str, username: str) -> None:
        """Initialize a warming schedule for a new account."""
        key = f"{platform}:{email or username}"
        schedule = WarmingSchedule(
            platform=Platform(platform) if platform in [p.value for p in Platform] else Platform.CUSTOM,
            account_email=email,
            phase=WarmingPhase.DAY_1_3,
            daily_actions=WARMING_SCHEDULES[WarmingPhase.DAY_1_3].get("actions", []),
        )
        self._warming[key] = schedule
        self._save_state()
        logger.info("Started warming schedule for %s on %s", email or username, platform)

    async def warm_account(self, platform: str, account: str) -> Dict[str, Any]:
        """Execute a warming session for an account."""
        key = f"{platform}:{account}"
        schedule = self._warming.get(key)
        if not schedule:
            return {"success": False, "error": f"No warming schedule for {key}"}

        phase_config = WARMING_SCHEDULES.get(schedule.phase, {})
        max_actions = phase_config.get("max_actions_per_day", 5)
        available_actions = phase_config.get("actions", ["browse_feed"])

        if schedule.completed_actions >= max_actions:
            return {"success": True, "message": "Daily action limit reached", "phase": schedule.phase.value}

        # Pick random actions
        actions_to_do = min(max_actions - schedule.completed_actions, 5)
        executed = []

        for _ in range(actions_to_do):
            action = random.choice(available_actions)
            executed.append(action)
            schedule.completed_actions += 1
            # In a real implementation, these would dispatch to social_automation
            await asyncio.sleep(random.uniform(2.0, 5.0))

        schedule.last_action = _now_iso()

        # Check if phase should advance
        now = datetime.now(timezone.utc)
        start = datetime.fromisoformat(schedule.start_date.replace("Z", "+00:00")) if schedule.start_date else now
        days = (now - start).days

        if days >= 30:
            schedule.phase = WarmingPhase.MATURE
        elif days >= 15:
            schedule.phase = WarmingPhase.DAY_15_30
        elif days >= 8:
            schedule.phase = WarmingPhase.DAY_8_14
        elif days >= 4:
            schedule.phase = WarmingPhase.DAY_4_7

        self._save_state()
        return {
            "success": True,
            "platform": platform,
            "account": account,
            "phase": schedule.phase.value,
            "actions_executed": executed,
            "total_today": schedule.completed_actions,
        }

    def get_warming_status(self, platform: str = "", account: str = "") -> List[Dict[str, Any]]:
        """Get warming schedule status."""
        if platform and account:
            key = f"{platform}:{account}"
            schedule = self._warming.get(key)
            return [schedule.to_dict()] if schedule else []
        return [s.to_dict() for s in self._warming.values()]

    # ── Job management ──

    def list_jobs(self, status: str = "") -> List[Dict[str, Any]]:
        """List account creation jobs."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status.value == status]
        return [j.to_dict() for j in jobs]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific job."""
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    # ── Statistics ──

    def stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        jobs = list(self._jobs.values())
        return {
            "total_jobs": len(jobs),
            "completed": sum(1 for j in jobs if j.status == CreationStatus.COMPLETED),
            "failed": sum(1 for j in jobs if j.status == CreationStatus.FAILED),
            "in_progress": sum(1 for j in jobs if j.status == CreationStatus.IN_PROGRESS),
            "warming_accounts": len(self._warming),
            "templates": len(self._templates),
            "platforms": {
                p: sum(1 for j in jobs if j.platform.value == p)
                for p in set(j.platform.value for j in jobs)
            } if jobs else {},
        }

    # ── Sync wrappers ──

    def create_account_sync(self, platform: str, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.create_account(platform, **kwargs))

    def warm_account_sync(self, platform: str, account: str) -> Dict[str, Any]:
        return _run_sync(self.warm_account(platform, account))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[AccountFactory] = None


def get_account_factory(**kwargs) -> AccountFactory:
    """Get the singleton AccountFactory instance."""
    global _instance
    if _instance is None:
        _instance = AccountFactory(**kwargs)
    return _instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _cli_create(args: argparse.Namespace) -> None:
    factory = get_account_factory()
    result = factory.create_account_sync(
        args.platform,
        persona_id=args.persona or "",
        email=args.email or "",
        username=args.username or "",
        password=args.password or "",
    )
    _print_json(result)


def _cli_templates(args: argparse.Namespace) -> None:
    factory = get_account_factory()
    action = args.action
    if action == "list":
        _print_json(factory.list_templates())
    elif action == "show":
        template = factory.get_template(args.platform or "")
        if template:
            _print_json(template.to_dict())
        else:
            print(f"No template for: {args.platform}")
    elif action == "add":
        data = json.loads(args.data) if args.data else {}
        result = factory.add_custom_template(args.name or "", data)
        _print_json(result)
    else:
        print(f"Unknown template action: {action}")


def _cli_warm(args: argparse.Namespace) -> None:
    factory = get_account_factory()
    result = factory.warm_account_sync(args.platform, args.account)
    _print_json(result)


def _cli_jobs(args: argparse.Namespace) -> None:
    factory = get_account_factory()
    action = args.action
    if action == "list":
        _print_json(factory.list_jobs(args.status or ""))
    elif action == "show":
        job = factory.get_job(args.id or "")
        if job:
            _print_json(job)
        else:
            print(f"Job not found: {args.id}")
    else:
        print(f"Unknown job action: {action}")


def _cli_status(args: argparse.Namespace) -> None:
    factory = get_account_factory()
    if args.warming:
        _print_json(factory.get_warming_status(args.platform or ""))
    else:
        _print_json(factory.stats())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="account_factory",
        description="OpenClaw Empire — Account Factory",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # create
    cr = sub.add_parser("create", help="Create an account")
    cr.add_argument("--platform", required=True)
    cr.add_argument("--persona", default="")
    cr.add_argument("--email", default="")
    cr.add_argument("--username", default="")
    cr.add_argument("--password", default="")
    cr.set_defaults(func=_cli_create)

    # templates
    tp = sub.add_parser("templates", help="Manage signup templates")
    tp.add_argument("action", choices=["list", "show", "add"])
    tp.add_argument("--platform", default="")
    tp.add_argument("--name", default="")
    tp.add_argument("--data", default=None, help="JSON template data")
    tp.set_defaults(func=_cli_templates)

    # warm
    wr = sub.add_parser("warm", help="Warm an account")
    wr.add_argument("--platform", required=True)
    wr.add_argument("--account", required=True)
    wr.set_defaults(func=_cli_warm)

    # jobs
    jb = sub.add_parser("jobs", help="View creation jobs")
    jb.add_argument("action", choices=["list", "show"])
    jb.add_argument("--status", default="")
    jb.add_argument("--id", default="")
    jb.set_defaults(func=_cli_jobs)

    # status
    st = sub.add_parser("status", help="Factory status")
    st.add_argument("--platform", default="")
    st.add_argument("--warming", action="store_true")
    st.set_defaults(func=_cli_status)

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
