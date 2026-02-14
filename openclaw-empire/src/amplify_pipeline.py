"""
AMPLIFY Pipeline — Android Phone Automation Enhancement Engine
==============================================================

A 6-stage enhancement pipeline that processes every Android automation task
before execution, ensuring reliability, resilience, and performance.

Stages:
    1. ENRICH   - App & context awareness
    2. EXPAND   - Edge case coverage
    3. FORTIFY  - Retry & error recovery
    4. ANTICIPATE - UI state prediction
    5. OPTIMIZE - Performance learning
    6. VALIDATE - Pre-execution gating

Usage:
    pipeline = AmplifyPipeline()
    enhanced_task = pipeline.full_pipeline(task_config)
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("amplify_pipeline")
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
DATA_DIR = BASE_DIR / "data" / "amplify"
TIMING_DATA_FILE = DATA_DIR / "timing_data.json"

# Safety limits
MAX_STEPS_PER_TASK = 50
MAX_WAIT_PER_STEP_S = 30.0
MAX_RETRIES = 5

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    TAP_ELEMENT = "tap_element"
    TYPE_TEXT = "type_text"
    SWIPE_SCROLL = "swipe_scroll"
    LAUNCH_APP = "launch_app"
    FIND_ELEMENT = "find_element"
    NAVIGATE = "navigate"
    BACK_PRESS = "back_press"
    SUBMIT_FORM = "submit_form"
    POST_CONTENT = "post_content"
    SEND_MESSAGE = "send_message"
    MAKE_PURCHASE = "make_purchase"
    DELETE_CONTENT = "delete_content"
    ACCOUNT_SETTINGS = "account_settings"


class VerificationMethod(str, Enum):
    VISION_CHECK = "vision_check"
    UI_DUMP_CHECK = "ui_dump_check"
    TEXT_PRESENCE_CHECK = "text_presence_check"


class AppCategory(str, Enum):
    SOCIAL_MEDIA = "social_media"
    MESSAGING = "messaging"
    PRODUCTIVITY = "productivity"
    BROWSERS = "browsers"
    EMPIRE_TOOLS = "empire_tools"


@dataclass
class AppProfile:
    """Metadata about an Android application relevant to automation."""
    name: str
    package_name: str
    category: AppCategory
    typical_load_time: float  # seconds
    auth_type: str  # "oauth", "credentials", "none", "biometric", "token"
    common_screens: list[str] = field(default_factory=list)
    known_quirks: list[str] = field(default_factory=list)


@dataclass
class RetryPolicy:
    """Defines how an action should be retried on failure."""
    max_attempts: int = 3
    base_delay: float = 0.5
    backoff_multiplier: float = 1.5
    pre_retry_action: Optional[str] = None  # e.g. "clear_field", "force_stop"


@dataclass
class StateExpectation:
    """Describes the expected UI state after an action."""
    description: str
    verification_method: VerificationMethod
    expected_elements: list[str] = field(default_factory=list)
    expected_text: Optional[str] = None
    timeout: float = 5.0


@dataclass
class PreflightCheck:
    """A single check in the VALIDATE stage."""
    name: str
    passed: bool
    message: str
    blocking: bool = False


@dataclass
class ValidationResult:
    """Outcome of the VALIDATE stage."""
    valid: bool
    checks: list[PreflightCheck] = field(default_factory=list)
    blocking_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TimingRecord:
    """A single recorded timing data point."""
    action_type: str
    app_name: str
    duration: float
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# App Profile Registry
# ---------------------------------------------------------------------------

APP_PROFILES: dict[str, AppProfile] = {
    # Social Media
    "facebook": AppProfile(
        name="Facebook",
        package_name="com.facebook.katana",
        category=AppCategory.SOCIAL_MEDIA,
        typical_load_time=4.0,
        auth_type="credentials",
        common_screens=["news_feed", "profile", "marketplace", "groups", "notifications"],
        known_quirks=[
            "Aggressive session timeout on background",
            "Frequent 'Rate this app' dialogs",
            "Dynamic feed — element positions shift on reload",
            "Stories tray can intercept taps near top of feed",
        ],
    ),
    "instagram": AppProfile(
        name="Instagram",
        package_name="com.instagram.android",
        category=AppCategory.SOCIAL_MEDIA,
        typical_load_time=3.5,
        auth_type="credentials",
        common_screens=["home_feed", "explore", "reels", "profile", "direct_messages"],
        known_quirks=[
            "Bottom sheet modals can block navigation bar",
            "Reels auto-play can consume focus",
            "Story creation has multi-step flow with animations",
            "Image upload compresses — sizes may differ from original",
        ],
    ),
    "twitter_x": AppProfile(
        name="X (Twitter)",
        package_name="com.twitter.android",
        category=AppCategory.SOCIAL_MEDIA,
        typical_load_time=3.0,
        auth_type="credentials",
        common_screens=["home_timeline", "explore", "notifications", "messages", "profile"],
        known_quirks=[
            "Pull-to-refresh triggers timeline jump",
            "Promoted tweets shift element positions",
            "Character counter can obscure post button",
            "Media upload requires wait for processing",
        ],
    ),
    "tiktok": AppProfile(
        name="TikTok",
        package_name="com.zhiliaoapp.musically",
        category=AppCategory.SOCIAL_MEDIA,
        typical_load_time=4.5,
        auth_type="credentials",
        common_screens=["for_you", "following", "create", "inbox", "profile"],
        known_quirks=[
            "Video autoplay on feed — can consume data / battery",
            "Creation flow is multi-step with effects selection",
            "Heavy on animations and transitions",
            "Age-gate may appear for certain content",
        ],
    ),
    "pinterest": AppProfile(
        name="Pinterest",
        package_name="com.pinterest",
        category=AppCategory.SOCIAL_MEDIA,
        typical_load_time=3.5,
        auth_type="credentials",
        common_screens=["home_feed", "search", "saved_pins", "profile", "create_pin"],
        known_quirks=[
            "Masonry layout — element positions are unpredictable",
            "Image-heavy — slow on poor connections",
            "Pin creation requires board selection step",
            "Search suggestions overlay can intercept taps",
        ],
    ),
    "linkedin": AppProfile(
        name="LinkedIn",
        package_name="com.linkedin.android",
        category=AppCategory.SOCIAL_MEDIA,
        typical_load_time=4.0,
        auth_type="credentials",
        common_screens=["feed", "my_network", "post_creation", "messaging", "notifications"],
        known_quirks=[
            "Connection suggestions popup on launch",
            "Rich text editor in post creation",
            "Premium upsell banners appear frequently",
            "Document uploads are slow and unreliable",
        ],
    ),

    # Messaging
    "whatsapp": AppProfile(
        name="WhatsApp",
        package_name="com.whatsapp",
        category=AppCategory.MESSAGING,
        typical_load_time=2.0,
        auth_type="biometric",
        common_screens=["chats_list", "chat_view", "status", "calls", "settings"],
        known_quirks=[
            "End-to-end encryption banner on new chats",
            "Media auto-download can slow interface",
            "Fingerprint lock may appear on resume",
            "Group info screen is long-scrollable",
        ],
    ),
    "telegram": AppProfile(
        name="Telegram",
        package_name="org.telegram.messenger",
        category=AppCategory.MESSAGING,
        typical_load_time=1.5,
        auth_type="credentials",
        common_screens=["chats_list", "chat_view", "channels", "settings", "contacts"],
        known_quirks=[
            "Sticker/GIF picker can overlay keyboard",
            "Channel posts have different layout than DMs",
            "Passcode lock if enabled",
            "File sharing has size-based upload delay",
        ],
    ),
    "discord": AppProfile(
        name="Discord",
        package_name="com.discord",
        category=AppCategory.MESSAGING,
        typical_load_time=3.0,
        auth_type="credentials",
        common_screens=["server_list", "channel_view", "direct_messages", "friends", "settings"],
        known_quirks=[
            "Server switching requires swipe or tap on sidebar",
            "Voice channel join prompts for microphone permission",
            "Thread navigation can be confusing",
            "Markdown rendering in messages",
        ],
    ),
    "slack": AppProfile(
        name="Slack",
        package_name="com.Slack",
        category=AppCategory.MESSAGING,
        typical_load_time=3.5,
        auth_type="oauth",
        common_screens=["home", "dms", "channels", "threads", "search"],
        known_quirks=[
            "Workspace switching menu at top",
            "Thread replies open in side panel",
            "Notification grouping can be confusing",
            "Slow initial sync on large workspaces",
        ],
    ),
    "sms": AppProfile(
        name="SMS (Messages)",
        package_name="com.google.android.apps.messaging",
        category=AppCategory.MESSAGING,
        typical_load_time=1.0,
        auth_type="none",
        common_screens=["conversations_list", "conversation_view", "contacts"],
        known_quirks=[
            "RCS features may show different UI than SMS",
            "Default app prompt if not set as default",
            "Group messages have different thread handling",
        ],
    ),

    # Productivity
    "gmail": AppProfile(
        name="Gmail",
        package_name="com.google.android.gm",
        category=AppCategory.PRODUCTIVITY,
        typical_load_time=3.0,
        auth_type="oauth",
        common_screens=["inbox", "compose", "search", "settings", "labels"],
        known_quirks=[
            "Account switcher at top right",
            "Promotional tab may hide emails",
            "Compose window can overlay inbox",
            "Attachment upload is asynchronous",
        ],
    ),
    "calendar": AppProfile(
        name="Google Calendar",
        package_name="com.google.android.calendar",
        category=AppCategory.PRODUCTIVITY,
        typical_load_time=2.0,
        auth_type="oauth",
        common_screens=["day_view", "week_view", "month_view", "event_creation", "settings"],
        known_quirks=[
            "View toggle can change element positions dramatically",
            "All-day events are in a collapsible section",
            "Time zone differences can cause confusion",
        ],
    ),
    "notes": AppProfile(
        name="Google Keep",
        package_name="com.google.android.keep",
        category=AppCategory.PRODUCTIVITY,
        typical_load_time=2.0,
        auth_type="oauth",
        common_screens=["notes_list", "note_editor", "reminders", "labels", "archive"],
        known_quirks=[
            "Grid vs list view toggle changes layout",
            "Color labels can make text hard to read",
            "Checklist mode changes input behavior",
        ],
    ),
    "files": AppProfile(
        name="Files by Google",
        package_name="com.google.android.apps.nbu.files",
        category=AppCategory.PRODUCTIVITY,
        typical_load_time=1.5,
        auth_type="none",
        common_screens=["browse", "clean", "share", "favorites"],
        known_quirks=[
            "Storage permission required",
            "Large folder listing can be slow",
            "File type icons vary by extension",
        ],
    ),
    "settings": AppProfile(
        name="Settings",
        package_name="com.android.settings",
        category=AppCategory.PRODUCTIVITY,
        typical_load_time=0.5,
        auth_type="none",
        common_screens=["main_settings", "wifi", "bluetooth", "display", "apps", "security"],
        known_quirks=[
            "Deep nesting — many sub-menus",
            "OEM customizations change layout per manufacturer",
            "Developer options hidden until unlocked",
            "Search is the most reliable navigation method",
        ],
    ),

    # Browsers
    "chrome": AppProfile(
        name="Google Chrome",
        package_name="com.android.chrome",
        category=AppCategory.BROWSERS,
        typical_load_time=2.0,
        auth_type="oauth",
        common_screens=["new_tab", "web_page", "tabs_overview", "settings", "downloads"],
        known_quirks=[
            "Tab overflow hides tab count",
            "Address bar moves to bottom on some versions",
            "Incognito mode has different theme",
            "Cookie consent banners on most sites",
            "Data saver mode can break some pages",
        ],
    ),
    "firefox": AppProfile(
        name="Firefox",
        package_name="org.mozilla.firefox",
        category=AppCategory.BROWSERS,
        typical_load_time=2.5,
        auth_type="credentials",
        common_screens=["home", "web_page", "tabs", "settings", "bookmarks"],
        known_quirks=[
            "Bottom toolbar by default",
            "Collection feature popup",
            "Enhanced tracking protection banner",
        ],
    ),
    "samsung_internet": AppProfile(
        name="Samsung Internet",
        package_name="com.sec.android.app.sbrowser",
        category=AppCategory.BROWSERS,
        typical_load_time=2.0,
        auth_type="none",
        common_screens=["home", "web_page", "tabs", "settings", "bookmarks"],
        known_quirks=[
            "Samsung account sync prompts",
            "Ad blocker built-in affects page rendering",
            "Secret mode has different theme and behavior",
        ],
    ),

    # Empire Tools
    "wordpress": AppProfile(
        name="WordPress",
        package_name="org.wordpress.android",
        category=AppCategory.EMPIRE_TOOLS,
        typical_load_time=3.0,
        auth_type="credentials",
        common_screens=["my_sites", "stats", "posts_list", "post_editor", "media", "settings"],
        known_quirks=[
            "Site switcher at top — must verify correct site selected",
            "Block editor can be slow with many blocks",
            "Image upload requires gallery permission",
            "Draft auto-save can conflict with manual saves",
            "Jetpack connection prompts",
        ],
    ),
    "analytics": AppProfile(
        name="Google Analytics",
        package_name="com.google.android.apps.giant",
        category=AppCategory.EMPIRE_TOOLS,
        typical_load_time=4.0,
        auth_type="oauth",
        common_screens=["home_overview", "realtime", "reports", "explore", "settings"],
        known_quirks=[
            "GA4 vs Universal Analytics — different layouts",
            "Property switcher can be confusing with many properties",
            "Date range picker is finicky",
            "Reports load asynchronously — wait for spinner",
        ],
    ),
    "adsense": AppProfile(
        name="Google AdSense",
        package_name="com.google.android.apps.ads.publisher",
        category=AppCategory.EMPIRE_TOOLS,
        typical_load_time=3.5,
        auth_type="oauth",
        common_screens=["earnings_overview", "reports", "payments", "settings"],
        known_quirks=[
            "Revenue data has 24-48 hour delay",
            "Currency formatting varies by locale",
            "Performance reports can take time to generate",
        ],
    ),
    "etsy": AppProfile(
        name="Etsy Seller",
        package_name="com.etsy.android",
        category=AppCategory.EMPIRE_TOOLS,
        typical_load_time=3.5,
        auth_type="credentials",
        common_screens=["shop_dashboard", "listings", "orders", "stats", "messages"],
        known_quirks=[
            "Buyer vs seller mode — must be in seller mode",
            "Listing creation is multi-step with photo upload",
            "Stats have significant lag",
            "Shipping profiles can be complex",
        ],
    ),
    "amazon_kdp": AppProfile(
        name="Amazon KDP (via browser)",
        package_name="com.android.chrome",  # KDP has no native app
        category=AppCategory.EMPIRE_TOOLS,
        typical_load_time=5.0,
        auth_type="credentials",
        common_screens=["bookshelf", "create_title", "pricing", "reports"],
        known_quirks=[
            "No native app — must use mobile browser",
            "Two-factor authentication on login",
            "Manuscript upload requires file picker",
            "Preview generation takes several minutes",
            "Session timeout is aggressive (30 min)",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Retry policies per action type
# ---------------------------------------------------------------------------

RETRY_POLICIES: dict[str, RetryPolicy] = {
    ActionType.TAP_ELEMENT: RetryPolicy(
        max_attempts=3,
        base_delay=0.5,
        backoff_multiplier=1.5,
    ),
    ActionType.TYPE_TEXT: RetryPolicy(
        max_attempts=2,
        base_delay=0.3,
        backoff_multiplier=1.0,
        pre_retry_action="clear_field",
    ),
    ActionType.SWIPE_SCROLL: RetryPolicy(
        max_attempts=3,
        base_delay=0.5,
        backoff_multiplier=1.0,
        pre_retry_action="adjust_coordinates",
    ),
    ActionType.LAUNCH_APP: RetryPolicy(
        max_attempts=3,
        base_delay=2.0,
        backoff_multiplier=2.0,
        pre_retry_action="force_stop",
    ),
    ActionType.FIND_ELEMENT: RetryPolicy(
        max_attempts=5,
        base_delay=1.0,
        backoff_multiplier=1.5,
    ),
    ActionType.NAVIGATE: RetryPolicy(
        max_attempts=3,
        base_delay=1.0,
        backoff_multiplier=1.5,
        pre_retry_action="go_home_and_restart",
    ),
}

# Fallback chains: ordered list of strategies when primary method fails
FALLBACK_STRATEGIES: dict[str, list[str]] = {
    "tap": ["coordinates", "accessibility_id", "text_match"],
    "text_input": ["input_text", "clipboard_paste", "key_events"],
    "navigation": ["intent_launch", "home_and_recent_apps", "search"],
}


# ---------------------------------------------------------------------------
# Stage 1: ENRICH
# ---------------------------------------------------------------------------

class EnrichStage:
    """Add context about the target Android app and fill missing metadata."""

    def __init__(self, profiles: dict[str, AppProfile] | None = None) -> None:
        self._profiles = profiles or APP_PROFILES

    def identify_app(self, app_hint: str) -> Optional[AppProfile]:
        """Resolve a user-provided app name or package to a known profile."""
        hint_lower = app_hint.lower().strip()

        # Direct key match
        if hint_lower in self._profiles:
            return self._profiles[hint_lower]

        # Match by package name
        for profile in self._profiles.values():
            if profile.package_name.lower() == hint_lower:
                return profile

        # Fuzzy match by display name
        for key, profile in self._profiles.items():
            if hint_lower in profile.name.lower() or hint_lower in key:
                return profile

        logger.warning("No app profile found for '%s'", app_hint)
        return None

    def enrich(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Stage 1: ENRICH -- Augment task config with app context.

        Fills in missing fields such as expected load times, known quirks,
        and package names without overriding anything explicitly set.
        """
        logger.info("STAGE 1 [ENRICH] -- Adding app & context awareness")

        app_name = task.get("app") or task.get("app_name") or task.get("target_app", "")
        profile = self.identify_app(app_name) if app_name else None

        if profile is None:
            logger.info("  No known profile for app '%s'; proceeding with defaults", app_name)
            task.setdefault("app_profile", None)
            task.setdefault("app_category", "unknown")
            task.setdefault("typical_load_time", 3.0)
            task.setdefault("auth_type", "unknown")
            task.setdefault("known_quirks", [])
            return task

        logger.info("  Matched profile: %s (%s)", profile.name, profile.package_name)

        # Fill defaults without overriding explicit values
        task.setdefault("app_profile", profile.name)
        task.setdefault("package_name", profile.package_name)
        task.setdefault("app_category", profile.category.value)
        task.setdefault("typical_load_time", profile.typical_load_time)
        task.setdefault("auth_type", profile.auth_type)
        task.setdefault("common_screens", profile.common_screens)
        task.setdefault("known_quirks", profile.known_quirks)

        # Inject quirk-based adjustments into steps
        steps: list[dict[str, Any]] = task.get("steps", [])
        for step in steps:
            if step.get("action") == ActionType.LAUNCH_APP:
                step.setdefault("wait_after", profile.typical_load_time)
            if profile.auth_type == "biometric":
                step.setdefault("expect_auth_prompt", True)

        task["_enriched"] = True
        return task


# ---------------------------------------------------------------------------
# Stage 2: EXPAND
# ---------------------------------------------------------------------------

class ExpandStage:
    """Handle Android-specific edge cases that cause automation failures."""

    # Edge case handlers: condition name -> handler configuration
    EDGE_CASES: dict[str, dict[str, Any]] = {
        "notification_shade": {
            "detection": "status_bar_expanded",
            "action": "swipe_up_from_center",
            "description": "Notification shade pulled down -- dismiss",
        },
        "permission_dialog": {
            "detection": "text_contains:Allow|Deny|While using|Only this time",
            "action": "handle_permission",
            "description": "Permission dialog appeared -- handle appropriately",
        },
        "app_update_dialog": {
            "detection": "text_contains:Update available|Update now|Later|Not now",
            "action": "dismiss_update",
            "description": "App update dialog -- dismiss or defer",
        },
        "anr_dialog": {
            "detection": "text_contains:isn't responding|Wait|Close app",
            "action": "handle_anr",
            "description": "App Not Responding dialog -- wait or force close",
        },
        "screen_rotation": {
            "detection": "orientation_changed",
            "action": "lock_orientation_portrait",
            "description": "Screen rotation changed -- lock to portrait",
        },
        "keyboard_covering_target": {
            "detection": "target_below_keyboard",
            "action": "scroll_target_into_view",
            "description": "Keyboard covering target element -- scroll up",
        },
        "cookie_consent": {
            "detection": "text_contains:Accept cookies|Accept all|Cookie|GDPR|Consent",
            "action": "accept_consent",
            "description": "Cookie/GDPR consent banner -- accept and continue",
        },
        "rate_limit_captcha": {
            "detection": "text_contains:Captcha|Verify|I'm not a robot|rate limit",
            "action": "pause_and_alert",
            "description": "Rate limiting or captcha detected -- pause and alert user",
        },
    }

    def expand(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Stage 2: EXPAND -- Inject edge-case handlers and enforce safety limits.

        Wraps each step with pre/post checks for common Android interruptions
        and adds timing metadata for slow operations.
        """
        logger.info("STAGE 2 [EXPAND] -- Adding edge case coverage")

        steps: list[dict[str, Any]] = task.get("steps", [])

        # Enforce safety limits
        if len(steps) > MAX_STEPS_PER_TASK:
            logger.warning(
                "  Task has %d steps; truncating to %d", len(steps), MAX_STEPS_PER_TASK
            )
            steps = steps[:MAX_STEPS_PER_TASK]
            task["steps"] = steps
            task.setdefault("warnings", []).append(
                f"Task truncated from {len(steps)} to {MAX_STEPS_PER_TASK} steps"
            )

        for i, step in enumerate(steps):
            step.setdefault("step_index", i)
            step.setdefault("max_wait", MAX_WAIT_PER_STEP_S)
            step.setdefault("max_retries", min(step.get("max_retries", 3), MAX_RETRIES))

            # Attach relevant edge-case guards
            step["edge_case_guards"] = self._select_guards(step)

            # Add timing metadata for slow operations
            action = step.get("action", "")
            step["timing_meta"] = self._timing_metadata(action, task)

        task["_expanded"] = True
        return task

    def _select_guards(self, step: dict[str, Any]) -> list[dict[str, Any]]:
        """Select edge-case guards relevant to the given step."""
        guards: list[dict[str, Any]] = []

        # Always guard against notification shade and ANR dialogs
        guards.append(self.EDGE_CASES["notification_shade"])
        guards.append(self.EDGE_CASES["anr_dialog"])

        action = step.get("action", "")

        # Permission dialog is relevant for app launches and first-time actions
        if action in (ActionType.LAUNCH_APP, ActionType.NAVIGATE):
            guards.append(self.EDGE_CASES["permission_dialog"])
            guards.append(self.EDGE_CASES["app_update_dialog"])

        # Keyboard issues when typing
        if action in (ActionType.TYPE_TEXT, ActionType.TAP_ELEMENT):
            guards.append(self.EDGE_CASES["keyboard_covering_target"])

        # Cookie/consent when navigating web content
        if action in (ActionType.NAVIGATE,) or step.get("is_web_content", False):
            guards.append(self.EDGE_CASES["cookie_consent"])

        # Captcha/rate-limit when doing high-frequency actions
        if step.get("high_frequency", False):
            guards.append(self.EDGE_CASES["rate_limit_captcha"])

        # Screen rotation for any step (can happen anytime)
        guards.append(self.EDGE_CASES["screen_rotation"])

        return guards

    @staticmethod
    def _timing_metadata(action: str, task: dict[str, Any]) -> dict[str, Any]:
        """Compute timing metadata for operations that need extra wait time."""
        meta: dict[str, Any] = {"needs_extra_wait": False}

        load_time = task.get("typical_load_time", 3.0)

        if action == ActionType.LAUNCH_APP:
            meta["needs_extra_wait"] = True
            meta["expected_duration"] = load_time
            meta["reason"] = "App launch — waiting for splash screen and main activity"

        elif action == ActionType.SUBMIT_FORM:
            meta["needs_extra_wait"] = True
            meta["expected_duration"] = max(load_time, 5.0)
            meta["reason"] = "Form submission — waiting for server response"

        elif action == ActionType.NAVIGATE:
            meta["needs_extra_wait"] = True
            meta["expected_duration"] = max(load_time * 0.7, 2.0)
            meta["reason"] = "Navigation — waiting for page transition"

        elif action == ActionType.TYPE_TEXT:
            meta["needs_extra_wait"] = False
            meta["expected_duration"] = 0.5
            meta["reason"] = "Text input — minimal wait"

        return meta


# ---------------------------------------------------------------------------
# Stage 3: FORTIFY
# ---------------------------------------------------------------------------

class FortifyStage:
    """Attach retry policies and fallback strategies to each step."""

    def __init__(
        self,
        retry_policies: dict[str, RetryPolicy] | None = None,
        fallback_strategies: dict[str, list[str]] | None = None,
    ) -> None:
        self._policies = retry_policies or RETRY_POLICIES
        self._fallbacks = fallback_strategies or FALLBACK_STRATEGIES

    def fortify(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Stage 3: FORTIFY -- Add retry policies and fallback strategies.

        Each step gets a retry policy based on its action type and a list of
        fallback strategies to try when the primary approach fails.
        """
        logger.info("STAGE 3 [FORTIFY] -- Adding retry & error recovery")

        steps: list[dict[str, Any]] = task.get("steps", [])

        for step in steps:
            action = step.get("action", "")

            # Attach retry policy
            policy = self._policies.get(action)
            if policy is not None:
                step["retry_policy"] = {
                    "max_attempts": policy.max_attempts,
                    "base_delay": policy.base_delay,
                    "backoff_multiplier": policy.backoff_multiplier,
                    "pre_retry_action": policy.pre_retry_action,
                }
            else:
                # Default conservative policy for unknown actions
                step["retry_policy"] = {
                    "max_attempts": 2,
                    "base_delay": 1.0,
                    "backoff_multiplier": 1.5,
                    "pre_retry_action": None,
                }

            # Attach fallback strategies
            step["fallback_chain"] = self._resolve_fallbacks(action)

        task["_fortified"] = True
        return task

    def _resolve_fallbacks(self, action: str) -> list[str]:
        """Map action type to its fallback chain."""
        if action in (ActionType.TAP_ELEMENT,):
            return list(self._fallbacks.get("tap", []))
        elif action in (ActionType.TYPE_TEXT,):
            return list(self._fallbacks.get("text_input", []))
        elif action in (ActionType.NAVIGATE, ActionType.LAUNCH_APP):
            return list(self._fallbacks.get("navigation", []))
        return []

    @staticmethod
    def wrap_with_retry(policy: RetryPolicy) -> Callable[[F], F]:
        """
        Decorator that wraps a callable with retry logic per the given policy.

        Usage:
            @FortifyStage.wrap_with_retry(RetryPolicy(max_attempts=3, base_delay=0.5))
            def tap_button(element_id: str) -> bool:
                ...
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception: Optional[Exception] = None
                delay = policy.base_delay

                for attempt in range(1, policy.max_attempts + 1):
                    try:
                        logger.debug(
                            "  [RETRY] Attempt %d/%d for %s",
                            attempt,
                            policy.max_attempts,
                            func.__name__,
                        )
                        return func(*args, **kwargs)
                    except Exception as exc:
                        last_exception = exc
                        logger.warning(
                            "  [RETRY] %s failed (attempt %d): %s",
                            func.__name__,
                            attempt,
                            exc,
                        )
                        if attempt < policy.max_attempts:
                            time.sleep(delay)
                            delay *= policy.backoff_multiplier

                raise RuntimeError(
                    f"{func.__name__} failed after {policy.max_attempts} attempts"
                ) from last_exception

            return wrapper  # type: ignore[return-value]

        return decorator


# ---------------------------------------------------------------------------
# Stage 4: ANTICIPATE
# ---------------------------------------------------------------------------

class AnticipateStage:
    """Predict what the screen should look like after each action."""

    # State transition templates per action type
    STATE_TRANSITIONS: dict[str, dict[str, Any]] = {
        ActionType.LAUNCH_APP: {
            "expect": "App splash screen or main activity",
            "verification": VerificationMethod.VISION_CHECK,
            "timeout_factor": 1.0,  # multiplied by typical_load_time
        },
        ActionType.TAP_ELEMENT: {
            "expect": "New screen, dialog, or state change",
            "verification": VerificationMethod.UI_DUMP_CHECK,
            "timeout_factor": 0.3,
        },
        ActionType.TYPE_TEXT: {
            "expect": "Text visible in target field",
            "verification": VerificationMethod.TEXT_PRESENCE_CHECK,
            "timeout_factor": 0.2,
        },
        ActionType.SWIPE_SCROLL: {
            "expect": "New content visible in scroll area",
            "verification": VerificationMethod.VISION_CHECK,
            "timeout_factor": 0.3,
        },
        ActionType.BACK_PRESS: {
            "expect": "Previous screen restored",
            "verification": VerificationMethod.UI_DUMP_CHECK,
            "timeout_factor": 0.3,
        },
        ActionType.SUBMIT_FORM: {
            "expect": "Success message, confirmation, or error feedback",
            "verification": VerificationMethod.TEXT_PRESENCE_CHECK,
            "timeout_factor": 1.5,
        },
        ActionType.NAVIGATE: {
            "expect": "Target screen or page loaded",
            "verification": VerificationMethod.VISION_CHECK,
            "timeout_factor": 1.0,
        },
        ActionType.FIND_ELEMENT: {
            "expect": "Element located on screen",
            "verification": VerificationMethod.UI_DUMP_CHECK,
            "timeout_factor": 0.5,
        },
    }

    def anticipate(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Stage 4: ANTICIPATE -- Predict expected UI state after each action.

        Each step receives a state_expectation object describing what the
        screen should look like post-execution. If actual state does not
        match, the FORTIFY retry mechanism is triggered.
        """
        logger.info("STAGE 4 [ANTICIPATE] -- Predicting UI state transitions")

        steps: list[dict[str, Any]] = task.get("steps", [])
        load_time = task.get("typical_load_time", 3.0)

        for i, step in enumerate(steps):
            action = step.get("action", "")
            transition = self.STATE_TRANSITIONS.get(action)

            if transition is not None:
                timeout = load_time * transition["timeout_factor"]
                timeout = max(timeout, 1.0)  # at least 1 second
                timeout = min(timeout, MAX_WAIT_PER_STEP_S)

                expectation = StateExpectation(
                    description=transition["expect"],
                    verification_method=transition["verification"],
                    expected_elements=self._predict_elements(step, action),
                    expected_text=self._predict_text(step, action),
                    timeout=timeout,
                )
            else:
                # Generic expectation for unknown action types
                expectation = StateExpectation(
                    description="Action completed without error",
                    verification_method=VerificationMethod.UI_DUMP_CHECK,
                    timeout=3.0,
                )

            step["state_expectation"] = {
                "description": expectation.description,
                "verification_method": expectation.verification_method.value,
                "expected_elements": expectation.expected_elements,
                "expected_text": expectation.expected_text,
                "timeout": expectation.timeout,
                "retry_on_mismatch": True,
            }

            # Link to next step for chained verification
            if i < len(steps) - 1:
                next_step = steps[i + 1]
                step["state_expectation"]["next_action"] = next_step.get("action", "unknown")

        task["_anticipated"] = True
        return task

    @staticmethod
    def _predict_elements(step: dict[str, Any], action: str) -> list[str]:
        """Predict which UI elements should be present after the action."""
        elements: list[str] = []

        if action == ActionType.LAUNCH_APP:
            app = step.get("app", step.get("target_app", ""))
            elements.append(f"{app}_main_activity")
            elements.append("action_bar")

        elif action == ActionType.TAP_ELEMENT:
            target = step.get("target", step.get("element", ""))
            if target:
                elements.append(f"result_of_{target}")

        elif action == ActionType.SUBMIT_FORM:
            elements.extend(["success_indicator", "confirmation_dialog"])

        elif action == ActionType.NAVIGATE:
            destination = step.get("destination", step.get("target_screen", ""))
            if destination:
                elements.append(destination)

        return elements

    @staticmethod
    def _predict_text(step: dict[str, Any], action: str) -> Optional[str]:
        """Predict what text should be visible after the action."""
        if action == ActionType.TYPE_TEXT:
            return step.get("text", step.get("input_text"))

        if action == ActionType.SUBMIT_FORM:
            return step.get("expected_success_text", "Success")

        return None


# ---------------------------------------------------------------------------
# Stage 5: OPTIMIZE
# ---------------------------------------------------------------------------

class OptimizeStage:
    """Learn optimal timing and batching from historical execution data."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or DATA_DIR
        self._timing_file = self._data_dir / "timing_data.json"
        self._timing_data: list[dict[str, Any]] = []
        self._load_timing_data()

    def _load_timing_data(self) -> None:
        """Load persisted timing data from disk."""
        if self._timing_file.exists():
            try:
                with open(self._timing_file, "r", encoding="utf-8") as f:
                    self._timing_data = json.load(f)
                logger.debug(
                    "  Loaded %d timing records from %s",
                    len(self._timing_data),
                    self._timing_file,
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("  Failed to load timing data: %s", exc)
                self._timing_data = []
        else:
            logger.debug("  No existing timing data at %s", self._timing_file)
            self._timing_data = []

    def _save_timing_data(self) -> None:
        """Persist timing data to disk."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._timing_file, "w", encoding="utf-8") as f:
                json.dump(self._timing_data, f, indent=2, default=str)
            logger.debug("  Saved %d timing records", len(self._timing_data))
        except OSError as exc:
            logger.warning("  Failed to save timing data: %s", exc)

    def record_timing(self, record: TimingRecord) -> None:
        """Record a timing observation for future optimization."""
        self._timing_data.append({
            "action_type": record.action_type,
            "app_name": record.app_name,
            "duration": record.duration,
            "success": record.success,
            "timestamp": record.timestamp,
        })
        # Keep data bounded — retain last 5000 records
        if len(self._timing_data) > 5000:
            self._timing_data = self._timing_data[-5000:]
        self._save_timing_data()

    def get_optimal_delay(self, action_type: str, app_name: str) -> Optional[float]:
        """
        Compute the 90th-percentile duration for a given action+app pair.

        Returns None if insufficient data (<3 successful observations).
        """
        durations = [
            r["duration"]
            for r in self._timing_data
            if r["action_type"] == action_type
            and r["app_name"] == app_name
            and r["success"]
        ]

        if len(durations) < 3:
            return None

        durations.sort()
        p90_index = int(len(durations) * 0.9)
        p90_value = durations[min(p90_index, len(durations) - 1)]
        return round(p90_value, 2)

    def get_app_performance_stats(self, app_name: str) -> dict[str, Any]:
        """Get aggregated performance statistics for a specific app."""
        app_records = [r for r in self._timing_data if r["app_name"] == app_name]

        if not app_records:
            return {"app_name": app_name, "total_records": 0, "message": "No data available"}

        total = len(app_records)
        successes = sum(1 for r in app_records if r["success"])
        durations = [r["duration"] for r in app_records if r["success"]]

        stats: dict[str, Any] = {
            "app_name": app_name,
            "total_records": total,
            "success_rate": round(successes / total, 3) if total > 0 else 0.0,
            "total_successes": successes,
            "total_failures": total - successes,
        }

        if durations:
            durations.sort()
            stats["avg_duration"] = round(sum(durations) / len(durations), 3)
            stats["min_duration"] = round(durations[0], 3)
            stats["max_duration"] = round(durations[-1], 3)
            stats["p50_duration"] = round(durations[len(durations) // 2], 3)
            p90_idx = int(len(durations) * 0.9)
            stats["p90_duration"] = round(durations[min(p90_idx, len(durations) - 1)], 3)

        # Per-action breakdown
        action_breakdown: dict[str, dict[str, Any]] = {}
        for r in app_records:
            action = r["action_type"]
            if action not in action_breakdown:
                action_breakdown[action] = {"count": 0, "successes": 0, "durations": []}
            action_breakdown[action]["count"] += 1
            if r["success"]:
                action_breakdown[action]["successes"] += 1
                action_breakdown[action]["durations"].append(r["duration"])

        for action, data in action_breakdown.items():
            data["success_rate"] = (
                round(data["successes"] / data["count"], 3) if data["count"] > 0 else 0.0
            )
            if data["durations"]:
                data["avg_duration"] = round(
                    sum(data["durations"]) / len(data["durations"]), 3
                )
            del data["durations"]  # Don't expose raw list in stats

        stats["action_breakdown"] = action_breakdown
        return stats

    def optimize(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Stage 5: OPTIMIZE -- Apply performance learning to task steps.

        Uses historical timing data to set optimal wait times, batches
        sequential same-app actions, and eliminates redundant screenshots.
        """
        logger.info("STAGE 5 [OPTIMIZE] -- Applying performance learning")

        steps: list[dict[str, Any]] = task.get("steps", [])
        app_name = task.get("app_profile", task.get("app", "unknown"))

        # --- Apply learned timing ---
        for step in steps:
            action = step.get("action", "")
            learned_delay = self.get_optimal_delay(action, app_name)

            if learned_delay is not None:
                original = step.get("state_expectation", {}).get("timeout", 3.0)
                optimized = max(learned_delay, 0.5)  # never below 0.5s
                step.setdefault("state_expectation", {})["timeout"] = optimized
                step["_optimization"] = {
                    "original_timeout": original,
                    "learned_timeout": optimized,
                    "data_source": "historical_p90",
                }
                logger.debug(
                    "  Optimized %s timeout: %.2fs -> %.2fs (learned)",
                    action,
                    original,
                    optimized,
                )
            else:
                step["_optimization"] = {"data_source": "default", "note": "Insufficient data"}

        # --- Batch optimization: group same-app sequential actions ---
        task["batch_groups"] = self._compute_batch_groups(steps)

        # --- Screenshot deduplication ---
        self._deduplicate_screenshots(steps)

        task["_optimized"] = True
        return task

    @staticmethod
    def _compute_batch_groups(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Group sequential steps targeting the same app into batches.

        Batched steps can share context and skip app-switch overhead.
        """
        groups: list[dict[str, Any]] = []
        current_group: list[int] = []
        current_app: Optional[str] = None

        for i, step in enumerate(steps):
            step_app = step.get("target_app", step.get("app", None))

            if step_app == current_app and current_app is not None:
                current_group.append(i)
            else:
                if current_group:
                    groups.append({
                        "app": current_app,
                        "step_indices": list(current_group),
                        "count": len(current_group),
                        "can_skip_app_switch": len(current_group) > 1,
                    })
                current_group = [i]
                current_app = step_app

        # Flush last group
        if current_group:
            groups.append({
                "app": current_app,
                "step_indices": list(current_group),
                "count": len(current_group),
                "can_skip_app_switch": len(current_group) > 1,
            })

        return groups

    @staticmethod
    def _deduplicate_screenshots(steps: list[dict[str, Any]]) -> None:
        """
        Mark redundant screenshot/verification steps that can be skipped
        when the previous screenshot was taken less than 2 seconds ago.
        """
        last_screenshot_time: Optional[float] = None
        screenshot_threshold = 2.0  # seconds

        for step in steps:
            verification = step.get("state_expectation", {}).get("verification_method")

            if verification == VerificationMethod.VISION_CHECK.value:
                step_timeout = step.get("state_expectation", {}).get("timeout", 3.0)

                if last_screenshot_time is not None:
                    elapsed = step_timeout  # estimated time since last
                    if elapsed < screenshot_threshold:
                        step["_skip_screenshot"] = True
                        step["_skip_reason"] = (
                            f"Previous screenshot <{screenshot_threshold}s ago"
                        )

                last_screenshot_time = step_timeout


# ---------------------------------------------------------------------------
# Stage 6: VALIDATE
# ---------------------------------------------------------------------------

class ValidateStage:
    """Pre-execution safety gate for irreversible actions."""

    # Preflight check definitions per action type
    PREFLIGHT_CHECKS: dict[str, list[dict[str, Any]]] = {
        ActionType.POST_CONTENT: [
            {"name": "content_not_empty", "field": "content", "check": "not_empty"},
            {"name": "correct_account", "field": "account", "check": "confirmed"},
            {"name": "correct_platform", "field": "platform", "check": "confirmed"},
            {"name": "draft_preview", "field": "preview_url", "check": "exists"},
        ],
        ActionType.SEND_MESSAGE: [
            {"name": "recipient_confirmed", "field": "recipient", "check": "not_empty"},
            {"name": "content_reviewed", "field": "message", "check": "not_empty"},
            {"name": "not_duplicate", "field": "message_hash", "check": "unique"},
        ],
        ActionType.MAKE_PURCHASE: [
            {"name": "amount_confirmed", "field": "amount", "check": "positive_number"},
            {"name": "payment_method_correct", "field": "payment_method", "check": "confirmed"},
            {"name": "user_approval", "field": "user_approved", "check": "is_true",
             "blocking": True},
        ],
        ActionType.DELETE_CONTENT: [
            {"name": "confirmation_required", "field": "confirmed", "check": "is_true",
             "blocking": True},
            {"name": "backup_available", "field": "backup_path", "check": "exists"},
        ],
        ActionType.ACCOUNT_SETTINGS: [
            {"name": "change_description_clear", "field": "description", "check": "not_empty"},
            {"name": "reversibility_noted", "field": "reversible", "check": "documented"},
        ],
    }

    def validate(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Stage 6: VALIDATE -- Pre-execution safety gating.

        Runs preflight checks on steps that perform irreversible actions.
        Blocks execution if any blocking check fails.
        """
        logger.info("STAGE 6 [VALIDATE] -- Pre-execution safety checks")

        steps: list[dict[str, Any]] = task.get("steps", [])
        task_validation = ValidationResult(valid=True)

        for step in steps:
            action = step.get("action", "")
            check_defs = self.PREFLIGHT_CHECKS.get(action)

            if check_defs is None:
                # No special validation needed for this action type
                step["validation"] = {"valid": True, "checks": [], "skipped": True}
                continue

            step_result = self._run_checks(step, check_defs)
            step["validation"] = {
                "valid": step_result.valid,
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "message": c.message,
                        "blocking": c.blocking,
                    }
                    for c in step_result.checks
                ],
                "blocking_failures": step_result.blocking_failures,
                "warnings": step_result.warnings,
            }

            # Propagate to task-level validation
            if not step_result.valid:
                task_validation.valid = False
            task_validation.blocking_failures.extend(step_result.blocking_failures)
            task_validation.warnings.extend(step_result.warnings)

        task["validation_summary"] = {
            "valid": task_validation.valid,
            "total_blocking_failures": len(task_validation.blocking_failures),
            "blocking_failures": task_validation.blocking_failures,
            "total_warnings": len(task_validation.warnings),
            "warnings": task_validation.warnings,
        }
        task["_validated"] = True

        if not task_validation.valid:
            logger.warning(
                "  VALIDATION FAILED: %d blocking issue(s)",
                len(task_validation.blocking_failures),
            )
        else:
            logger.info("  Validation passed")

        return task

    def _run_checks(
        self, step: dict[str, Any], check_defs: list[dict[str, Any]]
    ) -> ValidationResult:
        """Execute a list of preflight check definitions against a step."""
        result = ValidationResult(valid=True)

        for check_def in check_defs:
            name = check_def["name"]
            field_name = check_def["field"]
            check_type = check_def["check"]
            is_blocking = check_def.get("blocking", False)

            field_value = step.get(field_name)
            passed, message = self._evaluate_check(check_type, field_name, field_value)

            check = PreflightCheck(
                name=name,
                passed=passed,
                message=message,
                blocking=is_blocking,
            )
            result.checks.append(check)

            if not passed:
                if is_blocking:
                    result.valid = False
                    result.blocking_failures.append(f"{name}: {message}")
                else:
                    result.warnings.append(f"{name}: {message}")

        return result

    @staticmethod
    def _evaluate_check(
        check_type: str, field_name: str, value: Any
    ) -> tuple[bool, str]:
        """Evaluate a single check condition and return (passed, message)."""

        if check_type == "not_empty":
            if value is not None and str(value).strip():
                return True, f"'{field_name}' is present"
            return False, f"'{field_name}' is empty or missing"

        elif check_type == "confirmed":
            if value is not None and str(value).strip():
                return True, f"'{field_name}' is set to '{value}'"
            return False, f"'{field_name}' needs confirmation"

        elif check_type == "exists":
            if value is not None:
                return True, f"'{field_name}' exists"
            return False, f"'{field_name}' not found"

        elif check_type == "is_true":
            if value is True:
                return True, f"'{field_name}' is confirmed (True)"
            return False, f"'{field_name}' is not True — explicit approval required"

        elif check_type == "positive_number":
            try:
                num = float(value) if value is not None else 0
                if num > 0:
                    return True, f"'{field_name}' = {num}"
                return False, f"'{field_name}' must be a positive number, got {value}"
            except (TypeError, ValueError):
                return False, f"'{field_name}' is not a valid number: {value}"

        elif check_type == "unique":
            # In a real implementation, this would check against a dedup store.
            # For now, presence of the hash is treated as a pass.
            if value is not None:
                return True, f"'{field_name}' hash present for dedup check"
            return False, f"'{field_name}' missing — cannot verify uniqueness"

        elif check_type == "documented":
            if value is not None:
                return True, f"'{field_name}' is documented"
            return False, f"'{field_name}' should be documented for safety"

        else:
            return False, f"Unknown check type: '{check_type}'"


# ---------------------------------------------------------------------------
# Full Pipeline Orchestrator
# ---------------------------------------------------------------------------

class AmplifyPipeline:
    """
    AMPLIFY: the 6-stage Android automation enhancement pipeline.

    Processes every task through ENRICH -> EXPAND -> FORTIFY ->
    ANTICIPATE -> OPTIMIZE -> VALIDATE before execution.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or DATA_DIR
        self._enrich = EnrichStage()
        self._expand = ExpandStage()
        self._fortify = FortifyStage()
        self._anticipate = AnticipateStage()
        self._optimize = OptimizeStage(data_dir=self._data_dir)
        self._validate = ValidateStage()

        # Ensure data directory exists
        self._data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "AmplifyPipeline initialized (data_dir=%s)", self._data_dir
        )

    # -- Public API --

    def full_pipeline(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Run all 6 AMPLIFY stages on a task configuration.

        Args:
            task: Raw task dictionary. Must contain at minimum:
                - "app" or "target_app": the Android app to automate
                - "steps": list of action step dicts, each with "action" key

        Returns:
            Enhanced task dictionary with enriched context, edge-case guards,
            retry policies, state expectations, optimized timing, and
            validation results attached to each step.

        Raises:
            ValueError: If the task is missing required fields.
        """
        logger.info("=" * 60)
        logger.info("AMPLIFY PIPELINE -- Processing task")
        logger.info("=" * 60)

        # Basic validation of input
        self._validate_input(task)

        # Stamp metadata
        task["_amplify"] = {
            "version": "1.0.0",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stages_completed": [],
        }

        # Stage 1: ENRICH
        task = self._enrich.enrich(task)
        task["_amplify"]["stages_completed"].append("ENRICH")

        # Stage 2: EXPAND
        task = self._expand.expand(task)
        task["_amplify"]["stages_completed"].append("EXPAND")

        # Stage 3: FORTIFY
        task = self._fortify.fortify(task)
        task["_amplify"]["stages_completed"].append("FORTIFY")

        # Stage 4: ANTICIPATE
        task = self._anticipate.anticipate(task)
        task["_amplify"]["stages_completed"].append("ANTICIPATE")

        # Stage 5: OPTIMIZE
        task = self._optimize.optimize(task)
        task["_amplify"]["stages_completed"].append("OPTIMIZE")

        # Stage 6: VALIDATE
        task = self._validate.validate(task)
        task["_amplify"]["stages_completed"].append("VALIDATE")

        task["_amplify"]["finished_at"] = datetime.now(timezone.utc).isoformat()
        task["_amplify"]["fully_processed"] = True

        # Summary log
        validation = task.get("validation_summary", {})
        logger.info("=" * 60)
        logger.info(
            "AMPLIFY COMPLETE -- valid=%s, warnings=%d, blocking=%d",
            validation.get("valid", "N/A"),
            validation.get("total_warnings", 0),
            validation.get("total_blocking_failures", 0),
        )
        logger.info("=" * 60)

        return task

    def record_execution(
        self,
        action_type: str,
        app_name: str,
        duration: float,
        success: bool,
    ) -> None:
        """
        Record an execution timing observation for the OPTIMIZE stage.

        Call this after each step executes to feed the learning loop.
        """
        record = TimingRecord(
            action_type=action_type,
            app_name=app_name,
            duration=duration,
            success=success,
        )
        self._optimize.record_timing(record)

    def get_app_stats(self, app_name: str) -> dict[str, Any]:
        """Get aggregated performance stats for an app."""
        return self._optimize.get_app_performance_stats(app_name)

    def get_optimal_delay(self, action_type: str, app_name: str) -> Optional[float]:
        """Get the learned optimal delay for an action+app pair."""
        return self._optimize.get_optimal_delay(action_type, app_name)

    # -- Internal --

    @staticmethod
    def _validate_input(task: dict[str, Any]) -> None:
        """Ensure the task has the minimum required structure."""
        if not isinstance(task, dict):
            raise ValueError(f"Task must be a dict, got {type(task).__name__}")

        app = task.get("app") or task.get("target_app") or task.get("app_name")
        if not app:
            raise ValueError(
                "Task must specify an app via 'app', 'target_app', or 'app_name' key"
            )

        steps = task.get("steps")
        if steps is None:
            # Auto-create empty steps list so stages don't error
            task["steps"] = []
            logger.warning("Task has no 'steps' key; initialized to empty list")
        elif not isinstance(steps, list):
            raise ValueError(f"'steps' must be a list, got {type(steps).__name__}")


# ---------------------------------------------------------------------------
# Convenience: standalone stage runners
# ---------------------------------------------------------------------------

def enrich(task: dict[str, Any]) -> dict[str, Any]:
    """Run only Stage 1 (ENRICH) on a task."""
    return EnrichStage().enrich(task)


def expand(task: dict[str, Any]) -> dict[str, Any]:
    """Run only Stage 2 (EXPAND) on a task."""
    return ExpandStage().expand(task)


def fortify(task: dict[str, Any]) -> dict[str, Any]:
    """Run only Stage 3 (FORTIFY) on a task."""
    return FortifyStage().fortify(task)


def anticipate(task: dict[str, Any]) -> dict[str, Any]:
    """Run only Stage 4 (ANTICIPATE) on a task."""
    return AnticipateStage().anticipate(task)


def optimize(task: dict[str, Any]) -> dict[str, Any]:
    """Run only Stage 5 (OPTIMIZE) on a task."""
    return OptimizeStage().optimize(task)


def validate(task: dict[str, Any]) -> dict[str, Any]:
    """Run only Stage 6 (VALIDATE) on a task."""
    return ValidateStage().validate(task)


# ---------------------------------------------------------------------------
# CLI entry point (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Example task: post a Pin to Pinterest via the Android app
    example_task: dict[str, Any] = {
        "app": "pinterest",
        "task_description": "Create and publish a new pin for WitchcraftForBeginners",
        "steps": [
            {
                "action": ActionType.LAUNCH_APP,
                "target_app": "pinterest",
            },
            {
                "action": ActionType.TAP_ELEMENT,
                "target": "create_button",
                "element": "com.pinterest:id/create_pin_fab",
            },
            {
                "action": ActionType.TAP_ELEMENT,
                "target": "pin_option",
                "element": "Create Pin",
            },
            {
                "action": ActionType.TAP_ELEMENT,
                "target": "add_image",
                "element": "com.pinterest:id/add_image_button",
            },
            {
                "action": ActionType.TAP_ELEMENT,
                "target": "select_photo",
                "element": "gallery_image_0",
            },
            {
                "action": ActionType.TYPE_TEXT,
                "target": "title_field",
                "element": "com.pinterest:id/pin_title",
                "text": "Full Moon Protection Ritual for Beginners",
            },
            {
                "action": ActionType.TYPE_TEXT,
                "target": "description_field",
                "element": "com.pinterest:id/pin_description",
                "text": (
                    "Learn a simple yet powerful full moon protection ritual "
                    "perfect for beginner witches. Step-by-step guide with "
                    "supplies list and timing tips."
                ),
            },
            {
                "action": ActionType.TYPE_TEXT,
                "target": "link_field",
                "element": "com.pinterest:id/pin_link",
                "text": "https://witchcraftforbeginners.com/full-moon-protection-ritual/",
            },
            {
                "action": ActionType.TAP_ELEMENT,
                "target": "select_board",
                "element": "com.pinterest:id/board_selector",
            },
            {
                "action": ActionType.TAP_ELEMENT,
                "target": "board_name",
                "element": "Moon Rituals",
            },
            {
                "action": ActionType.POST_CONTENT,
                "target": "publish_button",
                "element": "com.pinterest:id/publish_button",
                "content": "Full Moon Protection Ritual for Beginners",
                "account": "witchcraftforbeginners",
                "platform": "pinterest",
            },
        ],
    }

    pipeline = AmplifyPipeline()
    enhanced = pipeline.full_pipeline(example_task)

    # Pretty print summary
    print("\n" + "=" * 60)
    print("ENHANCED TASK SUMMARY")
    print("=" * 60)
    print(f"App Profile: {enhanced.get('app_profile', 'N/A')}")
    print(f"Package: {enhanced.get('package_name', 'N/A')}")
    print(f"Category: {enhanced.get('app_category', 'N/A')}")
    print(f"Load Time: {enhanced.get('typical_load_time', 'N/A')}s")
    print(f"Steps: {len(enhanced.get('steps', []))}")
    print(f"Batch Groups: {len(enhanced.get('batch_groups', []))}")

    validation = enhanced.get("validation_summary", {})
    print(f"Valid: {validation.get('valid', 'N/A')}")
    print(f"Blocking Failures: {validation.get('total_blocking_failures', 0)}")
    print(f"Warnings: {validation.get('total_warnings', 0)}")

    amplify_meta = enhanced.get("_amplify", {})
    print(f"Stages: {', '.join(amplify_meta.get('stages_completed', []))}")
    print(f"Started: {amplify_meta.get('started_at', 'N/A')}")
    print(f"Finished: {amplify_meta.get('finished_at', 'N/A')}")

    if "--json" in sys.argv:
        print("\n" + json.dumps(enhanced, indent=2, default=str))
