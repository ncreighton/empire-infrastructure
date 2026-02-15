"""
Social Automation — In-App Social Media Workflows (OpenClaw Empire)
===================================================================

Pre-built automation workflows for social media tasks that REQUIRE in-app
interaction (no API available or API is restricted). Uses phone_controller.py
for ADB commands and vision_agent.py for screen analysis.

Complements social_publisher.py (API-based posting) with on-device automation
for engagement, analytics scraping, story/reel posting, DMs, and growth tasks.

Supported platforms:
    Instagram  (com.instagram.android)
    TikTok     (com.zhiliaoapp.musically)
    Pinterest  (com.pinterest)
    Facebook   (com.facebook.katana)
    Twitter/X  (com.twitter.android)
    LinkedIn   (com.linkedin.android)

Architecture:
    BasePlatformBot  -- shared open/login/navigate/popup handling
        InstagramBot -- photo, story, reel, engagement, DMs, insights
        TikTokBot    -- upload, live, FYP engagement, analytics
        PinterestBot -- pin creation, idea pins, boards, analytics
        FacebookBot  -- posts, groups, messenger, page insights
        TwitterBot   -- tweets, replies, engagement, analytics
        LinkedInBot  -- posts, articles, connections, messaging

    HumanBehavior   -- anti-detection delays and randomization
    ActionLimiter   -- per-platform daily action limits
    SocialCampaign  -- multi-step campaign orchestration
    AnalyticsScraper -- OCR-based analytics extraction

Data storage: data/social_automation/
    limits.json      -- daily action counts (reset at midnight)
    campaigns.json   -- campaign definitions and results
    analytics.json   -- scraped analytics snapshots
    growth.json      -- daily growth tracking

Usage:
    from src.social_automation import get_social_bot
    bot = get_social_bot()
    await bot.instagram.post_photo("/tmp/img.png", "Hello!", ["witchcraft"])
    await bot.pinterest.create_pin("/tmp/pin.png", "Moon Water", "Guide...", "https://...")

CLI:
    python -m src.social_automation instagram post-photo --image /tmp/img.png --caption "Hello"
    python -m src.social_automation pinterest create-pin --image /tmp/pin.png --title "Moon"
    python -m src.social_automation campaign create --name "Launch" --platform instagram
    python -m src.social_automation limits show
    python -m src.social_automation analytics scrape --platform instagram
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("social_automation")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "social_automation"
LIMITS_FILE = DATA_DIR / "limits.json"
CAMPAIGNS_FILE = DATA_DIR / "campaigns.json"
ANALYTICS_FILE = DATA_DIR / "analytics.json"
GROWTH_FILE = DATA_DIR / "growth.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_CAMPAIGNS = 1000
MAX_ANALYTICS = 5000
MAX_GROWTH = 3000

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PlatformName(str, Enum):
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    PINTEREST = "pinterest"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"


class CampaignStatus(str, Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class ActionCategory(str, Enum):
    POST = "post"
    LIKE = "like"
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"
    COMMENT = "comment"
    DM = "dm"
    SHARE = "share"
    VIEW = "view"
    SAVE = "save"
    STORY = "story"
    REEL = "reel"


# ---------------------------------------------------------------------------
# JSON Persistence (atomic via temp file + os.replace)
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default if default is not None else []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default if default is not None else []


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
        os.replace(str(tmp), str(path))
    except OSError as exc:
        logger.error("Failed to save %s: %s", path, exc)
        if tmp.exists():
            tmp.unlink()


def _bounded_append(path: Path, entries: List[Dict[str, Any]], cap: int) -> None:
    existing = _load_json(path, default=[])
    if not isinstance(existing, list):
        existing = []
    existing.extend(entries)
    if len(existing) > cap:
        existing = existing[-cap:]
    _save_json(path, existing)


def _run_sync(coro):
    """Run async coroutine synchronously, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ActionRecord:
    """A single recorded automation action."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    platform: str = ""
    category: str = ""
    description: str = ""
    success: bool = False
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class AnalyticsSnapshot:
    """A point-in-time analytics capture from a platform."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    platform: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    screenshot_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnalyticsSnapshot:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class CampaignAction:
    """A single step in a social campaign."""
    action_type: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    executed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CampaignAction:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SocialCampaign:
    """A multi-step social media campaign."""
    campaign_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    platform: str = ""
    actions: List[CampaignAction] = field(default_factory=list)
    schedule: Optional[str] = None
    status: str = CampaignStatus.DRAFT.value
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id, "name": self.name,
            "platform": self.platform,
            "actions": [a.to_dict() for a in self.actions],
            "schedule": self.schedule, "status": self.status,
            "results": self.results, "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SocialCampaign:
        actions_raw = data.pop("actions", [])
        actions = [CampaignAction.from_dict(a) for a in actions_raw]
        known = {f.name for f in cls.__dataclass_fields__.values()}
        scalars = {k: v for k, v in data.items() if k in known and k != "actions"}
        return cls(actions=actions, **scalars)

    def summary(self) -> str:
        done = sum(1 for a in self.actions if a.status == "completed")
        return (f"Campaign {self.campaign_id} | {self.name} | "
                f"{self.platform} | {done}/{len(self.actions)} actions | {self.status}")


# ---------------------------------------------------------------------------
# HumanBehavior — Anti-detection delays and randomization
# ---------------------------------------------------------------------------


class HumanBehavior:
    """
    Provides human-like randomization for automation actions to avoid
    detection by social media platform anti-bot systems.

    All delays are randomized within a configurable range. Typing is
    simulated character-by-character with random inter-key delays.
    Session and daily limits prevent excessive activity.
    """

    def __init__(self) -> None:
        self.session_limits: Dict[str, int] = {
            "instagram": 60,
            "tiktok": 50,
            "pinterest": 80,
            "facebook": 50,
            "twitter": 60,
            "linkedin": 40,
        }
        self.daily_limits: Dict[str, Dict[str, int]] = {
            "instagram": {"likes": 100, "follows": 50, "comments": 30,
                          "dms": 20, "posts": 10, "stories": 15, "reels": 5},
            "tiktok": {"likes": 80, "follows": 40, "comments": 20,
                       "posts": 5, "views": 200},
            "pinterest": {"pins": 50, "saves": 60, "follows": 30,
                          "boards": 5},
            "facebook": {"posts": 10, "likes": 80, "comments": 20,
                         "shares": 15, "messages": 25},
            "twitter": {"tweets": 30, "likes": 100, "retweets": 40,
                        "follows": 40, "dms": 20},
            "linkedin": {"posts": 5, "likes": 50, "comments": 15,
                         "connections": 25, "messages": 20},
        }
        self.rest_periods: Dict[str, Tuple[int, int]] = {
            "between_actions": (2, 5),
            "between_bursts": (30, 90),
            "burst_size": (5, 12),
            "session_break": (300, 600),
        }
        self._action_count: int = 0
        self._burst_count: int = 0
        self._burst_target: int = random.randint(5, 12)

    async def random_delay(self, min_s: float = 1.0, max_s: float = 3.0) -> float:
        """Wait a randomized human-like duration. Returns actual seconds waited."""
        delay = random.uniform(min_s, max_s)
        await asyncio.sleep(delay)
        return delay

    async def random_scroll_speed(self) -> int:
        """Return a randomized scroll distance in pixels."""
        return random.randint(300, 800)

    async def typing_delay(self, text: str, controller: Any) -> None:
        """Type text character-by-character with random inter-key gaps."""
        for char in text:
            await controller.type_text(char)
            gap = random.uniform(0.05, 0.25)
            await asyncio.sleep(gap)

    async def action_cooldown(self) -> None:
        """Apply appropriate delay based on current action/burst counts."""
        self._action_count += 1
        self._burst_count += 1

        if self._burst_count >= self._burst_target:
            lo, hi = self.rest_periods["between_bursts"]
            wait = random.uniform(lo, hi)
            logger.info("Burst rest: %.1fs after %d actions", wait, self._burst_count)
            await asyncio.sleep(wait)
            self._burst_count = 0
            self._burst_target = random.randint(*self.rest_periods["burst_size"])
        else:
            lo, hi = self.rest_periods["between_actions"]
            await asyncio.sleep(random.uniform(lo, hi))

    def reset_session(self) -> None:
        """Reset session counters."""
        self._action_count = 0
        self._burst_count = 0
        self._burst_target = random.randint(5, 12)


# ---------------------------------------------------------------------------
# ActionLimiter — Per-platform daily action limits
# ---------------------------------------------------------------------------


class ActionLimiter:
    """
    Tracks actions per platform per day. Warns at 80% of limit,
    hard stop at 100%. Counts persist to JSON and reset daily.
    """

    def __init__(self, limits: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        self._limits = limits or HumanBehavior().daily_limits
        self._counts: Dict[str, Dict[str, int]] = {}
        self._date: str = ""
        self._load()

    def _load(self) -> None:
        data = _load_json(LIMITS_FILE, default={})
        if isinstance(data, dict):
            stored_date = data.get("date", "")
            today = date.today().isoformat()
            if stored_date == today:
                self._counts = data.get("counts", {})
                self._date = stored_date
            else:
                self._counts = {}
                self._date = today
        else:
            self._counts = {}
            self._date = date.today().isoformat()

    def _save(self) -> None:
        _save_json(LIMITS_FILE, {"date": self._date, "counts": self._counts})

    def _ensure_today(self) -> None:
        today = date.today().isoformat()
        if self._date != today:
            self._counts = {}
            self._date = today

    def check(self, platform: str, action: str) -> Tuple[bool, str]:
        """
        Check if an action is allowed. Returns (allowed, message).
        Messages indicate warnings at 80% or blocks at 100%.
        """
        self._ensure_today()
        platform = platform.lower()
        action = action.lower()
        limit = self._limits.get(platform, {}).get(action, 999)
        current = self._counts.get(platform, {}).get(action, 0)

        if current >= limit:
            return False, f"BLOCKED: {platform}/{action} at {current}/{limit} daily limit"

        threshold = int(limit * 0.8)
        if current >= threshold:
            return True, f"WARNING: {platform}/{action} at {current}/{limit} (80% threshold)"

        return True, f"OK: {platform}/{action} at {current}/{limit}"

    def record(self, platform: str, action: str) -> None:
        """Record an action execution. Increments daily counter."""
        self._ensure_today()
        platform = platform.lower()
        action = action.lower()
        if platform not in self._counts:
            self._counts[platform] = {}
        self._counts[platform][action] = self._counts[platform].get(action, 0) + 1
        self._save()

    def get_counts(self, platform: Optional[str] = None) -> Dict[str, Any]:
        """Get current daily counts, optionally filtered by platform."""
        self._ensure_today()
        if platform:
            return {platform: self._counts.get(platform.lower(), {})}
        return dict(self._counts)

    def get_remaining(self, platform: str, action: str) -> int:
        """Get remaining allowed actions for today."""
        self._ensure_today()
        limit = self._limits.get(platform.lower(), {}).get(action.lower(), 999)
        current = self._counts.get(platform.lower(), {}).get(action.lower(), 0)
        return max(0, limit - current)

    def get_all_limits(self) -> Dict[str, Dict[str, int]]:
        """Return the full limits configuration."""
        return dict(self._limits)

    def set_limit(self, platform: str, action: str, limit: int) -> None:
        """Override a specific limit."""
        platform = platform.lower()
        action = action.lower()
        if platform not in self._limits:
            self._limits[platform] = {}
        self._limits[platform][action] = limit

    def summary(self) -> str:
        """Return a formatted summary of today's usage."""
        self._ensure_today()
        lines = [f"Action Limits ({self._date})"]
        lines.append("-" * 50)
        for platform in sorted(self._limits.keys()):
            plat_counts = self._counts.get(platform, {})
            plat_limits = self._limits[platform]
            for action in sorted(plat_limits.keys()):
                current = plat_counts.get(action, 0)
                limit = plat_limits[action]
                pct = int(current / limit * 100) if limit else 0
                bar = "#" * (pct // 5) + "." * (20 - pct // 5)
                flag = " !!!" if pct >= 100 else " !" if pct >= 80 else ""
                lines.append(f"  {platform:12s} {action:12s} [{bar}] {current:3d}/{limit:3d}{flag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BasePlatformBot — Shared automation primitives
# ---------------------------------------------------------------------------


class BasePlatformBot:
    """
    Base class for all platform-specific automation bots. Provides shared
    functionality: app launching, login flow, popup handling, navigation,
    screen identification, and human-like cooldowns.

    Subclasses must set ``package_name`` and ``app_label``.
    """

    package_name: str = ""
    app_label: str = ""

    def __init__(
        self,
        controller: Any,
        vision: Any,
        behavior: HumanBehavior,
        limiter: ActionLimiter,
    ) -> None:
        self.controller = controller
        self.vision = vision
        self.behavior = behavior
        self.limiter = limiter
        self._logged_in: bool = False
        self._current_screen: str = ""

    # -- App lifecycle -------------------------------------------------------

    async def open_app(self) -> bool:
        """Launch the app, wait for it to load, and handle initial dialogs."""
        logger.info("Opening %s (%s)", self.app_label, self.package_name)
        result = await self.controller.launch_app(self.package_name)
        if not result.success:
            logger.error("Failed to launch %s: %s", self.app_label, result.error)
            return False
        await self.cool_down(3.0)
        await self.handle_popup()
        self._current_screen = await self.get_current_screen()
        logger.info("%s opened on screen: %s", self.app_label, self._current_screen)
        return True

    async def login(self, username: str, password: str) -> bool:
        """
        Perform login flow. If 2FA is required, pauses for user input.
        Returns True if login succeeds.
        """
        logger.info("Logging into %s as %s", self.app_label, username)
        screen = await self.get_current_screen()
        if "home" in screen.lower() or "feed" in screen.lower():
            logger.info("Already logged in to %s", self.app_label)
            self._logged_in = True
            return True

        # Try to find and tap login fields
        username_el = await self.controller.find_element(
            text="username", partial_match=True
        ) or await self.controller.find_element(
            text="email", partial_match=True
        ) or await self.controller.find_element(
            text="phone", partial_match=True
        )

        if username_el:
            await self.controller.tap_element(username_el)
            await self.cool_down(0.5)
            await self.controller.type_text(username)
            await self.cool_down(0.5)

        password_el = await self.controller.find_element(
            text="password", partial_match=True
        )
        if password_el:
            await self.controller.tap_element(password_el)
            await self.cool_down(0.5)
            await self.controller.type_text(password)
            await self.cool_down(0.5)

        # Tap login/sign in button
        login_btn = await self.controller.find_element(
            text="log in", partial_match=True
        ) or await self.controller.find_element(
            text="sign in", partial_match=True
        )
        if login_btn:
            await self.controller.tap_element(login_btn)
            await self.cool_down(5.0)

        # Check for 2FA
        twofa = await self.controller.find_element(
            text="verification", partial_match=True
        ) or await self.controller.find_element(
            text="two-factor", partial_match=True
        ) or await self.controller.find_element(
            text="confirm", partial_match=True
        )
        if twofa:
            logger.warning("2FA detected for %s -- waiting 60s for user input",
                           self.app_label)
            await asyncio.sleep(60)

        self._logged_in = await self.verify_logged_in()
        return self._logged_in

    async def navigate_to(self, destination: str) -> bool:
        """Navigate to a specific screen within the app using vision."""
        logger.info("Navigating to '%s' in %s", destination, self.app_label)
        for attempt in range(5):
            screen = await self.get_current_screen()
            if destination.lower() in screen.lower():
                return True
            decision = await self.vision.decide_action(
                f"Navigate to the {destination} screen in {self.app_label}",
                context=f"Currently on: {screen}",
            )
            if decision and decision.get("action_type") != "done":
                await self.vision._execute_vision_action(
                    decision["action_type"], decision.get("params", {})
                )
                await self.cool_down(2.0)
            elif decision and decision.get("action_type") == "done":
                return True
            else:
                break
        return False

    async def handle_popup(self) -> bool:
        """Dismiss common popups: notifications, updates, cookies, etc."""
        dismiss_texts = [
            "not now", "skip", "dismiss", "close", "no thanks",
            "maybe later", "cancel", "got it", "ok", "allow",
            "don't allow", "deny", "i agree", "accept",
        ]
        dismissed = False
        for text in dismiss_texts:
            el = await self.controller.find_element(text=text, partial_match=True)
            if el and el.clickable:
                logger.info("Dismissing popup: '%s'", text)
                await self.controller.tap_element(el)
                await self.cool_down(1.5)
                dismissed = True
                break
        return dismissed

    async def verify_logged_in(self) -> bool:
        """Check if currently authenticated by analyzing the screen."""
        screen = await self.get_current_screen()
        logged_out_indicators = ["login", "sign in", "sign up", "create account",
                                 "log in", "register"]
        for indicator in logged_out_indicators:
            if indicator in screen.lower():
                return False
        return True

    async def get_current_screen(self) -> str:
        """Identify the current screen via vision analysis."""
        try:
            screenshot_path = await self.controller.screenshot()
            analysis = await self.vision.analyze_screen(
                f"What screen of {self.app_label} is currently shown? "
                "Return a short label like: home, feed, profile, explore, "
                "create_post, story, reel, settings, login, dm, search, "
                "notifications, analytics.",
                screenshot_path=screenshot_path,
            )
            self._current_screen = analysis.description.strip().lower()
            return self._current_screen
        except Exception as exc:
            logger.warning("Screen identification failed: %s", exc)
            return "unknown"

    async def safe_back(self) -> bool:
        """Navigate back with verification that the screen changed."""
        before = await self.get_current_screen()
        await self.controller.press_back()
        await self.cool_down(1.5)
        after = await self.get_current_screen()
        return before != after

    async def cool_down(self, seconds: float) -> None:
        """Human-like random delay: 0.8x to 1.5x of specified seconds."""
        actual = seconds * random.uniform(0.8, 1.5)
        await asyncio.sleep(actual)

    # -- Helpers -------------------------------------------------------------

    async def _check_limit(self, action: str) -> bool:
        """Check limiter and log. Returns True if action is allowed."""
        platform = self.package_name_to_platform()
        allowed, msg = self.limiter.check(platform, action)
        if not allowed:
            logger.warning(msg)
        elif "WARNING" in msg:
            logger.info(msg)
        return allowed

    async def _record_action(self, action: str) -> None:
        """Record an action to the limiter."""
        platform = self.package_name_to_platform()
        self.limiter.record(platform, action)

    def package_name_to_platform(self) -> str:
        """Map package name to platform string."""
        mapping = {
            "com.instagram.android": "instagram",
            "com.zhiliaoapp.musically": "tiktok",
            "com.pinterest": "pinterest",
            "com.facebook.katana": "facebook",
            "com.twitter.android": "twitter",
            "com.linkedin.android": "linkedin",
        }
        return mapping.get(self.package_name, "unknown")

    async def _vision_guided_flow(self, goal: str, max_steps: int = 15) -> bool:
        """Run a vision-guided automation flow until goal is achieved."""
        for step in range(max_steps):
            decision = await self.vision.decide_action(goal)
            if decision is None:
                logger.warning("Vision returned no action at step %d", step)
                return False
            if decision.get("action_type") == "done":
                logger.info("Goal achieved: %s", goal)
                return True
            await self.vision._execute_vision_action(
                decision["action_type"], decision.get("params", {})
            )
            await self.behavior.action_cooldown()
        logger.warning("Vision flow did not complete in %d steps: %s", max_steps, goal)
        return False

    async def _find_and_tap(self, text: str, timeout: float = 10.0) -> bool:
        """Find a UI element by text and tap it."""
        result = await self.controller.find_and_tap(text=text, timeout=timeout)
        return result.success

    async def _type_in_field(self, field_text: str, value: str) -> bool:
        """Find a text field, tap it, and type a value."""
        el = await self.controller.find_element(text=field_text, partial_match=True)
        if el:
            await self.controller.tap_element(el)
            await self.cool_down(0.5)
            await self.controller.type_text(value)
            return True
        return False

    async def _scroll_and_find(self, text: str, max_scrolls: int = 5) -> bool:
        """Scroll down repeatedly trying to find an element."""
        for _ in range(max_scrolls):
            el = await self.controller.find_element(text=text, partial_match=True)
            if el:
                await self.controller.tap_element(el)
                return True
            await self.controller.scroll_down()
            await self.cool_down(1.0)
        return False


# ---------------------------------------------------------------------------
# InstagramBot
# ---------------------------------------------------------------------------


class InstagramBot(BasePlatformBot):
    """
    Instagram automation via in-app interaction.

    Handles: photo posting, stories, reels, engagement (likes, follows,
    comments), DMs, analytics scraping, and post saving.
    """

    package_name = "com.instagram.android"
    app_label = "Instagram"

    async def post_photo(
        self,
        image_path: str,
        caption: str,
        hashtags: Optional[List[str]] = None,
        location: Optional[str] = None,
    ) -> ActionRecord:
        """Post a photo with caption, hashtags, and optional location tag."""
        record = ActionRecord(platform="instagram", category=ActionCategory.POST.value,
                              description=f"Post photo: {caption[:50]}...")
        start = time.monotonic()

        if not await self._check_limit("posts"):
            record.error = "Daily post limit reached"
            return record

        try:
            await self.open_app()
            await self.cool_down(1.0)

            # Tap the create/new post button (center bottom +)
            if not await self._find_and_tap("New post") and \
               not await self._find_and_tap("Create"):
                cx = self.controller.resolution[0] // 2
                await self.controller.tap(cx, self.controller.resolution[1] - 80)
                await self.cool_down(1.5)

            await self.cool_down(2.0)

            # Select "Post" if shown
            await self._find_and_tap("Post")
            await self.cool_down(1.0)

            # Navigate gallery: push the image via ADB, then select it
            await self._push_and_select_image(image_path)
            await self.cool_down(1.0)

            # Tap Next
            await self._find_and_tap("Next")
            await self.cool_down(1.5)

            # Skip filters -- tap Next again
            await self._find_and_tap("Next")
            await self.cool_down(1.5)

            # Write caption
            full_caption = caption
            if hashtags:
                tag_str = " ".join(f"#{t.lstrip('#')}" for t in hashtags)
                full_caption = f"{caption}\n\n{tag_str}"

            caption_el = await self.controller.find_element(
                text="Write a caption", partial_match=True
            ) or await self.controller.find_element(
                text="caption", partial_match=True
            )
            if caption_el:
                await self.controller.tap_element(caption_el)
                await self.cool_down(0.5)
                await self.controller.type_text(full_caption)
                await self.cool_down(0.5)

            # Add location if provided
            if location:
                if await self._find_and_tap("Add location"):
                    await self.cool_down(1.0)
                    await self.controller.type_text(location)
                    await self.cool_down(2.0)
                    # Tap first result
                    await self.controller.tap(
                        self.controller.resolution[0] // 2, 400
                    )
                    await self.cool_down(1.0)

            # Share
            await self._find_and_tap("Share")
            await self.cool_down(5.0)

            record.success = True
            await self._record_action("posts")

        except Exception as exc:
            record.error = str(exc)
            logger.error("Instagram post_photo failed: %s", exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def post_story(
        self,
        image_path: str,
        stickers: Optional[List[Dict[str, Any]]] = None,
        text_overlay: Optional[str] = None,
    ) -> ActionRecord:
        """Create an Instagram story with optional stickers and text."""
        record = ActionRecord(platform="instagram", category=ActionCategory.STORY.value,
                              description="Post story")
        start = time.monotonic()

        if not await self._check_limit("stories"):
            record.error = "Daily story limit reached"
            return record

        try:
            await self.open_app()
            await self.cool_down(1.0)

            # Tap Your Story or camera icon (top-left)
            if not await self._find_and_tap("Your story"):
                await self.controller.tap(60, 120)
            await self.cool_down(2.0)

            # Push image and select from gallery
            await self._push_and_select_image(image_path)
            await self.cool_down(1.5)

            # Add text overlay
            if text_overlay:
                # Tap screen to add text
                cx = self.controller.resolution[0] // 2
                cy = self.controller.resolution[1] // 2
                await self.controller.tap(cx, cy)
                await self.cool_down(0.5)
                if await self._find_and_tap("Aa"):
                    await self.cool_down(0.5)
                    await self.controller.type_text(text_overlay)
                    await self._find_and_tap("Done")
                    await self.cool_down(0.5)

            # Add stickers
            if stickers:
                for sticker in stickers:
                    sticker_type = sticker.get("type", "")
                    if await self._find_and_tap("sticker") or \
                       await self._find_and_tap("Stickers"):
                        await self.cool_down(1.0)
                        if sticker_type:
                            await self._find_and_tap(sticker_type)
                            await self.cool_down(1.0)
                        await self.safe_back()

            # Share to story
            if await self._find_and_tap("Your story") or \
               await self._find_and_tap("Share"):
                await self.cool_down(3.0)
                record.success = True
                await self._record_action("stories")

        except Exception as exc:
            record.error = str(exc)
            logger.error("Instagram post_story failed: %s", exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def post_reel(
        self,
        video_path: str,
        caption: str,
        hashtags: Optional[List[str]] = None,
        audio_name: Optional[str] = None,
    ) -> ActionRecord:
        """Upload a reel with caption and optional audio."""
        record = ActionRecord(platform="instagram", category=ActionCategory.REEL.value,
                              description=f"Post reel: {caption[:50]}...")
        start = time.monotonic()

        if not await self._check_limit("reels"):
            record.error = "Daily reel limit reached"
            return record

        try:
            await self.open_app()
            await self.cool_down(1.0)

            # Tap create button
            if not await self._find_and_tap("New post") and \
               not await self._find_and_tap("Create"):
                cx = self.controller.resolution[0] // 2
                await self.controller.tap(cx, self.controller.resolution[1] - 80)
            await self.cool_down(2.0)

            # Select Reel tab
            await self._find_and_tap("Reel")
            await self.cool_down(1.0)

            # Push video and select
            await self._push_media(video_path)
            await self.cool_down(2.0)

            # Select from gallery
            gallery_el = await self.controller.find_element(
                text="Gallery", partial_match=True
            )
            if gallery_el:
                await self.controller.tap_element(gallery_el)
                await self.cool_down(1.5)
            # Tap first video in gallery
            await self.controller.tap(200, 600)
            await self.cool_down(1.5)

            # Add audio if specified
            if audio_name:
                if await self._find_and_tap("Audio") or \
                   await self._find_and_tap("Music"):
                    await self.cool_down(1.0)
                    search_el = await self.controller.find_element(
                        text="Search", partial_match=True
                    )
                    if search_el:
                        await self.controller.tap_element(search_el)
                        await self.controller.type_text(audio_name)
                        await self.cool_down(2.0)
                        # Tap first result
                        await self.controller.tap(
                            self.controller.resolution[0] // 2, 400
                        )
                    await self.cool_down(1.0)

            # Next
            await self._find_and_tap("Next")
            await self.cool_down(2.0)

            # Write caption with hashtags
            full_caption = caption
            if hashtags:
                tag_str = " ".join(f"#{t.lstrip('#')}" for t in hashtags)
                full_caption = f"{caption}\n\n{tag_str}"

            caption_el = await self.controller.find_element(
                text="caption", partial_match=True
            )
            if caption_el:
                await self.controller.tap_element(caption_el)
                await self.cool_down(0.5)
                await self.controller.type_text(full_caption)

            # Share
            await self._find_and_tap("Share")
            await self.cool_down(5.0)

            record.success = True
            await self._record_action("reels")

        except Exception as exc:
            record.error = str(exc)
            logger.error("Instagram post_reel failed: %s", exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def like_posts(
        self,
        hashtag: str,
        count: int = 10,
        delay_range: Tuple[float, float] = (3.0, 8.0),
    ) -> ActionRecord:
        """Like posts from a hashtag feed."""
        record = ActionRecord(platform="instagram", category=ActionCategory.LIKE.value,
                              description=f"Like {count} posts from #{hashtag}")
        start = time.monotonic()
        liked = 0

        try:
            await self.open_app()
            await self._navigate_to_hashtag(hashtag)

            for i in range(count):
                if not await self._check_limit("likes"):
                    break

                # Double-tap to like the current post
                cx = self.controller.resolution[0] // 2
                cy = self.controller.resolution[1] // 2
                await self.controller.tap(cx, cy)
                await asyncio.sleep(0.15)
                await self.controller.tap(cx, cy)
                await self.cool_down(0.5)

                liked += 1
                await self._record_action("likes")

                # Scroll to next post
                await self.controller.scroll_down(random.randint(400, 700))
                await self.behavior.random_delay(*delay_range)

            record.success = True
            record.metadata = {"liked": liked, "hashtag": hashtag}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def follow_users(
        self,
        from_hashtag: str,
        count: int = 10,
        delay_range: Tuple[float, float] = (5.0, 15.0),
    ) -> ActionRecord:
        """Follow users from a hashtag feed."""
        record = ActionRecord(platform="instagram", category=ActionCategory.FOLLOW.value,
                              description=f"Follow {count} from #{from_hashtag}")
        start = time.monotonic()
        followed = 0

        try:
            await self.open_app()
            await self._navigate_to_hashtag(from_hashtag)

            for i in range(count):
                if not await self._check_limit("follows"):
                    break

                # Tap the username at top of post to open profile
                await self.controller.tap(200, 300)
                await self.cool_down(2.0)

                # Tap Follow button
                if await self._find_and_tap("Follow"):
                    followed += 1
                    await self._record_action("follows")
                    await self.cool_down(1.0)

                # Go back to feed
                await self.safe_back()
                await self.cool_down(1.0)

                # Scroll to next post
                await self.controller.scroll_down(random.randint(400, 700))
                await self.behavior.random_delay(*delay_range)

            record.success = True
            record.metadata = {"followed": followed, "hashtag": from_hashtag}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def unfollow_users(
        self,
        count: int = 10,
        skip_followers: bool = True,
        delay_range: Tuple[float, float] = (5.0, 15.0),
    ) -> ActionRecord:
        """Unfollow users from the following list."""
        record = ActionRecord(platform="instagram", category=ActionCategory.UNFOLLOW.value,
                              description=f"Unfollow {count} users")
        start = time.monotonic()
        unfollowed = 0

        try:
            await self.open_app()
            # Navigate to profile -> following
            await self._navigate_to_profile()
            await self._find_and_tap("following")
            await self.cool_down(2.0)

            for i in range(count):
                if not await self._check_limit("follows"):
                    break

                # Find and tap an "Following" button
                btn = await self.controller.find_element(
                    text="Following", partial_match=False
                )
                if not btn:
                    await self.controller.scroll_down()
                    await self.cool_down(1.0)
                    btn = await self.controller.find_element(
                        text="Following", partial_match=False
                    )

                if btn:
                    await self.controller.tap_element(btn)
                    await self.cool_down(1.0)
                    # Confirm unfollow
                    await self._find_and_tap("Unfollow")
                    unfollowed += 1
                    await self._record_action("follows")
                    await self.behavior.random_delay(*delay_range)

            record.success = True
            record.metadata = {"unfollowed": unfollowed}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def comment_on_posts(
        self,
        hashtag: str,
        comments: List[str],
        count: int = 5,
    ) -> ActionRecord:
        """Post comments on posts from a hashtag feed."""
        record = ActionRecord(platform="instagram", category=ActionCategory.COMMENT.value,
                              description=f"Comment on {count} posts from #{hashtag}")
        start = time.monotonic()
        commented = 0

        try:
            await self.open_app()
            await self._navigate_to_hashtag(hashtag)

            for i in range(count):
                if not await self._check_limit("comments"):
                    break

                # Tap comment icon (speech bubble)
                comment_el = await self.controller.find_element(
                    content_desc="Comment", partial_match=True
                )
                if comment_el:
                    await self.controller.tap_element(comment_el)
                    await self.cool_down(1.5)

                    # Type a random comment
                    comment = random.choice(comments)
                    await self.controller.type_text(comment)
                    await self.cool_down(0.5)

                    # Post comment
                    await self._find_and_tap("Post")
                    commented += 1
                    await self._record_action("comments")
                    await self.cool_down(1.0)

                    # Go back
                    await self.safe_back()

                # Scroll to next
                await self.controller.scroll_down(random.randint(400, 700))
                await self.behavior.random_delay(5.0, 15.0)

            record.success = True
            record.metadata = {"commented": commented, "hashtag": hashtag}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def send_dm(self, username: str, message: str) -> ActionRecord:
        """Send a direct message to a user."""
        record = ActionRecord(platform="instagram", category=ActionCategory.DM.value,
                              description=f"DM to @{username}")
        start = time.monotonic()

        if not await self._check_limit("dms"):
            record.error = "Daily DM limit reached"
            return record

        try:
            await self.open_app()
            # Tap DM icon (top right)
            dm_el = await self.controller.find_element(
                content_desc="Direct", partial_match=True
            )
            if dm_el:
                await self.controller.tap_element(dm_el)
            else:
                w = self.controller.resolution[0]
                await self.controller.tap(w - 60, 80)
            await self.cool_down(2.0)

            # Tap new message
            await self._find_and_tap("New message") or \
                await self._find_and_tap("compose")
            await self.cool_down(1.0)

            # Search for user
            await self.controller.type_text(username)
            await self.cool_down(2.0)

            # Tap first result
            await self.controller.tap(self.controller.resolution[0] // 2, 350)
            await self.cool_down(1.0)

            # Tap Next/Chat
            await self._find_and_tap("Next") or await self._find_and_tap("Chat")
            await self.cool_down(1.5)

            # Type message
            msg_el = await self.controller.find_element(
                text="Message", partial_match=True
            )
            if msg_el:
                await self.controller.tap_element(msg_el)
                await self.cool_down(0.3)
            await self.controller.type_text(message)
            await self.cool_down(0.5)

            # Send
            await self._find_and_tap("Send")
            await self.cool_down(1.0)

            record.success = True
            await self._record_action("dms")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def check_insights(self) -> AnalyticsSnapshot:
        """Navigate to insights and extract follower/reach data via OCR."""
        snapshot = AnalyticsSnapshot(platform="instagram")

        try:
            await self.open_app()
            await self._navigate_to_profile()
            await self.cool_down(1.0)

            # Tap Insights / Professional dashboard
            if not await self._find_and_tap("Insights") and \
               not await self._find_and_tap("Professional dashboard"):
                snapshot.metrics = {"error": "Insights not found"}
                return snapshot

            await self.cool_down(3.0)

            # Take screenshot and OCR the analytics
            screenshot_path = await self.controller.screenshot()
            snapshot.screenshot_path = screenshot_path

            analysis = await self.vision.analyze_screen(
                "Extract all analytics metrics visible on this Instagram Insights screen. "
                "Return JSON with keys like: followers, reach, impressions, "
                "profile_visits, website_clicks, content_interactions.",
                screenshot_path=screenshot_path,
            )
            try:
                import re as _re
                cleaned = analysis.raw_response.strip()
                if isinstance(cleaned, str) and cleaned.startswith("{"):
                    snapshot.metrics = json.loads(cleaned)
                else:
                    snapshot.metrics = {"raw": analysis.description}
            except (json.JSONDecodeError, TypeError):
                snapshot.metrics = {"raw": analysis.description}

        except Exception as exc:
            snapshot.metrics = {"error": str(exc)}

        return snapshot

    async def save_post(self, post_url: str) -> ActionRecord:
        """Save a post to a collection."""
        record = ActionRecord(platform="instagram", category=ActionCategory.SAVE.value,
                              description=f"Save post: {post_url}")
        start = time.monotonic()

        try:
            await self.open_app()
            # Use deep link to open post directly
            await self.controller._adb_shell(
                f"am start -a android.intent.action.VIEW -d '{post_url}'"
            )
            await self.cool_down(3.0)

            # Tap bookmark/save icon
            save_el = await self.controller.find_element(
                content_desc="Save", partial_match=True
            ) or await self.controller.find_element(
                content_desc="Bookmark", partial_match=True
            )
            if save_el:
                await self.controller.tap_element(save_el)
                record.success = True
            else:
                success = await self._vision_guided_flow(
                    "Tap the save/bookmark icon on this Instagram post"
                )
                record.success = success

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    # -- Internal helpers ----------------------------------------------------

    async def _push_and_select_image(self, image_path: str) -> None:
        """Push image to device and select it from gallery."""
        await self._push_media(image_path)
        # Tap gallery if visible
        gallery = await self.controller.find_element(text="Gallery", partial_match=True)
        if gallery:
            await self.controller.tap_element(gallery)
            await self.cool_down(1.5)
        # Tap first image in grid
        await self.controller.tap(200, 600)
        await self.cool_down(1.0)

    async def _push_media(self, local_path: str) -> None:
        """Push a file to the device's DCIM folder via ADB."""
        filename = Path(local_path).name
        device_path = f"/sdcard/DCIM/{filename}"
        await self.controller._invoke_node("file.push", {
            "local_path": local_path, "device_path": device_path,
        })
        # Trigger media scan so it appears in gallery
        await self.controller._adb_shell(
            f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
            f"-d file://{device_path}"
        )
        await self.cool_down(1.5)

    async def _navigate_to_hashtag(self, hashtag: str) -> None:
        """Navigate to a hashtag's feed."""
        # Tap search/explore
        explore_el = await self.controller.find_element(
            content_desc="Search", partial_match=True
        )
        if explore_el:
            await self.controller.tap_element(explore_el)
        else:
            w = self.controller.resolution[0]
            await self.controller.tap(w // 4, self.controller.resolution[1] - 60)
        await self.cool_down(1.5)

        # Search
        search_el = await self.controller.find_element(text="Search", partial_match=True)
        if search_el:
            await self.controller.tap_element(search_el)
            await self.cool_down(0.5)
        await self.controller.type_text(f"#{hashtag}")
        await self.cool_down(2.0)

        # Tap first hashtag result
        await self.controller.tap(self.controller.resolution[0] // 2, 350)
        await self.cool_down(2.0)

        # Tap "Recent" tab for latest posts
        await self._find_and_tap("Recent")
        await self.cool_down(1.5)

        # Tap first post in grid
        await self.controller.tap(180, 800)
        await self.cool_down(1.5)

    async def _navigate_to_profile(self) -> None:
        """Navigate to the user's own profile."""
        profile_el = await self.controller.find_element(
            content_desc="Profile", partial_match=True
        )
        if profile_el:
            await self.controller.tap_element(profile_el)
        else:
            w = self.controller.resolution[0]
            h = self.controller.resolution[1]
            await self.controller.tap(w - 60, h - 60)
        await self.cool_down(2.0)


# ---------------------------------------------------------------------------
# TikTokBot
# ---------------------------------------------------------------------------


class TikTokBot(BasePlatformBot):
    """
    TikTok automation via in-app interaction.

    Handles: video upload, live streaming, FYP engagement, following,
    comment replies, and analytics scraping.
    """

    package_name = "com.zhiliaoapp.musically"
    app_label = "TikTok"

    async def upload_video(
        self,
        video_path: str,
        caption: str,
        hashtags: Optional[List[str]] = None,
        sounds: Optional[str] = None,
    ) -> ActionRecord:
        """Upload a video with caption, hashtags, and optional sound."""
        record = ActionRecord(platform="tiktok", category=ActionCategory.POST.value,
                              description=f"Upload video: {caption[:50]}...")
        start = time.monotonic()

        if not await self._check_limit("posts"):
            record.error = "Daily post limit reached"
            return record

        try:
            await self.open_app()
            await self.cool_down(1.5)

            # Tap the + create button (center bottom)
            cx = self.controller.resolution[0] // 2
            h = self.controller.resolution[1]
            await self.controller.tap(cx, h - 60)
            await self.cool_down(2.0)

            # Tap Upload
            await self._find_and_tap("Upload")
            await self.cool_down(1.5)

            # Push video and select
            filename = Path(video_path).name
            device_path = f"/sdcard/DCIM/{filename}"
            await self.controller._invoke_node("file.push", {
                "local_path": video_path, "device_path": device_path,
            })
            await self.controller._adb_shell(
                f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
                f"-d file://{device_path}"
            )
            await self.cool_down(2.0)

            # Select first video
            await self.controller.tap(200, 500)
            await self.cool_down(1.5)

            # Tap Next
            await self._find_and_tap("Next")
            await self.cool_down(2.0)

            # Add sound if specified
            if sounds:
                if await self._find_and_tap("Sounds") or \
                   await self._find_and_tap("Add sound"):
                    await self.cool_down(1.0)
                    await self.controller.type_text(sounds)
                    await self.cool_down(2.0)
                    await self.controller.tap(cx, 400)
                    await self.cool_down(1.0)
                    await self._find_and_tap("Done") or await self._find_and_tap("Use")
                    await self.cool_down(1.0)

            # Tap Next again (past editing)
            await self._find_and_tap("Next")
            await self.cool_down(2.0)

            # Write caption
            full_caption = caption
            if hashtags:
                tag_str = " ".join(f"#{t.lstrip('#')}" for t in hashtags)
                full_caption = f"{caption} {tag_str}"

            desc_el = await self.controller.find_element(
                text="Describe your video", partial_match=True
            ) or await self.controller.find_element(
                text="caption", partial_match=True
            )
            if desc_el:
                await self.controller.tap_element(desc_el)
                await self.cool_down(0.5)
            await self.controller.type_text(full_caption)
            await self.cool_down(0.5)

            # Post
            await self._find_and_tap("Post")
            await self.cool_down(5.0)

            record.success = True
            await self._record_action("posts")

        except Exception as exc:
            record.error = str(exc)
            logger.error("TikTok upload_video failed: %s", exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def go_live(self, title: str) -> ActionRecord:
        """Start a TikTok live stream with the given title."""
        record = ActionRecord(platform="tiktok", category=ActionCategory.POST.value,
                              description=f"Go live: {title}")
        start = time.monotonic()

        try:
            await self.open_app()
            cx = self.controller.resolution[0] // 2
            h = self.controller.resolution[1]
            await self.controller.tap(cx, h - 60)
            await self.cool_down(2.0)

            # Select LIVE
            await self._find_and_tap("LIVE")
            await self.cool_down(2.0)

            # Set title
            title_el = await self.controller.find_element(
                text="title", partial_match=True
            )
            if title_el:
                await self.controller.tap_element(title_el)
                await self.cool_down(0.5)
                await self.controller.type_text(title)
                await self.cool_down(0.5)

            # Go live
            await self._find_and_tap("Go LIVE")
            await self.cool_down(3.0)

            record.success = True

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def engage_fyp(
        self,
        duration_minutes: int = 10,
        like_ratio: float = 0.3,
    ) -> ActionRecord:
        """Scroll FYP and like posts at the given ratio."""
        record = ActionRecord(platform="tiktok", category=ActionCategory.LIKE.value,
                              description=f"FYP engagement {duration_minutes}min")
        start = time.monotonic()
        liked = 0
        viewed = 0

        try:
            await self.open_app()
            # Ensure we are on For You page
            await self._find_and_tap("For You") or await self._find_and_tap("Home")
            await self.cool_down(1.5)

            end_time = time.monotonic() + duration_minutes * 60
            while time.monotonic() < end_time:
                viewed += 1

                # Randomly like based on ratio
                if random.random() < like_ratio:
                    if await self._check_limit("likes"):
                        # Double tap to like
                        cx = self.controller.resolution[0] // 2
                        cy = self.controller.resolution[1] // 2
                        await self.controller.tap(cx, cy)
                        await asyncio.sleep(0.15)
                        await self.controller.tap(cx, cy)
                        liked += 1
                        await self._record_action("likes")
                        await self.cool_down(0.5)

                # Watch for 3-8 seconds
                await self.behavior.random_delay(3.0, 8.0)

                # Swipe up to next video
                h = self.controller.resolution[1]
                cx = self.controller.resolution[0] // 2
                await self.controller.swipe(cx, h * 3 // 4, cx, h // 4, 300)
                await self.cool_down(1.0)

                await self._record_action("views")

            record.success = True
            record.metadata = {"viewed": viewed, "liked": liked}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def follow_from_video(self, count: int = 10) -> ActionRecord:
        """Follow creators from the FYP."""
        record = ActionRecord(platform="tiktok", category=ActionCategory.FOLLOW.value,
                              description=f"Follow {count} from FYP")
        start = time.monotonic()
        followed = 0

        try:
            await self.open_app()
            await self._find_and_tap("For You") or await self._find_and_tap("Home")
            await self.cool_down(1.5)

            for i in range(count):
                if not await self._check_limit("follows"):
                    break

                # Tap the + follow button on the right side of the video
                w = self.controller.resolution[0]
                await self.controller.tap(w - 50, self.controller.resolution[1] // 2 - 100)
                await self.cool_down(1.0)

                followed += 1
                await self._record_action("follows")

                # Swipe to next video
                h = self.controller.resolution[1]
                cx = w // 2
                await self.controller.swipe(cx, h * 3 // 4, cx, h // 4, 300)
                await self.behavior.random_delay(5.0, 12.0)

            record.success = True
            record.metadata = {"followed": followed}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def check_analytics(self) -> AnalyticsSnapshot:
        """Extract view/follower data from TikTok analytics."""
        snapshot = AnalyticsSnapshot(platform="tiktok")

        try:
            await self.open_app()
            # Navigate to profile
            w = self.controller.resolution[0]
            h = self.controller.resolution[1]
            await self.controller.tap(w - 40, h - 60)
            await self.cool_down(2.0)

            # Tap hamburger menu -> Creator tools -> Analytics
            await self.controller.tap(w - 40, 80)
            await self.cool_down(1.5)

            if await self._find_and_tap("Creator tools") or \
               await self._find_and_tap("Business suite"):
                await self.cool_down(1.5)
                await self._find_and_tap("Analytics")
                await self.cool_down(3.0)

            screenshot_path = await self.controller.screenshot()
            snapshot.screenshot_path = screenshot_path

            analysis = await self.vision.analyze_screen(
                "Extract TikTok analytics metrics. Return JSON: "
                "views, followers, likes, comments, shares, profile_views.",
                screenshot_path=screenshot_path,
            )
            try:
                cleaned = analysis.raw_response.strip()
                if isinstance(cleaned, str) and cleaned.startswith("{"):
                    snapshot.metrics = json.loads(cleaned)
                else:
                    snapshot.metrics = {"raw": analysis.description}
            except (json.JSONDecodeError, TypeError):
                snapshot.metrics = {"raw": analysis.description}

        except Exception as exc:
            snapshot.metrics = {"error": str(exc)}

        return snapshot

    async def reply_to_comments(
        self,
        video_id: str,
        replies_map: Dict[str, str],
    ) -> ActionRecord:
        """Reply to comments on a video. replies_map: {comment_text: reply_text}."""
        record = ActionRecord(platform="tiktok", category=ActionCategory.COMMENT.value,
                              description=f"Reply to {len(replies_map)} comments")
        start = time.monotonic()
        replied = 0

        try:
            await self.open_app()
            # Navigate to profile -> video
            w = self.controller.resolution[0]
            h = self.controller.resolution[1]
            await self.controller.tap(w - 40, h - 60)
            await self.cool_down(2.0)

            # Tap first video (or find by video_id if possible)
            await self.controller.tap(180, 600)
            await self.cool_down(2.0)

            # Tap comments icon
            await self.controller.tap(w - 50, h // 2 + 50)
            await self.cool_down(2.0)

            for comment_text, reply_text in replies_map.items():
                if not await self._check_limit("comments"):
                    break

                # Find the comment
                el = await self.controller.find_element(
                    text=comment_text, partial_match=True
                )
                if el:
                    # Long press to reply
                    cx, cy = el.center
                    await self.controller.long_press(cx, cy)
                    await self.cool_down(1.0)
                    await self._find_and_tap("Reply")
                    await self.cool_down(0.5)
                    await self.controller.type_text(reply_text)
                    await self.cool_down(0.5)
                    await self._find_and_tap("Send") or \
                        await self.controller.press_enter()
                    replied += 1
                    await self._record_action("comments")
                    await self.behavior.random_delay(3.0, 8.0)
                else:
                    await self.controller.scroll_down(300)
                    await self.cool_down(1.0)

            record.success = True
            record.metadata = {"replied": replied}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record


# ---------------------------------------------------------------------------
# PinterestBot
# ---------------------------------------------------------------------------


class PinterestBot(BasePlatformBot):
    """
    Pinterest automation via in-app interaction.

    Handles: pin creation, idea pins, board management, pin saving,
    board following, analytics scraping, and bulk pinning.
    """

    package_name = "com.pinterest"
    app_label = "Pinterest"

    async def create_pin(
        self,
        image_path: str,
        title: str,
        description: str,
        link: str,
        board_name: str,
    ) -> ActionRecord:
        """Create a standard pin with image, title, description, and link."""
        record = ActionRecord(platform="pinterest", category=ActionCategory.POST.value,
                              description=f"Create pin: {title[:50]}...")
        start = time.monotonic()

        if not await self._check_limit("pins"):
            record.error = "Daily pin limit reached"
            return record

        try:
            await self.open_app()
            await self.cool_down(1.0)

            # Tap + create button
            if not await self._find_and_tap("Create") and \
               not await self._find_and_tap("Add"):
                cx = self.controller.resolution[0] // 2
                h = self.controller.resolution[1]
                await self.controller.tap(cx, h - 60)
            await self.cool_down(1.5)

            # Select "Pin"
            await self._find_and_tap("Pin")
            await self.cool_down(1.5)

            # Push image and select
            await self._push_image(image_path)
            await self.cool_down(1.5)

            # Tap gallery pick
            gallery = await self.controller.find_element(
                text="Gallery", partial_match=True
            )
            if gallery:
                await self.controller.tap_element(gallery)
                await self.cool_down(1.0)
            await self.controller.tap(200, 500)
            await self.cool_down(1.5)

            # Fill in title
            await self._type_in_field("Title", title) or \
                await self._type_in_field("Add a title", title)
            await self.cool_down(0.5)

            # Fill in description
            await self._type_in_field("description", description) or \
                await self._type_in_field("Tell everyone", description)
            await self.cool_down(0.5)

            # Add link
            await self._type_in_field("link", link) or \
                await self._type_in_field("website", link) or \
                await self._type_in_field("destination", link)
            await self.cool_down(0.5)

            # Select board
            if await self._find_and_tap("Board") or \
               await self._find_and_tap("Select board"):
                await self.cool_down(1.0)
                if await self._find_and_tap(board_name):
                    await self.cool_down(1.0)
                else:
                    # Search for board
                    search = await self.controller.find_element(
                        text="Search", partial_match=True
                    )
                    if search:
                        await self.controller.tap_element(search)
                        await self.controller.type_text(board_name)
                        await self.cool_down(1.0)
                        await self.controller.tap(
                            self.controller.resolution[0] // 2, 350
                        )

            # Publish
            await self._find_and_tap("Publish") or \
                await self._find_and_tap("Save") or \
                await self._find_and_tap("Done")
            await self.cool_down(3.0)

            record.success = True
            await self._record_action("pins")

        except Exception as exc:
            record.error = str(exc)
            logger.error("Pinterest create_pin failed: %s", exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def create_idea_pin(
        self,
        images: List[str],
        title: str,
        description: str,
    ) -> ActionRecord:
        """Create a multi-image idea pin."""
        record = ActionRecord(platform="pinterest", category=ActionCategory.POST.value,
                              description=f"Idea pin: {title[:50]}...")
        start = time.monotonic()

        if not await self._check_limit("pins"):
            record.error = "Daily pin limit reached"
            return record

        try:
            await self.open_app()
            await self.cool_down(1.0)

            # Create -> Idea Pin
            if not await self._find_and_tap("Create"):
                cx = self.controller.resolution[0] // 2
                h = self.controller.resolution[1]
                await self.controller.tap(cx, h - 60)
            await self.cool_down(1.5)

            await self._find_and_tap("Idea Pin") or \
                await self._find_and_tap("Story Pin")
            await self.cool_down(1.5)

            # Push all images
            for img in images:
                await self._push_image(img)
            await self.cool_down(1.5)

            # Select images from gallery
            for i in range(len(images)):
                x = 200 + (i % 3) * 200
                await self.controller.tap(x, 500)
                await self.cool_down(0.5)
            await self.cool_down(1.0)

            # Next
            await self._find_and_tap("Next")
            await self.cool_down(2.0)

            # Title and description
            await self._type_in_field("Title", title) or \
                await self._type_in_field("title", title)
            await self.cool_down(0.5)

            # May need to scroll to description
            await self._scroll_and_find("description")
            desc_el = await self.controller.find_element(
                text="description", partial_match=True
            )
            if desc_el:
                await self.controller.tap_element(desc_el)
                await self.cool_down(0.3)
                await self.controller.type_text(description)
                await self.cool_down(0.5)

            # Publish
            await self._find_and_tap("Publish") or \
                await self._find_and_tap("Done")
            await self.cool_down(3.0)

            record.success = True
            await self._record_action("pins")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def create_board(
        self,
        name: str,
        description: str,
        category: Optional[str] = None,
    ) -> ActionRecord:
        """Create a new Pinterest board."""
        record = ActionRecord(platform="pinterest", category=ActionCategory.POST.value,
                              description=f"Create board: {name}")
        start = time.monotonic()

        if not await self._check_limit("boards"):
            record.error = "Daily board limit reached"
            return record

        try:
            await self.open_app()
            # Go to profile
            w = self.controller.resolution[0]
            h = self.controller.resolution[1]
            await self.controller.tap(w - 40, h - 60)
            await self.cool_down(2.0)

            # Tap + to create board
            if await self._find_and_tap("Create board") or \
               await self._find_and_tap("+"):
                await self.cool_down(1.5)

            # Enter name
            await self._type_in_field("Name", name) or \
                await self._type_in_field("Board name", name)
            await self.cool_down(0.5)

            # Create
            await self._find_and_tap("Create") or \
                await self._find_and_tap("Done")
            await self.cool_down(2.0)

            # Add description
            if await self._find_and_tap("Edit") or \
               await self._find_and_tap("description"):
                await self.cool_down(1.0)
                desc_el = await self.controller.find_element(
                    text="description", partial_match=True
                )
                if desc_el:
                    await self.controller.tap_element(desc_el)
                    await self.controller.type_text(description)
                    await self.cool_down(0.5)
                await self._find_and_tap("Done") or \
                    await self._find_and_tap("Save")

            record.success = True
            await self._record_action("boards")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def save_pin(self, pin_url: str, board_name: str) -> ActionRecord:
        """Save an external pin to a board."""
        record = ActionRecord(platform="pinterest", category=ActionCategory.SAVE.value,
                              description=f"Save pin to {board_name}")
        start = time.monotonic()

        if not await self._check_limit("saves"):
            record.error = "Daily save limit reached"
            return record

        try:
            await self.open_app()
            await self.controller._adb_shell(
                f"am start -a android.intent.action.VIEW -d '{pin_url}'"
            )
            await self.cool_down(3.0)

            # Tap Save
            if await self._find_and_tap("Save"):
                await self.cool_down(1.5)
                # Select board
                if await self._find_and_tap(board_name):
                    await self.cool_down(1.0)
                    record.success = True
                    await self._record_action("saves")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def follow_board(self, board_url: str) -> ActionRecord:
        """Follow a Pinterest board."""
        record = ActionRecord(platform="pinterest", category=ActionCategory.FOLLOW.value,
                              description=f"Follow board: {board_url}")
        start = time.monotonic()

        if not await self._check_limit("follows"):
            record.error = "Daily follow limit reached"
            return record

        try:
            await self.open_app()
            await self.controller._adb_shell(
                f"am start -a android.intent.action.VIEW -d '{board_url}'"
            )
            await self.cool_down(3.0)

            if await self._find_and_tap("Follow"):
                record.success = True
                await self._record_action("follows")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def check_analytics(self) -> AnalyticsSnapshot:
        """Extract impressions/saves data from Pinterest analytics."""
        snapshot = AnalyticsSnapshot(platform="pinterest")

        try:
            await self.open_app()
            # Profile -> Analytics
            w = self.controller.resolution[0]
            h = self.controller.resolution[1]
            await self.controller.tap(w - 40, h - 60)
            await self.cool_down(2.0)

            if not await self._find_and_tap("Analytics") and \
               not await self._find_and_tap("Business hub"):
                snapshot.metrics = {"error": "Analytics not found"}
                return snapshot

            await self.cool_down(3.0)

            screenshot_path = await self.controller.screenshot()
            snapshot.screenshot_path = screenshot_path

            analysis = await self.vision.analyze_screen(
                "Extract Pinterest analytics. Return JSON: impressions, "
                "saves, pin_clicks, outbound_clicks, followers, engagement_rate.",
                screenshot_path=screenshot_path,
            )
            try:
                cleaned = analysis.raw_response.strip()
                if isinstance(cleaned, str) and cleaned.startswith("{"):
                    snapshot.metrics = json.loads(cleaned)
                else:
                    snapshot.metrics = {"raw": analysis.description}
            except (json.JSONDecodeError, TypeError):
                snapshot.metrics = {"raw": analysis.description}

        except Exception as exc:
            snapshot.metrics = {"error": str(exc)}

        return snapshot

    async def bulk_pin(self, pins_data: List[Dict[str, Any]]) -> ActionRecord:
        """Create multiple pins with human-like delays between each."""
        record = ActionRecord(platform="pinterest", category=ActionCategory.POST.value,
                              description=f"Bulk pin: {len(pins_data)} pins")
        start = time.monotonic()
        created = 0

        for pin in pins_data:
            if not await self._check_limit("pins"):
                break

            result = await self.create_pin(
                image_path=pin.get("image_path", ""),
                title=pin.get("title", ""),
                description=pin.get("description", ""),
                link=pin.get("link", ""),
                board_name=pin.get("board_name", ""),
            )
            if result.success:
                created += 1

            # Rest between pins
            await self.behavior.random_delay(30.0, 90.0)

        record.success = created > 0
        record.metadata = {"created": created, "total": len(pins_data)}
        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    # -- Internal helpers ----------------------------------------------------

    async def _push_image(self, image_path: str) -> None:
        """Push an image to device DCIM."""
        filename = Path(image_path).name
        device_path = f"/sdcard/DCIM/{filename}"
        await self.controller._invoke_node("file.push", {
            "local_path": image_path, "device_path": device_path,
        })
        await self.controller._adb_shell(
            f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
            f"-d file://{device_path}"
        )
        await self.cool_down(1.0)


# ---------------------------------------------------------------------------
# FacebookBot
# ---------------------------------------------------------------------------


class FacebookBot(BasePlatformBot):
    """Facebook automation via in-app interaction."""

    package_name = "com.facebook.katana"
    app_label = "Facebook"

    async def create_post(
        self,
        text: str,
        images: Optional[List[str]] = None,
        link: Optional[str] = None,
    ) -> ActionRecord:
        """Create a feed post with optional images and link."""
        record = ActionRecord(platform="facebook", category=ActionCategory.POST.value,
                              description=f"Post: {text[:50]}...")
        start = time.monotonic()

        if not await self._check_limit("posts"):
            record.error = "Daily post limit reached"
            return record

        try:
            await self.open_app()
            await self.cool_down(1.5)

            # Tap "What's on your mind?"
            await self._find_and_tap("What's on your mind") or \
                await self._find_and_tap("Create a post")
            await self.cool_down(2.0)

            # Type the post text
            content = text
            if link:
                content = f"{text}\n\n{link}"
            await self.controller.type_text(content)
            await self.cool_down(0.5)

            # Add images if provided
            if images:
                for img_path in images:
                    await self._push_media_fb(img_path)
                if await self._find_and_tap("Photo") or \
                   await self._find_and_tap("Photo/Video"):
                    await self.cool_down(2.0)
                    # Select images from gallery
                    for i in range(len(images)):
                        x = 200 + (i % 3) * 200
                        await self.controller.tap(x, 500)
                        await self.cool_down(0.5)
                    await self._find_and_tap("Done") or \
                        await self._find_and_tap("Next")
                    await self.cool_down(1.0)

            # Post
            await self._find_and_tap("Post")
            await self.cool_down(3.0)

            record.success = True
            await self._record_action("posts")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def post_to_group(
        self,
        group_name: str,
        text: str,
        images: Optional[List[str]] = None,
    ) -> ActionRecord:
        """Post to a Facebook group."""
        record = ActionRecord(platform="facebook", category=ActionCategory.POST.value,
                              description=f"Group post: {group_name}")
        start = time.monotonic()

        if not await self._check_limit("posts"):
            record.error = "Daily post limit reached"
            return record

        try:
            await self.open_app()
            await self.cool_down(1.0)

            # Navigate to Groups
            await self._find_and_tap("Groups") or \
                await self._find_and_tap("Menu")
            await self.cool_down(2.0)

            # Search for the group
            if await self._find_and_tap("Search") or \
               await self._find_and_tap("Search groups"):
                await self.cool_down(0.5)
                await self.controller.type_text(group_name)
                await self.cool_down(2.0)
                # Tap first result
                await self.controller.tap(
                    self.controller.resolution[0] // 2, 350
                )
                await self.cool_down(2.0)

            # Write something
            await self._find_and_tap("Write something") or \
                await self._find_and_tap("What's on your mind")
            await self.cool_down(1.5)

            await self.controller.type_text(text)
            await self.cool_down(0.5)

            if images:
                for img in images:
                    await self._push_media_fb(img)
                await self._find_and_tap("Photo") or \
                    await self._find_and_tap("Photo/Video")
                await self.cool_down(2.0)
                for i in range(len(images)):
                    await self.controller.tap(200 + (i % 3) * 200, 500)
                    await self.cool_down(0.5)
                await self._find_and_tap("Done")
                await self.cool_down(1.0)

            await self._find_and_tap("Post")
            await self.cool_down(3.0)

            record.success = True
            await self._record_action("posts")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def join_group(self, group_name: str) -> ActionRecord:
        """Request to join a Facebook group."""
        record = ActionRecord(platform="facebook", category=ActionCategory.FOLLOW.value,
                              description=f"Join group: {group_name}")
        start = time.monotonic()

        try:
            await self.open_app()
            await self._find_and_tap("Groups")
            await self.cool_down(1.5)
            await self._find_and_tap("Search")
            await self.cool_down(0.5)
            await self.controller.type_text(group_name)
            await self.cool_down(2.0)
            await self.controller.tap(self.controller.resolution[0] // 2, 350)
            await self.cool_down(2.0)

            if await self._find_and_tap("Join") or \
               await self._find_and_tap("Join Group"):
                record.success = True
            await self.cool_down(1.0)

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def share_post(self, post_url: str, comment: Optional[str] = None) -> ActionRecord:
        """Share a post to your feed."""
        record = ActionRecord(platform="facebook", category=ActionCategory.SHARE.value,
                              description=f"Share: {post_url}")
        start = time.monotonic()

        if not await self._check_limit("shares"):
            record.error = "Daily share limit reached"
            return record

        try:
            await self.open_app()
            await self.controller._adb_shell(
                f"am start -a android.intent.action.VIEW -d '{post_url}'"
            )
            await self.cool_down(3.0)

            if await self._find_and_tap("Share"):
                await self.cool_down(1.5)
                if await self._find_and_tap("Share now") or \
                   await self._find_and_tap("Share to Feed"):
                    if comment:
                        comment_el = await self.controller.find_element(
                            text="Say something", partial_match=True
                        )
                        if comment_el:
                            await self.controller.tap_element(comment_el)
                            await self.controller.type_text(comment)
                            await self.cool_down(0.5)
                    await self._find_and_tap("Share") or \
                        await self._find_and_tap("Post")
                    record.success = True
                    await self._record_action("shares")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def respond_to_messages(
        self,
        auto_replies: Dict[str, str],
    ) -> ActionRecord:
        """Auto-reply to Messenger messages matching keywords."""
        record = ActionRecord(platform="facebook", category=ActionCategory.DM.value,
                              description=f"Auto-reply to {len(auto_replies)} patterns")
        start = time.monotonic()
        replied = 0

        try:
            await self.open_app()
            # Open Messenger
            await self._find_and_tap("Messenger") or \
                await self._find_and_tap("Chat")
            await self.cool_down(2.0)

            # Check recent conversations
            for keyword, reply in auto_replies.items():
                if not await self._check_limit("messages"):
                    break
                el = await self.controller.find_element(
                    text=keyword, partial_match=True
                )
                if el:
                    await self.controller.tap_element(el)
                    await self.cool_down(2.0)
                    msg_el = await self.controller.find_element(
                        text="Aa", partial_match=True
                    ) or await self.controller.find_element(
                        text="Type a message", partial_match=True
                    )
                    if msg_el:
                        await self.controller.tap_element(msg_el)
                        await self.controller.type_text(reply)
                        await self.cool_down(0.5)
                        await self._find_and_tap("Send") or \
                            await self.controller.press_enter()
                        replied += 1
                        await self._record_action("messages")
                    await self.safe_back()
                    await self.cool_down(1.0)

            record.success = True
            record.metadata = {"replied": replied}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def check_page_insights(self, page_name: str) -> AnalyticsSnapshot:
        """Extract Facebook page analytics via OCR."""
        snapshot = AnalyticsSnapshot(platform="facebook")

        try:
            await self.open_app()
            # Navigate to page
            await self._find_and_tap("Pages") or await self._find_and_tap("Menu")
            await self.cool_down(1.5)
            await self._find_and_tap(page_name)
            await self.cool_down(2.0)
            await self._find_and_tap("Insights") or \
                await self._find_and_tap("Professional dashboard")
            await self.cool_down(3.0)

            screenshot_path = await self.controller.screenshot()
            snapshot.screenshot_path = screenshot_path

            analysis = await self.vision.analyze_screen(
                "Extract Facebook page insights. Return JSON: page_likes, "
                "reach, engagement, post_impressions, page_views, followers.",
                screenshot_path=screenshot_path,
            )
            try:
                cleaned = analysis.raw_response.strip()
                if isinstance(cleaned, str) and cleaned.startswith("{"):
                    snapshot.metrics = json.loads(cleaned)
                else:
                    snapshot.metrics = {"raw": analysis.description}
            except (json.JSONDecodeError, TypeError):
                snapshot.metrics = {"raw": analysis.description}

        except Exception as exc:
            snapshot.metrics = {"error": str(exc)}

        return snapshot

    async def invite_to_page(self, page_name: str, count: int = 10) -> ActionRecord:
        """Invite friends to like a page."""
        record = ActionRecord(platform="facebook", category=ActionCategory.SHARE.value,
                              description=f"Invite {count} to {page_name}")
        start = time.monotonic()
        invited = 0

        try:
            await self.open_app()
            await self._find_and_tap("Pages") or await self._find_and_tap("Menu")
            await self.cool_down(1.5)
            await self._find_and_tap(page_name)
            await self.cool_down(2.0)

            if await self._find_and_tap("Invite") or \
               await self._find_and_tap("Invite friends"):
                await self.cool_down(2.0)
                for i in range(count):
                    invite_btns = await self.controller.find_elements(
                        text="Invite", partial_match=False
                    )
                    if invite_btns:
                        btn = invite_btns[0]
                        await self.controller.tap_element(btn)
                        invited += 1
                        await self.cool_down(1.0)
                    else:
                        await self.controller.scroll_down()
                        await self.cool_down(1.0)

            record.success = invited > 0
            record.metadata = {"invited": invited}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def _push_media_fb(self, path: str) -> None:
        """Push media file for Facebook."""
        filename = Path(path).name
        device_path = f"/sdcard/DCIM/{filename}"
        await self.controller._invoke_node("file.push", {
            "local_path": path, "device_path": device_path,
        })
        await self.controller._adb_shell(
            f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
            f"-d file://{device_path}"
        )
        await self.cool_down(1.0)


# ---------------------------------------------------------------------------
# TwitterBot (X)
# ---------------------------------------------------------------------------


class TwitterBot(BasePlatformBot):
    """Twitter/X automation via in-app interaction."""

    package_name = "com.twitter.android"
    app_label = "Twitter/X"

    async def post_tweet(
        self,
        text: str,
        images: Optional[List[str]] = None,
        poll_options: Optional[List[str]] = None,
    ) -> ActionRecord:
        """Compose and post a tweet."""
        record = ActionRecord(platform="twitter", category=ActionCategory.POST.value,
                              description=f"Tweet: {text[:50]}...")
        start = time.monotonic()

        if not await self._check_limit("tweets"):
            record.error = "Daily tweet limit reached"
            return record

        try:
            await self.open_app()
            await self.cool_down(1.0)

            # Tap compose button (floating + or quill icon)
            compose_el = await self.controller.find_element(
                content_desc="Compose", partial_match=True
            ) or await self.controller.find_element(
                content_desc="New post", partial_match=True
            )
            if compose_el:
                await self.controller.tap_element(compose_el)
            else:
                w = self.controller.resolution[0]
                h = self.controller.resolution[1]
                await self.controller.tap(w - 80, h - 160)
            await self.cool_down(1.5)

            # Type tweet
            await self.controller.type_text(text)
            await self.cool_down(0.5)

            # Add images
            if images:
                for img in images:
                    await self._push_media_tw(img)
                gallery_el = await self.controller.find_element(
                    content_desc="Gallery", partial_match=True
                ) or await self.controller.find_element(
                    content_desc="Media", partial_match=True
                )
                if gallery_el:
                    await self.controller.tap_element(gallery_el)
                    await self.cool_down(1.5)
                    for i in range(len(images)):
                        await self.controller.tap(200 + (i % 3) * 200, 500)
                        await self.cool_down(0.5)
                    await self._find_and_tap("Done")
                    await self.cool_down(1.0)

            # Add poll if specified
            if poll_options:
                poll_el = await self.controller.find_element(
                    content_desc="Poll", partial_match=True
                )
                if poll_el:
                    await self.controller.tap_element(poll_el)
                    await self.cool_down(1.0)
                    for i, option in enumerate(poll_options[:4]):
                        opt_el = await self.controller.find_element(
                            text=f"Choice {i + 1}", partial_match=True
                        )
                        if opt_el:
                            await self.controller.tap_element(opt_el)
                            await self.controller.type_text(option)
                            await self.cool_down(0.3)

            # Post
            await self._find_and_tap("Post") or \
                await self._find_and_tap("Tweet")
            await self.cool_down(3.0)

            record.success = True
            await self._record_action("tweets")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def reply_to_tweet(self, tweet_url: str, text: str) -> ActionRecord:
        """Reply to a specific tweet."""
        record = ActionRecord(platform="twitter", category=ActionCategory.COMMENT.value,
                              description=f"Reply: {text[:50]}...")
        start = time.monotonic()

        try:
            await self.open_app()
            await self.controller._adb_shell(
                f"am start -a android.intent.action.VIEW -d '{tweet_url}'"
            )
            await self.cool_down(3.0)

            reply_el = await self.controller.find_element(
                content_desc="Reply", partial_match=True
            )
            if reply_el:
                await self.controller.tap_element(reply_el)
                await self.cool_down(1.0)
                await self.controller.type_text(text)
                await self.cool_down(0.5)
                await self._find_and_tap("Reply") or \
                    await self._find_and_tap("Post")
                record.success = True

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def retweet(self, tweet_url: str) -> ActionRecord:
        """Retweet a post."""
        record = ActionRecord(platform="twitter", category=ActionCategory.SHARE.value,
                              description=f"Retweet: {tweet_url}")
        start = time.monotonic()

        if not await self._check_limit("retweets"):
            record.error = "Daily retweet limit reached"
            return record

        try:
            await self.open_app()
            await self.controller._adb_shell(
                f"am start -a android.intent.action.VIEW -d '{tweet_url}'"
            )
            await self.cool_down(3.0)

            rt_el = await self.controller.find_element(
                content_desc="Repost", partial_match=True
            ) or await self.controller.find_element(
                content_desc="Retweet", partial_match=True
            )
            if rt_el:
                await self.controller.tap_element(rt_el)
                await self.cool_down(1.0)
                await self._find_and_tap("Repost") or \
                    await self._find_and_tap("Retweet")
                record.success = True
                await self._record_action("retweets")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def quote_tweet(self, tweet_url: str, text: str) -> ActionRecord:
        """Quote retweet with comment."""
        record = ActionRecord(platform="twitter", category=ActionCategory.SHARE.value,
                              description=f"Quote: {text[:50]}...")
        start = time.monotonic()

        try:
            await self.open_app()
            await self.controller._adb_shell(
                f"am start -a android.intent.action.VIEW -d '{tweet_url}'"
            )
            await self.cool_down(3.0)

            rt_el = await self.controller.find_element(
                content_desc="Repost", partial_match=True
            ) or await self.controller.find_element(
                content_desc="Retweet", partial_match=True
            )
            if rt_el:
                await self.controller.tap_element(rt_el)
                await self.cool_down(1.0)
                await self._find_and_tap("Quote") or \
                    await self._find_and_tap("Quote Tweet")
                await self.cool_down(1.0)
                await self.controller.type_text(text)
                await self.cool_down(0.5)
                await self._find_and_tap("Post") or \
                    await self._find_and_tap("Tweet")
                record.success = True

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def like_tweets(
        self,
        hashtag: str,
        count: int = 10,
    ) -> ActionRecord:
        """Like tweets from a hashtag search."""
        record = ActionRecord(platform="twitter", category=ActionCategory.LIKE.value,
                              description=f"Like {count} from #{hashtag}")
        start = time.monotonic()
        liked = 0

        try:
            await self.open_app()
            # Search
            await self._find_and_tap("Search") or \
                await self._find_and_tap("Explore")
            await self.cool_down(1.5)
            search_el = await self.controller.find_element(
                text="Search", partial_match=True
            )
            if search_el:
                await self.controller.tap_element(search_el)
                await self.cool_down(0.5)
            await self.controller.type_text(f"#{hashtag}")
            await self.controller.press_enter()
            await self.cool_down(2.0)

            # Tap Latest tab
            await self._find_and_tap("Latest")
            await self.cool_down(1.5)

            for i in range(count):
                if not await self._check_limit("likes"):
                    break
                like_el = await self.controller.find_element(
                    content_desc="Like", partial_match=True
                )
                if like_el:
                    await self.controller.tap_element(like_el)
                    liked += 1
                    await self._record_action("likes")
                await self.controller.scroll_down(random.randint(300, 600))
                await self.behavior.random_delay(3.0, 8.0)

            record.success = True
            record.metadata = {"liked": liked, "hashtag": hashtag}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def follow_from_topic(self, topic: str, count: int = 10) -> ActionRecord:
        """Follow users from a topic feed."""
        record = ActionRecord(platform="twitter", category=ActionCategory.FOLLOW.value,
                              description=f"Follow {count} from topic: {topic}")
        start = time.monotonic()
        followed = 0

        try:
            await self.open_app()
            await self._find_and_tap("Search") or \
                await self._find_and_tap("Explore")
            await self.cool_down(1.5)
            search_el = await self.controller.find_element(
                text="Search", partial_match=True
            )
            if search_el:
                await self.controller.tap_element(search_el)
            await self.controller.type_text(topic)
            await self.controller.press_enter()
            await self.cool_down(2.0)

            # Go to People tab
            await self._find_and_tap("People")
            await self.cool_down(1.5)

            for i in range(count):
                if not await self._check_limit("follows"):
                    break
                follow_btn = await self.controller.find_element(
                    text="Follow", partial_match=False
                )
                if follow_btn:
                    await self.controller.tap_element(follow_btn)
                    followed += 1
                    await self._record_action("follows")
                    await self.behavior.random_delay(5.0, 12.0)
                else:
                    await self.controller.scroll_down()
                    await self.cool_down(1.0)

            record.success = True
            record.metadata = {"followed": followed, "topic": topic}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def check_analytics(self) -> AnalyticsSnapshot:
        """Extract Twitter analytics via OCR."""
        snapshot = AnalyticsSnapshot(platform="twitter")

        try:
            await self.open_app()
            # Profile -> Analytics
            await self._find_and_tap("Profile") or \
                await self.controller.tap(60, 80)
            await self.cool_down(2.0)

            # Look for analytics in menu
            await self._find_and_tap("Analytics") or \
                await self._find_and_tap("Creator Studio")
            await self.cool_down(3.0)

            screenshot_path = await self.controller.screenshot()
            snapshot.screenshot_path = screenshot_path

            analysis = await self.vision.analyze_screen(
                "Extract Twitter/X analytics. Return JSON: impressions, "
                "engagements, followers, profile_visits, mentions.",
                screenshot_path=screenshot_path,
            )
            try:
                cleaned = analysis.raw_response.strip()
                if isinstance(cleaned, str) and cleaned.startswith("{"):
                    snapshot.metrics = json.loads(cleaned)
                else:
                    snapshot.metrics = {"raw": analysis.description}
            except (json.JSONDecodeError, TypeError):
                snapshot.metrics = {"raw": analysis.description}

        except Exception as exc:
            snapshot.metrics = {"error": str(exc)}

        return snapshot

    async def send_dm(self, username: str, message: str) -> ActionRecord:
        """Send a direct message on Twitter/X."""
        record = ActionRecord(platform="twitter", category=ActionCategory.DM.value,
                              description=f"DM to @{username}")
        start = time.monotonic()

        if not await self._check_limit("dms"):
            record.error = "Daily DM limit reached"
            return record

        try:
            await self.open_app()
            # Tap Messages tab
            await self._find_and_tap("Messages")
            await self.cool_down(1.5)

            # New message
            compose_el = await self.controller.find_element(
                content_desc="New message", partial_match=True
            ) or await self.controller.find_element(
                content_desc="Compose", partial_match=True
            )
            if compose_el:
                await self.controller.tap_element(compose_el)
            await self.cool_down(1.0)

            # Search for user
            await self.controller.type_text(username)
            await self.cool_down(2.0)
            await self.controller.tap(self.controller.resolution[0] // 2, 350)
            await self.cool_down(1.0)
            await self._find_and_tap("Next")
            await self.cool_down(1.0)

            # Type and send message
            msg_el = await self.controller.find_element(
                text="Start a new message", partial_match=True
            )
            if msg_el:
                await self.controller.tap_element(msg_el)
            await self.controller.type_text(message)
            await self.cool_down(0.5)
            await self._find_and_tap("Send") or \
                await self.controller.press_enter()

            record.success = True
            await self._record_action("dms")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def _push_media_tw(self, path: str) -> None:
        """Push media file for Twitter."""
        filename = Path(path).name
        device_path = f"/sdcard/DCIM/{filename}"
        await self.controller._invoke_node("file.push", {
            "local_path": path, "device_path": device_path,
        })
        await self.controller._adb_shell(
            f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
            f"-d file://{device_path}"
        )
        await self.cool_down(1.0)


# ---------------------------------------------------------------------------
# LinkedInBot
# ---------------------------------------------------------------------------


class LinkedInBot(BasePlatformBot):
    """LinkedIn automation via in-app interaction."""

    package_name = "com.linkedin.android"
    app_label = "LinkedIn"

    async def create_post(
        self,
        text: str,
        images: Optional[List[str]] = None,
    ) -> ActionRecord:
        """Create a LinkedIn feed post."""
        record = ActionRecord(platform="linkedin", category=ActionCategory.POST.value,
                              description=f"Post: {text[:50]}...")
        start = time.monotonic()

        if not await self._check_limit("posts"):
            record.error = "Daily post limit reached"
            return record

        try:
            await self.open_app()
            await self.cool_down(1.0)

            # Tap Post/Create button
            await self._find_and_tap("Post") or \
                await self._find_and_tap("Create a post")
            await self.cool_down(1.5)

            # Type content
            await self.controller.type_text(text)
            await self.cool_down(0.5)

            # Add images
            if images:
                for img in images:
                    await self._push_media_li(img)
                img_btn = await self.controller.find_element(
                    content_desc="Photo", partial_match=True
                ) or await self.controller.find_element(
                    content_desc="Image", partial_match=True
                )
                if img_btn:
                    await self.controller.tap_element(img_btn)
                    await self.cool_down(1.5)
                    for i in range(len(images)):
                        await self.controller.tap(200 + (i % 3) * 200, 500)
                        await self.cool_down(0.5)
                    await self._find_and_tap("Done") or \
                        await self._find_and_tap("Add")
                    await self.cool_down(1.0)

            # Post
            await self._find_and_tap("Post")
            await self.cool_down(3.0)

            record.success = True
            await self._record_action("posts")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def share_article(self, url: str, comment: str) -> ActionRecord:
        """Share a link/article with a comment."""
        record = ActionRecord(platform="linkedin", category=ActionCategory.SHARE.value,
                              description=f"Share: {url[:50]}...")
        start = time.monotonic()

        try:
            await self.open_app()
            await self._find_and_tap("Post") or \
                await self._find_and_tap("Create a post")
            await self.cool_down(1.5)

            # Type comment and URL
            await self.controller.type_text(f"{comment}\n\n{url}")
            await self.cool_down(1.0)

            # Post
            await self._find_and_tap("Post")
            await self.cool_down(3.0)

            record.success = True
            await self._record_action("posts")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def connect_with(self, profile_url: str, note: Optional[str] = None) -> ActionRecord:
        """Send a connection request."""
        record = ActionRecord(platform="linkedin", category=ActionCategory.FOLLOW.value,
                              description=f"Connect: {profile_url}")
        start = time.monotonic()

        if not await self._check_limit("connections"):
            record.error = "Daily connection limit reached"
            return record

        try:
            await self.open_app()
            await self.controller._adb_shell(
                f"am start -a android.intent.action.VIEW -d '{profile_url}'"
            )
            await self.cool_down(3.0)

            if await self._find_and_tap("Connect"):
                await self.cool_down(1.0)
                if note:
                    if await self._find_and_tap("Add a note"):
                        await self.cool_down(0.5)
                        await self.controller.type_text(note)
                        await self.cool_down(0.5)
                await self._find_and_tap("Send") or \
                    await self._find_and_tap("Done")
                record.success = True
                await self._record_action("connections")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def engage_feed(
        self,
        duration_minutes: int = 10,
        like_ratio: float = 0.3,
    ) -> ActionRecord:
        """Scroll feed and like posts at the given ratio."""
        record = ActionRecord(platform="linkedin", category=ActionCategory.LIKE.value,
                              description=f"Feed engagement {duration_minutes}min")
        start = time.monotonic()
        liked = 0

        try:
            await self.open_app()
            await self._find_and_tap("Home")
            await self.cool_down(1.5)

            end_time = time.monotonic() + duration_minutes * 60
            while time.monotonic() < end_time:
                if random.random() < like_ratio:
                    if await self._check_limit("likes"):
                        like_el = await self.controller.find_element(
                            content_desc="Like", partial_match=True
                        )
                        if like_el:
                            await self.controller.tap_element(like_el)
                            liked += 1
                            await self._record_action("likes")

                await self.controller.scroll_down(random.randint(400, 700))
                await self.behavior.random_delay(3.0, 8.0)

            record.success = True
            record.metadata = {"liked": liked}

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def check_profile_views(self) -> AnalyticsSnapshot:
        """Check who viewed your LinkedIn profile."""
        snapshot = AnalyticsSnapshot(platform="linkedin")

        try:
            await self.open_app()
            await self._find_and_tap("Profile") or \
                await self.controller.tap(60, 80)
            await self.cool_down(2.0)

            await self._find_and_tap("Who viewed your profile") or \
                await self._find_and_tap("profile views")
            await self.cool_down(3.0)

            screenshot_path = await self.controller.screenshot()
            snapshot.screenshot_path = screenshot_path

            analysis = await self.vision.analyze_screen(
                "Extract LinkedIn profile view data. Return JSON: "
                "total_views, viewer_names[], viewer_companies[], trend.",
                screenshot_path=screenshot_path,
            )
            try:
                cleaned = analysis.raw_response.strip()
                if isinstance(cleaned, str) and cleaned.startswith("{"):
                    snapshot.metrics = json.loads(cleaned)
                else:
                    snapshot.metrics = {"raw": analysis.description}
            except (json.JSONDecodeError, TypeError):
                snapshot.metrics = {"raw": analysis.description}

        except Exception as exc:
            snapshot.metrics = {"error": str(exc)}

        return snapshot

    async def send_message(self, profile_url: str, message: str) -> ActionRecord:
        """Send a message to a LinkedIn connection."""
        record = ActionRecord(platform="linkedin", category=ActionCategory.DM.value,
                              description=f"Message: {profile_url[:40]}...")
        start = time.monotonic()

        if not await self._check_limit("messages"):
            record.error = "Daily message limit reached"
            return record

        try:
            await self.open_app()
            await self.controller._adb_shell(
                f"am start -a android.intent.action.VIEW -d '{profile_url}'"
            )
            await self.cool_down(3.0)

            await self._find_and_tap("Message")
            await self.cool_down(2.0)

            msg_el = await self.controller.find_element(
                text="Write a message", partial_match=True
            )
            if msg_el:
                await self.controller.tap_element(msg_el)
            await self.controller.type_text(message)
            await self.cool_down(0.5)
            await self._find_and_tap("Send") or \
                await self.controller.press_enter()

            record.success = True
            await self._record_action("messages")

        except Exception as exc:
            record.error = str(exc)

        record.duration_ms = (time.monotonic() - start) * 1000
        return record

    async def _push_media_li(self, path: str) -> None:
        """Push media file for LinkedIn."""
        filename = Path(path).name
        device_path = f"/sdcard/DCIM/{filename}"
        await self.controller._invoke_node("file.push", {
            "local_path": path, "device_path": device_path,
        })
        await self.controller._adb_shell(
            f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
            f"-d file://{device_path}"
        )
        await self.cool_down(1.0)
