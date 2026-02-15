"""
Social Media Agent — OpenClaw Empire AI Social Media Manager

Strategic intelligence layer on top of social_automation.py bots.
Generates content strategies via Sonnet, creates contextual comments
via Haiku, manages DMs, extracts analytics via OCR, tracks growth,
analyzes competitors, and coordinates multi-account engagement.

Data persisted to: data/social_agent/

Usage:
    from src.social_media_agent import SocialMediaAgent, get_social_agent

    agent = get_social_agent()
    strategy = await agent.generate_strategy("instagram", "witchcraft")
    await agent.smart_engage("instagram", strategy_id="abc123")
    analytics = await agent.extract_analytics("instagram")

CLI:
    python -m src.social_media_agent strategy --platform instagram --niche witchcraft
    python -m src.social_media_agent engage --platform instagram --duration 30
    python -m src.social_media_agent analytics --platform instagram
    python -m src.social_media_agent growth --platform instagram --days 30
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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("social_media_agent")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "social_agent"


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

class SocialPlatform(str, Enum):
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    PINTEREST = "pinterest"
    LINKEDIN = "linkedin"
    YOUTUBE = "youtube"
    REDDIT = "reddit"
    THREADS = "threads"


class ContentPillar(str, Enum):
    EDUCATIONAL = "educational"
    ENTERTAINING = "entertaining"
    INSPIRATIONAL = "inspirational"
    PROMOTIONAL = "promotional"
    COMMUNITY = "community"
    BTS = "behind_the_scenes"
    UGC = "user_generated_content"
    TRENDING = "trending"


class EngagementType(str, Enum):
    LIKE = "like"
    COMMENT = "comment"
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"
    SHARE = "share"
    SAVE = "save"
    DM = "dm"
    STORY_REACT = "story_react"
    VIEW = "view"


class GrowthTactic(str, Enum):
    FOLLOW_UNFOLLOW = "follow_unfollow"
    ENGAGEMENT_PODS = "engagement_pods"
    HASHTAG_TARGETING = "hashtag_targeting"
    COMPETITOR_AUDIENCE = "competitor_audience"
    TRENDING_CONTENT = "trending_content"
    COLLABORATION = "collaboration"
    GIVEAWAY = "giveaway"
    CROSS_PLATFORM = "cross_platform"


class DMCategory(str, Enum):
    INQUIRY = "inquiry"
    COLLABORATION = "collaboration"
    SPAM = "spam"
    SUPPORT = "support"
    PERSONAL = "personal"
    IMPORTANT = "important"
    AUTO_RESPONSE = "auto_response"


@dataclass
class ContentStrategy:
    """A content strategy for a platform/niche."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    platform: SocialPlatform = SocialPlatform.INSTAGRAM
    niche: str = ""
    pillars: List[ContentPillar] = field(default_factory=list)
    posting_frequency: str = ""  # e.g., "3x/week"
    optimal_times: List[str] = field(default_factory=list)  # e.g., ["09:00", "18:00"]
    hashtag_sets: Dict[str, List[str]] = field(default_factory=dict)
    content_ideas: List[Dict[str, str]] = field(default_factory=list)
    target_audience: Dict[str, Any] = field(default_factory=dict)
    growth_tactics: List[GrowthTactic] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    active: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        d["pillars"] = [p.value for p in self.pillars]
        d["growth_tactics"] = [t.value for t in self.growth_tactics]
        return d


@dataclass
class EngagementAction:
    """A recorded engagement action."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    platform: SocialPlatform = SocialPlatform.INSTAGRAM
    action_type: EngagementType = EngagementType.LIKE
    target_user: str = ""
    target_post: str = ""
    comment_text: str = ""
    success: bool = True
    timestamp: str = field(default_factory=_now_iso)
    strategy_id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        d["action_type"] = self.action_type.value
        return d


@dataclass
class AnalyticsSnapshot:
    """A snapshot of platform analytics."""
    platform: SocialPlatform = SocialPlatform.INSTAGRAM
    followers: int = 0
    following: int = 0
    posts: int = 0
    engagement_rate: float = 0.0
    reach: int = 0
    impressions: int = 0
    profile_views: int = 0
    website_clicks: int = 0
    top_posts: List[Dict[str, Any]] = field(default_factory=list)
    audience_demographics: Dict[str, Any] = field(default_factory=dict)
    captured_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        return d


@dataclass
class GrowthRecord:
    """Daily growth tracking record."""
    date: str = ""
    platform: SocialPlatform = SocialPlatform.INSTAGRAM
    followers: int = 0
    following: int = 0
    posts: int = 0
    engagement_rate: float = 0.0
    new_followers: int = 0
    lost_followers: int = 0
    actions_taken: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        return d


@dataclass
class CompetitorProfile:
    """Competitor analysis data."""
    username: str = ""
    platform: SocialPlatform = SocialPlatform.INSTAGRAM
    followers: int = 0
    following: int = 0
    posts: int = 0
    avg_engagement: float = 0.0
    content_themes: List[str] = field(default_factory=list)
    posting_frequency: str = ""
    top_hashtags: List[str] = field(default_factory=list)
    analyzed_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        return d


@dataclass
class DMConversation:
    """A DM conversation summary."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    platform: SocialPlatform = SocialPlatform.INSTAGRAM
    user: str = ""
    category: DMCategory = DMCategory.INQUIRY
    messages: List[Dict[str, str]] = field(default_factory=list)
    auto_responded: bool = False
    escalated: bool = False
    last_message: str = ""
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        d["category"] = self.category.value
        return d


# Comment templates by niche
COMMENT_TEMPLATES = {
    "witchcraft": [
        "Beautiful energy in this! {specific}",
        "This resonates so deeply. {specific}",
        "Love the {specific} — blessed be!",
        "Such powerful work! {specific}",
    ],
    "tech": [
        "Great insights on {specific}!",
        "This is really helpful — {specific}",
        "Interesting approach to {specific}",
    ],
    "default": [
        "Love this! {specific}",
        "Great content — {specific}",
        "Really appreciate {specific}",
        "This is amazing! {specific}",
    ],
}

# Auto-response templates
AUTO_RESPONSES = {
    DMCategory.INQUIRY: "Thanks for reaching out! I'll get back to you shortly.",
    DMCategory.COLLABORATION: "Thanks for the collab interest! Let me review and get back to you.",
    DMCategory.SUPPORT: "Thanks for contacting us! We'll look into this and respond soon.",
}


# ---------------------------------------------------------------------------
# SocialMediaAgent
# ---------------------------------------------------------------------------

class SocialMediaAgent:
    """
    AI-powered social media manager.

    Wraps social_automation.py bots with strategic intelligence:
    content strategy generation, smart engagement with contextual
    comments, DM management, analytics extraction, competitor
    analysis, and multi-account coordination.

    Usage:
        agent = get_social_agent()
        strategy = await agent.generate_strategy("instagram", "witchcraft")
        await agent.smart_engage("instagram")
    """

    def __init__(
        self,
        social_bot: Any = None,
        memory: Any = None,
        calendar: Any = None,
        account_mgr: Any = None,
        data_dir: Optional[Path] = None,
    ):
        self._social_bot = social_bot
        self._memory = memory
        self._calendar = calendar
        self._account_mgr = account_mgr
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._strategies: Dict[str, ContentStrategy] = {}
        self._actions: List[EngagementAction] = []
        self._analytics: Dict[str, List[AnalyticsSnapshot]] = {}
        self._growth: Dict[str, List[GrowthRecord]] = {}
        self._competitors: Dict[str, CompetitorProfile] = {}
        self._dms: Dict[str, DMConversation] = {}
        self._daily_limits: Dict[str, Dict[str, int]] = {}

        self._load_state()
        logger.info("SocialMediaAgent initialized (%d strategies)", len(self._strategies))

    # ── Property helpers ──

    @property
    def social_bot(self):
        if self._social_bot is None:
            try:
                from src.social_automation import get_social_bot
                self._social_bot = get_social_bot()
            except ImportError:
                logger.warning("SocialBot not available")
        return self._social_bot

    @property
    def memory(self):
        if self._memory is None:
            try:
                from src.agent_memory import get_memory
                self._memory = get_memory()
            except ImportError:
                pass
        return self._memory

    @property
    def calendar(self):
        if self._calendar is None:
            try:
                from src.content_calendar import get_calendar
                self._calendar = get_calendar()
            except ImportError:
                pass
        return self._calendar

    # ── Persistence ──

    def _load_state(self) -> None:
        state = _load_json(self._data_dir / "state.json")
        for sid, data in state.get("strategies", {}).items():
            if isinstance(data, dict):
                pillars = [ContentPillar(p) for p in data.pop("pillars", [])]
                tactics = [GrowthTactic(t) for t in data.pop("growth_tactics", [])]
                p = data.pop("platform", "instagram")
                self._strategies[sid] = ContentStrategy(
                    platform=SocialPlatform(p), pillars=pillars, growth_tactics=tactics, **data
                )
        self._actions = [
            EngagementAction(**a) if isinstance(a, dict) else a
            for a in state.get("actions", [])[-1000:]
        ]
        for platform, snapshots in state.get("analytics", {}).items():
            self._analytics[platform] = [
                AnalyticsSnapshot(**s) if isinstance(s, dict) else s
                for s in snapshots[-30:]
            ]
        for platform, records in state.get("growth", {}).items():
            self._growth[platform] = [
                GrowthRecord(**r) if isinstance(r, dict) else r
                for r in records[-90:]
            ]
        for key, data in state.get("competitors", {}).items():
            if isinstance(data, dict):
                self._competitors[key] = CompetitorProfile(**data)
        self._daily_limits = state.get("daily_limits", {})

    def _save_state(self) -> None:
        _save_json(self._data_dir / "state.json", {
            "strategies": {k: v.to_dict() for k, v in self._strategies.items()},
            "actions": [a.to_dict() for a in self._actions[-1000:]],
            "analytics": {k: [s.to_dict() for s in v[-30:]] for k, v in self._analytics.items()},
            "growth": {k: [r.to_dict() for r in v[-90:]] for k, v in self._growth.items()},
            "competitors": {k: v.to_dict() for k, v in self._competitors.items()},
            "daily_limits": self._daily_limits,
            "updated_at": _now_iso(),
        })

    # ── AI helpers ──

    async def _call_haiku(self, prompt: str, max_tokens: int = 200) -> str:
        """Call Haiku for lightweight AI tasks (comments, classification)."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                system=[{
                    "type": "text",
                    "text": "You are a social media engagement assistant. Be concise, authentic, and human-sounding. Never use emojis excessively.",
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("Haiku call failed: %s", exc)
            return ""

    async def _call_sonnet(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call Sonnet for strategy and analysis tasks."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=[{
                    "type": "text",
                    "text": "You are a social media strategist. Provide actionable, data-driven advice. Return JSON when asked.",
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("Sonnet call failed: %s", exc)
            return ""

    # ── Strategy generation ──

    async def generate_strategy(
        self,
        platform: str,
        niche: str,
        target_followers: int = 10000,
        current_followers: int = 0,
    ) -> ContentStrategy:
        """
        Generate a content strategy using Sonnet.

        Args:
            platform: Target platform.
            niche: Content niche (e.g., "witchcraft", "smart home").
            target_followers: Goal follower count.
            current_followers: Current follower count.

        Returns:
            ContentStrategy with pillars, posting schedule, hashtags, etc.
        """
        prompt = (
            f"Create a social media content strategy in JSON format for:\n"
            f"Platform: {platform}\nNiche: {niche}\n"
            f"Current followers: {current_followers}\nTarget: {target_followers}\n\n"
            f"Return JSON with keys: pillars (list of content types), "
            f"posting_frequency (string), optimal_times (list of HH:MM strings), "
            f"hashtag_sets (dict of set_name -> list of hashtags), "
            f"content_ideas (list of {{title, pillar, description}}), "
            f"growth_tactics (list of tactic names)"
        )

        response = await self._call_sonnet(prompt, max_tokens=1500)

        strategy = ContentStrategy(
            platform=SocialPlatform(platform) if platform in [p.value for p in SocialPlatform] else SocialPlatform.INSTAGRAM,
            niche=niche,
        )

        # Parse AI response
        try:
            # Extract JSON from response
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]

            data = json.loads(json_match)

            strategy.pillars = [
                ContentPillar(p) for p in data.get("pillars", [])
                if p in [cp.value for cp in ContentPillar]
            ]
            strategy.posting_frequency = data.get("posting_frequency", "3x/week")
            strategy.optimal_times = data.get("optimal_times", ["09:00", "12:00", "18:00"])
            strategy.hashtag_sets = data.get("hashtag_sets", {})
            strategy.content_ideas = data.get("content_ideas", [])
            strategy.growth_tactics = [
                GrowthTactic(t) for t in data.get("growth_tactics", [])
                if t in [gt.value for gt in GrowthTactic]
            ]
            strategy.target_audience = data.get("target_audience", {
                "niche": niche,
                "target_followers": target_followers,
            })
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to parse strategy response: %s", exc)
            # Fallback defaults
            strategy.pillars = [ContentPillar.EDUCATIONAL, ContentPillar.ENTERTAINING, ContentPillar.COMMUNITY]
            strategy.posting_frequency = "3x/week"
            strategy.optimal_times = ["09:00", "12:00", "18:00"]
            strategy.growth_tactics = [GrowthTactic.HASHTAG_TARGETING, GrowthTactic.TRENDING_CONTENT]

        self._strategies[strategy.id] = strategy
        self._save_state()

        # Store in memory
        if self.memory:
            try:
                self.memory.store_sync(
                    content=f"Generated {platform} strategy for {niche}: "
                            f"{len(strategy.pillars)} pillars, {strategy.posting_frequency}",
                    memory_type="task_result",
                    tags=["strategy", platform, niche],
                )
            except Exception:
                pass

        logger.info("Strategy generated: %s for %s/%s", strategy.id, platform, niche)
        return strategy

    # ── Smart engagement ──

    async def smart_engage(
        self,
        platform: str,
        strategy_id: str = "",
        duration_minutes: int = 30,
        max_actions: int = 50,
    ) -> Dict[str, Any]:
        """
        Run a smart engagement session.

        Likes, comments, and follows based on the content strategy.
        Uses Haiku to generate contextual comments.

        Args:
            platform: Target platform.
            strategy_id: Strategy to follow (uses latest if empty).
            duration_minutes: Session duration.
            max_actions: Maximum actions to take.

        Returns:
            Dict with session results.
        """
        # Get strategy
        strategy = None
        if strategy_id:
            strategy = self._strategies.get(strategy_id)
        if not strategy:
            # Find latest strategy for this platform
            for s in sorted(self._strategies.values(), key=lambda x: x.created_at, reverse=True):
                if s.platform.value == platform and s.active:
                    strategy = s
                    break

        if not strategy:
            return {"success": False, "error": f"No active strategy for {platform}"}

        if not self.social_bot:
            return {"success": False, "error": "SocialBot not available"}

        # Check daily limits
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        limit_key = f"{platform}:{today}"
        daily_counts = self._daily_limits.get(limit_key, {})
        total_today = sum(daily_counts.values())

        if total_today >= max_actions:
            return {"success": True, "message": "Daily action limit reached", "today_actions": total_today}

        session_actions = []
        start_time = time.monotonic()
        remaining = max_actions - total_today

        # Engagement mix: 50% likes, 25% comments, 15% follows, 10% saves
        action_mix = {
            EngagementType.LIKE: 0.50,
            EngagementType.COMMENT: 0.25,
            EngagementType.FOLLOW: 0.15,
            EngagementType.SAVE: 0.10,
        }

        bot = self.social_bot
        platform_bot = getattr(bot, platform, None) if bot else None

        actions_taken = 0
        while (actions_taken < remaining and
               time.monotonic() - start_time < duration_minutes * 60):

            # Pick action type
            r = random.random()
            cumulative = 0
            chosen_type = EngagementType.LIKE
            for action_type, weight in action_mix.items():
                cumulative += weight
                if r <= cumulative:
                    chosen_type = action_type
                    break

            try:
                action = EngagementAction(
                    platform=SocialPlatform(platform),
                    action_type=chosen_type,
                    strategy_id=strategy.id,
                )

                if chosen_type == EngagementType.LIKE:
                    if platform_bot and hasattr(platform_bot, 'like_post'):
                        # Browse feed and like
                        result = await platform_bot.like_post("")
                        action.success = getattr(result, 'success', True)
                    else:
                        action.success = False

                elif chosen_type == EngagementType.COMMENT:
                    # Generate contextual comment
                    comment = await self._generate_comment(platform, strategy.niche)
                    action.comment_text = comment
                    if platform_bot and hasattr(platform_bot, 'comment_on_post'):
                        result = await platform_bot.comment_on_post("", comment)
                        action.success = getattr(result, 'success', True)
                    else:
                        action.success = False

                elif chosen_type == EngagementType.FOLLOW:
                    if platform_bot and hasattr(platform_bot, 'follow_user'):
                        result = await platform_bot.follow_user("")
                        action.success = getattr(result, 'success', True)
                    else:
                        action.success = False

                elif chosen_type == EngagementType.SAVE:
                    # Save/bookmark action
                    action.success = True  # Placeholder

                self._actions.append(action)
                session_actions.append(action.to_dict())
                actions_taken += 1

                # Update daily limits
                action_key = chosen_type.value
                daily_counts[action_key] = daily_counts.get(action_key, 0) + 1
                self._daily_limits[limit_key] = daily_counts

                # Random delay between actions (human-like)
                await asyncio.sleep(random.uniform(5.0, 30.0))

            except Exception as exc:
                logger.warning("Engagement action failed: %s", exc)
                actions_taken += 1

        self._save_state()

        duration = time.monotonic() - start_time
        logger.info("Engagement session: %d actions in %.0f seconds", actions_taken, duration)
        return {
            "success": True,
            "platform": platform,
            "actions_taken": actions_taken,
            "duration_seconds": round(duration),
            "action_breakdown": daily_counts,
        }

    async def _generate_comment(self, platform: str, niche: str) -> str:
        """Generate a contextual comment using Haiku."""
        prompt = (
            f"Write a single genuine, human-sounding comment for a {niche} post "
            f"on {platform}. Make it specific and authentic. Max 100 characters. "
            f"No hashtags, no emojis, no generic phrases."
        )
        comment = await self._call_haiku(prompt, max_tokens=50)
        if comment:
            return comment

        # Fallback to templates
        templates = COMMENT_TEMPLATES.get(niche, COMMENT_TEMPLATES["default"])
        template = random.choice(templates)
        return template.replace("{specific}", "this content")

    # ── Analytics ──

    async def extract_analytics(self, platform: str) -> AnalyticsSnapshot:
        """
        Extract analytics from a platform via OCR.

        Opens the analytics/insights section of the platform app
        and reads the data via vision analysis.

        Args:
            platform: Target platform.

        Returns:
            AnalyticsSnapshot with extracted metrics.
        """
        snapshot = AnalyticsSnapshot(
            platform=SocialPlatform(platform) if platform in [p.value for p in SocialPlatform] else SocialPlatform.INSTAGRAM,
        )

        if self.social_bot:
            try:
                analytics = getattr(self.social_bot, 'analytics', None)
                if analytics and hasattr(analytics, 'scrape_analytics'):
                    result = await analytics.scrape_analytics(platform)
                    if result:
                        snapshot.followers = getattr(result, 'followers', 0)
                        snapshot.following = getattr(result, 'following', 0)
                        snapshot.posts = getattr(result, 'posts', 0)
                        snapshot.engagement_rate = getattr(result, 'engagement_rate', 0.0)
                        snapshot.reach = getattr(result, 'reach', 0)
                        snapshot.impressions = getattr(result, 'impressions', 0)
            except Exception as exc:
                logger.warning("Analytics extraction failed: %s", exc)

        # Store snapshot
        if platform not in self._analytics:
            self._analytics[platform] = []
        self._analytics[platform].append(snapshot)
        self._save_state()

        logger.info("Analytics snapshot for %s: %d followers, %.2f%% ER",
                     platform, snapshot.followers, snapshot.engagement_rate)
        return snapshot

    def get_analytics_history(
        self, platform: str, limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Get analytics history for a platform."""
        snapshots = self._analytics.get(platform, [])
        return [s.to_dict() for s in snapshots[-limit:]]

    # ── Growth tracking ──

    async def track_growth(self, platform: str) -> GrowthRecord:
        """Record daily growth metrics."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        snapshot = await self.extract_analytics(platform)

        # Calculate changes from previous record
        previous_records = self._growth.get(platform, [])
        prev = previous_records[-1] if previous_records else None

        record = GrowthRecord(
            date=today,
            platform=SocialPlatform(platform) if platform in [p.value for p in SocialPlatform] else SocialPlatform.INSTAGRAM,
            followers=snapshot.followers,
            following=snapshot.following,
            posts=snapshot.posts,
            engagement_rate=snapshot.engagement_rate,
            new_followers=snapshot.followers - prev.followers if prev else 0,
            lost_followers=max(0, (prev.followers - snapshot.followers)) if prev else 0,
            actions_taken=sum(
                self._daily_limits.get(f"{platform}:{today}", {}).values()
            ),
        )

        if platform not in self._growth:
            self._growth[platform] = []
        self._growth[platform].append(record)
        self._save_state()

        return record

    def get_growth_history(
        self, platform: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get growth history for a platform."""
        records = self._growth.get(platform, [])
        return [r.to_dict() for r in records[-days:]]

    # ── Competitor analysis ──

    async def analyze_competitor(
        self, platform: str, username: str
    ) -> CompetitorProfile:
        """Analyze a competitor's profile."""
        profile = CompetitorProfile(
            username=username,
            platform=SocialPlatform(platform) if platform in [p.value for p in SocialPlatform] else SocialPlatform.INSTAGRAM,
        )

        # Use AI to analyze if social_bot has profile viewing
        prompt = (
            f"What are the key competitive advantages of analyzing a {platform} "
            f"account like @{username}? List 3 content themes they likely use."
        )
        analysis = await self._call_haiku(prompt, max_tokens=150)
        if analysis:
            profile.content_themes = [analysis[:100]]

        key = f"{platform}:{username}"
        self._competitors[key] = profile
        self._save_state()

        return profile

    def list_competitors(self, platform: str = "") -> List[Dict[str, Any]]:
        """List tracked competitors."""
        competitors = list(self._competitors.values())
        if platform:
            competitors = [c for c in competitors if c.platform.value == platform]
        return [c.to_dict() for c in competitors]

    # ── DM management ──

    async def manage_dms(
        self, platform: str, auto_respond: bool = True
    ) -> Dict[str, Any]:
        """
        Check and manage DMs for a platform.

        Reads DMs via OCR, categorizes them, and auto-responds
        to common inquiries.

        Args:
            platform: Target platform.
            auto_respond: Whether to auto-respond to common queries.

        Returns:
            Dict with DM processing results.
        """
        processed = []
        escalated = []

        # In a real implementation, this would open DMs and read them via OCR
        # For now, we process any tracked DMs
        for dm in self._dms.values():
            if dm.platform.value != platform:
                continue

            if auto_respond and not dm.auto_responded:
                response = AUTO_RESPONSES.get(dm.category)
                if response:
                    dm.auto_responded = True
                    processed.append(dm.id)
                elif dm.category == DMCategory.IMPORTANT:
                    dm.escalated = True
                    escalated.append(dm.id)

        self._save_state()
        return {
            "success": True,
            "platform": platform,
            "processed": len(processed),
            "escalated": len(escalated),
        }

    def get_dm_conversations(self, platform: str = "") -> List[Dict[str, Any]]:
        """Get DM conversations."""
        dms = list(self._dms.values())
        if platform:
            dms = [d for d in dms if d.platform.value == platform]
        return [d.to_dict() for d in dms]

    # ── Multi-account coordination ──

    async def coordinate_posting(
        self,
        content: str,
        platforms: List[str],
        adapt_per_platform: bool = True,
    ) -> Dict[str, Any]:
        """
        Coordinate posting across multiple platforms.

        Adapts content for each platform's format and audience.

        Args:
            content: Base content to post.
            platforms: List of platform names.
            adapt_per_platform: Whether to adapt content per platform.

        Returns:
            Dict with posting results per platform.
        """
        results = {}
        for platform in platforms:
            adapted = content
            if adapt_per_platform:
                prompt = (
                    f"Adapt this content for {platform} (keep under 280 chars for Twitter, "
                    f"add hashtags for Instagram, be professional for LinkedIn):\n\n{content}"
                )
                adapted = await self._call_haiku(prompt, max_tokens=200) or content

            results[platform] = {
                "content": adapted[:500],
                "status": "prepared",
            }

        self._save_state()
        return {"success": True, "platforms": results}

    # ── Statistics ──

    def stats(self) -> Dict[str, Any]:
        """Get social media agent statistics."""
        return {
            "strategies": len(self._strategies),
            "total_actions": len(self._actions),
            "analytics_snapshots": sum(len(v) for v in self._analytics.values()),
            "growth_records": sum(len(v) for v in self._growth.values()),
            "competitors_tracked": len(self._competitors),
            "dm_conversations": len(self._dms),
            "platforms": list(set(
                a.platform.value for a in self._actions
            )),
            "actions_today": {
                k: sum(v.values())
                for k, v in self._daily_limits.items()
                if datetime.now(timezone.utc).strftime("%Y-%m-%d") in k
            },
        }

    # ── Sync wrappers ──

    def generate_strategy_sync(self, platform: str, niche: str, **kwargs) -> ContentStrategy:
        return _run_sync(self.generate_strategy(platform, niche, **kwargs))

    def smart_engage_sync(self, platform: str, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.smart_engage(platform, **kwargs))

    def extract_analytics_sync(self, platform: str) -> AnalyticsSnapshot:
        return _run_sync(self.extract_analytics(platform))

    def track_growth_sync(self, platform: str) -> GrowthRecord:
        return _run_sync(self.track_growth(platform))

    def manage_dms_sync(self, platform: str, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.manage_dms(platform, **kwargs))

    def coordinate_posting_sync(self, content: str, platforms: List[str], **kwargs) -> Dict[str, Any]:
        return _run_sync(self.coordinate_posting(content, platforms, **kwargs))

    def analyze_competitor_sync(self, platform: str, username: str) -> CompetitorProfile:
        return _run_sync(self.analyze_competitor(platform, username))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[SocialMediaAgent] = None


def get_social_agent(
    social_bot: Any = None,
    memory: Any = None,
) -> SocialMediaAgent:
    """Get the singleton SocialMediaAgent instance."""
    global _instance
    if _instance is None:
        _instance = SocialMediaAgent(social_bot=social_bot, memory=memory)
    return _instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _cli_strategy(args: argparse.Namespace) -> None:
    agent = get_social_agent()
    strategy = agent.generate_strategy_sync(
        args.platform, args.niche,
        target_followers=args.target,
        current_followers=args.current,
    )
    _print_json(strategy.to_dict())


def _cli_engage(args: argparse.Namespace) -> None:
    agent = get_social_agent()
    result = agent.smart_engage_sync(
        args.platform,
        duration_minutes=args.duration,
        max_actions=args.max_actions,
    )
    _print_json(result)


def _cli_analytics(args: argparse.Namespace) -> None:
    agent = get_social_agent()
    if args.history:
        _print_json(agent.get_analytics_history(args.platform, args.limit))
    else:
        snapshot = agent.extract_analytics_sync(args.platform)
        _print_json(snapshot.to_dict())


def _cli_growth(args: argparse.Namespace) -> None:
    agent = get_social_agent()
    if args.track:
        record = agent.track_growth_sync(args.platform)
        _print_json(record.to_dict())
    else:
        _print_json(agent.get_growth_history(args.platform, args.days))


def _cli_competitor(args: argparse.Namespace) -> None:
    agent = get_social_agent()
    action = args.action
    if action == "analyze":
        profile = agent.analyze_competitor_sync(args.platform, args.username or "")
        _print_json(profile.to_dict())
    elif action == "list":
        _print_json(agent.list_competitors(args.platform or ""))
    else:
        print(f"Unknown competitor action: {action}")


def _cli_dms(args: argparse.Namespace) -> None:
    agent = get_social_agent()
    if args.manage:
        result = agent.manage_dms_sync(args.platform)
        _print_json(result)
    else:
        _print_json(agent.get_dm_conversations(args.platform or ""))


def _cli_post(args: argparse.Namespace) -> None:
    agent = get_social_agent()
    platforms = [p.strip() for p in args.platforms.split(",")]
    result = agent.coordinate_posting_sync(args.content, platforms)
    _print_json(result)


def _cli_stats(args: argparse.Namespace) -> None:
    agent = get_social_agent()
    _print_json(agent.stats())


def _cli_strategies(args: argparse.Namespace) -> None:
    agent = get_social_agent()
    _print_json([s.to_dict() for s in agent._strategies.values()])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="social_media_agent",
        description="OpenClaw Empire — AI Social Media Manager",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # strategy
    st = sub.add_parser("strategy", help="Generate content strategy")
    st.add_argument("--platform", required=True)
    st.add_argument("--niche", required=True)
    st.add_argument("--target", type=int, default=10000)
    st.add_argument("--current", type=int, default=0)
    st.set_defaults(func=_cli_strategy)

    # engage
    en = sub.add_parser("engage", help="Run engagement session")
    en.add_argument("--platform", required=True)
    en.add_argument("--duration", type=int, default=30)
    en.add_argument("--max-actions", type=int, default=50)
    en.set_defaults(func=_cli_engage)

    # analytics
    an = sub.add_parser("analytics", help="Extract/view analytics")
    an.add_argument("--platform", required=True)
    an.add_argument("--history", action="store_true")
    an.add_argument("--limit", type=int, default=30)
    an.set_defaults(func=_cli_analytics)

    # growth
    gr = sub.add_parser("growth", help="Growth tracking")
    gr.add_argument("--platform", required=True)
    gr.add_argument("--days", type=int, default=30)
    gr.add_argument("--track", action="store_true")
    gr.set_defaults(func=_cli_growth)

    # competitor
    cp = sub.add_parser("competitor", help="Competitor analysis")
    cp.add_argument("action", choices=["analyze", "list"])
    cp.add_argument("--platform", default="")
    cp.add_argument("--username", default="")
    cp.set_defaults(func=_cli_competitor)

    # dms
    dm = sub.add_parser("dms", help="DM management")
    dm.add_argument("--platform", default="")
    dm.add_argument("--manage", action="store_true")
    dm.set_defaults(func=_cli_dms)

    # post
    po = sub.add_parser("post", help="Coordinate cross-platform posting")
    po.add_argument("--content", required=True)
    po.add_argument("--platforms", required=True, help="Comma-separated platforms")
    po.set_defaults(func=_cli_post)

    # stats
    ss = sub.add_parser("stats", help="Agent statistics")
    ss.set_defaults(func=_cli_stats)

    # strategies
    sl = sub.add_parser("strategies", help="List all strategies")
    sl.set_defaults(func=_cli_strategies)

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
