"""
Rate Limit Manager — OpenClaw Empire Edition

Platform-aware rate limiting for all external APIs used by
Nick Creighton's 16-site WordPress publishing empire.

Platforms managed:
    ANTHROPIC    — Claude API (messages, token budgets)
    WORDPRESS    — WP REST API (posts, media, comments)
    PINTEREST    — Pin creation, board management
    INSTAGRAM    — Post scheduling, stories
    FACEBOOK     — Page posts, ads
    TWITTER      — Tweets, DMs, searches
    LINKEDIN     — Posts, articles
    SUBSTACK     — Newsletter publishing
    GOOGLE       — GSC, Analytics, PageSpeed
    AMAZON       — KDP, PA-API (affiliate)
    ETSY         — Listings, shop management
    N8N          — Webhook triggers
    GEELARK      — Cloud phone API
    SCREENPIPE   — Local OCR/audio search
    INTERNAL     — Empire inter-service calls

Algorithms:
    FIXED_WINDOW    — N requests per discrete window
    SLIDING_WINDOW  — N requests in trailing time window
    TOKEN_BUCKET    — Smooth with burst capacity
    LEAKY_BUCKET    — Fixed-rate output smoothing

All data persisted to: data/rate_limits/

Usage:
    from src.rate_limit_manager import get_rate_limit_manager, Platform

    mgr = get_rate_limit_manager()

    # Check before calling Anthropic
    if await mgr.acquire(Platform.ANTHROPIC, "messages", cost=0.01):
        response = await client.messages.create(...)

    # Synchronous variant
    if mgr.acquire_sync(Platform.WORDPRESS, "posts"):
        wp_client.create_post(...)

    # Context manager
    from src.rate_limit_manager import rate_limited
    async with rate_limited(Platform.ANTHROPIC, "messages", cost=0.01):
        response = await client.messages.create(...)

    # CLI
    python -m src.rate_limit_manager status
    python -m src.rate_limit_manager platform anthropic
    python -m src.rate_limit_manager costs --days 7
    python -m src.rate_limit_manager reset --platform anthropic
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import math
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger("rate_limit_manager")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
RATELIMIT_DATA_DIR = BASE_DIR / "data" / "rate_limits"
STATE_FILE = RATELIMIT_DATA_DIR / "state.json"
CONFIG_FILE = RATELIMIT_DATA_DIR / "config.json"
DAILY_DIR = RATELIMIT_DATA_DIR / "daily"
COST_HISTORY_FILE = RATELIMIT_DATA_DIR / "cost_history.json"

# Ensure directories exist on import
RATELIMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TIMESTAMPS_KEPT = 5000          # per limiter, prune old entries
DAILY_RESET_HOUR_UTC = 5            # midnight US/Eastern approx
MAX_DAILY_HISTORY_FILES = 90        # 90 days of daily summaries
WARNING_THRESHOLD_DEFAULT = 0.80    # warn at 80% usage
STATE_SAVE_INTERVAL = 30.0          # seconds between auto-saves


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return _now_utc().isoformat()


def _today_str() -> str:
    """Return today's date as YYYY-MM-DD."""
    return _now_utc().strftime("%Y-%m-%d")


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* on any error."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(str(tmp), str(path))


def _run_sync(coro):
    """Run async coroutine synchronously, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _limiter_key(platform: Platform, endpoint: str) -> str:
    """Canonical key for a platform + endpoint pair."""
    return f"{platform.value}:{endpoint}"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Platform(str, Enum):
    """Platforms integrated into the empire."""
    ANTHROPIC = "anthropic"
    WORDPRESS = "wordpress"
    PINTEREST = "pinterest"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    SUBSTACK = "substack"
    GOOGLE = "google"
    AMAZON = "amazon"
    ETSY = "etsy"
    N8N = "n8n"
    GEELARK = "geelark"
    SCREENPIPE = "screenpipe"
    INTERNAL = "internal"


class RateLimitStrategy(str, Enum):
    """Algorithm used for rate limiting."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class LimitStatus(str, Enum):
    """Current state of a rate limiter."""
    OK = "ok"
    WARNING = "warning"
    THROTTLED = "throttled"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RateLimitConfig:
    """Configuration for a single rate-limited endpoint."""
    platform: Platform
    endpoint: str
    max_requests: int
    window_seconds: int
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_limit: Optional[int] = None
    retry_after_seconds: float = 1.0
    warning_threshold: float = WARNING_THRESHOLD_DEFAULT
    cost_per_request: float = 0.0
    daily_budget: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["platform"] = self.platform.value
        d["strategy"] = self.strategy.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RateLimitConfig:
        d = dict(d)
        d["platform"] = Platform(d["platform"])
        d["strategy"] = RateLimitStrategy(d.get("strategy", "sliding_window"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RateLimitState:
    """Runtime state for a single rate limiter."""
    config: RateLimitConfig
    request_timestamps: List[float] = field(default_factory=list)
    tokens_remaining: float = 0.0
    last_refill: float = 0.0
    total_requests_today: int = 0
    total_cost_today: float = 0.0
    blocked_count: int = 0
    last_request: Optional[float] = None
    status: LimitStatus = LimitStatus.OK
    today_date: str = ""

    def __post_init__(self):
        if not self.today_date:
            self.today_date = _today_str()
        if self.tokens_remaining <= 0 and self.last_refill == 0.0:
            self.tokens_remaining = float(self.config.max_requests)
            self.last_refill = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "request_timestamps": self.request_timestamps[-MAX_TIMESTAMPS_KEPT:],
            "tokens_remaining": self.tokens_remaining,
            "last_refill": self.last_refill,
            "total_requests_today": self.total_requests_today,
            "total_cost_today": self.total_cost_today,
            "blocked_count": self.blocked_count,
            "last_request": self.last_request,
            "status": self.status.value,
            "today_date": self.today_date,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RateLimitState:
        config = RateLimitConfig.from_dict(d["config"])
        return cls(
            config=config,
            request_timestamps=d.get("request_timestamps", []),
            tokens_remaining=d.get("tokens_remaining", float(config.max_requests)),
            last_refill=d.get("last_refill", time.time()),
            total_requests_today=d.get("total_requests_today", 0),
            total_cost_today=d.get("total_cost_today", 0.0),
            blocked_count=d.get("blocked_count", 0),
            last_request=d.get("last_request"),
            status=LimitStatus(d.get("status", "ok")),
            today_date=d.get("today_date", _today_str()),
        )


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class RateLimitExceeded(Exception):
    """Raised when a rate limit cannot be satisfied."""

    def __init__(self, message: str, platform: Platform, endpoint: str,
                 retry_after: float = 0.0):
        super().__init__(message)
        self.platform = platform
        self.endpoint = endpoint
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Default platform limits
# ---------------------------------------------------------------------------

DEFAULT_LIMITS: Dict[Platform, List[RateLimitConfig]] = {
    # -- Anthropic (Claude API) -----------------------------------------------
    Platform.ANTHROPIC: [
        RateLimitConfig(
            Platform.ANTHROPIC, "messages", max_requests=50, window_seconds=60,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            burst_limit=10, retry_after_seconds=2.0,
            cost_per_request=0.01, daily_budget=50.0,
            notes="Opus/Sonnet — Tier 2 rate limit",
        ),
        RateLimitConfig(
            Platform.ANTHROPIC, "messages_haiku", max_requests=100, window_seconds=60,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            burst_limit=20, retry_after_seconds=1.0,
            cost_per_request=0.002, daily_budget=20.0,
            notes="Haiku — higher throughput, lower cost",
        ),
        RateLimitConfig(
            Platform.ANTHROPIC, "batch", max_requests=500, window_seconds=3600,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            cost_per_request=0.005, daily_budget=30.0,
            notes="Batch API — 50% cost discount",
        ),
    ],
    # -- WordPress REST API ---------------------------------------------------
    Platform.WORDPRESS: [
        RateLimitConfig(
            Platform.WORDPRESS, "posts", max_requests=30, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            retry_after_seconds=2.0,
            notes="POST /wp-json/wp/v2/posts — rate varies by host",
        ),
        RateLimitConfig(
            Platform.WORDPRESS, "media", max_requests=10, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            retry_after_seconds=5.0,
            notes="POST /wp-json/wp/v2/media — upload-heavy",
        ),
        RateLimitConfig(
            Platform.WORDPRESS, "reads", max_requests=120, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            notes="GET requests — higher limit",
        ),
    ],
    # -- Pinterest API --------------------------------------------------------
    Platform.PINTEREST: [
        RateLimitConfig(
            Platform.PINTEREST, "pins", max_requests=10, window_seconds=3600,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            retry_after_seconds=60.0,
            notes="Pin creation — conservative to avoid bans",
        ),
        RateLimitConfig(
            Platform.PINTEREST, "boards", max_requests=5, window_seconds=3600,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            notes="Board operations",
        ),
    ],
    # -- Twitter / X ----------------------------------------------------------
    Platform.TWITTER: [
        RateLimitConfig(
            Platform.TWITTER, "tweets", max_requests=50, window_seconds=86400,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            retry_after_seconds=300.0,
            notes="Free tier — 1500/mo, ~50/day safe target",
        ),
        RateLimitConfig(
            Platform.TWITTER, "reads", max_requests=100, window_seconds=900,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            notes="Read endpoints — 15-minute windows",
        ),
    ],
    # -- Instagram Graph API --------------------------------------------------
    Platform.INSTAGRAM: [
        RateLimitConfig(
            Platform.INSTAGRAM, "posts", max_requests=25, window_seconds=86400,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            retry_after_seconds=600.0,
            notes="Content publishing — 25/day limit",
        ),
        RateLimitConfig(
            Platform.INSTAGRAM, "stories", max_requests=50, window_seconds=86400,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            notes="Story publishing",
        ),
        RateLimitConfig(
            Platform.INSTAGRAM, "reads", max_requests=200, window_seconds=3600,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            notes="Graph API reads — 200/hr",
        ),
    ],
    # -- Facebook Graph API ---------------------------------------------------
    Platform.FACEBOOK: [
        RateLimitConfig(
            Platform.FACEBOOK, "posts", max_requests=25, window_seconds=86400,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            retry_after_seconds=300.0,
            notes="Page posts — conservative daily limit",
        ),
        RateLimitConfig(
            Platform.FACEBOOK, "reads", max_requests=200, window_seconds=3600,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            notes="Graph API reads",
        ),
    ],
    # -- LinkedIn API ---------------------------------------------------------
    Platform.LINKEDIN: [
        RateLimitConfig(
            Platform.LINKEDIN, "posts", max_requests=20, window_seconds=86400,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            retry_after_seconds=600.0,
            notes="Post creation — 100/day max, 20 recommended",
        ),
        RateLimitConfig(
            Platform.LINKEDIN, "reads", max_requests=100, window_seconds=86400,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            notes="Profile/connection reads",
        ),
    ],
    # -- Substack -------------------------------------------------------------
    Platform.SUBSTACK: [
        RateLimitConfig(
            Platform.SUBSTACK, "posts", max_requests=5, window_seconds=86400,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            notes="Newsletter publishing — no official API rate, self-imposed",
        ),
    ],
    # -- Google (GSC, Analytics, PageSpeed) ------------------------------------
    Platform.GOOGLE: [
        RateLimitConfig(
            Platform.GOOGLE, "search_console", max_requests=200, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            notes="GSC API — 200 QPM default quota",
        ),
        RateLimitConfig(
            Platform.GOOGLE, "analytics", max_requests=100, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            notes="GA4 Data API",
        ),
        RateLimitConfig(
            Platform.GOOGLE, "pagespeed", max_requests=25, window_seconds=60,
            strategy=RateLimitStrategy.LEAKY_BUCKET,
            retry_after_seconds=3.0,
            notes="PageSpeed Insights — 25K/day, smoothed",
        ),
    ],
    # -- Amazon (KDP, PA-API) -------------------------------------------------
    Platform.AMAZON: [
        RateLimitConfig(
            Platform.AMAZON, "paapi", max_requests=1, window_seconds=1,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            burst_limit=1, retry_after_seconds=1.0,
            notes="Product Advertising API — 1 TPS base",
        ),
        RateLimitConfig(
            Platform.AMAZON, "kdp", max_requests=10, window_seconds=3600,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            notes="KDP management actions — self-imposed",
        ),
    ],
    # -- Etsy -----------------------------------------------------------------
    Platform.ETSY: [
        RateLimitConfig(
            Platform.ETSY, "listings", max_requests=30, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            notes="Listing management — Open API v3",
        ),
        RateLimitConfig(
            Platform.ETSY, "reads", max_requests=100, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            notes="Shop/listing reads",
        ),
    ],
    # -- n8n webhooks ---------------------------------------------------------
    Platform.N8N: [
        RateLimitConfig(
            Platform.N8N, "webhooks", max_requests=60, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            notes="Webhook triggers — 1/sec average",
        ),
    ],
    # -- GeeLark cloud phone --------------------------------------------------
    Platform.GEELARK: [
        RateLimitConfig(
            Platform.GEELARK, "api", max_requests=30, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            retry_after_seconds=2.0,
            notes="GeeLark management API",
        ),
        RateLimitConfig(
            Platform.GEELARK, "adb", max_requests=10, window_seconds=60,
            strategy=RateLimitStrategy.LEAKY_BUCKET,
            notes="ADB bridge commands — smoothed",
        ),
    ],
    # -- Screenpipe -----------------------------------------------------------
    Platform.SCREENPIPE: [
        RateLimitConfig(
            Platform.SCREENPIPE, "search", max_requests=60, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            notes="Local OCR/audio search — generous limit",
        ),
    ],
    # -- Internal (empire inter-service) --------------------------------------
    Platform.INTERNAL: [
        RateLimitConfig(
            Platform.INTERNAL, "default", max_requests=200, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            notes="Inter-service calls — high throughput",
        ),
    ],
}


# ---------------------------------------------------------------------------
# RateLimiter — per-endpoint rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Rate limiter for a single platform endpoint.

    Implements four strategies: fixed window, sliding window, token bucket,
    and leaky bucket.  Thread-safe through asyncio lock.
    """

    def __init__(self, config: RateLimitConfig, state: Optional[RateLimitState] = None):
        self._config = config
        self._state = state or RateLimitState(config=config)
        self._lock = asyncio.Lock()
        self._key = _limiter_key(config.platform, config.endpoint)

    # -- Properties -----------------------------------------------------------

    @property
    def config(self) -> RateLimitConfig:
        return self._config

    @property
    def state(self) -> RateLimitState:
        return self._state

    @property
    def key(self) -> str:
        return self._key

    # -- Daily reset ----------------------------------------------------------

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if the date has rolled over."""
        today = _today_str()
        if self._state.today_date != today:
            logger.info(
                "Daily reset for %s (was %s, now %s)",
                self._key, self._state.today_date, today,
            )
            self._state.total_requests_today = 0
            self._state.total_cost_today = 0.0
            self._state.blocked_count = 0
            self._state.today_date = today

    # -- Strategy implementations ---------------------------------------------

    def _prune_timestamps(self, now: float) -> None:
        """Remove timestamps older than the window."""
        cutoff = now - self._config.window_seconds
        ts = self._state.request_timestamps
        # Binary search for cutoff (timestamps are sorted)
        lo, hi = 0, len(ts)
        while lo < hi:
            mid = (lo + hi) // 2
            if ts[mid] < cutoff:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0:
            self._state.request_timestamps = ts[lo:]

    def _count_in_window(self, now: float) -> int:
        """Count requests within the current window."""
        cutoff = now - self._config.window_seconds
        count = 0
        for ts in reversed(self._state.request_timestamps):
            if ts >= cutoff:
                count += 1
            else:
                break
        return count

    def _fixed_window_check(self, now: float) -> tuple[bool, float]:
        """Fixed window: count resets at window boundaries.

        Returns (can_proceed, wait_seconds).
        """
        window = self._config.window_seconds
        window_start = (now // window) * window
        # Count requests in current window
        count = sum(
            1 for ts in self._state.request_timestamps
            if ts >= window_start
        )
        if count < self._config.max_requests:
            return True, 0.0
        # Wait until next window
        wait = (window_start + window) - now
        return False, max(wait, self._config.retry_after_seconds)

    def _sliding_window_check(self, now: float) -> tuple[bool, float]:
        """Sliding window: count in trailing window_seconds.

        Returns (can_proceed, wait_seconds).
        """
        self._prune_timestamps(now)
        count = len(self._state.request_timestamps)
        if count < self._config.max_requests:
            return True, 0.0
        # Wait until oldest request exits the window
        oldest = self._state.request_timestamps[0]
        wait = (oldest + self._config.window_seconds) - now
        return False, max(wait, self._config.retry_after_seconds)

    def _token_bucket_check(self, now: float) -> tuple[bool, float]:
        """Token bucket: steady refill with burst capacity.

        Returns (can_proceed, wait_seconds).
        """
        # Refill tokens
        elapsed = now - self._state.last_refill
        refill_rate = self._config.max_requests / self._config.window_seconds
        new_tokens = elapsed * refill_rate
        max_tokens = float(self._config.burst_limit or self._config.max_requests)
        self._state.tokens_remaining = min(
            max_tokens,
            self._state.tokens_remaining + new_tokens,
        )
        self._state.last_refill = now

        if self._state.tokens_remaining >= 1.0:
            return True, 0.0
        # Wait for one token to refill
        wait = (1.0 - self._state.tokens_remaining) / refill_rate
        return False, max(wait, self._config.retry_after_seconds)

    def _leaky_bucket_check(self, now: float) -> tuple[bool, float]:
        """Leaky bucket: fixed output rate, smoothing bursts.

        Returns (can_proceed, wait_seconds).
        """
        interval = self._config.window_seconds / self._config.max_requests
        if self._state.last_request is None:
            return True, 0.0
        elapsed = now - self._state.last_request
        if elapsed >= interval:
            return True, 0.0
        wait = interval - elapsed
        return False, max(wait, 0.01)

    def _check_strategy(self, now: float) -> tuple[bool, float]:
        """Dispatch to the configured strategy."""
        strategy = self._config.strategy
        if strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._fixed_window_check(now)
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window_check(now)
        elif strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._token_bucket_check(now)
        elif strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._leaky_bucket_check(now)
        else:
            logger.warning("Unknown strategy %s, defaulting to sliding window", strategy)
            return self._sliding_window_check(now)

    def _update_status(self, now: float) -> LimitStatus:
        """Compute and update the current status."""
        # Check daily budget first
        if self._config.daily_budget and self._state.total_cost_today >= self._config.daily_budget:
            self._state.status = LimitStatus.BLOCKED
            return LimitStatus.BLOCKED

        can_proceed, wait = self._check_strategy(now)
        if not can_proceed:
            if wait > self._config.retry_after_seconds * 5:
                self._state.status = LimitStatus.BLOCKED
            else:
                self._state.status = LimitStatus.THROTTLED
            return self._state.status

        # Check warning threshold
        usage = self._get_usage_ratio(now)
        if usage >= self._config.warning_threshold:
            self._state.status = LimitStatus.WARNING
        else:
            self._state.status = LimitStatus.OK
        return self._state.status

    def _get_usage_ratio(self, now: float) -> float:
        """Return fraction of rate limit consumed (0.0 to 1.0+)."""
        strategy = self._config.strategy
        if strategy == RateLimitStrategy.TOKEN_BUCKET:
            max_tokens = float(self._config.burst_limit or self._config.max_requests)
            if max_tokens <= 0:
                return 0.0
            return 1.0 - (self._state.tokens_remaining / max_tokens)
        elif strategy == RateLimitStrategy.LEAKY_BUCKET:
            interval = self._config.window_seconds / self._config.max_requests
            if self._state.last_request is None:
                return 0.0
            elapsed = now - self._state.last_request
            return max(0.0, 1.0 - (elapsed / interval))
        else:
            # Fixed or sliding window
            self._prune_timestamps(now)
            count = len(self._state.request_timestamps)
            if self._config.max_requests <= 0:
                return 0.0
            return count / self._config.max_requests

    def _record_request(self, now: float, cost: float) -> None:
        """Record a successful request acquisition."""
        self._state.request_timestamps.append(now)
        self._state.last_request = now
        self._state.total_requests_today += 1
        self._state.total_cost_today += cost

        # Consume token for bucket strategies
        if self._config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            self._state.tokens_remaining = max(0.0, self._state.tokens_remaining - 1.0)

        # Prune old timestamps to avoid unbounded growth
        if len(self._state.request_timestamps) > MAX_TIMESTAMPS_KEPT:
            self._state.request_timestamps = self._state.request_timestamps[-MAX_TIMESTAMPS_KEPT:]

    # -- Public API -----------------------------------------------------------

    async def acquire(self, cost: float = 0.0, max_wait: float = 30.0) -> bool:
        """Try to acquire a request slot.

        If the limiter is in THROTTLED state and *max_wait* > 0, waits up to
        *max_wait* seconds for a slot to become available.  Returns True if the
        request can proceed, False if blocked.
        """
        async with self._lock:
            self._maybe_reset_daily()
            now = time.time()

            # Check daily budget
            effective_cost = cost if cost > 0 else self._config.cost_per_request
            if self._config.daily_budget:
                if self._state.total_cost_today + effective_cost > self._config.daily_budget:
                    self._state.blocked_count += 1
                    self._state.status = LimitStatus.BLOCKED
                    logger.warning(
                        "Budget exceeded for %s: $%.2f / $%.2f daily",
                        self._key, self._state.total_cost_today, self._config.daily_budget,
                    )
                    return False

            can_proceed, wait = self._check_strategy(now)

            if can_proceed:
                self._record_request(now, effective_cost)
                self._update_status(now)
                return True

            # Cannot proceed immediately — try waiting
            if max_wait <= 0 or wait > max_wait:
                self._state.blocked_count += 1
                self._update_status(now)
                logger.info(
                    "Rate limited %s — need to wait %.1fs (max_wait=%.1f)",
                    self._key, wait, max_wait,
                )
                return False

        # Release lock while waiting
        logger.debug("Throttled %s — waiting %.2fs", self._key, wait)
        await asyncio.sleep(wait)

        # Re-acquire lock and try again
        async with self._lock:
            now = time.time()
            can_proceed, wait2 = self._check_strategy(now)
            if can_proceed:
                effective_cost = cost if cost > 0 else self._config.cost_per_request
                self._record_request(now, effective_cost)
                self._update_status(now)
                return True
            self._state.blocked_count += 1
            self._update_status(now)
            return False

    def acquire_sync(self, cost: float = 0.0, max_wait: float = 30.0) -> bool:
        """Synchronous wrapper around :meth:`acquire`."""
        return _run_sync(self.acquire(cost=cost, max_wait=max_wait))

    async def wait_if_needed(self) -> float:
        """Wait until the next request is allowed.

        Returns the number of seconds waited (0 if no wait was needed).
        """
        async with self._lock:
            self._maybe_reset_daily()
            now = time.time()
            can_proceed, wait = self._check_strategy(now)
            if can_proceed:
                return 0.0

        if wait > 0:
            await asyncio.sleep(wait)
        return wait

    def check_status(self) -> LimitStatus:
        """Return current status without consuming a request slot."""
        now = time.time()
        self._maybe_reset_daily()
        return self._update_status(now)

    def get_wait_time(self) -> float:
        """Return seconds until the next request would be allowed."""
        now = time.time()
        self._maybe_reset_daily()
        can_proceed, wait = self._check_strategy(now)
        if can_proceed:
            return 0.0
        return wait

    def reset(self) -> None:
        """Reset this limiter to initial state."""
        self._state = RateLimitState(config=self._config)
        logger.info("Reset limiter %s", self._key)

    def get_usage(self) -> Dict[str, Any]:
        """Return current usage statistics."""
        now = time.time()
        self._maybe_reset_daily()
        self._update_status(now)
        ratio = self._get_usage_ratio(now)

        # Count requests in current window
        if self._config.strategy in (RateLimitStrategy.FIXED_WINDOW, RateLimitStrategy.SLIDING_WINDOW):
            self._prune_timestamps(now)
            used = len(self._state.request_timestamps)
        elif self._config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            max_t = float(self._config.burst_limit or self._config.max_requests)
            used = int(max_t - self._state.tokens_remaining)
        else:
            used = self._state.total_requests_today

        return {
            "key": self._key,
            "platform": self._config.platform.value,
            "endpoint": self._config.endpoint,
            "strategy": self._config.strategy.value,
            "status": self._state.status.value,
            "used": used,
            "max": self._config.max_requests,
            "usage_percent": round(ratio * 100, 1),
            "window_seconds": self._config.window_seconds,
            "wait_seconds": self.get_wait_time(),
            "total_today": self._state.total_requests_today,
            "cost_today": round(self._state.total_cost_today, 4),
            "daily_budget": self._config.daily_budget,
            "budget_remaining": (
                round(self._config.daily_budget - self._state.total_cost_today, 4)
                if self._config.daily_budget else None
            ),
            "blocked_count": self._state.blocked_count,
            "last_request": (
                datetime.fromtimestamp(self._state.last_request, tz=timezone.utc).isoformat()
                if self._state.last_request else None
            ),
        }


# ---------------------------------------------------------------------------
# RateLimitManager — manages all limiters
# ---------------------------------------------------------------------------

class RateLimitManager:
    """Central manager for all platform rate limiters.

    Lazily initializes limiters from DEFAULT_LIMITS on first access.  State
    is persisted to disk and restored on startup.
    """

    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._daily_stats: Dict[str, Dict[str, Any]] = {}
        self._last_save: float = 0.0
        self._initialized = False
        self._init_lock = asyncio.Lock()

    # -- Initialization -------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Load saved state or create default limiters."""
        if self._initialized:
            return
        self._initialized = True

        # Try to load saved state
        saved = _load_json(STATE_FILE, default={})
        if saved and "limiters" in saved:
            for key, limiter_data in saved["limiters"].items():
                try:
                    state = RateLimitState.from_dict(limiter_data)
                    self._limiters[key] = RateLimiter(state.config, state)
                except Exception as exc:
                    logger.warning("Failed to restore limiter %s: %s", key, exc)

        # Load any custom config overrides
        custom_config = _load_json(CONFIG_FILE, default={})

        # Register defaults for any missing limiters
        for platform, configs in DEFAULT_LIMITS.items():
            for cfg in configs:
                key = _limiter_key(platform, cfg.endpoint)
                if key not in self._limiters:
                    # Check for custom override
                    if key in custom_config:
                        try:
                            cfg = RateLimitConfig.from_dict(custom_config[key])
                        except Exception:
                            pass
                    self._limiters[key] = RateLimiter(cfg)

        # Load daily stats
        self._daily_stats = _load_json(
            DAILY_DIR / f"{_today_str()}.json",
            default={},
        )
        logger.info("Rate limit manager initialized with %d limiters", len(self._limiters))

    def _get_limiter(self, platform: Platform, endpoint: str) -> RateLimiter:
        """Get or create a limiter for the given platform/endpoint."""
        self._ensure_initialized()
        key = _limiter_key(platform, endpoint)
        if key not in self._limiters:
            # Create a default limiter with generous limits
            cfg = RateLimitConfig(
                platform=platform,
                endpoint=endpoint,
                max_requests=60,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                notes="Auto-created default",
            )
            self._limiters[key] = RateLimiter(cfg)
            logger.info("Auto-created limiter for %s", key)
        return self._limiters[key]

    def _auto_save(self) -> None:
        """Save state if enough time has elapsed since last save."""
        now = time.time()
        if now - self._last_save >= STATE_SAVE_INTERVAL:
            self.save_state()

    # -- Core API -------------------------------------------------------------

    async def acquire(self, platform: Platform, endpoint: str = "default",
                      cost: float = 0.0, max_wait: float = 30.0) -> bool:
        """Try to acquire a request slot for *platform*:*endpoint*.

        Waits up to *max_wait* seconds if throttled.  Returns True if
        the request can proceed.
        """
        limiter = self._get_limiter(platform, endpoint)
        result = await limiter.acquire(cost=cost, max_wait=max_wait)
        self._auto_save()
        return result

    def acquire_sync(self, platform: Platform, endpoint: str = "default",
                     cost: float = 0.0, max_wait: float = 30.0) -> bool:
        """Synchronous wrapper for :meth:`acquire`."""
        return _run_sync(self.acquire(platform, endpoint, cost, max_wait))

    def check(self, platform: Platform, endpoint: str = "default") -> LimitStatus:
        """Check status without consuming a request slot."""
        limiter = self._get_limiter(platform, endpoint)
        return limiter.check_status()

    def get_wait_time(self, platform: Platform, endpoint: str = "default") -> float:
        """Seconds until the next request would be allowed."""
        limiter = self._get_limiter(platform, endpoint)
        return limiter.get_wait_time()

    def configure(self, platform: Platform, endpoint: str,
                  config: RateLimitConfig) -> None:
        """Override the config for a specific platform/endpoint.

        Persists the override to config.json so it survives restarts.
        """
        self._ensure_initialized()
        key = _limiter_key(platform, endpoint)

        # Update or create limiter
        if key in self._limiters:
            old_state = self._limiters[key].state
            old_state.config = config
            self._limiters[key] = RateLimiter(config, old_state)
        else:
            self._limiters[key] = RateLimiter(config)

        # Persist custom config
        custom = _load_json(CONFIG_FILE, default={})
        custom[key] = config.to_dict()
        _save_json(CONFIG_FILE, custom)
        logger.info("Configured %s: %d req / %ds", key, config.max_requests, config.window_seconds)

    def get_platform_status(self, platform: Platform) -> Dict[str, Any]:
        """Return status for all endpoints of a platform."""
        self._ensure_initialized()
        prefix = f"{platform.value}:"
        endpoints = {}
        for key, limiter in self._limiters.items():
            if key.startswith(prefix):
                endpoints[limiter.config.endpoint] = limiter.get_usage()
        return {
            "platform": platform.value,
            "endpoints": endpoints,
            "timestamp": _now_iso(),
        }

    def get_all_status(self) -> Dict[str, Any]:
        """Return status for all platforms and endpoints."""
        self._ensure_initialized()
        platforms = {}
        for platform in Platform:
            status = self.get_platform_status(platform)
            if status["endpoints"]:
                platforms[platform.value] = status
        return {
            "platforms": platforms,
            "total_limiters": len(self._limiters),
            "timestamp": _now_iso(),
        }

    def get_daily_report(self) -> Dict[str, Any]:
        """Return today's usage across all platforms."""
        self._ensure_initialized()
        report: Dict[str, Any] = {
            "date": _today_str(),
            "platforms": {},
            "totals": {
                "requests": 0,
                "cost": 0.0,
                "blocked": 0,
            },
        }
        for key, limiter in self._limiters.items():
            usage = limiter.get_usage()
            platform = usage["platform"]
            if platform not in report["platforms"]:
                report["platforms"][platform] = {
                    "endpoints": {},
                    "total_requests": 0,
                    "total_cost": 0.0,
                    "total_blocked": 0,
                }
            p = report["platforms"][platform]
            p["endpoints"][usage["endpoint"]] = {
                "requests": usage["total_today"],
                "cost": usage["cost_today"],
                "blocked": usage["blocked_count"],
                "status": usage["status"],
            }
            p["total_requests"] += usage["total_today"]
            p["total_cost"] += usage["cost_today"]
            p["total_blocked"] += usage["blocked_count"]
            report["totals"]["requests"] += usage["total_today"]
            report["totals"]["cost"] += usage["cost_today"]
            report["totals"]["blocked"] += usage["blocked_count"]

        report["totals"]["cost"] = round(report["totals"]["cost"], 4)
        return report

    def get_cost_report(self, days: int = 7) -> Dict[str, Any]:
        """Return cost breakdown for the last *days* days."""
        self._ensure_initialized()
        report: Dict[str, Any] = {
            "days": days,
            "daily": [],
            "platform_totals": {},
            "grand_total": 0.0,
        }

        today = date.today()
        for i in range(days):
            d = today - timedelta(days=i)
            day_str = d.isoformat()
            day_file = DAILY_DIR / f"{day_str}.json"
            day_data = _load_json(day_file, default={})

            day_entry = {"date": day_str, "total": 0.0, "platforms": {}}
            if "platforms" in day_data:
                for plat_name, plat_data in day_data.get("platforms", {}).items():
                    cost = plat_data.get("total_cost", 0.0)
                    day_entry["platforms"][plat_name] = cost
                    day_entry["total"] += cost
                    report["platform_totals"][plat_name] = (
                        report["platform_totals"].get(plat_name, 0.0) + cost
                    )
            elif i == 0:
                # Today — use live data
                daily = self.get_daily_report()
                for plat_name, plat_data in daily.get("platforms", {}).items():
                    cost = plat_data.get("total_cost", 0.0)
                    day_entry["platforms"][plat_name] = round(cost, 4)
                    day_entry["total"] += cost
                    report["platform_totals"][plat_name] = (
                        report["platform_totals"].get(plat_name, 0.0) + cost
                    )

            day_entry["total"] = round(day_entry["total"], 4)
            report["daily"].append(day_entry)
            report["grand_total"] += day_entry["total"]

        report["grand_total"] = round(report["grand_total"], 4)
        for k in report["platform_totals"]:
            report["platform_totals"][k] = round(report["platform_totals"][k], 4)
        return report

    def is_budget_exceeded(self, platform: Platform) -> bool:
        """Check if any endpoint's daily budget is exceeded."""
        self._ensure_initialized()
        prefix = f"{platform.value}:"
        for key, limiter in self._limiters.items():
            if key.startswith(prefix):
                cfg = limiter.config
                if cfg.daily_budget and limiter.state.total_cost_today >= cfg.daily_budget:
                    return True
        return False

    def reset_daily(self) -> None:
        """Reset all daily counters.  Call at midnight or on demand."""
        self._ensure_initialized()
        # Save yesterday's stats before resetting
        self._save_daily_snapshot()
        for limiter in self._limiters.values():
            limiter.state.total_requests_today = 0
            limiter.state.total_cost_today = 0.0
            limiter.state.blocked_count = 0
            limiter.state.today_date = _today_str()
        self.save_state()
        logger.info("Daily counters reset for all %d limiters", len(self._limiters))

    def reset_platform(self, platform: Platform) -> None:
        """Reset all limiters for a specific platform."""
        self._ensure_initialized()
        prefix = f"{platform.value}:"
        count = 0
        for key, limiter in self._limiters.items():
            if key.startswith(prefix):
                limiter.reset()
                count += 1
        self.save_state()
        logger.info("Reset %d limiters for platform %s", count, platform.value)

    def reset_endpoint(self, platform: Platform, endpoint: str) -> None:
        """Reset a single limiter."""
        limiter = self._get_limiter(platform, endpoint)
        limiter.reset()
        self.save_state()

    # -- Persistence ----------------------------------------------------------

    def save_state(self) -> None:
        """Persist all limiter states to disk."""
        self._ensure_initialized()
        data = {
            "limiters": {},
            "saved_at": _now_iso(),
            "version": 1,
        }
        for key, limiter in self._limiters.items():
            data["limiters"][key] = limiter.state.to_dict()
        _save_json(STATE_FILE, data)
        self._last_save = time.time()

    def load_state(self) -> None:
        """Reload state from disk (replaces in-memory state)."""
        self._initialized = False
        self._limiters.clear()
        self._ensure_initialized()

    def _save_daily_snapshot(self) -> None:
        """Save a daily summary file for historical cost tracking."""
        report = self.get_daily_report()
        day_str = report["date"]
        _save_json(DAILY_DIR / f"{day_str}.json", report)
        self._cleanup_old_daily_files()

    def _cleanup_old_daily_files(self) -> None:
        """Remove daily snapshot files older than MAX_DAILY_HISTORY_FILES days."""
        try:
            files = sorted(DAILY_DIR.glob("????-??-??.json"))
            if len(files) > MAX_DAILY_HISTORY_FILES:
                for f in files[: len(files) - MAX_DAILY_HISTORY_FILES]:
                    f.unlink()
                    logger.debug("Removed old daily file: %s", f.name)
        except OSError as exc:
            logger.warning("Failed to clean up daily files: %s", exc)

    # -- Stats ----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics for the manager."""
        self._ensure_initialized()
        total_requests = 0
        total_cost = 0.0
        total_blocked = 0
        status_counts: Dict[str, int] = {s.value: 0 for s in LimitStatus}
        platforms_active = set()

        for limiter in self._limiters.values():
            usage = limiter.get_usage()
            total_requests += usage["total_today"]
            total_cost += usage["cost_today"]
            total_blocked += usage["blocked_count"]
            status_counts[usage["status"]] += 1
            if usage["total_today"] > 0:
                platforms_active.add(usage["platform"])

        return {
            "total_limiters": len(self._limiters),
            "platforms_registered": len(set(
                l.config.platform.value for l in self._limiters.values()
            )),
            "platforms_active_today": len(platforms_active),
            "total_requests_today": total_requests,
            "total_cost_today": round(total_cost, 4),
            "total_blocked_today": total_blocked,
            "status_counts": status_counts,
            "state_file": str(STATE_FILE),
            "timestamp": _now_iso(),
        }


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def rate_limited(platform: Platform, endpoint: str = "default",
                       cost: float = 0.0, max_wait: float = 30.0) -> AsyncIterator[None]:
    """Async context manager that acquires a rate limit slot before executing.

    Raises :class:`RateLimitExceeded` if the limit cannot be satisfied within
    *max_wait* seconds.

    Usage::

        async with rate_limited(Platform.ANTHROPIC, "messages", cost=0.01):
            response = await client.messages.create(...)
    """
    manager = get_rate_limit_manager()
    acquired = await manager.acquire(platform, endpoint, cost=cost, max_wait=max_wait)
    if not acquired:
        wait = manager.get_wait_time(platform, endpoint)
        raise RateLimitExceeded(
            f"Rate limit exceeded for {platform.value}:{endpoint} "
            f"(retry after {wait:.1f}s)",
            platform=platform,
            endpoint=endpoint,
            retry_after=wait,
        )
    try:
        yield
    finally:
        # No explicit release needed — slots are time-based
        pass


def rate_limited_sync(platform: Platform, endpoint: str = "default",
                      cost: float = 0.0) -> bool:
    """Synchronous convenience: acquire and return True/False.

    Unlike the async context manager, this does NOT raise — just returns False
    if blocked.

    Usage::

        if rate_limited_sync(Platform.WORDPRESS, "posts"):
            wp_client.create_post(...)
        else:
            logger.warning("WordPress rate limited")
    """
    manager = get_rate_limit_manager()
    return manager.acquire_sync(platform, endpoint, cost=cost)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def with_rate_limit(platform: Platform, endpoint: str = "default",
                    cost: float = 0.0, max_wait: float = 30.0):
    """Decorator that rate-limits an async function.

    Usage::

        @with_rate_limit(Platform.ANTHROPIC, "messages", cost=0.01)
        async def call_claude(prompt: str) -> str:
            ...
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with rate_limited(platform, endpoint, cost=cost, max_wait=max_wait):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_rate_limit_manager: Optional[RateLimitManager] = None


def get_rate_limit_manager() -> RateLimitManager:
    """Return the singleton :class:`RateLimitManager` instance."""
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
    return _rate_limit_manager


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _format_status_table(statuses: Dict[str, Any]) -> str:
    """Format the all-status dict into a readable table."""
    lines: List[str] = []
    lines.append("")
    lines.append(f"{'Platform':<14} {'Endpoint':<20} {'Strategy':<15} "
                 f"{'Status':<10} {'Used/Max':<12} {'%':>6}  "
                 f"{'Wait':>7}  {'Today':>7}  {'Cost':>8}  {'Budget':>8}")
    lines.append("-" * 130)

    for plat_name, plat_data in sorted(statuses.get("platforms", {}).items()):
        for ep_name, ep_data in sorted(plat_data.get("endpoints", {}).items()):
            status_str = ep_data["status"].upper()
            if status_str == "OK":
                status_display = "OK"
            elif status_str == "WARNING":
                status_display = "WARN"
            elif status_str == "THROTTLED":
                status_display = "THROT"
            else:
                status_display = "BLOCK"

            budget_str = (
                f"${ep_data['daily_budget']:.2f}"
                if ep_data.get("daily_budget") else "---"
            )
            cost_str = f"${ep_data['cost_today']:.4f}" if ep_data["cost_today"] > 0 else "---"
            wait_str = f"{ep_data['wait_seconds']:.1f}s" if ep_data["wait_seconds"] > 0 else "---"

            lines.append(
                f"{plat_name:<14} {ep_name:<20} {ep_data['strategy']:<15} "
                f"{status_display:<10} {ep_data['used']}/{ep_data['max']:<10} "
                f"{ep_data['usage_percent']:>5.1f}%  "
                f"{wait_str:>7}  {ep_data['total_today']:>7}  "
                f"{cost_str:>8}  {budget_str:>8}"
            )

    lines.append("")
    return "\n".join(lines)


def _format_daily_report(report: Dict[str, Any]) -> str:
    """Format the daily report into readable text."""
    lines: List[str] = []
    lines.append(f"\n=== Daily Report: {report['date']} ===\n")

    for plat_name, plat_data in sorted(report.get("platforms", {}).items()):
        lines.append(f"  {plat_name}:")
        for ep_name, ep_data in sorted(plat_data.get("endpoints", {}).items()):
            cost_str = f" (${ep_data['cost']:.4f})" if ep_data.get("cost", 0) > 0 else ""
            blocked_str = f" [{ep_data['blocked']} blocked]" if ep_data.get("blocked", 0) > 0 else ""
            lines.append(
                f"    {ep_name:<20} {ep_data['requests']:>5} requests"
                f"{cost_str}{blocked_str}  [{ep_data['status']}]"
            )
        lines.append(
            f"    {'SUBTOTAL':<20} {plat_data['total_requests']:>5} requests"
            f"  ${plat_data['total_cost']:.4f}"
        )
        lines.append("")

    totals = report["totals"]
    lines.append(f"  TOTAL: {totals['requests']} requests  "
                 f"${totals['cost']:.4f} cost  "
                 f"{totals['blocked']} blocked")
    lines.append("")
    return "\n".join(lines)


def _format_cost_report(report: Dict[str, Any]) -> str:
    """Format the cost report into readable text."""
    lines: List[str] = []
    lines.append(f"\n=== Cost Report: Last {report['days']} Days ===\n")

    for day in report["daily"]:
        if day["total"] > 0:
            plat_parts = ", ".join(
                f"{p}: ${c:.4f}" for p, c in sorted(day["platforms"].items()) if c > 0
            )
            lines.append(f"  {day['date']}  ${day['total']:.4f}  ({plat_parts})")
        else:
            lines.append(f"  {day['date']}  $0.0000")

    lines.append("")
    lines.append("  Platform Totals:")
    for plat, total in sorted(
        report["platform_totals"].items(), key=lambda x: -x[1]
    ):
        lines.append(f"    {plat:<14}  ${total:.4f}")

    lines.append(f"\n  GRAND TOTAL: ${report['grand_total']:.4f}")
    lines.append("")
    return "\n".join(lines)


def _format_platform_status(status: Dict[str, Any]) -> str:
    """Format a single platform's status into readable text."""
    lines: List[str] = []
    plat = status["platform"]
    lines.append(f"\n=== {plat.upper()} Rate Limits ===\n")

    for ep_name, ep_data in sorted(status.get("endpoints", {}).items()):
        lines.append(f"  Endpoint: {ep_name}")
        lines.append(f"    Strategy:  {ep_data['strategy']}")
        lines.append(f"    Status:    {ep_data['status'].upper()}")
        lines.append(f"    Usage:     {ep_data['used']}/{ep_data['max']} "
                     f"({ep_data['usage_percent']}%)")
        lines.append(f"    Window:    {ep_data['window_seconds']}s")

        if ep_data["wait_seconds"] > 0:
            lines.append(f"    Wait:      {ep_data['wait_seconds']:.1f}s")
        lines.append(f"    Today:     {ep_data['total_today']} requests")

        if ep_data["cost_today"] > 0:
            lines.append(f"    Cost:      ${ep_data['cost_today']:.4f}")
        if ep_data.get("daily_budget"):
            remaining = ep_data.get("budget_remaining", 0)
            lines.append(f"    Budget:    ${ep_data['daily_budget']:.2f} "
                         f"(${remaining:.4f} remaining)")
        if ep_data["blocked_count"] > 0:
            lines.append(f"    Blocked:   {ep_data['blocked_count']} times")
        if ep_data.get("last_request"):
            lines.append(f"    Last req:  {ep_data['last_request']}")
        lines.append("")

    return "\n".join(lines)


def _format_stats(stats: Dict[str, Any]) -> str:
    """Format manager stats into readable text."""
    lines: List[str] = []
    lines.append("\n=== Rate Limit Manager Stats ===\n")
    lines.append(f"  Limiters registered:  {stats['total_limiters']}")
    lines.append(f"  Platforms registered: {stats['platforms_registered']}")
    lines.append(f"  Platforms active:     {stats['platforms_active_today']}")
    lines.append(f"  Requests today:       {stats['total_requests_today']}")
    lines.append(f"  Cost today:           ${stats['total_cost_today']:.4f}")
    lines.append(f"  Blocked today:        {stats['total_blocked_today']}")
    lines.append(f"  State file:           {stats['state_file']}")
    lines.append(f"  Timestamp:            {stats['timestamp']}")
    lines.append("")
    lines.append("  Status distribution:")
    for status, count in sorted(stats["status_counts"].items()):
        lines.append(f"    {status:<12}  {count}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for rate limit management."""
    parser = argparse.ArgumentParser(
        prog="rate_limit_manager",
        description="Rate Limit Manager — OpenClaw Empire Edition",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # -- status ---------------------------------------------------------------
    sub_status = subparsers.add_parser("status", help="Show all rate limit statuses")
    sub_status.add_argument("--json", action="store_true", help="Output raw JSON")

    # -- platform -------------------------------------------------------------
    sub_platform = subparsers.add_parser("platform", help="Show status for a specific platform")
    sub_platform.add_argument("name", type=str, help="Platform name (e.g., anthropic)")
    sub_platform.add_argument("--json", action="store_true", help="Output raw JSON")

    # -- daily ----------------------------------------------------------------
    sub_daily = subparsers.add_parser("daily", help="Show today's usage report")
    sub_daily.add_argument("--json", action="store_true", help="Output raw JSON")

    # -- costs ----------------------------------------------------------------
    sub_costs = subparsers.add_parser("costs", help="Show cost breakdown")
    sub_costs.add_argument("--days", type=int, default=7, help="Number of days (default: 7)")
    sub_costs.add_argument("--json", action="store_true", help="Output raw JSON")

    # -- configure ------------------------------------------------------------
    sub_configure = subparsers.add_parser("configure", help="Override rate limit config")
    sub_configure.add_argument("platform_name", type=str, help="Platform name")
    sub_configure.add_argument("endpoint", type=str, help="Endpoint name")
    sub_configure.add_argument("--max-requests", type=int, required=True, help="Max requests")
    sub_configure.add_argument("--window", type=int, required=True, help="Window in seconds")
    sub_configure.add_argument("--strategy", type=str, default="sliding_window",
                               choices=[s.value for s in RateLimitStrategy],
                               help="Rate limit strategy")
    sub_configure.add_argument("--burst", type=int, default=None, help="Burst limit")
    sub_configure.add_argument("--cost", type=float, default=0.0, help="Cost per request")
    sub_configure.add_argument("--budget", type=float, default=None, help="Daily budget")

    # -- reset ----------------------------------------------------------------
    sub_reset = subparsers.add_parser("reset", help="Reset rate limit counters")
    sub_reset.add_argument("--platform", type=str, default=None,
                           help="Platform to reset (all if omitted)")
    sub_reset.add_argument("--endpoint", type=str, default=None,
                           help="Specific endpoint to reset")
    sub_reset.add_argument("--daily", action="store_true",
                           help="Reset daily counters only")

    # -- stats ----------------------------------------------------------------
    sub_stats = subparsers.add_parser("stats", help="Show manager statistics")
    sub_stats.add_argument("--json", action="store_true", help="Output raw JSON")

    # -- check ----------------------------------------------------------------
    sub_check = subparsers.add_parser("check", help="Check if a request would be allowed")
    sub_check.add_argument("platform_name", type=str, help="Platform name")
    sub_check.add_argument("endpoint", type=str, nargs="?", default="default",
                           help="Endpoint name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    mgr = get_rate_limit_manager()

    # -- Dispatch commands ----------------------------------------------------

    if args.command == "status":
        data = mgr.get_all_status()
        if args.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            print(_format_status_table(data))

    elif args.command == "platform":
        try:
            platform = Platform(args.name.lower())
        except ValueError:
            print(f"Unknown platform: {args.name}")
            print(f"Valid platforms: {', '.join(p.value for p in Platform)}")
            sys.exit(1)
        data = mgr.get_platform_status(platform)
        if args.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            print(_format_platform_status(data))

    elif args.command == "daily":
        data = mgr.get_daily_report()
        if args.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            print(_format_daily_report(data))

    elif args.command == "costs":
        data = mgr.get_cost_report(days=args.days)
        if args.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            print(_format_cost_report(data))

    elif args.command == "configure":
        try:
            platform = Platform(args.platform_name.lower())
        except ValueError:
            print(f"Unknown platform: {args.platform_name}")
            sys.exit(1)
        config = RateLimitConfig(
            platform=platform,
            endpoint=args.endpoint,
            max_requests=args.max_requests,
            window_seconds=args.window,
            strategy=RateLimitStrategy(args.strategy),
            burst_limit=args.burst,
            cost_per_request=args.cost,
            daily_budget=args.budget,
        )
        mgr.configure(platform, args.endpoint, config)
        print(f"Configured {platform.value}:{args.endpoint} -> "
              f"{args.max_requests} req / {args.window}s "
              f"({args.strategy})")

    elif args.command == "reset":
        if args.daily:
            mgr.reset_daily()
            print("Daily counters reset for all platforms.")
        elif args.platform:
            try:
                platform = Platform(args.platform.lower())
            except ValueError:
                print(f"Unknown platform: {args.platform}")
                sys.exit(1)
            if args.endpoint:
                mgr.reset_endpoint(platform, args.endpoint)
                print(f"Reset {platform.value}:{args.endpoint}")
            else:
                mgr.reset_platform(platform)
                print(f"Reset all endpoints for {platform.value}")
        else:
            mgr.reset_daily()
            print("Daily counters reset for all platforms.")

    elif args.command == "stats":
        data = mgr.get_stats()
        if args.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            print(_format_stats(data))

    elif args.command == "check":
        try:
            platform = Platform(args.platform_name.lower())
        except ValueError:
            print(f"Unknown platform: {args.platform_name}")
            sys.exit(1)
        status = mgr.check(platform, args.endpoint)
        wait = mgr.get_wait_time(platform, args.endpoint)
        print(f"Platform:  {platform.value}")
        print(f"Endpoint:  {args.endpoint}")
        print(f"Status:    {status.value.upper()}")
        if wait > 0:
            print(f"Wait:      {wait:.1f}s")
        else:
            print("Wait:      none (ready)")

    # Save state on exit
    mgr.save_state()


if __name__ == "__main__":
    main()
