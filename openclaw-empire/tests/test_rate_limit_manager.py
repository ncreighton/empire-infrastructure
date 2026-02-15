"""Test rate_limit_manager — OpenClaw Empire."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_data(tmp_path, monkeypatch):
    """Redirect rate limit data to temp dir."""
    rl_dir = tmp_path / "rate_limits"
    rl_dir.mkdir(parents=True, exist_ok=True)
    (rl_dir / "daily").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.rate_limit_manager.RATELIMIT_DATA_DIR", rl_dir)
    monkeypatch.setattr("src.rate_limit_manager.STATE_FILE", rl_dir / "state.json")
    monkeypatch.setattr("src.rate_limit_manager.CONFIG_FILE", rl_dir / "config.json")
    monkeypatch.setattr("src.rate_limit_manager.DAILY_DIR", rl_dir / "daily")
    monkeypatch.setattr("src.rate_limit_manager.COST_HISTORY_FILE", rl_dir / "cost_history.json")
    # Reset singleton
    import src.rate_limit_manager as rl_mod
    rl_mod._rate_limit_manager = None
    yield


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.rate_limit_manager import (
    DEFAULT_LIMITS,
    LimitStatus,
    Platform,
    RateLimitConfig,
    RateLimitExceeded,
    RateLimitManager,
    RateLimitState,
    RateLimitStrategy,
    RateLimiter,
    get_rate_limit_manager,
)


# ===================================================================
# Enum tests
# ===================================================================

class TestPlatformEnum:
    """Test Platform enum values."""

    def test_all_15_platforms_exist(self):
        expected = [
            "anthropic", "wordpress", "pinterest", "instagram", "facebook",
            "twitter", "linkedin", "substack", "google", "amazon",
            "etsy", "n8n", "geelark", "screenpipe", "internal",
        ]
        actual = [p.value for p in Platform]
        assert sorted(actual) == sorted(expected)
        assert len(actual) == 15


class TestRateLimitStrategyEnum:
    def test_strategy_values(self):
        assert RateLimitStrategy.FIXED_WINDOW.value == "fixed_window"
        assert RateLimitStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert RateLimitStrategy.TOKEN_BUCKET.value == "token_bucket"
        assert RateLimitStrategy.LEAKY_BUCKET.value == "leaky_bucket"


class TestLimitStatusEnum:
    def test_status_values(self):
        assert LimitStatus.OK.value == "ok"
        assert LimitStatus.WARNING.value == "warning"
        assert LimitStatus.THROTTLED.value == "throttled"
        assert LimitStatus.BLOCKED.value == "blocked"


# ===================================================================
# RateLimitConfig
# ===================================================================

class TestRateLimitConfig:
    def test_to_dict_and_from_dict_roundtrip(self):
        cfg = RateLimitConfig(
            platform=Platform.ANTHROPIC,
            endpoint="messages",
            max_requests=50,
            window_seconds=60,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            burst_limit=10,
            cost_per_request=0.01,
            daily_budget=50.0,
        )
        d = cfg.to_dict()
        assert d["platform"] == "anthropic"
        assert d["strategy"] == "token_bucket"

        restored = RateLimitConfig.from_dict(d)
        assert restored.platform == Platform.ANTHROPIC
        assert restored.strategy == RateLimitStrategy.TOKEN_BUCKET
        assert restored.burst_limit == 10


# ===================================================================
# RateLimiter — Fixed Window
# ===================================================================

class TestFixedWindow:
    """Test fixed window rate limiting algorithm."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        cfg = RateLimitConfig(
            platform=Platform.INTERNAL,
            endpoint="test",
            max_requests=5,
            window_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW,
        )
        limiter = RateLimiter(cfg)
        for _ in range(5):
            result = await limiter.acquire(max_wait=0)
            assert result is True

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        cfg = RateLimitConfig(
            platform=Platform.INTERNAL,
            endpoint="test",
            max_requests=3,
            window_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW,
        )
        limiter = RateLimiter(cfg)
        for _ in range(3):
            await limiter.acquire(max_wait=0)
        result = await limiter.acquire(max_wait=0)
        assert result is False


# ===================================================================
# RateLimiter — Sliding Window
# ===================================================================

class TestSlidingWindow:
    """Test sliding window rate limiting algorithm."""

    @pytest.mark.asyncio
    async def test_allows_within_window(self):
        cfg = RateLimitConfig(
            platform=Platform.WORDPRESS,
            endpoint="posts",
            max_requests=5,
            window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        limiter = RateLimiter(cfg)
        for _ in range(5):
            result = await limiter.acquire(max_wait=0)
            assert result is True

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        cfg = RateLimitConfig(
            platform=Platform.WORDPRESS,
            endpoint="posts",
            max_requests=3,
            window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        limiter = RateLimiter(cfg)
        for _ in range(3):
            await limiter.acquire(max_wait=0)
        result = await limiter.acquire(max_wait=0)
        assert result is False

    @pytest.mark.asyncio
    async def test_recovers_after_window_expires(self):
        cfg = RateLimitConfig(
            platform=Platform.INTERNAL,
            endpoint="test",
            max_requests=2,
            window_seconds=0.1,  # 100ms window
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            retry_after_seconds=0.01,
        )
        limiter = RateLimiter(cfg)
        await limiter.acquire(max_wait=0)
        await limiter.acquire(max_wait=0)
        result = await limiter.acquire(max_wait=0)
        assert result is False
        # Wait for window to pass
        await asyncio.sleep(0.15)
        result = await limiter.acquire(max_wait=0)
        assert result is True


# ===================================================================
# RateLimiter — Token Bucket
# ===================================================================

class TestTokenBucket:
    """Test token bucket rate limiting algorithm."""

    @pytest.mark.asyncio
    async def test_burst_capacity(self):
        cfg = RateLimitConfig(
            platform=Platform.ANTHROPIC,
            endpoint="test_bucket",
            max_requests=10,
            window_seconds=10,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            burst_limit=5,
        )
        limiter = RateLimiter(cfg)
        # Should allow burst up to burst_limit
        successes = 0
        for _ in range(5):
            if await limiter.acquire(max_wait=0):
                successes += 1
        assert successes == 5

    @pytest.mark.asyncio
    async def test_blocks_after_tokens_exhausted(self):
        cfg = RateLimitConfig(
            platform=Platform.ANTHROPIC,
            endpoint="test_bucket",
            max_requests=10,
            window_seconds=10,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            burst_limit=3,
        )
        limiter = RateLimiter(cfg)
        for _ in range(3):
            await limiter.acquire(max_wait=0)
        # Tokens should be exhausted
        result = await limiter.acquire(max_wait=0)
        assert result is False

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self):
        cfg = RateLimitConfig(
            platform=Platform.INTERNAL,
            endpoint="refill_test",
            max_requests=100,
            window_seconds=1,  # 100 req/sec refill rate
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            burst_limit=3,
        )
        limiter = RateLimiter(cfg)
        # Consume all tokens
        for _ in range(3):
            await limiter.acquire(max_wait=0)
        # Wait for refill (100 req/sec = ~10ms per token)
        await asyncio.sleep(0.05)
        result = await limiter.acquire(max_wait=0)
        assert result is True


# ===================================================================
# RateLimiter — Leaky Bucket
# ===================================================================

class TestLeakyBucket:
    """Test leaky bucket rate limiting algorithm."""

    @pytest.mark.asyncio
    async def test_first_request_always_allowed(self):
        cfg = RateLimitConfig(
            platform=Platform.GOOGLE,
            endpoint="pagespeed",
            max_requests=10,
            window_seconds=10,
            strategy=RateLimitStrategy.LEAKY_BUCKET,
        )
        limiter = RateLimiter(cfg)
        result = await limiter.acquire(max_wait=0)
        assert result is True

    @pytest.mark.asyncio
    async def test_enforces_interval(self):
        cfg = RateLimitConfig(
            platform=Platform.GOOGLE,
            endpoint="pagespeed",
            max_requests=10,
            window_seconds=10,  # 1 req per second
            strategy=RateLimitStrategy.LEAKY_BUCKET,
        )
        limiter = RateLimiter(cfg)
        await limiter.acquire(max_wait=0)
        # Immediately try again -- should be blocked
        result = await limiter.acquire(max_wait=0)
        assert result is False


# ===================================================================
# RateLimiter — Status and usage
# ===================================================================

class TestLimiterStatus:
    def test_check_status_ok(self):
        cfg = RateLimitConfig(
            platform=Platform.INTERNAL, endpoint="status",
            max_requests=100, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        limiter = RateLimiter(cfg)
        status = limiter.check_status()
        assert status == LimitStatus.OK

    def test_get_usage_returns_dict(self):
        cfg = RateLimitConfig(
            platform=Platform.WORDPRESS, endpoint="posts",
            max_requests=30, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        limiter = RateLimiter(cfg)
        usage = limiter.get_usage()
        assert usage["platform"] == "wordpress"
        assert usage["endpoint"] == "posts"
        assert "used" in usage
        assert "max" in usage

    def test_reset_clears_state(self):
        cfg = RateLimitConfig(
            platform=Platform.INTERNAL, endpoint="reset",
            max_requests=10, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        limiter = RateLimiter(cfg)
        limiter._state.total_requests_today = 50
        limiter.reset()
        assert limiter.state.total_requests_today == 0


# ===================================================================
# RateLimiter — Daily budget
# ===================================================================

class TestDailyBudget:
    @pytest.mark.asyncio
    async def test_budget_blocks_when_exceeded(self):
        cfg = RateLimitConfig(
            platform=Platform.ANTHROPIC, endpoint="budget",
            max_requests=100, window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            cost_per_request=10.0,
            daily_budget=25.0,
        )
        limiter = RateLimiter(cfg)
        # First two should succeed ($10 + $10 = $20)
        r1 = await limiter.acquire(max_wait=0)
        r2 = await limiter.acquire(max_wait=0)
        assert r1 is True
        assert r2 is True
        # Third should be blocked ($30 > $25 budget)
        r3 = await limiter.acquire(max_wait=0)
        assert r3 is False


# ===================================================================
# RateLimitExceeded exception
# ===================================================================

class TestRateLimitExceeded:
    def test_attributes(self):
        exc = RateLimitExceeded(
            "rate limited", platform=Platform.TWITTER,
            endpoint="tweets", retry_after=30.0,
        )
        assert exc.platform == Platform.TWITTER
        assert exc.endpoint == "tweets"
        assert exc.retry_after == 30.0
        assert "rate limited" in str(exc)


# ===================================================================
# Platform-specific default configs
# ===================================================================

class TestDefaultLimits:
    """Verify default limits are configured for all 15 platforms."""

    def test_all_platforms_have_defaults(self):
        for platform in Platform:
            assert platform in DEFAULT_LIMITS, f"Missing defaults for {platform.value}"

    def test_anthropic_has_multiple_endpoints(self):
        anthropic_configs = DEFAULT_LIMITS[Platform.ANTHROPIC]
        endpoints = [c.endpoint for c in anthropic_configs]
        assert "messages" in endpoints
        assert "messages_haiku" in endpoints
        assert "batch" in endpoints

    def test_wordpress_has_posts_media_reads(self):
        wp_configs = DEFAULT_LIMITS[Platform.WORDPRESS]
        endpoints = [c.endpoint for c in wp_configs]
        assert "posts" in endpoints
        assert "media" in endpoints
        assert "reads" in endpoints


# ===================================================================
# RateLimitManager
# ===================================================================

class TestRateLimitManager:
    """Test the central rate limit manager."""

    @pytest.mark.asyncio
    async def test_acquire_succeeds_for_valid_platform(self):
        mgr = RateLimitManager()
        result = await mgr.acquire(Platform.INTERNAL, "default", max_wait=0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_auto_creates_limiter_for_unknown_endpoint(self):
        mgr = RateLimitManager()
        result = await mgr.acquire(Platform.INTERNAL, "custom_endpoint", max_wait=0)
        assert result is True

    def test_check_returns_status(self):
        mgr = RateLimitManager()
        status = mgr.check(Platform.WORDPRESS, "posts")
        assert isinstance(status, LimitStatus)

    def test_get_wait_time(self):
        mgr = RateLimitManager()
        wait = mgr.get_wait_time(Platform.WORDPRESS, "posts")
        assert wait == 0.0  # no requests yet, should be 0

    def test_acquire_sync(self):
        mgr = RateLimitManager()
        result = mgr.acquire_sync(Platform.INTERNAL, "default", max_wait=0)
        assert result is True


# ===================================================================
# State serialization
# ===================================================================

class TestStateSerialization:
    def test_rate_limit_state_roundtrip(self):
        cfg = RateLimitConfig(
            platform=Platform.TWITTER, endpoint="tweets",
            max_requests=50, window_seconds=86400,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        state = RateLimitState(config=cfg)
        state.total_requests_today = 10
        state.total_cost_today = 1.5
        d = state.to_dict()
        restored = RateLimitState.from_dict(d)
        assert restored.total_requests_today == 10
        assert restored.total_cost_today == 1.5
        assert restored.config.platform == Platform.TWITTER


# ===================================================================
# Singleton
# ===================================================================

class TestSingleton:
    def test_get_rate_limit_manager_returns_same_instance(self):
        m1 = get_rate_limit_manager()
        m2 = get_rate_limit_manager()
        assert m1 is m2
