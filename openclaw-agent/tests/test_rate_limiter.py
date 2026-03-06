"""Tests for openclaw/automation/rate_limiter.py — intelligent rate limiting."""

import time

import pytest

from openclaw.automation.rate_limiter import RateLimiter, RateLimitConfig


@pytest.fixture
def limiter():
    """Create a RateLimiter with tight limits for fast testing."""
    config = RateLimitConfig(
        global_min_delay_seconds=1,
        global_max_delay_seconds=2,
        platform_cooldown_hours=24,
        max_signups_per_hour=5,
        max_signups_per_day=20,
        max_failures_before_pause=3,
        failure_pause_minutes=30,
    )
    return RateLimiter(config)


class TestCanProceed:
    def test_can_proceed_initially(self, limiter):
        ok, reason = limiter.can_proceed("gumroad")
        assert ok is True
        assert reason == ""

    def test_blocked_after_hourly_limit(self):
        config = RateLimitConfig(max_signups_per_hour=2, max_signups_per_day=100)
        lim = RateLimiter(config)
        lim.record_attempt("a")
        lim.record_attempt("b")
        ok, reason = lim.can_proceed("c")
        assert ok is False
        assert "Hourly" in reason or "hourly" in reason.lower()

    def test_platform_cooldown_blocks(self):
        config = RateLimitConfig(
            platform_cooldown_hours=0.001,  # ~3.6 seconds
            max_signups_per_hour=100,
            max_signups_per_day=100,
        )
        lim = RateLimiter(config)
        lim.record_attempt("gumroad")
        ok, reason = lim.can_proceed("gumroad")
        assert ok is False
        assert "cooldown" in reason.lower()


class TestWaitTime:
    def test_wait_time_positive(self, limiter):
        wt = limiter.wait_time("gumroad")
        assert isinstance(wt, float)
        assert wt > 0


class TestRecordSuccessFailure:
    def test_record_success_resets_failures(self, limiter):
        limiter.record_failure("a")
        limiter.record_failure("a")
        assert limiter._consecutive_failures == 2
        limiter.record_success("a")
        assert limiter._consecutive_failures == 0

    def test_record_failure_increments(self, limiter):
        limiter.record_failure("a")
        assert limiter._consecutive_failures == 1
        limiter.record_failure("a")
        assert limiter._consecutive_failures == 2

    def test_pause_after_max_failures(self, limiter):
        assert limiter.is_paused is False
        limiter.record_failure("a")
        limiter.record_failure("a")
        limiter.record_failure("a")
        assert limiter.is_paused is True


class TestCounting:
    def test_signups_this_hour(self, limiter):
        assert limiter.signups_this_hour == 0
        limiter.record_attempt("a")
        limiter.record_attempt("b")
        assert limiter.signups_this_hour == 2

    def test_signups_today(self, limiter):
        assert limiter.signups_today == 0
        limiter.record_attempt("a")
        assert limiter.signups_today == 1


class TestReset:
    def test_reset_clears_all_state(self, limiter):
        limiter.record_attempt("a")
        limiter.record_failure("a")
        limiter.reset()
        assert limiter.signups_this_hour == 0
        assert limiter.signups_today == 0
        assert limiter._consecutive_failures == 0
        assert limiter.is_paused is False


class TestGetStats:
    def test_get_stats_keys(self, limiter):
        stats = limiter.get_stats()
        expected_keys = {
            "signups_this_hour",
            "signups_today",
            "max_per_hour",
            "max_per_day",
            "consecutive_failures",
            "is_paused",
            "paused_until",
            "platform_cooldowns",
            "config",
        }
        assert expected_keys.issubset(stats.keys())

    def test_get_stats_after_activity(self, limiter):
        limiter.record_attempt("gumroad")
        stats = limiter.get_stats()
        assert stats["signups_this_hour"] == 1
        assert stats["signups_today"] == 1
        assert "gumroad" in stats["platform_cooldowns"]
