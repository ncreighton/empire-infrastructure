"""Tests for openclaw/automation/retry_engine.py — smart retry logic."""

import asyncio

import pytest

from openclaw.automation.retry_engine import RetryEngine, RetryPolicy, ErrorCategory


@pytest.fixture
def engine():
    """Create a RetryEngine with fast settings for testing."""
    policy = RetryPolicy(
        max_retries=3,
        base_delay_seconds=0.01,
        max_delay_seconds=0.1,
        exponential_base=2.0,
        jitter=False,
    )
    return RetryEngine(policy)


class TestCategorizeError:
    def test_transient_timeout(self, engine):
        assert engine.categorize_error("Connection timeout") == ErrorCategory.TRANSIENT

    def test_rate_limited(self, engine):
        assert engine.categorize_error("429 Too Many Requests") == ErrorCategory.RATE_LIMITED

    def test_credential_error(self, engine):
        assert engine.categorize_error("Invalid password") == ErrorCategory.CREDENTIAL_ERROR

    def test_account_exists(self, engine):
        assert engine.categorize_error("Account already exists") == ErrorCategory.ACCOUNT_EXISTS

    def test_unknown_error(self, engine):
        assert engine.categorize_error("some random thing") == ErrorCategory.UNKNOWN

    def test_captcha_failed(self, engine):
        assert engine.categorize_error("captcha solve failed") == ErrorCategory.CAPTCHA_FAILED

    def test_blocked(self, engine):
        assert engine.categorize_error("IP blocked") == ErrorCategory.BLOCKED

    def test_platform_down(self, engine):
        assert engine.categorize_error("Site under maintenance") == ErrorCategory.PLATFORM_DOWN


class TestShouldRetry:
    def test_should_retry_transient(self, engine):
        assert engine.should_retry("p1", ErrorCategory.TRANSIENT) is True

    def test_should_not_retry_credential_error(self, engine):
        assert engine.should_retry("p1", ErrorCategory.CREDENTIAL_ERROR) is False

    def test_should_not_retry_account_exists(self, engine):
        assert engine.should_retry("p1", ErrorCategory.ACCOUNT_EXISTS) is False


class TestCalculateDelay:
    def test_exponential_increase(self, engine):
        d0 = engine.calculate_delay(0, ErrorCategory.TRANSIENT)
        d1 = engine.calculate_delay(1, ErrorCategory.TRANSIENT)
        d2 = engine.calculate_delay(2, ErrorCategory.TRANSIENT)
        assert d1 > d0
        assert d2 > d1

    def test_caps_at_max_delay(self, engine):
        delay = engine.calculate_delay(100, ErrorCategory.TRANSIENT)
        assert delay <= engine.policy.max_delay_seconds

    def test_rate_limited_multiplier(self, engine):
        d_transient = engine.calculate_delay(0, ErrorCategory.TRANSIENT)
        d_rate_limited = engine.calculate_delay(0, ErrorCategory.RATE_LIMITED)
        # Rate limited should have a 3x multiplier
        assert d_rate_limited > d_transient


class TestExecuteWithRetry:
    def test_success_on_first_try(self, engine):
        async def success_func():
            return {"ok": True}

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                engine.execute_with_retry(success_func, "p1")
            )
            assert result == {"ok": True}
        finally:
            loop.close()

    def test_fails_twice_then_succeeds(self, engine):
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection timeout")
            return "success"

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                engine.execute_with_retry(flaky_func, "p2")
            )
            assert result == "success"
            assert call_count == 3
        finally:
            loop.close()

    def test_raises_after_max_retries(self, engine):
        async def always_fails():
            raise ConnectionError("Connection timeout forever")

        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(ConnectionError):
                loop.run_until_complete(
                    engine.execute_with_retry(always_fails, "p3")
                )
        finally:
            loop.close()

    def test_no_retry_on_credential_error(self, engine):
        call_count = 0

        async def bad_creds():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid password")

        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(ValueError):
                loop.run_until_complete(
                    engine.execute_with_retry(bad_creds, "p4")
                )
            # Should fail immediately without retrying
            assert call_count == 1
        finally:
            loop.close()


class TestStatsAndHistory:
    def test_get_stats_returns_dict(self, engine):
        stats = engine.get_stats()
        assert isinstance(stats, dict)
        assert "total_attempts" in stats
        assert "total_successes" in stats
        assert "total_failures" in stats

    def test_clear_history_clears_platform(self, engine):
        async def success_func():
            return True

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                engine.execute_with_retry(success_func, "p1")
            )
        finally:
            loop.close()

        assert len(engine.history.get("p1", [])) > 0
        engine.clear_history("p1")
        assert len(engine.history.get("p1", [])) == 0

    def test_clear_all_history(self, engine):
        async def success_func():
            return True

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                engine.execute_with_retry(success_func, "p1")
            )
            loop.run_until_complete(
                engine.execute_with_retry(success_func, "p2")
            )
        finally:
            loop.close()

        engine.clear_history()
        assert len(engine.history) == 0
