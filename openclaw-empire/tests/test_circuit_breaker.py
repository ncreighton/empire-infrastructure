"""Test circuit_breaker — OpenClaw Empire."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# We need to redirect the module's data directory before importing
# so that tests don't touch production state.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_data(tmp_path, monkeypatch):
    """Redirect all circuit_breaker paths to a temp directory."""
    monkeypatch.setattr(
        "src.circuit_breaker.BREAKER_DATA_DIR", tmp_path / "circuit_breaker"
    )
    monkeypatch.setattr(
        "src.circuit_breaker.BREAKERS_FILE", tmp_path / "circuit_breaker" / "breakers.json"
    )
    monkeypatch.setattr(
        "src.circuit_breaker.HISTORY_FILE", tmp_path / "circuit_breaker" / "history.json"
    )
    monkeypatch.setattr(
        "src.circuit_breaker.STATS_FILE", tmp_path / "circuit_breaker" / "stats.json"
    )
    (tmp_path / "circuit_breaker").mkdir(parents=True, exist_ok=True)
    # Reset the singleton between tests
    import src.circuit_breaker as cb
    cb._registry = None
    yield


# ---------------------------------------------------------------------------
# Imports (after fixture definition so monkeypatch can intercept)
# ---------------------------------------------------------------------------
from src.circuit_breaker import (
    BreakerRegistry,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RECOVERY_TIMEOUT,
    DEFAULT_RETRYABLE_CODES,
    DEFAULT_SUCCESS_THRESHOLD,
    ErrorCode,
    ErrorContext,
    NON_RETRYABLE_CODES,
    NETWORK_RETRYABLE,
    RATE_LIMIT_RETRYABLE,
    RetryPolicy,
    classify_error,
    get_breaker,
    get_breaker_registry,
    with_retry,
)


# ===================================================================
# ErrorCode enum
# ===================================================================

class TestErrorCode:
    """Test ErrorCode enum values and categories."""

    def test_network_error_codes_exist(self):
        assert ErrorCode.E1001.value == "NETWORK_TIMEOUT"
        assert ErrorCode.E1002.value == "NETWORK_UNREACHABLE"
        assert ErrorCode.E1003.value == "DNS_RESOLUTION_FAILED"
        assert ErrorCode.E1004.value == "CONNECTION_REFUSED"
        assert ErrorCode.E1005.value == "SSL_ERROR"

    def test_auth_error_codes_exist(self):
        assert ErrorCode.E2001.value == "AUTH_EXPIRED"
        assert ErrorCode.E2002.value == "AUTH_INVALID"
        assert ErrorCode.E2003.value == "AUTH_INSUFFICIENT_SCOPE"

    def test_wordpress_error_codes_exist(self):
        assert ErrorCode.E3001.value == "WP_API_ERROR"
        assert ErrorCode.E3002.value == "WP_POST_NOT_FOUND"
        assert ErrorCode.E3003.value == "WP_MEDIA_UPLOAD_FAILED"
        assert ErrorCode.E3004.value == "WP_RATE_LIMITED"

    def test_anthropic_error_codes_exist(self):
        assert ErrorCode.E4001.value == "ANTHROPIC_RATE_LIMITED"
        assert ErrorCode.E4002.value == "ANTHROPIC_OVERLOADED"
        assert ErrorCode.E4003.value == "ANTHROPIC_CONTEXT_TOO_LONG"
        assert ErrorCode.E4004.value == "ANTHROPIC_INVALID_REQUEST"

    def test_platform_error_codes_exist(self):
        assert ErrorCode.E6001.value == "PLATFORM_BANNED"
        assert ErrorCode.E6002.value == "PLATFORM_CAPTCHA"
        assert ErrorCode.E6003.value == "PLATFORM_RATE_LIMITED"

    def test_internal_error_codes_exist(self):
        assert ErrorCode.E9001.value == "INTERNAL_ERROR"
        assert ErrorCode.E9002.value == "CONFIG_MISSING"
        assert ErrorCode.E9003.value == "DATA_CORRUPTION"
        assert ErrorCode.E9004.value == "DEPENDENCY_UNAVAILABLE"

    def test_retryable_code_sets_are_disjoint_from_non_retryable(self):
        retryable = NETWORK_RETRYABLE | RATE_LIMIT_RETRYABLE
        assert retryable.isdisjoint(NON_RETRYABLE_CODES)


# ===================================================================
# ErrorContext
# ===================================================================

class TestErrorContext:
    """Test structured error context creation and serialization."""

    def test_create_with_defaults(self):
        ctx = ErrorContext(
            code=ErrorCode.E1001,
            message="timeout",
            module="wp",
            operation="publish",
        )
        assert ctx.code == ErrorCode.E1001
        assert ctx.retryable is True
        assert len(ctx.recovery_hints) > 0
        assert ctx.timestamp != ""

    def test_non_retryable_code_sets_retryable_false(self):
        ctx = ErrorContext(
            code=ErrorCode.E2002,
            message="invalid key",
            module="auth",
            operation="login",
        )
        assert ctx.retryable is False

    def test_to_dict_and_from_dict_roundtrip(self):
        ctx = ErrorContext(
            code=ErrorCode.E3001,
            message="API error",
            module="wordpress",
            operation="create_post",
            metadata={"site_id": "witchcraft"},
        )
        d = ctx.to_dict()
        assert d["code"] == "E3001"
        assert d["code_value"] == "WP_API_ERROR"
        restored = ErrorContext.from_dict(d)
        assert restored.code == ErrorCode.E3001
        assert restored.module == "wordpress"

    def test_str_representation(self):
        ctx = ErrorContext(
            code=ErrorCode.E4001,
            message="rate limit hit",
            module="anthropic",
            operation="call",
        )
        s = str(ctx)
        assert "E4001" in s
        assert "ANTHROPIC_RATE_LIMITED" in s


# ===================================================================
# classify_error
# ===================================================================

class TestClassifyError:
    """Test the error classifier mapping exceptions to ErrorCodes."""

    def test_timeout_error(self):
        ctx = classify_error(TimeoutError("request timed out"), "api", "fetch")
        assert ctx.code == ErrorCode.E1001

    def test_connection_refused(self):
        ctx = classify_error(ConnectionRefusedError("conn refused"), "api", "fetch")
        assert ctx.code == ErrorCode.E1004

    def test_ssl_error(self):
        ctx = classify_error(Exception("SSL certificate verify failed"), "api", "fetch")
        assert ctx.code == ErrorCode.E1005

    def test_unauthorized_error(self):
        ctx = classify_error(Exception("HTTP 401 Unauthorized"), "auth", "check")
        assert ctx.code == ErrorCode.E2001

    def test_unknown_falls_back_to_internal(self):
        ctx = classify_error(Exception("something weird"), "misc", "op")
        assert ctx.code == ErrorCode.E9001

    def test_wp_post_not_found(self):
        ctx = classify_error(Exception("HTTP 404 post not found"), "wp", "get")
        assert ctx.code == ErrorCode.E3002


# ===================================================================
# CircuitState enum
# ===================================================================

class TestCircuitState:
    def test_values(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


# ===================================================================
# CircuitBreaker — state transitions
# ===================================================================

class TestCircuitBreaker:
    """Test the CircuitBreaker state machine transitions."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_failures_trigger_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_open_rejects_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_recovery_timeout_allows_half_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_restores_closed(self):
        cb = CircuitBreaker(
            name="test", failure_threshold=1, success_threshold=2, recovery_timeout=0.05
        )
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.1)
        cb.can_execute()  # transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker(
            name="test", failure_threshold=1, recovery_timeout=0.05
        )
        cb.record_failure()
        time.sleep(0.1)
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_reset_returns_to_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_get_stats_returns_expected_keys(self):
        cb = CircuitBreaker(name="test")
        cb.record_success()
        cb.record_failure()
        stats = cb.get_stats()
        assert stats["name"] == "test"
        assert stats["total_calls"] == 2
        assert stats["total_failures"] == 1
        assert "failure_rate_pct" in stats

    def test_to_dict_and_from_dict_roundtrip(self):
        cb = CircuitBreaker(name="round-trip", failure_threshold=10)
        cb.record_success()
        d = cb.to_dict()
        restored = CircuitBreaker.from_dict(d)
        assert restored.name == "round-trip"
        assert restored.failure_threshold == 10
        assert restored.total_successes == 1

    def test_record_failure_with_error_context(self):
        cb = CircuitBreaker(name="test", failure_threshold=10)
        err = ErrorContext(
            code=ErrorCode.E1001,
            message="timeout",
            module="api",
            operation="call",
        )
        cb.record_failure(err)
        assert len(cb.recent_errors) == 1
        assert cb.recent_errors[0]["code"] == "E1001"

    def test_success_decrements_failure_count_in_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=10)
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2
        cb.record_success()
        assert cb.failure_count == 1


# ===================================================================
# CircuitOpenError
# ===================================================================

class TestCircuitOpenError:
    def test_message_contains_breaker_name(self):
        exc = CircuitOpenError("my-service", recovery_timeout=60.0)
        assert "my-service" in str(exc)
        assert "OPEN" in str(exc)


# ===================================================================
# RetryPolicy
# ===================================================================

class TestRetryPolicy:
    """Test retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        func = AsyncMock(return_value="ok")
        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        result = await policy.execute(func)
        assert result == "ok"
        func.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self):
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timeout")
            return "success"

        policy = RetryPolicy(
            max_retries=5,
            base_delay=0.01,
            max_delay=0.05,
            jitter=False,
            module_name="test",
        )
        result = await policy.execute(flaky)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_does_not_retry_non_retryable_code(self):
        async def fatal():
            raise Exception("invalid_api_key from anthropic")

        policy = RetryPolicy(
            max_retries=5,
            base_delay=0.01,
            module_name="anthropic",
        )
        with pytest.raises(Exception, match="invalid_api_key"):
            await policy.execute(fatal)

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        breaker = CircuitBreaker(name="test-retry", failure_threshold=10)
        func = AsyncMock(return_value="ok")
        policy = RetryPolicy(
            max_retries=2,
            base_delay=0.01,
            circuit_breaker=breaker,
        )
        result = await policy.execute(func)
        assert result == "ok"
        assert breaker.total_successes == 1

    @pytest.mark.asyncio
    async def test_circuit_open_raises_circuit_open_error(self):
        breaker = CircuitBreaker(name="blocked", failure_threshold=1)
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        policy = RetryPolicy(
            max_retries=2,
            base_delay=0.01,
            circuit_breaker=breaker,
        )
        with pytest.raises(CircuitOpenError):
            await policy.execute(AsyncMock())

    def test_calculate_delay_exponential(self):
        policy = RetryPolicy(
            base_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=False
        )
        assert policy._calculate_delay(0) == 1.0
        assert policy._calculate_delay(1) == 2.0
        assert policy._calculate_delay(2) == 4.0

    def test_calculate_delay_capped_at_max(self):
        policy = RetryPolicy(
            base_delay=1.0, exponential_base=2.0, max_delay=5.0, jitter=False
        )
        assert policy._calculate_delay(10) == 5.0


# ===================================================================
# @with_retry decorator
# ===================================================================

class TestWithRetryDecorator:
    """Test the @with_retry decorator on async functions."""

    @pytest.mark.asyncio
    async def test_decorated_function_succeeds(self):
        @with_retry(service_name="test-deco", max_retries=2, base_delay=0.01)
        async def my_func(x: int) -> int:
            return x * 2

        result = await my_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_decorated_function_retries_on_timeout(self):
        call_count = 0

        @with_retry(
            service_name="test-retry-deco",
            max_retries=3,
            base_delay=0.01,
            use_circuit_breaker=False,
        )
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("boom")
            return "ok"

        result = await flaky_func()
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorated_function_raises_after_exhaustion(self):
        @with_retry(
            service_name="fail-deco",
            max_retries=2,
            base_delay=0.01,
            use_circuit_breaker=False,
        )
        async def always_fail():
            raise TimeoutError("permanent timeout")

        with pytest.raises(TimeoutError, match="permanent timeout"):
            await always_fail()


# ===================================================================
# BreakerRegistry
# ===================================================================

class TestBreakerRegistry:
    """Test the global breaker registry singleton behavior."""

    def test_get_breaker_returns_same_instance(self):
        registry = BreakerRegistry()
        registry._loaded = True
        b1 = registry.get_breaker("api-service")
        b2 = registry.get_breaker("api-service")
        assert b1 is b2

    def test_get_breaker_creates_new_for_new_name(self):
        registry = BreakerRegistry()
        registry._loaded = True
        b1 = registry.get_breaker("service-a")
        b2 = registry.get_breaker("service-b")
        assert b1 is not b2
        assert b1.name == "service-a"
        assert b2.name == "service-b"

    def test_record_error_adds_to_history(self):
        registry = BreakerRegistry()
        ctx = ErrorContext(
            code=ErrorCode.E1001,
            message="timeout",
            module="api",
            operation="call",
        )
        registry.record_error(ctx)
        history = registry.get_error_history(limit=10)
        assert len(history) == 1
        assert history[0]["code"] == "E1001"

    def test_get_error_history_filters_by_code(self):
        registry = BreakerRegistry()
        registry.record_error(
            ErrorContext(code=ErrorCode.E1001, message="t1", module="a", operation="o")
        )
        registry.record_error(
            ErrorContext(code=ErrorCode.E4001, message="t2", module="b", operation="o")
        )
        filtered = registry.get_error_history(code="E1001")
        assert len(filtered) == 1
        assert filtered[0]["code"] == "E1001"

    def test_reset_all_clears_state(self, tmp_path):
        registry = BreakerRegistry()
        registry._loaded = True
        b = registry.get_breaker("svc")
        b.record_failure()
        registry.record_error(
            ErrorContext(code=ErrorCode.E9001, message="err", module="m", operation="o")
        )
        registry.reset_all()
        assert b.state == CircuitState.CLOSED
        assert len(registry.get_error_history()) == 0

    def test_remove_breaker(self):
        registry = BreakerRegistry()
        registry._loaded = True
        registry.get_breaker("to-remove")
        assert registry.remove_breaker("to-remove") is True
        assert "to-remove" not in registry.list_breakers()
        assert registry.remove_breaker("nonexistent") is False

    def test_get_stats_returns_summary(self):
        registry = BreakerRegistry()
        registry._loaded = True
        b = registry.get_breaker("svc")
        b.record_success()
        stats = registry.get_stats()
        assert "summary" in stats
        assert "breakers" in stats
        assert stats["summary"]["total_breakers"] == 1
        assert stats["summary"]["total_successes"] == 1

    def test_save_and_load_state_persists(self, tmp_path):
        registry = BreakerRegistry()
        registry._loaded = True
        b = registry.get_breaker("persistent-svc", failure_threshold=7)
        b.record_success()
        registry.save_state()

        # Create a new registry and load from same files
        registry2 = BreakerRegistry()
        registry2.load_state()
        assert "persistent-svc" in registry2.list_breakers()
        b2 = registry2.get_breaker("persistent-svc")
        assert b2.total_successes == 1

    def test_get_policy_returns_retry_policy(self):
        registry = BreakerRegistry()
        registry._loaded = True
        policy = registry.get_policy("api", max_retries=5, base_delay=2.0)
        assert isinstance(policy, RetryPolicy)
        assert policy.max_retries == 5
        # Calling again returns same instance
        policy2 = registry.get_policy("api")
        assert policy2 is policy


# ===================================================================
# Singleton
# ===================================================================

class TestSingleton:
    def test_get_breaker_registry_returns_same_instance(self):
        r1 = get_breaker_registry()
        r2 = get_breaker_registry()
        assert r1 is r2

    def test_get_breaker_shortcut(self):
        b = get_breaker("shortcut-test")
        assert b.name == "shortcut-test"
        b2 = get_breaker("shortcut-test")
        assert b2 is b
