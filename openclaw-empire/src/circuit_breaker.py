"""
Circuit Breaker — Reliability Foundation for OpenClaw Empire

Fault-tolerance primitives for Nick Creighton's 16-site WordPress publishing
empire.  Provides circuit breakers, retry policies with exponential backoff,
structured error classification, and a global registry that persists state
across process restarts.

Patterns:
    - Circuit Breaker: prevents cascading failures by short-circuiting calls
      to services that are known to be down.
    - Retry with Backoff: retries transient failures with exponential delay
      and optional jitter.
    - Error Classification: maps raw exceptions to structured ErrorContext
      objects with recovery hints.

All state persisted to: data/circuit_breaker/

Usage:
    from src.circuit_breaker import get_breaker_registry, get_breaker, with_retry

    # Get a circuit breaker for a service
    breaker = get_breaker("wordpress-api")
    if breaker.can_execute():
        try:
            result = call_wordpress()
            breaker.record_success()
        except Exception as exc:
            breaker.record_failure(classify_error(exc, "wp", "publish"))

    # Retry policy with circuit breaker integration
    registry = get_breaker_registry()
    policy = registry.get_policy("anthropic", max_retries=5, base_delay=2.0)
    result = await policy.execute(call_anthropic, prompt="Hello")

    # Decorator for automatic retry + circuit breaker
    @with_retry(service_name="wordpress", max_retries=3)
    async def publish_post(site_id: str, content: str) -> dict:
        return await wp_client.create_post(site_id, content)

CLI:
    python -m src.circuit_breaker status          # Show all breaker states
    python -m src.circuit_breaker reset           # Reset all breakers
    python -m src.circuit_breaker reset --name X  # Reset a specific breaker
    python -m src.circuit_breaker history          # Show recent error history
    python -m src.circuit_breaker stats            # Show aggregate statistics
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import json
import logging
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("circuit_breaker")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
BREAKER_DATA_DIR = BASE_DIR / "data" / "circuit_breaker"
BREAKERS_FILE = BREAKER_DATA_DIR / "breakers.json"
HISTORY_FILE = BREAKER_DATA_DIR / "history.json"
STATS_FILE = BREAKER_DATA_DIR / "stats.json"

# Ensure data directory exists on import
BREAKER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum error history entries to keep on disk
MAX_HISTORY_ENTRIES = 1000

# Default circuit breaker thresholds
DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_SUCCESS_THRESHOLD = 3
DEFAULT_RECOVERY_TIMEOUT = 60.0  # seconds

# Retry defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_EXPONENTIAL_BASE = 2.0


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    """Return the current time in UTC, timezone-aware."""
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return _now_utc().isoformat()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string back to a timezone-aware datetime."""
    if s is None or s == "":
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


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
            # Windows: os.replace is atomic on same volume
            os.replace(str(tmp), str(path))
        else:
            tmp.replace(path)
    except Exception:
        # Clean up temp file on failure
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Async / sync bridge
# ---------------------------------------------------------------------------

def _run_sync(coro):
    """Run an async coroutine in a sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ===================================================================
# ENUMS
# ===================================================================

class ErrorCode(str, Enum):
    """Structured error codes for the OpenClaw Empire.

    Ranges:
        E1xxx — Network errors
        E2xxx — Authentication errors
        E3xxx — WordPress errors
        E4xxx — Anthropic API errors
        E5xxx — Phone/Android errors
        E6xxx — Platform errors
        E9xxx — Internal errors
    """

    # Network errors E1xxx
    E1001 = "NETWORK_TIMEOUT"
    E1002 = "NETWORK_UNREACHABLE"
    E1003 = "DNS_RESOLUTION_FAILED"
    E1004 = "CONNECTION_REFUSED"
    E1005 = "SSL_ERROR"

    # Auth errors E2xxx
    E2001 = "AUTH_EXPIRED"
    E2002 = "AUTH_INVALID"
    E2003 = "AUTH_INSUFFICIENT_SCOPE"

    # WordPress errors E3xxx
    E3001 = "WP_API_ERROR"
    E3002 = "WP_POST_NOT_FOUND"
    E3003 = "WP_MEDIA_UPLOAD_FAILED"
    E3004 = "WP_RATE_LIMITED"

    # Anthropic errors E4xxx
    E4001 = "ANTHROPIC_RATE_LIMITED"
    E4002 = "ANTHROPIC_OVERLOADED"
    E4003 = "ANTHROPIC_CONTEXT_TOO_LONG"
    E4004 = "ANTHROPIC_INVALID_REQUEST"

    # Phone errors E5xxx
    E5001 = "PHONE_DISCONNECTED"
    E5002 = "PHONE_SCREEN_OFF"
    E5003 = "PHONE_APP_CRASH"
    E5004 = "PHONE_LOW_BATTERY"

    # Platform errors E6xxx
    E6001 = "PLATFORM_BANNED"
    E6002 = "PLATFORM_CAPTCHA"
    E6003 = "PLATFORM_RATE_LIMITED"
    E6004 = "PLATFORM_MAINTENANCE"

    # Internal errors E9xxx
    E9001 = "INTERNAL_ERROR"
    E9002 = "CONFIG_MISSING"
    E9003 = "DATA_CORRUPTION"
    E9004 = "DEPENDENCY_UNAVAILABLE"


class CircuitState(str, Enum):
    """State machine for circuit breakers."""
    CLOSED = "closed"         # Normal operation — calls pass through
    OPEN = "open"             # Failing — calls are rejected immediately
    HALF_OPEN = "half_open"   # Testing recovery — limited calls allowed


# ---------------------------------------------------------------------------
# Sets of retryable error codes
# ---------------------------------------------------------------------------

NETWORK_RETRYABLE: Set[ErrorCode] = {
    ErrorCode.E1001,  # NETWORK_TIMEOUT
    ErrorCode.E1002,  # NETWORK_UNREACHABLE
    ErrorCode.E1003,  # DNS_RESOLUTION_FAILED
    ErrorCode.E1004,  # CONNECTION_REFUSED
}

RATE_LIMIT_RETRYABLE: Set[ErrorCode] = {
    ErrorCode.E3004,  # WP_RATE_LIMITED
    ErrorCode.E4001,  # ANTHROPIC_RATE_LIMITED
    ErrorCode.E4002,  # ANTHROPIC_OVERLOADED
    ErrorCode.E6003,  # PLATFORM_RATE_LIMITED
}

TRANSIENT_RETRYABLE: Set[ErrorCode] = {
    ErrorCode.E5002,  # PHONE_SCREEN_OFF
    ErrorCode.E6004,  # PLATFORM_MAINTENANCE
    ErrorCode.E9004,  # DEPENDENCY_UNAVAILABLE
}

DEFAULT_RETRYABLE_CODES: Set[ErrorCode] = (
    NETWORK_RETRYABLE | RATE_LIMIT_RETRYABLE | TRANSIENT_RETRYABLE
)

# Codes that should never be retried — terminal failures
NON_RETRYABLE_CODES: Set[ErrorCode] = {
    ErrorCode.E2002,  # AUTH_INVALID
    ErrorCode.E2003,  # AUTH_INSUFFICIENT_SCOPE
    ErrorCode.E4003,  # ANTHROPIC_CONTEXT_TOO_LONG
    ErrorCode.E4004,  # ANTHROPIC_INVALID_REQUEST
    ErrorCode.E6001,  # PLATFORM_BANNED
    ErrorCode.E9002,  # CONFIG_MISSING
    ErrorCode.E9003,  # DATA_CORRUPTION
}

# ---------------------------------------------------------------------------
# Recovery hints per error code
# ---------------------------------------------------------------------------

RECOVERY_HINTS: Dict[ErrorCode, List[str]] = {
    ErrorCode.E1001: ["Check network connectivity", "Increase timeout value", "Verify target host is up"],
    ErrorCode.E1002: ["Check internet connection", "Verify VPN status", "Check firewall rules"],
    ErrorCode.E1003: ["Verify domain name", "Check DNS configuration", "Try alternate DNS server"],
    ErrorCode.E1004: ["Check if service is running", "Verify port number", "Check firewall rules"],
    ErrorCode.E1005: ["Update SSL certificates", "Check system clock", "Verify TLS version compatibility"],
    ErrorCode.E2001: ["Refresh authentication token", "Re-authenticate with the service"],
    ErrorCode.E2002: ["Check credentials in .env", "Regenerate API key or app password"],
    ErrorCode.E2003: ["Request additional permissions", "Check API scope configuration"],
    ErrorCode.E3001: ["Check WordPress REST API endpoint", "Verify site is accessible"],
    ErrorCode.E3002: ["Verify post ID exists", "Check if post was deleted or trashed"],
    ErrorCode.E3003: ["Check file size limits", "Verify media upload permissions", "Check disk space"],
    ErrorCode.E3004: ["Wait and retry", "Reduce request frequency", "Check rate limit headers"],
    ErrorCode.E4001: ["Wait for rate limit reset", "Reduce request frequency", "Use batch API"],
    ErrorCode.E4002: ["Wait and retry in 30-60 seconds", "Switch to fallback model"],
    ErrorCode.E4003: ["Reduce input length", "Summarize context", "Split into multiple requests"],
    ErrorCode.E4004: ["Check request format", "Validate parameters", "Review API documentation"],
    ErrorCode.E5001: ["Check USB/WiFi connection", "Restart Termux:API", "Verify WebSocket endpoint"],
    ErrorCode.E5002: ["Wake screen via ADB", "Check auto-lock settings", "Send wake command"],
    ErrorCode.E5003: ["Restart the app", "Clear app cache", "Check for updates"],
    ErrorCode.E5004: ["Connect charger", "Reduce task frequency", "Pause non-critical tasks"],
    ErrorCode.E6001: ["Switch account or proxy", "Wait cooldown period", "Review platform ToS"],
    ErrorCode.E6002: ["Solve CAPTCHA manually", "Use CAPTCHA service", "Reduce automation rate"],
    ErrorCode.E6003: ["Wait and retry", "Reduce posting frequency", "Spread across accounts"],
    ErrorCode.E6004: ["Wait for maintenance to end", "Check platform status page"],
    ErrorCode.E9001: ["Check logs for stack trace", "Restart the service"],
    ErrorCode.E9002: ["Create missing config file", "Check config path in .env"],
    ErrorCode.E9003: ["Restore from backup", "Delete corrupt data file and re-initialize"],
    ErrorCode.E9004: ["Start the dependency service", "Check service health endpoint"],
}


# ===================================================================
# ERROR CONTEXT
# ===================================================================

@dataclass
class ErrorContext:
    """Structured error report with classification, recovery hints, and metadata.

    Every failure in the empire should be captured as an ErrorContext so that
    the circuit breaker and retry systems can make informed decisions.
    """

    code: ErrorCode
    message: str
    module: str
    operation: str
    timestamp: str = ""
    recovery_hints: List[str] = field(default_factory=list)
    retryable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _now_iso()
        if not self.recovery_hints:
            self.recovery_hints = list(RECOVERY_HINTS.get(self.code, []))
        if self.code in NON_RETRYABLE_CODES:
            self.retryable = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        d = asdict(self)
        d["code"] = self.code.name
        d["code_value"] = self.code.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ErrorContext:
        """Deserialize from a dictionary."""
        code_name = d.get("code", "E9001")
        try:
            code = ErrorCode[code_name]
        except KeyError:
            code = ErrorCode.E9001
        return cls(
            code=code,
            message=d.get("message", ""),
            module=d.get("module", "unknown"),
            operation=d.get("operation", "unknown"),
            timestamp=d.get("timestamp", ""),
            recovery_hints=d.get("recovery_hints", []),
            retryable=d.get("retryable", True),
            metadata=d.get("metadata", {}),
        )

    def __str__(self) -> str:
        return f"[{self.code.name}] {self.code.value}: {self.message} (module={self.module}, op={self.operation})"


# ===================================================================
# ERROR CLASSIFIER
# ===================================================================

def classify_error(
    exception: Exception,
    module: str = "unknown",
    operation: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
) -> ErrorContext:
    """Map a raw exception to a structured ErrorContext.

    Inspects the exception type and message to determine the most appropriate
    ErrorCode.  Falls back to E9001 (INTERNAL_ERROR) for unrecognized errors.
    """
    exc_type = type(exception).__name__
    exc_msg = str(exception).lower()
    meta = metadata or {}
    meta["exception_type"] = exc_type
    meta["exception_message"] = str(exception)

    # --- Network errors ---
    if exc_type in ("TimeoutError", "asyncio.TimeoutError", "ConnectTimeoutError", "ReadTimeoutError"):
        code = ErrorCode.E1001
    elif "timeout" in exc_msg:
        code = ErrorCode.E1001
    elif exc_type == "ConnectionRefusedError" or "connection refused" in exc_msg:
        code = ErrorCode.E1004
    elif "unreachable" in exc_msg or "network is unreachable" in exc_msg:
        code = ErrorCode.E1002
    elif "name or service not known" in exc_msg or "getaddrinfo" in exc_msg or "dns" in exc_msg:
        code = ErrorCode.E1003
    elif "ssl" in exc_msg or "certificate" in exc_msg or exc_type == "SSLError":
        code = ErrorCode.E1005
    elif exc_type in ("ConnectionError", "ConnectionResetError", "BrokenPipeError"):
        code = ErrorCode.E1002

    # --- Auth errors ---
    elif "401" in exc_msg or "unauthorized" in exc_msg:
        code = ErrorCode.E2001
    elif "403" in exc_msg or "forbidden" in exc_msg:
        code = ErrorCode.E2003
    elif "invalid api key" in exc_msg or "invalid_api_key" in exc_msg:
        code = ErrorCode.E2002

    # --- WordPress errors ---
    elif "404" in exc_msg and ("post" in exc_msg or "wp" in module.lower()):
        code = ErrorCode.E3002
    elif "media" in exc_msg and ("upload" in exc_msg or "failed" in exc_msg):
        code = ErrorCode.E3003
    elif "429" in exc_msg and "wp" in module.lower():
        code = ErrorCode.E3004

    # --- Anthropic errors ---
    elif "rate_limit" in exc_msg or ("429" in exc_msg and "anthropic" in module.lower()):
        code = ErrorCode.E4001
    elif "overloaded" in exc_msg or "529" in exc_msg:
        code = ErrorCode.E4002
    elif "context" in exc_msg and "too long" in exc_msg:
        code = ErrorCode.E4003
    elif "invalid_request" in exc_msg and "anthropic" in module.lower():
        code = ErrorCode.E4004

    # --- Platform errors ---
    elif "banned" in exc_msg or "suspended" in exc_msg:
        code = ErrorCode.E6001
    elif "captcha" in exc_msg:
        code = ErrorCode.E6002
    elif "429" in exc_msg or "rate limit" in exc_msg:
        code = ErrorCode.E6003
    elif "maintenance" in exc_msg or "503" in exc_msg:
        code = ErrorCode.E6004

    # --- Internal fallback ---
    elif exc_type == "FileNotFoundError" or "config" in exc_msg:
        code = ErrorCode.E9002
    elif "corrupt" in exc_msg or "decode" in exc_msg:
        code = ErrorCode.E9003
    else:
        code = ErrorCode.E9001

    return ErrorContext(
        code=code,
        message=str(exception),
        module=module,
        operation=operation,
        metadata=meta,
    )


# ===================================================================
# CIRCUIT BREAKER
# ===================================================================

@dataclass
class CircuitBreaker:
    """Circuit breaker for a named service endpoint.

    State machine:
        CLOSED  -- failures exceed threshold --> OPEN
        OPEN    -- recovery_timeout elapsed  --> HALF_OPEN
        HALF_OPEN -- success_threshold met   --> CLOSED
        HALF_OPEN -- any failure             --> OPEN
    """

    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    success_threshold: int = DEFAULT_SUCCESS_THRESHOLD
    recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change: Optional[float] = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    total_rejected: int = 0
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)

    # Maximum recent errors to keep per breaker
    MAX_RECENT_ERRORS: int = 20

    def can_execute(self) -> bool:
        """Check whether a call is allowed through this breaker.

        Returns True if the call should proceed, False if it should be
        rejected (circuit is OPEN and recovery timeout has not elapsed).
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open to test recovery
            return True

        # State is OPEN — check if recovery timeout has elapsed
        if self.state == CircuitState.OPEN:
            self._check_recovery()
            return self.state == CircuitState.HALF_OPEN

        return False

    def record_success(self) -> None:
        """Record a successful call. May transition HALF_OPEN -> CLOSED."""
        self.total_calls += 1
        self.total_successes += 1
        self.last_success_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition(CircuitState.CLOSED)
                logger.info(
                    "Circuit breaker '%s' CLOSED after %d consecutive successes",
                    self.name, self.success_count,
                )
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success in closed state
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self, error: Optional[ErrorContext] = None) -> None:
        """Record a failed call. May transition CLOSED -> OPEN or HALF_OPEN -> OPEN."""
        self.total_calls += 1
        self.total_failures += 1
        self.last_failure_time = time.time()

        # Store error in recent history
        if error is not None:
            err_dict = error.to_dict()
            self.recent_errors.append(err_dict)
            if len(self.recent_errors) > self.MAX_RECENT_ERRORS:
                self.recent_errors = self.recent_errors[-self.MAX_RECENT_ERRORS:]

        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately re-opens the circuit
            self._transition(CircuitState.OPEN)
            self.success_count = 0
            logger.warning(
                "Circuit breaker '%s' re-OPENED from HALF_OPEN on failure: %s",
                self.name, error or "unknown",
            )

        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self._transition(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker '%s' OPENED after %d failures (threshold=%d)",
                    self.name, self.failure_count, self.failure_threshold,
                )

    def reset(self) -> None:
        """Force-reset the breaker to CLOSED state, clearing all counters."""
        self._transition(CircuitState.CLOSED)
        self.failure_count = 0
        self.success_count = 0
        self.recent_errors.clear()
        logger.info("Circuit breaker '%s' manually RESET to CLOSED", self.name)

    def get_stats(self) -> Dict[str, Any]:
        """Return a dictionary of metrics for this breaker."""
        now = time.time()
        time_in_state = None
        if self.last_state_change is not None:
            time_in_state = round(now - self.last_state_change, 1)

        time_since_failure = None
        if self.last_failure_time is not None:
            time_since_failure = round(now - self.last_failure_time, 1)

        time_since_success = None
        if self.last_success_time is not None:
            time_since_success = round(now - self.last_success_time, 1)

        failure_rate = 0.0
        if self.total_calls > 0:
            failure_rate = round(self.total_failures / self.total_calls * 100, 2)

        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "recovery_timeout": self.recovery_timeout,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_rejected": self.total_rejected,
            "failure_rate_pct": failure_rate,
            "time_in_state_seconds": time_in_state,
            "time_since_last_failure_seconds": time_since_failure,
            "time_since_last_success_seconds": time_since_success,
            "recent_error_count": len(self.recent_errors),
        }

    def _check_recovery(self) -> None:
        """Auto-transition OPEN -> HALF_OPEN if recovery timeout has elapsed."""
        if self.state != CircuitState.OPEN:
            return
        if self.last_failure_time is None:
            return

        elapsed = time.time() - self.last_failure_time
        if elapsed >= self.recovery_timeout:
            self._transition(CircuitState.HALF_OPEN)
            self.success_count = 0
            logger.info(
                "Circuit breaker '%s' transitioned to HALF_OPEN after %.1fs recovery timeout",
                self.name, elapsed,
            )

    def _transition(self, new_state: CircuitState) -> None:
        """Perform a state transition with timestamp tracking."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()
        if old_state != new_state:
            logger.debug(
                "Circuit breaker '%s': %s -> %s",
                self.name, old_state.value, new_state.value,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize breaker state for persistence."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "last_state_change": self.last_state_change,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_rejected": self.total_rejected,
            "recent_errors": self.recent_errors,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CircuitBreaker:
        """Deserialize breaker state from a dictionary."""
        state_val = d.get("state", "closed")
        try:
            state = CircuitState(state_val)
        except ValueError:
            state = CircuitState.CLOSED

        return cls(
            name=d.get("name", "unknown"),
            state=state,
            failure_count=d.get("failure_count", 0),
            success_count=d.get("success_count", 0),
            failure_threshold=d.get("failure_threshold", DEFAULT_FAILURE_THRESHOLD),
            success_threshold=d.get("success_threshold", DEFAULT_SUCCESS_THRESHOLD),
            recovery_timeout=d.get("recovery_timeout", DEFAULT_RECOVERY_TIMEOUT),
            last_failure_time=d.get("last_failure_time"),
            last_success_time=d.get("last_success_time"),
            last_state_change=d.get("last_state_change"),
            total_calls=d.get("total_calls", 0),
            total_failures=d.get("total_failures", 0),
            total_successes=d.get("total_successes", 0),
            total_rejected=d.get("total_rejected", 0),
            recent_errors=d.get("recent_errors", []),
        )


# ===================================================================
# CIRCUIT BREAKER EXCEPTION
# ===================================================================

class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit breaker is OPEN."""

    def __init__(self, breaker_name: str, recovery_timeout: float, last_failure_time: Optional[float] = None):
        self.breaker_name = breaker_name
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = last_failure_time
        remaining = 0.0
        if last_failure_time is not None:
            remaining = max(0.0, recovery_timeout - (time.time() - last_failure_time))
        self.remaining_seconds = round(remaining, 1)
        super().__init__(
            f"Circuit breaker '{breaker_name}' is OPEN. "
            f"Retry in {self.remaining_seconds}s (recovery_timeout={recovery_timeout}s)"
        )


# ===================================================================
# RETRY POLICY
# ===================================================================

@dataclass
class RetryPolicy:
    """Configurable retry policy with exponential backoff and circuit breaker integration.

    Wraps an async callable, retrying on transient failures up to *max_retries*
    times with exponential backoff and optional jitter.  When paired with a
    CircuitBreaker, failures and successes are recorded and calls are rejected
    when the circuit is open.
    """

    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE
    jitter: bool = True
    retryable_codes: Set[ErrorCode] = field(default_factory=lambda: set(DEFAULT_RETRYABLE_CODES))
    circuit_breaker: Optional[CircuitBreaker] = None
    module_name: str = "unknown"

    async def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *func* with retry logic and circuit breaker integration.

        Args:
            func: An async callable to execute.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The return value of *func* on success.

        Raises:
            CircuitOpenError: If the circuit breaker is open.
            Exception: The last exception if all retries are exhausted.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            # Check circuit breaker before each attempt
            if self.circuit_breaker is not None:
                if not self.circuit_breaker.can_execute():
                    self.circuit_breaker.total_rejected += 1
                    raise CircuitOpenError(
                        breaker_name=self.circuit_breaker.name,
                        recovery_timeout=self.circuit_breaker.recovery_timeout,
                        last_failure_time=self.circuit_breaker.last_failure_time,
                    )

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success — record and return
                if self.circuit_breaker is not None:
                    self.circuit_breaker.record_success()

                if attempt > 0:
                    logger.info(
                        "Succeeded on attempt %d/%d for %s",
                        attempt + 1, self.max_retries + 1, self.module_name,
                    )

                return result

            except Exception as exc:
                last_exception = exc
                error_ctx = classify_error(exc, module=self.module_name, operation="retry_execute")

                # Record failure on circuit breaker
                if self.circuit_breaker is not None:
                    self.circuit_breaker.record_failure(error_ctx)

                # Check if we should retry
                if not self._should_retry(error_ctx, attempt):
                    logger.warning(
                        "Not retrying %s after attempt %d: %s (code=%s, retryable=%s)",
                        self.module_name, attempt + 1, exc, error_ctx.code.value, error_ctx.retryable,
                    )
                    raise

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.info(
                    "Retry %d/%d for %s in %.1fs: %s [%s]",
                    attempt + 1, self.max_retries, self.module_name,
                    delay, exc, error_ctx.code.value,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        if last_exception is not None:
            raise last_exception

    def execute_sync(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Synchronous wrapper for execute()."""
        return _run_sync(self.execute(func, *args, **kwargs))

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt using exponential backoff with optional jitter.

        delay = min(base_delay * exponential_base^attempt, max_delay)
        With jitter: delay * random(0.5, 1.5)
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            jitter_factor = 0.5 + random.random()  # 0.5 to 1.5
            delay *= jitter_factor

        return round(delay, 3)

    def _should_retry(self, error: ErrorContext, attempt: int) -> bool:
        """Determine whether the given error on the given attempt should be retried.

        Returns False when:
            - The attempt count has reached max_retries
            - The error is not in the retryable_codes set
            - The error is explicitly marked as non-retryable
        """
        if attempt >= self.max_retries:
            return False

        if not error.retryable:
            return False

        if error.code not in self.retryable_codes:
            return False

        return True


# ===================================================================
# DECORATOR
# ===================================================================

def with_retry(
    service_name: str = "default",
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retryable_codes: Optional[Set[ErrorCode]] = None,
    use_circuit_breaker: bool = True,
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT,
) -> Callable:
    """Decorator that wraps async functions with retry logic and circuit breaker protection.

    Usage:
        @with_retry(service_name="wordpress", max_retries=3)
        async def publish_post(site_id: str, content: str) -> dict:
            return await wp_client.create_post(site_id, content)

    Args:
        service_name: Name for the circuit breaker (used as registry key).
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        retryable_codes: Set of ErrorCodes that should trigger retries.
            Defaults to DEFAULT_RETRYABLE_CODES.
        use_circuit_breaker: Whether to attach a circuit breaker.
        failure_threshold: Failures before the breaker opens.
        recovery_timeout: Seconds before the breaker transitions OPEN -> HALF_OPEN.

    Returns:
        A decorator that wraps the target function.
    """
    codes = retryable_codes if retryable_codes is not None else set(DEFAULT_RETRYABLE_CODES)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            registry = get_breaker_registry()

            breaker = None
            if use_circuit_breaker:
                breaker = registry.get_breaker(
                    service_name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )

            policy = RetryPolicy(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retryable_codes=codes,
                circuit_breaker=breaker,
                module_name=service_name,
            )

            try:
                return await policy.execute(func, *args, **kwargs)
            except Exception as exc:
                # Record error in global registry history
                error_ctx = classify_error(exc, module=service_name, operation=func.__name__)
                registry.record_error(error_ctx)
                raise

        # Also provide a sync version via attribute
        @functools.wraps(func)
        def wrapper_sync(*args: Any, **kwargs: Any) -> Any:
            return _run_sync(wrapper(*args, **kwargs))

        wrapper.sync = wrapper_sync  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ===================================================================
# BREAKER REGISTRY
# ===================================================================

class BreakerRegistry:
    """Global registry for circuit breakers, retry policies, and error history.

    Provides centralized management of all reliability primitives across the
    empire.  State is persisted to JSON for cross-restart continuity.
    """

    def __init__(self) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._policies: Dict[str, RetryPolicy] = {}
        self._error_history: List[Dict[str, Any]] = []
        self._max_history: int = MAX_HISTORY_ENTRIES
        self._dirty: bool = False
        self._loaded: bool = False

    def get_breaker(
        self,
        name: str,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        success_threshold: int = DEFAULT_SUCCESS_THRESHOLD,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT,
    ) -> CircuitBreaker:
        """Get or create a named circuit breaker.

        If the breaker already exists, it is returned as-is (thresholds are
        not updated to avoid mid-session config drift).  New breakers are
        created with the provided thresholds.
        """
        if not self._loaded:
            self.load_state()

        if name in self._breakers:
            return self._breakers[name]

        breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=recovery_timeout,
        )
        self._breakers[name] = breaker
        self._dirty = True
        logger.debug("Created circuit breaker '%s' (threshold=%d, timeout=%.0fs)",
                      name, failure_threshold, recovery_timeout)
        return breaker

    def get_policy(
        self,
        name: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        retryable_codes: Optional[Set[ErrorCode]] = None,
        use_circuit_breaker: bool = True,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT,
    ) -> RetryPolicy:
        """Get or create a named retry policy with optional circuit breaker.

        Like get_breaker(), existing policies are returned as-is.
        """
        if name in self._policies:
            return self._policies[name]

        breaker = None
        if use_circuit_breaker:
            breaker = self.get_breaker(
                name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )

        codes = retryable_codes if retryable_codes is not None else set(DEFAULT_RETRYABLE_CODES)

        policy = RetryPolicy(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            retryable_codes=codes,
            circuit_breaker=breaker,
            module_name=name,
        )
        self._policies[name] = policy
        logger.debug("Created retry policy '%s' (retries=%d, delay=%.1fs)", name, max_retries, base_delay)
        return policy

    def record_error(self, ctx: ErrorContext) -> None:
        """Store an error in the global history and trigger auto-save when dirty.

        The history is bounded at MAX_HISTORY_ENTRIES to prevent unbounded growth.
        """
        entry = ctx.to_dict()
        self._error_history.append(entry)
        if len(self._error_history) > self._max_history:
            self._error_history = self._error_history[-self._max_history:]

        self._dirty = True
        logger.debug("Recorded error: %s", ctx)

        # Auto-save periodically
        if len(self._error_history) % 10 == 0:
            self.save_state()

    def get_error_history(
        self,
        limit: int = 50,
        code: Optional[str] = None,
        module: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent error history, optionally filtered by code or module.

        Args:
            limit: Maximum number of entries to return.
            code: Filter by ErrorCode name (e.g., "E1001").
            module: Filter by module name.

        Returns:
            List of error context dictionaries, newest first.
        """
        results = list(reversed(self._error_history))

        if code is not None:
            results = [e for e in results if e.get("code") == code]

        if module is not None:
            mod_lower = module.lower()
            results = [e for e in results if mod_lower in e.get("module", "").lower()]

        return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all breakers and the error history."""
        breaker_stats = {}
        for name, breaker in self._breakers.items():
            breaker_stats[name] = breaker.get_stats()

        # Error code distribution
        code_counts: Dict[str, int] = {}
        module_counts: Dict[str, int] = {}
        for entry in self._error_history:
            code_name = entry.get("code", "unknown")
            code_counts[code_name] = code_counts.get(code_name, 0) + 1
            mod_name = entry.get("module", "unknown")
            module_counts[mod_name] = module_counts.get(mod_name, 0) + 1

        total_calls = sum(b.total_calls for b in self._breakers.values())
        total_failures = sum(b.total_failures for b in self._breakers.values())
        total_successes = sum(b.total_successes for b in self._breakers.values())
        total_rejected = sum(b.total_rejected for b in self._breakers.values())

        open_breakers = [n for n, b in self._breakers.items() if b.state == CircuitState.OPEN]
        half_open_breakers = [n for n, b in self._breakers.items() if b.state == CircuitState.HALF_OPEN]

        return {
            "summary": {
                "total_breakers": len(self._breakers),
                "breakers_closed": len(self._breakers) - len(open_breakers) - len(half_open_breakers),
                "breakers_open": len(open_breakers),
                "breakers_half_open": len(half_open_breakers),
                "open_breaker_names": open_breakers,
                "half_open_breaker_names": half_open_breakers,
                "total_calls": total_calls,
                "total_failures": total_failures,
                "total_successes": total_successes,
                "total_rejected": total_rejected,
                "overall_failure_rate_pct": round(total_failures / max(total_calls, 1) * 100, 2),
                "error_history_size": len(self._error_history),
                "policies_count": len(self._policies),
            },
            "breakers": breaker_stats,
            "error_distribution": {
                "by_code": dict(sorted(code_counts.items(), key=lambda x: x[1], reverse=True)),
                "by_module": dict(sorted(module_counts.items(), key=lambda x: x[1], reverse=True)),
            },
        }

    def save_state(self) -> None:
        """Persist all breaker states and error history to JSON files."""
        # Save breaker states
        breakers_data = {}
        for name, breaker in self._breakers.items():
            breakers_data[name] = breaker.to_dict()
        _save_json(BREAKERS_FILE, breakers_data)

        # Save error history
        _save_json(HISTORY_FILE, self._error_history)

        # Save stats snapshot
        _save_json(STATS_FILE, self.get_stats())

        self._dirty = False
        logger.debug("Saved circuit breaker state (%d breakers, %d errors)",
                      len(self._breakers), len(self._error_history))

    def load_state(self) -> None:
        """Restore breaker states and error history from JSON files."""
        self._loaded = True

        # Load breaker states
        breakers_data = _load_json(BREAKERS_FILE, {})
        for name, bdata in breakers_data.items():
            if name not in self._breakers:
                try:
                    self._breakers[name] = CircuitBreaker.from_dict(bdata)
                except Exception as exc:
                    logger.warning("Failed to load breaker '%s': %s", name, exc)

        # Load error history
        history_data = _load_json(HISTORY_FILE, [])
        if isinstance(history_data, list):
            self._error_history = history_data[-self._max_history:]

        logger.debug("Loaded circuit breaker state (%d breakers, %d errors)",
                      len(self._breakers), len(self._error_history))

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED and clear error history."""
        for breaker in self._breakers.values():
            breaker.reset()
        self._error_history.clear()
        self._dirty = True
        self.save_state()
        logger.info("Reset all %d circuit breakers and cleared error history", len(self._breakers))

    def reset_breaker(self, name: str) -> bool:
        """Reset a specific breaker by name. Returns True if found and reset."""
        if name in self._breakers:
            self._breakers[name].reset()
            self._dirty = True
            self.save_state()
            return True
        return False

    def remove_breaker(self, name: str) -> bool:
        """Remove a breaker entirely from the registry."""
        if name in self._breakers:
            del self._breakers[name]
            if name in self._policies:
                del self._policies[name]
            self._dirty = True
            self.save_state()
            logger.info("Removed circuit breaker '%s'", name)
            return True
        return False

    def list_breakers(self) -> List[str]:
        """Return sorted list of breaker names."""
        if not self._loaded:
            self.load_state()
        return sorted(self._breakers.keys())


# ===================================================================
# SINGLETON
# ===================================================================

_registry: Optional[BreakerRegistry] = None


def get_breaker_registry() -> BreakerRegistry:
    """Return the global BreakerRegistry singleton, creating it on first call."""
    global _registry
    if _registry is None:
        _registry = BreakerRegistry()
        _registry.load_state()
    return _registry


def get_breaker(name: str, **kwargs: Any) -> CircuitBreaker:
    """Shortcut to get a named circuit breaker from the global registry."""
    return get_breaker_registry().get_breaker(name, **kwargs)


# ===================================================================
# CLI COMMAND HANDLERS
# ===================================================================

def _cmd_status(args: argparse.Namespace) -> None:
    """Show the status of all circuit breakers."""
    registry = get_breaker_registry()
    breakers = registry.list_breakers()

    if not breakers:
        print("No circuit breakers registered.")
        return

    # Header
    print(f"\n{'Name':<30} {'State':<12} {'Failures':<10} {'Calls':<10} {'Fail%':<8} {'Rejected':<10}")
    print("-" * 80)

    for name in breakers:
        breaker = registry.get_breaker(name)
        stats = breaker.get_stats()

        # Color indicators for state
        state_str = stats["state"].upper()
        if stats["state"] == "open":
            state_str = f"[!] {state_str}"
        elif stats["state"] == "half_open":
            state_str = f"[~] {state_str}"
        else:
            state_str = f"[+] {state_str}"

        print(
            f"{name:<30} {state_str:<12} "
            f"{stats['failure_count']}/{stats['failure_threshold']:<7} "
            f"{stats['total_calls']:<10} "
            f"{stats['failure_rate_pct']:<8} "
            f"{stats['total_rejected']:<10}"
        )

    # Summary
    summary = registry.get_stats()["summary"]
    print(f"\n  Total: {summary['total_breakers']} breakers | "
          f"{summary['breakers_open']} open | "
          f"{summary['breakers_half_open']} half-open | "
          f"{summary['total_calls']} calls | "
          f"{summary['overall_failure_rate_pct']}% failure rate")

    if summary["open_breaker_names"]:
        print(f"  OPEN breakers: {', '.join(summary['open_breaker_names'])}")

    print()


def _cmd_reset(args: argparse.Namespace) -> None:
    """Reset circuit breakers."""
    registry = get_breaker_registry()

    if args.name:
        if registry.reset_breaker(args.name):
            print(f"Reset circuit breaker '{args.name}' to CLOSED.")
        else:
            print(f"Circuit breaker '{args.name}' not found.")
            print(f"Available breakers: {', '.join(registry.list_breakers())}")
    elif args.all:
        count = len(registry.list_breakers())
        registry.reset_all()
        print(f"Reset all {count} circuit breakers to CLOSED and cleared error history.")
    else:
        print("Specify --name <breaker> to reset a specific breaker, or --all to reset everything.")


def _cmd_history(args: argparse.Namespace) -> None:
    """Show recent error history."""
    registry = get_breaker_registry()
    errors = registry.get_error_history(
        limit=args.limit,
        code=args.code,
        module=args.module,
    )

    if not errors:
        print("No errors in history.")
        return

    print(f"\nRecent Errors ({len(errors)} shown):\n")

    for i, entry in enumerate(errors, 1):
        code = entry.get("code", "?")
        code_value = entry.get("code_value", "?")
        message = entry.get("message", "")
        module = entry.get("module", "?")
        operation = entry.get("operation", "?")
        timestamp = entry.get("timestamp", "?")
        retryable = entry.get("retryable", True)

        # Truncate message for display
        if len(message) > 100:
            message = message[:97] + "..."

        retry_marker = "R" if retryable else "X"
        print(f"  {i:3}. [{code}] {code_value}")
        print(f"       {message}")
        print(f"       module={module} op={operation} [{retry_marker}] {timestamp}")

        # Show recovery hints if verbose
        if args.verbose:
            hints = entry.get("recovery_hints", [])
            if hints:
                for hint in hints:
                    print(f"         -> {hint}")
        print()


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show aggregate statistics."""
    registry = get_breaker_registry()
    stats = registry.get_stats()
    summary = stats["summary"]

    print("\n=== Circuit Breaker Statistics ===\n")
    print(f"  Breakers:     {summary['total_breakers']} total "
          f"({summary['breakers_closed']} closed, "
          f"{summary['breakers_open']} open, "
          f"{summary['breakers_half_open']} half-open)")
    print(f"  Policies:     {summary['policies_count']}")
    print(f"  Total Calls:  {summary['total_calls']}")
    print(f"  Successes:    {summary['total_successes']}")
    print(f"  Failures:     {summary['total_failures']} ({summary['overall_failure_rate_pct']}%)")
    print(f"  Rejected:     {summary['total_rejected']}")
    print(f"  Error Log:    {summary['error_history_size']} entries")

    # Error distribution
    by_code = stats["error_distribution"]["by_code"]
    if by_code:
        print("\n  Error Distribution (by code):")
        for code, count in list(by_code.items())[:15]:
            bar = "#" * min(count, 40)
            print(f"    {code:<8} {count:>5}  {bar}")

    by_module = stats["error_distribution"]["by_module"]
    if by_module:
        print("\n  Error Distribution (by module):")
        for mod, count in list(by_module.items())[:10]:
            bar = "#" * min(count, 40)
            print(f"    {mod:<25} {count:>5}  {bar}")

    # Per-breaker details
    if stats["breakers"] and args.detail:
        print("\n  --- Per-Breaker Detail ---")
        for name, bstats in sorted(stats["breakers"].items()):
            print(f"\n  [{bstats['state'].upper()}] {name}")
            print(f"    Failures: {bstats['failure_count']}/{bstats['failure_threshold']} "
                  f"| Successes: {bstats['success_count']}/{bstats['success_threshold']}")
            print(f"    Total: {bstats['total_calls']} calls, "
                  f"{bstats['total_failures']} failures ({bstats['failure_rate_pct']}%), "
                  f"{bstats['total_rejected']} rejected")
            if bstats["time_in_state_seconds"] is not None:
                print(f"    In state for: {bstats['time_in_state_seconds']}s")
            if bstats["time_since_last_failure_seconds"] is not None:
                print(f"    Last failure: {bstats['time_since_last_failure_seconds']}s ago")

    print()


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def main() -> None:
    """CLI entry point for the circuit breaker module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="circuit_breaker",
        description="OpenClaw Empire Circuit Breaker — Reliability Foundation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    sp_status = subparsers.add_parser("status", help="Show all circuit breaker states")
    sp_status.set_defaults(func=_cmd_status)

    # reset
    sp_reset = subparsers.add_parser("reset", help="Reset circuit breakers")
    sp_reset.add_argument("--name", type=str, default=None, help="Reset a specific breaker by name")
    sp_reset.add_argument("--all", action="store_true", help="Reset all breakers and clear history")
    sp_reset.set_defaults(func=_cmd_reset)

    # history
    sp_history = subparsers.add_parser("history", help="Show recent error history")
    sp_history.add_argument("--limit", type=int, default=20, help="Max entries to show (default: 20)")
    sp_history.add_argument("--code", type=str, default=None, help="Filter by ErrorCode name (e.g., E1001)")
    sp_history.add_argument("--module", type=str, default=None, help="Filter by module name")
    sp_history.add_argument("--verbose", action="store_true", help="Show recovery hints")
    sp_history.set_defaults(func=_cmd_history)

    # stats
    sp_stats = subparsers.add_parser("stats", help="Show aggregate statistics")
    sp_stats.add_argument("--detail", action="store_true", help="Show per-breaker detail")
    sp_stats.set_defaults(func=_cmd_stats)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
