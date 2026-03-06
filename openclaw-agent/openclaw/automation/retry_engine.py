"""RetryEngine — smart retry logic with exponential backoff and error categorization.

Categorizes errors from signup attempts into actionable categories (transient, rate
limited, CAPTCHA failed, etc.) and applies the appropriate retry strategy. Supports
both sync and async execution with full history tracking per platform.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ─── Error Categories ────────────────────────────────────────────────────────


class ErrorCategory(str, Enum):
    """Categories of errors for retry strategy selection."""

    TRANSIENT = "transient"               # Network timeout, 502/503 — retry quickly
    RATE_LIMITED = "rate_limited"          # 429 — retry with long backoff
    CAPTCHA_FAILED = "captcha_failed"     # CAPTCHA solve failed — retry with new solve
    CREDENTIAL_ERROR = "credential_error" # Wrong password — don't retry
    PLATFORM_DOWN = "platform_down"       # Site maintenance — retry much later
    ACCOUNT_EXISTS = "account_exists"     # Already registered — skip entirely
    BLOCKED = "blocked"                   # IP/account banned — switch proxy, long wait
    VALIDATION_ERROR = "validation_error" # Form validation error — fix input, maybe retry
    UNKNOWN = "unknown"                   # Unrecognized error


# Pattern → ErrorCategory mapping for auto-classification
_ERROR_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    # Transient / network
    (re.compile(r"(?i)timeout|timed?\s*out"), ErrorCategory.TRANSIENT),
    (re.compile(r"(?i)connection\s*(reset|refused|abort)"), ErrorCategory.TRANSIENT),
    (re.compile(r"(?i)502|503|504|bad\s*gateway|service\s*unavailable"), ErrorCategory.TRANSIENT),
    (re.compile(r"(?i)temporary|temporarily"), ErrorCategory.TRANSIENT),
    (re.compile(r"(?i)network\s*(error|unreachable)"), ErrorCategory.TRANSIENT),
    (re.compile(r"(?i)dns\s*(resolution|lookup)\s*failed"), ErrorCategory.TRANSIENT),
    (re.compile(r"(?i)ssl|tls|certificate"), ErrorCategory.TRANSIENT),
    # Rate limited
    (re.compile(r"(?i)429|rate\s*limit|too\s*many\s*requests"), ErrorCategory.RATE_LIMITED),
    (re.compile(r"(?i)throttl|slow\s*down"), ErrorCategory.RATE_LIMITED),
    # CAPTCHA
    (re.compile(r"(?i)captcha.*fail|captcha.*error|captcha.*timeout"), ErrorCategory.CAPTCHA_FAILED),
    (re.compile(r"(?i)recaptcha|hcaptcha|turnstile.*fail"), ErrorCategory.CAPTCHA_FAILED),
    # Credentials
    (re.compile(r"(?i)invalid\s*(password|credentials|login)"), ErrorCategory.CREDENTIAL_ERROR),
    (re.compile(r"(?i)authentication\s*fail|unauthorized|401"), ErrorCategory.CREDENTIAL_ERROR),
    (re.compile(r"(?i)wrong\s*password"), ErrorCategory.CREDENTIAL_ERROR),
    # Platform down
    (re.compile(r"(?i)maintenance|under\s*construction|coming\s*soon"), ErrorCategory.PLATFORM_DOWN),
    (re.compile(r"(?i)500\s*internal\s*server\s*error"), ErrorCategory.PLATFORM_DOWN),
    # Account exists
    (re.compile(r"(?i)already\s*(registered|exists|taken|in\s*use)"), ErrorCategory.ACCOUNT_EXISTS),
    (re.compile(r"(?i)email.*already|username.*taken"), ErrorCategory.ACCOUNT_EXISTS),
    (re.compile(r"(?i)duplicate.*account"), ErrorCategory.ACCOUNT_EXISTS),
    # Blocked
    (re.compile(r"(?i)blocked|banned|forbidden|403|blacklist"), ErrorCategory.BLOCKED),
    (re.compile(r"(?i)access\s*denied|ip\s*(blocked|banned)"), ErrorCategory.BLOCKED),
    (re.compile(r"(?i)suspicious\s*activity"), ErrorCategory.BLOCKED),
    # Validation
    (re.compile(r"(?i)invalid\s*(email|format|input|field)"), ErrorCategory.VALIDATION_ERROR),
    (re.compile(r"(?i)required\s*field|field\s*required"), ErrorCategory.VALIDATION_ERROR),
]


# ─── Data Classes ────────────────────────────────────────────────────────────


@dataclass
class RetryPolicy:
    """Configures retry behavior — delays, limits, and which errors to retry on."""

    max_retries: int = 3
    base_delay_seconds: float = 5.0
    max_delay_seconds: float = 300.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: set[ErrorCategory] = field(
        default_factory=lambda: {
            ErrorCategory.TRANSIENT,
            ErrorCategory.RATE_LIMITED,
            ErrorCategory.CAPTCHA_FAILED,
            ErrorCategory.PLATFORM_DOWN,
        }
    )
    no_retry_on: set[ErrorCategory] = field(
        default_factory=lambda: {
            ErrorCategory.CREDENTIAL_ERROR,
            ErrorCategory.ACCOUNT_EXISTS,
        }
    )


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""

    attempt_number: int
    error_category: ErrorCategory
    error_message: str
    delay_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)
    succeeded: bool = False


# ─── Retry Engine ────────────────────────────────────────────────────────────


class RetryEngine:
    """Execute async callables with intelligent retry logic.

    Usage::

        engine = RetryEngine()
        result = await engine.execute_with_retry(
            my_signup_function, "gumroad", arg1, arg2, key=val
        )
    """

    def __init__(self, policy: RetryPolicy | None = None):
        self.policy = policy or RetryPolicy()
        self.history: dict[str, list[RetryAttempt]] = defaultdict(list)

    # ─── Error Classification ────────────────────────────────────────────

    def categorize_error(self, error: Exception | str) -> ErrorCategory:
        """Categorize an error for retry strategy selection.

        Matches the error message against known patterns. Falls back to UNKNOWN.
        """
        error_str = str(error)

        for pattern, category in _ERROR_PATTERNS:
            if pattern.search(error_str):
                return category

        return ErrorCategory.UNKNOWN

    # ─── Retry Decisions ─────────────────────────────────────────────────

    def should_retry(
        self, platform_id: str, error_category: ErrorCategory
    ) -> bool:
        """Decide whether to retry based on error category, policy, and history."""
        # Never retry these categories
        if error_category in self.policy.no_retry_on:
            logger.debug(
                f"No retry for {platform_id}: category {error_category.value} "
                f"is in no_retry_on"
            )
            return False

        # Only retry categories explicitly in the retry set (plus UNKNOWN)
        if (
            error_category not in self.policy.retry_on
            and error_category != ErrorCategory.UNKNOWN
        ):
            logger.debug(
                f"No retry for {platform_id}: category {error_category.value} "
                f"not in retry_on set"
            )
            return False

        # Check retry count — the current (just-recorded) failure counts as an
        # attempt, so we allow retrying until failed_attempts > max_retries.
        # max_retries=3 means: 1 initial attempt + 3 retries = 4 total.
        platform_history = self.history.get(platform_id, [])
        failed_attempts = sum(
            1 for a in platform_history if not a.succeeded
        )
        if failed_attempts > self.policy.max_retries:
            logger.debug(
                f"No retry for {platform_id}: max retries reached "
                f"({failed_attempts}/{self.policy.max_retries})"
            )
            return False

        return True

    # ─── Delay Calculation ───────────────────────────────────────────────

    def calculate_delay(
        self, attempt_number: int, error_category: ErrorCategory
    ) -> float:
        """Calculate delay with exponential backoff and optional jitter.

        Certain error categories get longer base delays:
        - RATE_LIMITED: 3x base delay
        - PLATFORM_DOWN: 5x base delay
        - BLOCKED: 4x base delay
        """
        base = self.policy.base_delay_seconds

        # Category-specific multipliers
        category_multipliers = {
            ErrorCategory.RATE_LIMITED: 3.0,
            ErrorCategory.PLATFORM_DOWN: 5.0,
            ErrorCategory.BLOCKED: 4.0,
            ErrorCategory.CAPTCHA_FAILED: 1.5,
        }
        multiplier = category_multipliers.get(error_category, 1.0)

        # Exponential backoff: base * multiplier * exponential_base^attempt
        delay = base * multiplier * (self.policy.exponential_base ** attempt_number)

        # Cap at max delay
        delay = min(delay, self.policy.max_delay_seconds)

        # Apply jitter (+-30%)
        if self.policy.jitter:
            jitter_factor = random.uniform(0.7, 1.3)
            delay *= jitter_factor

        return round(delay, 2)

    # ─── Execution ───────────────────────────────────────────────────────

    async def execute_with_retry(
        self,
        func: Callable,
        platform_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an async callable with retry logic.

        On each failure:
        1. Categorize the error
        2. Decide whether to retry based on policy
        3. Calculate the backoff delay
        4. Wait, then retry

        Raises the last exception if all retries are exhausted.
        Returns the function's result on success.
        """
        last_error: Exception | None = None

        for attempt in range(self.policy.max_retries + 1):
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Record success
                success_attempt = RetryAttempt(
                    attempt_number=attempt,
                    error_category=ErrorCategory.TRANSIENT,  # placeholder
                    error_message="",
                    delay_seconds=0,
                    succeeded=True,
                )
                self.history[platform_id].append(success_attempt)

                if attempt > 0:
                    logger.info(
                        f"Succeeded for {platform_id} on attempt {attempt + 1}"
                    )

                return result

            except Exception as e:
                last_error = e
                category = self.categorize_error(e)
                delay = self.calculate_delay(attempt, category)

                # Record the failed attempt
                retry_attempt = RetryAttempt(
                    attempt_number=attempt,
                    error_category=category,
                    error_message=str(e),
                    delay_seconds=delay,
                    succeeded=False,
                )
                self.history[platform_id].append(retry_attempt)

                logger.warning(
                    f"Attempt {attempt + 1}/{self.policy.max_retries + 1} failed "
                    f"for {platform_id}: [{category.value}] {e}"
                )

                # Check if we should retry
                if not self.should_retry(platform_id, category):
                    logger.error(
                        f"Not retrying {platform_id}: "
                        f"category={category.value}, error={e}"
                    )
                    raise

                # If this was the last attempt, don't wait — just raise
                if attempt >= self.policy.max_retries:
                    break

                logger.info(
                    f"Retrying {platform_id} in {delay:.1f}s "
                    f"(attempt {attempt + 2}/{self.policy.max_retries + 1})"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            f"All {self.policy.max_retries + 1} attempts failed for {platform_id}"
        )
        if last_error:
            raise last_error
        raise RuntimeError(f"All retries exhausted for {platform_id}")

    # ─── History & Stats ─────────────────────────────────────────────────

    def get_platform_history(self, platform_id: str) -> list[dict[str, Any]]:
        """Get retry history for a specific platform."""
        return [
            {
                "attempt_number": a.attempt_number,
                "error_category": a.error_category.value,
                "error_message": a.error_message,
                "delay_seconds": a.delay_seconds,
                "timestamp": a.timestamp.isoformat(),
                "succeeded": a.succeeded,
            }
            for a in self.history.get(platform_id, [])
        ]

    def get_stats(self) -> dict[str, Any]:
        """Overall retry statistics across all platforms."""
        total_attempts = 0
        total_successes = 0
        total_failures = 0
        category_counts: dict[str, int] = defaultdict(int)
        platforms_attempted = set()

        for platform_id, attempts in self.history.items():
            platforms_attempted.add(platform_id)
            for attempt in attempts:
                total_attempts += 1
                if attempt.succeeded:
                    total_successes += 1
                else:
                    total_failures += 1
                    category_counts[attempt.error_category.value] += 1

        return {
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": (
                round(total_successes / total_attempts, 3)
                if total_attempts > 0
                else 0.0
            ),
            "platforms_attempted": len(platforms_attempted),
            "error_category_counts": dict(category_counts),
            "policy": {
                "max_retries": self.policy.max_retries,
                "base_delay_seconds": self.policy.base_delay_seconds,
                "max_delay_seconds": self.policy.max_delay_seconds,
                "exponential_base": self.policy.exponential_base,
                "jitter": self.policy.jitter,
                "retry_on": sorted(c.value for c in self.policy.retry_on),
                "no_retry_on": sorted(c.value for c in self.policy.no_retry_on),
            },
        }

    def clear_history(self, platform_id: str = "") -> None:
        """Clear retry history, optionally for a specific platform only."""
        if platform_id:
            self.history.pop(platform_id, None)
            logger.debug(f"Cleared retry history for {platform_id}")
        else:
            self.history.clear()
            logger.debug("Cleared all retry history")
