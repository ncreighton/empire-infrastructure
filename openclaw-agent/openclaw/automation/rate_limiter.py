"""RateLimiter — intelligent rate limiting for signup operations.

Enforces per-platform cooldowns, hourly/daily caps, and automatic pausing after
consecutive failures. All delays include configurable jitter to avoid predictable
timing patterns that could trigger anti-bot detection.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration with sensible defaults for signup automation."""

    global_min_delay_seconds: int = 30       # Min seconds between any two signups
    global_max_delay_seconds: int = 120      # Max random delay between signups
    platform_cooldown_hours: float = 24      # Don't retry same platform within
    max_signups_per_hour: int = 5            # Hourly signup cap
    max_signups_per_day: int = 20            # Daily signup cap
    max_failures_before_pause: int = 3       # Pause after N consecutive failures
    failure_pause_minutes: int = 30          # How long to pause after failure streak
    jitter_range: tuple[float, float] = (0.8, 1.3)  # Multiply delays by random factor


class RateLimiter:
    """Enforce rate limits on signup operations to avoid detection and bans.

    Usage::

        limiter = RateLimiter()
        ok, reason = limiter.can_proceed("gumroad")
        if ok:
            await limiter.wait("gumroad")
            limiter.record_attempt("gumroad")
            # ... do signup ...
            limiter.record_success("gumroad")
        else:
            print(f"Cannot proceed: {reason}")
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._signup_times: list[datetime] = []
        self._platform_last_attempt: dict[str, datetime] = {}
        self._platform_attempts: dict[str, list[datetime]] = defaultdict(list)
        self._consecutive_failures: int = 0
        self._paused_until: datetime | None = None

    # ─── Gate Checks ─────────────────────────────────────────────────────

    def can_proceed(self, platform_id: str = "") -> tuple[bool, str]:
        """Check if a signup attempt can proceed right now.

        Returns a tuple of (allowed, reason_if_denied).
        """
        # Check if we are in a failure-induced pause
        if self.is_paused:
            remaining = 0.0
            if self._paused_until:
                remaining = (self._paused_until - datetime.now()).total_seconds()
            return (
                False,
                f"Paused due to {self._consecutive_failures} consecutive failures. "
                f"Resumes in {remaining:.0f}s",
            )

        # Check hourly limit
        if self.signups_this_hour >= self.config.max_signups_per_hour:
            return (
                False,
                f"Hourly signup limit reached ({self.config.max_signups_per_hour}/hr)",
            )

        # Check daily limit
        if self.signups_today >= self.config.max_signups_per_day:
            return (
                False,
                f"Daily signup limit reached ({self.config.max_signups_per_day}/day)",
            )

        # Check platform-specific cooldown
        if platform_id and platform_id in self._platform_last_attempt:
            last = self._platform_last_attempt[platform_id]
            cooldown = timedelta(hours=self.config.platform_cooldown_hours)
            if datetime.now() - last < cooldown:
                remaining = (last + cooldown - datetime.now()).total_seconds()
                return (
                    False,
                    f"Platform {platform_id} cooldown active. "
                    f"Wait {remaining:.0f}s ({self.config.platform_cooldown_hours}h cooldown)",
                )

        return (True, "")

    # ─── Delay Calculation ───────────────────────────────────────────────

    def wait_time(self, platform_id: str = "") -> float:
        """Calculate how many seconds to wait before the next signup.

        Combines the global random delay with jitter. If the last signup was
        recent, the delay accounts for the remaining cooldown.
        """
        # Base delay: random between min and max
        base_delay = random.uniform(
            self.config.global_min_delay_seconds,
            self.config.global_max_delay_seconds,
        )

        # Apply jitter multiplier
        jitter = random.uniform(*self.config.jitter_range)
        delay = base_delay * jitter

        # If we have a recent signup, ensure we wait at least the min delay
        if self._signup_times:
            last_signup = self._signup_times[-1]
            elapsed = (datetime.now() - last_signup).total_seconds()
            min_remaining = max(
                0, self.config.global_min_delay_seconds - elapsed
            )
            delay = max(delay, min_remaining)

        # If paused, wait until pause expires
        if self._paused_until and datetime.now() < self._paused_until:
            pause_remaining = (
                self._paused_until - datetime.now()
            ).total_seconds()
            delay = max(delay, pause_remaining)

        return round(delay, 2)

    async def wait(self, platform_id: str = "") -> None:
        """Async wait the appropriate time before proceeding with a signup.

        Logs the wait duration so operators can monitor pacing.
        """
        delay = self.wait_time(platform_id)
        if delay > 0:
            logger.info(f"Rate limiter: waiting {delay:.1f}s before next signup")
            await asyncio.sleep(delay)

    # ─── Recording ───────────────────────────────────────────────────────

    def record_attempt(self, platform_id: str) -> None:
        """Record that a signup attempt was made."""
        now = datetime.now()
        self._signup_times.append(now)
        self._platform_last_attempt[platform_id] = now
        self._platform_attempts[platform_id].append(now)
        logger.debug(
            f"Recorded signup attempt: {platform_id} "
            f"(hour={self.signups_this_hour}, day={self.signups_today})"
        )

    def record_success(self, platform_id: str) -> None:
        """Record a successful signup — resets the consecutive failure counter."""
        self._consecutive_failures = 0
        self._paused_until = None
        logger.debug(f"Signup success recorded: {platform_id}")

    def record_failure(self, platform_id: str) -> None:
        """Record a failed signup. Triggers a pause after too many consecutive failures."""
        self._consecutive_failures += 1
        logger.debug(
            f"Signup failure recorded: {platform_id} "
            f"(consecutive={self._consecutive_failures})"
        )

        if self._consecutive_failures >= self.config.max_failures_before_pause:
            pause_duration = timedelta(
                minutes=self.config.failure_pause_minutes
            )
            self._paused_until = datetime.now() + pause_duration
            logger.warning(
                f"Rate limiter paused for {self.config.failure_pause_minutes}m "
                f"after {self._consecutive_failures} consecutive failures"
            )

    def reset(self) -> None:
        """Reset all rate limiting state."""
        self._signup_times.clear()
        self._platform_last_attempt.clear()
        self._platform_attempts.clear()
        self._consecutive_failures = 0
        self._paused_until = None
        logger.info("Rate limiter state reset")

    # ─── Properties ──────────────────────────────────────────────────────

    @property
    def is_paused(self) -> bool:
        """True if the limiter is in a failure-induced pause."""
        if self._paused_until is None:
            return False
        if datetime.now() >= self._paused_until:
            # Pause has expired
            self._paused_until = None
            return False
        return True

    @property
    def signups_this_hour(self) -> int:
        """Number of signup attempts in the last 60 minutes."""
        cutoff = datetime.now() - timedelta(hours=1)
        return sum(1 for t in self._signup_times if t > cutoff)

    @property
    def signups_today(self) -> int:
        """Number of signup attempts in the last 24 hours."""
        cutoff = datetime.now() - timedelta(hours=24)
        return sum(1 for t in self._signup_times if t > cutoff)

    # ─── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Current rate limiter state for monitoring and debugging."""
        # Per-platform last attempt times
        platform_cooldowns: dict[str, dict[str, Any]] = {}
        for pid, last_time in self._platform_last_attempt.items():
            cooldown_ends = last_time + timedelta(
                hours=self.config.platform_cooldown_hours
            )
            is_active = datetime.now() < cooldown_ends
            platform_cooldowns[pid] = {
                "last_attempt": last_time.isoformat(),
                "cooldown_ends": cooldown_ends.isoformat(),
                "cooldown_active": is_active,
                "total_attempts": len(self._platform_attempts.get(pid, [])),
            }

        return {
            "signups_this_hour": self.signups_this_hour,
            "signups_today": self.signups_today,
            "max_per_hour": self.config.max_signups_per_hour,
            "max_per_day": self.config.max_signups_per_day,
            "consecutive_failures": self._consecutive_failures,
            "is_paused": self.is_paused,
            "paused_until": (
                self._paused_until.isoformat() if self._paused_until else None
            ),
            "platform_cooldowns": platform_cooldowns,
            "config": {
                "global_min_delay_seconds": self.config.global_min_delay_seconds,
                "global_max_delay_seconds": self.config.global_max_delay_seconds,
                "platform_cooldown_hours": self.config.platform_cooldown_hours,
                "max_failures_before_pause": self.config.max_failures_before_pause,
                "failure_pause_minutes": self.config.failure_pause_minutes,
            },
        }
