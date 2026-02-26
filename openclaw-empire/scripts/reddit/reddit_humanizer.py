"""Enhanced humanization engine — gamma-distributed delays + micro-behaviors.

Adapted from geelark-automation/src/humanized_delays.py + micro_behaviors.py
with key improvements that prevent detection:
- Session fatigue actually integrated (not just defined)
- Circadian rhythm speed multiplier
- Abandoned comment starts (unique)
- Much longer warmup before engagement
"""

import math
import random
import time
import logging
from datetime import datetime

logger = logging.getLogger("reddit_humanizer")

# ---------------------------------------------------------------------------
# Gamma-distributed delay profiles
# ---------------------------------------------------------------------------
# Each profile: (shape, scale, min_clamp, max_clamp)
# shape < 1: heavy left skew (mostly short, occasional long)
# shape = 1: exponential
# shape > 1: bell-shaped (peaks at (shape-1)*scale)

DELAY_PROFILES = {
    # Scrolling & browsing
    "scroll_feed":          (2.0, 1.0, 0.8, 5.0),
    "scroll_comments":      (2.5, 0.8, 0.5, 4.0),
    "between_scrolls":      (1.5, 0.4, 0.3, 2.0),

    # Reading & dwelling
    "read_title":           (2.0, 1.5, 1.0, 8.0),
    "read_post":            (3.0, 3.0, 3.0, 25.0),
    "read_comment":         (2.0, 1.5, 1.0, 8.0),

    # Actions
    "before_vote":          (1.5, 0.5, 0.2, 2.0),
    "after_vote":           (2.0, 0.6, 0.3, 2.5),
    "before_comment":       (3.0, 2.0, 2.0, 12.0),
    "typing_pause":         (1.5, 1.0, 0.3, 4.0),
    "after_comment":        (2.5, 1.5, 1.5, 8.0),
    "before_post":          (3.0, 3.0, 3.0, 15.0),
    "after_post":           (3.0, 2.0, 2.0, 10.0),

    # Navigation
    "open_subreddit":       (2.0, 1.2, 1.0, 5.0),
    "between_subreddits":   (3.0, 5.0, 5.0, 30.0),
    "return_to_feed":       (2.0, 1.0, 0.8, 4.0),

    # Rests
    "micro_rest":           (2.0, 3.0, 2.0, 15.0),
    "burst_rest":           (3.0, 15.0, 20.0, 90.0),
    "session_break":        (3.0, 30.0, 30.0, 180.0),
}


def _gamma_delay(profile_name: str) -> float:
    """Generate a gamma-distributed delay for a named profile."""
    shape, scale, min_c, max_c = DELAY_PROFILES[profile_name]
    value = random.gammavariate(shape, scale)
    return max(min_c, min(max_c, value))


# ---------------------------------------------------------------------------
# Circadian rhythm — speed varies by time of day
# ---------------------------------------------------------------------------

def _circadian_multiplier() -> float:
    """Return a speed multiplier based on time of day.

    Early morning / late night: slower (1.2-1.5x longer delays)
    Mid-morning / afternoon: faster (0.8-0.9x)
    Evening: normal (1.0x)
    """
    hour = datetime.now().hour
    if hour < 7:
        return random.uniform(1.3, 1.6)
    elif hour < 10:
        return random.uniform(0.85, 1.0)
    elif hour < 14:
        return random.uniform(0.8, 0.95)
    elif hour < 18:
        return random.uniform(0.85, 1.0)
    elif hour < 21:
        return random.uniform(0.95, 1.1)
    else:
        return random.uniform(1.1, 1.4)


# ---------------------------------------------------------------------------
# Session fatigue — delays increase toward end of session
# ---------------------------------------------------------------------------

class SessionFatigue:
    """Track session progress and apply fatigue multiplier."""

    def __init__(self, planned_actions: int):
        self.planned = max(1, planned_actions)
        self.completed = 0

    def tick(self):
        self.completed += 1

    @property
    def multiplier(self) -> float:
        """1.0 in first 60%, 1.2-1.5 in final 40%."""
        progress = self.completed / self.planned
        if progress < 0.6:
            return 1.0
        # Linear ramp from 1.0 to 1.5 across final 40%
        fatigue_progress = (progress - 0.6) / 0.4
        return 1.0 + fatigue_progress * random.uniform(0.2, 0.5)


# Module-level fatigue tracker, reset per session
_fatigue: SessionFatigue | None = None


def start_session(planned_actions: int):
    """Call at session start to initialize fatigue tracking."""
    global _fatigue
    _fatigue = SessionFatigue(planned_actions)


def tick_action():
    """Call after each action to advance fatigue counter."""
    if _fatigue:
        _fatigue.tick()


# ---------------------------------------------------------------------------
# Main delay function
# ---------------------------------------------------------------------------

def sleep_humanized(profile_name: str, dry_run: bool = False) -> float:
    """Sleep for a humanized duration based on named profile.

    Returns the actual sleep duration (useful for logging).
    """
    base = _gamma_delay(profile_name)
    circadian = _circadian_multiplier()
    fatigue = _fatigue.multiplier if _fatigue else 1.0

    duration = base * circadian * fatigue

    # Clamp to reasonable bounds
    duration = max(0.1, min(duration, 300.0))

    if not dry_run:
        time.sleep(duration)
    else:
        logger.debug(f"[DRY] sleep_humanized({profile_name}) = {duration:.1f}s")

    return duration


# ---------------------------------------------------------------------------
# Micro-behaviors — probabilistic human-like quirks
# ---------------------------------------------------------------------------

def maybe_re_read_scroll(dry_run: bool = False) -> bool:
    """12% chance to scroll back up slightly while 're-reading'.
    Returns True if the behavior was triggered."""
    if random.random() < 0.12:
        logger.debug("Micro: re-read scroll-back")
        if not dry_run:
            pause = random.uniform(1.5, 4.0)
            time.sleep(pause)
        return True
    return False


def maybe_micro_pause(dry_run: bool = False) -> bool:
    """20% chance for a tiny hesitation suggesting thought."""
    if random.random() < 0.20:
        logger.debug("Micro: micro-pause")
        if not dry_run:
            time.sleep(random.uniform(0.3, 2.0))
        return True
    return False


def maybe_overshoot_correction(dry_run: bool = False) -> bool:
    """8% chance to overshoot scroll then correct back."""
    if random.random() < 0.08:
        logger.debug("Micro: overshoot correction")
        return True  # Caller handles the actual swipe correction
    return False


def should_abandon_comment() -> bool:
    """5% chance to start typing a comment then abandon it.
    Unique to this system — simulates changed mind."""
    return random.random() < 0.05


def should_skip_action() -> bool:
    """10% base chance to randomly skip any single action.
    Adds natural inconsistency to behavior patterns."""
    return random.random() < 0.10


# ---------------------------------------------------------------------------
# Variable session activity counts
# ---------------------------------------------------------------------------

def session_activity_count(mode: str = "browse") -> int:
    """Generate gamma-distributed activity count for a session type.

    Returns how many main actions (scrolls, votes, comments) to perform.
    """
    profiles = {
        "browse":   (3.0, 3.0, 5, 20),   # 5-20 actions
        "comment":  (2.5, 4.0, 8, 25),    # 8-25 actions
        "post":     (2.0, 5.0, 10, 30),   # 10-30 actions (includes bracketing)
    }
    shape, scale, min_c, max_c = profiles.get(mode, profiles["browse"])
    count = random.gammavariate(shape, scale)
    return max(min_c, min(max_c, int(count)))


def randomized_swipe_params() -> dict:
    """Generate variable swipe parameters for natural scrolling."""
    return {
        "start_x": random.randint(400, 680),
        "start_y": random.randint(1400, 1800),
        "end_y": random.randint(400, 900),
        "duration_ms": random.randint(250, 750),
    }
