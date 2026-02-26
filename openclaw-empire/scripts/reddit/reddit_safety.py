"""Anti-ban engine — every action must pass ALL safety gates.

4-phase warmup (much more conservative than GeeLark):
  lurk (0-14d): browse + upvote only
  comment (15-28d): + comments
  active (29-56d): + posts (no promo)
  established (57+d): + promo posts

8 safety gates checked before every action.
Ban detection via inbox keyword scanning.
"""

import hashlib
import logging
import random
from datetime import datetime, timedelta, date

from .reddit_state import RedditState
from .reddit_config import PROMO_RULES

logger = logging.getLogger("reddit_safety")

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASE_LIMITS = {
    "lurk": {
        "allowed_actions": {"browse", "upvote", "save"},
        "daily_upvotes": 10,
        "daily_comments": 0,
        "daily_posts": 0,
        "max_sessions": 2,
    },
    "comment": {
        "allowed_actions": {"browse", "upvote", "save", "comment"},
        "daily_upvotes": 20,
        "daily_comments": 3,
        "daily_posts": 0,
        "max_sessions": 3,
    },
    "active": {
        "allowed_actions": {"browse", "upvote", "save", "comment", "post"},
        "daily_upvotes": 30,
        "daily_comments": 8,
        "daily_posts": 2,  # per week, enforced separately
        "max_sessions": 4,
    },
    "established": {
        "allowed_actions": {"browse", "upvote", "save", "comment", "post", "promo"},
        "daily_upvotes": 40,
        "daily_comments": 12,
        "daily_posts": 4,  # per week
        "max_sessions": 5,
    },
}


class SafetyEngine:
    """Central safety gate — every action passes through here."""

    def __init__(self, state: RedditState | None = None):
        self.state = state or RedditState()
        self._blacklisted_subs: set[str] = set()
        self._load_blacklist()

    def _load_blacklist(self):
        bl = self.state.get("blacklisted_subs", "")
        if bl:
            self._blacklisted_subs = set(bl.split(","))

    def _save_blacklist(self):
        self.state.set("blacklisted_subs", ",".join(self._blacklisted_subs))

    # --- Main gate ---

    def can_do(self, action: str, subreddit: str = "") -> bool:
        """Check ALL safety gates. Returns True only if ALL pass."""
        phase = self.state.get_phase()
        limits = PHASE_LIMITS.get(phase, PHASE_LIMITS["lurk"])

        # Gate 1: Phase allows this action type
        if action not in limits["allowed_actions"]:
            logger.debug(f"Gate 1 FAIL: {action} not in {phase} phase")
            return False

        # Gate 2: Daily limit not exceeded
        if not self._check_daily_limit(action, limits):
            logger.debug(f"Gate 2 FAIL: daily {action} limit reached")
            return False

        # Gate 3: Self-promo ratio check (for promo actions)
        if action == "promo" and not self._check_promo_ratio():
            logger.debug("Gate 3 FAIL: promo ratio >= 10%")
            return False

        # Gate 4: Promo cooldown (48h between promo posts)
        if action == "promo" and not self._check_promo_cooldown():
            logger.debug("Gate 4 FAIL: promo cooldown not elapsed")
            return False

        # Gate 5: Subreddit not blacklisted
        if subreddit and subreddit.lower() in {s.lower() for s in self._blacklisted_subs}:
            logger.debug(f"Gate 5 FAIL: {subreddit} is blacklisted")
            return False

        # Gate 6: Max comments per sub per day
        if action == "comment" and subreddit:
            sub_comments = self.state.get_daily_count("comment", subreddit)
            if sub_comments >= PROMO_RULES["max_same_sub_comments_day"]:
                logger.debug(f"Gate 6 FAIL: {sub_comments} comments in {subreddit} today")
                return False

        # Gate 7: Karma minimum for promo
        if action == "promo":
            karma = self.state.get_int("karma_estimate", 0)
            if karma < PROMO_RULES["min_karma_for_promo"]:
                logger.debug(f"Gate 7 FAIL: karma {karma} < {PROMO_RULES['min_karma_for_promo']}")
                return False

        # Gate 8: Skip day check
        if self._is_skip_day():
            logger.debug("Gate 8 FAIL: skip day")
            return False

        return True

    def _check_daily_limit(self, action: str, limits: dict) -> bool:
        limit_key = f"daily_{action}s" if action != "upvote" else "daily_upvotes"
        max_count = limits.get(limit_key, 999)

        # For posts, check weekly instead of daily
        if action == "post":
            return self._check_weekly_posts(max_count)

        today_count = self.state.get_daily_count(action)
        return today_count < max_count

    def _check_weekly_posts(self, max_per_week: int) -> bool:
        """Check posts in the last 7 days."""
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        # Use total from post_history
        state = self.state
        rows = state._conn.execute(
            "SELECT COUNT(*) as cnt FROM post_history WHERE timestamp > ?",
            (cutoff,),
        ).fetchone()
        return rows["cnt"] < max_per_week

    def _check_promo_ratio(self) -> bool:
        return self.state.get_promo_ratio() < PROMO_RULES["max_promo_ratio"]

    def _check_promo_cooldown(self) -> bool:
        last_promo = self.state.get_last_promo_time()
        if last_promo is None:
            return True
        hours_since = (datetime.now() - last_promo).total_seconds() / 3600
        return hours_since >= PROMO_RULES["promo_cooldown_hours"]

    def _is_skip_day(self) -> bool:
        """20% base skip chance + forced after 5 consecutive active days."""
        today = date.today().isoformat()
        skip_key = f"skip_day_{today}"
        cached = self.state.get(skip_key, "")
        if cached:
            return cached == "yes"

        # Check consecutive active days
        consecutive = 0
        for i in range(1, 8):
            d = (date.today() - timedelta(days=i)).isoformat()
            if self.state.get(f"active_day_{d}", "") == "yes":
                consecutive += 1
            else:
                break

        if consecutive >= 5:
            # Force skip after 5 consecutive days
            self.state.set(skip_key, "yes")
            logger.info(f"Forced skip day after {consecutive} consecutive active days")
            return True

        # 20% random skip
        skip = random.random() < 0.20
        self.state.set(skip_key, "yes" if skip else "no")
        if not skip:
            self.state.set(f"active_day_{today}", "yes")
        return skip

    # --- Comment dedup ---

    def is_comment_unique(self, text: str) -> bool:
        """Check if comment is sufficiently unique using n-gram hashing."""
        ngram_hash = self._compute_ngram_hash(text)
        recent_hashes = self.state.get_recent_ngram_hashes(50)
        return ngram_hash not in recent_hashes

    @staticmethod
    def _compute_ngram_hash(text: str, n: int = 3) -> str:
        """Compute a hash from the n-grams of a text."""
        words = text.lower().split()
        if len(words) < n:
            return hashlib.md5(text.lower().encode()).hexdigest()[:12]
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngrams.add(" ".join(words[i:i + n]))
        # Sort for deterministic hashing
        ngram_str = "|".join(sorted(ngrams))
        return hashlib.md5(ngram_str.encode()).hexdigest()[:12]

    # --- Ban detection ---

    def check_ban_signals(self, inbox_text: str) -> dict | None:
        """Scan inbox/notification text for ban signals.

        Returns dict with severity + recommended action, or None if clean.
        """
        text_lower = inbox_text.lower()

        # Severity levels
        if any(kw in text_lower for kw in [
            "permanently banned", "account suspended", "suspended from reddit",
        ]):
            return {
                "severity": "account_ban",
                "action": "full_stop",
                "cooldown_hours": 999999,
                "message": "Account-level ban detected. Full stop.",
            }

        if any(kw in text_lower for kw in [
            "banned from", "you have been banned from",
        ]):
            return {
                "severity": "subreddit_ban",
                "action": "blacklist_sub",
                "cooldown_hours": 0,
                "message": "Subreddit ban detected.",
            }

        if any(kw in text_lower for kw in [
            "removed", "your post has been removed", "your comment has been removed",
            "violates our rules",
        ]):
            return {
                "severity": "content_removed",
                "action": "avoid_sub_7d",
                "cooldown_hours": 168,  # 7 days
                "message": "Content removed. Avoiding sub for 7 days.",
            }

        if any(kw in text_lower for kw in [
            "rate limit", "you are doing that too much", "try again later",
            "too many requests",
        ]):
            return {
                "severity": "rate_limit",
                "action": "backoff_6h",
                "cooldown_hours": 6,
                "message": "Rate limited. Backing off 6 hours.",
            }

        return None

    def handle_ban_signal(self, signal: dict, subreddit: str = ""):
        """Take action based on a detected ban signal."""
        severity = signal["severity"]
        logger.warning(f"Ban signal: {severity} — {signal['message']}")

        if severity == "account_ban":
            self.state.set("account_banned", "true")
            self.state.set("ban_detected_at", datetime.now().isoformat())

        elif severity == "subreddit_ban" and subreddit:
            self._blacklisted_subs.add(subreddit.lower())
            self._save_blacklist()
            logger.warning(f"Blacklisted r/{subreddit}")

        elif severity == "content_removed":
            if subreddit:
                avoid_until = (datetime.now() + timedelta(hours=signal["cooldown_hours"])).isoformat()
                self.state.set(f"avoid_sub_{subreddit.lower()}", avoid_until)

        elif severity == "rate_limit":
            resume_at = (datetime.now() + timedelta(hours=signal["cooldown_hours"])).isoformat()
            self.state.set("rate_limit_until", resume_at)

    def is_banned(self) -> bool:
        """Check if account is banned."""
        return self.state.get("account_banned", "") == "true"

    def is_rate_limited(self) -> bool:
        """Check if we're in a rate limit backoff period."""
        until = self.state.get("rate_limit_until", "")
        if not until:
            return False
        return datetime.now() < datetime.fromisoformat(until)

    # --- Recording ---

    def record_action(self, action: str, subreddit: str = ""):
        """Record that an action was performed (for limit tracking)."""
        self.state.increment_daily(action, subreddit)

    def get_status(self) -> dict:
        """Return current safety status for display."""
        phase = self.state.get_phase()
        limits = PHASE_LIMITS.get(phase, PHASE_LIMITS["lurk"])
        return {
            "phase": phase,
            "account_age_days": self.state.get_account_age_days(),
            "karma_estimate": self.state.get_int("karma_estimate"),
            "promo_ratio": f"{self.state.get_promo_ratio():.1%}",
            "allowed_actions": sorted(limits["allowed_actions"]),
            "daily_counts": self.state.get_all_daily_counts(),
            "banned": self.is_banned(),
            "rate_limited": self.is_rate_limited(),
            "blacklisted_subs": sorted(self._blacklisted_subs),
        }
