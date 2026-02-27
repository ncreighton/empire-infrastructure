"""Session engine — browse, vote, comment, and post sessions.

Three session types:
1. Browse: scroll feed, upvote 3-8 posts, read 2-5 posts. 5-15 min.
2. Comment: browse + 1-3 comments through safety pipeline. 10-25 min.
3. Post: create one own-content post, bracketed by natural browsing.
"""

import logging
import random
import time
from datetime import datetime

from .reddit_adb import check_adb_connection, ensure_screen_on, go_home
from .reddit_browser import (
    launch_reddit, close_reddit, navigate_to_subreddit, scroll_feed,
    read_post, extract_visible_post_text, upvote_current,
    type_comment, create_post, check_inbox,
)
from .reddit_config import (
    TIER1_SUBREDDITS, TIER2_SUBREDDITS, TIER3_SUBREDDITS,
    EXPERTISE_TOPICS,
)
from .reddit_content import generate_comment, generate_post
from .reddit_humanizer import (
    sleep_humanized, start_session, tick_action,
    session_activity_count, should_skip_action, should_abandon_comment,
)
from .reddit_safety import SafetyEngine
from .reddit_state import RedditState

logger = logging.getLogger("reddit_engagement")


class RedditSession:
    """Base class for Reddit automation sessions."""

    def __init__(self, state: RedditState, safety: SafetyEngine,
                 dry_run: bool = False):
        self.state = state
        self.safety = safety
        self.dry_run = dry_run
        self.actions = {
            "scrolls": 0, "upvotes": 0, "comments": 0,
            "posts": 0, "subs_visited": [],
        }
        self.start_time = None

    def _pick_subreddits(self, count: int, tiers: list[list]) -> list[str]:
        """Pick random subreddits from given tiers."""
        pool = []
        for tier in tiers:
            pool.extend([s["name"] for s in tier])
        random.shuffle(pool)
        return pool[:count]

    def _browse_and_vote(self, sub: str, scroll_count: int = 5,
                         vote_chance: float = 0.4) -> int:
        """Browse feed, scrolling and voting. Returns votes cast.

        Varies time per post: quick glance, normal read, or longer
        engagement (simulating reading a full post or watching a video).
        """
        votes = 0
        for i in range(scroll_count):
            if should_skip_action():
                continue

            actual = scroll_feed(1, dry_run=self.dry_run)
            self.actions["scrolls"] += actual
            tick_action()

            # Variable reading time — simulate different engagement levels
            engagement = random.random()
            if engagement < 0.15:
                # Quick scroll past (15%) — barely glance
                sleep_humanized("between_scrolls", dry_run=self.dry_run)
            elif engagement < 0.70:
                # Normal read (55%) — read title, glance at image
                sleep_humanized("read_title", dry_run=self.dry_run)
            elif engagement < 0.90:
                # Longer read (20%) — read post content, look at image
                sleep_humanized("read_post", dry_run=self.dry_run)
            else:
                # Deep engagement (10%) — watch video or read comments
                sleep_humanized("read_post", dry_run=self.dry_run)
                sleep_humanized("read_post", dry_run=self.dry_run)

            # Vote (with randomized chance)
            if random.random() < vote_chance and self.safety.can_do("upvote", sub):
                if upvote_current(dry_run=self.dry_run):
                    votes += 1
                    self.actions["upvotes"] += 1
                    self.safety.record_action("upvote", sub)
                    tick_action()

        return votes

    def _duration(self) -> float:
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0


class BrowseSession(RedditSession):
    """Natural browsing: mix of home feed scrolling + occasional subreddit visits.

    Mimics real behavior: scroll home feed, read posts, look at images/videos,
    upvote interesting things across various topics (not just niche), occasionally
    visit a targeted subreddit. Vary time spent per post naturally.
    """

    def run(self) -> dict:
        self.start_time = datetime.now()
        planned = session_activity_count("browse")
        start_session(planned)
        logger.info(f"Browse session starting (planned={planned} actions)")

        if not self.dry_run:
            if not check_adb_connection():
                logger.error("ADB connection failed")
                return self.actions
            launch_reddit()

        # Phase 1: Scroll home feed naturally (this is what most Reddit time is)
        home_scrolls = random.randint(5, 15)
        logger.info(f"Browsing home feed ({home_scrolls} scrolls)")
        self.actions["subs_visited"].append("home")
        self._browse_and_vote("home", home_scrolls, vote_chance=0.25)

        # Phase 2: Maybe visit 1-2 specific subreddits (not every session)
        if random.random() < 0.6:  # 60% chance to visit subreddits
            subs = self._pick_subreddits(
                random.randint(1, 2),
                [TIER1_SUBREDDITS, TIER2_SUBREDDITS, TIER3_SUBREDDITS],
            )
            for sub in subs:
                if not self.dry_run:
                    if not navigate_to_subreddit(sub):
                        continue
                self.actions["subs_visited"].append(sub)
                scroll_count = random.randint(3, 8)
                self._browse_and_vote(sub, scroll_count, vote_chance=0.35)
                sleep_humanized("between_subreddits", dry_run=self.dry_run)

            # Go back to home feed after subreddit visits
            if not self.dry_run:
                go_home()
                time.sleep(1)
                launch_reddit()

        # Phase 3: Final home feed scroll (wind down naturally)
        if random.random() < 0.5:
            wind_down = random.randint(2, 6)
            logger.info(f"Wind-down home scrolling ({wind_down} scrolls)")
            self._browse_and_vote("home", wind_down, vote_chance=0.20)

        if not self.dry_run:
            close_reddit()

        duration = self._duration()
        self.state.log_session("browse", duration, self.actions, self.actions["subs_visited"])
        logger.info(f"Browse session complete ({duration:.0f}s): {self.actions}")
        return self.actions


class CommentSession(RedditSession):
    """Browse + 1-3 comments through full safety pipeline."""

    def run(self) -> dict:
        self.start_time = datetime.now()
        planned = session_activity_count("comment")
        start_session(planned)
        target_comments = random.randint(1, 3)
        logger.info(f"Comment session starting (planned={planned}, target_comments={target_comments})")

        if not self.dry_run:
            if not check_adb_connection():
                logger.error("ADB connection failed")
                return self.actions
            launch_reddit()

        # Mix of browsing + commenting subs
        subs = self._pick_subreddits(
            random.randint(2, 4),
            [TIER1_SUBREDDITS, TIER2_SUBREDDITS],
        )

        comments_made = 0
        recent_comments = [c["comment_text"] for c in self.state.get_recent_comments(10)]

        for sub in subs:
            if not self.dry_run:
                if not navigate_to_subreddit(sub):
                    continue
            self.actions["subs_visited"].append(sub)

            scroll_count = random.randint(4, 10)
            self._browse_and_vote(sub, scroll_count, vote_chance=0.40)

            # Try to comment if we haven't hit target
            if comments_made < target_comments and self.safety.can_do("comment", sub):
                post_info = read_post() if not self.dry_run else {
                    "title": "Simulated post",
                    "body": "Simulated body about 3D printing",
                    "subreddit": sub,
                }

                if post_info:
                    # Abandoned comment micro-behavior
                    if should_abandon_comment():
                        logger.info("Abandoned comment (micro-behavior)")
                        sleep_humanized("typing_pause", dry_run=self.dry_run)
                        continue

                    comment = generate_comment(
                        post_info.get("title", ""),
                        post_info.get("body", ""),
                        sub,
                        recent_comments=recent_comments,
                    )

                    if comment and self.safety.is_comment_unique(comment):
                        if type_comment(comment, dry_run=self.dry_run):
                            comments_made += 1
                            self.actions["comments"] += 1
                            self.safety.record_action("comment", sub)
                            ngram_hash = self.safety._compute_ngram_hash(comment)
                            self.state.add_comment(
                                sub, post_info.get("title", ""), comment,
                                ngram_hash=ngram_hash,
                            )
                            recent_comments.append(comment)
                            tick_action()
                            # Karma estimate: assume each comment gets 1-3 upvotes
                            self.state.increment("karma_estimate", random.randint(1, 3))

            sleep_humanized("between_subreddits", dry_run=self.dry_run)

        # Check inbox for ban signals
        if not self.dry_run:
            inbox_text = check_inbox()
            if inbox_text:
                signal = self.safety.check_ban_signals(inbox_text)
                if signal:
                    self.safety.handle_ban_signal(signal)

            close_reddit()

        duration = self._duration()
        self.state.log_session("comment", duration, self.actions, self.actions["subs_visited"])
        logger.info(f"Comment session complete ({duration:.0f}s): {self.actions}")
        return self.actions


class PostSession(RedditSession):
    """Create one own-content post, bracketed by natural browsing."""

    def run(self, topic: str = "", is_promo: bool = False) -> dict:
        self.start_time = datetime.now()
        planned = session_activity_count("post")
        start_session(planned)
        action_type = "promo" if is_promo else "post"
        logger.info(f"Post session starting (type={action_type})")

        if not self.dry_run:
            if not check_adb_connection():
                logger.error("ADB connection failed")
                return self.actions
            launch_reddit()

        # Pre-browsing (look natural before posting)
        pre_subs = self._pick_subreddits(
            random.randint(1, 2),
            [TIER1_SUBREDDITS, TIER2_SUBREDDITS, TIER3_SUBREDDITS],
        )
        for sub in pre_subs:
            if not self.dry_run:
                if not navigate_to_subreddit(sub):
                    continue
            self.actions["subs_visited"].append(sub)
            self._browse_and_vote(sub, random.randint(2, 5))
            sleep_humanized("between_subreddits", dry_run=self.dry_run)

        # Pick a posting subreddit
        target_sub = random.choice([s["name"] for s in TIER1_SUBREDDITS])

        if self.safety.can_do(action_type, target_sub):
            post_data = generate_post(target_sub, topic=topic, is_promo=is_promo)

            if create_post(
                target_sub, post_data["title"], post_data["body"],
                dry_run=self.dry_run,
            ):
                self.actions["posts"] += 1
                self.safety.record_action("post", target_sub)
                self.state.add_post(
                    target_sub, post_data["title"], post_data["body"][:500],
                    is_promo=is_promo,
                    etsy_link="etsy" in post_data["body"].lower(),
                )
                self.actions["subs_visited"].append(target_sub)
                tick_action()
                # Karma: assume posts get 5-20 upvotes
                self.state.increment("karma_estimate", random.randint(5, 20))
        else:
            logger.info(f"Safety gate blocked {action_type} in r/{target_sub}")

        # Post-browsing (continue looking natural)
        post_subs = self._pick_subreddits(
            random.randint(1, 2),
            [TIER1_SUBREDDITS, TIER2_SUBREDDITS],
        )
        for sub in post_subs:
            if not self.dry_run:
                if not navigate_to_subreddit(sub):
                    continue
            self.actions["subs_visited"].append(sub)
            self._browse_and_vote(sub, random.randint(2, 4))
            sleep_humanized("between_subreddits", dry_run=self.dry_run)

        if not self.dry_run:
            close_reddit()

        duration = self._duration()
        self.state.log_session("post", duration, self.actions, self.actions["subs_visited"])
        logger.info(f"Post session complete ({duration:.0f}s): {self.actions}")
        return self.actions


def run_session(session_type: str, dry_run: bool = False, **kwargs) -> dict:
    """Run a Reddit session of the given type."""
    state = RedditState()
    safety = SafetyEngine(state)

    # Pre-flight checks
    if safety.is_banned():
        logger.error("Account is banned. No sessions will run.")
        return {}
    if safety.is_rate_limited():
        logger.warning("Rate limited. Skipping session.")
        return {}

    try:
        if session_type == "browse":
            return BrowseSession(state, safety, dry_run).run()
        elif session_type == "comment":
            return CommentSession(state, safety, dry_run).run()
        elif session_type == "post":
            return PostSession(state, safety, dry_run).run(**kwargs)
        else:
            logger.error(f"Unknown session type: {session_type}")
            return {}
    finally:
        state.close()
