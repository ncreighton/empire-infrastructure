"""Reddit app navigation via ADB — launch, browse, vote, comment, post.

Uses UI dump as primary element finder with vision AI as fallback.
"""

import logging
import random
import re
import time

from .reddit_adb import (
    adb_shell, adb_tap, adb_swipe, adb_type, adb_keyevent,
    ensure_screen_on, go_home, dump_ui, find_node, find_all_nodes, screenshot,
)
from .reddit_config import REDDIT_PACKAGE
from .reddit_humanizer import (
    sleep_humanized, maybe_re_read_scroll, maybe_micro_pause,
    maybe_overshoot_correction, randomized_swipe_params,
)

logger = logging.getLogger("reddit_browser")


def launch_reddit():
    """Open the Reddit app."""
    ensure_screen_on()
    go_home()
    adb_shell(f"monkey -p {REDDIT_PACKAGE} -c android.intent.category.LAUNCHER 1")
    sleep_humanized("open_subreddit")
    time.sleep(3)
    logger.info("Reddit app launched")


def close_reddit():
    """Return home and force-stop Reddit."""
    go_home()
    adb_shell(f"am force-stop {REDDIT_PACKAGE}")
    time.sleep(1)


def navigate_to_subreddit(name: str) -> bool:
    """Navigate to a specific subreddit via search."""
    logger.info(f"Navigating to r/{name}")

    # Tap search icon (usually top bar)
    root = dump_ui()
    search = find_node(root, desc="Search") or find_node(root, text="Search")
    if search:
        adb_tap(*search)
    else:
        # Fallback: tap search area in top bar
        adb_tap(540, 120)
    sleep_humanized("open_subreddit")

    # Type subreddit name
    adb_type(f"r/{name}")
    time.sleep(2)

    # Find and tap the subreddit result
    root = dump_ui()
    sub_node = find_node(root, text=f"r/{name}")
    if not sub_node:
        # Try case-insensitive or partial
        nodes = find_all_nodes(root, text=name)
        if nodes:
            sub_node = (nodes[0][0], nodes[0][1])

    if sub_node:
        adb_tap(*sub_node)
        sleep_humanized("open_subreddit")
        logger.info(f"Entered r/{name}")
        return True
    else:
        logger.warning(f"Could not find r/{name} in search results")
        adb_keyevent(4)  # Back
        time.sleep(1)
        return False


def scroll_feed(count: int = 1, dry_run: bool = False) -> int:
    """Scroll the feed with variable swipe patterns. Returns actual scrolls."""
    actual = 0
    for _ in range(count):
        if dry_run:
            actual += 1
            continue

        params = randomized_swipe_params()
        adb_swipe(params["start_x"], params["start_y"],
                  params["start_x"], params["end_y"],
                  params["duration_ms"])
        actual += 1

        # Micro-behaviors
        if maybe_overshoot_correction():
            # Scroll back slightly
            adb_swipe(params["start_x"], params["end_y"],
                      params["start_x"], params["end_y"] + random.randint(100, 300),
                      random.randint(200, 400))

        if maybe_re_read_scroll():
            # Already sleeps internally
            pass

        maybe_micro_pause()

        sleep_humanized("between_scrolls")

    return actual


def read_post() -> dict | None:
    """Open and read the currently visible post. Returns post info or None."""
    root = dump_ui()
    if root is None:
        return None

    # Extract visible post content
    texts = []
    title = ""
    subreddit = ""

    for node in root.iter("node"):
        node_text = node.get("text", "")
        node_desc = node.get("content-desc", "")
        bounds = node.get("bounds", "")

        if not bounds:
            continue

        m = re.findall(r"\[(\d+),(\d+)\]", bounds)
        if len(m) != 2:
            continue

        y1 = int(m[0][1])
        y2 = int(m[1][1])

        # Only visible area
        if y1 < 200 or y2 > 2200:
            continue

        # Subreddit name (starts with r/)
        if node_text.startswith("r/") and not subreddit:
            subreddit = node_text

        # Title-like: longer text in upper portion
        if len(node_text) > 15 and y1 < 1000 and not title:
            title = node_text

        # Body text
        if len(node_text) > 30:
            texts.append(node_text)

    if not texts and not title:
        return None

    return {
        "title": title,
        "subreddit": subreddit,
        "body": " ".join(texts[:5])[:500],
        "raw_texts": texts,
    }


def extract_visible_post_text() -> str:
    """Extract text from visible feed posts (for relevance checking)."""
    root = dump_ui()
    if root is None:
        return ""

    texts = []
    for node in root.iter("node"):
        node_text = node.get("text", "")
        bounds = node.get("bounds", "")
        if len(node_text) > 20 and bounds:
            m = re.findall(r"\[(\d+),(\d+)\]", bounds)
            if len(m) == 2:
                y1 = int(m[0][1])
                if 200 < y1 < 2000:
                    texts.append(node_text)
    return " ".join(texts[:5])


def upvote_current(dry_run: bool = False) -> bool:
    """Find and tap the upvote button for the current visible post."""
    if dry_run:
        return True

    root = dump_ui()
    # Look for upvote button
    upvote = find_node(root, desc="Upvote") or find_node(root, desc="upvote")
    if upvote:
        sleep_humanized("before_vote")
        adb_tap(*upvote)
        sleep_humanized("after_vote")
        return True

    # Fallback: look for arrow icon buttons
    nodes = find_all_nodes(root, desc="Vote")
    if nodes:
        # First vote button is typically upvote
        sleep_humanized("before_vote")
        adb_tap(nodes[0][0], nodes[0][1])
        sleep_humanized("after_vote")
        return True

    logger.debug("Upvote button not found")
    return False


def type_comment(text: str, dry_run: bool = False) -> bool:
    """Type a comment on the currently open post."""
    if dry_run:
        logger.info(f"[DRY] Would type comment: {text[:60]}...")
        return True

    root = dump_ui()

    # Find comment input area
    comment_input = (
        find_node(root, text="Add a comment") or
        find_node(root, desc="Add a comment") or
        find_node(root, text="comment")
    )

    if comment_input:
        adb_tap(*comment_input)
        sleep_humanized("before_comment")
    else:
        logger.warning("Comment input not found")
        return False

    # Type in chunks with humanized pauses
    words = text.split()
    chunk_size = random.randint(3, 8)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    for chunk in chunks:
        adb_type(chunk + " ")
        sleep_humanized("typing_pause")

    sleep_humanized("after_comment")

    # Find and tap post/submit button
    root = dump_ui()
    post_btn = (
        find_node(root, text="Post") or
        find_node(root, desc="Post") or
        find_node(root, text="Reply") or
        find_node(root, desc="Submit")
    )

    if post_btn:
        adb_tap(*post_btn)
        time.sleep(3)
        logger.info("Comment submitted")
        return True

    logger.warning("Post/Submit button not found for comment")
    return False


def create_post(subreddit: str, title: str, body: str,
                image_path: str = "", dry_run: bool = False) -> bool:
    """Create a new post in a subreddit."""
    if dry_run:
        logger.info(f"[DRY] Would post to r/{subreddit}: {title[:60]}")
        return True

    # Navigate to subreddit first
    if not navigate_to_subreddit(subreddit):
        return False

    sleep_humanized("before_post")

    # Find create post button (+ or pencil icon)
    root = dump_ui()
    create_btn = (
        find_node(root, desc="Create post") or
        find_node(root, desc="Create") or
        find_node(root, desc="New post")
    )

    if not create_btn:
        # Try floating action button
        for node in root.iter("node"):
            bounds = node.get("bounds", "")
            clickable = node.get("clickable", "")
            if clickable == "true" and bounds:
                m = re.findall(r"\[(\d+),(\d+)\]", bounds)
                if len(m) == 2:
                    x1 = int(m[0][0])
                    y1 = int(m[0][1])
                    x2 = int(m[1][0])
                    y2 = int(m[1][1])
                    # FAB at bottom-right
                    if x1 > 800 and y1 > 1900 and (x2 - x1) < 200:
                        create_btn = ((x1 + x2) // 2, (y1 + y2) // 2)
                        break

    if not create_btn:
        logger.error("Create post button not found")
        screenshot("reddit_no_create.png")
        return False

    adb_tap(*create_btn)
    time.sleep(3)

    # Select text post type if prompted
    root = dump_ui()
    text_option = find_node(root, text="Text") or find_node(root, desc="Text")
    if text_option:
        adb_tap(*text_option)
        time.sleep(2)

    # Type title
    root = dump_ui()
    title_field = find_node(root, text="Title") or find_node(root, desc="Title")
    if title_field:
        adb_tap(*title_field)
        time.sleep(0.5)
    adb_type(title)
    sleep_humanized("typing_pause")

    # Type body (tap body area first)
    root = dump_ui()
    body_field = (
        find_node(root, text="body text") or
        find_node(root, text="Body") or
        find_node(root, desc="Optional body text")
    )
    if body_field:
        adb_tap(*body_field)
        time.sleep(0.5)
    else:
        # Tab to next field
        adb_keyevent(61)  # TAB
        time.sleep(0.5)

    # Type body in chunks
    words = body.split()
    chunk_size = random.randint(5, 12)
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        adb_type(chunk + " ")
        sleep_humanized("typing_pause")

    sleep_humanized("before_post")

    # Submit
    root = dump_ui()
    submit_btn = (
        find_node(root, text="Post") or
        find_node(root, desc="Post") or
        find_node(root, text="Submit")
    )
    if submit_btn:
        adb_tap(*submit_btn)
        time.sleep(5)
        logger.info(f"Post submitted to r/{subreddit}: {title[:50]}")
        sleep_humanized("after_post")
        return True

    logger.error("Submit button not found for post")
    screenshot("reddit_no_submit.png")
    return False


def check_inbox() -> str:
    """Open inbox and return visible text for ban detection."""
    root = dump_ui()
    inbox = find_node(root, desc="Inbox") or find_node(root, desc="Notifications")
    if inbox:
        adb_tap(*inbox)
        time.sleep(3)

        root = dump_ui()
        texts = []
        for node in root.iter("node"):
            t = node.get("text", "")
            if len(t) > 10:
                texts.append(t)

        # Go back
        adb_keyevent(4)
        time.sleep(1)
        return " ".join(texts[:20])

    return ""
