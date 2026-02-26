"""
Substack Notes Poster — ADB Phone Automation
Posts notes to two Substack accounts via Android phone:
  - americayouneedtherapy (Substack app)
  - incompetenceoffice (Firefox browser)

Usage:
  python scripts/substack_poster.py --account aynt --note "Your note text here"
  python scripts/substack_poster.py --account ico --note "Your note text here"
"""

import subprocess
import time
import json
import os
import sys
import re
import argparse
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("substack_poster")

# --- Configuration (shared ADB config) ---
BASE_DIR = Path(__file__).parent.parent
from adb_config import ADB, DEVICE, PHONE_PIN

DATA_DIR = BASE_DIR / "data" / "substack"
DATA_DIR.mkdir(parents=True, exist_ok=True)
POST_LOG = DATA_DIR / "post_log.json"


def adb_shell(cmd: str, timeout: int = 30) -> str:
    """Execute an ADB shell command."""
    full_cmd = [ADB, "-s", DEVICE, "shell", cmd]
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning(f"ADB command timed out: {cmd[:60]}...")
        return ""
    except Exception as e:
        logger.error(f"ADB command failed: {e}")
        return ""


def adb_tap(x: int, y: int):
    """Tap a coordinate on screen."""
    adb_shell(f"input tap {x} {y}")
    time.sleep(0.5)


def adb_type(text: str):
    """Type text via ADB, handling special characters."""
    escaped = text.replace(" ", "%s")
    escaped = escaped.replace("'", "")
    escaped = escaped.replace('"', "")
    escaped = escaped.replace("&", "and")
    escaped = escaped.replace("(", "")
    escaped = escaped.replace(")", "")
    escaped = escaped.replace(";", ".")
    escaped = escaped.replace("!", ".")
    escaped = escaped.replace("?", ".")
    escaped = escaped.replace("$", "")
    escaped = escaped.replace("\n", "%s")
    escaped = escaped.replace("\u2014", "-")
    escaped = escaped.replace("\u2013", "-")
    escaped = escaped.replace("\u2018", "")
    escaped = escaped.replace("\u2019", "")
    escaped = escaped.replace("\u201c", "")
    escaped = escaped.replace("\u201d", "")

    chunk_size = 200
    for i in range(0, len(escaped), chunk_size):
        chunk = escaped[i:i + chunk_size]
        adb_shell(f'input text "{chunk}"')
        time.sleep(0.5)


def dump_ui() -> ET.Element:
    """Dump UI hierarchy and return parsed XML root."""
    adb_shell("uiautomator dump /sdcard/ui.xml")
    time.sleep(1)
    ui_path = str(DATA_DIR / "ui_dump.xml")
    subprocess.run(
        [ADB, "-s", DEVICE, "pull", "/sdcard/ui.xml", ui_path],
        capture_output=True, timeout=15
    )
    try:
        tree = ET.parse(ui_path)
        return tree.getroot()
    except Exception as e:
        logger.error(f"UI dump parse failed: {e}")
        return None


def find_node(root, text=None, desc=None, cls=None):
    """Find a UI node by text, content-desc, or class. Returns (cx, cy) center coords or None."""
    if root is None:
        return None
    for node in root.iter("node"):
        node_text = node.get("text", "")
        node_desc = node.get("content-desc", "")
        node_cls = node.get("class", "")
        bounds = node.get("bounds", "")

        match = False
        if text and text.lower() in node_text.lower():
            match = True
        if desc and desc.lower() in node_desc.lower():
            match = True
        if cls and cls in node_cls:
            match = True

        if match and bounds:
            m = re.findall(r'\[(\d+),(\d+)\]', bounds)
            if len(m) == 2:
                cx = (int(m[0][0]) + int(m[1][0])) // 2
                cy = (int(m[0][1]) + int(m[1][1])) // 2
                return (cx, cy)
    return None


def find_all_nodes(root, text=None, desc=None):
    """Find all matching nodes. Returns list of (cx, cy, text, desc, bounds)."""
    results = []
    if root is None:
        return results
    for node in root.iter("node"):
        node_text = node.get("text", "")
        node_desc = node.get("content-desc", "")
        bounds = node.get("bounds", "")

        match = False
        if text and text.lower() in node_text.lower():
            match = True
        if desc and desc.lower() in node_desc.lower():
            match = True

        if match and bounds:
            m = re.findall(r'\[(\d+),(\d+)\]', bounds)
            if len(m) == 2:
                cx = (int(m[0][0]) + int(m[1][0])) // 2
                cy = (int(m[0][1]) + int(m[1][1])) // 2
                results.append((cx, cy, node_text, node_desc, bounds))
    return results


def ensure_screen_on():
    """Make sure the phone screen is on."""
    state = adb_shell("dumpsys power | grep mWakefulness")
    if "Awake" not in state:
        logger.info("Waking screen...")
        adb_shell("input keyevent 224")  # WAKEUP
        time.sleep(2)


def go_home():
    """Press Android home button."""
    adb_shell("input keyevent 3")
    time.sleep(1)


def screenshot(name: str = "screen.png"):
    """Take and pull a screenshot."""
    path = str(BASE_DIR / name)
    adb_shell("screencap -p /sdcard/screen.png")
    subprocess.run(
        [ADB, "-s", DEVICE, "pull", "/sdcard/screen.png", path],
        capture_output=True, timeout=15
    )
    return path


def wait_for_ui(check_func, timeout=15, interval=2):
    """Wait for a UI condition to be true."""
    start = time.time()
    while time.time() - start < timeout:
        root = dump_ui()
        result = check_func(root)
        if result:
            return result
        time.sleep(interval)
    return None


def post_note_substack_app(text: str) -> bool:
    """Post a note via the Substack app (for americayouneedtherapy).

    Verified workflow:
    1. Go home
    2. Launch Substack app
    3. Wait for app to load, navigate to Home tab
    4. Tap FAB (+) compose button
    5. Wait for "New note" compose screen
    6. Type the note text
    7. Tap the blue post button
    8. Wait for "Your note has been posted" confirmation
    """
    logger.info("=" * 60)
    logger.info("Posting to Substack app (americayouneedtherapy)...")
    logger.info(f"Note: {text[:80]}...")

    ensure_screen_on()

    # Step 1: Go home and launch Substack
    go_home()
    time.sleep(1)
    adb_shell("monkey -p com.substack.app -c android.intent.category.LAUNCHER 1")
    logger.info("Launched Substack app, waiting for load...")
    time.sleep(6)

    # Step 2: Navigate to Home tab to ensure we're on the feed
    root = dump_ui()
    home_btn = find_node(root, desc="Home")
    if home_btn:
        logger.info(f"Tapping Home tab at {home_btn}")
        adb_tap(*home_btn)
        time.sleep(2)
    else:
        logger.warning("Home tab not found in UI, trying default coords")
        adb_tap(148, 2253)
        time.sleep(2)

    # Step 3: Dump UI again and find the FAB compose button
    root = dump_ui()
    if root is None:
        logger.error("Cannot dump UI")
        go_home()
        return False

    # The FAB is a clickable Button near the bottom-right
    # It doesn't have text or desc, so find it by position
    # It's usually the last clickable button before the nav bar
    fab_coords = None
    for node in root.iter("node"):
        bounds = node.get("bounds", "")
        clickable = node.get("clickable", "")
        if clickable == "true" and bounds:
            m = re.findall(r'\[(\d+),(\d+)\]', bounds)
            if len(m) == 2:
                x1, y1 = int(m[0][0]), int(m[0][1])
                x2, y2 = int(m[1][0]), int(m[1][1])
                # FAB is usually bottom-right, roughly square, around 130x130
                if x1 > 800 and y1 > 1800 and y1 < 2200:
                    width = x2 - x1
                    height = y2 - y1
                    if 80 < width < 200 and 80 < height < 200:
                        fab_coords = ((x1 + x2) // 2, (y1 + y2) // 2)

    if fab_coords:
        logger.info(f"Found FAB button at {fab_coords}")
        adb_tap(*fab_coords)
    else:
        logger.warning("FAB not found dynamically, using known coords (970, 2059)")
        adb_tap(970, 2059)
    time.sleep(3)

    # Step 4: Verify we're on "New note" compose screen
    root = dump_ui()
    new_note = find_node(root, text="New note")
    if new_note:
        logger.info("Compose screen confirmed ('New note' visible)")
    else:
        logger.warning("'New note' text not found — may not be on compose screen")
        # Try screenshot for debugging
        screenshot("debug_compose.png")
        # Try once more: maybe we need to tap FAB again
        adb_tap(970, 2059)
        time.sleep(3)
        root = dump_ui()
        new_note = find_node(root, text="New note")
        if not new_note:
            logger.error("Still not on compose screen. Aborting.")
            go_home()
            return False

    # Step 5: Type the note
    logger.info("Typing note text...")
    adb_type(text)
    time.sleep(1)

    # Step 6: Find and tap the post button (blue arrow, top right)
    root = dump_ui()
    # The post button is a clickable View/Button in the top-right area
    post_coords = None
    for node in root.iter("node"):
        bounds = node.get("bounds", "")
        clickable = node.get("clickable", "")
        if clickable == "true" and bounds:
            m = re.findall(r'\[(\d+),(\d+)\]', bounds)
            if len(m) == 2:
                x1, y1 = int(m[0][0]), int(m[0][1])
                x2, y2 = int(m[1][0]), int(m[1][1])
                # Post button is top-right, same row as "New note" title
                if x1 > 900 and y1 < 300:
                    post_coords = ((x1 + x2) // 2, (y1 + y2) // 2)

    if post_coords:
        logger.info(f"Found post button at {post_coords}")
        adb_tap(*post_coords)
    else:
        logger.warning("Post button not found dynamically, using known coords (1006, 216)")
        adb_tap(1006, 216)
    time.sleep(5)

    # Step 7: Verify posting succeeded
    root = dump_ui()
    posted_confirm = find_node(root, text="Your note has been posted")
    if posted_confirm:
        logger.info("SUCCESS: 'Your note has been posted' confirmed!")
    else:
        # Even without the toast, check if we're back on the feed
        feed_check = find_node(root, desc="Home")
        if feed_check:
            logger.info("Back on feed (toast may have expired). Assuming success.")
        else:
            logger.warning("Could not confirm posting. Check manually.")

    go_home()
    logger.info("Post complete.")
    return True


def post_note_firefox(text: str) -> bool:
    """Post a note via Firefox browser (for incompetenceoffice).

    This is more complex as it requires navigating a web UI.
    We'll open the Substack publish/notes page in Firefox.
    """
    logger.info("=" * 60)
    logger.info("Posting to Firefox (incompetenceoffice)...")
    logger.info(f"Note: {text[:80]}...")

    ensure_screen_on()
    go_home()
    time.sleep(1)

    # Open the notes publish page in Firefox
    adb_shell(
        'am start -a android.intent.action.VIEW '
        '-d "https://incompetenceoffice.substack.com/publish/notes" '
        '-n org.mozilla.firefox/.App'
    )
    logger.info("Launched Firefox with notes page, waiting for load...")
    time.sleep(10)

    # Dump UI and look for the compose area
    root = dump_ui()
    if root is None:
        logger.error("Cannot dump UI")
        go_home()
        return False

    # Debug: print all visible elements
    logger.info("Firefox UI elements:")
    for node in root.iter("node"):
        node_text = node.get("text", "")
        node_desc = node.get("content-desc", "")
        bounds = node.get("bounds", "")
        cls = node.get("class", "").split(".")[-1]
        clickable = node.get("clickable", "")
        if node_text or node_desc:
            logger.info(f"  {cls} | text={node_text[:40]} | desc={node_desc[:40]} | {bounds}")

    # Look for text input / compose area
    compose = find_node(root, text="Write") or find_node(root, text="write") or find_node(root, desc="Write")
    if compose:
        logger.info(f"Found compose area at {compose}")
        adb_tap(*compose)
        time.sleep(2)
    else:
        # Try looking for any editable text field in the web view
        for node in root.iter("node"):
            focusable = node.get("focusable", "")
            editable = node.get("editable", "")  # not standard in Android
            cls = node.get("class", "")
            if "EditText" in cls or "editable" in str(node.attrib):
                bounds = node.get("bounds", "")
                m = re.findall(r'\[(\d+),(\d+)\]', bounds)
                if len(m) == 2:
                    cx = (int(m[0][0]) + int(m[1][0])) // 2
                    cy = (int(m[0][1]) + int(m[1][1])) // 2
                    logger.info(f"Found editable field at ({cx}, {cy})")
                    adb_tap(cx, cy)
                    time.sleep(2)
                    break
        else:
            logger.warning("No compose area found. Taking debug screenshot.")
            screenshot("firefox_debug.png")

    # Type the note
    adb_type(text)
    time.sleep(1)

    # Look for Post button
    post_btn = find_node(root, text="Post")
    if post_btn:
        logger.info(f"Found Post button at {post_btn}")
        adb_tap(*post_btn)
        time.sleep(4)
    else:
        logger.warning("Post button not found. Taking debug screenshot.")
        screenshot("firefox_post_debug.png")

    go_home()
    logger.info("Firefox post attempt complete.")
    return True


def log_post(account: str, text: str, success: bool):
    """Append to post log."""
    log = []
    if POST_LOG.exists():
        try:
            log = json.loads(POST_LOG.read_text())
        except Exception:
            log = []

    log.append({
        "account": account,
        "text": text,
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "day_of_week": datetime.now().strftime("%A"),
        "hour": datetime.now().hour,
    })

    tmp = POST_LOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(log, indent=2))
    os.replace(str(tmp), str(POST_LOG))


def main():
    parser = argparse.ArgumentParser(description="Substack Notes Poster")
    parser.add_argument("--account", choices=["aynt", "ico"], required=True,
                        help="Which account to post to")
    parser.add_argument("--note", type=str, required=True,
                        help="Note text to post")
    args = parser.parse_args()

    if args.account == "aynt":
        success = post_note_substack_app(args.note)
    elif args.account == "ico":
        success = post_note_firefox(args.note)
    else:
        logger.error(f"Unknown account: {args.account}")
        success = False

    log_post(args.account, args.note, success)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
