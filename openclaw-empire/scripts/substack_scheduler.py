"""
Substack Notes Scheduler — Posts one note per invocation.

Called by Windows Task Scheduler every 15 minutes. Each run:
1. Checks if today's schedule exists (generates if not)
2. Finds the next unposted note whose scheduled time has passed
3. Posts it via ADB (Substack app for AYNT, Firefox for ICO)
4. Updates schedule.json and post_log.json
5. Exits

Usage:
  python scripts/substack_scheduler.py           # Normal: post next due note
  python scripts/substack_scheduler.py --status   # Show schedule status
  python scripts/substack_scheduler.py --force    # Post next note regardless of time
"""

import json
import os
import sys
import re
import time
import subprocess
import argparse
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, date
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "substack"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATA_DIR
LOG_FILE = LOG_DIR / "scheduler.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger("substack_scheduler")

# --- Configuration (shared ADB config) ---
from adb_config import ADB, DEVICE, PHONE_PIN, update_device, ENV_FILE

SCHEDULE_FILE = DATA_DIR / "schedule.json"
POST_LOG = DATA_DIR / "post_log.json"

# Cooldown: don't post two notes within this many seconds
MIN_POST_GAP_SECONDS = 120


# ---------------------------------------------------------------------------
# ADB helpers
# ---------------------------------------------------------------------------

def adb_shell(cmd: str, timeout: int = 30) -> str:
    """Execute an ADB shell command."""
    full_cmd = [ADB, "-s", DEVICE, "shell", cmd]
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0 and result.stderr:
            logger.warning(f"ADB stderr: {result.stderr.strip()[:100]}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning(f"ADB timeout: {cmd[:60]}...")
        return ""
    except Exception as e:
        logger.error(f"ADB error: {e}")
        return ""


def adb_tap(x: int, y: int):
    """Tap screen coordinates."""
    adb_shell(f"input tap {x} {y}")
    time.sleep(0.5)


def adb_type(text: str):
    """Type text via ADB, escaping special characters for shell safety."""
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

    # Send in chunks to avoid ADB input length limits
    chunk_size = 200
    for i in range(0, len(escaped), chunk_size):
        chunk = escaped[i : i + chunk_size]
        adb_shell(f'input text "{chunk}"')
        time.sleep(0.4)


def ensure_screen_on():
    """Wake the phone screen and unlock with PIN if needed."""
    state = adb_shell("dumpsys power | grep mWakefulness")
    if "Awake" not in state:
        logger.info("Waking screen...")
        adb_shell("input keyevent 224")  # WAKEUP
        time.sleep(1)
        adb_shell("input swipe 540 1800 540 800 300")  # Swipe up to unlock
        time.sleep(1)

    # Check if device is still locked (PIN/pattern screen showing)
    lock_state = adb_shell("dumpsys window | grep mDreamingLockscreen")
    showing_lock = "mDreamingLockscreen=true" in lock_state
    if not showing_lock:
        # Also check via keyguard
        kg = adb_shell("dumpsys window | grep isStatusBarKeyguard")
        showing_lock = "isStatusBarKeyguard=true" in kg
    if not showing_lock:
        kg2 = adb_shell("dumpsys trust | grep deviceLocked")
        showing_lock = "deviceLocked=true" in kg2

    if showing_lock and PHONE_PIN:
        logger.info("Lock screen detected, entering PIN...")
        adb_shell(f"input text {PHONE_PIN}")
        time.sleep(0.5)
        adb_shell("input keyevent 66")  # ENTER to confirm PIN
        time.sleep(2)
        # Verify unlock succeeded
        kg_after = adb_shell("dumpsys trust | grep deviceLocked")
        if "deviceLocked=true" in kg_after:
            logger.warning("PIN unlock may have failed, retrying swipe + PIN...")
            adb_shell("input swipe 540 1800 540 800 300")
            time.sleep(1)
            adb_shell(f"input text {PHONE_PIN}")
            time.sleep(0.5)
            adb_shell("input keyevent 66")
            time.sleep(2)
        else:
            logger.info("Phone unlocked successfully")


def go_home():
    """Press Android home button."""
    adb_shell("input keyevent 3")
    time.sleep(1)


def dump_ui() -> ET.Element | None:
    """Dump and parse the UI hierarchy."""
    adb_shell("uiautomator dump /sdcard/ui.xml")
    time.sleep(1)
    ui_path = str(DATA_DIR / "ui_dump.xml")
    subprocess.run(
        [ADB, "-s", DEVICE, "pull", "/sdcard/ui.xml", ui_path],
        capture_output=True,
        timeout=15,
    )
    try:
        tree = ET.parse(ui_path)
        return tree.getroot()
    except Exception as e:
        logger.error(f"UI dump parse failed: {e}")
        return None


def find_node(root, text=None, desc=None):
    """Find a UI node by text or content-desc. Returns (cx, cy) or None."""
    if root is None:
        return None
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
            m = re.findall(r"\[(\d+),(\d+)\]", bounds)
            if len(m) == 2:
                cx = (int(m[0][0]) + int(m[1][0])) // 2
                cy = (int(m[0][1]) + int(m[1][1])) // 2
                return (cx, cy)
    return None


def screenshot(name: str = "debug_post.png") -> str:
    """Take a debug screenshot."""
    path = str(DATA_DIR / name)
    adb_shell("screencap -p /sdcard/screen.png")
    subprocess.run(
        [ADB, "-s", DEVICE, "pull", "/sdcard/screen.png", path],
        capture_output=True,
        timeout=15,
    )
    return path


def adb_try_connect(host: str, port: int) -> bool:
    """Try connecting ADB to a specific host:port. Returns True if successful."""
    target = f"{host}:{port}"
    try:
        result = subprocess.run(
            [ADB, "connect", target],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.strip()
        if "connected" in output.lower() and "cannot" not in output.lower():
            time.sleep(1)
            test = subprocess.run(
                [ADB, "-s", target, "shell", "echo ok"],
                capture_output=True, text=True, timeout=10,
            )
            if test.stdout.strip() == "ok":
                return True
    except Exception:
        pass
    return False


def adb_port_scan(host: str) -> int | None:
    """Scan common Android wireless debugging ports to find the active one.

    Android assigns random ports in the 37000-44999 range when wireless
    debugging is toggled. We use a quick TCP socket check before attempting
    a full ADB connect to keep it fast.
    """
    import socket

    # Common ranges for Android wireless debugging
    port_ranges = [(37000, 44999)]
    # Also check well-known ports first
    quick_ports = [5555, 5037]

    logger.info(f"Scanning {host} for ADB port...")

    # Quick ports first
    for port in quick_ports:
        try:
            sock = socket.create_connection((host, port), timeout=1)
            sock.close()
            logger.info(f"Port {port} open, trying ADB connect...")
            if adb_try_connect(host, port):
                return port
        except (socket.timeout, ConnectionRefusedError, OSError):
            pass

    # Scan the wireless debugging range in batches using socket
    open_ports = []
    for range_start, range_end in port_ranges:
        for port in range(range_start, range_end + 1, 1):
            try:
                sock = socket.create_connection((host, port), timeout=0.15)
                sock.close()
                open_ports.append(port)
            except (socket.timeout, ConnectionRefusedError, OSError):
                pass

    if open_ports:
        logger.info(f"Open ports found: {open_ports}")
        for port in open_ports:
            if adb_try_connect(host, port):
                return port

    return None


def update_device_env(new_device: str):
    """Update DEVICE in .env file so future runs use the new port."""
    import adb_config
    update_device(new_device)
    # Keep module-level DEVICE in sync
    global DEVICE
    DEVICE = adb_config.DEVICE
    logger.info(f"Updated .env: OPENCLAW_ADB_DEVICE={new_device}")


def check_adb_connection() -> bool:
    """Verify ADB connection is alive. Auto-reconnects with port scan if needed."""
    # Quick check on current device
    result = adb_shell("echo ok")
    if result == "ok":
        return True

    logger.warning("ADB connection lost, attempting auto-reconnect...")
    host = DEVICE.split(":")[0]

    # Attempt 1: reconnect to same port
    if adb_try_connect(host, int(DEVICE.split(":")[1])):
        logger.info("ADB reconnected on same port")
        return True

    # Attempt 2: restart ADB server, retry same port
    logger.info("Restarting ADB server...")
    try:
        subprocess.run([ADB, "kill-server"], capture_output=True, timeout=10)
        time.sleep(3)
        subprocess.run([ADB, "start-server"], capture_output=True, timeout=10)
        time.sleep(2)
    except Exception as e:
        logger.warning(f"ADB server restart error: {e}")

    if adb_try_connect(host, int(DEVICE.split(":")[1])):
        logger.info("ADB reconnected after server restart")
        return True

    # Attempt 3: port scan — find the new wireless debugging port
    logger.info("Port changed — scanning for new ADB port...")
    new_port = adb_port_scan(host)
    if new_port:
        new_device = f"{host}:{new_port}"
        logger.info(f"Found ADB on port {new_port}! Updating config...")
        update_device_env(new_device)
        return True

    logger.error("ADB auto-reconnect failed: device unreachable or wireless debugging off")
    return False


# ---------------------------------------------------------------------------
# Posting methods
# ---------------------------------------------------------------------------

def post_via_substack_app(text: str) -> bool:
    """Post a note via the Substack mobile app (AYNT account).

    Flow:
    1. Wake screen, go home
    2. Launch Substack app
    3. Find and tap FAB compose button
    4. Verify "New note" compose screen
    5. Type the note
    6. Find and tap the post button (top right)
    7. Verify success
    """
    logger.info("Posting via Substack app...")
    ensure_screen_on()
    go_home()

    # Launch Substack
    adb_shell("monkey -p com.substack.app -c android.intent.category.LAUNCHER 1")
    time.sleep(6)

    # Navigate to Home tab (first nav item — no text/desc so use known coords)
    adb_tap(148, 2253)
    time.sleep(3)

    # Find FAB compose button (orange + button, bottom-right, ABOVE the nav bar)
    # Nav bar is at the very bottom. FAB floats above it.
    # Known good FAB coords: (969, 2058) with bounds roughly [903,1992][1035,2124]
    root = dump_ui()
    if root is None:
        logger.error("Cannot dump UI after launching app")
        go_home()
        return False

    fab_coords = None
    fab_candidates = []
    for node in root.iter("node"):
        bounds = node.get("bounds", "")
        clickable = node.get("clickable", "")
        node_desc = node.get("content-desc", "")
        if clickable == "true" and bounds:
            m = re.findall(r"\[(\d+),(\d+)\]", bounds)
            if len(m) == 2:
                x1, y1 = int(m[0][0]), int(m[0][1])
                x2, y2 = int(m[1][0]), int(m[1][1])
                width = x2 - x1
                height = y2 - y1
                # FAB is roughly square (120-140px), right side, above nav bar
                # Must NOT be in the nav bar row (y2 < 2170 excludes nav items)
                # Must be right-aligned (x1 > 850)
                # Must be square-ish (aspect ratio close to 1:1)
                if (x1 > 850 and 1900 < y1 < 2170 and y2 < 2170
                        and 90 < width < 200 and 90 < height < 200
                        and 0.7 < (width / height) < 1.4):
                    # Exclude nav bar items — they span full width or are small icons
                    # Nav items typically have content-desc like "Home", "Search" etc
                    if node_desc.lower() not in ("home", "search", "activity", "profile", ""):
                        fab_candidates.append(((x1 + x2) // 2, (y1 + y2) // 2, bounds))
                    elif node_desc == "":
                        # Unnamed square button in bottom-right = likely FAB
                        fab_candidates.append(((x1 + x2) // 2, (y1 + y2) // 2, bounds))

    if fab_candidates:
        # Prefer the one closest to known good coords (969, 2058)
        fab_candidates.sort(key=lambda c: abs(c[0] - 969) + abs(c[1] - 2058))
        fab_coords = (fab_candidates[0][0], fab_candidates[0][1])
        logger.info(f"FAB at {fab_coords} (from {len(fab_candidates)} candidates, bounds={fab_candidates[0][2]})")
        adb_tap(*fab_coords)
    else:
        logger.info("FAB not found dynamically, using fallback (969, 2058)")
        adb_tap(969, 2058)
    time.sleep(3)

    # Verify compose screen
    root = dump_ui()
    new_note = find_node(root, text="New note")
    if not new_note:
        logger.warning("Compose screen not detected, retrying FAB tap...")
        adb_tap(970, 2059)
        time.sleep(3)
        root = dump_ui()
        new_note = find_node(root, text="New note")
        if not new_note:
            logger.error("Cannot reach compose screen. Aborting.")
            screenshot("fail_compose.png")
            go_home()
            return False

    logger.info("Compose screen confirmed")

    # Type the note
    adb_type(text)
    time.sleep(1)

    # Find post button (top-right clickable element)
    root = dump_ui()
    post_coords = None
    if root is not None:
        for node in root.iter("node"):
            bounds = node.get("bounds", "")
            clickable = node.get("clickable", "")
            if clickable == "true" and bounds:
                m = re.findall(r"\[(\d+),(\d+)\]", bounds)
                if len(m) == 2:
                    x1, y1 = int(m[0][0]), int(m[0][1])
                    x2, y2 = int(m[1][0]), int(m[1][1])
                    if x1 > 900 and y1 < 300:
                        post_coords = ((x1 + x2) // 2, (y1 + y2) // 2)

    if post_coords:
        logger.info(f"Post button at {post_coords}")
        adb_tap(*post_coords)
    else:
        logger.info("Post button not found, using fallback (1006, 216)")
        adb_tap(1006, 216)
    time.sleep(5)

    # Check for success
    root = dump_ui()
    posted = find_node(root, text="Your note has been posted")
    if posted:
        logger.info("SUCCESS: confirmed 'Your note has been posted'")
    else:
        feed = find_node(root, desc="Home")
        if feed:
            logger.info("Back on feed — assuming success (toast may have expired)")
        else:
            logger.warning("Could not confirm post success")
            screenshot("uncertain_post.png")

    go_home()
    return True


def close_firefox_tabs():
    """Close all Firefox tabs to start fresh."""
    # Open tab switcher
    adb_shell("input keyevent 82")  # MENU key
    time.sleep(1)

    root = dump_ui()
    tabs_btn = find_node(root, desc="Tap to switch tabs") if root else None
    if tabs_btn:
        adb_tap(*tabs_btn)
        time.sleep(1)

        # Look for "Close all tabs" in the menu
        root = dump_ui()
        close_all = find_node(root, text="Close all tabs") if root else None
        if close_all:
            adb_tap(*close_all)
            time.sleep(1)
            # Confirm if dialog appears
            root = dump_ui()
            confirm = find_node(root, text="CLOSE") or find_node(root, text="Close") if root else None
            if confirm:
                adb_tap(*confirm)
                time.sleep(1)
            logger.info("Closed all Firefox tabs")
            return True

    # Fallback: press back to dismiss any menu
    adb_shell("input keyevent 4")
    time.sleep(0.5)
    return False


def post_via_chrome_cdp(text: str) -> bool:
    """Post a note via Chrome browser using Chrome DevTools Protocol (ICO account).

    Flow:
    1. Open Substack dashboard in Chrome
    2. Click "Create new" button
    3. Bottom sheet appears with 6 options — click "New note"
    4. Type note text into the compose area
    5. Click Post button (top right)
    6. Verify success
    """
    import json as _json
    import urllib.request as _urllib
    try:
        import websocket as _ws
    except ImportError:
        logger.error("websocket-client not installed: pip install websocket-client")
        return False

    CDP_PORT = 9222
    DASHBOARD_URL = "https://incompetenceoffice.substack.com/publish"

    logger.info("Posting via Chrome CDP (dashboard -> New note)...")
    ensure_screen_on()
    go_home()

    # Open dashboard in Chrome (preserve session/cookies)
    adb_shell(
        'am start -a android.intent.action.VIEW '
        f'-d "{DASHBOARD_URL}" '
        '-n com.android.chrome/com.google.android.apps.chrome.Main'
    )
    logger.info("Opened Substack dashboard, waiting for page load...")
    time.sleep(10)

    # Forward CDP port
    subprocess.run(
        [ADB, "-s", DEVICE, "forward", f"tcp:{CDP_PORT}",
         "localabstract:chrome_devtools_remote"],
        capture_output=True, timeout=10,
    )

    # Find the dashboard page in CDP
    try:
        resp = _urllib.urlopen(f"http://localhost:{CDP_PORT}/json", timeout=5)
        pages = _json.loads(resp.read())
    except Exception as e:
        logger.error(f"CDP connection failed: {e}")
        go_home()
        return False

    dash_page = next(
        (p for p in pages if "substack.com" in p.get("url", "")),
        None,
    )
    if not dash_page:
        logger.error(f"Substack page not found in CDP. Pages: {[p.get('url','')[:60] for p in pages]}")
        go_home()
        return False

    ws_url = dash_page["webSocketDebuggerUrl"]
    logger.info(f"CDP connected: {dash_page['url'][:60]}")

    # Connect WebSocket
    try:
        ws = _ws.create_connection(ws_url, suppress_origin=True, timeout=15)
    except Exception as e:
        logger.error(f"CDP WebSocket failed: {e}")
        go_home()
        return False

    msg_id = 0

    def cdp(method, params=None):
        nonlocal msg_id
        msg_id += 1
        msg = {"id": msg_id, "method": method}
        if params:
            msg["params"] = params
        ws.send(_json.dumps(msg))
        deadline = time.time() + 15
        while time.time() < deadline:
            raw = ws.recv()
            result = _json.loads(raw)
            if result.get("id") == msg_id:
                return result
        return {"error": "timeout"}

    def js(expression):
        r = cdp("Runtime.evaluate", {"expression": expression})
        return r.get("result", {}).get("result", {}).get("value", "")

    post_clicked = False
    try:
        # Step 1: Wait for dashboard to load
        for attempt in range(8):
            ready = js("document.readyState === 'complete'")
            if ready:
                break
            time.sleep(2)
        time.sleep(3)  # Extra settle time for JS to render

        # Step 2: Click "Create new" button on dashboard
        result = js("""
            (function() {
                // Look for "Create new", "New", "Create", or "Write" button
                var btns = document.querySelectorAll('button, a[role=button], [class*=create], [class*=new]');
                for (var b of btns) {
                    var t = (b.textContent || '').trim().toLowerCase();
                    var aria = (b.getAttribute('aria-label') || '').toLowerCase();
                    if (t.includes('create new') || t.includes('create') || t === 'new'
                        || aria.includes('create') || aria.includes('new post')) {
                        b.click();
                        return 'clicked: ' + (b.textContent || aria).trim().substring(0, 40);
                    }
                }
                // Try looking for a FAB / plus icon button
                var icons = document.querySelectorAll('button svg, button [class*=icon], button [class*=plus]');
                for (var icon of icons) {
                    var btn = icon.closest('button');
                    if (btn) {
                        btn.click();
                        return 'clicked-icon-btn';
                    }
                }
                // Debug: list all visible buttons
                var visible = [];
                for (var b of document.querySelectorAll('button')) {
                    var t = b.textContent.trim();
                    if (t && b.offsetParent !== null) visible.push(t.substring(0, 30));
                }
                return 'not found. buttons: ' + visible.join(', ');
            })()
        """)
        logger.info(f"Create new: {result}")

        if "not found" in str(result):
            logger.error(f"Create new button not found: {result}")
            screenshot("cdp_no_create.png")
            ws.close()
            go_home()
            return False

        # Step 3: Wait for bottom sheet / modal with content type options
        time.sleep(3)

        # Click "New note" from the bottom sheet options
        result = js("""
            (function() {
                // The bottom sheet shows options: Article, Video, Podcast, New note, etc.
                var items = document.querySelectorAll('button, a, [role=menuitem], [role=option], li, [class*=menu] *, [class*=modal] *, [class*=sheet] *, [class*=overlay] *');
                for (var el of items) {
                    var t = (el.textContent || '').trim().toLowerCase();
                    if (t === 'new note' || t === 'note') {
                        el.click();
                        return 'clicked: ' + el.textContent.trim();
                    }
                }
                // Second pass: partial match
                for (var el of items) {
                    var t = (el.textContent || '').trim().toLowerCase();
                    if (t.includes('new note') || t.includes('note')) {
                        el.click();
                        return 'clicked-partial: ' + el.textContent.trim().substring(0, 40);
                    }
                }
                // Debug: list visible text
                var visible = [];
                var all = document.querySelectorAll('button, a, [role=menuitem], li');
                for (var el of all) {
                    var t = el.textContent.trim();
                    if (t && t.length < 40 && el.offsetParent !== null) visible.push(t);
                }
                return 'not found. items: ' + visible.slice(0, 15).join(', ');
            })()
        """)
        logger.info(f"New note: {result}")

        if "not found" in str(result):
            logger.error(f"New note option not found in bottom sheet: {result}")
            screenshot("cdp_no_newnote.png")
            ws.close()
            go_home()
            return False

        # Step 4: Wait for note compose area to appear
        time.sleep(4)

        # Find and type into the note compose editor
        escaped_text = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        result = js(f"""
            (function() {{
                // Try ProseMirror editor (Substack uses this)
                var editor = document.querySelector('.ProseMirror');
                if (editor) {{
                    editor.focus();
                    editor.innerHTML = '<p>{escaped_text}</p>';
                    editor.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    return 'prosemirror:' + editor.innerText.substring(0, 30);
                }}
                // Try contenteditable div
                var editable = document.querySelector('[contenteditable=true]');
                if (editable) {{
                    editable.focus();
                    editable.innerHTML = '<p>{escaped_text}</p>';
                    editable.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    return 'contenteditable:' + editable.innerText.substring(0, 30);
                }}
                // Try textarea
                var ta = document.querySelector('textarea');
                if (ta) {{
                    ta.focus();
                    ta.value = '{escaped_text}';
                    ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    ta.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return 'textarea:' + ta.value.substring(0, 30);
                }}
                return 'no editor found';
            }})()
        """)
        logger.info(f"Text entry: {result}")

        if "no editor" in str(result):
            logger.error(f"Note compose editor not found: {result}")
            screenshot("cdp_no_note_editor.png")
            ws.close()
            go_home()
            return False

        time.sleep(2)

        # Step 5: Click the Post button (top right)
        result = js("""
            (function() {
                var btns = document.querySelectorAll('button');
                for (var b of btns) {
                    var t = b.textContent.trim().toLowerCase();
                    if (t === 'post' || t === 'publish' || t === 'send' || t === 'post note') {
                        b.click();
                        return 'posted: ' + b.textContent.trim();
                    }
                }
                // Debug: list buttons
                var visible = [];
                for (var b of btns) {
                    var t = b.textContent.trim();
                    if (t && b.offsetParent !== null) visible.push(t.substring(0, 30));
                }
                return 'not found. buttons: ' + visible.join(', ');
            })()
        """)
        logger.info(f"Post button: {result}")

        if "not found" in str(result):
            logger.error(f"Post button not found: {result}")
            screenshot("cdp_no_post_btn.png")
            ws.close()
            go_home()
            return False

        post_clicked = True
        time.sleep(5)

        # Step 6: Verify success
        try:
            # Check if note compose area is gone (posted successfully)
            success_check = js("""
                (function() {
                    var editor = document.querySelector('.ProseMirror') || document.querySelector('[contenteditable=true]');
                    if (!editor) return 'editor-gone';
                    if (editor.innerText.trim().length === 0) return 'editor-empty';
                    return 'editor-still-has-text';
                })()
            """)
            logger.info(f"Post verify: {success_check}")

            if success_check in ('editor-gone', 'editor-empty'):
                logger.info("SUCCESS: note posted (editor cleared/gone)")
                ws.close()
                go_home()
                return True

            # Check for success toast/message
            toast = js("""
                document.body.innerText.includes('posted') ||
                document.body.innerText.includes('Published') ||
                document.body.innerText.includes('Your note')
            """)
            if toast == "true":
                logger.info("SUCCESS: post confirmed via page content")
                ws.close()
                go_home()
                return True

            logger.warning("Could not confirm post success via CDP")
            screenshot("cdp_post_uncertain.png")
            ws.close()
            go_home()
            # Post button was clicked, likely succeeded
            return True

        except Exception as verify_err:
            # WebSocket died during verification — page likely changed after posting
            logger.info(f"CDP connection lost during verification (expected after post): {verify_err}")
            logger.info("SUCCESS: post button was clicked, treating as success")
            try:
                ws.close()
            except Exception:
                pass
            go_home()
            return True

    except Exception as e:
        if post_clicked:
            logger.info(f"CDP connection lost after post click (expected): {e}")
            logger.info("SUCCESS: treating post-click disconnect as success")
            try:
                ws.close()
            except Exception:
                pass
            go_home()
            return True
        logger.error(f"CDP error: {e}")
        screenshot("cdp_error.png")
        try:
            ws.close()
        except Exception:
            pass
        go_home()
        return False


# Keep post_via_firefox as alias that delegates to Chrome CDP
def post_via_firefox(text: str) -> bool:
    """Post a note via browser (ICO account). Uses Chrome CDP for reliability."""
    return post_via_chrome_cdp(text)


# ---------------------------------------------------------------------------
# Schedule management
# ---------------------------------------------------------------------------

def load_schedule() -> list:
    """Load posting schedule from disk."""
    if SCHEDULE_FILE.exists():
        try:
            return json.loads(SCHEDULE_FILE.read_text())
        except (json.JSONDecodeError, Exception):
            return []
    return []


def save_schedule(schedule: list):
    """Save schedule atomically."""
    tmp = SCHEDULE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(schedule, indent=2))
    os.replace(str(tmp), str(SCHEDULE_FILE))


def log_post(account: str, text: str, success: bool):
    """Append to post_log.json."""
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


def ensure_today_schedule():
    """If no schedule exists for today, generate one."""
    today_str = date.today().isoformat()
    schedule = load_schedule()

    if schedule and schedule[0].get("date") == today_str:
        return  # Already have today's schedule

    logger.info("No schedule for today. Generating...")
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "scripts" / "generate_schedule.py"), "--reset"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(BASE_DIR),
        )
        if result.returncode == 0:
            logger.info("Schedule generated successfully")
        else:
            logger.error(f"Schedule generation failed: {result.stderr[:200]}")
    except Exception as e:
        logger.error(f"Could not generate schedule: {e}")


def get_last_post_time() -> datetime | None:
    """Get the timestamp of the most recent post from the log."""
    if not POST_LOG.exists():
        return None
    try:
        log = json.loads(POST_LOG.read_text())
        if log:
            return datetime.fromisoformat(log[-1]["timestamp"])
    except Exception:
        pass
    return None


def run_scheduled_post(force: bool = False):
    """Find and post the next due note."""
    # Ensure we have a schedule for today
    ensure_today_schedule()

    schedule = load_schedule()
    if not schedule:
        logger.info("No notes scheduled. Nothing to do.")
        return

    today_str = date.today().isoformat()
    if schedule[0].get("date") != today_str:
        logger.info("Schedule is for a different day. Waiting for regeneration.")
        return

    # Check ADB connection
    if not check_adb_connection():
        logger.error("ADB connection failed. Cannot post.")
        return

    # Check cooldown
    if not force:
        last_post = get_last_post_time()
        if last_post:
            elapsed = (datetime.now() - last_post).total_seconds()
            if elapsed < MIN_POST_GAP_SECONDS:
                logger.info(f"Cooldown: {MIN_POST_GAP_SECONDS - elapsed:.0f}s remaining since last post")
                return

    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute

    # Find next unposted note whose time has passed
    for entry in schedule:
        if entry.get("posted"):
            continue

        scheduled_minutes = entry.get("hour", 0) * 60 + entry.get("minute", 0)

        if force or current_minutes >= scheduled_minutes:
            account = entry["account"]
            method = entry.get("method", "app")
            text = entry["text"]

            logger.info("=" * 60)
            logger.info(f"Posting for {account.upper()} (scheduled {entry.get('scheduled_time', '??:??')})")
            logger.info(f"Method: {method}")
            logger.info(f"Note: {text[:80]}...")

            # Post via appropriate method
            if method == "app":
                success = post_via_substack_app(text)
            elif method == "firefox":
                success = post_via_firefox(text)
            else:
                logger.error(f"Unknown method: {method}")
                continue

            # Update schedule entry
            entry["posted"] = True
            entry["posted_at"] = now.isoformat()
            entry["success"] = success
            entry["attempt_count"] = entry.get("attempt_count", 0) + 1
            save_schedule(schedule)

            # Log the post
            log_post(account, text, success)

            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Result: {status}")
            logger.info("=" * 60)

            return  # Only post ONE note per invocation

    # All done for today
    pending = sum(1 for e in schedule if not e.get("posted"))
    posted = sum(1 for e in schedule if e.get("posted"))
    if pending == 0:
        logger.info(f"All {posted} notes posted for today. Done.")
    else:
        next_entry = next((e for e in schedule if not e.get("posted")), None)
        if next_entry:
            logger.info(
                f"Next note at {next_entry.get('scheduled_time', '??:??')} "
                f"for {next_entry['account'].upper()}. "
                f"({posted} posted, {pending} remaining)"
            )


def show_status():
    """Print current schedule status."""
    schedule = load_schedule()
    if not schedule:
        print("No schedule loaded.")
        return

    print(f"\nSchedule for {schedule[0].get('date', 'unknown')}:")
    print(f"{'Time':>5} | {'Account':>6} | {'Method':>7} | {'Status':>8} | Note")
    print("-" * 90)

    for entry in schedule:
        status = "POSTED" if entry.get("posted") else "PENDING"
        time_str = entry.get("scheduled_time", f"{entry.get('hour', 0):02d}:00")
        account = "AYNT" if entry["account"] == "aynt" else "ICO"
        method = entry.get("method", "app")
        text = entry["text"][:50] + "..."
        extra = ""
        if entry.get("posted_at"):
            extra = f" @ {entry['posted_at'][11:16]}"
        if entry.get("posted") and not entry.get("success"):
            extra += " FAIL"
        print(f"{time_str:>5} | {account:>6} | {method:>7} | {status:>8}{extra} | {text}")

    pending = sum(1 for e in schedule if not e.get("posted"))
    posted = sum(1 for e in schedule if e.get("posted"))
    failed = sum(1 for e in schedule if e.get("posted") and not e.get("success"))
    print(f"\nPosted: {posted} | Pending: {pending} | Failed: {failed} | Total: {len(schedule)}")


def retry_failed():
    """Retry all failed posts from today's schedule."""
    schedule = load_schedule()
    if not schedule:
        print("No schedule loaded.")
        return

    failed = [e for e in schedule if e.get("posted") and not e.get("success")]
    if not failed:
        print("No failed posts to retry.")
        return

    if not check_adb_connection():
        logger.error("ADB connection failed. Cannot retry.")
        return

    print(f"Retrying {len(failed)} failed posts...")

    for entry in failed:
        account = entry["account"]
        method = entry.get("method", "app")
        text = entry["text"]

        logger.info("=" * 60)
        logger.info(f"RETRY: {account.upper()} (originally {entry.get('scheduled_time', '??:??')})")
        logger.info(f"Method: {method}")
        logger.info(f"Note: {text[:80]}...")

        if method == "app":
            success = post_via_substack_app(text)
        elif method == "firefox":
            success = post_via_firefox(text)
        else:
            continue

        entry["success"] = success
        entry["attempt_count"] = entry.get("attempt_count", 0) + 1
        entry["posted_at"] = datetime.now().isoformat()
        save_schedule(schedule)

        log_post(account, text, success)

        status = "SUCCESS" if success else "STILL FAILED"
        logger.info(f"Retry result: {status}")
        logger.info("=" * 60)

        # Gap between retries
        time.sleep(MIN_POST_GAP_SECONDS)


def post_all_pending(force: bool = False):
    """Post all pending notes from today's schedule with gaps between each."""
    ensure_today_schedule()
    schedule = load_schedule()
    if not schedule:
        logger.info("No schedule loaded.")
        return

    today_str = date.today().isoformat()
    if schedule[0].get("date") != today_str:
        logger.info("Schedule is for a different day.")
        return

    if not check_adb_connection():
        logger.error("ADB connection failed.")
        return

    pending = [e for e in schedule if not e.get("posted")]
    if not pending:
        logger.info("All notes already posted.")
        return

    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute

    # Filter to only due notes (unless force)
    if not force:
        pending = [e for e in pending
                   if e.get("hour", 0) * 60 + e.get("minute", 0) <= current_minutes]
        if not pending:
            logger.info("No due notes to post.")
            return

    logger.info(f"Posting {len(pending)} notes with {MIN_POST_GAP_SECONDS}s gaps...")

    for i, entry in enumerate(pending):
        account = entry["account"]
        method = entry.get("method", "app")
        text = entry["text"]

        logger.info(f"\n[{i+1}/{len(pending)}] Posting for {account.upper()} ({entry.get('scheduled_time', '??:??')})")

        if method == "app":
            success = post_via_substack_app(text)
        elif method == "firefox":
            success = post_via_firefox(text)
        else:
            continue

        entry["posted"] = True
        entry["posted_at"] = datetime.now().isoformat()
        entry["success"] = success
        entry["attempt_count"] = entry.get("attempt_count", 0) + 1
        save_schedule(schedule)
        log_post(account, text, success)

        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Result: {status}")

        # Wait between posts (skip after last one)
        if i < len(pending) - 1:
            logger.info(f"Waiting {MIN_POST_GAP_SECONDS}s before next post...")
            time.sleep(MIN_POST_GAP_SECONDS)

    posted = sum(1 for e in schedule if e.get("posted"))
    succeeded = sum(1 for e in schedule if e.get("posted") and e.get("success"))
    failed = posted - succeeded
    logger.info(f"\nDone. Total posted: {posted} (succeeded: {succeeded}, failed: {failed})")


def main():
    parser = argparse.ArgumentParser(description="Substack Notes Scheduler")
    parser.add_argument("--status", action="store_true", help="Show schedule status")
    parser.add_argument("--force", action="store_true", help="Post next note regardless of time")
    parser.add_argument("--retry-failed", action="store_true", help="Retry all failed posts")
    parser.add_argument("--post-all", action="store_true", help="Post all pending due notes in sequence")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.retry_failed:
        retry_failed()
        return

    if args.post_all:
        post_all_pending(force=args.force)
        return

    run_scheduled_post(force=args.force)


if __name__ == "__main__":
    main()
