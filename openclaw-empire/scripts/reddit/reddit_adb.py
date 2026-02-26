"""ADB helpers for Reddit automation — wraps phone interaction primitives.

Imports shared config from adb_config.py. Adds phone lock file for
mutual exclusion with Substack automation on the same Pixel 8.
"""

import json
import os
import re
import socket
import subprocess
import sys
import time
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# Add scripts/ to path for adb_config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import adb_config

logger = logging.getLogger("reddit_adb")

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "reddit"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOCK_FILE = DATA_DIR / ".phone.lock"
LOCK_TIMEOUT_SECONDS = 1800  # 30 min stale lock


# ---------------------------------------------------------------------------
# Phone lock — mutual exclusion with Substack
# ---------------------------------------------------------------------------

class PhoneLock:
    """File-based lock preventing simultaneous phone use across automations."""

    def __init__(self):
        self._locked = False

    def acquire(self, owner: str = "reddit") -> bool:
        if LOCK_FILE.exists():
            try:
                data = json.loads(LOCK_FILE.read_text())
                lock_time = datetime.fromisoformat(data.get("time", "2000-01-01"))
                age = (datetime.now() - lock_time).total_seconds()
                if age > LOCK_TIMEOUT_SECONDS:
                    logger.warning(f"Stale phone lock ({age:.0f}s old), breaking it")
                    LOCK_FILE.unlink()
                else:
                    logger.info(
                        f"Phone locked by {data.get('owner', '?')} "
                        f"(pid {data.get('pid', '?')}) since {data.get('time', '?')}"
                    )
                    return False
            except Exception:
                LOCK_FILE.unlink()

        LOCK_FILE.write_text(json.dumps({
            "owner": owner,
            "pid": os.getpid(),
            "time": datetime.now().isoformat(),
        }))
        self._locked = True
        return True

    def release(self):
        if self._locked and LOCK_FILE.exists():
            LOCK_FILE.unlink(missing_ok=True)
            self._locked = False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Could not acquire phone lock")
        return self

    def __exit__(self, *args):
        self.release()


# ---------------------------------------------------------------------------
# Core ADB commands
# ---------------------------------------------------------------------------

def adb_shell(cmd: str, timeout: int = 30) -> str:
    """Execute an ADB shell command."""
    full_cmd = [adb_config.ADB, "-s", adb_config.DEVICE, "shell", cmd]
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


def adb_swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 400):
    """Swipe between two points."""
    adb_shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")


def adb_type(text: str, chunk_size: int = 200):
    """Type text via ADB with shell-safe escaping."""
    escaped = text.replace(" ", "%s")
    for ch, repl in [
        ("'", ""), ('"', ""), ("&", "and"), ("(", ""), (")", ""),
        (";", "."), ("!", "."), ("?", "."), ("$", ""),
        ("\n", "%s"), ("\u2014", "-"), ("\u2013", "-"),
        ("\u2018", ""), ("\u2019", ""), ("\u201c", ""), ("\u201d", ""),
    ]:
        escaped = escaped.replace(ch, repl)

    for i in range(0, len(escaped), chunk_size):
        chunk = escaped[i:i + chunk_size]
        adb_shell(f'input text "{chunk}"')
        time.sleep(0.4)


def adb_keyevent(code: int | str):
    """Send a keyevent."""
    adb_shell(f"input keyevent {code}")


# ---------------------------------------------------------------------------
# Screen / unlock
# ---------------------------------------------------------------------------

def ensure_screen_on():
    """Wake the phone screen and unlock with PIN if needed."""
    state = adb_shell("dumpsys power | grep mWakefulness")
    if "Awake" not in state:
        logger.info("Waking screen...")
        adb_shell("input keyevent 224")  # WAKEUP
        time.sleep(1)
        adb_shell("input swipe 540 1800 540 800 300")  # Swipe up
        time.sleep(1)

    # Check lock state
    lock_state = adb_shell("dumpsys trust | grep deviceLocked")
    if "deviceLocked=true" in lock_state and adb_config.PHONE_PIN:
        logger.info("Lock screen detected, entering PIN...")
        adb_shell(f"input text {adb_config.PHONE_PIN}")
        time.sleep(0.5)
        adb_shell("input keyevent 66")  # ENTER
        time.sleep(2)
        # Verify
        kg_after = adb_shell("dumpsys trust | grep deviceLocked")
        if "deviceLocked=true" in kg_after:
            logger.warning("PIN unlock may have failed, retrying...")
            adb_shell("input swipe 540 1800 540 800 300")
            time.sleep(1)
            adb_shell(f"input text {adb_config.PHONE_PIN}")
            time.sleep(0.5)
            adb_shell("input keyevent 66")
            time.sleep(2)


def go_home():
    """Press Android home button."""
    adb_shell("input keyevent 3")
    time.sleep(1)


# ---------------------------------------------------------------------------
# UI dump helpers
# ---------------------------------------------------------------------------

def dump_ui() -> ET.Element | None:
    """Dump and parse the UI hierarchy."""
    adb_shell("uiautomator dump /sdcard/ui.xml")
    time.sleep(1)
    ui_path = str(DATA_DIR / "ui_dump.xml")
    subprocess.run(
        [adb_config.ADB, "-s", adb_config.DEVICE, "pull", "/sdcard/ui.xml", ui_path],
        capture_output=True, timeout=15,
    )
    try:
        tree = ET.parse(ui_path)
        return tree.getroot()
    except Exception as e:
        logger.error(f"UI dump parse failed: {e}")
        return None


def find_node(root, text=None, desc=None) -> tuple[int, int] | None:
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


def find_all_nodes(root, text=None, desc=None, class_name=None):
    """Find all matching UI nodes. Returns list of (cx, cy, text, desc)."""
    results = []
    if root is None:
        return results
    for node in root.iter("node"):
        node_text = node.get("text", "")
        node_desc = node.get("content-desc", "")
        node_class = node.get("class", "")
        bounds = node.get("bounds", "")
        match = False
        if text and text.lower() in node_text.lower():
            match = True
        if desc and desc.lower() in node_desc.lower():
            match = True
        if class_name and class_name.lower() in node_class.lower():
            match = True
        if match and bounds:
            m = re.findall(r"\[(\d+),(\d+)\]", bounds)
            if len(m) == 2:
                cx = (int(m[0][0]) + int(m[1][0])) // 2
                cy = (int(m[0][1]) + int(m[1][1])) // 2
                results.append((cx, cy, node_text, node_desc))
    return results


def screenshot(name: str = "reddit_debug.png") -> str:
    """Take a debug screenshot."""
    path = str(DATA_DIR / name)
    adb_shell("screencap -p /sdcard/screen.png")
    subprocess.run(
        [adb_config.ADB, "-s", adb_config.DEVICE, "pull", "/sdcard/screen.png", path],
        capture_output=True, timeout=15,
    )
    return path


# ---------------------------------------------------------------------------
# ADB connection management
# ---------------------------------------------------------------------------

def adb_try_connect(host: str, port: int) -> bool:
    """Try connecting ADB to a specific host:port."""
    target = f"{host}:{port}"
    try:
        result = subprocess.run(
            [adb_config.ADB, "connect", target],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.strip()
        if "connected" in output.lower() and "cannot" not in output.lower():
            time.sleep(1)
            test = subprocess.run(
                [adb_config.ADB, "-s", target, "shell", "echo ok"],
                capture_output=True, text=True, timeout=10,
            )
            if test.stdout.strip() == "ok":
                return True
    except Exception:
        pass
    return False


def adb_port_scan(host: str) -> int | None:
    """Scan for active ADB wireless debugging port."""
    logger.info(f"Scanning {host} for ADB port...")

    for port in [5555, 5037]:
        try:
            sock = socket.create_connection((host, port), timeout=1)
            sock.close()
            if adb_try_connect(host, port):
                return port
        except (socket.timeout, ConnectionRefusedError, OSError):
            pass

    open_ports = []
    for port in range(37000, 45000):
        try:
            sock = socket.create_connection((host, port), timeout=0.15)
            sock.close()
            open_ports.append(port)
        except (socket.timeout, ConnectionRefusedError, OSError):
            pass

    if open_ports:
        logger.info(f"Open ports: {open_ports}")
        for port in open_ports:
            if adb_try_connect(host, port):
                return port

    return None


def check_adb_connection() -> bool:
    """Verify ADB connection. Auto-reconnects with port scan if needed."""
    if adb_shell("echo ok") == "ok":
        return True

    logger.warning("ADB connection lost, attempting auto-reconnect...")
    host = adb_config.DEVICE.split(":")[0]

    # Attempt 1: same port
    if adb_try_connect(host, int(adb_config.DEVICE.split(":")[1])):
        logger.info("ADB reconnected on same port")
        return True

    # Attempt 2: restart server + retry
    try:
        subprocess.run([adb_config.ADB, "kill-server"], capture_output=True, timeout=10)
        time.sleep(3)
        subprocess.run([adb_config.ADB, "start-server"], capture_output=True, timeout=10)
        time.sleep(2)
    except Exception:
        pass

    if adb_try_connect(host, int(adb_config.DEVICE.split(":")[1])):
        return True

    # Attempt 3: port scan
    new_port = adb_port_scan(host)
    if new_port:
        new_device = f"{host}:{new_port}"
        logger.info(f"Found ADB on port {new_port}! Updating config...")
        adb_config.update_device(new_device)
        return True

    logger.error("ADB auto-reconnect failed")
    return False
