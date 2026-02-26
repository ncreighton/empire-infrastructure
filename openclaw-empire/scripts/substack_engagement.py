"""
Substack Engagement — Organic browsing, liking, restacking, and following.

Simulates human-like engagement on Substack to build organic presence.
AYNT uses the Substack mobile app (ADB), ICO uses Chrome CDP.

Usage:
  python scripts/substack_engagement.py --auto           # Task Scheduler mode
  python scripts/substack_engagement.py --account aynt   # Run one session manually
  python scripts/substack_engagement.py --account ico    # Run ICO session manually
  python scripts/substack_engagement.py --status         # Show engagement stats
  python scripts/substack_engagement.py --dry-run        # Log actions without executing
"""

import json
import os
import sys
import re
import time
import random
import argparse
import logging
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "substack"
ENGAGEMENT_DIR = DATA_DIR / "engagement"
ENGAGEMENT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = DATA_DIR / "engagement.log"
SESSIONS_FILE = ENGAGEMENT_DIR / "sessions_today.json"
STATE_FILE = ENGAGEMENT_DIR / "state.json"
LIMITS_FILE = ENGAGEMENT_DIR / "daily_limits.json"
HISTORY_FILE = ENGAGEMENT_DIR / "engagement_log.json"
NICHE_CONFIG_FILE = ENGAGEMENT_DIR / "niche_config.json"
LOCK_FILE = ENGAGEMENT_DIR / ".engagement.lock"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger("substack_engagement")

# ADB config — shared module
from adb_config import ADB, DEVICE, PHONE_PIN, ANTHROPIC_API_KEY, update_device, ENV_FILE

# Daily action limits per account
DAILY_LIMITS = {
    "likes": 50,
    "follows": 3,
    "restacks": 2,
    "sessions": 3,
}

# Warmup schedule — multipliers by week number (0-indexed from account start)
WARMUP_SCHEDULE = {
    0: {"multiplier": 0.3, "allowed": {"browse", "like"}},
    1: {"multiplier": 0.5, "allowed": {"browse", "like", "restack"}},
    2: {"multiplier": 0.75, "allowed": {"browse", "like", "restack", "follow"}},
    3: {"multiplier": 1.0, "allowed": {"browse", "like", "restack", "follow"}},
}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: Path, default=None):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, Exception):
            pass
    return default if default is not None else {}


def save_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# ADB helpers (imported pattern from substack_scheduler.py)
# ---------------------------------------------------------------------------

def adb_shell(cmd: str, timeout: int = 30) -> str:
    full_cmd = [ADB, "-s", DEVICE, "shell", cmd]
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning(f"ADB timeout: {cmd[:60]}...")
        return ""
    except Exception as e:
        logger.error(f"ADB error: {e}")
        return ""


def adb_tap(x: int, y: int):
    adb_shell(f"input tap {x} {y}")
    time.sleep(0.5)


def ensure_screen_on():
    """Wake the phone screen and unlock with PIN if needed."""
    state = adb_shell("dumpsys power | grep mWakefulness")
    if "Awake" not in state:
        logger.info("Waking screen...")
        adb_shell("input keyevent 224")
        time.sleep(1)
        adb_shell("input swipe 540 1800 540 800 300")
        time.sleep(1)

    # Check if device is still locked (PIN/pattern screen showing)
    lock_state = adb_shell("dumpsys window | grep mDreamingLockscreen")
    showing_lock = "mDreamingLockscreen=true" in lock_state
    if not showing_lock:
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
    adb_shell("input keyevent 3")
    time.sleep(1)


def dump_ui() -> Optional[ET.Element]:
    adb_shell("uiautomator dump /sdcard/ui.xml")
    time.sleep(1)
    ui_path = str(DATA_DIR / "ui_dump_engagement.xml")
    subprocess.run(
        [ADB, "-s", DEVICE, "pull", "/sdcard/ui.xml", ui_path],
        capture_output=True, timeout=15,
    )
    try:
        tree = ET.parse(ui_path)
        return tree.getroot()
    except Exception as e:
        logger.error(f"UI dump parse failed: {e}")
        return None


def find_node(root, text=None, desc=None):
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
    """Find all matching UI nodes. Returns list of (cx, cy, node_text, node_desc)."""
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


def screenshot(name: str = "engagement_debug.png") -> str:
    path = str(ENGAGEMENT_DIR / name)
    adb_shell("screencap -p /sdcard/screen.png")
    subprocess.run(
        [ADB, "-s", DEVICE, "pull", "/sdcard/screen.png", path],
        capture_output=True, timeout=15,
    )
    return path


def adb_try_connect(host: str, port: int) -> bool:
    """Try connecting ADB to a specific host:port."""
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
    """Scan Android wireless debugging port range (37000-44999) to find the active port."""
    import socket

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
        logger.info(f"Open ports found: {open_ports}")
        for port in open_ports:
            if adb_try_connect(host, port):
                return port

    return None


def update_device_env(new_device: str):
    """Update DEVICE in .env so future runs use the new port."""
    import adb_config
    update_device(new_device)
    global DEVICE
    DEVICE = adb_config.DEVICE
    logger.info(f"Updated .env: OPENCLAW_ADB_DEVICE={new_device}")


def check_adb_connection() -> bool:
    """Verify ADB connection is alive. Auto-reconnects with port scan if needed."""
    if adb_shell("echo ok") == "ok":
        return True

    logger.warning("ADB connection lost, attempting auto-reconnect...")
    host = DEVICE.split(":")[0]

    # Attempt 1: same port
    if adb_try_connect(host, int(DEVICE.split(":")[1])):
        logger.info("ADB reconnected on same port")
        return True

    # Attempt 2: restart server, retry
    logger.info("Restarting ADB server...")
    try:
        subprocess.run([ADB, "kill-server"], capture_output=True, timeout=10)
        time.sleep(3)
        subprocess.run([ADB, "start-server"], capture_output=True, timeout=10)
        time.sleep(2)
    except Exception:
        pass

    if adb_try_connect(host, int(DEVICE.split(":")[1])):
        logger.info("ADB reconnected after server restart")
        return True

    # Attempt 3: port scan
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
# HumanDelay — Sync random delays for natural behavior
# ---------------------------------------------------------------------------

class HumanDelay:
    """Random delays to simulate human browsing patterns."""

    @staticmethod
    def between_actions():
        """2-8 seconds between discrete actions."""
        time.sleep(random.uniform(2.0, 8.0))

    @staticmethod
    def reading_dwell():
        """3-15 seconds reading/dwelling on a note."""
        time.sleep(random.uniform(3.0, 15.0))

    @staticmethod
    def burst_rest():
        """30-90 second rest after a burst of actions."""
        rest = random.uniform(30.0, 90.0)
        logger.info(f"Burst rest: {rest:.0f}s")
        time.sleep(rest)

    @staticmethod
    def scroll_pause():
        """0.5-2 seconds between scrolls."""
        time.sleep(random.uniform(0.5, 2.0))

    @staticmethod
    def random_swipe():
        """Execute a random-speed scroll on the phone."""
        # Vary start position, distance, and duration
        start_x = random.randint(400, 680)
        start_y = random.randint(1400, 1800)
        end_y = random.randint(400, 900)
        duration = random.randint(300, 800)  # ms
        adb_shell(f"input swipe {start_x} {start_y} {start_x} {end_y} {duration}")


# ---------------------------------------------------------------------------
# EngagementLimiter — Daily action caps with persistence
# ---------------------------------------------------------------------------

class EngagementLimiter:
    """Track daily action counts per account. Resets at midnight."""

    def __init__(self):
        self._data = load_json(LIMITS_FILE, {"date": "", "accounts": {}})
        self._ensure_today()

    def _ensure_today(self):
        today = date.today().isoformat()
        if self._data.get("date") != today:
            self._data = {"date": today, "accounts": {}}
            self._save()

    def _save(self):
        save_json(LIMITS_FILE, self._data)

    def _get_account(self, account: str) -> dict:
        if account not in self._data["accounts"]:
            self._data["accounts"][account] = {
                "likes": 0, "follows": 0, "restacks": 0, "sessions": 0,
            }
        return self._data["accounts"][account]

    def can_do(self, account: str, action: str, warmup_mult: float = 1.0) -> bool:
        self._ensure_today()
        counts = self._get_account(account)
        base_limit = DAILY_LIMITS.get(action, 999)
        effective_limit = max(1, int(base_limit * warmup_mult))
        return counts.get(action, 0) < effective_limit

    def record(self, account: str, action: str):
        self._ensure_today()
        counts = self._get_account(account)
        counts[action] = counts.get(action, 0) + 1
        self._save()

    def get_counts(self, account: str) -> dict:
        self._ensure_today()
        return dict(self._get_account(account))

    def get_remaining(self, account: str, warmup_mult: float = 1.0) -> dict:
        self._ensure_today()
        counts = self._get_account(account)
        remaining = {}
        for action, base_limit in DAILY_LIMITS.items():
            effective = max(1, int(base_limit * warmup_mult))
            remaining[action] = max(0, effective - counts.get(action, 0))
        return remaining


# ---------------------------------------------------------------------------
# WarmupManager — Gradual ramp over 4 weeks
# ---------------------------------------------------------------------------

class WarmupManager:
    """Track account age and enforce warmup restrictions."""

    def __init__(self):
        self._state = load_json(STATE_FILE, {
            "accounts": {},
            "skip_days": {},
            "cumulative_stats": {},
        })

    def _save(self):
        save_json(STATE_FILE, self._state)

    def get_start_date(self, account: str) -> date:
        accounts = self._state.setdefault("accounts", {})
        if account not in accounts:
            accounts[account] = {"start_date": date.today().isoformat()}
            self._save()
        return date.fromisoformat(accounts[account]["start_date"])

    def get_week(self, account: str) -> int:
        start = self.get_start_date(account)
        days = (date.today() - start).days
        return min(days // 7, 3)  # Cap at week 3 (full cadence)

    def get_warmup(self, account: str) -> dict:
        week = self.get_week(account)
        return WARMUP_SCHEDULE.get(week, WARMUP_SCHEDULE[3])

    def is_action_allowed(self, account: str, action: str) -> bool:
        warmup = self.get_warmup(account)
        return action in warmup["allowed"]

    def get_multiplier(self, account: str) -> float:
        return self.get_warmup(account)["multiplier"]

    def should_skip_today(self, account: str) -> bool:
        """Randomly skip 1-2 days per week for natural patterns."""
        skip_data = self._state.setdefault("skip_days", {})
        today = date.today().isoformat()
        week_key = date.today().strftime("%Y-W%W")

        acct_skips = skip_data.setdefault(account, {})

        # If we already decided for today, return cached decision
        if today in acct_skips:
            return acct_skips[today]

        # Count skips this week
        week_skips = sum(
            1 for d, skipped in acct_skips.items()
            if skipped and d >= (date.today() - timedelta(days=date.today().weekday())).isoformat()
        )

        # Skip 1-2 days per week (random chance decreases if already skipped)
        if week_skips >= 2:
            skip = False
        elif week_skips >= 1:
            skip = random.random() < 0.15  # 15% chance of second skip
        else:
            skip = random.random() < 0.25  # 25% chance of first skip

        acct_skips[today] = skip
        self._save()
        return skip

    def record_session(self, account: str, actions: dict):
        """Record cumulative stats for the account."""
        stats = self._state.setdefault("cumulative_stats", {})
        acct_stats = stats.setdefault(account, {
            "total_sessions": 0, "total_likes": 0,
            "total_follows": 0, "total_restacks": 0,
            "total_scrolls": 0, "first_session": None,
            "last_session": None,
        })
        acct_stats["total_sessions"] += 1
        acct_stats["total_likes"] += actions.get("likes", 0)
        acct_stats["total_follows"] += actions.get("follows", 0)
        acct_stats["total_restacks"] += actions.get("restacks", 0)
        acct_stats["total_scrolls"] += actions.get("scrolls", 0)
        now = datetime.now().isoformat()
        if not acct_stats["first_session"]:
            acct_stats["first_session"] = now
        acct_stats["last_session"] = now
        self._save()


# ---------------------------------------------------------------------------
# NicheRelevanceChecker — Keyword match + Claude Haiku fallback
# ---------------------------------------------------------------------------

class NicheRelevanceChecker:
    """Check if a note is relevant to an account's niche."""

    def __init__(self):
        self._config = load_json(NICHE_CONFIG_FILE, {})

    def is_relevant(self, account: str, note_text: str) -> bool:
        """Check relevance. Keyword match first, AI fallback for ambiguous."""
        config = self._config.get(account, {})
        if not config:
            return True  # No config = allow everything

        text_lower = note_text.lower()

        # Check exclude keywords first
        for kw in config.get("exclude_keywords", []):
            if kw.lower() in text_lower:
                logger.debug(f"Excluded by keyword: {kw}")
                return False

        # Check target authors (always relevant)
        for author in config.get("target_authors", []):
            if author.lower() in text_lower:
                return True

        # Check niche keywords
        keyword_hits = sum(1 for kw in config.get("niche_keywords", []) if kw.lower() in text_lower)
        if keyword_hits >= 2:
            return True
        if keyword_hits == 0:
            return False

        # Ambiguous (1 keyword hit) — use AI if available
        return self._ai_check(account, note_text, config)

    def _ai_check(self, account: str, note_text: str, config: dict) -> bool:
        """Use Claude Haiku for ambiguous relevance decisions."""
        if not ANTHROPIC_API_KEY:
            return False  # Conservative: skip if no API key

        prompt = config.get("ai_relevance_prompt", "")
        if not prompt:
            return False

        try:
            # Import call_anthropic from generate_schedule
            sys.path.insert(0, str(BASE_DIR / "scripts"))
            from generate_schedule import call_anthropic

            result = call_anthropic(prompt, note_text[:500], max_tokens=10)
            return "relevant" in result.lower()
        except Exception as e:
            logger.debug(f"AI relevance check failed: {e}")
            return False  # Conservative fallback


# ---------------------------------------------------------------------------
# EngagementHistory — Action log with cap
# ---------------------------------------------------------------------------

class EngagementHistory:
    """Append-only log of engagement actions, capped at 5000 entries."""

    MAX_ENTRIES = 5000

    def __init__(self):
        self._log = load_json(HISTORY_FILE, [])
        if not isinstance(self._log, list):
            self._log = []

    def record(self, account: str, action: str, target: str = "", details: str = ""):
        self._log.append({
            "timestamp": datetime.now().isoformat(),
            "account": account,
            "action": action,
            "target": target,
            "details": details[:200],
        })
        # Cap at MAX_ENTRIES
        if len(self._log) > self.MAX_ENTRIES:
            self._log = self._log[-self.MAX_ENTRIES:]
        save_json(HISTORY_FILE, self._log)

    def get_recent(self, account: str = None, limit: int = 20) -> list:
        entries = self._log
        if account:
            entries = [e for e in entries if e.get("account") == account]
        return entries[-limit:]


# ---------------------------------------------------------------------------
# Lock file — prevent overlapping sessions
# ---------------------------------------------------------------------------

class SessionLock:
    """File-based lock to prevent concurrent engagement sessions."""

    def __init__(self):
        self._locked = False

    def acquire(self) -> bool:
        if LOCK_FILE.exists():
            # Check if lock is stale (> 25 min old)
            try:
                lock_data = json.loads(LOCK_FILE.read_text())
                lock_time = datetime.fromisoformat(lock_data.get("time", "2000-01-01"))
                if (datetime.now() - lock_time).total_seconds() > 1500:
                    logger.warning("Stale lock detected, removing")
                    LOCK_FILE.unlink()
                else:
                    logger.info(f"Session locked by {lock_data.get('account', '?')} since {lock_data.get('time', '?')}")
                    return False
            except Exception:
                LOCK_FILE.unlink()

        LOCK_FILE.write_text(json.dumps({
            "time": datetime.now().isoformat(),
            "pid": os.getpid(),
            "account": "pending",
        }))
        self._locked = True
        return True

    def update(self, account: str):
        if self._locked:
            LOCK_FILE.write_text(json.dumps({
                "time": datetime.now().isoformat(),
                "pid": os.getpid(),
                "account": account,
            }))

    def release(self):
        if self._locked and LOCK_FILE.exists():
            LOCK_FILE.unlink(missing_ok=True)
            self._locked = False


# ---------------------------------------------------------------------------
# SubstackAppEngagement — AYNT via ADB (Substack mobile app)
# ---------------------------------------------------------------------------

class SubstackAppEngagement:
    """Engage with Substack feed via the mobile app using ADB."""

    def __init__(self, account: str, dry_run: bool = False):
        self.account = account
        self.dry_run = dry_run
        self.limiter = EngagementLimiter()
        self.warmup = WarmupManager()
        self.relevance = NicheRelevanceChecker()
        self.history = EngagementHistory()
        self.actions = {"likes": 0, "follows": 0, "restacks": 0, "scrolls": 0}
        self._consecutive_likes = 0

    def run_session(self) -> dict:
        """Run a complete engagement session via the Substack app."""
        mult = self.warmup.get_multiplier(self.account)
        warmup_info = self.warmup.get_warmup(self.account)
        week = self.warmup.get_week(self.account)
        logger.info(f"Session start: {self.account.upper()} (week {week}, mult={mult}, allowed={warmup_info['allowed']})")

        if not self.dry_run:
            if not check_adb_connection():
                logger.error("ADB connection failed")
                return self.actions
            ensure_screen_on()
            go_home()

            # Launch Substack app
            adb_shell("monkey -p com.substack.app -c android.intent.category.LAUNCHER 1")
            time.sleep(6)

            # Navigate to Home/feed tab
            adb_tap(148, 2253)
            time.sleep(3)

        # Scroll feed and engage
        total_scrolls = random.randint(5, 15)
        notes_seen = 0
        actions_this_burst = 0
        burst_target = random.randint(5, 12)

        # Skip the first note before engaging (don't like the very first thing)
        if not self.dry_run:
            HumanDelay.random_swipe()
            HumanDelay.scroll_pause()
        self.actions["scrolls"] += 1

        for scroll_i in range(total_scrolls):
            # Scroll to reveal new content
            if not self.dry_run:
                HumanDelay.random_swipe()
                HumanDelay.scroll_pause()
            self.actions["scrolls"] += 1

            # Dump UI and look for notes
            if not self.dry_run:
                root = dump_ui()
            else:
                root = None

            # Try to find note content and engagement buttons
            note_text = self._extract_visible_note_text(root)
            if note_text:
                notes_seen += 1

                # Check relevance
                if not self.relevance.is_relevant(self.account, note_text):
                    logger.info(f"  Skipping irrelevant note: {note_text[:50]}...")
                    HumanDelay.between_actions()
                    continue

                # Dwell on the note (reading)
                logger.info(f"  Reading note: {note_text[:60]}...")
                if not self.dry_run:
                    HumanDelay.reading_dwell()

                # Occasional false start — read and back out
                if random.random() < 0.15:
                    logger.info("  False start: read but no action")
                    continue

                # Decide whether to like
                like_chance = random.uniform(0.40, 0.60)
                if (random.random() < like_chance
                        and self.limiter.can_do(self.account, "likes", mult)
                        and self.warmup.is_action_allowed(self.account, "like")):

                    # Anti-detection: don't like 3+ consecutive without scrolling past one
                    if self._consecutive_likes >= 2:
                        logger.info("  Cooldown: skipping like (3 consecutive)")
                        self._consecutive_likes = 0
                        continue

                    self._do_like(root, note_text)
                    actions_this_burst += 1
                else:
                    self._consecutive_likes = 0

                # Rarely restack (5% chance)
                if (random.random() < 0.05
                        and self.limiter.can_do(self.account, "restacks", mult)
                        and self.warmup.is_action_allowed(self.account, "restack")):
                    self._do_restack(root, note_text)
                    actions_this_burst += 1

                # Very rarely follow (2% chance)
                if (random.random() < 0.02
                        and self.limiter.can_do(self.account, "follows", mult)
                        and self.warmup.is_action_allowed(self.account, "follow")):
                    self._do_follow(root, note_text)
                    actions_this_burst += 1

            # Burst rest
            if actions_this_burst >= burst_target:
                if not self.dry_run:
                    HumanDelay.burst_rest()
                actions_this_burst = 0
                burst_target = random.randint(5, 12)

        # Go home
        if not self.dry_run:
            go_home()

        # Record session
        self.limiter.record(self.account, "sessions")
        self.warmup.record_session(self.account, self.actions)
        logger.info(f"Session complete: {self.actions}")
        return self.actions

    def _extract_visible_note_text(self, root) -> str:
        """Extract text content from visible notes in the UI dump."""
        if root is None:
            if self.dry_run:
                # Simulate a note for dry run
                return "Simulated note about politics and democracy for dry run testing"
            return ""

        # Look for text views that contain note content (longer text blocks)
        texts = []
        for node in root.iter("node"):
            node_text = node.get("text", "")
            if len(node_text) > 30:  # Notes are typically 40+ chars
                bounds = node.get("bounds", "")
                if bounds:
                    m = re.findall(r"\[(\d+),(\d+)\]", bounds)
                    if len(m) == 2:
                        y1 = int(m[0][1])
                        y2 = int(m[1][1])
                        # Only visible notes (in middle of screen)
                        if 300 < y1 < 2000 and y2 < 2200:
                            texts.append(node_text)
        return " ".join(texts[:3]) if texts else ""

    def _do_like(self, root, note_text: str):
        """Find and tap the Like button."""
        logger.info(f"  LIKE: {note_text[:50]}...")
        if not self.dry_run:
            # Look for like/heart button in UI
            like_btn = find_node(root, desc="Like") or find_node(root, desc="like")
            if like_btn:
                adb_tap(*like_btn)
            else:
                # Try finding heart icon by looking for small clickable elements
                # near the bottom of the visible note area
                logger.debug("  Like button not found by desc, trying position scan")
                # Fallback: notes typically have like button at bottom-left of note card
        self.actions["likes"] += 1
        self._consecutive_likes += 1
        self.limiter.record(self.account, "likes")
        self.history.record(self.account, "like", note_text[:100])
        if not self.dry_run:
            HumanDelay.between_actions()

    def _do_restack(self, root, note_text: str):
        """Find and tap the Restack button."""
        logger.info(f"  RESTACK: {note_text[:50]}...")
        if not self.dry_run:
            restack_btn = find_node(root, desc="Restack") or find_node(root, desc="restack")
            if restack_btn:
                adb_tap(*restack_btn)
                time.sleep(2)
                # Confirm restack if dialog appears
                root2 = dump_ui()
                confirm = find_node(root2, text="Restack") or find_node(root2, text="restack")
                if confirm:
                    adb_tap(*confirm)
        self.actions["restacks"] += 1
        self.limiter.record(self.account, "restacks")
        self.history.record(self.account, "restack", note_text[:100])
        if not self.dry_run:
            HumanDelay.between_actions()

    def _do_follow(self, root, note_text: str):
        """Find and tap Follow button for the note's author."""
        logger.info(f"  FOLLOW: author of '{note_text[:50]}...'")
        if not self.dry_run:
            follow_btn = find_node(root, text="Follow") or find_node(root, desc="Follow")
            if follow_btn:
                adb_tap(*follow_btn)
        self.actions["follows"] += 1
        self.limiter.record(self.account, "follows")
        self.history.record(self.account, "follow", note_text[:100])
        if not self.dry_run:
            HumanDelay.between_actions()


# ---------------------------------------------------------------------------
# ChromeCDPEngagement — ICO via Chrome DevTools Protocol
# ---------------------------------------------------------------------------

class ChromeCDPEngagement:
    """Engage with Substack feed via Chrome browser using CDP."""

    def __init__(self, account: str, dry_run: bool = False):
        self.account = account
        self.dry_run = dry_run
        self.limiter = EngagementLimiter()
        self.warmup = WarmupManager()
        self.relevance = NicheRelevanceChecker()
        self.history = EngagementHistory()
        self.actions = {"likes": 0, "follows": 0, "restacks": 0, "scrolls": 0}
        self._consecutive_likes = 0

    def run_session(self) -> dict:
        """Run a complete engagement session via Chrome CDP."""
        mult = self.warmup.get_multiplier(self.account)
        warmup_info = self.warmup.get_warmup(self.account)
        week = self.warmup.get_week(self.account)
        logger.info(f"Session start: {self.account.upper()} via CDP (week {week}, mult={mult})")

        if self.dry_run:
            return self._dry_run_session(mult, warmup_info)

        if not check_adb_connection():
            logger.error("ADB connection failed")
            return self.actions

        ensure_screen_on()
        go_home()

        # Open Substack notes feed in Chrome
        feed_url = "https://substack.com/notes"
        adb_shell(
            f'am start -a android.intent.action.VIEW '
            f'-d "{feed_url}" '
            f'-n com.android.chrome/com.google.android.apps.chrome.Main'
        )
        time.sleep(10)

        # Forward CDP port
        CDP_PORT = 9222
        subprocess.run(
            [ADB, "-s", DEVICE, "forward", f"tcp:{CDP_PORT}",
             "localabstract:chrome_devtools_remote"],
            capture_output=True, timeout=10,
        )

        # Connect to Chrome
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json", timeout=5)
            pages = json.loads(resp.read())
        except Exception as e:
            logger.error(f"CDP connection failed: {e}")
            go_home()
            return self.actions

        notes_page = next(
            (p for p in pages if "notes" in p.get("url", "").lower() or "substack" in p.get("url", "").lower()),
            None,
        )
        if not notes_page:
            logger.error("Notes feed page not found in CDP")
            go_home()
            return self.actions

        try:
            import websocket
            ws = websocket.create_connection(
                notes_page["webSocketDebuggerUrl"],
                suppress_origin=True, timeout=10,
            )
        except Exception as e:
            logger.error(f"CDP WebSocket failed: {e}")
            go_home()
            return self.actions

        msg_id = 0

        def cdp(method, params=None):
            nonlocal msg_id
            msg_id += 1
            msg = {"id": msg_id, "method": method}
            if params:
                msg["params"] = params
            ws.send(json.dumps(msg))
            deadline = time.time() + 10
            while time.time() < deadline:
                raw = ws.recv()
                result = json.loads(raw)
                if result.get("id") == msg_id:
                    return result
            return {"error": "timeout"}

        def js(expression):
            r = cdp("Runtime.evaluate", {"expression": expression})
            return r.get("result", {}).get("result", {}).get("value", "")

        try:
            # Wait for feed to load
            for attempt in range(5):
                ready = js("document.querySelectorAll('[class*=feed], [class*=note], article').length > 0")
                if ready:
                    break
                time.sleep(2)

            # Scroll and engage
            total_scrolls = random.randint(5, 15)
            actions_this_burst = 0
            burst_target = random.randint(5, 12)

            # Initial scroll past first content
            js("window.scrollBy(0, 600)")
            time.sleep(random.uniform(1.0, 2.0))
            self.actions["scrolls"] += 1

            for scroll_i in range(total_scrolls):
                # Scroll
                scroll_dist = random.randint(300, 800)
                js(f"window.scrollBy(0, {scroll_dist})")
                time.sleep(random.uniform(0.5, 2.0))
                self.actions["scrolls"] += 1

                # Extract visible note text
                note_text = js("""
                    (function() {
                        var notes = document.querySelectorAll('[class*=feed] [class*=body], [class*=note] p, article p');
                        var texts = [];
                        for (var n of notes) {
                            var rect = n.getBoundingClientRect();
                            if (rect.top > 0 && rect.top < window.innerHeight && n.textContent.length > 30) {
                                texts.push(n.textContent.substring(0, 200));
                            }
                        }
                        return texts.slice(0, 3).join(' ');
                    })()
                """)

                if not note_text or len(note_text) < 30:
                    continue

                if not self.relevance.is_relevant(self.account, note_text):
                    logger.info(f"  Skipping irrelevant: {note_text[:50]}...")
                    continue

                logger.info(f"  Reading: {note_text[:60]}...")
                HumanDelay.reading_dwell()

                # False start
                if random.random() < 0.15:
                    logger.info("  False start")
                    continue

                # Like
                like_chance = random.uniform(0.40, 0.60)
                if (random.random() < like_chance
                        and self.limiter.can_do(self.account, "likes", mult)
                        and self.warmup.is_action_allowed(self.account, "like")):
                    if self._consecutive_likes >= 2:
                        self._consecutive_likes = 0
                        continue
                    self._cdp_like(js, note_text)
                    actions_this_burst += 1
                else:
                    self._consecutive_likes = 0

                # Restack (5%)
                if (random.random() < 0.05
                        and self.limiter.can_do(self.account, "restacks", mult)
                        and self.warmup.is_action_allowed(self.account, "restack")):
                    self._cdp_restack(js, note_text)
                    actions_this_burst += 1

                # Follow (2%)
                if (random.random() < 0.02
                        and self.limiter.can_do(self.account, "follows", mult)
                        and self.warmup.is_action_allowed(self.account, "follow")):
                    self._cdp_follow(js, note_text)
                    actions_this_burst += 1

                # Burst rest
                if actions_this_burst >= burst_target:
                    HumanDelay.burst_rest()
                    actions_this_burst = 0
                    burst_target = random.randint(5, 12)

            ws.close()
        except Exception as e:
            logger.error(f"CDP engagement error: {e}")
            try:
                ws.close()
            except Exception:
                pass

        go_home()
        self.limiter.record(self.account, "sessions")
        self.warmup.record_session(self.account, self.actions)
        logger.info(f"Session complete: {self.actions}")
        return self.actions

    def _dry_run_session(self, mult: float, warmup_info: dict) -> dict:
        """Simulate a session for dry run."""
        total_scrolls = random.randint(5, 15)
        for _ in range(total_scrolls):
            self.actions["scrolls"] += 1
            note_text = "Simulated political note about government accountability and oversight"
            if self.relevance.is_relevant(self.account, note_text):
                if random.random() < 0.5 and self.limiter.can_do(self.account, "likes", mult):
                    logger.info(f"  [DRY] LIKE: {note_text[:50]}...")
                    self.actions["likes"] += 1
                    self.limiter.record(self.account, "likes")
                    self.history.record(self.account, "like", note_text[:100], "dry-run")
                if random.random() < 0.05 and self.limiter.can_do(self.account, "restacks", mult):
                    logger.info(f"  [DRY] RESTACK: {note_text[:50]}...")
                    self.actions["restacks"] += 1
                    self.limiter.record(self.account, "restacks")
                if random.random() < 0.02 and self.limiter.can_do(self.account, "follows", mult):
                    logger.info(f"  [DRY] FOLLOW")
                    self.actions["follows"] += 1
                    self.limiter.record(self.account, "follows")

        self.limiter.record(self.account, "sessions")
        self.warmup.record_session(self.account, self.actions)
        logger.info(f"[DRY RUN] Session complete: {self.actions}")
        return self.actions

    def _cdp_like(self, js, note_text: str):
        """Click a like/heart button via CDP JavaScript."""
        logger.info(f"  LIKE: {note_text[:50]}...")
        result = js("""
            (function() {
                // Find like buttons that are visible and in viewport
                var buttons = document.querySelectorAll('[aria-label*=Like], [aria-label*=like], button[class*=like]');
                for (var b of buttons) {
                    var rect = b.getBoundingClientRect();
                    if (rect.top > 0 && rect.top < window.innerHeight) {
                        b.click();
                        return 'liked';
                    }
                }
                // Fallback: look for heart SVGs
                var hearts = document.querySelectorAll('svg[class*=heart], [class*=heart] svg');
                for (var h of hearts) {
                    var rect = h.getBoundingClientRect();
                    if (rect.top > 0 && rect.top < window.innerHeight) {
                        var btn = h.closest('button') || h.parentElement;
                        if (btn) { btn.click(); return 'liked-heart'; }
                    }
                }
                return 'not-found';
            })()
        """)
        if "liked" in str(result):
            self.actions["likes"] += 1
            self._consecutive_likes += 1
            self.limiter.record(self.account, "likes")
            self.history.record(self.account, "like", note_text[:100])
        else:
            logger.debug(f"  Like button not found: {result}")
            self._consecutive_likes = 0
        HumanDelay.between_actions()

    def _cdp_restack(self, js, note_text: str):
        """Click restack button via CDP."""
        logger.info(f"  RESTACK: {note_text[:50]}...")
        result = js("""
            (function() {
                var buttons = document.querySelectorAll('[aria-label*=Restack], [aria-label*=restack], button[class*=restack]');
                for (var b of buttons) {
                    var rect = b.getBoundingClientRect();
                    if (rect.top > 0 && rect.top < window.innerHeight) {
                        b.click();
                        return 'restacked';
                    }
                }
                return 'not-found';
            })()
        """)
        if "restacked" in str(result):
            self.actions["restacks"] += 1
            self.limiter.record(self.account, "restacks")
            self.history.record(self.account, "restack", note_text[:100])
            # Wait for and confirm restack dialog
            time.sleep(2)
            js("""
                (function() {
                    var btns = document.querySelectorAll('button');
                    for (var b of btns) {
                        if (b.textContent.trim().toLowerCase().includes('restack')) {
                            b.click(); return 'confirmed';
                        }
                    }
                })()
            """)
        HumanDelay.between_actions()

    def _cdp_follow(self, js, note_text: str):
        """Click follow button for note author via CDP."""
        logger.info(f"  FOLLOW: author of '{note_text[:50]}...'")
        result = js("""
            (function() {
                var buttons = document.querySelectorAll('button');
                for (var b of buttons) {
                    var t = b.textContent.trim();
                    if (t === 'Follow' || t === 'Subscribe') {
                        var rect = b.getBoundingClientRect();
                        if (rect.top > 0 && rect.top < window.innerHeight) {
                            b.click();
                            return 'followed';
                        }
                    }
                }
                return 'not-found';
            })()
        """)
        if "followed" in str(result):
            self.actions["follows"] += 1
            self.limiter.record(self.account, "follows")
            self.history.record(self.account, "follow", note_text[:100])
        HumanDelay.between_actions()


# ---------------------------------------------------------------------------
# Session runner — ties everything together
# ---------------------------------------------------------------------------

ACCOUNT_METHODS = {
    "aynt": "app",   # Substack mobile app via ADB
    "ico": "cdp",    # Chrome browser via CDP
}


def run_session(account: str, dry_run: bool = False) -> dict:
    """Run an engagement session for a single account."""
    method = ACCOUNT_METHODS.get(account)
    if not method:
        logger.error(f"Unknown account: {account}")
        return {}

    warmup = WarmupManager()
    limiter = EngagementLimiter()
    mult = warmup.get_multiplier(account)

    # Check if we should skip today
    if warmup.should_skip_today(account):
        logger.info(f"Skipping {account.upper()} today (random skip day)")
        return {}

    # Check session limit
    if not limiter.can_do(account, "sessions", mult):
        logger.info(f"Session limit reached for {account.upper()} today")
        return {}

    # Lock to prevent overlap
    lock = SessionLock()
    if not dry_run and not lock.acquire():
        logger.info("Another session is running, skipping")
        return {}

    try:
        lock.update(account)

        if method == "app":
            eng = SubstackAppEngagement(account, dry_run=dry_run)
        else:
            eng = ChromeCDPEngagement(account, dry_run=dry_run)

        return eng.run_session()
    finally:
        if not dry_run:
            lock.release()


def run_auto():
    """Auto mode: check sessions_today.json and run if a session is due."""
    if not SESSIONS_FILE.exists():
        logger.info("No sessions file. Run generate_schedule.py first.")
        return

    data = load_json(SESSIONS_FILE, {})
    today = date.today().isoformat()

    if data.get("date") != today:
        logger.info("Sessions file is for a different day")
        return

    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute

    # Find sessions due within +-30 min window
    for session in data.get("sessions", []):
        if session.get("completed"):
            continue

        scheduled = session["hour"] * 60 + session["minute"]
        diff = abs(current_minutes - scheduled)

        if diff <= 30:
            account = session["account"]
            logger.info(f"Session due: {account.upper()} at {session['scheduled_time']} (current: {now.strftime('%H:%M')})")

            # Apply +-30 min jitter (randomize actual start)
            jitter = random.randint(0, 5)  # Small additional jitter in minutes
            if jitter > 0:
                logger.info(f"Jitter: waiting {jitter} minutes")
                time.sleep(jitter * 60)

            result = run_session(account)

            # Mark session as completed
            session["completed"] = True
            session["completed_at"] = datetime.now().isoformat()
            session["result"] = result
            save_json(SESSIONS_FILE, data)

            return  # Only run ONE session per invocation

    # No sessions due
    next_session = None
    for session in data.get("sessions", []):
        if not session.get("completed"):
            s_min = session["hour"] * 60 + session["minute"]
            if s_min > current_minutes:
                next_session = session
                break

    if next_session:
        logger.info(f"Next session: {next_session['account'].upper()} at {next_session['scheduled_time']}")
    else:
        completed = sum(1 for s in data.get("sessions", []) if s.get("completed"))
        total = len(data.get("sessions", []))
        logger.info(f"All sessions done for today ({completed}/{total})")


def show_status():
    """Display engagement status and stats."""
    warmup = WarmupManager()
    limiter = EngagementLimiter()
    history = EngagementHistory()

    print("\n=== Substack Engagement Status ===\n")

    for account in ["aynt", "ico"]:
        week = warmup.get_week(account)
        mult = warmup.get_multiplier(account)
        info = warmup.get_warmup(account)
        counts = limiter.get_counts(account)
        remaining = limiter.get_remaining(account, mult)
        skip = warmup.should_skip_today(account)

        name = "AYNT" if account == "aynt" else "ICO"
        method = "App (ADB)" if account == "aynt" else "Chrome (CDP)"

        print(f"  {name} ({method})")
        print(f"    Warmup week: {week} | Multiplier: {mult} | Allowed: {', '.join(sorted(info['allowed']))}")
        print(f"    Today's counts: likes={counts.get('likes', 0)}, follows={counts.get('follows', 0)}, "
              f"restacks={counts.get('restacks', 0)}, sessions={counts.get('sessions', 0)}")
        print(f"    Remaining: likes={remaining.get('likes', 0)}, follows={remaining.get('follows', 0)}, "
              f"restacks={remaining.get('restacks', 0)}, sessions={remaining.get('sessions', 0)}")
        if skip:
            print(f"    ** SKIP DAY (random rest day) **")

        # Cumulative stats
        state = load_json(STATE_FILE, {})
        cum = state.get("cumulative_stats", {}).get(account, {})
        if cum:
            print(f"    Lifetime: {cum.get('total_sessions', 0)} sessions, "
                  f"{cum.get('total_likes', 0)} likes, "
                  f"{cum.get('total_follows', 0)} follows, "
                  f"{cum.get('total_restacks', 0)} restacks")
            if cum.get("first_session"):
                print(f"    First session: {cum['first_session'][:10]}")
        print()

    # Today's sessions
    if SESSIONS_FILE.exists():
        data = load_json(SESSIONS_FILE, {})
        if data.get("date") == date.today().isoformat():
            print("  Today's Sessions:")
            for s in data.get("sessions", []):
                acct = "AYNT" if s["account"] == "aynt" else "ICO"
                status = "DONE" if s.get("completed") else "PENDING"
                completed_at = f" @ {s['completed_at'][11:16]}" if s.get("completed_at") else ""
                print(f"    {s['scheduled_time']} {acct:>4} [{status}]{completed_at}")
            print()

    # Recent activity
    recent = history.get_recent(limit=10)
    if recent:
        print("  Recent Activity (last 10):")
        for entry in recent:
            acct = "AYNT" if entry["account"] == "aynt" else "ICO"
            ts = entry["timestamp"][11:16]
            action = entry["action"].upper()
            target = entry.get("target", "")[:40]
            print(f"    {ts} {acct} {action:>8} | {target}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Substack Engagement — organic browsing and interaction")
    parser.add_argument("--auto", action="store_true", help="Auto mode: check sessions_today.json")
    parser.add_argument("--account", choices=["aynt", "ico"], help="Run a session for a specific account")
    parser.add_argument("--status", action="store_true", help="Show engagement stats")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without executing")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.account:
        logger.info(f"Manual session: {args.account.upper()} (dry_run={args.dry_run})")
        result = run_session(args.account, dry_run=args.dry_run)
        print(f"\nSession result: {result}")
        return

    if args.auto:
        run_auto()
        return

    # Default: show status
    parser.print_help()


if __name__ == "__main__":
    main()
