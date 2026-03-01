"""ADB connection monitor — keeps phone connected for all automations.

Runs every 5 minutes via Task Scheduler. Self-heals connection drops by:
1. Checking if the fixed port (5555) is reachable
2. If not, scanning for wireless debugging ports (34000-50000)
3. If found, connecting and re-establishing fixed port via `adb tcpip 5555`
4. Updating .env so all scripts use the correct device address

This eliminates the need to manually re-pair after port changes.
"""

import logging
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import adb_config

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = DATA_DIR / "adb_monitor.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger("adb_monitor")

PHONE_IP = "100.79.124.62"
FIXED_PORT = 5555
SCAN_RANGE = (34000, 50001)


def run_adb(*args, timeout=10) -> str:
    """Run an ADB command, return stdout."""
    cmd = [adb_config.ADB] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception as e:
        logger.debug(f"ADB command failed: {e}")
        return ""


def is_device_connected() -> bool:
    """Check if any device is connected and responsive."""
    output = run_adb("devices")
    if f"{PHONE_IP}" in output and "device" in output and "offline" not in output:
        # Verify it actually responds
        test = run_adb("-s", adb_config.DEVICE, "shell", "echo ok")
        return test.strip() == "ok"
    return False


def is_host_reachable() -> bool:
    """Check if the phone IP is reachable at all (Tailscale up)."""
    try:
        result = subprocess.run(
            ["ping", "-n", "1", "-w", "2000", PHONE_IP],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_port(port: int) -> int | None:
    """Check if a single TCP port is open."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        result = s.connect_ex((PHONE_IP, port))
        s.close()
        return port if result == 0 else None
    except Exception:
        return None


def scan_for_adb_port() -> int | None:
    """Scan for wireless debugging port on the phone."""
    # Try fixed port first
    if check_port(FIXED_PORT):
        return FIXED_PORT

    # Scan wireless debugging range
    logger.info(f"Scanning {PHONE_IP} ports {SCAN_RANGE[0]}-{SCAN_RANGE[1]}...")
    with ThreadPoolExecutor(max_workers=200) as ex:
        futures = {ex.submit(check_port, p): p for p in range(*SCAN_RANGE)}
        for f in as_completed(futures):
            port = f.result()
            if port:
                return port
    return None


def try_connect(port: int) -> bool:
    """Try to connect ADB to the given port."""
    target = f"{PHONE_IP}:{port}"
    output = run_adb("connect", target)
    if "connected" in output.lower() and "cannot" not in output.lower():
        time.sleep(1)
        test = run_adb("-s", target, "shell", "echo ok")
        return test.strip() == "ok"
    return False


def setup_fixed_port(connected_port: int) -> bool:
    """Once connected on any port, set up fixed port 5555 via adb tcpip."""
    if connected_port == FIXED_PORT:
        logger.info(f"Already on fixed port {FIXED_PORT}")
        return True

    target = f"{PHONE_IP}:{connected_port}"
    logger.info(f"Setting up fixed port {FIXED_PORT} via adb tcpip...")

    # Run adb tcpip 5555 — this restarts adbd on the phone to listen on 5555
    output = run_adb("-s", target, "tcpip", str(FIXED_PORT), timeout=15)
    logger.info(f"tcpip output: {output}")

    # The current connection will drop, wait for phone to restart adbd
    time.sleep(5)

    # Connect to the new fixed port
    run_adb("disconnect", target)
    time.sleep(1)

    if try_connect(FIXED_PORT):
        new_device = f"{PHONE_IP}:{FIXED_PORT}"
        adb_config.update_device(new_device)
        logger.info(f"Fixed port {FIXED_PORT} established and saved to .env")
        return True

    logger.error(f"Failed to connect on fixed port {FIXED_PORT} after tcpip")
    return False


def main():
    """Main monitor loop — check and heal ADB connection."""
    logger.info("ADB monitor check starting...")

    # Step 1: Already connected?
    if is_device_connected():
        # Verify we're on the fixed port
        if f":{FIXED_PORT}" in adb_config.DEVICE:
            logger.info(f"Connected on fixed port {FIXED_PORT} — all good")
            return
        else:
            # Connected but not on fixed port — fix it
            current_port = int(adb_config.DEVICE.split(":")[1])
            logger.info(f"Connected on port {current_port}, migrating to {FIXED_PORT}")
            setup_fixed_port(current_port)
            return

    # Step 2: Phone reachable?
    if not is_host_reachable():
        logger.warning(f"Phone {PHONE_IP} unreachable (Tailscale down or phone off)")
        return

    # Step 3: Try fixed port first
    logger.info("Connection lost. Attempting reconnect...")

    # Kill stale server state
    run_adb("kill-server")
    time.sleep(2)
    run_adb("start-server")
    time.sleep(2)

    if check_port(FIXED_PORT) and try_connect(FIXED_PORT):
        new_device = f"{PHONE_IP}:{FIXED_PORT}"
        adb_config.update_device(new_device)
        logger.info(f"Reconnected on fixed port {FIXED_PORT}")
        return

    # Step 4: Scan for wireless debugging port
    found_port = scan_for_adb_port()
    if found_port:
        logger.info(f"Found ADB on port {found_port}")
        if try_connect(found_port):
            # Re-establish fixed port
            setup_fixed_port(found_port)
            return
        else:
            logger.error(f"Port {found_port} open but ADB connect failed")
    else:
        logger.warning("No ADB ports found. Wireless debugging may be off on phone.")

    logger.error("Could not restore ADB connection")


if __name__ == "__main__":
    main()
