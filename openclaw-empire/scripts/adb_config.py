"""Shared ADB configuration — single source of truth for device connection.

All scripts that interact with the phone via ADB should import from here
instead of duplicating .env parsing logic.

Usage:
    from adb_config import ADB, DEVICE, PHONE_PIN, ANTHROPIC_API_KEY
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
ENV_FILE = BASE_DIR / ".env"

# Defaults
ADB = r"C:\Users\ncreighton\AppData\Local\Android\Sdk\platform-tools\adb.exe"
DEVICE = "100.79.124.62:34647"
PHONE_PIN = ""
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Parse .env once at import time
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "OPENCLAW_ADB_DEVICE":
            DEVICE = value
        elif key == "PHONE_PIN":
            PHONE_PIN = value
        elif key == "ADB_PATH" and value:
            ADB = value
        elif key == "ANTHROPIC_API_KEY" and not ANTHROPIC_API_KEY:
            ANTHROPIC_API_KEY = value


def update_device(new_device: str):
    """Update DEVICE in .env and in this module's global."""
    global DEVICE
    DEVICE = new_device
    if ENV_FILE.exists():
        content = ENV_FILE.read_text()
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("OPENCLAW_ADB_DEVICE=") and not line.strip().startswith("#"):
                lines[i] = f"OPENCLAW_ADB_DEVICE={new_device}"
                break
        ENV_FILE.write_text("\n".join(lines) + "\n")
