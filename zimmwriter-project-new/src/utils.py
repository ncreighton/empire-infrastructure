"""
Shared utilities: clipboard management, logging, retry logic, process helpers.
"""

import os
import sys
import time
import ctypes
import logging
import functools
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(ch)

        # File handler
        log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# CLIPBOARD (Windows-only)
# ─────────────────────────────────────────────

def set_clipboard(text: str):
    """Set text to Windows clipboard. Uses pyperclip (avoids 64-bit GlobalAlloc bug)."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return
    except ImportError:
        pass

    # Fallback to ctypes (may fail on 64-bit Python with 32-bit targets)
    CF_UNICODETEXT = 13
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    user32.OpenClipboard(0)
    user32.EmptyClipboard()

    hMem = kernel32.GlobalAlloc(0x0042, (len(text) + 1) * 2)
    pMem = kernel32.GlobalLock(hMem)
    ctypes.cdll.msvcrt.wcscpy(ctypes.c_wchar_p(pMem), text)
    kernel32.GlobalUnlock(hMem)
    user32.SetClipboardData(CF_UNICODETEXT, hMem)
    user32.CloseClipboard()


def get_clipboard() -> str:
    """Get text from Windows clipboard."""
    try:
        import pyperclip
        return pyperclip.paste()
    except ImportError:
        pass

    CF_UNICODETEXT = 13
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    user32.OpenClipboard(0)
    handle = user32.GetClipboardData(CF_UNICODETEXT)
    if handle:
        pMem = kernel32.GlobalLock(handle)
        text = ctypes.c_wchar_p(pMem).value
        kernel32.GlobalUnlock(handle)
    else:
        text = ""
    user32.CloseClipboard()
    return text


# ─────────────────────────────────────────────
# RETRY LOGIC
# ─────────────────────────────────────────────

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
          exceptions: tuple = (Exception,)):
    """
    Decorator for retrying operations with exponential backoff.

    Usage:
        @retry(max_attempts=3, delay=1.0)
        def flaky_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception
        return wrapper
    return decorator


# ─────────────────────────────────────────────
# PROCESS UTILITIES
# ─────────────────────────────────────────────

def is_zimmwriter_running() -> bool:
    """Check if ZimmWriter (AutoIt3) process is running."""
    try:
        import psutil
        for proc in psutil.process_iter(["name", "exe"]):
            name = (proc.info.get("name") or "").lower()
            if "autoit3" in name:
                return True
    except ImportError:
        # Fallback without psutil
        import subprocess
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq AutoIt3.exe"],
            capture_output=True, text=True
        )
        return "AutoIt3.exe" in result.stdout
    return False


def find_zimmwriter_exe() -> Optional[str]:
    """Search common paths for ZimmWriter installation directory."""
    search_paths = [
        r"D:\zimmwriter",
        r"C:\zimmwriter",
        r"C:\Program Files\ZimmWriter",
        r"C:\Program Files (x86)\ZimmWriter",
        os.path.expanduser(r"~\Desktop\ZimmWriter"),
        os.path.expanduser(r"~\Downloads\ZimmWriter"),
        os.path.expanduser(r"~\AppData\Local\ZimmWriter"),
    ]

    for path in search_paths:
        # Check for AutoIt3.exe launcher (ZimmWriter's actual executable)
        autoit_path = os.path.join(path, "bin", "util", "AutoIt3.exe")
        if os.path.exists(autoit_path):
            return path
        # Also check for direct exe
        for exe in ["ZimmWriter.exe", "zimmwriter.exe"]:
            exe_path = os.path.join(path, exe)
            if os.path.exists(exe_path):
                return path

    return None


# ─────────────────────────────────────────────
# PATH UTILITIES
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIGS_DIR = PROJECT_ROOT / "configs"


def timestamp() -> str:
    """Return ISO timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_output_dir() -> Path:
    """Ensure output directory exists and return it."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR
