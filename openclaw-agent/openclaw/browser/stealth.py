"""Stealth configuration — anti-detection args, UA rotation, viewport randomization."""

from __future__ import annotations

import random

# Realistic user agents (Chrome on Windows/Mac, updated periodically)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Common screen resolutions
VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1680, "height": 1050},
    {"width": 2560, "height": 1440},
    {"width": 1280, "height": 720},
]

# Chromium args for anti-detection
STEALTH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-popup-blocking",
    "--disable-extensions",
    "--disable-component-update",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-hang-monitor",
    "--disable-ipc-flooding-protection",
    "--disable-client-side-phishing-detection",
    "--password-store=basic",
    "--use-mock-keychain",
]

# Additional args for headless mode
HEADLESS_ARGS = [
    "--headless=new",
    "--disable-gpu",
    "--window-size=1920,1080",
]


def get_random_user_agent() -> str:
    """Return a random realistic user agent string."""
    return random.choice(USER_AGENTS)


def get_random_viewport() -> dict[str, int]:
    """Return a random common viewport size."""
    return random.choice(VIEWPORTS).copy()


def get_stealth_args(headless: bool = True) -> list[str]:
    """Get Chromium launch args for stealth operation."""
    args = STEALTH_ARGS.copy()
    if headless:
        args.extend(HEADLESS_ARGS)
    return args


def get_browser_config(headless: bool = True) -> dict:
    """Get complete browser configuration for browser-use."""
    viewport = get_random_viewport()
    return {
        "headless": headless,
        "args": get_stealth_args(headless),
        "user_agent": get_random_user_agent(),
        "viewport": viewport,
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "color_scheme": "light",
        "extra_http_headers": {
            "Accept-Language": "en-US,en;q=0.9",
        },
    }


def add_human_delays() -> dict[str, tuple[float, float]]:
    """Return randomized delay ranges to simulate human behavior."""
    return {
        "typing_delay": (0.05, 0.15),       # seconds between keystrokes
        "click_delay": (0.3, 1.2),           # seconds before clicking
        "page_load_wait": (1.0, 3.0),        # seconds after page load
        "form_field_pause": (0.5, 2.0),      # seconds between form fields
        "scroll_pause": (0.3, 0.8),          # seconds during scroll
        "submit_pause": (1.0, 2.5),          # seconds before submit
    }


def randomize_delay(delay_range: tuple[float, float]) -> float:
    """Get a random delay within range."""
    return random.uniform(delay_range[0], delay_range[1])
