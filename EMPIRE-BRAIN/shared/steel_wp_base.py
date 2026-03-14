"""
Shared Steel.dev Browser Automation Base for all Empire WordPress sites.

Replaces 16 identical per-site steel_automation.py files.
Loads credentials from config/sites.json — no hardcoded creds.

Usage:
    from EMPIRE_BRAIN.shared.steel_wp_base import SteelWPBase

    # From sites.json config
    auto = SteelWPBase("witchcraftforbeginners")

    # As context manager
    with SteelWPBase("smarthomewizards") as auto:
        auto.login_wordpress()
        auto.take_screenshot("dashboard.png")
"""

import json
import os
from pathlib import Path

# Config paths (resolve relative to this file's location)
_SHARED_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SHARED_DIR.parent.parent / "config" / "sites.json"


def _load_site_config(site_id):
    """Load site config from config/sites.json."""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Sites config not found: {_CONFIG_PATH}")
    with open(_CONFIG_PATH) as f:
        data = json.load(f)
    sites = data.get("sites", data)
    if site_id not in sites:
        raise KeyError(
            f"Site '{site_id}' not found in {_CONFIG_PATH}. "
            f"Available: {', '.join(sorted(sites.keys()))}"
        )
    return sites[site_id]


class SteelWPBase:
    """Steel.dev browser automation for any Empire WordPress site."""

    def __init__(self, site_id, *, site_url=None, wp_username=None):
        """
        Args:
            site_id: Key in config/sites.json (e.g. "witchcraftforbeginners").
            site_url: Override site URL (skips sites.json lookup for this field).
            wp_username: Override WP username (skips sites.json lookup for this field).
        """
        config = _load_site_config(site_id)
        self.site_id = site_id
        self.site_url = site_url or f"https://{config['domain']}"
        self.wp_username = wp_username or config["wordpress"]["user"]
        self.wp_password = os.getenv("WP_PASSWORD", "")

        api_key = os.environ.get("STEEL_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "STEEL_API_KEY environment variable is required"
            )
        from steel_sdk import Steel
        self.steel = Steel(api_key=api_key)
        self.session = None

    # --- Context manager ---

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()
        return False

    # --- Session management ---

    def create_session(self):
        """Create a new Steel browser session."""
        self.session = self.steel.sessions.create()
        return self.session

    def close_session(self):
        """Release the browser session."""
        if self.session:
            self.steel.sessions.release(self.session.id)
            self.session = None

    # --- WordPress operations ---

    def login_wordpress(self):
        """Login to WordPress admin."""
        if not self.session:
            self.create_session()

        self.steel.navigate(self.session.id, f"{self.site_url}/wp-admin")
        self.steel.type(self.session.id, "#user_login", self.wp_username)
        self.steel.type(self.session.id, "#user_pass", self.wp_password)
        self.steel.click(self.session.id, "#wp-submit")
        self.steel.wait_for(self.session.id, "#adminmenu")
        print(f"Logged into {self.site_url}")

    def create_post(self, title, content, status="draft"):
        """Create a new WordPress post via the classic editor."""
        if not self.session:
            self.login_wordpress()

        self.steel.navigate(
            self.session.id, f"{self.site_url}/wp-admin/post-new.php"
        )
        self.steel.type(self.session.id, "#title", title)
        self.steel.type(self.session.id, "#content", content)

        if status == "publish":
            self.steel.click(self.session.id, "#publish")
        else:
            self.steel.click(self.session.id, "#save-post")

        print(f"Created post: {title}")

    def clear_cache(self):
        """Clear LiteSpeed cache."""
        if not self.session:
            self.login_wordpress()

        self.steel.navigate(
            self.session.id,
            f"{self.site_url}/wp-admin/admin.php?page=litespeed-cache",
        )
        self.steel.click(self.session.id, ".litespeed-purge-all")
        print(f"Cache cleared for {self.site_url}")

    def take_screenshot(self, filename="screenshot.png"):
        """Take a screenshot of the current page."""
        if not self.session:
            self.create_session()

        screenshot = self.steel.screenshot(self.session.id)
        with open(filename, "wb") as f:
            f.write(screenshot)
        print(f"Screenshot saved: {filename}")


if __name__ == "__main__":
    import sys

    site = sys.argv[1] if len(sys.argv) > 1 else "witchcraftforbeginners"
    with SteelWPBase(site) as auto:
        auto.login_wordpress()
        auto.take_screenshot("wp-admin-screenshot.png")
