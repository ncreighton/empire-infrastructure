"""
Set up WordPress credentials for all 14 sites in ZimmWriter.
Navigates to Options Menu -> WordPress Settings, then enters each site.

ZimmWriter WordPress Settings screen controls:
  Text Fields:
    auto_id=79  "WordPress Site URL:"
    auto_id=81  "WordPress User:"
    auto_id=83  "WordPress User App Pass:"
  Buttons:
    auto_id=86  "Save New Site"
    auto_id=90  "Save New User"
  Dropdowns:
    auto_id=94  "Load a Saved WordPress Site to Update:"
    auto_id=96  "Load a Saved WordPress User to Update:"
"""

import json
import sys
import os
import subprocess
import time
import ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pywinauto import Application
    from pywinauto.keyboard import send_keys
except ImportError:
    print("ERROR: pip install pywinauto")
    sys.exit(1)

try:
    import pyperclip
except ImportError:
    pyperclip = None

# Load credentials from sites.json
SITES_JSON = r"D:\Claude Code Projects\config\sites.json"

# Control auto_ids for WordPress Settings screen
WP_URL_FIELD = "79"
WP_USER_FIELD = "81"
WP_PASS_FIELD = "83"
SAVE_SITE_BTN = "86"
SAVE_USER_BTN = "90"
SITE_DROPDOWN = "94"
USER_DROPDOWN = "96"


def load_credentials():
    """Load WordPress credentials from sites.json."""
    with open(SITES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    sites = []
    for site_id, config in data["sites"].items():
        wp = config.get("wordpress", {})
        domain = config.get("domain", "")
        if wp.get("user") and wp.get("app_password") and domain:
            sites.append({
                "site_id": site_id,
                "domain": domain,
                "url": f"https://{domain}",
                "user": wp["user"],
                "app_password": wp["app_password"],
            })
    return sites


def connect():
    """Connect to ZimmWriter via PID."""
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | "
         "Select-Object -First 1 -ExpandProperty Id"],
        capture_output=True, text=True, timeout=10
    )
    pid = int(result.stdout.strip())
    app = Application(backend="uia").connect(process=pid)
    window = app.top_window()
    print(f"Connected to PID {pid}: {window.window_text()}")
    return app, window


def navigate_to_wordpress(app, window):
    """Navigate to WordPress Settings screen."""
    title = window.window_text()

    # From main Menu -> Options Menu
    if "Menu" in title and "Option" not in title and "WordPress" not in title:
        print("  Menu -> Options Menu...")
        btn = window.child_window(title="Options Menu", control_type="Button")
        btn.invoke()
        time.sleep(3)
        window = app.top_window()
        title = window.window_text()

    # From Options Menu -> WordPress Settings
    if "Option" in title and "WordPress" not in title:
        print("  Options Menu -> WordPress Settings...")
        btn = window.child_window(auto_id="55", control_type="Button")
        btn.invoke()
        time.sleep(3)
        window = app.top_window()
        title = window.window_text()

    print(f"  Now on: {title}")
    return window


def set_text_via_clipboard(edit_ctrl, value):
    """Clear field and paste value via clipboard."""
    edit_ctrl.set_focus()
    time.sleep(0.1)
    send_keys("^a", pause=0.05)
    time.sleep(0.05)

    if pyperclip:
        pyperclip.copy(value)
    else:
        # Fallback to ctypes
        import win32clipboard
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(value)
        win32clipboard.CloseClipboard()

    send_keys("^v", pause=0.05)
    time.sleep(0.2)


def set_text_via_keys(edit_ctrl, value):
    """Clear field and type value character by character."""
    edit_ctrl.set_focus()
    time.sleep(0.1)
    send_keys("^a", pause=0.05)
    time.sleep(0.05)
    send_keys("{DELETE}", pause=0.05)
    time.sleep(0.1)
    # Type the value, escaping pywinauto special chars
    safe_value = value.replace("{", "{{").replace("}", "}}").replace("(", "{(}").replace(")", "{)}")
    safe_value = safe_value.replace("+", "{+}").replace("^", "{^}").replace("%", "{%}")
    edit_ctrl.type_keys(safe_value, with_spaces=True, pause=0.02)
    time.sleep(0.2)


def add_wordpress_site(window, url, username, app_password):
    """Add a single WordPress site with credentials."""
    # 1. Enter Site URL
    url_field = window.child_window(auto_id=WP_URL_FIELD, control_type="Edit")
    set_text_via_clipboard(url_field, url)
    time.sleep(0.3)

    # 2. Click "Save New Site"
    save_site_btn = window.child_window(auto_id=SAVE_SITE_BTN, control_type="Button")
    try:
        save_site_btn.invoke()
    except Exception:
        save_site_btn.click_input()
    time.sleep(2)

    # Handle any popup/confirmation dialog
    try:
        from pywinauto import Desktop
        for w in Desktop(backend="uia").windows():
            wtitle = w.window_text()
            if any(kw in wtitle.lower() for kw in ["error", "warning", "info", "confirm", "success"]):
                print(f"    Dialog: '{wtitle}' - clicking OK")
                try:
                    w.child_window(title="OK", control_type="Button").click_input()
                except Exception:
                    try:
                        w.child_window(title_re=".*OK.*|.*Yes.*", control_type="Button").click_input()
                    except Exception:
                        pass
                time.sleep(1)
                break
    except Exception:
        pass

    # Re-grab window reference (may have changed after dialog)
    time.sleep(0.5)

    # 3. Enter Username
    user_field = window.child_window(auto_id=WP_USER_FIELD, control_type="Edit")
    set_text_via_clipboard(user_field, username)
    time.sleep(0.3)

    # 4. Enter App Password
    pass_field = window.child_window(auto_id=WP_PASS_FIELD, control_type="Edit")
    set_text_via_clipboard(pass_field, app_password)
    time.sleep(0.3)

    # 5. Click "Save New User"
    save_user_btn = window.child_window(auto_id=SAVE_USER_BTN, control_type="Button")
    try:
        save_user_btn.invoke()
    except Exception:
        save_user_btn.click_input()
    time.sleep(2)

    # Handle any popup after user save
    try:
        from pywinauto import Desktop
        for w in Desktop(backend="uia").windows():
            wtitle = w.window_text()
            if any(kw in wtitle.lower() for kw in ["error", "warning", "info", "confirm", "success"]):
                print(f"    Dialog: '{wtitle}' - clicking OK")
                try:
                    w.child_window(title="OK", control_type="Button").click_input()
                except Exception:
                    try:
                        w.child_window(title_re=".*OK.*|.*Yes.*", control_type="Button").click_input()
                    except Exception:
                        pass
                time.sleep(1)
                break
    except Exception:
        pass


def verify_saved_sites(window):
    """Check what sites are saved in the dropdown."""
    try:
        site_dd = window.child_window(auto_id=SITE_DROPDOWN, control_type="ComboBox")
        items = site_dd.item_texts()
        print(f"\n  Saved sites in dropdown ({len(items)} total):")
        for item in items:
            print(f"    - {item}")
        return items
    except Exception as e:
        print(f"  Could not read site dropdown: {e}")
        return []


def verify_saved_users(window):
    """Check what users are saved in the dropdown."""
    try:
        user_dd = window.child_window(auto_id=USER_DROPDOWN, control_type="ComboBox")
        items = user_dd.item_texts()
        print(f"\n  Saved users in dropdown ({len(items)} total):")
        for item in items:
            print(f"    - {item}")
        return items
    except Exception as e:
        print(f"  Could not read user dropdown: {e}")
        return []


def main():
    print("=" * 60)
    print("  ZimmWriter WordPress Credential Setup")
    print("=" * 60)

    # Load credentials
    sites = load_credentials()
    print(f"\nLoaded {len(sites)} sites from {SITES_JSON}\n")
    for s in sites:
        print(f"  {s['domain']:35s}  user={s['user']}")

    # Connect
    app, window = connect()
    window = navigate_to_wordpress(app, window)

    # Check existing saved sites
    existing = verify_saved_sites(window)

    print(f"\n{'=' * 60}")
    print(f"  Adding {len(sites)} WordPress sites...")
    print(f"{'=' * 60}\n")

    success = 0
    failed = []

    for i, site in enumerate(sites, 1):
        # Skip if already saved
        if any(site["domain"] in item for item in existing):
            print(f"[{i:2d}/{len(sites)}] SKIP {site['domain']} (already saved)")
            success += 1
            continue

        print(f"[{i:2d}/{len(sites)}] Adding {site['domain']}...")
        print(f"         URL:  {site['url']}")
        print(f"         User: {site['user']}")
        print(f"         Pass: {site['app_password'][:8]}...")

        try:
            add_wordpress_site(window, site["url"], site["user"], site["app_password"])
            print(f"         SAVED!")
            success += 1
        except Exception as e:
            print(f"         FAILED: {e}")
            failed.append(site["domain"])

        # Brief pause between sites
        time.sleep(1)

    # Verify final state
    print(f"\n{'=' * 60}")
    print("  VERIFICATION")
    print(f"{'=' * 60}")
    final_sites = verify_saved_sites(window)
    final_users = verify_saved_users(window)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {success} saved, {len(failed)} failed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
