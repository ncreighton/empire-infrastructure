"""
WordPress credential setup v2.
Fixed flow: Save site first, then select it from dropdown, then save user.

ZimmWriter WordPress flow:
1. Enter site URL -> Save New Site
2. Select saved site from dropdown
3. Enter username + app password -> Save New User
"""

import json
import sys
import os
import subprocess
import time
import ctypes
from ctypes import wintypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pywinauto import Application, Desktop
    from pywinauto.keyboard import send_keys
except ImportError:
    print("ERROR: pip install pywinauto")
    sys.exit(1)

try:
    import pyperclip
except ImportError:
    pyperclip = None

try:
    import pyautogui
except ImportError:
    pyautogui = None

SITES_JSON = r"D:\Claude Code Projects\config\sites.json"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# WordPress Settings auto_ids
WP_URL = "79"
WP_USER = "81"
WP_PASS = "83"
BTN_SAVE_SITE = "86"
BTN_SAVE_USER = "90"
DD_SITES = "94"
DD_USERS = "96"

# Win32 ComboBox messages
CB_GETCOUNT = 0x0146
CB_GETCURSEL = 0x0147
CB_SETCURSEL = 0x014E
CB_GETLBTEXTLEN = 0x0149
CB_GETLBTEXT = 0x0148
CB_FINDSTRINGEXACT = 0x0158


def send_msg(hwnd, msg, wp=0, lp=0):
    """Send Win32 message with proper 64-bit types."""
    SendMessage = ctypes.windll.user32.SendMessageW
    SendMessage.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMessage.restype = ctypes.c_long
    return SendMessage(hwnd, msg, wp, lp)


def send_msg_text(hwnd, msg, wp, text_buf):
    """Send Win32 message with text buffer LP."""
    SendMessage = ctypes.windll.user32.SendMessageW
    SendMessage.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, ctypes.c_wchar_p]
    SendMessage.restype = ctypes.c_long
    return SendMessage(hwnd, msg, wp, text_buf)


def combo_get_items(combo):
    """Read all items from a combo box using Win32."""
    hwnd = combo.handle
    count = send_msg(hwnd, CB_GETCOUNT)
    items = []
    for i in range(count):
        length = send_msg(hwnd, CB_GETLBTEXTLEN, i)
        if length >= 0:
            buf = ctypes.create_unicode_buffer(length + 2)
            send_msg_text(hwnd, CB_GETLBTEXT, i, buf)
            items.append(buf.value)
    return items


def combo_select_by_index(combo, idx):
    """Select combo item by index."""
    send_msg(combo.handle, CB_SETCURSEL, idx)


def combo_select_by_text(combo, text):
    """Select combo item matching text."""
    hwnd = combo.handle
    idx = send_msg_text(hwnd, CB_FINDSTRINGEXACT, -1, text)
    if idx >= 0:
        send_msg(hwnd, CB_SETCURSEL, idx)
        return True
    return False


def load_credentials():
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
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | "
         "Select-Object -First 1 -ExpandProperty Id"],
        capture_output=True, text=True, timeout=10
    )
    pid = int(result.stdout.strip())
    app = Application(backend="uia").connect(process=pid)
    window = app.top_window()
    print(f"Connected: {window.window_text()}")
    return app, window


def navigate_to_wordpress(app, window):
    title = window.window_text()
    if "Setup WordPress" in title:
        return window
    if "Menu" in title and "Option" not in title:
        window.child_window(title="Options Menu", control_type="Button").invoke()
        time.sleep(3)
        window = app.top_window()
    if "Option" in window.window_text() and "Setup" not in window.window_text():
        window.child_window(auto_id="55", control_type="Button").invoke()
        time.sleep(3)
        window = app.top_window()
    print(f"On: {window.window_text()}")
    return window


def dismiss_dialog(app, max_wait=2):
    """Dismiss any popup dialog (Error, Info, etc). Returns dialog text."""
    dialog_text = None
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            for w in Desktop(backend="uia").windows():
                wtitle = w.window_text()
                if "ZimmWriter" in wtitle and any(kw in wtitle for kw in ["Error", "Info", "Warning", "Confirm"]):
                    # Try to read the dialog content
                    try:
                        for static in w.descendants(control_type="Text"):
                            txt = static.window_text()
                            if txt and len(txt) > 5 and txt != wtitle:
                                dialog_text = txt
                                break
                    except Exception:
                        pass

                    # Click OK/Yes button
                    for btn_title in ["OK", "Yes", "&OK", "&Yes"]:
                        try:
                            w.child_window(title=btn_title, control_type="Button").click_input()
                            time.sleep(0.5)
                            return dialog_text or wtitle
                        except Exception:
                            continue
                    # Fallback: click first button
                    try:
                        btns = w.descendants(control_type="Button")
                        if btns:
                            btns[0].click_input()
                            time.sleep(0.5)
                            return dialog_text or wtitle
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(0.3)
    return None


def paste_text(field, value):
    """Set text field value via clipboard paste."""
    field.set_focus()
    time.sleep(0.15)
    send_keys("^a", pause=0.05)
    time.sleep(0.05)
    if pyperclip:
        pyperclip.copy(value)
    send_keys("^v", pause=0.05)
    time.sleep(0.3)


def take_screenshot(window, name="wp_screenshot"):
    """Take screenshot of ZimmWriter window."""
    try:
        window.set_focus()
        time.sleep(0.5)
        img = window.capture_as_image()
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        img.save(path)
        print(f"  Screenshot: {path}")
        return path
    except Exception as e:
        print(f"  Screenshot failed: {e}")
        if pyautogui:
            try:
                window.set_focus()
                time.sleep(0.5)
                rect = window.rectangle()
                ss = pyautogui.screenshot(region=(rect.left, rect.top, rect.width(), rect.height()))
                path = os.path.join(OUTPUT_DIR, f"{name}.png")
                ss.save(path)
                print(f"  Screenshot (pyautogui): {path}")
                return path
            except Exception as e2:
                print(f"  Fallback screenshot failed: {e2}")
    return None


def main():
    print("=" * 60)
    print("  WordPress Credential Setup v2")
    print("=" * 60)

    sites = load_credentials()
    print(f"\nLoaded {len(sites)} sites\n")

    app, window = connect()
    window = navigate_to_wordpress(app, window)

    # Take initial screenshot
    take_screenshot(window, "wp_before_setup")

    # First: check what sites are already saved
    try:
        dd_sites = window.child_window(auto_id=DD_SITES, control_type="ComboBox")
        existing_sites = combo_get_items(dd_sites)
        print(f"\nExisting saved sites ({len(existing_sites)}):")
        for s in existing_sites:
            print(f"  - {s}")
    except Exception as e:
        print(f"\nCouldn't read saved sites: {e}")
        existing_sites = []

    # Check saved users
    try:
        dd_users = window.child_window(auto_id=DD_USERS, control_type="ComboBox")
        existing_users = combo_get_items(dd_users)
        print(f"\nExisting saved users ({len(existing_users)}):")
        for u in existing_users:
            print(f"  - {u}")
    except Exception as e:
        print(f"\nCouldn't read saved users: {e}")
        existing_users = []

    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Save Sites ({len(sites)} to add)")
    print(f"{'=' * 60}\n")

    sites_added = 0
    sites_skipped = 0

    for i, site in enumerate(sites, 1):
        domain = site["domain"]

        # Check if already in dropdown
        if any(domain in s for s in existing_sites):
            print(f"  [{i:2d}] SKIP {domain} (already saved)")
            sites_skipped += 1
            continue

        print(f"  [{i:2d}] Saving site: {site['url']}")

        # Enter URL
        url_field = window.child_window(auto_id=WP_URL, control_type="Edit")
        paste_text(url_field, site["url"])

        # Click Save New Site
        btn = window.child_window(auto_id=BTN_SAVE_SITE, control_type="Button")
        try:
            btn.invoke()
        except Exception:
            btn.click_input()
        time.sleep(1.5)

        # Handle dialog
        dialog = dismiss_dialog(app)
        if dialog:
            print(f"       Dialog: {dialog[:100]}")

        sites_added += 1
        time.sleep(0.5)

    print(f"\n  Sites: {sites_added} added, {sites_skipped} skipped")

    # Re-read saved sites after adding
    time.sleep(1)
    try:
        dd_sites = window.child_window(auto_id=DD_SITES, control_type="ComboBox")
        saved_sites = combo_get_items(dd_sites)
        print(f"\n  Saved sites now ({len(saved_sites)}):")
        for s in saved_sites:
            print(f"    - {s}")
    except Exception as e:
        print(f"\n  Couldn't re-read sites: {e}")
        saved_sites = []

    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Save Users (linked to sites)")
    print(f"{'=' * 60}\n")

    users_added = 0
    users_failed = []

    for i, site in enumerate(sites, 1):
        domain = site["domain"]
        print(f"  [{i:2d}] User for {domain}: {site['user']}")

        # Select the site from dropdown first
        try:
            dd_sites = window.child_window(auto_id=DD_SITES, control_type="ComboBox")
            # Try exact match first, then partial
            found = combo_select_by_text(dd_sites, site["url"])
            if not found:
                found = combo_select_by_text(dd_sites, f"https://{domain}")
            if not found:
                found = combo_select_by_text(dd_sites, domain)
            if not found:
                # Try by index matching
                for idx, item in enumerate(saved_sites):
                    if domain in item:
                        combo_select_by_index(dd_sites, idx)
                        found = True
                        break

            if found:
                print(f"       Selected site in dropdown")
            else:
                print(f"       WARNING: Could not find {domain} in site dropdown")
        except Exception as e:
            print(f"       Error selecting site: {e}")

        time.sleep(0.5)

        # Enter username
        user_field = window.child_window(auto_id=WP_USER, control_type="Edit")
        paste_text(user_field, site["user"])

        # Enter app password
        pass_field = window.child_window(auto_id=WP_PASS, control_type="Edit")
        paste_text(pass_field, site["app_password"])

        # Click Save New User
        btn = window.child_window(auto_id=BTN_SAVE_USER, control_type="Button")
        try:
            btn.invoke()
        except Exception:
            btn.click_input()
        time.sleep(1.5)

        # Handle dialog
        dialog = dismiss_dialog(app)
        if dialog:
            print(f"       Dialog: {dialog[:100]}")
            if "error" in (dialog or "").lower():
                users_failed.append(domain)
            else:
                users_added += 1
        else:
            users_added += 1

        time.sleep(0.5)

    # Final verification
    print(f"\n{'=' * 60}")
    print(f"  VERIFICATION")
    print(f"{'=' * 60}")

    try:
        dd_sites = window.child_window(auto_id=DD_SITES, control_type="ComboBox")
        final_sites = combo_get_items(dd_sites)
        print(f"\n  Final saved sites ({len(final_sites)}):")
        for s in final_sites:
            print(f"    - {s}")
    except Exception as e:
        print(f"\n  Sites read error: {e}")

    try:
        dd_users = window.child_window(auto_id=DD_USERS, control_type="ComboBox")
        final_users = combo_get_items(dd_users)
        print(f"\n  Final saved users ({len(final_users)}):")
        for u in final_users:
            print(f"    - {u}")
    except Exception as e:
        print(f"\n  Users read error: {e}")

    # Take final screenshot
    take_screenshot(window, "wp_after_setup")

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"  Sites added: {sites_added}")
    print(f"  Users added: {users_added}")
    if users_failed:
        print(f"  Users failed: {', '.join(users_failed)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
