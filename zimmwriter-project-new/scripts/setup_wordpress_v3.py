"""
WordPress credential setup v3.
Fix: Fill ALL THREE fields before saving. Use set_edit_text() for URL field.
ZimmWriter validates the full connection before saving.
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

# Auto IDs
WP_URL = "79"
WP_USER = "81"
WP_PASS = "83"
BTN_SAVE_SITE = "86"
BTN_SAVE_USER = "90"
DD_SITES = "94"
DD_USERS = "96"


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


def dismiss_dialog(app, max_wait=3):
    """Dismiss popup and return its text content."""
    time.sleep(0.5)
    dialog_text = None
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            for w in Desktop(backend="uia").windows():
                wtitle = w.window_text()
                if "ZimmWriter" in wtitle and any(kw in wtitle for kw in ["Error", "Info", "Warning"]):
                    # Read dialog content
                    try:
                        for static in w.descendants(control_type="Text"):
                            txt = static.window_text()
                            if txt and len(txt) > 10 and txt != wtitle:
                                dialog_text = txt
                                break
                    except Exception:
                        pass

                    # Click OK
                    for btn_name in ["OK", "&OK", "Yes", "&Yes"]:
                        try:
                            w.child_window(title=btn_name, control_type="Button").click_input()
                            time.sleep(0.5)
                            return dialog_text or wtitle
                        except Exception:
                            continue
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


def set_field(window, auto_id, value, label=""):
    """Set a text field using multiple strategies."""
    field = window.child_window(auto_id=auto_id, control_type="Edit")

    # Strategy 1: set_edit_text (direct Win32 WM_SETTEXT)
    try:
        field.set_edit_text(value)
        time.sleep(0.2)
        # Verify
        current = field.get_value() if hasattr(field, 'get_value') else ""
        if current and value in current:
            print(f"    {label}: SET via set_edit_text = '{current[:60]}'")
            return True
    except Exception as e:
        print(f"    {label}: set_edit_text failed: {e}")

    # Strategy 2: Win32 WM_SETTEXT directly
    try:
        hwnd = field.handle
        WM_SETTEXT = 0x000C
        SendMessage = ctypes.windll.user32.SendMessageW
        SendMessage.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, ctypes.c_wchar_p]
        SendMessage.restype = ctypes.c_long
        SendMessage(hwnd, WM_SETTEXT, 0, value)
        time.sleep(0.2)
        current = field.get_value() if hasattr(field, 'get_value') else ""
        if current and value in current:
            print(f"    {label}: SET via WM_SETTEXT = '{current[:60]}'")
            return True
        else:
            print(f"    {label}: WM_SETTEXT sent, field shows '{current[:60]}'")
    except Exception as e:
        print(f"    {label}: WM_SETTEXT failed: {e}")

    # Strategy 3: clipboard paste
    try:
        field.set_focus()
        time.sleep(0.2)
        send_keys("^a", pause=0.05)
        time.sleep(0.1)
        if pyperclip:
            pyperclip.copy(value)
        send_keys("^v", pause=0.05)
        time.sleep(0.3)
        current = field.get_value() if hasattr(field, 'get_value') else ""
        print(f"    {label}: SET via clipboard = '{current[:60]}'")
        return True
    except Exception as e:
        print(f"    {label}: clipboard paste failed: {e}")

    # Strategy 4: type_keys character by character
    try:
        field.set_focus()
        time.sleep(0.1)
        send_keys("^a{DELETE}", pause=0.05)
        time.sleep(0.1)
        # Escape pywinauto special chars
        safe = value.replace("{", "{{").replace("}", "}}")
        safe = safe.replace("(", "{(}").replace(")", "{)}")
        safe = safe.replace("+", "{+}").replace("^", "{^}").replace("%", "{%}")
        field.type_keys(safe, with_spaces=True, pause=0.02)
        time.sleep(0.3)
        current = field.get_value() if hasattr(field, 'get_value') else ""
        print(f"    {label}: SET via type_keys = '{current[:60]}'")
        return True
    except Exception as e:
        print(f"    {label}: type_keys failed: {e}")

    return False


def take_screenshot(window, name):
    try:
        window.set_focus()
        time.sleep(0.3)
        img = window.capture_as_image()
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        img.save(path)
        print(f"  Screenshot: {path}")
        return path
    except Exception:
        if pyautogui:
            try:
                rect = window.rectangle()
                ss = pyautogui.screenshot(region=(rect.left, rect.top, rect.width(), rect.height()))
                path = os.path.join(OUTPUT_DIR, f"{name}.png")
                ss.save(path)
                return path
            except Exception:
                pass
    return None


def combo_get_items_win32(combo):
    """Read combo items using proper Win32 calls."""
    hwnd = combo.handle
    SendMessage = ctypes.windll.user32.SendMessageW
    SendMessage.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMessage.restype = ctypes.c_long

    CB_GETCOUNT = 0x0146
    count = SendMessage(hwnd, CB_GETCOUNT, 0, 0)
    items = []

    for i in range(count):
        # Get text length
        CB_GETLBTEXTLEN = 0x0149
        length = SendMessage(hwnd, CB_GETLBTEXTLEN, i, 0)
        if length >= 0:
            buf = ctypes.create_unicode_buffer(length + 2)
            CB_GETLBTEXT = 0x0148
            SendMsg2 = ctypes.windll.user32.SendMessageW
            SendMsg2.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, ctypes.c_wchar_p]
            SendMsg2.restype = ctypes.c_long
            SendMsg2(hwnd, CB_GETLBTEXT, i, buf)
            items.append(buf.value)
    return items


def main():
    print("=" * 60)
    print("  WordPress Setup v3 — All Fields Before Save")
    print("=" * 60)

    sites = load_credentials()
    print(f"\nLoaded {len(sites)} sites\n")

    app, window = connect()
    window = navigate_to_wordpress(app, window)

    # Test with just ONE site first
    test_site = sites[0]
    print(f"\n--- TEST: {test_site['domain']} ---")
    print(f"  URL:  {test_site['url']}")
    print(f"  User: {test_site['user']}")
    print(f"  Pass: {test_site['app_password'][:12]}...")

    # Fill all three fields
    print("\n  Filling all 3 fields...")
    set_field(window, WP_URL, test_site["url"], "URL")
    set_field(window, WP_USER, test_site["user"], "User")
    set_field(window, WP_PASS, test_site["app_password"], "Pass")

    # Take screenshot to verify fields are filled
    take_screenshot(window, "wp_v3_fields_filled")

    # Verify all fields have content
    print("\n  Verifying field contents...")
    for aid, label in [(WP_URL, "URL"), (WP_USER, "User"), (WP_PASS, "Pass")]:
        try:
            f = window.child_window(auto_id=aid, control_type="Edit")
            val = f.get_value() if hasattr(f, 'get_value') else f.window_text()
            print(f"    {label}: '{val[:80]}'")
        except Exception as e:
            print(f"    {label}: read error: {e}")

    # Now click Save New Site
    print("\n  Clicking 'Save New Site'...")
    btn = window.child_window(auto_id=BTN_SAVE_SITE, control_type="Button")
    try:
        btn.invoke()
    except Exception:
        btn.click_input()
    time.sleep(3)

    # Check for dialog
    dialog = dismiss_dialog(app, max_wait=5)
    if dialog:
        print(f"  Dialog: {dialog[:200]}")

    # Take screenshot after save attempt
    take_screenshot(window, "wp_v3_after_save_site")

    # Try Save New User too
    print("\n  Clicking 'Save New User'...")
    btn = window.child_window(auto_id=BTN_SAVE_USER, control_type="Button")
    try:
        btn.invoke()
    except Exception:
        btn.click_input()
    time.sleep(3)

    dialog = dismiss_dialog(app, max_wait=5)
    if dialog:
        print(f"  Dialog: {dialog[:200]}")

    # Take screenshot after user save attempt
    take_screenshot(window, "wp_v3_after_save_user")

    # Check dropdowns
    print("\n  Checking saved sites dropdown...")
    try:
        dd = window.child_window(auto_id=DD_SITES, control_type="ComboBox")
        items = combo_get_items_win32(dd)
        print(f"  Saved sites ({len(items)}):")
        for item in items:
            print(f"    - {item}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n  Checking saved users dropdown...")
    try:
        dd = window.child_window(auto_id=DD_USERS, control_type="ComboBox")
        items = combo_get_items_win32(dd)
        print(f"  Saved users ({len(items)}):")
        for item in items:
            print(f"    - {item}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nDone with test. Check screenshots.")


if __name__ == "__main__":
    main()
