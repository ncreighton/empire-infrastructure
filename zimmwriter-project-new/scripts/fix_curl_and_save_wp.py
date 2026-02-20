"""
1. Navigate to Options Menu
2. Enable 'Bypass Windows CURL' checkbox
3. Save Options
4. Go to WordPress Settings
5. Save all 14 sites
"""

import json
import sys
import os
import subprocess
import time
import ctypes
from ctypes import wintypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pywinauto import Application, Desktop
from pywinauto.keyboard import send_keys

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


def dismiss_dialog(max_wait=5):
    """Dismiss popup dialogs and return text."""
    time.sleep(0.5)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            for w in Desktop(backend="uia").windows():
                wtitle = w.window_text()
                if "ZimmWriter" in wtitle and any(kw in wtitle for kw in ["Error", "Info", "Warning", "Success"]):
                    # Read content
                    dialog_text = None
                    try:
                        for static in w.descendants(control_type="Text"):
                            txt = static.window_text()
                            if txt and len(txt) > 10 and txt != wtitle:
                                dialog_text = txt
                                break
                    except Exception:
                        pass

                    # Click OK
                    for btn_name in ["OK", "&OK", "Yes"]:
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


def take_screenshot(window, name):
    try:
        window.set_focus()
        time.sleep(0.3)
        img = window.capture_as_image()
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        img.save(path)
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


def main():
    sites = load_credentials()
    app, window = connect()
    title = window.window_text()

    # === STEP 1: Navigate to Options Menu ===
    print("\n=== STEP 1: Enable Bypass Windows CURL ===")

    # If on WordPress Settings, close to go back to Options Menu
    if "Setup WordPress" in title:
        print("  Closing WordPress Settings...")
        try:
            close_btn = window.child_window(title="Close", control_type="Button")
            close_btn.click_input()
            time.sleep(2)
            window = app.top_window()
            title = window.window_text()
            print(f"  Now on: {title}")
        except Exception as e:
            print(f"  Close failed: {e}")

    # Navigate to Options Menu if needed
    if "Menu" in title and "Option" not in title:
        print("  Going to Options Menu...")
        window.child_window(title="Options Menu", control_type="Button").invoke()
        time.sleep(3)
        window = app.top_window()
        title = window.window_text()

    if "Option Menu" in title or "Option" in title:
        print(f"  On Options Menu: {title}")

        # Check current state of Bypass CURL checkbox (auto_id=33)
        bypass_cb = window.child_window(auto_id="33", control_type="CheckBox")
        current_state = bypass_cb.get_toggle_state()
        print(f"  Bypass Windows CURL: {'CHECKED' if current_state == 1 else 'UNCHECKED'}")

        if current_state == 0:
            print("  Enabling Bypass Windows CURL...")
            bypass_cb.toggle()
            time.sleep(0.5)
            new_state = bypass_cb.get_toggle_state()
            print(f"  Now: {'CHECKED' if new_state == 1 else 'UNCHECKED'}")

        # Save Options
        print("  Saving Options...")
        save_btn = window.child_window(auto_id="54", control_type="Button")
        try:
            save_btn.invoke()
        except Exception:
            save_btn.click_input()
        time.sleep(2)

        dialog = dismiss_dialog()
        if dialog:
            print(f"  Dialog: {dialog[:100]}")

        take_screenshot(window, "options_after_bypass_curl")

    # === STEP 2: Navigate to WordPress Settings ===
    print("\n=== STEP 2: WordPress Settings ===")

    # Refresh window reference
    window = app.top_window()
    title = window.window_text()

    if "Option" in title and "Setup" not in title:
        print("  Going to WordPress Settings...")
        wp_btn = window.child_window(auto_id="55", control_type="Button")
        wp_btn.invoke()
        time.sleep(3)
        window = app.top_window()
        title = window.window_text()
        print(f"  Now on: {title}")

    # === STEP 3: Test save with first site ===
    print("\n=== STEP 3: Test first site ===")

    test_site = sites[0]
    print(f"  Site: {test_site['url']}")
    print(f"  User: {test_site['user']}")

    # Fill fields
    url_field = window.child_window(auto_id="79", control_type="Edit")
    url_field.set_edit_text(test_site["url"])
    time.sleep(0.2)

    user_field = window.child_window(auto_id="81", control_type="Edit")
    user_field.set_edit_text(test_site["user"])
    time.sleep(0.2)

    pass_field = window.child_window(auto_id="83", control_type="Edit")
    pass_field.set_edit_text(test_site["app_password"])
    time.sleep(0.2)

    # Verify
    print(f"  URL field:  '{url_field.get_value()}'")
    print(f"  User field: '{user_field.get_value()}'")
    print(f"  Pass field: '{pass_field.get_value()[:20]}...'")

    take_screenshot(window, "wp_v4_test_filled")

    # Try Save New Site
    print("  Clicking Save New Site...")
    save_btn = window.child_window(auto_id="86", control_type="Button")
    try:
        save_btn.invoke()
    except Exception:
        save_btn.click_input()

    # Wait longer for connection test
    time.sleep(8)

    dialog = dismiss_dialog(max_wait=10)
    if dialog:
        print(f"  Dialog: {dialog[:200]}")
        if "could not connect" in dialog.lower():
            print("  STILL FAILING - trying with trailing slash...")
            # Try with trailing slash
            url_field = window.child_window(auto_id="79", control_type="Edit")
            url_field.set_edit_text(test_site["url"] + "/")
            time.sleep(0.2)
            save_btn = window.child_window(auto_id="86", control_type="Button")
            try:
                save_btn.invoke()
            except Exception:
                save_btn.click_input()
            time.sleep(8)
            dialog2 = dismiss_dialog(max_wait=10)
            if dialog2:
                print(f"  Dialog (trailing /): {dialog2[:200]}")
    else:
        print("  No error dialog - might have succeeded!")

    take_screenshot(window, "wp_v4_after_save")

    # Check dropdowns
    print("\n  Checking saved sites...")
    try:
        dd = window.child_window(auto_id="94", control_type="ComboBox")
        hwnd = dd.handle
        SendMsg = ctypes.windll.user32.SendMessageW
        SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        SendMsg.restype = ctypes.c_long
        count = SendMsg(hwnd, 0x0146, 0, 0)  # CB_GETCOUNT

        SendMsg2 = ctypes.windll.user32.SendMessageW
        SendMsg2.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, ctypes.c_wchar_p]
        SendMsg2.restype = ctypes.c_long

        print(f"  Saved sites count: {count}")
        for i in range(count):
            length = SendMsg(hwnd, 0x0149, i, 0)  # CB_GETLBTEXTLEN
            if length >= 0:
                buf = ctypes.create_unicode_buffer(length + 2)
                SendMsg2(hwnd, 0x0148, i, buf)  # CB_GETLBTEXT
                print(f"    [{i}] {buf.value}")
    except Exception as e:
        print(f"  Error reading dropdown: {e}")

    # === If still failing, try without HTTPS (just domain) ===
    if dialog and "could not connect" in (dialog or "").lower():
        print("\n=== STEP 4: Try alternate URL formats ===")
        for url_variant in [
            f"https://www.{test_site['domain']}",
            test_site['domain'],
        ]:
            print(f"  Trying URL: {url_variant}")
            url_field = window.child_window(auto_id="79", control_type="Edit")
            url_field.set_edit_text(url_variant)
            time.sleep(0.2)
            save_btn = window.child_window(auto_id="86", control_type="Button")
            try:
                save_btn.invoke()
            except Exception:
                save_btn.click_input()
            time.sleep(8)
            d = dismiss_dialog(max_wait=10)
            if d:
                print(f"  Dialog: {d[:150]}")
                if "could not connect" not in d.lower():
                    print("  POSSIBLE SUCCESS!")
                    break
            else:
                print("  No error - SUCCESS?")
                break

    print("\nDone. Check screenshots in output/")


if __name__ == "__main__":
    main()
