"""
Verify WordPress settings in ZimmWriter by reading dropdown contents
and taking a screenshot of the current state.
"""

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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def read_combo_items_win32(combo):
    """Read combo box items using Win32 messages (works with AutoIt combos)."""
    hwnd = combo.handle
    CB_GETCOUNT = 0x0146
    CB_GETLBTEXTLEN = 0x0149
    CB_GETLBTEXT = 0x0148

    count = ctypes.windll.user32.SendMessageW(hwnd, CB_GETCOUNT, 0, 0)
    items = []
    for i in range(count):
        length = ctypes.windll.user32.SendMessageW(hwnd, CB_GETLBTEXTLEN, i, 0)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.SendMessageW(hwnd, CB_GETLBTEXT, i, ctypes.addressof(buf))
            items.append(buf.value)
        else:
            items.append("")
    return items


def get_combo_selected_win32(combo):
    """Get currently selected item text using Win32."""
    hwnd = combo.handle
    CB_GETCURSEL = 0x0147
    CB_GETLBTEXTLEN = 0x0149
    CB_GETLBTEXT = 0x0148

    idx = ctypes.windll.user32.SendMessageW(hwnd, CB_GETCURSEL, 0, 0)
    if idx < 0:
        return "(none selected)"
    length = ctypes.windll.user32.SendMessageW(hwnd, CB_GETLBTEXTLEN, idx, 0)
    if length > 0:
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.SendMessageW(hwnd, CB_GETLBTEXT, idx, ctypes.addressof(buf))
        return buf.value
    return "(empty)"


def main():
    app, window = connect()
    title = window.window_text()
    print(f"Screen: {title}")

    # If not on WordPress Settings, navigate there
    if "WordPress" not in title and "Setup" not in title:
        print("Not on WordPress screen. Navigating...")
        if "Menu" in title and "Option" not in title:
            window.child_window(title="Options Menu", control_type="Button").invoke()
            time.sleep(3)
            window = app.top_window()
        if "Option" in window.window_text():
            window.child_window(auto_id="55", control_type="Button").invoke()
            time.sleep(3)
            window = app.top_window()
        print(f"Now on: {window.window_text()}")

    # Take screenshot
    try:
        img = window.capture_as_image()
        ss_path = os.path.join(OUTPUT_DIR, "wordpress_settings_screenshot.png")
        img.save(ss_path)
        print(f"\nScreenshot saved: {ss_path}")
    except Exception as e:
        print(f"Screenshot failed: {e}")

    # Read all dropdowns using Win32
    print("\n" + "=" * 60)
    print("  DROPDOWN CONTENTS (Win32 CB messages)")
    print("=" * 60)

    for auto_id, label in [("85", "TimeZone"), ("94", "Saved Sites"), ("96", "Saved Users")]:
        try:
            combo = window.child_window(auto_id=auto_id, control_type="ComboBox")
            items = read_combo_items_win32(combo)
            selected = get_combo_selected_win32(combo)
            print(f"\n  [{auto_id}] {label} (selected: '{selected}', {len(items)} items):")
            for item in items:
                print(f"    - {item}")
        except Exception as e:
            print(f"\n  [{auto_id}] {label}: Error - {e}")

    # Read text fields
    print("\n" + "=" * 60)
    print("  TEXT FIELD CONTENTS")
    print("=" * 60)

    for auto_id, label in [("79", "Site URL"), ("81", "Username"), ("83", "App Password")]:
        try:
            field = window.child_window(auto_id=auto_id, control_type="Edit")
            # Try get_value first
            try:
                val = field.get_value()
            except Exception:
                val = field.window_text()
            print(f"  [{auto_id}] {label}: '{val}'")
        except Exception as e:
            print(f"  [{auto_id}] {label}: Error - {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
