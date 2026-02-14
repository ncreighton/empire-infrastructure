"""
Discover all controls on each Options Menu sub-screen.
Navigates: Menu -> Options Menu -> clicks each sub-screen button (auto_ids 56-63)
-> scans all descendants -> saves JSON -> closes via WM_CLOSE -> repeats.

Output: output/zimmwriter_{screen}_controls.json for each
        output/zimmwriter_all_subscreens.json combined

Pattern: Close Bulk Writer first (WM_CLOSE), navigate Menu -> Options Menu,
iterate sub-screens, then reopen Bulk Writer when done.
"""

import json
import sys
import os
import subprocess
import time
import ctypes
from ctypes import wintypes
from io import StringIO
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pywinauto import Application
except ImportError:
    print("ERROR: pip install pywinauto")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WM_CLOSE = 0x0010
SendMsg = ctypes.windll.user32.SendMessageW
SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
SendMsg.restype = ctypes.c_long

# Sub-screens to discover (auto_id -> screen name)
SUBSCREENS = [
    ("56", "text_api_settings"),
    ("57", "image_api_settings"),
    ("58", "scraping_api_settings"),
    ("59", "scraping_surgeon_settings"),
    ("60", "scraping_domain_settings"),
    ("61", "ai_image_prompt_settings"),
    ("62", "ai_words_to_nuke_settings"),
    ("63", "secure_mode_settings"),
]


def find_pid():
    """Find ZimmWriter's AutoIt3 PID."""
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | "
         "Select-Object -First 1 -ExpandProperty Id"],
        capture_output=True, text=True, timeout=10
    )
    return int(result.stdout.strip())


def connect(backend="win32"):
    """Connect to ZimmWriter via PID with specified backend."""
    pid = find_pid()
    app = Application(backend=backend).connect(process=pid)
    return pid, app


def connect_uia():
    """Connect with UIA backend (for detailed control scanning)."""
    return connect(backend="uia")


def connect_win32():
    """Connect with win32 backend (reliable for all screens)."""
    return connect(backend="win32")


def find_window_by_title(app, keyword):
    """Find a window containing keyword in title."""
    for w in app.windows():
        if keyword.lower() in w.window_text().lower():
            return w
    return None


def close_window_by_handle(handle):
    """Close a window via WM_CLOSE."""
    SendMsg(handle, WM_CLOSE, 0, 0)
    time.sleep(1)


def discover_screen(window, screen_name):
    """Map all controls on the current screen."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "window_title": window.window_text(),
        "screen": screen_name,
        "buttons": [], "checkboxes": [], "dropdowns": [],
        "text_fields": [], "labels": [], "other": [],
    }

    all_ctrls = window.descendants()
    print(f"  Scanning {len(all_ctrls)} elements...", flush=True)

    for ctrl in all_ctrls:
        try:
            ct = ctrl.friendly_class_name()
            # automation_id() only exists on UIA; use control_id() for win32
            try:
                aid = ctrl.automation_id()
            except (AttributeError, Exception):
                try:
                    aid = str(ctrl.control_id())
                except Exception:
                    aid = ""
            info = {
                "name": ctrl.window_text()[:300],
                "auto_id": aid,
                "control_type": ct,
                "visible": ctrl.is_visible(),
                "enabled": ctrl.is_enabled(),
            }
            try:
                r = ctrl.rectangle()
                info["rect"] = {"l": r.left, "t": r.top, "w": r.width(), "h": r.height()}
            except Exception:
                pass

            if ct == "Button":
                report["buttons"].append(info)
            elif ct == "CheckBox":
                try:
                    info["checked"] = ctrl.get_toggle_state() == 1
                except Exception:
                    info["checked"] = None
                report["checkboxes"].append(info)
            elif ct in ["ComboBox", "ListBox"]:
                try:
                    info["selected"] = ctrl.selected_text()
                except Exception:
                    info["selected"] = "?"
                try:
                    info["items"] = ctrl.item_texts()
                except Exception:
                    info["items"] = []
                report["dropdowns"].append(info)
            elif ct in ["Edit", "TextBox"]:
                try:
                    info["value"] = ctrl.get_value()[:500] if hasattr(ctrl, "get_value") else ""
                except Exception:
                    info["value"] = ""
                report["text_fields"].append(info)
            elif ct in ["Static", "Text"]:
                report["labels"].append(info)
            else:
                report["other"].append(info)
        except Exception:
            pass

    # Print summary
    visible_buttons = [b for b in report["buttons"] if b.get("visible") and b.get("name")]
    visible_cbs = [c for c in report["checkboxes"] if c.get("visible")]
    visible_dds = [d for d in report["dropdowns"] if d.get("visible")]
    visible_tfs = [t for t in report["text_fields"] if t.get("visible")]

    print(f"  Buttons: {len(visible_buttons)}, Checkboxes: {len(visible_cbs)}, "
          f"Dropdowns: {len(visible_dds)}, Text fields: {len(visible_tfs)}", flush=True)

    for b in sorted(visible_buttons, key=lambda x: x.get("auto_id", "")):
        print(f"    Button id={b['auto_id']:>5s} '{b['name']}'", flush=True)
    for cb in sorted(visible_cbs, key=lambda x: x.get("auto_id", "")):
        s = "[X]" if cb.get("checked") else "[ ]"
        print(f"    Checkbox id={cb['auto_id']:>5s} {s} '{cb['name']}'", flush=True)
    for dd in sorted(visible_dds, key=lambda x: x.get("auto_id", "")):
        print(f"    Dropdown id={dd['auto_id']:>5s} '{dd['name']}' = {dd.get('selected', '?')}", flush=True)
    for tf in sorted(visible_tfs, key=lambda x: x.get("auto_id", "")):
        print(f"    Edit id={tf['auto_id']:>5s} '{tf['name']}' = '{tf.get('value', '')[:60]}'", flush=True)

    # Save individual file
    filepath = os.path.join(OUTPUT_DIR, f"zimmwriter_{screen_name}_controls.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {filepath}", flush=True)

    return report


def navigate_to_options_menu():
    """Navigate to Options Menu from wherever we are. Uses win32 backend."""
    _, app = connect_win32()
    window = app.top_window()
    title = window.window_text()

    # If on Bulk Writer, close it to get back to Menu
    if "Bulk" in title:
        print("Closing Bulk Writer to get to Menu...", flush=True)
        close_window_by_handle(window.handle)
        time.sleep(3)
        _, app = connect_win32()
        window = app.top_window()
        title = window.window_text()

    # If on Menu, click Options Menu
    if "Menu" in title and "Option" not in title:
        print("On Menu screen, clicking 'Options Menu'...", flush=True)
        btn = window["Options Menu"]
        btn.click()
        time.sleep(3)
        _, app = connect_win32()
        window = app.top_window()

    print(f"Current screen: {window.window_text()}", flush=True)
    return app, window


def main():
    print("=" * 70)
    print("  OPTIONS MENU SUB-SCREEN DISCOVERY")
    print("=" * 70)

    pid = find_pid()
    print(f"Connected to PID {pid}", flush=True)

    # Navigate to Options Menu (uses win32 backend)
    app, options_window = navigate_to_options_menu()
    options_handle = options_window.handle
    options_title = options_window.window_text()

    if "Option" not in options_title:
        print(f"ERROR: Expected Options Menu, got '{options_title}'")
        return

    all_reports = {}

    for auto_id, screen_name in SUBSCREENS:
        print(f"\n{'-'*60}")
        print(f"[{screen_name}] Clicking button auto_id={auto_id}...", flush=True)

        try:
            # Use win32 for navigation (reliable on all screens)
            _, app32 = connect_win32()
            options_win = app32.window(handle=options_handle)

            # Find and click the sub-screen button by iterating children
            children = options_win.children()
            btn = None
            for child in children:
                try:
                    if child.friendly_class_name() == "Button":
                        text = child.window_text()
                        # Match by position in button list or by text
                        cid = child.control_id()
                        if str(cid) == auto_id:
                            btn = child
                            break
                except Exception:
                    pass

            if btn is None:
                # Fallback: try by title text matching
                button_map = {
                    "56": "Text API", "57": "Image API", "58": "Scraping API",
                    "59": "Scraping Surgeon", "60": "Scraping Domain",
                    "61": "AI Image Prompt", "62": "AI Words to Nuke",
                    "63": "Secure Mode",
                }
                search_text = button_map.get(auto_id, "")
                for child in children:
                    try:
                        if search_text.lower() in child.window_text().lower():
                            btn = child
                            break
                    except Exception:
                        pass

            if btn is None:
                print(f"  WARNING: Could not find button auto_id={auto_id}", flush=True)
                all_reports[screen_name] = {"error": f"button auto_id={auto_id} not found"}
                continue

            btn_text = btn.window_text()
            print(f"  Button text: '{btn_text}'", flush=True)
            btn.click()
            time.sleep(3)

            # Try UIA backend first for richer control info, fall back to win32
            sub_window = None
            sub_handle = None
            backend_used = None

            # Try UIA
            try:
                _, app_uia = connect_uia()
                for w in app_uia.windows():
                    h = w.handle
                    if h != options_handle:
                        sub_window = w
                        sub_handle = h
                        backend_used = "uia"
                        break
            except Exception as e:
                print(f"  UIA failed ({e}), trying win32...", flush=True)

            # Fall back to win32
            if sub_window is None:
                _, app32b = connect_win32()
                for w in app32b.windows():
                    h = w.handle
                    if h != options_handle:
                        sub_window = w
                        sub_handle = h
                        backend_used = "win32"
                        break

            if sub_window:
                print(f"  Opened: '{sub_window.window_text()}' (via {backend_used})", flush=True)

                # Screenshot via pyautogui (more reliable across backends)
                try:
                    import pyautogui
                    sub_window.set_focus()
                    time.sleep(0.5)
                    screenshot = pyautogui.screenshot()
                    screenshot.save(os.path.join(OUTPUT_DIR, f"options_{screen_name}.png"))
                    print(f"  Screenshot saved", flush=True)
                except Exception as e:
                    print(f"  Screenshot error: {e}", flush=True)

                # Discover controls
                report = discover_screen(sub_window, screen_name)
                report["backend_used"] = backend_used
                all_reports[screen_name] = report

                # Close the sub-screen
                close_window_by_handle(sub_handle)
                time.sleep(2)

                # Verify we're back on Options Menu
                _, app32c = connect_win32()
                curr = app32c.top_window().window_text()
                if "Option" not in curr:
                    print(f"  WARNING: Not back on Options Menu (got '{curr}'), re-navigating...", flush=True)
                    _, options_win = navigate_to_options_menu()
                    options_handle = options_win.handle
            else:
                print(f"  WARNING: No new window opened for {screen_name}", flush=True)
                all_reports[screen_name] = {"error": "no window opened"}

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            all_reports[screen_name] = {"error": str(e)}
            # Try to recover
            try:
                _, app32r = connect_win32()
                for w in app32r.windows():
                    h = w.handle
                    if h != options_handle:
                        close_window_by_handle(h)
                time.sleep(1)
            except Exception:
                pass

    # Save combined report
    combined_path = os.path.join(OUTPUT_DIR, "zimmwriter_all_subscreens.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nCombined report: {combined_path}")

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for screen_name, report in all_reports.items():
        if "error" in report:
            print(f"  {screen_name:35s} ERROR: {report['error']}")
        else:
            n_btns = len([b for b in report.get("buttons", []) if b.get("visible")])
            n_cbs = len([c for c in report.get("checkboxes", []) if c.get("visible")])
            n_dds = len([d for d in report.get("dropdowns", []) if d.get("visible")])
            n_tfs = len([t for t in report.get("text_fields", []) if t.get("visible")])
            print(f"  {screen_name:35s} B={n_btns} CB={n_cbs} DD={n_dds} TF={n_tfs}")

    print(f"\n{'='*70}")
    print("Done! Review outputs in the output/ folder.")
    print("Run Bulk Writer again if needed: navigate from Menu screen.")


if __name__ == "__main__":
    main()
