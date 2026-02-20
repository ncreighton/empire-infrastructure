"""
Navigate to Options Menu -> WordPress Settings and discover all controls.
Then map the WordPress settings sub-screen.
"""

import json
import sys
import os
import subprocess
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pywinauto import Application
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
    print(f"Connected to PID {pid}: {window.window_text()}")
    return app, window


def navigate_to_wordpress_settings(app, window):
    """Navigate from wherever we are to WordPress Settings."""
    title = window.window_text()
    print(f"Current screen: {title}")

    # If on main Menu, go to Options Menu first
    if "Menu" in title and "Option" not in title:
        print("Navigating: Menu -> Options Menu...")
        btn = window.child_window(title="Options Menu", control_type="Button")
        btn.invoke()
        time.sleep(3)
        window = app.top_window()
        title = window.window_text()
        print(f"Now on: {title}")

    # If on Options Menu, click WordPress Settings
    if "Option" in title:
        print("Navigating: Options Menu -> WordPress Settings...")
        btn = window.child_window(auto_id="55", control_type="Button")
        btn.invoke()
        time.sleep(3)
        window = app.top_window()
        title = window.window_text()
        print(f"Now on: {title}")

    return window


def discover_wordpress_screen(window):
    """Map all controls on the WordPress settings screen."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "window_title": window.window_text(),
        "screen": "wordpress_settings",
        "buttons": [], "checkboxes": [], "dropdowns": [],
        "text_fields": [], "labels": [], "other": [],
    }

    all_ctrls = window.descendants()
    print(f"\nScanning {len(all_ctrls)} elements...\n")

    for ctrl in all_ctrls:
        try:
            ct = ctrl.friendly_class_name()
            info = {
                "name": ctrl.window_text()[:300],
                "auto_id": ctrl.automation_id(),
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

    # Print everything
    print("=" * 70)
    print("  WORDPRESS SETTINGS CONTROLS")
    print("=" * 70)

    print("\n  ALL BUTTONS:")
    for b in sorted(report["buttons"], key=lambda x: x.get("auto_id", "")):
        if b["visible"]:
            print(f'    auto_id={b["auto_id"]:>5s}  "{b["name"]}"')

    print("\n  ALL CHECKBOXES:")
    for cb in sorted(report["checkboxes"], key=lambda x: x.get("auto_id", "")):
        if cb["visible"]:
            s = "[X]" if cb.get("checked") else "[ ]"
            print(f'    auto_id={cb["auto_id"]:>5s}  {s} "{cb["name"]}"')

    print("\n  ALL DROPDOWNS:")
    for dd in sorted(report["dropdowns"], key=lambda x: x.get("auto_id", "")):
        if dd["visible"]:
            print(f'    auto_id={dd["auto_id"]:>5s}  "{dd["name"]}" = {dd.get("selected", "?")}')
            if dd.get("items"):
                for item in dd["items"][:20]:
                    print(f'              - "{item}"')

    print("\n  ALL TEXT FIELDS:")
    for tf in sorted(report["text_fields"], key=lambda x: x.get("auto_id", "")):
        if tf["visible"]:
            val = tf.get("value", "")[:100]
            print(f'    auto_id={tf["auto_id"]:>5s}  "{tf["name"]}" = "{val}"')

    print("\n  ALL LABELS:")
    for lbl in sorted(report["labels"], key=lambda x: x.get("auto_id", "")):
        if lbl["visible"] and lbl["name"]:
            print(f'    auto_id={lbl["auto_id"]:>5s}  "{lbl["name"][:120]}"')

    # Summary
    print("\n" + "=" * 70)
    for cat in ["buttons", "checkboxes", "dropdowns", "text_fields", "labels"]:
        visible = [c for c in report[cat] if c.get("visible")]
        print(f"  {cat:15s}: {len(visible)} visible / {len(report[cat])} total")

    # Save
    map_file = os.path.join(OUTPUT_DIR, "zimmwriter_wordpress_settings.json")
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {map_file}")

    # Save control tree to file
    tree_file = os.path.join(OUTPUT_DIR, "zimmwriter_wordpress_tree.txt")
    try:
        window.print_control_identifiers(depth=10, filename=tree_file)
        print(f"Saved: {tree_file}")
    except Exception as e:
        print(f"Tree save error (non-fatal): {e}")

    return report


if __name__ == "__main__":
    app, window = connect()
    window = navigate_to_wordpress_settings(app, window)
    report = discover_wordpress_screen(window)
    print("\nDone!")
