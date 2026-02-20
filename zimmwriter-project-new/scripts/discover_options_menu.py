"""
Discover all controls on the ZimmWriter Options Menu screen.
Navigates from Menu -> Options Menu, then maps every control.
"""

import json
import sys
import os
import subprocess
import time
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


def navigate_to_options(app, window):
    """Navigate to Options Menu from main menu."""
    title = window.window_text()
    if "Menu" in title and "Options" not in title:
        print("On main Menu screen, clicking 'Options Menu'...")
        try:
            btn = window.child_window(title="Options Menu", control_type="Button")
            btn.invoke()
            time.sleep(3)
            window = app.top_window()
            print(f"Now on: {window.window_text()}")
        except Exception as e:
            print(f"Could not find Options Menu button, trying all buttons...")
            for b in window.descendants(control_type="Button"):
                txt = b.window_text()
                if "option" in txt.lower():
                    print(f"  Found: '{txt}' - clicking...")
                    b.invoke()
                    time.sleep(3)
                    window = app.top_window()
                    print(f"  Now on: {window.window_text()}")
                    break
    elif "Options" in title or "Settings" in title:
        print(f"Already on Options screen: {title}")
    else:
        print(f"Unknown screen: {title}")
        print("Attempting to find Options button anyway...")
        for b in window.descendants(control_type="Button"):
            txt = b.window_text()
            if "option" in txt.lower() or "menu" in txt.lower():
                print(f"  Found button: '{txt}'")

    return window


def discover_screen(window, screen_name="options_menu"):
    """Map all controls on current screen."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "window_title": window.window_text(),
        "screen": screen_name,
        "buttons": [], "checkboxes": [], "dropdowns": [],
        "text_fields": [], "labels": [], "tabs": [], "other": [],
    }

    all_ctrls = window.descendants()
    print(f"\nScanning {len(all_ctrls)} elements on '{screen_name}'...\n")

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
            elif ct in ["TabControl", "Tab", "TabItem"]:
                report["tabs"].append(info)
            else:
                report["other"].append(info)
        except Exception:
            pass

    # Print visible controls
    print("=" * 70)
    print(f"  OPTIONS MENU CONTROLS ({screen_name})")
    print("=" * 70)

    print("\n  BUTTONS:")
    print("  " + "-" * 60)
    for b in sorted(report["buttons"], key=lambda x: x.get("auto_id", "")):
        if b["visible"] and b["name"]:
            print(f'    auto_id={b["auto_id"]:>5s}  "{b["name"]}"')

    print("\n  CHECKBOXES:")
    print("  " + "-" * 60)
    for cb in sorted(report["checkboxes"], key=lambda x: x.get("auto_id", "")):
        if cb["visible"] and cb["name"]:
            s = "[X]" if cb.get("checked") else "[ ]"
            print(f'    auto_id={cb["auto_id"]:>5s}  {s} "{cb["name"]}"')

    print("\n  DROPDOWNS:")
    print("  " + "-" * 60)
    for dd in sorted(report["dropdowns"], key=lambda x: x.get("auto_id", "")):
        if dd["visible"]:
            print(f'    auto_id={dd["auto_id"]:>5s}  "{dd["name"]}" = {dd.get("selected", "?")}')
            if dd.get("items"):
                items_str = str(dd["items"][:10])
                print(f'              Options: {items_str}')

    print("\n  TEXT FIELDS:")
    print("  " + "-" * 60)
    for tf in sorted(report["text_fields"], key=lambda x: x.get("auto_id", "")):
        if tf["visible"]:
            val_preview = tf.get("value", "")[:80]
            print(f'    auto_id={tf["auto_id"]:>5s}  "{tf["name"]}" = "{val_preview}"')

    print("\n  TABS:")
    print("  " + "-" * 60)
    for tab in report["tabs"]:
        if tab["visible"]:
            print(f'    auto_id={tab["auto_id"]:>5s}  "{tab["name"]}" ({tab["control_type"]})')

    print("\n  LABELS (WordPress/API related):")
    print("  " + "-" * 60)
    for lbl in report["labels"]:
        if lbl["visible"] and lbl["name"]:
            name_lower = lbl["name"].lower()
            if any(kw in name_lower for kw in ["wordpress", "api", "key", "url", "user", "pass",
                                                  "site", "pexel", "stabil", "openai", "anthrop",
                                                  "groq", "open router", "scrape", "perp", "save",
                                                  "delete", "update", "config", "model"]):
                print(f'    auto_id={lbl["auto_id"]:>5s}  "{lbl["name"][:100]}"')

    # Summary
    print("\n" + "=" * 70)
    for cat in ["buttons", "checkboxes", "dropdowns", "text_fields", "tabs"]:
        visible = [c for c in report[cat] if c.get("visible")]
        print(f"  {cat:15s}: {len(visible)} visible / {len(report[cat])} total")

    # Save
    map_file = os.path.join(OUTPUT_DIR, f"zimmwriter_{screen_name}_controls.json")
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {map_file}")

    tree_file = os.path.join(OUTPUT_DIR, f"zimmwriter_{screen_name}_tree.txt")
    buf = StringIO()
    window.print_control_identifiers(depth=10, filename=buf)
    with open(tree_file, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"Saved: {tree_file}")

    return report


if __name__ == "__main__":
    app, window = connect()
    window = navigate_to_options(app, window)
    report = discover_screen(window, "options_menu")
    print("\nDone! Review outputs in the output/ folder.")
