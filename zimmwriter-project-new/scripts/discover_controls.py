"""
ZimmWriter UI Discovery Tool
Run with ZimmWriter open to map every control.
Produces: output/zimmwriter_control_map.json, output/zimmwriter_cheatsheet.txt
"""

import json
import sys
import os
from datetime import datetime
from io import StringIO

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pywinauto import Application
except ImportError:
    print("ERROR: pip install pywinauto")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def discover():
    print("=" * 60)
    print("  ZimmWriter UI Discovery Tool")
    print("=" * 60)

    # ZimmWriter runs as AutoIt3.exe — use PID-based connection
    import subprocess
    app = None

    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | "
             "Where-Object { $_.MainWindowTitle -like '*ZimmWriter*' } | "
             "Select-Object -First 1 -ExpandProperty Id"],
            capture_output=True, text=True, timeout=10
        )
        pid_str = result.stdout.strip()
        if pid_str:
            pid = int(pid_str)
            app = Application(backend="uia").connect(process=pid)
            print(f"Connected to PID {pid}")
    except Exception as e:
        print(f"PID connection failed: {e}")

    if not app:
        # Fallback: title matching
        for title in ["ZimmWriter", "Zimm Writer", "Bulk Blog Writer"]:
            try:
                app = Application(backend="uia").connect(title_re=f".*{title}.*", timeout=5)
                print(f"Connected: {title}")
                break
            except Exception:
                continue

    if not app:
        print("ZimmWriter not running!")
        sys.exit(1)

    window = app.top_window()
    print(f"   Window: {window.window_text()}\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "window_title": window.window_text(),
        "buttons": [], "checkboxes": [], "dropdowns": [],
        "text_fields": [], "labels": [], "other": [],
    }

    all_ctrls = window.descendants()
    print(f"Scanning {len(all_ctrls)} elements...\n")

    for ctrl in all_ctrls:
        try:
            ct = ctrl.friendly_class_name()
            info = {
                "name": ctrl.window_text()[:200],
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

    # Summary
    for cat in ["buttons", "checkboxes", "dropdowns", "text_fields"]:
        visible = [c for c in report[cat] if c.get("visible")]
        print(f"  {cat:15s}: {len(visible)} visible / {len(report[cat])} total")

    # Print visible controls
    print("\n" + "=" * 60)
    print("  VISIBLE BUTTONS")
    print("=" * 60)
    for b in report["buttons"]:
        if b["visible"] and b["name"]:
            print(f'  📌 "{b["name"]}" | auto_id: {b["auto_id"]}')

    print("\n  VISIBLE CHECKBOXES")
    print("  " + "-" * 40)
    for cb in report["checkboxes"]:
        if cb["visible"] and cb["name"]:
            s = "☑" if cb.get("checked") else "☐"
            print(f'  {s} "{cb["name"]}" | auto_id: {cb["auto_id"]}')

    print("\n  VISIBLE DROPDOWNS")
    print("  " + "-" * 40)
    for dd in report["dropdowns"]:
        if dd["visible"]:
            print(f'  📋 "{dd["name"]}" = {dd.get("selected", "?")} | auto_id: {dd["auto_id"]}')
            if dd.get("items"):
                print(f'      Options: {dd["items"][:8]}')

    print("\n  VISIBLE TEXT FIELDS")
    print("  " + "-" * 40)
    for tf in report["text_fields"]:
        if tf["visible"]:
            print(f'  ✏️  "{tf["name"]}" | auto_id: {tf["auto_id"]}')

    # Save files
    map_file = os.path.join(OUTPUT_DIR, "zimmwriter_control_map.json")
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ Control map: {map_file}")

    tree_file = os.path.join(OUTPUT_DIR, "zimmwriter_control_tree.txt")
    buf = StringIO()
    window.print_control_identifiers(depth=10, filename=buf)
    with open(tree_file, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"✅ Control tree: {tree_file}")

    cheat_file = os.path.join(OUTPUT_DIR, "zimmwriter_cheatsheet.txt")
    with open(cheat_file, "w", encoding="utf-8") as f:
        f.write("ZIMMWRITER CONTROL CHEATSHEET\n" + "=" * 50 + "\n\n")
        f.write("BUTTONS:\n")
        for b in report["buttons"]:
            if b["visible"] and b["name"]:
                f.write(f'  click_button(name="{b["name"]}")\n')
        f.write("\nCHECKBOXES:\n")
        for cb in report["checkboxes"]:
            if cb["visible"] and cb["name"]:
                f.write(f'  set_checkbox(name="{cb["name"]}", checked=True)\n')
        f.write("\nDROPDOWNS:\n")
        for dd in report["dropdowns"]:
            if dd["visible"] and dd["name"]:
                f.write(f'  set_dropdown(name="{dd["name"]}", value="...")\n')
                if dd.get("items"):
                    f.write(f"    Options: {dd['items']}\n")
        f.write("\nTEXT FIELDS:\n")
        for tf in report["text_fields"]:
            if tf["visible"]:
                f.write(f'  set_text_field(name="{tf["name"]}", value="...")\n')
    print(f"✅ Cheatsheet: {cheat_file}")

    print("\nDONE. Review outputs in the output/ folder.")
    return report


if __name__ == "__main__":
    discover()
