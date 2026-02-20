"""
Discover all controls in ZimmWriter's 11 feature config windows.

Opens each feature toggle button, captures the resulting config window,
and enumerates all controls (buttons, checkboxes, dropdowns, text fields)
with their auto_ids, names, control types, and current values.

Output: JSON file at output/feature_window_controls.json

Usage:
    python scripts/discover_feature_windows.py
    python scripts/discover_feature_windows.py --feature serp_scraping
    python scripts/discover_feature_windows.py --feature deep_research --list-dropdown-values

REQUIREMENTS:
    ZimmWriter must be running on the Bulk Writer screen.
"""

import sys
import os
import json
import time
import ctypes
from ctypes import wintypes
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController


# Feature toggle button auto_ids and expected config window titles (v10.870)
FEATURES = {
    "wordpress":      {"auto_id": "95", "window_title": "Enable WordPress Uploads"},
    "link_pack":      {"auto_id": "96", "window_title": "Load Link Pack"},
    "serp_scraping":  {"auto_id": "97", "window_title": "Enable SERP Scraping"},
    "deep_research":  {"auto_id": "98", "window_title": "Deep Research"},
    "style_mimic":    {"auto_id": "99", "window_title": "Style Mimic"},
    "custom_outline": {"auto_id": "100", "window_title": "Set Custom Outline"},
    "custom_prompt":  {"auto_id": "101", "window_title": "Set Custom Prompts"},
    "youtube_videos": {"auto_id": "102", "window_title": "Enable YouTube Videos"},
    "webhook":        {"auto_id": "103", "window_title": "Enable Webhook"},
    "alt_images":     {"auto_id": "104", "window_title": "Enable Alt Images"},
    "seo_csv":        {"auto_id": "105", "window_title": "Set Bulk SEO CSV"},
}


def read_combo_items(hwnd: int) -> list:
    """Read all items from a ComboBox handle via Win32 CB messages."""
    SendMsg = ctypes.windll.user32.SendMessageW
    SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMsg.restype = ctypes.c_long

    count = SendMsg(hwnd, 0x0146, 0, 0)  # CB_GETCOUNT
    items = []
    for i in range(count):
        length = SendMsg(hwnd, 0x0149, i, 0)  # CB_GETLBTEXTLEN
        if length >= 0:
            buf = ctypes.create_unicode_buffer(length + 2)
            SendMsg(hwnd, 0x0148, i, ctypes.addressof(buf))  # CB_GETLBTEXT
            items.append(buf.value)
    return items


def read_combo_selected(hwnd: int) -> str:
    """Read the currently selected item from a ComboBox."""
    SendMsg = ctypes.windll.user32.SendMessageW
    SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMsg.restype = ctypes.c_long

    cur = SendMsg(hwnd, 0x0147, 0, 0)  # CB_GETCURSEL
    if cur < 0:
        return ""
    length = SendMsg(hwnd, 0x0149, cur, 0)  # CB_GETLBTEXTLEN
    if length < 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 2)
    SendMsg(hwnd, 0x0148, cur, ctypes.addressof(buf))  # CB_GETLBTEXT
    return buf.value


def enumerate_window_controls(win, list_dropdown_values: bool = False) -> list:
    """Enumerate all child controls of a window."""
    controls = []
    try:
        children = win.children()
    except Exception:
        return controls

    for child in children:
        try:
            info = {
                "control_type": child.friendly_class_name(),
                "auto_id": str(child.control_id()),
                "name": child.window_text()[:200],
                "rect": str(child.rectangle()),
                "is_visible": child.is_visible(),
                "is_enabled": child.is_enabled(),
            }

            # Read combo box details
            if child.friendly_class_name() in ("ComboBox",):
                try:
                    info["selected"] = read_combo_selected(child.handle)
                    if list_dropdown_values:
                        info["items"] = read_combo_items(child.handle)
                        info["item_count"] = len(info["items"])
                except Exception:
                    pass

            # Read checkbox state
            if child.friendly_class_name() in ("CheckBox",):
                try:
                    info["checked"] = child.get_check_state() == 1
                except Exception:
                    pass

            # Read text field content
            if child.friendly_class_name() in ("Edit",):
                try:
                    text = child.window_text()
                    info["text_value"] = text[:500] if text else ""
                except Exception:
                    pass

            controls.append(info)
        except Exception as e:
            controls.append({"error": str(e)})

    return controls


def discover_feature_window(zw: ZimmWriterController, feature_name: str,
                             list_dropdown_values: bool = False) -> dict:
    """
    Open a feature config window, enumerate its controls, then close it.
    Returns dict with feature name, window title, and controls list.
    """
    feature = FEATURES.get(feature_name)
    if not feature:
        return {"feature": feature_name, "error": f"Unknown feature: {feature_name}"}

    result = {
        "feature": feature_name,
        "auto_id": feature["auto_id"],
        "expected_window": feature["window_title"],
        "status": "unknown",
        "controls": [],
    }

    try:
        # Click the feature toggle button to open config window
        btn = zw._find_child(control_type="Button", auto_id=feature["auto_id"])
        btn_text = btn.window_text()
        result["button_text"] = btn_text

        btn.click_input()
        time.sleep(1.5)

        # Find the config window
        import re
        title_pattern = re.escape(feature["window_title"])
        win = zw._wait_for_window(title_pattern, timeout=5)

        if win:
            result["actual_window_title"] = win.window_text()
            result["controls"] = enumerate_window_controls(win, list_dropdown_values)
            result["control_count"] = len(result["controls"])
            result["status"] = "OK"

            # Close the config window
            WM_CLOSE = 0x0010
            SendMsg = ctypes.windll.user32.SendMessageW
            SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            SendMsg.restype = ctypes.c_long
            SendMsg(win.handle, WM_CLOSE, 0, 0)
            time.sleep(0.5)

            # Dismiss any follow-up dialogs
            zw._dismiss_dialog(timeout=2)
        else:
            result["status"] = "window_not_found"
            # Try to dismiss if toggle opened something unexpected
            zw._dismiss_dialog(timeout=2)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        try:
            zw._dismiss_dialog(timeout=2)
        except Exception:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Discover controls in ZimmWriter feature config windows"
    )
    parser.add_argument("--feature", type=str,
                        help="Discover a single feature (e.g., serp_scraping, deep_research)")
    parser.add_argument("--list-dropdown-values", action="store_true",
                        help="Also enumerate all dropdown values (slower)")
    args = parser.parse_args()

    # Connect
    zw = ZimmWriterController()
    if not zw.connect():
        print("ERROR: Could not connect to ZimmWriter. Is it running?")
        sys.exit(1)

    print(f"Connected to: {zw.get_window_title()}")

    # Determine which features to discover
    if args.feature:
        if args.feature not in FEATURES:
            print(f"ERROR: Unknown feature '{args.feature}'")
            print(f"Available: {', '.join(FEATURES.keys())}")
            sys.exit(1)
        feature_list = [args.feature]
    else:
        feature_list = list(FEATURES.keys())

    # Discover each feature window
    all_results = {}
    for i, feat in enumerate(feature_list, 1):
        print(f"[{i}/{len(feature_list)}] {feat}...", end=" ", flush=True)
        result = discover_feature_window(zw, feat, args.list_dropdown_values)
        all_results[feat] = result
        icon = "OK" if result["status"] == "OK" else "XX"
        ctrl_count = result.get("control_count", 0)
        print(f"[{icon}] {ctrl_count} controls")

        if result["status"] == "OK":
            # Print control summary
            for ctrl in result["controls"]:
                ctype = ctrl.get("control_type", "?")
                cid = ctrl.get("auto_id", "?")
                name = ctrl.get("name", "")[:60]
                extra = ""
                if "selected" in ctrl:
                    extra = f" (selected: {ctrl['selected']})"
                if "checked" in ctrl:
                    extra = f" (checked: {ctrl['checked']})"
                if "item_count" in ctrl:
                    extra += f" [{ctrl['item_count']} items]"
                print(f"    {ctype:15s} id={cid:4s}  {name}{extra}")

        time.sleep(1)

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "feature_window_controls.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Summary
    ok = sum(1 for r in all_results.values() if r["status"] == "OK")
    print(f"\nDiscovered: {ok}/{len(all_results)} feature windows")


if __name__ == "__main__":
    main()
