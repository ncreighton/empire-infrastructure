"""
Discover ALL feature config window controls in ZimmWriter v10.872.

Opens each feature toggle button, captures all child controls (type, ID, text),
then closes the window. Saves results to output/feature_window_controls.json.

Usage:
    python scripts/discover_all_feature_windows.py
"""
import sys
import os
import json
import time
import ctypes
from ctypes import wintypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Win32 message constants
_SM = ctypes.windll.user32.SendMessageW
_SM.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_SM.restype = ctypes.c_long

CB_GETCOUNT = 0x0146
CB_GETLBTEXTLEN = 0x0149
CB_GETLBTEXT = 0x0148
CB_GETCURSEL = 0x0147
WM_GETTEXTLENGTH = 0x000E
WM_GETTEXT = 0x000D
BM_GETCHECK = 0x00F0


def read_combo_items(hwnd, max_items=50):
    """Read all items from a ComboBox."""
    count = _SM(hwnd, CB_GETCOUNT, 0, 0)
    items = []
    for i in range(min(count, max_items)):
        item_len = _SM(hwnd, CB_GETLBTEXTLEN, i, 0)
        if item_len <= 0:
            continue
        buf = ctypes.create_unicode_buffer(item_len + 2)
        _SM(hwnd, CB_GETLBTEXT, i, ctypes.addressof(buf))
        items.append(buf.value)
    cur = _SM(hwnd, CB_GETCURSEL, 0, 0)
    return {"count": count, "selected_index": cur, "items": items}


def read_edit_text(hwnd, max_chars=500):
    """Read text from an Edit control."""
    text_len = _SM(hwnd, WM_GETTEXTLENGTH, 0, 0)
    if text_len <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(min(text_len, max_chars) + 2)
    _SM(hwnd, WM_GETTEXT, min(text_len, max_chars) + 1, ctypes.addressof(buf))
    return buf.value


def read_checkbox_state(hwnd):
    """Read checkbox state (0=unchecked, 1=checked)."""
    return _SM(hwnd, BM_GETCHECK, 0, 0)


def enumerate_window_controls(win):
    """Enumerate all child controls in a window."""
    controls = []
    try:
        children = win.children()
        for child in children:
            try:
                ctype = child.friendly_class_name()
                cid = child.control_id()
                text = child.window_text()[:200]
                hwnd = child.handle

                ctrl_info = {
                    "type": ctype,
                    "id": cid,
                    "text": text,
                    "handle": hwnd,
                }

                # Read extra info based on control type
                if ctype == "ComboBox":
                    ctrl_info["combo_data"] = read_combo_items(hwnd)
                elif ctype == "Edit":
                    ctrl_info["edit_text"] = read_edit_text(hwnd)
                elif ctype == "CheckBox":
                    ctrl_info["checked"] = read_checkbox_state(hwnd)

                controls.append(ctrl_info)
            except Exception as e:
                controls.append({"type": "ERROR", "error": str(e)})
    except Exception as e:
        controls.append({"type": "WINDOW_ERROR", "error": str(e)})
    return controls


def discover_feature_window(zw, feature_key, feature_id, window_title):
    """Open a feature config window and enumerate its controls."""
    print(f"\n{'='*60}")
    print(f"Feature: {feature_key} (button id={feature_id})")
    print(f"Expected window: '{window_title}'")
    print(f"{'='*60}")

    zw.ensure_connected()
    zw.bring_to_front()
    time.sleep(0.3)

    # Click the feature toggle button
    try:
        btn = zw._find_child(control_type="Button", auto_id=feature_id)
        btn_text = btn.window_text()
        print(f"  Button text: '{btn_text}'")
        btn.click_input()
        time.sleep(2)
    except Exception as e:
        print(f"  ERROR clicking button: {e}")
        return None

    # Wait for config window
    import re
    escaped = re.escape(window_title)
    win = zw._wait_for_window(escaped, timeout=10)
    if not win:
        # Try partial match
        for w in zw.app.windows():
            t = w.window_text()
            if t and "Bulk" not in t and "Menu" not in t and t != "":
                win = w
                print(f"  Found alternate window: '{t}'")
                break

    if not win:
        print(f"  ERROR: Config window not found!")
        zw._dismiss_dialog(timeout=2)
        return None

    actual_title = win.window_text()
    print(f"  Window opened: '{actual_title}'")

    # Enumerate all controls
    controls = enumerate_window_controls(win)
    print(f"  Found {len(controls)} controls:")
    for ctrl in controls:
        if ctrl["type"] == "ERROR":
            print(f"    [ERROR] {ctrl['error']}")
            continue
        line = f"    {ctrl['type']:15s} id={ctrl['id']:4d}  '{ctrl['text'][:80]}'"
        if ctrl["type"] == "ComboBox" and "combo_data" in ctrl:
            cd = ctrl["combo_data"]
            line += f"  [{cd['count']} items, sel={cd['selected_index']}]"
            if cd["items"]:
                for item in cd["items"][:10]:
                    print(f"    {'':15s}       -> '{item}'")
        elif ctrl["type"] == "Edit" and "edit_text" in ctrl:
            et = ctrl["edit_text"]
            if et:
                line += f"  [{len(et)} chars: '{et[:80]}']"
        elif ctrl["type"] == "CheckBox" and "checked" in ctrl:
            line += f"  [{'CHECKED' if ctrl['checked'] else 'unchecked'}]"
        print(line)

    # Close the window
    zw._close_config_window(win)
    time.sleep(0.5)
    zw._dismiss_dialog(timeout=2)

    # Remove non-serializable handle field
    for ctrl in controls:
        ctrl.pop("handle", None)

    return {
        "feature_key": feature_key,
        "button_id": feature_id,
        "expected_title": window_title,
        "actual_title": actual_title,
        "controls": controls,
    }


def discover_image_options_window(zw, button_id, label):
    """Open an Image Options (O button) window and enumerate controls."""
    print(f"\n{'='*60}")
    print(f"Image Options: {label} (button id={button_id})")
    print(f"{'='*60}")

    zw.ensure_connected()
    zw.bring_to_front()
    time.sleep(0.3)

    try:
        btn = zw._find_child(control_type="Button", auto_id=button_id)
        btn_text = btn.window_text()
        print(f"  Button text: '{btn_text}'")
        btn.click_input()
        time.sleep(2)
    except Exception as e:
        print(f"  ERROR clicking button: {e}")
        return None

    win = zw._wait_for_window("Image Options", timeout=8)
    if not win:
        for w in zw.app.windows():
            t = w.window_text()
            if t and "Bulk" not in t and "Menu" not in t and t != "":
                win = w
                print(f"  Found alternate window: '{t}'")
                break

    if not win:
        print(f"  ERROR: Image Options window not found!")
        zw._dismiss_dialog(timeout=2)
        return None

    actual_title = win.window_text()
    print(f"  Window opened: '{actual_title}'")

    controls = enumerate_window_controls(win)
    print(f"  Found {len(controls)} controls:")
    for ctrl in controls:
        if ctrl["type"] == "ERROR":
            print(f"    [ERROR] {ctrl['error']}")
            continue
        line = f"    {ctrl['type']:15s} id={ctrl['id']:4d}  '{ctrl['text'][:80]}'"
        if ctrl["type"] == "ComboBox" and "combo_data" in ctrl:
            cd = ctrl["combo_data"]
            line += f"  [{cd['count']} items, sel={cd['selected_index']}]"
            for item in cd["items"][:15]:
                print(f"    {'':15s}       -> '{item}'")
        elif ctrl["type"] == "Edit" and "edit_text" in ctrl:
            et = ctrl["edit_text"]
            if et:
                line += f"  [{len(et)} chars: '{et[:80]}']"
        elif ctrl["type"] == "CheckBox" and "checked" in ctrl:
            line += f"  [{'CHECKED' if ctrl['checked'] else 'unchecked'}]"
        print(line)

    zw._close_config_window(win)
    time.sleep(0.5)
    zw._dismiss_dialog(timeout=2)

    for ctrl in controls:
        ctrl.pop("handle", None)

    return {
        "window_type": "image_options",
        "label": label,
        "button_id": button_id,
        "actual_title": actual_title,
        "controls": controls,
    }


def main():
    zw = ZimmWriterController()
    if not zw.connect():
        print("ERROR: Could not connect to ZimmWriter")
        sys.exit(1)

    title = zw.get_window_title()
    print(f"Connected: {title}")

    if "Bulk" not in title:
        print("Not on Bulk Writer, navigating...")
        zw.open_bulk_writer()
        zw.connect()

    # Load a profile first so all features are accessible
    print("\nLoading profile: clearainews.com")
    zw.load_profile("clearainews.com")
    time.sleep(1)

    results = {"zimmwriter_version": title, "discovered": []}

    # Feature config windows
    features = {
        "wordpress":      "95",
        "link_pack":      "96",
        "serp_scraping":  "97",
        "deep_research":  "98",
        "style_mimic":    "99",
        "custom_outline": "100",
        "custom_prompt":  "101",
        "youtube_videos": "102",
        "webhook":        "103",
        "alt_images":     "104",
        "seo_csv":        "105",
    }

    window_titles = {
        "wordpress":      "Enable WordPress Uploads",
        "link_pack":      "Load Link Pack",
        "serp_scraping":  "Enable SERP Scraping",
        "deep_research":  "Deep Research",
        "style_mimic":    "Style Mimic",
        "custom_outline": "Set Custom Outline",
        "custom_prompt":  "Set Custom Prompts",
        "youtube_videos": "Enable YouTube Videos",
        "webhook":        "Enable Webhook",
        "alt_images":     "Enable Alt Images",
        "seo_csv":        "Set Bulk SEO CSV",
    }

    for key in features:
        try:
            result = discover_feature_window(zw, key, features[key], window_titles[key])
            if result:
                results["discovered"].append(result)
        except Exception as e:
            print(f"  FATAL ERROR on {key}: {e}")
            # Try to recover
            try:
                zw._dismiss_dialog(timeout=2)
                zw.connect()
            except Exception:
                pass
        time.sleep(1)

    # Image Options windows (O buttons)
    print("\n\n" + "="*60)
    print("IMAGE OPTIONS WINDOWS")
    print("="*60)

    for btn_id, label in [("80", "Featured O"), ("86", "Subheading O")]:
        try:
            result = discover_image_options_window(zw, btn_id, label)
            if result:
                results["discovered"].append(result)
        except Exception as e:
            print(f"  FATAL ERROR on {label}: {e}")
            try:
                zw._dismiss_dialog(timeout=2)
                zw.connect()
            except Exception:
                pass
        time.sleep(1)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "feature_window_controls.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to: {output_path}")
    print(f"Total windows discovered: {len(results['discovered'])}")


if __name__ == "__main__":
    main()
