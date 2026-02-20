"""
Discover control IDs in ZimmWriter's Image Options (O) and Image Prompt (P) sub-windows.

The 4 sub-windows opened by O/P buttons don't have known control maps:
  - O button (id=78): "Image Options" for featured image model
  - P button (id=79): "Set Featured Image Prompt" window
  - O button (id=84): "Image Options" for subheading image model
  - P button (id=85): "Set Subheading Image Prompt" window

O window controls vary by the currently selected image model (ideogram models
have extra dropdowns for magic_prompt, style, similarity).

Usage:
    python scripts/discover_image_windows.py
    python scripts/discover_image_windows.py --button 78     # Just featured O
    python scripts/discover_image_windows.py --button 79     # Just featured P
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
from src.utils import ensure_output_dir, timestamp


# Button auto_ids and expected window title patterns
IMAGE_BUTTONS = {
    "78": {"label": "Featured Image Options (O)", "title_re": "Image Options"},
    "79": {"label": "Featured Image Prompt (P)", "title_re": "Set Featured Image Prompt"},
    "84": {"label": "Subheading Image Options (O)", "title_re": "Image Options"},
    "85": {"label": "Subheading Image Prompt (P)", "title_re": "Set Subheading Image Prompt"},
}


def dump_window_controls(zw, win) -> dict:
    """Enumerate all controls in a window and return structured data."""
    controls = {
        "buttons": [],
        "checkboxes": [],
        "comboboxes": [],
        "edits": [],
        "labels": [],
        "other": [],
    }

    SendMsg = ctypes.windll.user32.SendMessageW
    SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMsg.restype = ctypes.c_long

    for child in win.children():
        try:
            ctrl_type = child.friendly_class_name()
            ctrl_id = str(child.control_id())
            text = child.window_text()
            visible = child.is_visible()

            info = {
                "control_id": ctrl_id,
                "class": ctrl_type,
                "text": text,
                "visible": visible,
            }

            if ctrl_type == "Button":
                # Check if it's actually a checkbox
                style = child.get_properties().get("style", 0)
                if isinstance(style, int) and (style & 0x0003) in (2, 3):
                    # BS_CHECKBOX or BS_AUTOCHECKBOX
                    info["type"] = "checkbox"
                    try:
                        info["checked"] = SendMsg(child.handle, 0x00F0, 0, 0) == 1
                    except Exception:
                        pass
                    controls["checkboxes"].append(info)
                else:
                    controls["buttons"].append(info)

            elif ctrl_type == "CheckBox":
                try:
                    info["checked"] = SendMsg(child.handle, 0x00F0, 0, 0) == 1
                except Exception:
                    pass
                controls["checkboxes"].append(info)

            elif ctrl_type == "ComboBox":
                # Read items
                hwnd = child.handle
                count = SendMsg(hwnd, 0x0146, 0, 0)  # CB_GETCOUNT
                items = []
                for i in range(min(count, 50)):  # Cap at 50
                    length = SendMsg(hwnd, 0x0149, i, 0)
                    if length >= 0:
                        buf = ctypes.create_unicode_buffer(length + 2)
                        SendMsg(hwnd, 0x0148, i, ctypes.addressof(buf))
                        items.append(buf.value)

                # Read current selection
                cur = SendMsg(hwnd, 0x0147, 0, 0)
                selected = ""
                if 0 <= cur < len(items):
                    selected = items[cur]

                info["items"] = items
                info["selected"] = selected
                info["item_count"] = count
                controls["comboboxes"].append(info)

            elif ctrl_type == "Edit":
                try:
                    info["value"] = child.window_text()[:500]
                except Exception:
                    info["value"] = ""
                controls["edits"].append(info)

            elif ctrl_type in ("Static", "Label"):
                controls["labels"].append(info)
            else:
                info["type"] = ctrl_type
                controls["other"].append(info)

        except Exception as e:
            controls["other"].append({"error": str(e)})

    return controls


def discover_button(zw, button_id: str, info: dict) -> dict:
    """Click a button, discover the opened window's controls, then close it."""
    result = {
        "button_id": button_id,
        "label": info["label"],
        "expected_title": info["title_re"],
        "window_found": False,
        "controls": {},
    }

    print(f"\n  Clicking button id={button_id} ({info['label']})...")

    try:
        # Click the button (must use click_input for 32/64-bit AutoIt compat)
        zw.bring_to_front()
        time.sleep(0.3)
        btn = zw.main_window.child_window(control_id=int(button_id))
        btn.click_input()
        time.sleep(2)

        # Find the opened window (title includes "ZimmWriter v10.xxx: ...")
        win = zw._wait_for_window(info["title_re"], timeout=8)
        if not win:
            # Try broader search — look for any new non-Bulk window
            for w in zw.app.windows():
                t = w.window_text()
                if t and "Bulk" not in t and "Menu" not in t and len(t) > 5:
                    win = w
                    break

        if win:
            result["window_found"] = True
            result["window_title"] = win.window_text()
            print(f"  Window found: '{win.window_text()}'")

            # Enumerate controls
            result["controls"] = dump_window_controls(zw, win)

            # Print summary
            for cat, items in result["controls"].items():
                if items:
                    print(f"    {cat}: {len(items)}")
                    for item in items:
                        ctrl_id = item.get("control_id", "?")
                        text = item.get("text", "")[:60]
                        print(f"      id={ctrl_id}: {text}")

            # Close the window
            WM_CLOSE = 0x0010
            SendMsg = ctypes.windll.user32.SendMessageW
            SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            SendMsg.restype = ctypes.c_long
            SendMsg(win.handle, WM_CLOSE, 0, 0)
            time.sleep(1)

            # Dismiss any dialogs
            zw._dismiss_dialog(timeout=2)
        else:
            print("  WARNING: No window found after click")
            # Try dismissing anything that opened
            zw._dismiss_dialog(timeout=3)

    except Exception as e:
        result["error"] = str(e)
        print(f"  ERROR: {e}")
        zw._dismiss_dialog(timeout=3)

    time.sleep(1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Discover Image O/P sub-window controls")
    parser.add_argument("--button", type=str, help="Single button ID to discover (78/79/84/85)")
    args = parser.parse_args()

    zw = ZimmWriterController()
    if not zw.connect():
        print("ERROR: Could not connect to ZimmWriter. Is it running?")
        sys.exit(1)

    print(f"Connected to: {zw.get_window_title()}")

    # Ensure on Bulk Writer screen
    title = zw.get_window_title()
    if "Bulk" not in title:
        print(f"On '{title}' — navigating to Bulk Writer...")
        zw.open_bulk_writer()
        time.sleep(2)
        title = zw.get_window_title()
        if "Bulk" not in title:
            print("ERROR: Could not navigate to Bulk Writer")
            sys.exit(1)

    # Determine which buttons to discover
    if args.button:
        if args.button not in IMAGE_BUTTONS:
            print(f"ERROR: Unknown button '{args.button}'. Use: {', '.join(IMAGE_BUTTONS.keys())}")
            sys.exit(1)
        buttons = {args.button: IMAGE_BUTTONS[args.button]}
    else:
        buttons = IMAGE_BUTTONS

    print(f"\nDiscovering {len(buttons)} image sub-window(s)...")
    print("=" * 65)

    results = {}
    for btn_id, info in buttons.items():
        result = discover_button(zw, btn_id, info)
        results[btn_id] = result
        time.sleep(1)

    # Save results
    out_dir = ensure_output_dir()
    out_path = str(out_dir / f"image_window_controls_{timestamp()}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 65}")
    print(f"Results saved to: {out_path}")

    # Summary
    print("\nSUMMARY:")
    for btn_id, result in results.items():
        found = "FOUND" if result.get("window_found") else "NOT FOUND"
        print(f"  Button {btn_id} ({result['label']}): {found}")
        if result.get("window_found"):
            ctrls = result.get("controls", {})
            total = sum(len(v) for v in ctrls.values())
            print(f"    Controls: {total} total")


if __name__ == "__main__":
    main()
