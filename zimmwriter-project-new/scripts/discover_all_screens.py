"""
Discover all controls on every ZimmWriter screen.

Navigates from Menu to each screen, captures the full control tree
(buttons, checkboxes, dropdowns, text fields), then returns to Menu.
Saves a comprehensive JSON map at output/all_screens_control_map.json.

Usage:
    python scripts/discover_all_screens.py
    python scripts/discover_all_screens.py --screen bulk_writer
    python scripts/discover_all_screens.py --screen seo_writer --list-dropdown-values
    python scripts/discover_all_screens.py --menu-only

REQUIREMENTS:
    ZimmWriter must be running (any screen — script navigates from Menu).
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
from src.screen_navigator import ScreenNavigator, Screen, MENU_BUTTONS


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


def enumerate_controls(win, list_dropdown_values: bool = False, depth: int = 0) -> list:
    """Recursively enumerate all child controls of a window."""
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
                "class_name": child.class_name(),
                "rect": {
                    "l": child.rectangle().left,
                    "t": child.rectangle().top,
                    "w": child.rectangle().width(),
                    "h": child.rectangle().height(),
                },
                "is_visible": child.is_visible(),
                "is_enabled": child.is_enabled(),
                "depth": depth,
            }

            # ComboBox details
            if child.friendly_class_name() in ("ComboBox",):
                try:
                    info["selected"] = read_combo_selected(child.handle)
                    if list_dropdown_values:
                        info["items"] = read_combo_items(child.handle)
                        info["item_count"] = len(info["items"])
                except Exception:
                    pass

            # CheckBox state
            if child.friendly_class_name() in ("CheckBox",):
                try:
                    info["checked"] = child.get_check_state() == 1
                except Exception:
                    pass

            # Edit field content
            if child.friendly_class_name() in ("Edit",):
                try:
                    text = child.window_text()
                    info["text_value"] = text[:500] if text else ""
                except Exception:
                    pass

            # Button state (for toggle buttons)
            if child.friendly_class_name() in ("Button",):
                try:
                    info["button_text"] = child.window_text()[:200]
                except Exception:
                    pass

            controls.append(info)
        except Exception as e:
            controls.append({"error": str(e), "depth": depth})

    return controls


def discover_screen(zw, nav, screen: Screen,
                    list_dropdown_values: bool = False) -> dict:
    """Navigate to a screen, enumerate its controls, then return to Menu."""
    result = {
        "screen": screen.value,
        "status": "unknown",
        "window_title": "",
        "controls": [],
        "summary": {},
    }

    try:
        # Navigate to the screen
        if screen == Screen.MENU:
            if not nav.is_on_menu():
                nav.back_to_menu()
        else:
            if not nav.navigate_to(screen):
                result["status"] = "navigation_failed"
                return result

        time.sleep(1)

        # Capture window title
        result["window_title"] = zw.get_window_title()

        # Enumerate controls
        controls = enumerate_controls(
            zw.main_window, list_dropdown_values=list_dropdown_values
        )
        result["controls"] = controls
        result["control_count"] = len(controls)
        result["status"] = "OK"

        # Build summary counts by control type
        type_counts = {}
        for ctrl in controls:
            ct = ctrl.get("control_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1
        result["summary"] = type_counts

        # Return to Menu for next screen (unless we're discovering Menu itself)
        if screen != Screen.MENU:
            time.sleep(0.5)
            nav.back_to_menu()
            time.sleep(1)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        try:
            nav.back_to_menu()
        except Exception:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Discover controls on all ZimmWriter screens"
    )
    parser.add_argument(
        "--screen", type=str,
        help="Discover a single screen (e.g., bulk_writer, seo_writer, menu)"
    )
    parser.add_argument(
        "--menu-only", action="store_true",
        help="Only discover the Menu screen"
    )
    parser.add_argument(
        "--list-dropdown-values", action="store_true",
        help="Enumerate all dropdown values (slower)"
    )
    parser.add_argument(
        "--skip", type=str, nargs="*", default=[],
        help="Screen names to skip (e.g., secret_training free_gpts)"
    )
    args = parser.parse_args()

    # Connect
    zw = ZimmWriterController()
    if not zw.connect():
        print("ERROR: Could not connect to ZimmWriter. Is it running?")
        sys.exit(1)

    nav = ScreenNavigator(zw)
    print(f"Connected to: {zw.get_window_title()}")
    print(f"Current screen: {nav.detect_screen().value}")

    # Determine which screens to discover
    if args.menu_only:
        screens_to_discover = [Screen.MENU]
    elif args.screen:
        try:
            target = Screen(args.screen)
        except ValueError:
            print(f"ERROR: Unknown screen '{args.screen}'")
            print(f"Available: {', '.join(s.value for s in Screen if s != Screen.UNKNOWN)}")
            sys.exit(1)
        screens_to_discover = [target]
    else:
        # All navigable screens + Menu
        screens_to_discover = [Screen.MENU] + list(MENU_BUTTONS.keys())

    skip_set = set(args.skip)

    # First, get to Menu
    if not nav.is_on_menu():
        print("Navigating to Menu first...")
        if not nav.back_to_menu():
            print("ERROR: Could not reach Menu screen")
            sys.exit(1)

    # Discover each screen
    all_results = {}
    for i, screen in enumerate(screens_to_discover, 1):
        if screen.value in skip_set:
            print(f"[{i}/{len(screens_to_discover)}] {screen.value}... SKIPPED")
            continue

        print(f"[{i}/{len(screens_to_discover)}] {screen.value}...", end=" ", flush=True)
        result = discover_screen(zw, nav, screen, args.list_dropdown_values)
        all_results[screen.value] = result

        icon = "OK" if result["status"] == "OK" else "XX"
        ctrl_count = result.get("control_count", 0)
        summary = result.get("summary", {})
        summary_str = ", ".join(f"{k}={v}" for k, v in sorted(summary.items()))
        print(f"[{icon}] {ctrl_count} controls ({summary_str})")

        if result["status"] == "OK":
            # Print notable controls
            for ctrl in result["controls"]:
                ct = ctrl.get("control_type", "?")
                cid = ctrl.get("auto_id", "?")
                name = ctrl.get("name", "")[:60]
                if ct in ("Button", "ComboBox", "CheckBox", "Edit"):
                    extra = ""
                    if "selected" in ctrl:
                        extra = f" (selected: {ctrl['selected'][:40]})"
                    if "checked" in ctrl:
                        extra = f" (checked: {ctrl['checked']})"
                    if "item_count" in ctrl:
                        extra += f" [{ctrl['item_count']} items]"
                    print(f"    {ct:15s} id={cid:4s}  {name}{extra}")

        time.sleep(0.5)

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    if args.screen or args.menu_only:
        output_name = f"{(args.screen or 'menu')}_control_map.json"
    else:
        output_name = "all_screens_control_map.json"

    output_path = os.path.join(output_dir, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Summary
    ok = sum(1 for r in all_results.values() if r["status"] == "OK")
    total = len(all_results)
    print(f"\nDiscovered: {ok}/{total} screens")


if __name__ == "__main__":
    main()
