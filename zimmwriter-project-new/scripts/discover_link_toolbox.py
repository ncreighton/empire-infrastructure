"""
Discover all controls in ZimmWriter's Link Toolbox screen.

Navigates to Link Toolbox via Menu, enumerates all controls,
and saves the map to output/link_toolbox_control_map.json.

Usage:
    python scripts/discover_link_toolbox.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.screen_navigator import ScreenNavigator, Screen


def main():
    zw = ZimmWriterController()
    zw.connect()
    print(f"Connected to: {zw.get_window_title()}")

    nav = ScreenNavigator(zw)

    # Navigate to Link Toolbox
    print("Navigating to Link Toolbox...")
    if not nav.navigate_to(Screen.LINK_TOOLBOX):
        print("ERROR: Could not navigate to Link Toolbox")
        return

    time.sleep(2)
    print(f"Current window: {zw.get_window_title()}")

    # Enumerate all controls
    print("\nDiscovering controls...")
    controls = zw.dump_controls()

    # Save to file
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "link_toolbox_control_map.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(controls, f, indent=2, default=str)

    print(f"\nSaved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("LINK TOOLBOX CONTROLS")
    print("=" * 70)

    if isinstance(controls, list):
        by_type = {}
        for c in controls:
            ctrl_type = c.get("control_type", "unknown")
            by_type.setdefault(ctrl_type, []).append(c)

        for ctrl_type, items in sorted(by_type.items()):
            print(f"\n--- {ctrl_type} ({len(items)}) ---")
            for item in items:
                auto_id = item.get("auto_id", "?")
                name = item.get("name", item.get("text", ""))
                print(f"  [{auto_id:>5s}] {name}")
    elif isinstance(controls, dict):
        for key, value in controls.items():
            print(f"  {key}: {value}")

    # Take screenshot
    try:
        screenshot_path = zw.take_screenshot()
        print(f"\nScreenshot: {screenshot_path}")
    except Exception:
        pass

    # Navigate back to Bulk Writer
    print("\nNavigating back to Bulk Writer...")
    nav.navigate_to(Screen.BULK_WRITER)


if __name__ == "__main__":
    main()
