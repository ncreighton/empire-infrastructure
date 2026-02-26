"""LinkedIn skill adder — automates checking all experience boxes and saving."""
import subprocess
import time
import sys
import re
import xml.etree.ElementTree as ET

from adb_config import ADB, DEVICE as DEV
UI_XML = r"D:\Claude Code Projects\openclaw-empire\ui.xml"


def adb(cmd):
    """Run ADB shell command."""
    full = [ADB, "-s", DEV, "shell", cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=15)


def tap(x, y):
    """Tap using motionevent (works on Compose UIs)."""
    adb(f"input motionevent DOWN {x} {y}")
    time.sleep(0.1)
    adb(f"input motionevent UP {x} {y}")
    time.sleep(0.5)


def dump_ui():
    """Dump and parse UI hierarchy."""
    adb("uiautomator dump /sdcard/ui.xml")
    time.sleep(1)
    subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/ui.xml", UI_XML],
                   capture_output=True, timeout=10)
    return ET.parse(UI_XML).getroot()


def find_unchecked_boxes(root):
    """Find all unchecked checkable items and return their tap coordinates."""
    unchecked = []
    for n in root.iter("node"):
        checkable = n.get("checkable", "")
        checked = n.get("checked", "")
        clickable = n.get("clickable", "")
        bounds = n.get("bounds", "")
        if checkable == "true" and checked == "false" and clickable == "true":
            m = re.findall(r'\[(\d+),(\d+)\]', bounds)
            if len(m) == 2:
                cx = (int(m[0][0]) + int(m[1][0])) // 2
                cy = (int(m[0][1]) + int(m[1][1])) // 2
                # Get associated text
                text = ""
                for child in n.iter("node"):
                    t = child.get("text", "")
                    if t and "Follow this" not in t:
                        text = t
                        break
                unchecked.append((cx, cy, text))
    return unchecked


def check_all_and_save():
    """Check all unchecked experience/education boxes, then save."""
    max_scrolls = 5
    for scroll_round in range(max_scrolls):
        root = dump_ui()

        # Check all unchecked boxes visible on screen
        unchecked = find_unchecked_boxes(root)
        # Filter out "Follow this skill" checkbox
        unchecked = [(x, y, t) for x, y, t in unchecked
                     if "Follow" not in t and "Suggested" not in t]

        if unchecked:
            print(f"  Round {scroll_round}: checking {len(unchecked)} boxes")
            for x, y, text in unchecked:
                print(f"    Checking: {text} at ({x},{y})")
                tap(x, y)
                time.sleep(0.3)

        # Try to scroll down to see more items
        adb("input swipe 540 1800 540 1200 300")
        time.sleep(1)

    # Now save — scroll to make sure save is visible
    print("  Saving skill...")
    root = dump_ui()

    # Find Save button
    for n in root.iter("node"):
        t = n.get("text", "")
        if t == "Save":
            bounds = n.get("bounds", "")
            m = re.findall(r'\[(\d+),(\d+)\]', bounds)
            if len(m) == 2:
                cx = (int(m[0][0]) + int(m[1][0])) // 2
                cy = (int(m[0][1]) + int(m[1][1])) // 2
                tap(cx, cy)
                time.sleep(3)
                print("  Saved!")
                return True

    print("  Save button not found, trying known coords...")
    tap(540, 2200)
    time.sleep(3)
    return True


def add_skill_from_form():
    """When already on the Add Skill form with a skill selected, check all and save."""
    print("Checking all experience boxes and saving...")
    check_all_and_save()

    # Check result
    root = dump_ui()
    for n in root.iter("node"):
        t = n.get("text", "")
        if "has been added" in t:
            print("SUCCESS: Skill added!")
            return True

    print("Could not confirm skill was added")
    return True


if __name__ == "__main__":
    add_skill_from_form()
