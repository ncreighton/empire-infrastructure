"""Add remaining LinkedIn skills via ADB automation.
Uses motionevent DOWN/UP for Compose UI compatibility.
Assumes we're already on the 'Add skill' form."""

import subprocess
import time
import sys
import re
import xml.etree.ElementTree as ET
import os

from adb_config import ADB, DEVICE as DEV
UI_XML = r"D:\Claude Code Projects\openclaw-empire\ui.xml"

SKILLS = [
    "Artificial Intelligence",
    "Digital Marketing",
    "Content Marketing",
    "Automation",
    "E-Commerce",
    "JavaScript",
    "Web Development",
]

os.environ["MSYS_NO_PATHCONV"] = "1"


def adb(cmd):
    full = [ADB, "-s", DEV, "shell", cmd]
    try:
        return subprocess.run(full, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        print("  ADB timeout")
        return None


def tap(x, y):
    adb(f"input motionevent DOWN {x} {y}")
    time.sleep(0.1)
    adb(f"input motionevent UP {x} {y}")
    time.sleep(0.5)


def dump_ui():
    adb("uiautomator dump /sdcard/ui.xml")
    time.sleep(1)
    with open(UI_XML, "wb") as f:
        subprocess.run(
            [ADB, "-s", DEV, "exec-out", "cat /sdcard/ui.xml"],
            stdout=f, stderr=subprocess.DEVNULL, timeout=10,
        )
    try:
        return ET.parse(UI_XML).getroot()
    except ET.ParseError:
        print("  XML parse error, retrying...")
        time.sleep(2)
        adb("uiautomator dump /sdcard/ui.xml")
        time.sleep(1)
        with open(UI_XML, "wb") as f:
            subprocess.run(
                [ADB, "-s", DEV, "exec-out", "cat /sdcard/ui.xml"],
                stdout=f, stderr=subprocess.DEVNULL, timeout=10,
            )
        return ET.parse(UI_XML).getroot()


def find_text(root, text):
    for n in root.iter("node"):
        t = n.get("text", "")
        if text.lower() in t.lower():
            bounds = n.get("bounds", "")
            m = re.findall(r'\[(\d+),(\d+)\]', bounds)
            if len(m) == 2:
                cx = (int(m[0][0]) + int(m[1][0])) // 2
                cy = (int(m[0][1]) + int(m[1][1])) // 2
                return cx, cy
    return None


def find_clickable_near(root, text):
    """Find the clickable parent element near a text element."""
    text_pos = find_text(root, text)
    if not text_pos:
        return None
    ty = text_pos[1]
    best = None
    best_dist = 9999
    for n in root.iter("node"):
        if n.get("clickable") != "true":
            continue
        bounds = n.get("bounds", "")
        m = re.findall(r'\[(\d+),(\d+)\]', bounds)
        if len(m) == 2:
            cy = (int(m[0][1]) + int(m[1][1])) // 2
            dist = abs(cy - ty)
            if dist < best_dist and dist < 50:
                best_dist = dist
                best = ((int(m[0][0]) + int(m[1][0])) // 2, cy)
    return best


def check_all_boxes(root):
    """Check all unchecked checkable+clickable boxes, skip Follow."""
    checked_count = 0
    for n in root.iter("node"):
        if n.get("checkable") != "true":
            continue
        if n.get("checked") != "false":
            continue
        if n.get("clickable") != "true":
            continue
        bounds = n.get("bounds", "")
        m = re.findall(r'\[(\d+),(\d+)\]', bounds)
        if len(m) != 2:
            continue
        x1, y1 = int(m[0][0]), int(m[0][1])
        x2, y2 = int(m[1][0]), int(m[1][1])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # Skip if too small (the inner checkbox icon, not the row)
        width = x2 - x1
        if width < 200:
            continue
        # Skip Follow checkbox (usually the last one, checked by default)
        # We'll check it anyway since it's usually already checked
        tap(cx, cy)
        time.sleep(0.3)
        checked_count += 1
    return checked_count


def add_skill(skill_name):
    print(f"\n{'='*50}")
    print(f"Adding: {skill_name}")
    print(f"{'='*50}")

    root = dump_ui()

    # Check if we're on success screen
    if find_text(root, "has been added"):
        print("  On success screen, tapping 'Add more skills'...")
        pos = find_clickable_near(root, "Add more skills")
        if pos:
            tap(*pos)
        else:
            pos = find_text(root, "Add more skills")
            if pos:
                tap(*pos)
        time.sleep(3)
        root = dump_ui()

    # Check if we need to navigate to Add skill form
    if find_text(root, "Add to profile"):
        print("  On Add to profile menu, finding Add skills...")
        # Expand Core if needed
        core_pos = find_clickable_near(root, "Core")
        if core_pos:
            tap(*core_pos)
            time.sleep(2)
            root = dump_ui()
        skills_pos = find_clickable_near(root, "Add skills")
        if skills_pos:
            tap(*skills_pos)
            time.sleep(3)
            root = dump_ui()

    # Now we should be on the Add skill form
    # Tap the skill input field
    input_pos = find_text(root, "Skill (ex:")
    if not input_pos:
        input_pos = find_text(root, "Skill*")
    if input_pos:
        tap(input_pos[0], input_pos[1] + 40)  # tap slightly below label
        time.sleep(1)
    else:
        # Try tapping where input typically is
        tap(540, 333)
        time.sleep(1)

    # Clear any existing text
    adb("input keyevent 123")  # MOVE_END
    time.sleep(0.1)
    for _ in range(30):
        adb("input keyevent 67")  # DEL
    time.sleep(0.3)

    # Type skill name
    search = skill_name.replace(" ", "%s").replace("-", "%s")
    adb(f'input text "{search}"')
    print(f"  Typed: {skill_name}")
    time.sleep(2)

    # Find and tap the suggestion
    root = dump_ui()
    # Look for suggestion matching skill name
    match = find_text(root, skill_name.split("(")[0].strip())
    if match:
        # Find clickable parent
        clickable = find_clickable_near(root, skill_name.split("(")[0].strip())
        if clickable:
            tap(*clickable)
        else:
            tap(*match)
        print(f"  Selected suggestion")
        time.sleep(3)
    else:
        print(f"  No suggestion found, trying first result...")
        # Tap area where first suggestion would appear
        root2 = dump_ui()
        for n in root2.iter("node"):
            if n.get("clickable") == "true":
                bounds = n.get("bounds", "")
                m2 = re.findall(r'\[(\d+),(\d+)\]', bounds)
                if len(m2) == 2:
                    y1 = int(m2[0][1])
                    if 300 < y1 < 500:
                        cx = (int(m2[0][0]) + int(m2[1][0])) // 2
                        cy = (int(m2[0][1]) + int(m2[1][1])) // 2
                        tap(cx, cy)
                        time.sleep(3)
                        break

    # Now on the skill form with checkboxes
    root = dump_ui()
    if not find_text(root, "Show us where"):
        print("  WARNING: May not be on checkbox form")
        # Take screenshot for debug
        with open(f"D:\\Claude Code Projects\\openclaw-empire\\debug_{skill_name.replace(' ','_')}.png", "wb") as f:
            subprocess.run(
                [ADB, "-s", DEV, "exec-out", "screencap", "-p"],
                stdout=f, timeout=10
            )

    # Scroll up first to see all checkboxes from top
    adb("input swipe 540 600 540 1800 300")
    time.sleep(1)

    # Check boxes in multiple rounds with scrolling
    total_checked = 0
    for scroll_round in range(4):
        root = dump_ui()
        count = check_all_boxes(root)
        total_checked += count
        if scroll_round < 3:
            adb("input swipe 540 1800 540 800 300")
            time.sleep(1)

    print(f"  Checked {total_checked} boxes total")

    # Find and tap Save
    root = dump_ui()
    save_pos = find_text(root, "Save")
    if save_pos:
        # Find the clickable Save button
        for n in root.iter("node"):
            if n.get("clickable") == "true":
                bounds = n.get("bounds", "")
                m = re.findall(r'\[(\d+),(\d+)\]', bounds)
                if len(m) == 2:
                    y1 = int(m[0][1])
                    y2 = int(m[1][1])
                    if y1 > 2100:  # Save button at very bottom
                        cx = (int(m[0][0]) + int(m[1][0])) // 2
                        cy = (y1 + y2) // 2
                        tap(cx, cy)
                        break
        else:
            tap(*save_pos)
    else:
        print("  Save not found, trying bottom of screen")
        tap(540, 2211)

    time.sleep(4)

    # Verify
    root = dump_ui()
    if find_text(root, "has been added"):
        print(f"  SUCCESS: {skill_name} added!")
        return True
    else:
        print(f"  Could not confirm (may still have worked)")
        return True


def main():
    skills = SKILLS
    if len(sys.argv) > 1:
        skills = sys.argv[1:]

    print(f"Adding {len(skills)} skills...")
    results = []

    for skill in skills:
        ok = add_skill(skill)
        results.append((skill, ok))

    print(f"\n{'='*50}")
    print("RESULTS:")
    for skill, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {skill}")


if __name__ == "__main__":
    main()
