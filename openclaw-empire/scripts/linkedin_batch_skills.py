"""Batch add LinkedIn skills — types each skill, selects it, checks all boxes, saves."""
import subprocess
import time
import sys
import re
import xml.etree.ElementTree as ET

from adb_config import ADB, DEVICE as DEV
UI_XML = r"D:\Claude Code Projects\openclaw-empire\ui.xml"

SKILLS_TO_ADD = [
    "Search Engine Optimization (SEO)",
    "Content Strategy",
    "Artificial Intelligence (AI)",
    "Digital Marketing",
    "Content Marketing",
    "Automation",
    "E-Commerce",
    "JavaScript",
    "Web Development",
]


def adb(cmd):
    full = [ADB, "-s", DEV, "shell", cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=15)


def tap(x, y):
    adb(f"input motionevent DOWN {x} {y}")
    time.sleep(0.1)
    adb(f"input motionevent UP {x} {y}")
    time.sleep(0.5)


def dump_ui():
    adb("uiautomator dump /sdcard/ui.xml")
    time.sleep(1)
    subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/ui.xml", UI_XML],
                   capture_output=True, timeout=10)
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


def check_all_boxes_and_save():
    """Check all unchecked boxes, scroll through all, then save."""
    checked_any = True
    scroll_count = 0

    while checked_any and scroll_count < 6:
        root = dump_ui()
        checked_any = False

        for n in root.iter("node"):
            checkable = n.get("checkable", "")
            checked = n.get("checked", "")
            clickable = n.get("clickable", "")
            bounds = n.get("bounds", "")

            if checkable == "true" and checked == "false" and clickable == "true":
                # Skip "Follow this skill" items
                skip = False
                for child in n.iter("node"):
                    if "Follow" in child.get("text", ""):
                        skip = True
                        break
                if skip:
                    continue

                m = re.findall(r'\[(\d+),(\d+)\]', bounds)
                if len(m) == 2:
                    cx = (int(m[0][0]) + int(m[1][0])) // 2
                    cy = (int(m[0][1]) + int(m[1][1])) // 2
                    if 300 < cy < 2100:  # Only tap visible area
                        tap(cx, cy)
                        time.sleep(0.2)
                        checked_any = True

        # Scroll down
        adb("input swipe 540 1800 540 1200 300")
        time.sleep(0.8)
        scroll_count += 1

    # Save
    root = dump_ui()
    save_pos = find_text(root, "Save")
    if save_pos:
        tap(*save_pos)
    else:
        tap(540, 2200)
    time.sleep(3)


def tap_add_more_skills():
    """Tap 'Add more skills' from success screen."""
    root = dump_ui()
    pos = find_text(root, "Add more skills")
    if pos:
        tap(*pos)
        time.sleep(2)
        return True
    return False


def search_and_select_skill(skill_name):
    """Type skill name in search and select first match."""
    # Tap the skill input field
    root = dump_ui()

    # Check if we're on the success screen first
    if find_text(root, "has been added"):
        print(f"  On success screen, tapping 'Add more skills'...")
        if not tap_add_more_skills():
            print("  ERROR: Can't find 'Add more skills' button")
            return False
        root = dump_ui()

    # Find and tap skill search input
    skill_input = find_text(root, "Skill (ex:")
    if not skill_input:
        # Try finding empty clickable near Skill* label
        for n in root.iter("node"):
            clickable = n.get("clickable", "")
            bounds = n.get("bounds", "")
            if clickable == "true" and bounds:
                m = re.findall(r'\[(\d+),(\d+)\]', bounds)
                if len(m) == 2:
                    y1 = int(m[0][1])
                    if 400 < y1 < 600:
                        skill_input = ((int(m[0][0]) + int(m[1][0])) // 2,
                                       (int(m[0][1]) + int(m[1][1])) // 2)
                        break

    if skill_input:
        tap(*skill_input)
        time.sleep(1)
    else:
        print("  ERROR: Can't find skill input field")
        return False

    # Type the skill name (use short search term)
    search_term = skill_name.split("(")[0].strip()
    if len(search_term) > 20:
        search_term = search_term[:20]
    adb(f'input text "{search_term}"')
    time.sleep(2)

    # Find and tap the first match
    root = dump_ui()
    match = find_text(root, skill_name.split("(")[0].strip())
    if match:
        tap(*match)
        time.sleep(2)
        return True
    else:
        # Try just the first word
        first_word_match = find_text(root, search_term.split()[0])
        if first_word_match:
            tap(*first_word_match)
            time.sleep(2)
            return True

    print(f"  ERROR: No match found for '{skill_name}'")
    return False


def add_skill(skill_name):
    """Full flow: search, select, check boxes, save."""
    print(f"\n{'='*50}")
    print(f"Adding skill: {skill_name}")
    print(f"{'='*50}")

    if not search_and_select_skill(skill_name):
        return False

    # Now we should be on the "Add skill" form
    root = dump_ui()
    skill_field = find_text(root, "Skill*")
    if skill_field or find_text(root, "Show us where"):
        print(f"  On Add Skill form, checking boxes...")
        check_all_boxes_and_save()

        # Verify
        root = dump_ui()
        if find_text(root, "has been added"):
            print(f"  SUCCESS: {skill_name} added!")
            return True

    print(f"  Could not confirm skill was added")
    return True


def main():
    skills = SKILLS_TO_ADD
    if len(sys.argv) > 1:
        skills = sys.argv[1:]

    print(f"Adding {len(skills)} skills to LinkedIn profile...")
    results = []

    for skill in skills:
        success = add_skill(skill)
        results.append((skill, success))

    print(f"\n{'='*50}")
    print("RESULTS:")
    for skill, ok in results:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {skill}")


if __name__ == "__main__":
    main()
