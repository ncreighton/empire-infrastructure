"""
Test the P button (Image Prompt) flow step-by-step with screenshots.
Captures a screenshot after each action so we can see exactly what ZimmWriter shows.

Usage:
    python scripts/test_p_button.py
"""
import sys
import os
import time
import ctypes
from ctypes import wintypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
import pyautogui

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "p_button_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def screenshot(name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    pyautogui.screenshot(path)
    print(f"  [screenshot] {name}.png")
    return path

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

    # Load a profile first
    print("\n1. Loading profile: clearainews.com")
    zw.load_profile("clearainews.com")
    time.sleep(1)
    screenshot("01_profile_loaded")

    # Click the Featured Image P button (id=81)
    print("\n2. Clicking Featured Image P button (id=81)...")
    zw.bring_to_front()
    time.sleep(0.3)
    btn = zw._find_child(control_type="Button", auto_id="81")
    print(f"   Button text: '{btn.window_text()}'")
    btn.click_input()
    time.sleep(2)
    screenshot("02_after_p_click")

    # Find the Image Prompt window
    import re
    win = zw._wait_for_window("Image Prompt", timeout=8)
    if not win:
        print("   ERROR: Image Prompt window not found!")
        screenshot("02b_no_window")
        zw._dismiss_dialog(timeout=3)
        return

    print(f"   Window: '{win.window_text()}'")
    screenshot("03_prompt_window_open")

    # Read current contents of the text fields
    _SM = ctypes.windll.user32.SendMessageW
    _SM.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    _SM.restype = ctypes.c_long
    WM_GETTEXT = 0x000D
    WM_GETTEXTLENGTH = 0x000E
    WM_SETTEXT = 0x000C

    # Read prompt text (cid=114)
    prompt_edit = zw._find_child(win, control_type="Edit", auto_id="114")
    text_len = _SM(prompt_edit.handle, WM_GETTEXTLENGTH, 0, 0)
    text_buf = ctypes.create_unicode_buffer(text_len + 2)
    _SM(prompt_edit.handle, WM_GETTEXT, text_len + 1, ctypes.addressof(text_buf))
    print(f"\n3. Current prompt text ({text_len} chars):")
    print(f"   '{text_buf.value[:200]}...'") if text_len > 200 else print(f"   '{text_buf.value}'")

    # Read prompt name (cid=116)
    name_edit = zw._find_child(win, control_type="Edit", auto_id="116")
    name_len = _SM(name_edit.handle, WM_GETTEXTLENGTH, 0, 0)
    name_buf = ctypes.create_unicode_buffer(name_len + 2)
    _SM(name_edit.handle, WM_GETTEXT, name_len + 1, ctypes.addressof(name_buf))
    print(f"\n4. Current prompt name: '{name_buf.value}'")

    # Read loaded prompt dropdown (cid=118)
    try:
        loaded_combo = zw._find_child(win, control_type="ComboBox", auto_id="118")
        cur = _SM(loaded_combo.handle, 0x0147, 0, 0)  # CB_GETCURSEL
        if cur >= 0:
            item_len = _SM(loaded_combo.handle, 0x0149, cur, 0)  # CB_GETLBTEXTLEN
            item_buf = ctypes.create_unicode_buffer(item_len + 2)
            _SM(loaded_combo.handle, 0x0148, cur, ctypes.addressof(item_buf))  # CB_GETLBTEXT
            print(f"\n5. Loaded prompt dropdown selected: '{item_buf.value}'")
        else:
            print(f"\n5. Loaded prompt dropdown: no selection (index={cur})")

        # List all saved prompts
        count = _SM(loaded_combo.handle, 0x0146, 0, 0)  # CB_GETCOUNT
        print(f"   Total saved prompts: {count}")
        for i in range(min(count, 20)):
            item_len = _SM(loaded_combo.handle, 0x0149, i, 0)
            item_buf = ctypes.create_unicode_buffer(item_len + 2)
            _SM(loaded_combo.handle, 0x0148, i, ctypes.addressof(item_buf))
            print(f"   [{i}] {item_buf.value}")
    except Exception as e:
        print(f"\n5. Could not read dropdown: {e}")

    # List ALL controls in the window
    print("\n6. All controls in Image Prompt window:")
    try:
        children = win.children()
        for child in children:
            try:
                ctype = child.friendly_class_name()
                cid = child.control_id()
                text = child.window_text()[:100]
                print(f"   {ctype:15s} id={cid:4d}  '{text}'")
            except Exception:
                pass
    except Exception as e:
        print(f"   Error: {e}")

    # Now write the new prompt text via WM_SETTEXT
    test_prompt = (
        'Read and deeply analyze the article title "{title}". '
        "Identify the specific real-world subject. "
        "Then write a 40-60 word photojournalistic editorial shot prompt. "
        "Color palette: high-contrast black and white with vivid accent color pops. "
        "Do not include any text, words, watermarks, or logos in the image."
    )
    test_name = "clearainews.com_featured"

    print(f"\n7. Writing prompt text via WM_SETTEXT ({len(test_prompt)} chars)...")
    text_buf2 = ctypes.create_unicode_buffer(test_prompt)
    _SM(prompt_edit.handle, WM_SETTEXT, 0, ctypes.addressof(text_buf2))
    time.sleep(0.3)
    screenshot("04_after_text_set")

    # Verify it was written
    verify_len = _SM(prompt_edit.handle, WM_GETTEXTLENGTH, 0, 0)
    verify_buf = ctypes.create_unicode_buffer(verify_len + 2)
    _SM(prompt_edit.handle, WM_GETTEXT, verify_len + 1, ctypes.addressof(verify_buf))
    print(f"   Verified: {verify_len} chars written")
    if verify_buf.value[:50] != test_prompt[:50]:
        print(f"   MISMATCH! Got: '{verify_buf.value[:100]}'")
    else:
        print(f"   Text matches OK")

    print(f"\n8. Writing prompt name via WM_SETTEXT: '{test_name}'...")
    name_buf2 = ctypes.create_unicode_buffer(test_name)
    _SM(name_edit.handle, WM_SETTEXT, 0, ctypes.addressof(name_buf2))
    time.sleep(0.3)
    screenshot("05_after_name_set")

    # Verify prompt text is STILL correct (not overwritten)
    verify_len2 = _SM(prompt_edit.handle, WM_GETTEXTLENGTH, 0, 0)
    verify_buf2 = ctypes.create_unicode_buffer(verify_len2 + 2)
    _SM(prompt_edit.handle, WM_GETTEXT, verify_len2 + 1, ctypes.addressof(verify_buf2))
    print(f"   Prompt text still intact: {verify_len2} chars")
    if verify_buf2.value[:50] != test_prompt[:50]:
        print(f"   WARNING: PROMPT TEXT WAS OVERWRITTEN!")
        print(f"   Now contains: '{verify_buf2.value[:100]}'")
    else:
        print(f"   Text still matches OK")

    # Click Save New Prompt
    print(f"\n9. Clicking 'Save New Prompt' (cid=120)...")
    save_btn = zw._find_child(win, control_type="Button", auto_id="120")
    print(f"   Button text: '{save_btn.window_text()}'")
    save_btn.click_input()
    time.sleep(1.5)
    screenshot("06_after_save_click")

    # Check for any popup dialogs
    print(f"\n10. Checking for popup dialogs...")
    try:
        for w in zw.app.windows():
            t = w.window_text()
            print(f"   Window: '{t}'")
    except Exception as e:
        print(f"   Error listing windows: {e}")
    screenshot("07_final_state")

    # Dismiss any dialogs
    dismissed = zw._dismiss_dialog(timeout=3)
    print(f"   Dismissed dialog: {dismissed}")

    # Close the prompt window
    print(f"\n11. Closing prompt window...")
    zw._close_config_window(win)
    time.sleep(0.5)
    zw._dismiss_dialog(timeout=2)
    screenshot("08_closed")

    print(f"\nDone! Screenshots saved to: {OUTPUT_DIR}")
    print("Review the screenshots to verify the P button flow.")


if __name__ == "__main__":
    main()
