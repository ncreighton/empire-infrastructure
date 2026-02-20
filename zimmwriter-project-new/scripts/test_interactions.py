"""
Test ZimmWriter controller interactions.
Uses UIA for checkboxes/buttons, keyboard for dropdowns.
"""
import subprocess
import time
import ctypes
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pywinauto import Application
from pywinauto.keyboard import send_keys


def get_pid():
    result = subprocess.run(
        ['powershell', '-Command',
         "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def set_clipboard(text):
    CF_UNICODETEXT = 13
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    user32.OpenClipboard(0)
    user32.EmptyClipboard()
    hMem = kernel32.GlobalAlloc(0x0042, (len(text) + 1) * 2)
    pMem = kernel32.GlobalLock(hMem)
    ctypes.cdll.msvcrt.wcscpy(ctypes.c_wchar_p(pMem), text)
    kernel32.GlobalUnlock(hMem)
    user32.SetClipboardData(CF_UNICODETEXT, hMem)
    user32.CloseClipboard()


def main():
    pid = get_pid()
    print(f"PID: {pid}")

    app = Application(backend="uia").connect(process=pid)

    # Ensure Bulk Writer is open
    try:
        bulk_win = app.window(title_re=".*Bulk Blog Writer.*")
        bulk_win.window_text()
        print(f"Bulk Writer open: {bulk_win.window_text()}")
    except Exception:
        menu_win = app.window(title_re=".*Menu.*")
        menu_win.child_window(title="Bulk Writer", control_type="Button").invoke()
        print("Opening Bulk Writer...")
        time.sleep(4)
        bulk_win = app.window(title_re=".*Bulk Blog Writer.*")
        print(f"Opened: {bulk_win.window_text()}")

    passed = 0
    failed = 0

    # ── TEST 1: Checkbox toggle by auto_id ──
    print("\n=== TEST 1: Toggle 'Enable Lists (?)' by auto_id ===")
    try:
        cb = bulk_win.child_window(auto_id="48", control_type="CheckBox")
        before = cb.get_toggle_state()
        print(f"  Before: {before}")
        cb.toggle()
        time.sleep(0.3)
        after = cb.get_toggle_state()
        print(f"  After: {after}")
        assert after != before, "State unchanged!"
        cb.toggle()
        time.sleep(0.3)
        print(f"  Reset: {cb.get_toggle_state()}")
        print("  PASS")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── TEST 2: Checkbox by name ──
    print("\n=== TEST 2: Toggle 'Nuke AI Words' by name ===")
    try:
        cb = bulk_win.child_window(title="Nuke AI Words", control_type="CheckBox")
        before = cb.get_toggle_state()
        print(f"  Before: {before}")
        cb.toggle()
        time.sleep(0.3)
        print(f"  After: {cb.get_toggle_state()}")
        cb.toggle()
        time.sleep(0.3)
        print(f"  Reset: {cb.get_toggle_state()}")
        print("  PASS")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── TEST 3: Dropdown via keyboard ──
    print("\n=== TEST 3: Set 'Section Length' dropdown via keyboard ===")
    try:
        combo = bulk_win.child_window(auto_id="46", control_type="ComboBox")
        print(f"  Before: {combo.selected_text()}")
        # Focus and use Alt+Down to open, then type first letter
        combo.set_focus()
        time.sleep(0.2)
        # Type 'L' to select "Long"
        send_keys("L")
        time.sleep(0.3)
        print(f"  After 'L': {combo.selected_text()}")
        # Type 'M' to go back to Medium
        send_keys("M")
        time.sleep(0.3)
        print(f"  After 'M': {combo.selected_text()}")
        print("  PASS")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── TEST 4: Paste titles ──
    print("\n=== TEST 4: Paste titles into text area ===")
    try:
        titles = "Test Title 1: Smart Home Setup\nTest Title 2: AI Writing Guide\nTest Title 3: ZimmWriter Automation"
        title_field = bulk_win.child_window(auto_id="36", control_type="Edit")
        title_field.set_focus()
        time.sleep(0.2)
        set_clipboard(titles)
        send_keys("^a")
        time.sleep(0.1)
        send_keys("^v")
        time.sleep(0.5)
        print(f"  Pasted 3 titles ({len(titles)} chars)")
        # Clear
        send_keys("^a")
        time.sleep(0.1)
        send_keys("{DELETE}")
        time.sleep(0.3)
        print("  Cleared")
        print("  PASS")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── TEST 5: Feature toggle button ──
    print("\n=== TEST 5: Toggle 'WordPress' feature button ===")
    try:
        wp_btn = bulk_win.child_window(title_re=".*WordPress.*", control_type="Button")
        before = wp_btn.window_text()
        print(f"  Before: '{before}'")
        wp_btn.invoke()
        time.sleep(0.5)
        after = wp_btn.window_text()
        print(f"  After: '{after}'")
        # Toggle back
        wp_btn = bulk_win.child_window(title_re=".*WordPress.*", control_type="Button")
        wp_btn.invoke()
        time.sleep(0.5)
        reset = wp_btn.window_text()
        print(f"  Reset: '{reset}'")
        print("  PASS")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── TEST 6: Set AI Model dropdown ──
    print("\n=== TEST 6: Set 'AI Model for Writing' dropdown ===")
    try:
        model_combo = bulk_win.child_window(auto_id="67", control_type="ComboBox")
        print(f"  Before: {model_combo.selected_text()}")
        # Focus and type 'C' to jump to Claude models
        model_combo.set_focus()
        time.sleep(0.2)
        send_keys("C")
        time.sleep(0.3)
        after = model_combo.selected_text()
        print(f"  After 'C': {after}")
        # Go back to GPT-4o Mini
        send_keys("G")
        time.sleep(0.3)
        reset = model_combo.selected_text()
        print(f"  After 'G': {reset}")
        print("  PASS")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
