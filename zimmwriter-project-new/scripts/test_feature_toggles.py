"""
Live-test all 11 feature toggle buttons on the Bulk Writer screen.
Tests click_input() for each toggle:
  1. Read initial state (Enabled/Disabled from button text)
  2. Click to toggle
  3. Verify state changed
  4. Click again to restore original state
  5. Verify restored
"""
import subprocess
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pywinauto import Application, Desktop

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# Feature toggle buttons: name -> auto_id (Bulk Writer screen)
FEATURE_TOGGLES = {
    "WordPress":       "93",
    "Link Pack":       "94",
    "SERP Scraping":   "95",
    "Deep Research":   "96",
    "Style Mimic":     "97",
    "Custom Outline":  "98",
    "Custom Prompt":   "99",
    "YouTube Videos":  "100",
    "Webhook":         "101",
    "Alt Images":      "102",
    "SEO CSV":         "103",
}


def dismiss_any_dialog(max_wait=3):
    """Dismiss any ZimmWriter dialog that pops up."""
    deadline = time.time() + max_wait
    dismissed = []
    while time.time() < deadline:
        for w in Desktop(backend="uia").windows():
            t = w.window_text()
            if "ZimmWriter" in t and any(k in t for k in ["Error", "Info", "Warning", "Success"]):
                try:
                    w.child_window(title="OK", control_type="Button").click_input()
                except Exception:
                    try:
                        btns = w.descendants(control_type="Button")
                        if btns:
                            btns[0].click_input()
                    except Exception:
                        pass
                dismissed.append(t)
                time.sleep(0.3)
        time.sleep(0.3)
    return dismissed


def close_feature_panel(bulk_win, feature_name, max_wait=2):
    """Some features open a panel/dialog when enabled. Try to close it."""
    time.sleep(0.5)
    # Check for any new windows that appeared
    dismissed = dismiss_any_dialog(max_wait=1)
    if dismissed:
        return dismissed

    # Some features open a panel within the Bulk Writer window itself
    # We don't need to close those - they're part of the UI
    return []


def main():
    # Connect to ZimmWriter
    result = subprocess.run(
        ['powershell', '-Command',
         "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
        capture_output=True, text=True, timeout=10
    )
    pid = int(result.stdout.strip())
    app = Application(backend="uia").connect(process=pid)

    # Find all windows
    print("=== ZimmWriter Windows ===")
    bulk_win = None
    menu_win = None
    for w in app.windows():
        title = w.window_text()
        handle = w.handle
        print(f"  {title} (handle={handle})")
        if "Bulk" in title and ("Writer" in title or "Blog" in title):
            bulk_win = app.window(handle=handle)
        if title and "Menu" in title and "Option" not in title and "Setup" not in title:
            menu_win = app.window(handle=handle)

    # Navigate to Bulk Writer if not already there
    if not bulk_win:
        # First close Options Menu / WordPress Settings if blocking
        import ctypes
        from ctypes import wintypes
        WM_CLOSE = 0x0010
        SendMsg = ctypes.windll.user32.SendMessageW
        SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        SendMsg.restype = ctypes.c_long

        for w in app.windows():
            title = w.window_text()
            if "Option Menu" in title or "Setup WordPress" in title:
                print(f"  Closing blocking window: {title}")
                SendMsg(w.handle, WM_CLOSE, 0, 0)
                time.sleep(1)

        time.sleep(2)
        app = Application(backend="uia").connect(process=pid)

        # Now find Menu and click Bulk Writer
        for w in app.windows():
            title = w.window_text()
            if "Menu" in title and "Option" not in title:
                menu_win = app.window(handle=w.handle)
                break

        if menu_win:
            print("\nNavigating Menu -> Bulk Writer...")
            menu_win.set_focus()
            time.sleep(0.5)
            menu_win.child_window(auto_id="14", control_type="Button").click_input()
            time.sleep(5)
            app = Application(backend="uia").connect(process=pid)
            for w in app.windows():
                title = w.window_text()
                if "Bulk" in title and ("Writer" in title or "Blog" in title):
                    bulk_win = app.window(handle=w.handle)
                    break

    if not bulk_win:
        print("\nERROR: Could not find or navigate to Bulk Writer screen!")
        return

    print(f"\nBulk Writer window: {bulk_win.window_text()}")

    # Test each feature toggle
    print(f"\n{'=' * 70}")
    print(f"TESTING {len(FEATURE_TOGGLES)} FEATURE TOGGLE BUTTONS")
    print(f"{'=' * 70}\n")

    results = []
    for name, auto_id in FEATURE_TOGGLES.items():
        print(f"[{name}] (auto_id={auto_id})")

        try:
            btn = bulk_win.child_window(auto_id=auto_id, control_type="Button")
            initial_text = btn.window_text()
            initial_enabled = "Enabled" in initial_text
            print(f"  Initial: '{initial_text}' ({'ENABLED' if initial_enabled else 'DISABLED'})")

            # Click to toggle
            btn.click_input()
            time.sleep(1)

            # Dismiss any dialog that appeared
            dismissed = dismiss_any_dialog(max_wait=2)
            if dismissed:
                print(f"  Dismissed dialog: {dismissed}")

            # Read new state
            btn = bulk_win.child_window(auto_id=auto_id, control_type="Button")
            toggled_text = btn.window_text()
            toggled_enabled = "Enabled" in toggled_text
            print(f"  After toggle: '{toggled_text}' ({'ENABLED' if toggled_enabled else 'DISABLED'})")

            state_changed = initial_enabled != toggled_enabled
            if state_changed:
                print(f"  STATE CHANGED: OK")
            else:
                print(f"  WARNING: State did NOT change!")

            # Restore original state
            if state_changed:
                btn.click_input()
                time.sleep(1)
                dismissed = dismiss_any_dialog(max_wait=2)
                if dismissed:
                    print(f"  Dismissed dialog: {dismissed}")

                btn = bulk_win.child_window(auto_id=auto_id, control_type="Button")
                restored_text = btn.window_text()
                restored_enabled = "Enabled" in restored_text
                restored_ok = restored_enabled == initial_enabled
                print(f"  Restored: '{restored_text}' ({'ENABLED' if restored_enabled else 'DISABLED'}) {'OK' if restored_ok else 'MISMATCH!'}")
            else:
                restored_ok = False

            results.append({
                "name": name,
                "auto_id": auto_id,
                "initial": initial_text,
                "toggled": toggled_text,
                "state_changed": state_changed,
                "restored": restored_ok if state_changed else False,
                "status": "PASS" if state_changed else "FAIL",
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "name": name,
                "auto_id": auto_id,
                "status": "ERROR",
                "error": str(e),
            })

        print()

    # Summary
    print(f"{'=' * 70}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 70}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    for r in results:
        status = r["status"]
        name = r["name"]
        if status == "PASS":
            print(f"  PASS  {name} (id={r['auto_id']})")
        elif status == "FAIL":
            print(f"  FAIL  {name} (id={r['auto_id']}) - state did not change")
        else:
            print(f"  ERROR {name} (id={r['auto_id']}) - {r.get('error', 'unknown')}")

    print(f"\n  Total: {passed} PASS, {failed} FAIL, {errors} ERROR out of {len(results)}")
    print(f"{'=' * 70}")

    # Take final screenshot
    try:
        img = bulk_win.wrapper_object().capture_as_image()
        screenshot_path = os.path.join(OUTPUT_DIR, "feature_toggles_test.png")
        img.save(screenshot_path)
        print(f"\nScreenshot: {screenshot_path}")
    except Exception as e:
        print(f"\nScreenshot error: {e}")


if __name__ == "__main__":
    main()
