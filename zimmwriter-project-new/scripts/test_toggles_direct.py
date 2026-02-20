"""Direct test of feature toggle buttons on the already-open Bulk Blog Writer."""
import subprocess
import time
import ctypes
from ctypes import wintypes
from pywinauto import Application, Desktop

OUTPUT_DIR = r"D:\Claude Code Projects\zimmwriter-project-new\output"

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


def dismiss_dialogs():
    """Quick dismiss of any ZimmWriter popups."""
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
            time.sleep(0.3)


def main():
    result = subprocess.run(
        ['powershell', '-Command',
         "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
        capture_output=True, text=True, timeout=10
    )
    pid = int(result.stdout.strip())
    print(f"PID: {pid}")
    app = Application(backend="uia").connect(process=pid)

    # Find Bulk Blog Writer window directly
    bulk_win = None
    for w in app.windows():
        title = w.window_text()
        handle = w.handle
        print(f"Window: {title} (handle={handle})")
        if "Bulk" in title and "Blog" in title:
            bulk_win = app.window(handle=handle)

    if not bulk_win:
        print("ERROR: Bulk Blog Writer not found!")
        return

    print(f"\nUsing: {bulk_win.window_text()}")
    bulk_win.set_focus()
    time.sleep(0.5)

    # Test each toggle
    print(f"\n{'='*60}")
    results = []

    for name, auto_id in FEATURE_TOGGLES.items():
        print(f"\n[{name}] auto_id={auto_id}")
        try:
            btn = bulk_win.child_window(auto_id=auto_id, control_type="Button")
            initial = btn.window_text()
            was_disabled = "Disabled" in initial
            print(f"  Before: {initial}")

            # Toggle ON
            btn.click_input()
            time.sleep(1.5)
            dismiss_dialogs()

            btn2 = bulk_win.child_window(auto_id=auto_id, control_type="Button")
            after = btn2.window_text()
            print(f"  After:  {after}")

            changed = ("Disabled" in initial) != ("Disabled" in after)

            # Restore: toggle back OFF
            if changed:
                btn2.click_input()
                time.sleep(1)
                dismiss_dialogs()
                btn3 = bulk_win.child_window(auto_id=auto_id, control_type="Button")
                restored = btn3.window_text()
                print(f"  Restored: {restored}")
                restore_ok = ("Disabled" in restored) == was_disabled
            else:
                restore_ok = False

            status = "PASS" if changed else "FAIL"
            results.append((name, auto_id, status, initial, after))
            print(f"  Result: {status}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, auto_id, "ERROR", "", str(e)))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = failed = errors = 0
    for name, aid, status, before, after in results:
        icon = {"PASS": "OK", "FAIL": "!!", "ERROR": "XX"}[status]
        print(f"  [{icon}] {name:20s} (id={aid}) {status}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        else:
            errors += 1

    print(f"\n  {passed} PASS / {failed} FAIL / {errors} ERROR = {len(results)} total")
    print(f"{'='*60}")

    # Screenshot
    try:
        bulk_win.set_focus()
        time.sleep(0.3)
        img = bulk_win.wrapper_object().capture_as_image()
        img.save(f"{OUTPUT_DIR}\\feature_toggles_test.png")
        print(f"\nScreenshot: {OUTPUT_DIR}\\feature_toggles_test.png")
    except Exception as e:
        print(f"Screenshot error: {e}")


if __name__ == "__main__":
    main()
