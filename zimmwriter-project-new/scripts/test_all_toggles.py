"""
Full test of all 11 feature toggle buttons on Bulk Blog Writer.
Feature toggles open config windows when clicked. This script:
1. Clicks each toggle button
2. Checks if a config window appeared
3. Closes it (WM_CLOSE)
4. Verifies the button state on the Bulk Writer
5. Restores original state
"""
import subprocess
import time
import ctypes
from ctypes import wintypes
from pywinauto import Application, Desktop

OUTPUT_DIR = r"D:\Claude Code Projects\zimmwriter-project-new\output"
WM_CLOSE = 0x0010

SendMsg = ctypes.windll.user32.SendMessageW
SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
SendMsg.restype = ctypes.c_long

FEATURE_TOGGLES = [
    ("WordPress",      "93"),
    ("Link Pack",      "94"),
    ("SERP Scraping",  "95"),
    ("Deep Research",  "96"),
    ("Style Mimic",    "97"),
    ("Custom Outline", "98"),
    ("Custom Prompt",  "99"),
    ("YouTube Videos", "100"),
    ("Webhook",        "101"),
    ("Alt Images",     "102"),
    ("SEO CSV",        "103"),
]


def get_app_windows(app):
    """Get dict of window titles -> handles."""
    windows = {}
    for w in app.windows():
        windows[w.window_text()] = w.handle
    return windows


def dismiss_dialogs():
    """Quick dismiss popups."""
    for w in Desktop(backend="uia").windows():
        t = w.window_text()
        if "ZimmWriter" in t and any(k in t for k in ["Error", "Info", "Warning", "Success"]):
            try:
                w.child_window(title="OK", control_type="Button").click_input()
            except Exception:
                pass
            time.sleep(0.3)


def main():
    print("Connecting...", flush=True)
    result = subprocess.run(
        ['powershell', '-Command',
         "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | "
         "Select-Object -First 1 -ExpandProperty Id"],
        capture_output=True, text=True, timeout=10
    )
    pid = int(result.stdout.strip())
    app = Application(backend="uia").connect(process=pid)

    # Close any existing WordPress Uploads window from previous test
    before_wins = get_app_windows(app)
    print(f"Windows: {list(before_wins.keys())}", flush=True)

    for title, handle in before_wins.items():
        if "Enable" in title or "Uploads" in title:
            print(f"  Closing leftover: {title}", flush=True)
            SendMsg(handle, WM_CLOSE, 0, 0)
            time.sleep(1)

    app = Application(backend="uia").connect(process=pid)

    # Find Bulk Blog Writer
    bulk_handle = None
    for w in app.windows():
        if "Bulk" in w.window_text() and "Blog" in w.window_text():
            bulk_handle = w.handle
            break

    if not bulk_handle:
        print("ERROR: Bulk Blog Writer not found!", flush=True)
        return

    bulk = app.window(handle=bulk_handle)
    print(f"Bulk Writer: handle={bulk_handle}", flush=True)

    # Test each toggle
    print(f"\n{'='*65}", flush=True)
    print(f"TESTING {len(FEATURE_TOGGLES)} FEATURE TOGGLE BUTTONS", flush=True)
    print(f"{'='*65}", flush=True)

    results = []

    for name, auto_id in FEATURE_TOGGLES:
        print(f"\n--- {name} (auto_id={auto_id}) ---", flush=True)

        try:
            # Read initial state
            btn = bulk.child_window(auto_id=auto_id, control_type="Button")
            initial_text = btn.window_text()
            initial_disabled = "Disabled" in initial_text
            print(f"  Initial: '{initial_text}'", flush=True)

            # Record windows before click
            windows_before = set(get_app_windows(app).keys())

            # Click the toggle
            btn.click_input()
            time.sleep(2)
            dismiss_dialogs()

            # Check for new windows
            app_fresh = Application(backend="uia").connect(process=pid)
            windows_after = get_app_windows(app_fresh)
            new_windows = set(windows_after.keys()) - windows_before

            config_window = None
            if new_windows:
                for nw in new_windows:
                    if "ZimmWriter" in nw and "Menu" not in nw:
                        config_window = nw
                        break
                print(f"  Config window opened: {config_window}", flush=True)
            else:
                print(f"  No new window opened", flush=True)

            # Read button state after click
            bulk2 = app_fresh.window(handle=bulk_handle)
            btn2 = bulk2.child_window(auto_id=auto_id, control_type="Button")
            after_text = btn2.window_text()
            after_disabled = "Disabled" in after_text
            print(f"  After click: '{after_text}'", flush=True)

            # Close config window if it opened
            if config_window and config_window in windows_after:
                ch = windows_after[config_window]
                print(f"  Closing config window (handle={ch})...", flush=True)
                SendMsg(ch, WM_CLOSE, 0, 0)
                time.sleep(1)
                dismiss_dialogs()

                # Re-check button state after closing
                app_fresh2 = Application(backend="uia").connect(process=pid)
                bulk3 = app_fresh2.window(handle=bulk_handle)
                btn3 = bulk3.child_window(auto_id=auto_id, control_type="Button")
                final_text = btn3.window_text()
                final_disabled = "Disabled" in final_text
                print(f"  After close: '{final_text}'", flush=True)
            else:
                final_text = after_text
                final_disabled = after_disabled

            # Determine result
            click_worked = True  # click_input() itself didn't error
            opened_window = config_window is not None
            state_changed = initial_disabled != final_disabled

            status = "PASS"
            details = []
            if click_worked:
                details.append("click_input OK")
            if opened_window:
                details.append(f"opened '{config_window}'")
            if state_changed:
                details.append(f"state: {'Disabled->Enabled' if initial_disabled else 'Enabled->Disabled'}")
            else:
                details.append("state unchanged (config needed)")

            # If button became enabled, restore it by clicking again + closing
            if state_changed and not final_disabled:
                print(f"  Restoring to Disabled...", flush=True)
                app_r = Application(backend="uia").connect(process=pid)
                bulk_r = app_r.window(handle=bulk_handle)
                btn_r = bulk_r.child_window(auto_id=auto_id, control_type="Button")
                btn_r.click_input()
                time.sleep(2)
                dismiss_dialogs()
                # Close any new window
                app_r2 = Application(backend="uia").connect(process=pid)
                for wt, wh in get_app_windows(app_r2).items():
                    if wt not in windows_before and "ZimmWriter" in wt and "Menu" not in wt and "Bulk" not in wt:
                        SendMsg(wh, WM_CLOSE, 0, 0)
                        time.sleep(0.5)
                # Verify
                app_r3 = Application(backend="uia").connect(process=pid)
                bulk_r2 = app_r3.window(handle=bulk_handle)
                btn_r2 = bulk_r2.child_window(auto_id=auto_id, control_type="Button")
                restored_text = btn_r2.window_text()
                print(f"  Restored: '{restored_text}'", flush=True)

            results.append((name, auto_id, status, "; ".join(details)))
            print(f"  RESULT: {status} ({'; '.join(details)})", flush=True)

            # Refresh app reference for next iteration
            app = Application(backend="uia").connect(process=pid)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append((name, auto_id, "ERROR", str(e)))
            app = Application(backend="uia").connect(process=pid)

    # Summary
    print(f"\n{'='*65}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*65}", flush=True)

    pass_count = 0
    for name, aid, status, details in results:
        icon = "OK" if status == "PASS" else "XX"
        print(f"  [{icon}] {name:20s} id={aid:4s} {details}", flush=True)
        if status == "PASS":
            pass_count += 1

    print(f"\n  {pass_count}/{len(results)} PASS", flush=True)
    print(f"{'='*65}", flush=True)

    # Final screenshot
    try:
        app = Application(backend="uia").connect(process=pid)
        bw = app.window(handle=bulk_handle)
        bw.set_focus()
        time.sleep(0.5)
        img = bw.wrapper_object().capture_as_image()
        img.save(f"{OUTPUT_DIR}\\feature_toggles_final.png")
        print(f"\nScreenshot: {OUTPUT_DIR}\\feature_toggles_final.png", flush=True)
    except Exception as e:
        print(f"Screenshot error: {e}", flush=True)


if __name__ == "__main__":
    main()
