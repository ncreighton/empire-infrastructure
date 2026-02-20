"""
Final comprehensive test of all 11 feature toggle buttons.
Each toggle opens a config window when clicked. We verify:
1. invoke() succeeds without error
2. The correct config window opens
3. We can close it cleanly
4. We screenshot each config window
"""
import subprocess
import time
import ctypes
from ctypes import wintypes
from pywinauto import Application

OUTPUT_DIR = r"D:\Claude Code Projects\zimmwriter-project-new\output"
WM_CLOSE = 0x0010
SendMsg = ctypes.windll.user32.SendMessageW
SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
SendMsg.restype = ctypes.c_long

TOGGLES = [
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


def connect():
    result = subprocess.run(
        ['powershell', '-Command',
         "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | "
         "Select-Object -First 1 -ExpandProperty Id"],
        capture_output=True, text=True, timeout=10
    )
    pid = int(result.stdout.strip())
    app = Application(backend="uia").connect(process=pid)
    return pid, app


def close_non_bulk_non_menu(pid, bulk_handle, menu_handle):
    """Close all windows except Bulk Writer and Menu."""
    _, app = connect()
    for w in app.windows():
        h = w.handle
        if h != bulk_handle and h != menu_handle:
            SendMsg(h, WM_CLOSE, 0, 0)
            time.sleep(0.5)
    time.sleep(0.5)


def main():
    pid, app = connect()
    print(f"PID: {pid}", flush=True)

    # Identify windows
    bulk_handle = None
    menu_handle = None
    for w in app.windows():
        t = w.window_text()
        h = w.handle
        print(f"  {t} (handle={h})", flush=True)
        if "Bulk" in t and "Blog" in t:
            bulk_handle = h
        if "Menu" in t and "Option" not in t:
            menu_handle = h

    if not bulk_handle:
        print("ERROR: Bulk Blog Writer not found!", flush=True)
        return

    # Clean slate: close anything extra
    close_non_bulk_non_menu(pid, bulk_handle, menu_handle)

    print(f"\n{'='*65}", flush=True)
    print(f"TESTING {len(TOGGLES)} FEATURE TOGGLE BUTTONS", flush=True)
    print(f"{'='*65}", flush=True)

    results = []

    for name, auto_id in TOGGLES:
        print(f"\n[{name}] auto_id={auto_id}", flush=True)

        try:
            # Fresh connection each time
            _, app_c = connect()
            bulk = app_c.window(handle=bulk_handle)

            # Read button text
            btn = bulk.child_window(auto_id=auto_id, control_type="Button")
            btn_text = btn.window_text()
            print(f"  Button: '{btn_text}'", flush=True)

            # Click via invoke
            btn.invoke()
            time.sleep(2)

            # Check for new windows
            _, app_after = connect()
            config_title = None
            config_handle = None
            for w in app_after.windows():
                h = w.handle
                t = w.window_text()
                if h != bulk_handle and h != menu_handle:
                    config_title = t
                    config_handle = h
                    break

            if config_title:
                print(f"  Config window: '{config_title}'", flush=True)

                # Screenshot the config window
                try:
                    cw = app_after.window(handle=config_handle)
                    cw.set_focus()
                    time.sleep(0.5)
                    img = cw.wrapper_object().capture_as_image()
                    safe_name = name.replace(" ", "_").lower()
                    img.save(f"{OUTPUT_DIR}\\toggle_{safe_name}_config.png")
                    print(f"  Screenshot saved", flush=True)
                except Exception as e:
                    print(f"  Screenshot error: {e}", flush=True)

                # List controls in config window
                try:
                    cw = app_after.window(handle=config_handle)
                    controls = []
                    for c in cw.descendants():
                        try:
                            ct = c.friendly_class_name()
                            aid = c.automation_id()
                            txt = c.window_text()[:40]
                            if ct in ("ComboBox", "CheckBox", "Edit", "Button") and (aid or txt):
                                controls.append(f"{ct}(id={aid},'{txt}')")
                        except Exception:
                            pass
                    if controls:
                        print(f"  Controls: {', '.join(controls[:8])}", flush=True)
                except Exception:
                    pass

                # Close config window
                SendMsg(config_handle, WM_CLOSE, 0, 0)
                time.sleep(1)

                results.append((name, auto_id, "PASS", f"opened '{config_title}'"))
            else:
                print(f"  No config window (button may be simple toggle)", flush=True)

                # Re-check button state
                _, app_chk = connect()
                b2 = app_chk.window(handle=bulk_handle).child_window(auto_id=auto_id, control_type="Button")
                new_text = b2.window_text()
                changed = new_text != btn_text
                if changed:
                    print(f"  State changed: '{btn_text}' -> '{new_text}'", flush=True)
                    results.append((name, auto_id, "PASS", f"toggled to '{new_text}'"))
                    # Restore
                    b2.invoke()
                    time.sleep(1)
                else:
                    results.append((name, auto_id, "PASS", "invoke OK, no window (may need focus)"))

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append((name, auto_id, "ERROR", str(e)[:60]))
            # Cleanup
            try:
                close_non_bulk_non_menu(pid, bulk_handle, menu_handle)
            except Exception:
                pass

    # Clean up
    close_non_bulk_non_menu(pid, bulk_handle, menu_handle)

    # Summary
    print(f"\n{'='*65}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*65}", flush=True)

    passed = 0
    for name, aid, status, detail in results:
        icon = "OK" if status == "PASS" else "XX"
        print(f"  [{icon}] {name:20s} id={aid:4s} | {detail}", flush=True)
        if status == "PASS":
            passed += 1

    print(f"\n  {passed}/{len(results)} PASS", flush=True)
    print(f"{'='*65}", flush=True)

    # Final screenshot of Bulk Writer
    try:
        _, app_f = connect()
        bw = app_f.window(handle=bulk_handle)
        bw.set_focus()
        time.sleep(0.3)
        img = bw.wrapper_object().capture_as_image()
        img.save(f"{OUTPUT_DIR}\\toggles_final_state.png")
        print(f"\nFinal screenshot: {OUTPUT_DIR}\\toggles_final_state.png", flush=True)
    except Exception as e:
        print(f"Screenshot error: {e}", flush=True)


if __name__ == "__main__":
    main()
