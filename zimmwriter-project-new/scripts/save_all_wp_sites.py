"""
Save all 14 WordPress sites in ZimmWriter.
Requires Bypass CURL already enabled (done by bypass_curl_and_test.py).
Accesses WordPress Settings window directly by handle.
"""
import json
import subprocess
import time
import ctypes
from ctypes import wintypes
from pywinauto import Application, Desktop

SITES_JSON = r"D:\Claude Code Projects\config\sites.json"
OUTPUT_DIR = r"D:\Claude Code Projects\zimmwriter-project-new\output"


def load_credentials():
    with open(SITES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    sites = []
    for site_id, config in data["sites"].items():
        wp = config.get("wordpress", {})
        domain = config.get("domain", "")
        if wp.get("user") and wp.get("app_password") and domain:
            sites.append({
                "domain": domain,
                "url": f"https://{domain}",
                "user": wp["user"],
                "app_password": wp["app_password"],
            })
    return sites


def dismiss_dialog(max_wait=15):
    """Wait for and dismiss dialog. Return text content."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        for w in Desktop(backend="uia").windows():
            t = w.window_text()
            if "ZimmWriter" in t and any(k in t for k in ["Error", "Info", "Warning", "Success"]):
                content = None
                try:
                    for s in w.descendants(control_type="Text"):
                        txt = s.window_text()
                        if txt and len(txt) > 10 and txt != t:
                            content = txt
                            break
                except Exception:
                    pass
                try:
                    w.child_window(title="OK", control_type="Button").click_input()
                except Exception:
                    try:
                        btns = w.descendants(control_type="Button")
                        if btns:
                            btns[0].click_input()
                    except Exception:
                        pass
                time.sleep(0.5)
                return content or t
        time.sleep(0.5)
    return None


def read_combo_items(combo_ctrl):
    """Read combo box items using Win32 messages."""
    hwnd = combo_ctrl.handle

    SendMsgInt = ctypes.windll.user32.SendMessageW
    SendMsgInt.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMsgInt.restype = ctypes.c_long

    SendMsgStr = ctypes.windll.user32.SendMessageW
    SendMsgStr.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, ctypes.c_wchar_p]
    SendMsgStr.restype = ctypes.c_long

    count = SendMsgInt(hwnd, 0x0146, 0, 0)  # CB_GETCOUNT
    items = []
    for i in range(count):
        length = SendMsgInt(hwnd, 0x0149, i, 0)  # CB_GETLBTEXTLEN
        if length >= 0:
            buf = ctypes.create_unicode_buffer(length + 2)
            SendMsgStr(hwnd, 0x0148, i, buf)  # CB_GETLBTEXT
            items.append(buf.value)
    return items


def main():
    sites = load_credentials()
    print(f"Loaded {len(sites)} sites\n")

    # Connect
    result = subprocess.run(
        ['powershell', '-Command',
         "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
        capture_output=True, text=True, timeout=10
    )
    pid = int(result.stdout.strip())
    app = Application(backend="uia").connect(process=pid)

    # Find WordPress Settings window
    wp_win = None
    for w in app.windows():
        if "Setup WordPress" in w.window_text():
            wp_win = app.window(handle=w.handle)
            break

    if not wp_win:
        print("WordPress Settings window not found!")
        return

    print(f"WordPress window found\n")

    # Check what's already saved
    try:
        dd = wp_win.child_window(auto_id="94", control_type="ComboBox")
        existing = read_combo_items(dd.wrapper_object())
        print(f"Already saved sites ({len(existing)}):")
        for s in existing:
            print(f"  - {s}")
    except Exception as e:
        print(f"Couldn't read existing sites: {e}")
        existing = []

    # Save each site
    success_sites = 0
    success_users = 0
    failed = []

    for i, site in enumerate(sites, 1):
        # Skip if already in dropdown
        if any(site["domain"] in s for s in existing):
            print(f"[{i:2d}/{len(sites)}] SKIP {site['domain']} (already saved)")
            success_sites += 1
            success_users += 1
            continue

        print(f"[{i:2d}/{len(sites)}] {site['domain']}")

        # Fill all 3 fields
        url_f = wp_win.child_window(auto_id="79", control_type="Edit")
        user_f = wp_win.child_window(auto_id="81", control_type="Edit")
        pass_f = wp_win.child_window(auto_id="83", control_type="Edit")

        url_f.set_edit_text(site["url"])
        time.sleep(0.1)
        user_f.set_edit_text(site["user"])
        time.sleep(0.1)
        pass_f.set_edit_text(site["app_password"])
        time.sleep(0.1)

        # Save New Site
        wp_win.child_window(auto_id="86", control_type="Button").invoke()
        dialog = dismiss_dialog(max_wait=15)

        if dialog and "could not connect" in dialog.lower():
            print(f"  SITE FAILED: {dialog[:80]}")
            failed.append(site["domain"])
            continue
        elif dialog:
            print(f"  Site dialog: {dialog[:80]}")
        else:
            pass  # No dialog = success

        success_sites += 1
        time.sleep(0.5)

        # Save New User
        # Re-fill user and password (may have been cleared after site save)
        user_f = wp_win.child_window(auto_id="81", control_type="Edit")
        pass_f = wp_win.child_window(auto_id="83", control_type="Edit")
        user_f.set_edit_text(site["user"])
        time.sleep(0.1)
        pass_f.set_edit_text(site["app_password"])
        time.sleep(0.1)

        wp_win.child_window(auto_id="90", control_type="Button").invoke()
        dialog = dismiss_dialog(max_wait=15)

        if dialog and "could not connect" in dialog.lower():
            print(f"  USER FAILED: {dialog[:80]}")
        elif dialog:
            print(f"  User dialog: {dialog[:80]}")
            success_users += 1
        else:
            success_users += 1

        time.sleep(1)
        print(f"  DONE")

    # Final verification
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Sites saved: {success_sites}/{len(sites)}")
    print(f"  Users saved: {success_users}/{len(sites)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    # Read final dropdown contents
    try:
        dd = wp_win.child_window(auto_id="94", control_type="ComboBox")
        final_sites = read_combo_items(dd.wrapper_object())
        print(f"\nFinal sites dropdown ({len(final_sites)} items):")
        for s in final_sites:
            print(f"  - {s}")
    except Exception as e:
        print(f"  Dropdown read error: {e}")

    try:
        dd = wp_win.child_window(auto_id="96", control_type="ComboBox")
        final_users = read_combo_items(dd.wrapper_object())
        print(f"\nFinal users dropdown ({len(final_users)} items):")
        for u in final_users:
            print(f"  - {u}")
    except Exception as e:
        print(f"  Users dropdown error: {e}")

    # Screenshot
    try:
        wp_wrapper = wp_win.wrapper_object()
        img = wp_wrapper.capture_as_image()
        img.save(f"{OUTPUT_DIR}\\wp_all_sites_saved.png")
        print(f"\nScreenshot: {OUTPUT_DIR}\\wp_all_sites_saved.png")
    except Exception:
        pass

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
