"""Step 2: Navigate to WordPress Settings and save all 14 sites."""
import json
import subprocess
import time
import sys
import os
import ctypes
from ctypes import wintypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pywinauto import Application, Desktop

SITES_JSON = r"D:\Claude Code Projects\config\sites.json"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


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


def dismiss_dialog(max_wait=12):
    """Wait for and dismiss dialog. Return text content."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        for w in Desktop(backend='uia').windows():
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


def main():
    sites = load_credentials()
    print(f"Loaded {len(sites)} sites")

    # Connect
    result = subprocess.run(
        ['powershell', '-Command',
         "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
        capture_output=True, text=True, timeout=10
    )
    pid = int(result.stdout.strip())
    app = Application(backend='uia').connect(process=pid)
    window = app.top_window()
    title = window.window_text()
    print(f"Connected: {title}")

    # Navigate to WordPress Settings
    if "Setup WordPress" not in title:
        if "Option Menu" in title:
            window.child_window(auto_id="55", control_type="Button").invoke()
            time.sleep(3)
            app = Application(backend='uia').connect(process=pid)
            window = app.top_window()
        elif "Menu" in title:
            window.child_window(title="Options Menu", control_type="Button").invoke()
            time.sleep(3)
            app = Application(backend='uia').connect(process=pid)
            window = app.top_window()
            window.child_window(auto_id="55", control_type="Button").invoke()
            time.sleep(3)
            app = Application(backend='uia').connect(process=pid)
            window = app.top_window()

    print(f"On: {window.window_text()}")

    # Save each site (all 3 fields + Save New Site + Save New User)
    success_sites = 0
    success_users = 0
    failed = []

    for i, site in enumerate(sites, 1):
        print(f"\n[{i:2d}/{len(sites)}] {site['domain']}")

        # Fill ALL 3 fields
        url_f = window.child_window(auto_id="79", control_type="Edit")
        user_f = window.child_window(auto_id="81", control_type="Edit")
        pass_f = window.child_window(auto_id="83", control_type="Edit")

        url_f.set_edit_text(site["url"])
        time.sleep(0.15)
        user_f.set_edit_text(site["user"])
        time.sleep(0.15)
        pass_f.set_edit_text(site["app_password"])
        time.sleep(0.15)

        # Verify fields
        url_val = url_f.get_value()
        user_val = user_f.get_value()
        pass_val = pass_f.get_value()
        print(f"  URL:  {url_val}")
        print(f"  User: {user_val}")
        print(f"  Pass: {pass_val[:12]}...")

        if not url_val or not user_val or not pass_val:
            print("  ERROR: Empty field(s)!")
            failed.append(site["domain"])
            continue

        # Click Save New Site
        print("  Save New Site...")
        btn = window.child_window(auto_id="86", control_type="Button")
        try:
            btn.invoke()
        except Exception:
            btn.click_input()

        # Wait for validation (ZimmWriter tests the connection)
        dialog = dismiss_dialog(max_wait=15)
        if dialog:
            short = dialog[:100]
            print(f"  Dialog: {short}")
            if "could not connect" in dialog.lower():
                failed.append(site["domain"])
                continue
            else:
                success_sites += 1
        else:
            print("  No dialog - site saved!")
            success_sites += 1

        # Now Save New User
        print("  Save New User...")
        btn = window.child_window(auto_id="90", control_type="Button")
        try:
            btn.invoke()
        except Exception:
            btn.click_input()

        dialog = dismiss_dialog(max_wait=15)
        if dialog:
            short = dialog[:100]
            print(f"  Dialog: {short}")
            if "could not connect" not in dialog.lower():
                success_users += 1
        else:
            print("  No dialog - user saved!")
            success_users += 1

        time.sleep(1)

    # Final verification
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Sites saved:  {success_sites}/{len(sites)}")
    print(f"  Users saved:  {success_users}/{len(sites)}")
    print(f"  Failed:       {len(failed)}")
    if failed:
        print(f"  Failed sites: {', '.join(failed)}")
    print(f"{'=' * 60}")

    # Check dropdowns
    try:
        dd = window.child_window(auto_id="94", control_type="ComboBox")
        hwnd = dd.handle
        SendMsg = ctypes.windll.user32.SendMessageW
        SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        SendMsg.restype = ctypes.c_long
        count = SendMsg(hwnd, 0x0146, 0, 0)  # CB_GETCOUNT
        print(f"\nSaved sites dropdown: {count} items")

        SendMsg2 = ctypes.windll.user32.SendMessageW
        SendMsg2.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, ctypes.c_wchar_p]
        SendMsg2.restype = ctypes.c_long
        for idx in range(count):
            length = SendMsg(hwnd, 0x0149, idx, 0)
            if length >= 0:
                buf = ctypes.create_unicode_buffer(length + 2)
                SendMsg2(hwnd, 0x0148, idx, buf)
                print(f"  [{idx}] {buf.value}")
    except Exception as e:
        print(f"Dropdown read error: {e}")

    # Screenshot
    try:
        img = window.capture_as_image()
        img.save(os.path.join(OUTPUT_DIR, "wp_final_result.png"))
        print(f"Screenshot: {OUTPUT_DIR}/wp_final_result.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
