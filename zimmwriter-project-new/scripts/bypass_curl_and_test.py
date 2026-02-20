"""
Access the Options Menu window directly (by handle) to enable Bypass CURL,
then access the WordPress Settings window to test saving a site.
"""
import subprocess
import time
import ctypes
from ctypes import wintypes
from pywinauto import Application, Desktop

OUTPUT_DIR = "D:\\Claude Code Projects\\zimmwriter-project-new\\output"

result = subprocess.run(
    ['powershell', '-Command',
     "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
    capture_output=True, text=True, timeout=10
)
pid = int(result.stdout.strip())
app = Application(backend='uia').connect(process=pid)

# Find all windows - use handle-based window specs for child_window support
raw_windows = app.windows()
options_win = None
wp_win = None

for w in raw_windows:
    title = w.window_text()
    handle = w.handle
    print(f"Window: {title} (handle={handle})")
    if "Option Menu" in title:
        options_win = app.window(handle=handle)
    if "Setup WordPress" in title:
        wp_win = app.window(handle=handle)

# === Enable Bypass CURL on Options Menu ===
if options_win:
    print(f"\nOptions Menu window found: handle={options_win.handle}")
    cb = options_win.child_window(auto_id="33", control_type="CheckBox")
    state = cb.get_toggle_state()
    print(f"Bypass CURL: {'ON' if state == 1 else 'OFF'}")

    if state == 0:
        cb.toggle()
        time.sleep(0.3)
        print(f"Toggled to: {'ON' if cb.get_toggle_state() == 1 else 'OFF'}")

    # Save Options
    print("Saving options...")
    options_win.child_window(auto_id="54", control_type="Button").invoke()
    time.sleep(2)

    # Dismiss dialog
    for d in Desktop(backend="uia").windows():
        t = d.window_text()
        if "ZimmWriter" in t and any(k in t for k in ["Info", "Error", "Success", "Warning"]):
            print(f"Dialog: {t}")
            try:
                d.child_window(title="OK", control_type="Button").click_input()
            except Exception:
                pass
            time.sleep(1)

    print("Options saved!")
else:
    print("Options Menu window not found!")

# === Test WordPress save ===
if wp_win:
    print(f"\nWordPress window found: handle={wp_win.handle}")

    # Fill fields
    url_f = wp_win.child_window(auto_id="79", control_type="Edit")
    user_f = wp_win.child_window(auto_id="81", control_type="Edit")
    pass_f = wp_win.child_window(auto_id="83", control_type="Edit")

    url_f.set_edit_text("https://smarthomewizards.com")
    time.sleep(0.15)
    user_f.set_edit_text("SmartHomeGuru")
    time.sleep(0.15)
    pass_f.set_edit_text("E2Pe BGoq 7nOd I2eP BbCu BtGm")
    time.sleep(0.15)

    print(f"URL:  {url_f.get_value()}")
    print(f"User: {user_f.get_value()}")
    print(f"Pass: {pass_f.get_value()[:15]}...")

    # Save New Site
    print("\nClicking Save New Site...")
    wp_win.child_window(auto_id="86", control_type="Button").invoke()
    print("Waiting for validation (up to 15s)...")
    time.sleep(12)

    # Check for dialog
    found = False
    for d in Desktop(backend="uia").windows():
        t = d.window_text()
        if "ZimmWriter" in t and any(k in t for k in ["Error", "Info", "Success", "Warning"]):
            # Read content
            for s in d.descendants(control_type="Text"):
                txt = s.window_text()
                if txt and len(txt) > 10 and t not in txt:
                    print(f"Dialog content: {txt[:250]}")
                    break
            try:
                d.child_window(title="OK", control_type="Button").click_input()
            except Exception:
                btns = d.descendants(control_type="Button")
                if btns:
                    btns[0].click_input()
            found = True
            time.sleep(1)
            break

    if not found:
        print("NO ERROR DIALOG! Site may have saved successfully!")

    # Take screenshot
    try:
        img = wp_win.capture_as_image()
        img.save(f"{OUTPUT_DIR}\\wp_after_bypass_test.png")
        print(f"Screenshot saved: {OUTPUT_DIR}\\wp_after_bypass_test.png")
    except Exception as e:
        print(f"Screenshot error: {e}")

    # Check dropdown
    print("\nChecking saved sites dropdown...")
    dd = wp_win.child_window(auto_id="94", control_type="ComboBox")
    hwnd = dd.handle
    SendMsg = ctypes.windll.user32.SendMessageW
    SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMsg.restype = ctypes.c_long
    count = SendMsg(hwnd, 0x0146, 0, 0)  # CB_GETCOUNT
    print(f"Items in dropdown: {count}")

    SendMsg2 = ctypes.windll.user32.SendMessageW
    SendMsg2.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, ctypes.c_wchar_p]
    SendMsg2.restype = ctypes.c_long
    for i in range(count):
        length = SendMsg(hwnd, 0x0149, i, 0)
        if length >= 0:
            buf = ctypes.create_unicode_buffer(length + 2)
            SendMsg2(hwnd, 0x0148, i, buf)
            print(f"  [{i}] {buf.value}")
else:
    print("WordPress Settings window not found!")

print("\nDone!")
