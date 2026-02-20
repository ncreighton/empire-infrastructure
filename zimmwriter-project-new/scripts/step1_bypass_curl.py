"""Step 1: Enable Bypass Windows CURL and save options."""
import subprocess, time, sys
sys.path.insert(0, '.')
from pywinauto import Application, Desktop

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

# Close WP Settings if on that screen
if "Setup WordPress" in title:
    print("Closing WordPress Settings...")
    window.child_window(title="Close", control_type="Button").click_input()
    time.sleep(3)
    app = Application(backend='uia').connect(process=pid)
    window = app.top_window()
    title = window.window_text()
    print(f"Now: {title}")

# Navigate to Options Menu
if "Option Menu" not in title:
    if "Menu" in title:
        window.child_window(title="Options Menu", control_type="Button").invoke()
        time.sleep(3)
        app = Application(backend='uia').connect(process=pid)
        window = app.top_window()
        print(f"Navigated to: {window.window_text()}")

# Toggle Bypass Windows CURL (auto_id=33)
cb = window.child_window(auto_id="33", control_type="CheckBox")
state = cb.get_toggle_state()
print(f"Bypass CURL: {'ON' if state == 1 else 'OFF'}")
if state == 0:
    cb.toggle()
    time.sleep(0.5)
    print(f"Toggled to: {'ON' if cb.get_toggle_state() == 1 else 'OFF'}")

# Save Options (auto_id=54)
print("Saving options...")
window.child_window(auto_id="54", control_type="Button").invoke()
time.sleep(3)

# Dismiss any dialog
for w in Desktop(backend='uia').windows():
    t = w.window_text()
    if "ZimmWriter" in t and any(k in t for k in ["Info", "Error", "Warning", "Success"]):
        print(f"Dialog: {t}")
        try:
            w.child_window(title="OK", control_type="Button").click_input()
        except Exception:
            btns = w.descendants(control_type="Button")
            if btns:
                btns[0].click_input()
        time.sleep(1)

print("Done! Bypass CURL enabled and saved.")
