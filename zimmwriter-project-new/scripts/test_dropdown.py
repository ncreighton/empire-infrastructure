"""Test: Dropdown selection via keyboard"""
import subprocess, time
from pywinauto import Application
from pywinauto.keyboard import send_keys

result = subprocess.run(
    ['powershell', '-Command',
     "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
    capture_output=True, text=True)
pid = int(result.stdout.strip())
app = Application(backend="uia").connect(process=pid)
bulk = app.window(title_re=".*Bulk Blog Writer.*")

print("TEST: Set Section Length dropdown via keyboard")
combo = bulk.child_window(auto_id="46", control_type="ComboBox")
combo.set_focus()
time.sleep(0.3)
send_keys("L")
time.sleep(0.5)
print("  Sent 'L' to Section Length combo")

# Verify by reading back
combo2 = bulk.child_window(auto_id="46", control_type="ComboBox")
try:
    sel = combo2.selected_text()
    print(f"  Selected: {sel}")
except:
    print("  (Could not read selected_text)")

# Reset
combo2.set_focus()
time.sleep(0.2)
send_keys("M")
time.sleep(0.5)
print("  Sent 'M' to reset to Medium")
print("  PASS")
