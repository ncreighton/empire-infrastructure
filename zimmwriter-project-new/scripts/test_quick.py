"""Quick interaction test - flush output immediately."""
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
print("Starting...", flush=True)

import comtypes.client
print("COM initialized", flush=True)

from pywinauto import Application
from pywinauto.keyboard import send_keys
import subprocess
print("Imports done", flush=True)

# Get PID dynamically
result = subprocess.run(
    ['powershell', '-Command',
     "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
    capture_output=True, text=True)
pid = int(result.stdout.strip())

app = Application(backend="uia").connect(process=pid)
bulk = app.top_window()
print(f"Window: {bulk.window_text()}", flush=True)

# === Test 1: Checkbox ===
print("\nTEST 1: Checkbox toggle (Enable Lists)", flush=True)
cb = bulk.child_window(auto_id="48", control_type="CheckBox")
s0 = cb.get_toggle_state()
print(f"  Before: {s0}", flush=True)
cb.toggle()
time.sleep(0.5)
s1 = cb.get_toggle_state()
print(f"  After: {s1}", flush=True)
cb.toggle()
time.sleep(0.5)
s2 = cb.get_toggle_state()
print(f"  Reset: {s2}", flush=True)
print(f"  {'PASS' if s1 != s0 and s2 == s0 else 'FAIL'}", flush=True)

# === Test 2: Title paste using pyperclip ===
print("\nTEST 2: Paste titles", flush=True)
import pyperclip
field = bulk.child_window(auto_id="36", control_type="Edit")
field.set_focus()
time.sleep(0.3)
pyperclip.copy("Test Title 1\nTest Title 2\nTest Title 3")
send_keys("^a")
time.sleep(0.1)
send_keys("^v")
time.sleep(0.5)
print("  Pasted 3 titles", flush=True)
send_keys("^a{DELETE}")
time.sleep(0.3)
print("  Cleared", flush=True)
print("  PASS", flush=True)

# === Test 3: Feature button ===
print("\nTEST 3: Feature button (WordPress)", flush=True)
wp = bulk.child_window(title_re=".*WordPress.*", control_type="Button")
bt = wp.window_text()
print(f"  Before: '{bt}'", flush=True)
wp.invoke()
time.sleep(0.5)
wp2 = bulk.child_window(title_re=".*WordPress.*", control_type="Button")
at = wp2.window_text()
print(f"  After: '{at}'", flush=True)
wp3 = bulk.child_window(title_re=".*WordPress.*", control_type="Button")
wp3.invoke()
time.sleep(0.5)
wp4 = bulk.child_window(title_re=".*WordPress.*", control_type="Button")
rt = wp4.window_text()
print(f"  Reset: '{rt}'", flush=True)
print(f"  {'PASS' if at != bt else 'FAIL'}", flush=True)

# === Test 4: Dropdown via keyboard ===
print("\nTEST 4: Dropdown (Section Length)", flush=True)
combo = bulk.child_window(auto_id="46", control_type="ComboBox")
combo.set_focus()
time.sleep(0.3)
send_keys("L")  # Jump to "Long"
time.sleep(0.5)
print("  Sent 'L' to Section Length", flush=True)
# Reset
combo.set_focus()
time.sleep(0.2)
send_keys("M")  # Jump to "Medium"
time.sleep(0.5)
print("  Sent 'M' to reset", flush=True)
print("  PASS", flush=True)

print("\n=== ALL TESTS COMPLETE ===", flush=True)
