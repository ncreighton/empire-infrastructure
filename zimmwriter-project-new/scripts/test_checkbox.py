"""Test: Checkbox toggle via UIA"""
import subprocess, time
from pywinauto import Application

result = subprocess.run(
    ['powershell', '-Command',
     "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
    capture_output=True, text=True)
pid = int(result.stdout.strip())
app = Application(backend="uia").connect(process=pid)
bulk = app.window(title_re=".*Bulk Blog Writer.*")

print("TEST 1: Checkbox toggle by auto_id (Enable Lists)")
cb = bulk.child_window(auto_id="48", control_type="CheckBox")
b = cb.get_toggle_state()
print(f"  Before: {b}")
cb.toggle()
time.sleep(0.3)
a = cb.get_toggle_state()
print(f"  After: {a}")
cb.toggle()
time.sleep(0.3)
r = cb.get_toggle_state()
print(f"  Reset: {r}")
ok = a != b and r == b
print(f"  {'PASS' if ok else 'FAIL'}")

print("\nTEST 2: Checkbox toggle by name (Nuke AI Words)")
cb2 = bulk.child_window(title="Nuke AI Words", control_type="CheckBox")
b2 = cb2.get_toggle_state()
print(f"  Before: {b2}")
cb2.toggle()
time.sleep(0.3)
a2 = cb2.get_toggle_state()
print(f"  After: {a2}")
cb2.toggle()
time.sleep(0.3)
r2 = cb2.get_toggle_state()
print(f"  Reset: {r2}")
ok2 = a2 != b2 and r2 == b2
print(f"  {'PASS' if ok2 else 'FAIL'}")

print("\nTEST 3: Feature button toggle (WordPress)")
wp = bulk.child_window(title_re=".*WordPress.*", control_type="Button")
bt = wp.window_text()
print(f"  Before: '{bt}'")
wp.invoke()
time.sleep(0.5)
at = wp.window_text()
print(f"  After: '{at}'")
wp2 = bulk.child_window(title_re=".*WordPress.*", control_type="Button")
wp2.invoke()
time.sleep(0.5)
rt = wp2.window_text()
print(f"  Reset: '{rt}'")
print(f"  {'PASS' if at != bt else 'FAIL'}")
