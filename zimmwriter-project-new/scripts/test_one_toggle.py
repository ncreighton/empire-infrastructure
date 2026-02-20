"""Test a single feature toggle button to verify click_input() works."""
import subprocess
import time
import sys
from pywinauto import Application

print("Starting...", flush=True)

result = subprocess.run(
    ['powershell', '-Command',
     "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
    capture_output=True, text=True, timeout=10
)
pid = int(result.stdout.strip())
print(f"PID: {pid}", flush=True)

app = Application(backend="uia").connect(process=pid)
print("Connected", flush=True)

# Find Bulk Blog Writer
bulk = None
for w in app.windows():
    title = w.window_text()
    print(f"  Window: {title} (handle={w.handle})", flush=True)
    if "Bulk" in title and "Blog" in title:
        bulk = app.window(handle=w.handle)

if not bulk:
    print("ERROR: Bulk Blog Writer not found!", flush=True)
    sys.exit(1)

print(f"\nUsing: {bulk.window_text()}", flush=True)

# Test WordPress toggle (auto_id=93)
print("\n--- Testing WordPress toggle (auto_id=93) ---", flush=True)
btn = bulk.child_window(auto_id="93", control_type="Button")
text_before = btn.window_text()
print(f"  Before: '{text_before}'", flush=True)

print("  Clicking...", flush=True)
btn.click_input()
print("  Waiting 2s...", flush=True)
time.sleep(2)

btn2 = bulk.child_window(auto_id="93", control_type="Button")
text_after = btn2.window_text()
print(f"  After:  '{text_after}'", flush=True)

changed = text_before != text_after
print(f"  Changed: {changed}", flush=True)

if changed:
    print("  Restoring...", flush=True)
    btn2.click_input()
    time.sleep(1)
    btn3 = bulk.child_window(auto_id="93", control_type="Button")
    print(f"  Restored: '{btn3.window_text()}'", flush=True)

print("\nDone!", flush=True)
