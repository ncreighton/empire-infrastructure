"""Test feature toggle with explicit focus and screenshot verification."""
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

print("Connecting...", flush=True)
result = subprocess.run(
    ['powershell', '-Command',
     "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | "
     "Select-Object -First 1 -ExpandProperty Id"],
    capture_output=True, text=True, timeout=10
)
pid = int(result.stdout.strip())
app = Application(backend="uia").connect(process=pid)

# List windows
for w in app.windows():
    print(f"  {w.window_text()} (handle={w.handle})", flush=True)

# Find Bulk Writer
bulk = None
for w in app.windows():
    if "Bulk" in w.window_text() and "Blog" in w.window_text():
        bulk = app.window(handle=w.handle)
        break

if not bulk:
    print("ERROR: Bulk Blog Writer not found!", flush=True)
    exit(1)

# Focus the window
print("\nFocusing Bulk Writer...", flush=True)
bulk.set_focus()
bulk.restore()
time.sleep(1)

# Screenshot before
img = bulk.wrapper_object().capture_as_image()
img.save(f"{OUTPUT_DIR}\\toggle_before.png")
print("Screenshot before saved", flush=True)

# Read WordPress button
btn = bulk.child_window(auto_id="93", control_type="Button")
print(f"\nWordPress button: '{btn.window_text()}'", flush=True)
print(f"  Is visible: {btn.is_visible()}", flush=True)
print(f"  Is enabled: {btn.is_enabled()}", flush=True)

# Get button rectangle for precise click
rect = btn.rectangle()
print(f"  Rectangle: left={rect.left} top={rect.top} right={rect.right} bottom={rect.bottom}", flush=True)
center_x = (rect.left + rect.right) // 2
center_y = (rect.top + rect.bottom) // 2
print(f"  Center: ({center_x}, {center_y})", flush=True)

# Try click_input with explicit coords
print("\nClicking button at center...", flush=True)
btn.click_input()
time.sleep(3)

# Check windows after
app2 = Application(backend="uia").connect(process=pid)
print("\nWindows after click:", flush=True)
for w in app2.windows():
    title = w.window_text()
    print(f"  {title} (handle={w.handle})", flush=True)
    if "Enable" in title or "Upload" in title or "Config" in title:
        print(f"    -> CONFIG WINDOW FOUND!", flush=True)
        # Screenshot it
        cw = app2.window(handle=w.handle)
        cw.set_focus()
        time.sleep(0.5)
        img2 = cw.wrapper_object().capture_as_image()
        img2.save(f"{OUTPUT_DIR}\\toggle_config_window.png")
        print(f"    Screenshot saved", flush=True)
        # Close it
        SendMsg(w.handle, WM_CLOSE, 0, 0)
        time.sleep(1)

# Re-check button state
app3 = Application(backend="uia").connect(process=pid)
bulk3 = app3.window(handle=bulk.handle)
btn3 = bulk3.child_window(auto_id="93", control_type="Button")
print(f"\nWordPress after: '{btn3.window_text()}'", flush=True)

# Screenshot after
bulk3.set_focus()
time.sleep(0.3)
img3 = bulk3.wrapper_object().capture_as_image()
img3.save(f"{OUTPUT_DIR}\\toggle_after.png")
print("Screenshot after saved", flush=True)

print("\nDone!", flush=True)
