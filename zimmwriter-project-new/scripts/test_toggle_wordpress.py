"""Explore the WordPress toggle window that opens when clicking the button."""
import subprocess
import time
import os
from pywinauto import Application

OUTPUT_DIR = r"D:\Claude Code Projects\zimmwriter-project-new\output"

print("Connecting...", flush=True)
result = subprocess.run(
    ['powershell', '-Command',
     "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
    capture_output=True, text=True, timeout=10
)
pid = int(result.stdout.strip())
app = Application(backend="uia").connect(process=pid)

# List all windows
print("\n=== All Windows ===", flush=True)
for w in app.windows():
    title = w.window_text()
    handle = w.handle
    print(f"  {title} (handle={handle})", flush=True)

# Find the WordPress Uploads window
wp_upload = None
bulk_win = None
for w in app.windows():
    title = w.window_text()
    if "WordPress Uploads" in title or "Enable WordPress" in title:
        wp_upload = app.window(handle=w.handle)
    if "Bulk" in title and "Blog" in title:
        bulk_win = app.window(handle=w.handle)

if wp_upload:
    print(f"\n=== WordPress Uploads Window ===", flush=True)
    wp_upload.set_focus()
    time.sleep(0.5)

    # Screenshot
    try:
        img = wp_upload.wrapper_object().capture_as_image()
        img.save(os.path.join(OUTPUT_DIR, "wp_uploads_window.png"))
        print("Screenshot saved", flush=True)
    except Exception as e:
        print(f"Screenshot error: {e}", flush=True)

    # List all controls
    print("\nControls:", flush=True)
    for ctrl in wp_upload.descendants():
        try:
            ctype = ctrl.friendly_class_name()
            text = ctrl.window_text()[:60]
            aid = ctrl.automation_id()
            if text or aid:
                print(f"  {ctype:15s} id={aid:5s} text='{text}'", flush=True)
        except Exception:
            pass
else:
    print("WordPress Uploads window not found", flush=True)

print("\nDone!", flush=True)
