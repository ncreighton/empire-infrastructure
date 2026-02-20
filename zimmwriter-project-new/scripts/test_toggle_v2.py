"""Test toggle with pyautogui click and invoke fallback."""
import subprocess
import time
import ctypes
from ctypes import wintypes
from pywinauto import Application
import pyautogui

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

# Find Bulk Writer
bulk = None
for w in app.windows():
    t = w.window_text()
    print(f"  {t} (handle={w.handle})", flush=True)
    if "Bulk" in t and "Blog" in t:
        bulk = app.window(handle=w.handle)

if not bulk:
    print("ERROR: No Bulk Writer!", flush=True)
    exit(1)

# Focus it
bulk.set_focus()
bulk.restore()
time.sleep(1)

# Test WordPress button (auto_id=93) with 3 different click methods
btn = bulk.child_window(auto_id="93", control_type="Button")
text0 = btn.window_text()
print(f"\nWordPress initial: '{text0}'", flush=True)

# Method 1: invoke()
print("\n--- Method 1: invoke() ---", flush=True)
try:
    btn.invoke()
    time.sleep(2)
    app2 = Application(backend="uia").connect(process=pid)
    wins = [w.window_text() for w in app2.windows()]
    print(f"  Windows: {wins}", flush=True)
    new_config = [w for w in wins if "Enable" in w or "Upload" in w]
    if new_config:
        print(f"  CONFIG WINDOW: {new_config[0]}", flush=True)
        for w in app2.windows():
            if w.window_text() in new_config:
                SendMsg(w.handle, WM_CLOSE, 0, 0)
                time.sleep(0.5)
    btn_check = app2.window(handle=bulk.handle).child_window(auto_id="93", control_type="Button")
    print(f"  After: '{btn_check.window_text()}'", flush=True)
except Exception as e:
    print(f"  invoke() error: {e}", flush=True)

# Method 2: click_input() with explicit focus
print("\n--- Method 2: click_input() ---", flush=True)
try:
    app3 = Application(backend="uia").connect(process=pid)
    bw3 = app3.window(handle=bulk.handle)
    bw3.set_focus()
    time.sleep(0.5)
    btn3 = bw3.child_window(auto_id="93", control_type="Button")
    text1 = btn3.window_text()
    print(f"  Before: '{text1}'", flush=True)
    btn3.click_input()
    time.sleep(2)
    app4 = Application(backend="uia").connect(process=pid)
    wins4 = [w.window_text() for w in app4.windows()]
    print(f"  Windows: {wins4}", flush=True)
    new_config = [w for w in wins4 if "Enable" in w or "Upload" in w]
    if new_config:
        print(f"  CONFIG WINDOW: {new_config[0]}", flush=True)
        for w in app4.windows():
            if w.window_text() in new_config:
                SendMsg(w.handle, WM_CLOSE, 0, 0)
                time.sleep(0.5)
    btn_check = app4.window(handle=bulk.handle).child_window(auto_id="93", control_type="Button")
    print(f"  After: '{btn_check.window_text()}'", flush=True)
except Exception as e:
    print(f"  click_input() error: {e}", flush=True)

# Method 3: pyautogui coordinate click
print("\n--- Method 3: pyautogui.click() ---", flush=True)
try:
    app5 = Application(backend="uia").connect(process=pid)
    bw5 = app5.window(handle=bulk.handle)
    bw5.set_focus()
    time.sleep(0.5)
    btn5 = bw5.child_window(auto_id="93", control_type="Button")
    text2 = btn5.window_text()
    print(f"  Before: '{text2}'", flush=True)
    rect = btn5.rectangle()
    cx = (rect.left + rect.right) // 2
    cy = (rect.top + rect.bottom) // 2
    print(f"  Clicking at ({cx}, {cy})", flush=True)
    pyautogui.click(cx, cy)
    time.sleep(3)
    app6 = Application(backend="uia").connect(process=pid)
    wins6 = [w.window_text() for w in app6.windows()]
    print(f"  Windows: {wins6}", flush=True)
    new_config = [w for w in wins6 if "Enable" in w or "Upload" in w]
    if new_config:
        print(f"  CONFIG WINDOW: {new_config[0]}", flush=True)
        for w in app6.windows():
            if w.window_text() in new_config:
                SendMsg(w.handle, WM_CLOSE, 0, 0)
                time.sleep(0.5)
    btn_check = app6.window(handle=bulk.handle).child_window(auto_id="93", control_type="Button")
    print(f"  After: '{btn_check.window_text()}'", flush=True)
except Exception as e:
    print(f"  pyautogui error: {e}", flush=True)

# Method 4: BM_CLICK Win32 message
print("\n--- Method 4: Win32 BM_CLICK ---", flush=True)
try:
    BM_CLICK = 0x00F5
    app7 = Application(backend="uia").connect(process=pid)
    bw7 = app7.window(handle=bulk.handle)
    bw7.set_focus()
    time.sleep(0.5)
    btn7 = bw7.child_window(auto_id="93", control_type="Button")
    text3 = btn7.window_text()
    print(f"  Before: '{text3}'", flush=True)
    hwnd = btn7.handle
    print(f"  Button handle: {hwnd}", flush=True)
    SendMsg(hwnd, BM_CLICK, 0, 0)
    time.sleep(3)
    app8 = Application(backend="uia").connect(process=pid)
    wins8 = [w.window_text() for w in app8.windows()]
    print(f"  Windows: {wins8}", flush=True)
    new_config = [w for w in wins8 if "Enable" in w or "Upload" in w]
    if new_config:
        print(f"  CONFIG WINDOW: {new_config[0]}", flush=True)
        for w in app8.windows():
            if w.window_text() in new_config:
                SendMsg(w.handle, WM_CLOSE, 0, 0)
                time.sleep(0.5)
    btn_check = app8.window(handle=bulk.handle).child_window(auto_id="93", control_type="Button")
    print(f"  After: '{btn_check.window_text()}'", flush=True)
except Exception as e:
    print(f"  BM_CLICK error: {e}", flush=True)

print("\nDone!", flush=True)
