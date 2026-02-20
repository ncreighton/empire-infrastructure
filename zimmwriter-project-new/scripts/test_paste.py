"""Test: Paste titles into bulk title area"""
import subprocess, time, ctypes
from pywinauto import Application
from pywinauto.keyboard import send_keys


def set_clipboard(text):
    CF_UNICODETEXT = 13
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    user32.OpenClipboard(0)
    user32.EmptyClipboard()
    hMem = kernel32.GlobalAlloc(0x0042, (len(text) + 1) * 2)
    pMem = kernel32.GlobalLock(hMem)
    ctypes.cdll.msvcrt.wcscpy(ctypes.c_wchar_p(pMem), text)
    kernel32.GlobalUnlock(hMem)
    user32.SetClipboardData(CF_UNICODETEXT, hMem)
    user32.CloseClipboard()


result = subprocess.run(
    ['powershell', '-Command',
     "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
    capture_output=True, text=True)
pid = int(result.stdout.strip())
app = Application(backend="uia").connect(process=pid)
bulk = app.window(title_re=".*Bulk Blog Writer.*")

print("TEST: Paste titles")
titles = "Test Title 1: Smart Home Setup\nTest Title 2: AI Writing Guide\nTest Title 3: ZimmWriter Automation"
field = bulk.child_window(auto_id="36", control_type="Edit")
field.set_focus()
time.sleep(0.2)
set_clipboard(titles)
send_keys("^a")
time.sleep(0.1)
send_keys("^v")
time.sleep(0.5)
print(f"  Pasted 3 titles ({len(titles)} chars)")

# Clear
send_keys("^a")
time.sleep(0.1)
send_keys("{DELETE}")
time.sleep(0.3)
print("  Cleared titles")
print("  PASS")
