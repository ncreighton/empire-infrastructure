' Vision Service — Port 8002 (uses Python 3.12 venv)
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""D:\Claude Code Projects\geelark-automation\.venv-vision\Scripts\pythonw.exe D:\Claude Code Projects\geelark-automation\services\vision_service.py""", 0, False
