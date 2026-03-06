' VideoForge Intelligence API — Port 8090
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\videoforge-engine && set PYTHONPATH=D:\Claude Code Projects\videoforge-engine && D:\Python314\pythonw.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8090""", 0, False
