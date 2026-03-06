' Empire Dashboard — Port 8000
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\empire-dashboard && set PYTHONPATH=D:\Claude Code Projects\empire-dashboard && D:\Python314\pythonw.exe -m uvicorn main:app --port 8000 >> D:\EmpireLogs\empire_dashboard.log 2>&1""", 0, False
