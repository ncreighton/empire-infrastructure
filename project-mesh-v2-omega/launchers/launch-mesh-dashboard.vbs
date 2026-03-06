' Mesh Dashboard v3.0 — Port 8100
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\project-mesh-v2-omega && set PYTHONPATH=D:\Claude Code Projects\project-mesh-v2-omega && D:\Python314\pythonw.exe -m uvicorn dashboard.api:app --host 0.0.0.0 --port 8100""", 0, False
