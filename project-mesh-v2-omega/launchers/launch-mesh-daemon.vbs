' Mesh Daemon v3.0 — background mode
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\project-mesh-v2-omega && set PYTHONPATH=D:\Claude Code Projects\project-mesh-v2-omega && D:\Python314\pythonw.exe mesh_daemon.py --background""", 0, False
