' Revid Forge — Publish Scheduler
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\revid-forge && set PYTHONPATH=D:\Claude Code Projects\revid-forge && D:\Python314\pythonw.exe scripts/publish_scheduler.py >> D:\EmpireLogs\revid_publish.log 2>&1""", 0, False
