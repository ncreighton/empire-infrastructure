' Revid Forge — Publish Scheduler
' Uses python.exe (NOT pythonw.exe which silently swallows errors)
' Window hidden by VBS Run param 0
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\revid-forge && D:\Python314\python.exe scripts/publish_scheduler.py >> logs\publish_scheduler_task.log 2>&1""", 0, False
