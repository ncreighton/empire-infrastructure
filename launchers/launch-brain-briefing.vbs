' EMPIRE-BRAIN Daily Briefing
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\EMPIRE-BRAIN && set PYTHONPATH=D:\Claude Code Projects\EMPIRE-BRAIN && D:\Python314\pythonw.exe agents/briefing_agent.py >> D:\EmpireLogs\briefing.log 2>&1""", 0, False
