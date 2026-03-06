' EMPIRE-BRAIN Daily Briefing
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\EMPIRE-BRAIN && D:\Python314\pythonw.exe agents/briefing_agent.py""", 0, False
