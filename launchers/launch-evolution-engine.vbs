' EMPIRE-BRAIN Evolution Engine — daemon mode
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\EMPIRE-BRAIN && set PYTHONPATH=D:\Claude Code Projects\EMPIRE-BRAIN && D:\Python314\pythonw.exe agents/evolution_agent.py >> D:\EmpireLogs\evolution.log 2>&1""", 0, False
