' EMPIRE-BRAIN Evolution Engine — daemon mode
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\EMPIRE-BRAIN && D:\Python314\pythonw.exe agents/evolution_agent.py""", 0, False
