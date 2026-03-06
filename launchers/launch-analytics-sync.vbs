' Daily Analytics Sync
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects && D:\Python314\pythonw.exe daily_analytics_sync.py""", 0, False
