' Printables Empire — Daily Publish
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\printables-empire && D:\Python314\pythonw.exe daily_publish.py""", 0, False
