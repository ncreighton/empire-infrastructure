' Printables Empire — Daily Publish
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\printables-empire && set PYTHONPATH=D:\Claude Code Projects\printables-empire && D:\Python314\pythonw.exe daily_publish.py >> D:\EmpireLogs\printables_publish.log 2>&1""", 0, False
