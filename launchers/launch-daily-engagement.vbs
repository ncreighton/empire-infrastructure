' Daily Engagement Runner (Reddit + Quora via GeeLark)
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\geelark-automation && D:\Python314\pythonw.exe run_daily.py --reddit --quora --extended --inbox >> D:\EmpireLogs\daily_engagement.log 2>&1""", 0, False
