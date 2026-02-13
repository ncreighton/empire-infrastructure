Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File ""D:\Claude Code Projects\start-screenpipe.ps1""", 0, False
