Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File ""D:\Claude Code Projects\start-vision-service.ps1""", 0, False
