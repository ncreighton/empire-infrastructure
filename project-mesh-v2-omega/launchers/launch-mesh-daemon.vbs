Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File ""D:\Claude Code Projects\project-mesh-v2-omega\scripts\start-mesh-daemon.ps1""", 0, False
