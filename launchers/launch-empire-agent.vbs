Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File ""C:\Claude Code Projects\empire-claude-agent.ps1"" -once", 0, False
