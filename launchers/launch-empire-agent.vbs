' Empire Claude Agent — PowerShell-only agent, must keep PS1
' Uses -NonInteractive to suppress any prompts
Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File ""C:\Claude Code Projects\empire-claude-agent.ps1"" -once", 0, False
