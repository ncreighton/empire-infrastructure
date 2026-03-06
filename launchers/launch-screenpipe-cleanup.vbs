' Screenpipe Cleanup — PowerShell-only (no Python), must keep PS1
' Uses -NonInteractive to suppress any prompts
Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File ""D:\Claude Code Projects\scripts\screenpipe-cleanup.ps1""", 0, False
