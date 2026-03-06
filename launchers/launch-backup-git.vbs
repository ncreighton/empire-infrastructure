' Backup Git Projects — PowerShell-only, must keep PS1
Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File ""C:\backup-system\Backup-GitProjects.ps1""", 0, False
