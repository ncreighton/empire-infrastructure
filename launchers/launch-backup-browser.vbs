' Backup Browser Profiles — PowerShell-only, must keep PS1
Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File ""C:\backup-system\Backup-BrowserProfiles.ps1"" -WaitForClose -MaxWaitMinutes 2", 0, False
