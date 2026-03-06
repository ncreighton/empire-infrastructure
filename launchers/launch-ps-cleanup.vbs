' CleanupOrphanedPowerShell — runs PS cleanup script hidden
' PowerShell with -NonInteractive -NoProfile to minimize window flash
Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File ""C:\Users\ncreighton\Scripts\PowerShellCleanup\CleanupOrphanedPowerShell.ps1""", 0, False
