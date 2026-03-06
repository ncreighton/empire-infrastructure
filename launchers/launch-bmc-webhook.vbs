' BMC Webhook Handler — Port 8095
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\bmc-witchcraft\automation && set PYTHONPATH=D:\Claude Code Projects\bmc-witchcraft\automation && D:\Python314\pythonw.exe -m uvicorn bmc_webhook_handler:app --host 127.0.0.1 --port 8095 >> D:\EmpireLogs\bmc_webhook.log 2>&1""", 0, False
