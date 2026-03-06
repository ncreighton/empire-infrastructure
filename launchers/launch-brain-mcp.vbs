' EMPIRE-BRAIN MCP Server — Port 8200
' Launches pythonw directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c ""cd /d D:\Claude Code Projects\EMPIRE-BRAIN && set PYTHONPATH=D:\Claude Code Projects\EMPIRE-BRAIN && D:\Python314\pythonw.exe -m uvicorn api.brain_mcp:app --host 0.0.0.0 --port 8200 >> D:\EmpireLogs\brain_mcp.log 2>&1""", 0, False
