# EMPIRE-BRAIN MCP Server Startup Script
# Launched by Task Scheduler via VBS wrapper

$ErrorActionPreference = "SilentlyContinue"
$BrainDir = "D:\Claude Code Projects\EMPIRE-BRAIN"

# Kill any existing instance on port 8200
$existing = Get-NetTCPConnection -LocalPort 8200 -ErrorAction SilentlyContinue
if ($existing) {
    $existing | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

# Start MCP server
Set-Location $BrainDir
& python -m uvicorn api.brain_mcp:app --host 0.0.0.0 --port 8200 2>&1 | Tee-Object -FilePath "$BrainDir\logs\mcp-server.log"
