# EMPIRE-BRAIN Evolution Engine Startup Script
# Launched by Task Scheduler via VBS wrapper

$ErrorActionPreference = "SilentlyContinue"
$BrainDir = "D:\Claude Code Projects\EMPIRE-BRAIN"
$LogFile = "$BrainDir\logs\evolution.log"

# Ensure log directory exists
New-Item -ItemType Directory -Force -Path "$BrainDir\logs" | Out-Null

# Kill any existing evolution agent
$existing = Get-Process python*, pythonw* -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*evolution_agent*"
}
if ($existing) {
    $existing | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Start evolution engine daemon
Set-Location $BrainDir
& pythonw agents/evolution_agent.py 2>&1 >> $LogFile
