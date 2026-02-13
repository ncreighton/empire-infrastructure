# Empire Dashboard Startup Script
# Starts the FastAPI dashboard on port 8000

$dashboardDir = "D:\Claude Code Projects\empire-dashboard"
$logPath = "$env:LOCALAPPDATA\EmpireArchitect\dashboard.log"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting Empire Dashboard..."

if (-not (Test-Path $dashboardDir)) {
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: Dashboard directory not found at $dashboardDir"
    exit 1
}

Set-Location $dashboardDir

try {
    python -m uvicorn main:app --port 8000
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
