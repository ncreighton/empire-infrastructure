# Mesh Dashboard v3.0 Startup Script
# Starts the Project Mesh web dashboard (FastAPI on port 8100)

$meshDir = "D:\Claude Code Projects\project-mesh-v2-omega"
$logPath = "$env:LOCALAPPDATA\EmpireArchitect\mesh-dashboard.log"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting Mesh Dashboard v3.0..."

if (-not (Test-Path $meshDir)) {
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: Mesh directory not found at $meshDir"
    exit 1
}

Set-Location $meshDir
$env:PYTHONPATH = $meshDir

try {
    pythonw -m uvicorn dashboard.api:app --host 0.0.0.0 --port 8100
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
