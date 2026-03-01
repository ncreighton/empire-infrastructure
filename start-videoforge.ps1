# VideoForge Intelligence API Startup Script
# Starts the FastAPI server on port 8090

$videoforgeDir = "D:\Claude Code Projects\videoforge-engine"
$logPath = "$env:LOCALAPPDATA\EmpireArchitect\videoforge-api.log"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting VideoForge Intelligence API..."

if (-not (Test-Path $videoforgeDir)) {
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: VideoForge directory not found at $videoforgeDir"
    exit 1
}

Set-Location $videoforgeDir
$env:PYTHONPATH = $videoforgeDir

try {
    pythonw -m uvicorn api.app:app --host 127.0.0.1 --port 8090
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
