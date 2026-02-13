# Vision Service Startup Script
# Starts the Vision AI FastAPI service on port 8002

$venvPython = "D:\Claude Code Projects\geelark-automation\.venv-vision\Scripts\python.exe"
$servicePath = "D:\Claude Code Projects\geelark-automation\services\vision_service.py"
$logPath = "$env:LOCALAPPDATA\EmpireArchitect\vision-service.log"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# Log startup
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting Vision Service..."

# Verify venv python exists
if (-not (Test-Path $venvPython)) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: Vision venv python not found at $venvPython"
    exit 1
}

# Start Vision Service
try {
    & $venvPython $servicePath
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
