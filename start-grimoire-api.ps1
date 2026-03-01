# Grimoire Intelligence API Startup Script
# Starts the FastAPI server on port 8080

$grimoireDir = "D:\Claude Code Projects\grimoire-intelligence"
$logPath = "$env:LOCALAPPDATA\EmpireArchitect\grimoire-api.log"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting Grimoire Intelligence API..."

if (-not (Test-Path $grimoireDir)) {
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: Grimoire directory not found at $grimoireDir"
    exit 1
}

Set-Location $grimoireDir
$env:PYTHONPATH = $grimoireDir

try {
    pythonw -m uvicorn api.app:app --host 127.0.0.1 --port 8080
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
