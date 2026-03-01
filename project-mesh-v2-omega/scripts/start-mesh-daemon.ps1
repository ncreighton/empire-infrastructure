# Mesh Daemon v3.0 Startup Script
# Starts the Project Mesh daemon in background mode (9 loops)

$meshDir = "D:\Claude Code Projects\project-mesh-v2-omega"
$logPath = "$env:LOCALAPPDATA\EmpireArchitect\mesh-daemon.log"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting Mesh Daemon v3.0..."

if (-not (Test-Path $meshDir)) {
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: Mesh directory not found at $meshDir"
    exit 1
}

Set-Location $meshDir
$env:PYTHONPATH = $meshDir

try {
    pythonw "$meshDir\mesh_daemon.py" --background
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
