# Screenpipe Startup Script
# Starts Screenpipe screen recording in the background

$screenpipeBin = "$env:USERPROFILE\screenpipe\bin\screenpipe.exe"
$logPath = "$env:LOCALAPPDATA\EmpireArchitect\screenpipe.log"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# Log startup
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting Screenpipe..."

# Verify binary exists
if (-not (Test-Path $screenpipeBin)) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: screenpipe.exe not found at $screenpipeBin"
    exit 1
}

# Ensure bun is on PATH (required for pipes)
$bunPath = "$env:USERPROFILE\.bun\bin"
if (Test-Path $bunPath) {
    $env:Path = "$bunPath;$env:Path"
}

# Start Screenpipe with pipe manager enabled
try {
    & $screenpipeBin --enable-pipe-manager
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
