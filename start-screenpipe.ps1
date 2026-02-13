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

# Start Screenpipe with pipe manager and optimized settings
# --fps 1: 1 frame/sec = ~30GB/month (default, lowest storage)
# --video-chunk-duration 120: 2-min chunks (fewer files)
# --ignored-windows: skip noisy/private apps from OCR
try {
    & $screenpipeBin `
        --enable-pipe-manager `
        --fps 1 `
        --video-chunk-duration 120 `
        --ignored-windows "Bitwarden" `
        --ignored-windows "1Password" `
        --ignored-windows "KeePass" `
        --ignored-windows "Private" `
        --ignored-windows "Incognito"
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
