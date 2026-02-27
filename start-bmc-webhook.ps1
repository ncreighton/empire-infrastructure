# BMC Webhook Handler Startup Script
# Starts the FastAPI server on port 8095

$bmcDir = "D:\Claude Code Projects\bmc-witchcraft\automation"
$logPath = "$env:LOCALAPPDATA\EmpireArchitect\bmc-webhook.log"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting BMC Webhook Handler..."

if (-not (Test-Path $bmcDir)) {
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: BMC automation directory not found at $bmcDir"
    exit 1
}

Set-Location $bmcDir
$env:PYTHONPATH = $bmcDir

try {
    python -m uvicorn bmc_webhook_handler:app --host 127.0.0.1 --port 8095
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
