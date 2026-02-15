# ============================================================
# OpenClaw Empire — Start API Server
# Usage: .\start-empire.ps1
# ============================================================

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  OpenClaw Empire API Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Load .env file
$envFile = Join-Path $ProjectDir ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $val = $matches[2].Trim()
            if ($val -ne "") {
                [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
            }
        }
    }
    Write-Host "  [OK] Loaded .env" -ForegroundColor Green
} else {
    Write-Host "  [WARN] No .env file found" -ForegroundColor Yellow
}

# Check key vars
$apiKey = $env:ANTHROPIC_API_KEY
if ($apiKey -and $apiKey.StartsWith("sk-ant-")) {
    Write-Host "  [OK] ANTHROPIC_API_KEY set" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] ANTHROPIC_API_KEY not set!" -ForegroundColor Red
    exit 1
}

$nodeUrl = $env:OPENCLAW_NODE_URL
Write-Host "  [INFO] Phone gateway: $nodeUrl" -ForegroundColor Gray
Write-Host "  [INFO] Auth disabled: $($env:OPENCLAW_AUTH_DISABLED)" -ForegroundColor Gray
Write-Host ""

# Check if ADB is accessible
$adbPath = $env:ADB_PATH
if ($adbPath -and (Test-Path $adbPath)) {
    $adbVer = & $adbPath version 2>$null | Select-Object -First 1
    Write-Host "  [OK] ADB: $adbVer" -ForegroundColor Green
} else {
    Write-Host "  [WARN] ADB not found at $adbPath" -ForegroundColor Yellow
}

# Check local services
Write-Host ""
Write-Host "  Checking local services..." -ForegroundColor Gray

try {
    $null = Invoke-RestMethod -Uri "http://localhost:3030/health" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "  [OK] Screenpipe (3030)" -ForegroundColor Green
} catch {
    Write-Host "  [--] Screenpipe (3030) not running" -ForegroundColor DarkGray
}

try {
    $null = Invoke-RestMethod -Uri "http://localhost:8002/health" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "  [OK] Vision Service (8002)" -ForegroundColor Green
} catch {
    Write-Host "  [--] Vision Service (8002) not running" -ForegroundColor DarkGray
}

# Check phone gateway
try {
    $null = Invoke-RestMethod -Uri "$nodeUrl/health" -TimeoutSec 3 -ErrorAction Stop
    Write-Host "  [OK] Phone Gateway ($nodeUrl)" -ForegroundColor Green
} catch {
    Write-Host "  [--] Phone Gateway ($nodeUrl) not reachable yet" -ForegroundColor Yellow
    Write-Host "       Start gateway on phone: openclaw gateway --port 18789 --verbose" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "  Starting API server on port $($env:OPENCLAW_API_PORT)..." -ForegroundColor Cyan
Write-Host "  Dashboard: http://localhost:$($env:OPENCLAW_API_PORT)" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start uvicorn
Set-Location $ProjectDir
python -m uvicorn src.api:app --host 0.0.0.0 --port $env:OPENCLAW_API_PORT --reload
