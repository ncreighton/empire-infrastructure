# ============================================================
# Test Phone Connection — Quick verification
# Usage: .\scripts\test-phone-connection.ps1 [phone-ip]
# ============================================================

param(
    [string]$PhoneIP = ""
)

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Load .env
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
}

if ($PhoneIP) {
    $NodeURL = "http://${PhoneIP}:18789"
} else {
    $NodeURL = $env:OPENCLAW_NODE_URL
    if (-not $NodeURL) { $NodeURL = "http://192.168.1.100:18789" }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Phone Connection Test" -ForegroundColor Cyan
Write-Host "  Target: $NodeURL" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$pass = 0
$fail = 0

# Test 1: Gateway reachable
Write-Host "  [1/5] Gateway reachable..." -NoNewline
try {
    $resp = Invoke-RestMethod -Uri "$NodeURL/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host " OK" -ForegroundColor Green
    $pass++
} catch {
    Write-Host " FAIL - Gateway not responding at $NodeURL" -ForegroundColor Red
    Write-Host "        Make sure 'openclaw gateway --port 18789' is running on the phone" -ForegroundColor DarkGray
    $fail++
}

# Test 2: ADB local
Write-Host "  [2/5] ADB (local USB)..." -NoNewline
$adbPath = $env:ADB_PATH
if ($adbPath -and (Test-Path $adbPath)) {
    $devices = & $adbPath devices 2>$null
    if ($devices -match "\tdevice") {
        Write-Host " OK - Device connected via USB" -ForegroundColor Green
        $pass++
    } else {
        Write-Host " SKIP - No USB device (using network mode)" -ForegroundColor Yellow
    }
} else {
    Write-Host " SKIP - ADB not found" -ForegroundColor Yellow
}

# Test 3: Node listing
Write-Host "  [3/5] Node listing..." -NoNewline
try {
    $nodes = Invoke-RestMethod -Uri "$NodeURL/api/nodes" -TimeoutSec 5 -ErrorAction Stop
    Write-Host " OK - $($nodes.Count) node(s)" -ForegroundColor Green
    $pass++
} catch {
    Write-Host " FAIL" -ForegroundColor Red
    $fail++
}

# Test 4: Device status
Write-Host "  [4/5] Device status..." -NoNewline
try {
    $body = @{node="android"; command="battery"} | ConvertTo-Json
    $status = Invoke-RestMethod -Uri "$NodeURL/api/nodes/invoke" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10 -ErrorAction Stop
    Write-Host " OK" -ForegroundColor Green
    $pass++
} catch {
    Write-Host " FAIL or SKIP (node may not be registered yet)" -ForegroundColor Yellow
}

# Test 5: Screenshot
Write-Host "  [5/5] Screenshot capability..." -NoNewline
try {
    $body = @{node="android"; command="screenshot"} | ConvertTo-Json
    $shot = Invoke-RestMethod -Uri "$NodeURL/api/nodes/invoke" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 15 -ErrorAction Stop
    Write-Host " OK" -ForegroundColor Green
    $pass++
} catch {
    Write-Host " FAIL or SKIP" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Results: $pass passed, $fail failed" -ForegroundColor $(if ($fail -eq 0) {"Green"} else {"Yellow"})
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($fail -gt 0) {
    Write-Host "  Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Is the phone on the same WiFi network?" -ForegroundColor Gray
    Write-Host "  2. What's the phone's IP? (Settings > About Phone > IP)" -ForegroundColor Gray
    Write-Host "  3. Is the gateway running? (tmux attach -t openclaw)" -ForegroundColor Gray
    Write-Host "  4. Update OPENCLAW_NODE_URL in .env with correct IP" -ForegroundColor Gray
    Write-Host ""
}
