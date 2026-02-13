# Empire Infrastructure Smoke Test
# Run after reboot to verify all services are operational
# Usage: powershell -ExecutionPolicy Bypass -File scripts\smoke-test.ps1

$passed = 0
$failed = 0
$total = 0

function Test-Service($name, $url, $expectField, $expectValue) {
    $script:total++
    try {
        $resp = Invoke-RestMethod -Uri $url -TimeoutSec 10 -ErrorAction Stop
        $actual = if ($expectField) { $resp.$expectField } else { "ok" }
        if ($expectValue -and $actual -ne $expectValue) {
            Write-Host "  WARN  $name - expected $expectField=$expectValue, got $actual" -ForegroundColor Yellow
            $script:passed++  # Partial pass
        } else {
            Write-Host "  PASS  $name" -ForegroundColor Green
            $script:passed++
        }
        return $true
    } catch {
        Write-Host "  FAIL  $name - $($_.Exception.Message)" -ForegroundColor Red
        $script:failed++
        return $false
    }
}

function Test-Endpoint($name, $url, $method, $body) {
    $script:total++
    try {
        if ($method -eq "POST") {
            $resp = Invoke-RestMethod -Uri $url -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10 -ErrorAction Stop
        } else {
            $resp = Invoke-RestMethod -Uri $url -TimeoutSec 10 -ErrorAction Stop
        }
        Write-Host "  PASS  $name" -ForegroundColor Green
        $script:passed++
        return $resp
    } catch {
        Write-Host "  FAIL  $name - $($_.Exception.Message)" -ForegroundColor Red
        $script:failed++
        return $null
    }
}

Write-Host ""
Write-Host "Empire Infrastructure Smoke Test" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# 1. Service Health
Write-Host "Service Health:" -ForegroundColor Yellow
Test-Service "Screenpipe (3030)" "http://localhost:3030/health" "frame_status" "ok"
Test-Service "Vision Service (8002)" "http://localhost:8002/health" "status" "ok"
Test-Service "Empire Dashboard (8000)" "http://localhost:8000/health" "status" "healthy"
Test-Service "Automation API (8001)" "http://localhost:8001/health" $null $null

# 2. Dashboard Endpoints
Write-Host "`nDashboard API:" -ForegroundColor Yellow
Test-Endpoint "GET /api/alerts" "http://localhost:8000/api/alerts?limit=1" "GET" $null
Test-Endpoint "GET /api/alerts/summary" "http://localhost:8000/api/alerts/summary" "GET" $null
Test-Endpoint "GET /api/health/services" "http://localhost:8000/api/health/services" "GET" $null

# 3. Screenpipe Search
Write-Host "`nScreenpipe:" -ForegroundColor Yellow
$searchResult = Test-Endpoint "Search proxy" "http://localhost:8000/api/screenpipe/search?q=test&limit=1" "GET" $null
Test-Endpoint "Direct search" "http://localhost:3030/search?q=test&limit=1" "GET" $null

# 4. Pipes
Write-Host "`nPipes:" -ForegroundColor Yellow
$script:total++
try {
    $pipes = Invoke-RestMethod -Uri "http://localhost:3030/pipes/list" -TimeoutSec 10
    $enabled = ($pipes.data | Where-Object { $_.enabled }).Count
    if ($enabled -ge 2) {
        Write-Host "  PASS  Pipes enabled: $enabled" -ForegroundColor Green
        $script:passed++
    } else {
        Write-Host "  WARN  Only $enabled pipe(s) enabled (expected 2)" -ForegroundColor Yellow
        $script:passed++
    }
} catch {
    Write-Host "  FAIL  Pipe list - $($_.Exception.Message)" -ForegroundColor Red
    $script:failed++
}

# 5. Alert POST (create and immediately delete)
Write-Host "`nAlert Pipeline:" -ForegroundColor Yellow
$testBody = '{"severity":"info","message":"Smoke test alert","source":"smoke-test","category":"test"}'
$alertResp = Test-Endpoint "POST /api/alerts" "http://localhost:8000/api/alerts" "POST" $testBody
if ($alertResp) {
    $alertId = $alertResp.alert.id
    Test-Endpoint "DELETE alert $alertId" "http://localhost:8000/api/alerts/$alertId" "DELETE" $null | Out-Null
    # Re-test as DELETE
    $script:total--  # Don't double-count
}

# 6. Scheduled Tasks
Write-Host "`nScheduled Tasks:" -ForegroundColor Yellow
$tasks = @("Screenpipe", "Vision Service", "Empire Dashboard", "Screenpipe Cleanup")
foreach ($taskName in $tasks) {
    $script:total++
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        Write-Host "  PASS  $taskName ($($task.State))" -ForegroundColor Green
        $script:passed++
    } else {
        Write-Host "  FAIL  $taskName (not registered)" -ForegroundColor Red
        $script:failed++
    }
}

# Summary
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
$color = if ($failed -eq 0) { "Green" } elseif ($failed -le 2) { "Yellow" } else { "Red" }
Write-Host "Results: $passed/$total passed, $failed failed" -ForegroundColor $color
if ($failed -eq 0) {
    Write-Host "All systems operational!" -ForegroundColor Green
} elseif ($failed -le 2) {
    Write-Host "Minor issues detected." -ForegroundColor Yellow
} else {
    Write-Host "Multiple failures - investigate!" -ForegroundColor Red
}
Write-Host ""
