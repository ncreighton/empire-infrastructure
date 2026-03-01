# Restart Grimoire Intelligence API
# Find and kill existing process, then start fresh

$found = $false
Get-Process -Name python -ErrorAction SilentlyContinue | ForEach-Object {
    $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue).CommandLine
    if ($cmdline -match "uvicorn.*8080|grimoire") {
        Write-Host "Stopping Grimoire API (PID $($_.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $_.Id -Force
        $found = $true
    }
}

if (-not $found) {
    Write-Host "No running Grimoire API found" -ForegroundColor Gray
}

Start-Sleep -Seconds 2

Write-Host "Starting Grimoire API..." -ForegroundColor Cyan
Set-Location "D:\Claude Code Projects\grimoire-intelligence"
$env:PYTHONPATH = "D:\Claude Code Projects\grimoire-intelligence"
Start-Process -FilePath "pythonw" -ArgumentList "-m","uvicorn","api.app:app","--host","127.0.0.1","--port","8080" -WindowStyle Hidden
Start-Sleep -Seconds 3

$health = Invoke-RestMethod -Uri "http://localhost:8080/health" -ErrorAction SilentlyContinue
if ($health.status -eq "ok") {
    Write-Host "Grimoire API started successfully" -ForegroundColor Green
} else {
    Write-Host "Grimoire API may not have started properly" -ForegroundColor Red
}
