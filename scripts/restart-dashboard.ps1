# Restart empire dashboard
# Find and kill existing dashboard process, then start fresh

$found = $false
Get-Process -Name python -ErrorAction SilentlyContinue | ForEach-Object {
    $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue).CommandLine
    if ($cmdline -match "uvicorn.*8000|empire-dashboard") {
        Write-Host "Stopping dashboard (PID $($_.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $_.Id -Force
        $found = $true
    }
}

if (-not $found) {
    Write-Host "No running dashboard found" -ForegroundColor Gray
}

Start-Sleep -Seconds 2

Write-Host "Starting dashboard..." -ForegroundColor Cyan
Set-Location "D:\Claude Code Projects\empire-dashboard"
Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","main:app","--port","8000" -WindowStyle Hidden
Start-Sleep -Seconds 3

$health = Invoke-RestMethod -Uri "http://localhost:8000/health" -ErrorAction SilentlyContinue
if ($health.status -eq "healthy") {
    Write-Host "Dashboard started successfully" -ForegroundColor Green
} else {
    Write-Host "Dashboard may not have started properly" -ForegroundColor Red
}
