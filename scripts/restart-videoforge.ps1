# Restart VideoForge Intelligence API
# Find and kill existing process, then start fresh

$found = $false
Get-Process -Name python -ErrorAction SilentlyContinue | ForEach-Object {
    $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue).CommandLine
    if ($cmdline -match "uvicorn.*8090|videoforge") {
        Write-Host "Stopping VideoForge API (PID $($_.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $_.Id -Force
        $found = $true
    }
}

if (-not $found) {
    Write-Host "No running VideoForge API found" -ForegroundColor Gray
}

Start-Sleep -Seconds 2

Write-Host "Starting VideoForge API..." -ForegroundColor Cyan
Set-Location "D:\Claude Code Projects\videoforge-engine"
$env:PYTHONPATH = "D:\Claude Code Projects\videoforge-engine"
Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","api.app:app","--host","127.0.0.1","--port","8090" -WindowStyle Hidden
Start-Sleep -Seconds 3

$health = Invoke-RestMethod -Uri "http://localhost:8090/health" -ErrorAction SilentlyContinue
if ($health.status -eq "ok") {
    Write-Host "VideoForge API started successfully" -ForegroundColor Green
} else {
    Write-Host "VideoForge API may not have started properly" -ForegroundColor Red
}
