# Setup Analytics Sync Scheduled Task
# Registers "Empire-AnalyticsSync" to run at 6:00 AM daily
# Run: powershell -ExecutionPolicy Bypass -File scripts\setup-analytics-sync.ps1

$taskName = "Empire-AnalyticsSync"
$vbsPath = "D:\Claude Code Projects\launchers\launch-analytics-sync.vbs"
$description = "Syncs GSC, GA4, Bing analytics to Supabase (6:00 AM daily)"

Write-Host ""
Write-Host "Setting up: $taskName" -ForegroundColor Cyan

# Remove existing task if it exists
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# Verify VBS launcher exists
if (-not (Test-Path $vbsPath)) {
    Write-Host "  ERROR: VBS launcher not found at $vbsPath" -ForegroundColor Red
    exit 1
}

# Create the action
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument """$vbsPath"""

# Create trigger for daily at 6:00 AM
$trigger = New-ScheduledTaskTrigger -Daily -At "6:00AM"

# Create settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# Create principal (run as current user)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

# Register the task
$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description $description
Register-ScheduledTask -TaskName $taskName -InputObject $task | Out-Null

Write-Host "  Registered: $taskName" -ForegroundColor Green
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Analytics Sync Task Created!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Task: $taskName" -ForegroundColor White
Write-Host "  Schedule: Daily at 6:00 AM" -ForegroundColor White
Write-Host "  Script: daily_analytics_sync.py (GSC + GA4 + Bing -> Supabase)" -ForegroundColor White
Write-Host "  Log: $env:LOCALAPPDATA\EmpireArchitect\analytics-sync.log" -ForegroundColor Gray
Write-Host ""

# Show task status
Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue |
    Select-Object TaskName, State |
    Format-Table -AutoSize
