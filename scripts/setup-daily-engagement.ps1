# Setup Daily Engagement Scheduled Task
# Registers "Empire-DailyEngagement" to run at 10:00 AM daily
# Run: powershell -ExecutionPolicy Bypass -File scripts\setup-daily-engagement.ps1

$taskName = "Empire-DailyEngagement"
$vbsPath = "D:\Claude Code Projects\launchers\launch-daily-engagement.vbs"
$description = "Runs daily Reddit + Quora engagement via GeeLark (10:00 AM)"

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

# Create trigger for daily at 10:00 AM
$trigger = New-ScheduledTaskTrigger -Daily -At "10:00AM"

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
Write-Host "  Daily Engagement Task Created!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Task: $taskName" -ForegroundColor White
Write-Host "  Schedule: Daily at 10:00 AM" -ForegroundColor White
Write-Host "  Command: python run_daily.py --reddit --quora --extended --inbox --skip-window" -ForegroundColor White
Write-Host "  Log: $env:LOCALAPPDATA\EmpireArchitect\daily-engagement.log" -ForegroundColor Gray
Write-Host ""

# Show task status
Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue |
    Select-Object TaskName, State |
    Format-Table -AutoSize
