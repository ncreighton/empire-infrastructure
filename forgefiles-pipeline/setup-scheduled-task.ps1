# Set up Windows Scheduled Task for ForgeFiles batch processing
# Processes one pending STL every 6 hours
# Usage: powershell -ExecutionPolicy Bypass -File setup-scheduled-task.ps1

$pipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = "python"
$catalogScript = Join-Path $pipelineRoot "scripts\catalog.py"
$logFile = Join-Path $pipelineRoot "logs\scheduled_batch.log"

# Ensure logs directory exists
New-Item -ItemType Directory -Force -Path (Join-Path $pipelineRoot "logs") | Out-Null

$taskName = "ForgeFiles-BatchProcess"
$description = "Process next pending STL file in the ForgeFiles pipeline (every 6 hours)"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing task: $taskName" -ForegroundColor Yellow
}

# Create the action
$action = New-ScheduledTaskAction `
    -Execute $pythonPath `
    -Argument "`"$catalogScript`" --process-next --fast" `
    -WorkingDirectory $pipelineRoot

# Trigger: every 6 hours
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddHours(8) -RepetitionInterval (New-TimeSpan -Hours 6)

# Settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# Register
Register-ScheduledTask `
    -TaskName $taskName `
    -Description $description `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -RunLevel Highest

Write-Host ""
Write-Host "Scheduled task created: $taskName" -ForegroundColor Green
Write-Host "  Schedule: Every 6 hours starting at 8 AM"
Write-Host "  Action: python catalog.py --process-next --fast"
Write-Host "  Log: $logFile"
Write-Host ""
Write-Host "To run manually: schtasks /run /tn `"$taskName`"" -ForegroundColor Cyan
Write-Host "To remove: Unregister-ScheduledTask -TaskName `"$taskName`"" -ForegroundColor Cyan
