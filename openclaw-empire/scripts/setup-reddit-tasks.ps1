# Setup Windows Task Scheduler for Reddit ForgeFiles automation
#
# Creates three scheduled tasks:
#   1. RedditGeneratePlan — Runs daily at 6:45 AM to generate the day's sessions
#   2. RedditEngagement — Runs every 45 minutes from 8 AM to 11 PM to run sessions
#   3. RedditMaintenance — Runs daily at 2 AM for database cleanup
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\setup-reddit-tasks.ps1

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = "D:\Python314\python.exe"
$SchedulerScript = Join-Path $ProjectDir "scripts\reddit\reddit_scheduler.py"
$LogDir = Join-Path $ProjectDir "data\reddit"

# Ensure log directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Host "Setting up Reddit ForgeFiles automation tasks..." -ForegroundColor Cyan
Write-Host "Project: $ProjectDir"
Write-Host "Python: $Python"
Write-Host ""

# --- Task 1: Daily Plan Generation (6:45 AM) ---
$taskName1 = "RedditGeneratePlan"

Write-Host "Creating task: $taskName1" -ForegroundColor Yellow

$existing = Get-ScheduledTask -TaskName $taskName1 -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName1 -Confirm:$false
    Write-Host "  Removed existing task"
}

$action1 = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "`"$SchedulerScript`" --generate" `
    -WorkingDirectory $ProjectDir

$trigger1 = New-ScheduledTaskTrigger -Daily -At "6:45AM"

$settings1 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Register-ScheduledTask `
    -TaskName $taskName1 `
    -Action $action1 `
    -Trigger $trigger1 `
    -Settings $settings1 `
    -Description "Generate daily Reddit session plan for ForgeFiles (subreddits, session types, times)" `
    -RunLevel Limited | Out-Null

Write-Host "  Created: Daily at 6:45 AM" -ForegroundColor Green

# --- Task 2: Engagement (every 45 min, 8 AM - 11 PM) ---
$taskName2 = "RedditEngagement"

Write-Host "Creating task: $taskName2" -ForegroundColor Yellow

$existing2 = Get-ScheduledTask -TaskName $taskName2 -ErrorAction SilentlyContinue
if ($existing2) {
    Unregister-ScheduledTask -TaskName $taskName2 -Confirm:$false
    Write-Host "  Removed existing task"
}

$action2 = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "`"$SchedulerScript`"" `
    -WorkingDirectory $ProjectDir

# Trigger: every 45 minutes, starting at 8 AM, for 15 hours (until 11 PM)
$trigger2 = New-ScheduledTaskTrigger -Daily -At "8:00AM"
$trigger2.Repetition = (New-ScheduledTaskTrigger -Once -At "8:00AM" `
    -RepetitionInterval (New-TimeSpan -Minutes 45) `
    -RepetitionDuration (New-TimeSpan -Hours 15)).Repetition

$settings2 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 25) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $taskName2 `
    -Action $action2 `
    -Trigger $trigger2 `
    -Settings $settings2 `
    -Description "Reddit ForgeFiles engagement — runs due session from daily plan (browse/comment/post)" `
    -RunLevel Limited | Out-Null

Write-Host "  Created: Every 45 min, 8 AM - 11 PM" -ForegroundColor Green

# --- Task 3: Maintenance (daily at 2 AM) ---
$taskName3 = "RedditMaintenance"

Write-Host "Creating task: $taskName3" -ForegroundColor Yellow

$existing3 = Get-ScheduledTask -TaskName $taskName3 -ErrorAction SilentlyContinue
if ($existing3) {
    Unregister-ScheduledTask -TaskName $taskName3 -Confirm:$false
    Write-Host "  Removed existing task"
}

$action3 = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "`"$SchedulerScript`" --maintenance" `
    -WorkingDirectory $ProjectDir

$trigger3 = New-ScheduledTaskTrigger -Daily -At "2:00AM"

$settings3 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Register-ScheduledTask `
    -TaskName $taskName3 `
    -Action $action3 `
    -Trigger $trigger3 `
    -Settings $settings3 `
    -Description "Reddit state database cleanup — removes data older than 90 days" `
    -RunLevel Limited | Out-Null

Write-Host "  Created: Daily at 2:00 AM" -ForegroundColor Green

# --- Summary ---
Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Tasks created:" -ForegroundColor Cyan
Write-Host "  1. $taskName1 - Daily at 6:45 AM (generates session plan)"
Write-Host "  2. $taskName2 - Every 45 min, 8 AM-11 PM (runs due sessions)"
Write-Host "  3. $taskName3 - Daily at 2:00 AM (database maintenance)"
Write-Host ""
Write-Host "Manual commands:" -ForegroundColor Cyan
Write-Host "  Generate plan:     python scripts\reddit\reddit_scheduler.py --generate"
Write-Host "  Show status:       python scripts\reddit\reddit_scheduler.py --status"
Write-Host "  Dry run session:   python scripts\reddit\reddit_scheduler.py --dry-run"
Write-Host "  Run maintenance:   python scripts\reddit\reddit_scheduler.py --maintenance"
Write-Host ""
Write-Host "Staggered 45-min cycle (vs Substack's 30-min) to reduce phone contention." -ForegroundColor DarkGray
Write-Host "Phone lock file at data\reddit\.phone.lock prevents simultaneous use." -ForegroundColor DarkGray
Write-Host ""
Write-Host "Logs: $LogDir\scheduler.log" -ForegroundColor Cyan
