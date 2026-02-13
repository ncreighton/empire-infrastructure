# Register weekly screenpipe cleanup task
$taskName = "Screenpipe Cleanup"
$vbsPath = "D:\Claude Code Projects\launchers\launch-screenpipe-cleanup.vbs"

# Remove existing task if it exists
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument """$vbsPath"""

# Run every Sunday at 3 AM
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "3:00AM"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "Weekly cleanup of Screenpipe data older than 30 days"

Register-ScheduledTask -TaskName $taskName -InputObject $task | Out-Null

Write-Host "Registered: $taskName (every Sunday at 3 AM)" -ForegroundColor Green
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State | Format-Table -AutoSize
