# Setup BMC Webhook Handler Scheduled Task
# Run: powershell -ExecutionPolicy Bypass -File "D:\Claude Code Projects\bmc-witchcraft\setup-task.ps1"

$taskName = "BMC Webhook Handler"
$description = "Starts BMC Webhook Handler at login (port 8095)"
$vbsPath = "D:\Claude Code Projects\launchers\launch-bmc-webhook.vbs"

Write-Host ""
Write-Host "Setting up: $taskName" -ForegroundColor Cyan

# Remove existing task if present
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

# Create action
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument """$vbsPath"""

# Trigger at logon with 15s delay
$trigger = New-ScheduledTaskTrigger -AtLogOn
$trigger.Delay = "PT15S"

# Settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# Principal (run as current user)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

# Register
$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description $description
Register-ScheduledTask -TaskName $taskName -InputObject $task | Out-Null

Write-Host "  Registered: $taskName" -ForegroundColor Green
Write-Host ""

# Verify
Get-ScheduledTask -TaskName $taskName | Format-Table TaskName, State -AutoSize
