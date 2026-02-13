# Setup Screenpipe + Vision Service Auto-Start Tasks
# Creates scheduled tasks to start both services at Windows login
# Run: powershell -ExecutionPolicy Bypass -File setup-services-autostart.ps1

$services = @(
    @{
        Name = "Screenpipe"
        Description = "Starts Screenpipe screen recording at login (port 3030)"
        VbsPath = "D:\Claude Code Projects\launchers\launch-screenpipe.vbs"
    },
    @{
        Name = "Vision Service"
        Description = "Starts Vision AI service at login (port 8002)"
        VbsPath = "D:\Claude Code Projects\launchers\launch-vision-service.vbs"
    }
)

foreach ($svc in $services) {
    Write-Host ""
    Write-Host "Setting up: $($svc.Name)" -ForegroundColor Cyan

    # Remove existing task if it exists
    $existing = Get-ScheduledTask -TaskName $svc.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  Removing existing task..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $svc.Name -Confirm:$false
    }

    # Verify VBS launcher exists
    if (-not (Test-Path $svc.VbsPath)) {
        Write-Host "  ERROR: VBS launcher not found at $($svc.VbsPath)" -ForegroundColor Red
        continue
    }

    # Create the action
    $action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument """$($svc.VbsPath)"""

    # Create trigger for user logon with 15-second delay
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $trigger.Delay = "PT15S"

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
    $task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description $svc.Description

    Register-ScheduledTask -TaskName $svc.Name -InputObject $task | Out-Null

    Write-Host "  Registered: $($svc.Name)" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Auto-Start Tasks Created!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services will start automatically at login:" -ForegroundColor White
Write-Host "  - Screenpipe     -> http://localhost:3030" -ForegroundColor White
Write-Host "  - Vision Service -> http://localhost:8002" -ForegroundColor White
Write-Host ""
Write-Host "Logs at: $env:LOCALAPPDATA\EmpireArchitect\" -ForegroundColor Gray
Write-Host ""

# Show task status
Get-ScheduledTask -TaskName "Screenpipe", "Vision Service" -ErrorAction SilentlyContinue |
    Select-Object TaskName, State |
    Format-Table -AutoSize
