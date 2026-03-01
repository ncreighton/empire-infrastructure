# Setup Mesh Daemon Auto-Start Task
# Creates a scheduled task to start the Mesh Daemon at Windows login
# Run: powershell -ExecutionPolicy Bypass -File scripts\setup-daemon-autostart.ps1

$taskName = "Mesh Daemon v3.0"
$taskDescription = "Starts Project Mesh v3.0 daemon at login (9 sync loops, knowledge graph, service monitor)"
$vbsPath = "D:\Claude Code Projects\project-mesh-v2-omega\launchers\launch-mesh-daemon.vbs"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Mesh Daemon v3.0 - Auto-Start Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Verify VBS launcher exists
if (-not (Test-Path $vbsPath)) {
    Write-Host "ERROR: VBS launcher not found at $vbsPath" -ForegroundColor Red
    Write-Host "Expected file: launchers\launch-mesh-daemon.vbs" -ForegroundColor Yellow
    exit 1
}

# Verify mesh_daemon.py exists
$daemonPath = "D:\Claude Code Projects\project-mesh-v2-omega\mesh_daemon.py"
if (-not (Test-Path $daemonPath)) {
    Write-Host "ERROR: mesh_daemon.py not found at $daemonPath" -ForegroundColor Red
    exit 1
}

Write-Host "Setting up: $taskName" -ForegroundColor Cyan

# Remove existing task if it exists
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# Create the action (run VBS launcher via wscript for hidden window)
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument """$vbsPath"""

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
$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description $taskDescription

Register-ScheduledTask -TaskName $taskName -InputObject $task | Out-Null

Write-Host "  Registered: $taskName" -ForegroundColor Green

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Mesh Daemon Auto-Start Created!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The Mesh Daemon will start automatically at login:" -ForegroundColor White
Write-Host "  - Task Name:    $taskName" -ForegroundColor White
Write-Host "  - Trigger:      At Logon + 15s delay" -ForegroundColor White
Write-Host "  - Window:       Hidden (via VBS launcher)" -ForegroundColor White
Write-Host "  - Launcher:     $vbsPath" -ForegroundColor White
Write-Host ""
Write-Host "Daemon loops (9):" -ForegroundColor White
Write-Host "  - Sync (3s debounce)          - Compile (10s debounce)" -ForegroundColor Gray
Write-Host "  - Sentinel (5 min)            - Harvest (1 hour)" -ForegroundColor Gray
Write-Host "  - Health (30 min)             - Index (5 min)" -ForegroundColor Gray
Write-Host "  - Service Discovery (2 min)   - Drift Detection (15 min)" -ForegroundColor Gray
Write-Host "  - Heartbeat (60s)" -ForegroundColor Gray
Write-Host ""
Write-Host "Logs at: $env:LOCALAPPDATA\EmpireArchitect\mesh-daemon.log" -ForegroundColor Gray
Write-Host ""
Write-Host "Manual commands:" -ForegroundColor White
Write-Host "  python mesh_daemon.py --status   # Check if running" -ForegroundColor Gray
Write-Host "  python mesh_daemon.py --stop     # Stop daemon" -ForegroundColor Gray
Write-Host ""

# Show task status
Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue |
    Select-Object TaskName, State |
    Format-Table -AutoSize
