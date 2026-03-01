# Setup ADB Connection Monitor — runs every 5 minutes to keep phone connected
# This is the foundation all other automations (Reddit, Substack) depend on.

$python = "D:\Python314\python.exe"
$script = "D:\Claude Code Projects\openclaw-empire\scripts\adb_monitor.py"
$workDir = "D:\Claude Code Projects\openclaw-empire"
$taskName = "AdbConnectionMonitor"

# Remove existing task if present
schtasks /delete /tn $taskName /f 2>$null

# Create task: every 5 minutes, 8 AM to 11 PM
schtasks /create `
    /tn $taskName `
    /tr "$python `"$script`"" `
    /sc daily `
    /st 07:00 `
    /ri 5 `
    /du 18:00 `
    /sd (Get-Date -Format "MM/dd/yyyy") `
    /rl HIGHEST `
    /f

# Set working directory and other options via XML modification
$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <CalendarTrigger>
      <Repetition>
        <Interval>PT5M</Interval>
        <Duration>PT18H</Duration>
      </Repetition>
      <StartBoundary>$(Get-Date -Format "yyyy-MM-dd")T07:00:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT5M</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions>
    <Exec>
      <Command>$python</Command>
      <Arguments>"$script"</Arguments>
      <WorkingDirectory>$workDir</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

$xmlPath = "$env:TEMP\adb_monitor_task.xml"
$xml | Out-File -FilePath $xmlPath -Encoding Unicode -Force

schtasks /delete /tn $taskName /f 2>$null
schtasks /create /tn $taskName /xml $xmlPath /f

Remove-Item $xmlPath -Force

# Verify
Write-Host ""
Write-Host "=== ADB Monitor Task ===" -ForegroundColor Cyan
schtasks /query /tn $taskName /fo LIST

Write-Host ""
Write-Host "Monitor will:" -ForegroundColor Green
Write-Host "  - Run every 5 minutes from 7 AM to 1 AM"
Write-Host "  - Check ADB connection to phone"
Write-Host "  - Auto-reconnect if dropped"
Write-Host "  - Scan for new ports if wireless debugging restarted"
Write-Host "  - Re-establish fixed port 5555 via adb tcpip"
Write-Host "  - Update .env so Reddit/Substack scripts use correct port"
