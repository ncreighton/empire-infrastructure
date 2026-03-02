$taskName = 'Empire-Evolution-Engine'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument '"D:\Claude Code Projects\launchers\launch-evolution-engine.vbs"'
$trigger = New-ScheduledTaskTrigger -AtLogOn
$trigger.Delay = 'PT30S'
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description 'EMPIRE-BRAIN Evolution Engine - Autonomous self-evolving intelligence'
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
