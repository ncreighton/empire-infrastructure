$vbs = "D:\Claude Code Projects\launchers\run-hidden.vbs"
$logDir = "D:\Claude Code Projects\EMPIRE-BRAIN\logs"
$py = "D:\Python314\pythonw.exe"

$tasks = @(
    @{ Name = "AdbConnectionMonitor"; WorkDir = "D:\Claude Code Projects\openclaw-empire"; Script = "scripts\adb_monitor.py"; Args = ""; LogFile = "adb_monitor.log" },
    @{ Name = "DesignConsistency-DailyAudit"; WorkDir = "D:\Claude Code Projects\design-consistency-system"; Script = "scripts\daily_automation.py"; Args = ""; LogFile = "daily_automation.log" },
    @{ Name = "GeeLark Daily Evening"; WorkDir = "D:\Claude Code Projects\geelark-automation"; Script = "run_daily.py"; Args = "--core"; LogFile = "geelark_evening.log" },
    @{ Name = "GeeLark Daily Morning"; WorkDir = "D:\Claude Code Projects\geelark-automation"; Script = "run_daily.py"; Args = "--all --parasite --extended --publish --inbox"; LogFile = "geelark_morning.log" },
    @{ Name = "RedditEngagement"; WorkDir = "D:\Claude Code Projects\openclaw-empire"; Script = "scripts\reddit\reddit_scheduler.py"; Args = ""; LogFile = "reddit_engagement.log" },
    @{ Name = "RedditGeneratePlan"; WorkDir = "D:\Claude Code Projects\openclaw-empire"; Script = "scripts\reddit\reddit_scheduler.py"; Args = "--generate"; LogFile = "reddit_generate_plan.log" },
    @{ Name = "RedditMaintenance"; WorkDir = "D:\Claude Code Projects\openclaw-empire"; Script = "scripts\reddit\reddit_scheduler.py"; Args = "--maintenance"; LogFile = "reddit_maintenance.log" },
    @{ Name = "SubstackEngagement"; WorkDir = "D:\Claude Code Projects\openclaw-empire"; Script = "scripts\substack_engagement.py"; Args = "--auto"; LogFile = "substack_engagement.log" },
    @{ Name = "SubstackGenerateSchedule"; WorkDir = "D:\Claude Code Projects\openclaw-empire"; Script = "scripts\generate_schedule.py"; Args = ""; LogFile = "substack_generate_schedule.log" },
    @{ Name = "SubstackPostScheduler"; WorkDir = "D:\Claude Code Projects\openclaw-empire"; Script = "scripts\substack_scheduler.py"; Args = ""; LogFile = "substack_scheduler.log" }
)

foreach ($t in $tasks) {
    $logPath = Join-Path $logDir $t.LogFile
    $cmd = "cd /d $($t.WorkDir) && $py $($t.Script)"
    if ($t.Args) { $cmd += " $($t.Args)" }
    $cmd += " >> $logPath 2>&1"

    $argument = """$vbs"" ""cmd /c """"$cmd"""""""

    $action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument $argument
    Set-ScheduledTask -TaskName $t.Name -Action $action | Out-Null
    Write-Output "FIXED: $($t.Name) -> $($t.LogFile)"
}

# Verify
Write-Output ""
Write-Output "=== VERIFICATION ==="
foreach ($t in $tasks) {
    $task = Get-ScheduledTask -TaskName $t.Name
    $args = $task.Actions[0].Arguments
    if ($args -match "logs\\[a-z]") {
        Write-Output "OK: $($t.Name)"
    } else {
        Write-Output "STILL BROKEN: $($t.Name) -> $args"
    }
}
