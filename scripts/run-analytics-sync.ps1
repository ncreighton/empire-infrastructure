# Daily Analytics Sync Runner
# Pulls GSC, GA4, Bing data and syncs to Supabase
# Triggered daily at 6:00 AM via Task Scheduler

$logPath = "$env:LOCALAPPDATA\EmpireArchitect\analytics-sync.log"
$projectDir = "D:\Claude Code Projects"
$pythonExe = "python"
$script = "daily_analytics_sync.py"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting analytics sync..."

# Verify script exists
if (-not (Test-Path "$projectDir\$script")) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $script not found in $projectDir"
    exit 1
}

try {
    Set-Location $projectDir
    & $pythonExe $script 2>&1 |
        ForEach-Object {
            $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Add-Content -Path $logPath -Value "[$ts] $_"
        }

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] Analytics sync completed (exit code: $LASTEXITCODE)"
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
