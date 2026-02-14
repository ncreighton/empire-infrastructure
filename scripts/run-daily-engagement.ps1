# Daily Engagement Runner
# Runs Reddit + Quora engagement via GeeLark cloud phones
# Triggered daily at 10:00 AM via Task Scheduler

$logPath = "$env:LOCALAPPDATA\EmpireArchitect\daily-engagement.log"
$projectDir = "D:\Claude Code Projects\geelark-automation"
$pythonExe = "python"
$script = "run_daily.py"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting daily engagement..."

# Verify project directory exists
if (-not (Test-Path $projectDir)) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: Project dir not found at $projectDir"
    exit 1
}

# Verify script exists
if (-not (Test-Path "$projectDir\$script")) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $script not found in $projectDir"
    exit 1
}

try {
    Set-Location $projectDir
    & $pythonExe $script --reddit --quora --extended --inbox 2>&1 |
        ForEach-Object {
            $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Add-Content -Path $logPath -Value "[$ts] $_"
        }

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] Daily engagement completed (exit code: $LASTEXITCODE)"
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] ERROR: $_"
}
