# EMPIRE-BRAIN Scanner — Hidden background execution
# Launched by Task Scheduler via VBS wrapper

$ErrorActionPreference = "SilentlyContinue"
$BrainDir = "D:\Claude Code Projects\EMPIRE-BRAIN"
$LogFile = "$BrainDir\logs\scanner.log"

# Ensure logs directory exists
New-Item -ItemType Directory -Path "$BrainDir\logs" -Force | Out-Null

# Run scanner (single pass)
Set-Location $BrainDir
& python agents/scanner_agent.py --once 2>&1 | Out-File -Append -FilePath $LogFile -Encoding utf8
