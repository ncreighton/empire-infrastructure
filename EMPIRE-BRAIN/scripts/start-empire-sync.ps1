# Empire Analytics Sync — Hidden background execution
# Launched by Task Scheduler via VBS wrapper

$ErrorActionPreference = "SilentlyContinue"
$EmpireDir = "D:\Claude Code Projects"
$LogFile = "$EmpireDir\EMPIRE-BRAIN\logs\empire-sync.log"

# Ensure logs directory exists
New-Item -ItemType Directory -Path "$EmpireDir\EMPIRE-BRAIN\logs" -Force | Out-Null

# Run sync
Set-Location $EmpireDir
& pythonw empire_sync.py
