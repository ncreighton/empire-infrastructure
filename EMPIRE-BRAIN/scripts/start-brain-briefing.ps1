# EMPIRE-BRAIN Daily Briefing — Hidden background execution
# Launched by Task Scheduler via VBS wrapper

$ErrorActionPreference = "SilentlyContinue"
$BrainDir = "D:\Claude Code Projects\EMPIRE-BRAIN"
$LogFile = "$BrainDir\logs\briefing.log"

# Ensure logs directory exists
New-Item -ItemType Directory -Path "$BrainDir\logs" -Force | Out-Null

# Run briefing
Set-Location $BrainDir
& pythonw agents/briefing_agent.py
