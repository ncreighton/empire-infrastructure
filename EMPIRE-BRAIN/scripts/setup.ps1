# EMPIRE-BRAIN 3.0 — Setup Script
# Run: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

param(
    [switch]$Full,           # Full setup including Docker deploy
    [switch]$LocalOnly,      # Only local setup, no VPS
    [switch]$ScheduleTask,   # Create Windows scheduled tasks
    [switch]$InitDB          # Initialize database only
)

$ErrorActionPreference = "Stop"
$BrainRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$BrainDir = Join-Path $BrainRoot "EMPIRE-BRAIN"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  EMPIRE-BRAIN 3.0 — Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create local cache directory
$LocalCache = Join-Path $env:LOCALAPPDATA "EmpireBrain"
if (-not (Test-Path $LocalCache)) {
    New-Item -ItemType Directory -Path $LocalCache -Force | Out-Null
    Write-Host "[OK] Created local cache: $LocalCache" -ForegroundColor Green
} else {
    Write-Host "[OK] Local cache exists: $LocalCache" -ForegroundColor Green
}

# Step 2: Install Python dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
Push-Location $BrainDir
pip install -r requirements.txt 2>&1 | Out-Null
Pop-Location
Write-Host "[OK] Dependencies installed" -ForegroundColor Green

# Step 3: Create .env file if not exists
$EnvFile = Join-Path $BrainDir ".env"
if (-not (Test-Path $EnvFile)) {
    @"
# EMPIRE-BRAIN Environment Variables
# Fill in your actual credentials

# n8n Server (Contabo)
N8N_BASE_URL=https://vmi2976539.contaboserver.net
N8N_API_KEY=

# PostgreSQL (UpCloud)
BRAIN_PG_HOST=209.151.152.98
BRAIN_PG_DB=empire_architect
BRAIN_PG_USER=
BRAIN_PG_PASS=

# Qdrant (Contabo Docker)
QDRANT_HOST=vmi2976539.contaboserver.net
QDRANT_PORT=6333

# GitHub
GITHUB_PAT=

# Composio
COMPOSIO_API_KEY=
"@ | Set-Content $EnvFile -Encoding UTF8
    Write-Host "[OK] Created .env template — FILL IN CREDENTIALS" -ForegroundColor Yellow
} else {
    Write-Host "[OK] .env file exists" -ForegroundColor Green
}

# Step 4: Initialize local SQLite database
Write-Host ""
Write-Host "Initializing Brain database..." -ForegroundColor Yellow
python -c "import sys; sys.path.insert(0, r'$BrainDir'); from knowledge.brain_db import BrainDB; db = BrainDB(); print(f'DB initialized: {db.stats()}')"
Write-Host "[OK] Brain database ready" -ForegroundColor Green

# Step 5: Run initial scan
Write-Host ""
Write-Host "Running initial empire scan..." -ForegroundColor Yellow
python "$BrainDir\agents\scanner_agent.py" --once --no-webhook
Write-Host "[OK] Initial scan complete" -ForegroundColor Green

# Step 6: Create scheduled task for scanner
if ($ScheduleTask -or $Full) {
    Write-Host ""
    Write-Host "Creating scheduled tasks..." -ForegroundColor Yellow

    # Scanner: runs every 6 hours
    $ScannerAction = New-ScheduledTaskAction `
        -Execute "python" `
        -Argument "$BrainDir\agents\scanner_agent.py --once" `
        -WorkingDirectory $BrainDir

    $ScannerTrigger = New-ScheduledTaskTrigger -Once -At "06:00AM" `
        -RepetitionInterval (New-TimeSpan -Hours 6)

    Register-ScheduledTask `
        -TaskName "Empire Brain Scanner" `
        -Action $ScannerAction `
        -Trigger $ScannerTrigger `
        -Description "EMPIRE-BRAIN 3.0 — Scans empire every 6 hours" `
        -RunLevel Highest `
        -Force

    # Briefing: runs at 6 AM daily
    $BriefingAction = New-ScheduledTaskAction `
        -Execute "python" `
        -Argument "$BrainDir\agents\briefing_agent.py --webhook" `
        -WorkingDirectory $BrainDir

    $BriefingTrigger = New-ScheduledTaskTrigger -Daily -At "06:00AM"

    Register-ScheduledTask `
        -TaskName "Empire Brain Briefing" `
        -Action $BriefingAction `
        -Trigger $BriefingTrigger `
        -Description "EMPIRE-BRAIN 3.0 — Daily morning briefing" `
        -RunLevel Highest `
        -Force

    # MCP Server: starts at login
    $MCPAction = New-ScheduledTaskAction `
        -Execute "python" `
        -Argument "-m uvicorn api.brain_mcp:app --port 8200" `
        -WorkingDirectory $BrainDir

    $MCPTrigger = New-ScheduledTaskTrigger -AtLogOn
    $MCPSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd

    Register-ScheduledTask `
        -TaskName "Empire Brain MCP Server" `
        -Action $MCPAction `
        -Trigger $MCPTrigger `
        -Settings $MCPSettings `
        -Description "EMPIRE-BRAIN 3.0 — MCP Server on port 8200" `
        -RunLevel Highest `
        -Force

    Write-Host "[OK] Scheduled tasks created" -ForegroundColor Green
}

# Step 7: Docker deploy to VPS
if ($Full) {
    Write-Host ""
    Write-Host "Deploying Docker services to VPS..." -ForegroundColor Yellow
    scp "$BrainDir\docker-compose.yml" empire@vmi2976539.contaboserver.net:/opt/empire/brain/
    ssh empire@vmi2976539.contaboserver.net "cd /opt/empire/brain && docker compose up -d"
    Write-Host "[OK] Docker services deployed" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  EMPIRE-BRAIN 3.0 — Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Brain DB:     $BrainDir\knowledge\brain.db" -ForegroundColor White
Write-Host "  Local Cache:  $LocalCache" -ForegroundColor White
Write-Host "  MCP Server:   http://localhost:8200" -ForegroundColor White
Write-Host "  Logs:         $BrainDir\logs\brain.log" -ForegroundColor White
Write-Host ""
Write-Host "  Quick Start:" -ForegroundColor Yellow
Write-Host "    python agents\scanner_agent.py --once    # Full scan" -ForegroundColor White
Write-Host "    python agents\briefing_agent.py          # Daily briefing" -ForegroundColor White
Write-Host "    python -m uvicorn api.brain_mcp:app --port 8200  # MCP server" -ForegroundColor White
Write-Host ""
