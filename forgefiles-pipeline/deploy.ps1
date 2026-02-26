# Deploy ForgeFiles Pipeline to Contabo VPS
# Usage: powershell -ExecutionPolicy Bypass -File deploy.ps1
# Dry run: powershell -ExecutionPolicy Bypass -File deploy.ps1 -DryRun

param(
    [string]$VpsIp = "217.216.84.245",
    [string]$User = "empire",
    [switch]$DryRun
)

$pipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dest = "${User}@${VpsIp}"
$remotePath = "/opt/empire/forgefiles-pipeline"

function Run-Cmd {
    param([string]$Cmd)
    if ($DryRun) {
        Write-Host "  [DRY] $Cmd" -ForegroundColor DarkCyan
    } else {
        Invoke-Expression $Cmd
    }
}

$gnuTar = "C:\Program Files\Git\usr\bin\tar.exe"

$mode = if ($DryRun) { "DRY RUN" } else { "LIVE" }

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  ForgeFiles Pipeline Deploy [$mode]" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Upload pipeline code
Write-Host "[1/5] Uploading pipeline code..." -ForegroundColor Yellow
$excludes = "--exclude=.git --exclude=__pycache__ --exclude=.claude --exclude=*.pyc --exclude=output --exclude=logs --exclude=.locks --exclude=models/*.stl --exclude=brand_assets/music"
$cmd = "& `"$gnuTar`" -cf - -C `"$pipelineRoot`" $excludes . | ssh `"$dest`" `"mkdir -p $remotePath && tar -xf - -C $remotePath`""
Write-Host "  tar+ssh: $pipelineRoot -> $remotePath" -ForegroundColor DarkGray
Run-Cmd $cmd

# Step 2: Upload brand assets (including music)
Write-Host "[2/5] Uploading brand assets..." -ForegroundColor Yellow
$cmd = "& `"$gnuTar`" -cf - -C `"$pipelineRoot\brand_assets`" . | ssh `"$dest`" `"mkdir -p $remotePath/brand_assets && tar -xf - -C $remotePath/brand_assets`""
Run-Cmd $cmd

# Step 3: Install Python dependencies on VPS
Write-Host "[3/5] Installing dependencies..." -ForegroundColor Yellow
$pipCmd = "cd $remotePath && pip3 install --user fastapi uvicorn pillow numpy 2>&1 | tail -1"
Run-Cmd "ssh `"$dest`" `"$pipCmd`""

# Step 4: Install Blender on VPS (if not present)
Write-Host "[4/5] Checking Blender on VPS..." -ForegroundColor Yellow
$blenderCheck = "which blender 2>/dev/null || (echo 'Installing Blender...' && sudo apt-get update -qq && sudo apt-get install -y -qq blender 2>&1 | tail -3)"
Run-Cmd "ssh `"$dest`" `"$blenderCheck`""

# Step 5: Set up systemd service for the API
Write-Host "[5/5] Setting up systemd service..." -ForegroundColor Yellow
$serviceContent = @"
[Unit]
Description=ForgeFiles Pipeline API
After=network.target

[Service]
Type=simple
User=$User
WorkingDirectory=$remotePath
ExecStart=/usr/bin/python3 -m uvicorn api:app --host 0.0.0.0 --port 8090
Restart=always
RestartSec=10
Environment=PYTHONPATH=$remotePath/scripts

[Install]
WantedBy=multi-user.target
"@

$escapedService = $serviceContent -replace '"', '\"' -replace "`n", "\n"
Run-Cmd "ssh `"$dest`" `"echo '$escapedService' | sudo tee /etc/systemd/system/forgefiles-pipeline.service > /dev/null && sudo systemctl daemon-reload && sudo systemctl enable forgefiles-pipeline && sudo systemctl restart forgefiles-pipeline`""

# Step 6: Set up cron for batch processing
Write-Host "[BONUS] Setting up cron job..." -ForegroundColor Yellow
$cronCmd = "(crontab -l 2>/dev/null | grep -v forgefiles; echo '0 */6 * * * cd $remotePath && python3 scripts/catalog.py --process-next --fast 2>&1 >> /var/log/forgefiles-cron.log') | crontab -"
Run-Cmd "ssh `"$dest`" `"$cronCmd`""

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Deploy Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  API: http://${VpsIp}:8090/health"
Write-Host "  Docs: http://${VpsIp}:8090/docs"
Write-Host "  Cron: Every 6 hours (process-next)"
Write-Host ""
