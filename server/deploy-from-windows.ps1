# Deploy Empire Services to Contabo VPS
# Usage: powershell -ExecutionPolicy Bypass -File server\deploy-from-windows.ps1 -VpsIp "YOUR_VPS_IP"
# Dry run: powershell -ExecutionPolicy Bypass -File server\deploy-from-windows.ps1 -VpsIp "YOUR_VPS_IP" -DryRun

param(
    [Parameter(Mandatory=$true)]
    [string]$VpsIp,
    [string]$User = "empire",
    [switch]$DryRun
)

$projectRoot = "D:\Claude Code Projects"
$dest = "${User}@${VpsIp}"

# Run or print a command depending on DryRun mode
function Run-Cmd {
    param([string]$Cmd)
    if ($DryRun) {
        Write-Host "  $Cmd" -ForegroundColor DarkCyan
    } else {
        Invoke-Expression $Cmd
    }
}

# Pipe a tar stream over SSH — single connection per directory
# Use Git's GNU tar (Windows System32 bsdtar produces incompatible streams)
$gnuTar = "C:\Program Files\Git\usr\bin\tar.exe"
function Tar-Upload {
    param([string]$LocalDir, [string]$RemotePath)
    if (-not (Test-Path $LocalDir)) {
        Write-Host "  SKIP: $LocalDir not found" -ForegroundColor DarkGray
        return
    }
    $excludes = "--exclude=.git --exclude=__pycache__ --exclude=.claude --exclude=*.pyc --exclude=node_modules --exclude=.venv --exclude=nul"
    $cmd = "& `"$gnuTar`" -cf - -C `"$LocalDir`" $excludes . | ssh `"$dest`" `"mkdir -p $RemotePath && tar -xf - -C $RemotePath`""
    Write-Host "  tar+ssh: $LocalDir -> $RemotePath" -ForegroundColor DarkGray
    Run-Cmd $cmd
}

$mode = if ($DryRun) { "DRY RUN" } else { "LIVE" }

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Deploying Empire to $VpsIp [$mode]" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Upload docker-compose and configs
Write-Host "[1/9] Uploading server configs..." -ForegroundColor Yellow
Run-Cmd "scp `"$projectRoot\server\docker-compose.yml`" `"${dest}:/opt/empire/`""
Run-Cmd "scp `"$projectRoot\server\nginx-empire.conf`" `"${dest}:/opt/empire/`""
if (Test-Path "$projectRoot\server\.env") {
    Run-Cmd "scp `"$projectRoot\server\.env`" `"${dest}:/opt/empire/`""
}

# Upload dashboard
Write-Host "[2/9] Uploading dashboard..." -ForegroundColor Yellow
Tar-Upload "$projectRoot\empire-dashboard" "/opt/empire/dashboard"
Run-Cmd "scp `"$projectRoot\server\dashboard\Dockerfile`" `"${dest}:/opt/empire/dashboard/`""

# Upload article-audit
Write-Host "[3/9] Uploading article-audit..." -ForegroundColor Yellow
Tar-Upload "$projectRoot\article-audit-system" "/opt/empire/article-audit"
Run-Cmd "scp `"$projectRoot\server\article-audit\Dockerfile`" `"${dest}:/opt/empire/article-audit/`""

# Upload toolbox
Write-Host "[4/9] Uploading empire-toolbox..." -ForegroundColor Yellow
Tar-Upload "$projectRoot\empire-toolbox" "/opt/empire/toolbox"

# Upload openclaw
Write-Host "[5/9] Uploading openclaw..." -ForegroundColor Yellow
Tar-Upload "$projectRoot\openclaw-empire" "/opt/empire/openclaw"

# Upload shared config
Write-Host "[6/9] Uploading shared config..." -ForegroundColor Yellow
Tar-Upload "$projectRoot\config" "/opt/empire/config"

# Upload skill library (FORGE/AMPLIFY intelligence)
Write-Host "[7/9] Uploading skill library..." -ForegroundColor Yellow
Run-Cmd "ssh `"$dest`" `"mkdir -p /opt/empire/skill-library/configs`""
Run-Cmd "scp `"$projectRoot\empire-skill-library\skills\forge-amplify-intelligence\forge_amplify_engine.py`" `"${dest}:/opt/empire/skill-library/`""
Run-Cmd "scp `"$projectRoot\empire-skill-library\skills\forge-amplify-intelligence\skill.json`" `"${dest}:/opt/empire/skill-library/`""
Tar-Upload "$projectRoot\empire-skill-library\skills\forge-amplify-intelligence\configs" "/opt/empire/skill-library/configs"

# Upload n8n workflows (article-audit + empire-master)
Write-Host "[8/9] Uploading n8n workflows..." -ForegroundColor Yellow
Run-Cmd "ssh `"$dest`" `"mkdir -p /opt/empire/n8n-workflows`""
Tar-Upload "$projectRoot\article-audit-system\n8n-workflows" "/opt/empire/n8n-workflows"
Tar-Upload "$projectRoot\empire-master\workflows" "/opt/empire/n8n-workflows"
if (Test-Path "$projectRoot\n8n") {
    Tar-Upload "$projectRoot\n8n" "/opt/empire/n8n-workflows"
}

# Rebuild and restart containers
Write-Host "[9/9] Rebuilding containers on VPS..." -ForegroundColor Yellow
Run-Cmd "ssh `"$dest`" `"cd /opt/empire && docker compose up -d --build`""

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "  Dry Run Complete! (nothing uploaded)" -ForegroundColor Yellow
} else {
    Write-Host "  Deploy Complete!" -ForegroundColor Green
}
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Verify:" -ForegroundColor White
Write-Host "  curl http://${VpsIp}:8000/api/health/services" -ForegroundColor Gray
Write-Host "  curl http://${VpsIp}:8001/forge/health" -ForegroundColor Gray
Write-Host "  curl http://${VpsIp}:8300/health" -ForegroundColor Gray
Write-Host ""
