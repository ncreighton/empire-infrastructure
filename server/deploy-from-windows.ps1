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

# scp on Windows doesn't expand * globs — use PowerShell to resolve them
function Scp-Dir {
    param([string]$LocalDir, [string]$RemotePath)
    if (-not (Test-Path $LocalDir)) {
        Write-Host "  SKIP: $LocalDir not found" -ForegroundColor DarkGray
        return
    }
    $exclude = @('.git', '__pycache__', 'nul', '.claude', '*.pyc', 'node_modules')
    $items = Get-ChildItem -Path $LocalDir | Where-Object {
        $name = $_.Name
        -not ($exclude | Where-Object { $name -like $_ })
    }
    Write-Host "  $($items.Count) items from $LocalDir" -ForegroundColor DarkGray
    foreach ($item in $items) {
        if ($item.PSIsContainer) {
            Run-Cmd "scp -r `"$($item.FullName)`" `"${RemotePath}`""
        } else {
            Run-Cmd "scp `"$($item.FullName)`" `"${RemotePath}`""
        }
    }
}

$mode = if ($DryRun) { "DRY RUN" } else { "LIVE" }

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Deploying Empire to $VpsIp [$mode]" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Upload docker-compose and configs
Write-Host "[1/7] Uploading server configs..." -ForegroundColor Yellow
Run-Cmd "scp `"$projectRoot\server\docker-compose.yml`" `"${dest}:/opt/empire/`""
Run-Cmd "scp `"$projectRoot\server\nginx-empire.conf`" `"${dest}:/opt/empire/`""

# Upload dashboard
Write-Host "[2/7] Uploading dashboard..." -ForegroundColor Yellow
Scp-Dir "$projectRoot\empire-dashboard" "${dest}:/opt/empire/dashboard/"
Run-Cmd "scp `"$projectRoot\server\dashboard\Dockerfile`" `"${dest}:/opt/empire/dashboard/`""

# Upload article-audit
Write-Host "[3/7] Uploading article-audit..." -ForegroundColor Yellow
Scp-Dir "$projectRoot\article-audit-system" "${dest}:/opt/empire/article-audit/"
Run-Cmd "scp `"$projectRoot\server\article-audit\Dockerfile`" `"${dest}:/opt/empire/article-audit/`""

# Upload shared config
Write-Host "[4/7] Uploading shared config..." -ForegroundColor Yellow
Scp-Dir "$projectRoot\config" "${dest}:/opt/empire/config/"

# Upload skill library (FORGE/AMPLIFY intelligence)
Write-Host "[5/7] Uploading skill library..." -ForegroundColor Yellow
Run-Cmd "ssh `"$dest`" `"mkdir -p /opt/empire/skill-library/configs`""
Run-Cmd "scp `"$projectRoot\empire-skill-library\skills\forge-amplify-intelligence\forge_amplify_engine.py`" `"${dest}:/opt/empire/skill-library/`""
Run-Cmd "scp `"$projectRoot\empire-skill-library\skills\forge-amplify-intelligence\skill.json`" `"${dest}:/opt/empire/skill-library/`""
Scp-Dir "$projectRoot\empire-skill-library\skills\forge-amplify-intelligence\configs" "${dest}:/opt/empire/skill-library/configs/"

# Upload n8n workflows (article-audit + empire-master)
Write-Host "[6/7] Uploading n8n workflows..." -ForegroundColor Yellow
Run-Cmd "ssh `"$dest`" `"mkdir -p /opt/empire/n8n-workflows`""
Scp-Dir "$projectRoot\article-audit-system\n8n-workflows" "${dest}:/opt/empire/n8n-workflows/"
Scp-Dir "$projectRoot\empire-master\workflows" "${dest}:/opt/empire/n8n-workflows/"
if (Test-Path "$projectRoot\n8n") {
    Scp-Dir "$projectRoot\n8n" "${dest}:/opt/empire/n8n-workflows/"
}

# Rebuild and restart containers
Write-Host "[7/7] Rebuilding containers on VPS..." -ForegroundColor Yellow
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
Write-Host ""
