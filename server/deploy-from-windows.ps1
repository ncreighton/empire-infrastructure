# Deploy Empire Services to Contabo VPS
# Usage: powershell -ExecutionPolicy Bypass -File server\deploy-from-windows.ps1 -VpsIp "YOUR_VPS_IP"

param(
    [Parameter(Mandatory=$true)]
    [string]$VpsIp,
    [string]$User = "empire"
)

$projectRoot = "D:\Claude Code Projects"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Deploying Empire to $VpsIp" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Upload docker-compose and configs
Write-Host "[1/7] Uploading server configs..." -ForegroundColor Yellow
scp "$projectRoot\server\docker-compose.yml" "${User}@${VpsIp}:/opt/empire/"
scp "$projectRoot\server\nginx-empire.conf" "${User}@${VpsIp}:/opt/empire/"

# Upload dashboard
Write-Host "[2/7] Uploading dashboard..." -ForegroundColor Yellow
scp -r "$projectRoot\empire-dashboard\*" "${User}@${VpsIp}:/opt/empire/dashboard/"
scp "$projectRoot\server\dashboard\Dockerfile" "${User}@${VpsIp}:/opt/empire/dashboard/"

# Upload article-audit
Write-Host "[3/7] Uploading article-audit..." -ForegroundColor Yellow
scp -r "$projectRoot\article-audit-system\*" "${User}@${VpsIp}:/opt/empire/article-audit/"
scp "$projectRoot\server\article-audit\Dockerfile" "${User}@${VpsIp}:/opt/empire/article-audit/"

# Upload shared config
Write-Host "[4/7] Uploading shared config..." -ForegroundColor Yellow
scp -r "$projectRoot\config\*" "${User}@${VpsIp}:/opt/empire/config/"

# Upload skill library (FORGE/AMPLIFY intelligence)
Write-Host "[5/7] Uploading skill library..." -ForegroundColor Yellow
ssh "${User}@${VpsIp}" "mkdir -p /opt/empire/skill-library/configs"
scp "$projectRoot\empire-skill-library\skills\forge-amplify-intelligence\forge_amplify_engine.py" "${User}@${VpsIp}:/opt/empire/skill-library/"
scp "$projectRoot\empire-skill-library\skills\forge-amplify-intelligence\skill.json" "${User}@${VpsIp}:/opt/empire/skill-library/"
scp -r "$projectRoot\empire-skill-library\skills\forge-amplify-intelligence\configs\*" "${User}@${VpsIp}:/opt/empire/skill-library/configs/"

# Upload n8n workflows (article-audit + empire-master)
Write-Host "[6/7] Uploading n8n workflows..." -ForegroundColor Yellow
ssh "${User}@${VpsIp}" "mkdir -p /opt/empire/n8n-workflows"
scp -r "$projectRoot\article-audit-system\n8n-workflows\*" "${User}@${VpsIp}:/opt/empire/n8n-workflows/"
scp -r "$projectRoot\empire-master\workflows\*" "${User}@${VpsIp}:/opt/empire/n8n-workflows/"
if (Test-Path "$projectRoot\n8n") {
    scp -r "$projectRoot\n8n\*" "${User}@${VpsIp}:/opt/empire/n8n-workflows/"
}

# Rebuild and restart containers
Write-Host "[7/7] Rebuilding containers on VPS..." -ForegroundColor Yellow
ssh "${User}@${VpsIp}" "cd /opt/empire && docker compose up -d --build"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Deploy Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Verify:" -ForegroundColor White
Write-Host "  curl http://${VpsIp}:8000/api/health/services" -ForegroundColor Gray
Write-Host "  curl http://${VpsIp}:8001/forge/health" -ForegroundColor Gray
Write-Host ""
