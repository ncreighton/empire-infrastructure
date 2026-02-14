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
Write-Host "[1/5] Uploading server configs..." -ForegroundColor Yellow
scp "$projectRoot\server\docker-compose.yml" "${User}@${VpsIp}:/opt/empire/"
scp "$projectRoot\server\nginx-empire.conf" "${User}@${VpsIp}:/opt/empire/"

# Upload dashboard
Write-Host "[2/5] Uploading dashboard..." -ForegroundColor Yellow
scp -r "$projectRoot\empire-dashboard\*" "${User}@${VpsIp}:/opt/empire/dashboard/"
scp "$projectRoot\server\dashboard\Dockerfile" "${User}@${VpsIp}:/opt/empire/dashboard/"

# Upload article-audit
Write-Host "[3/5] Uploading article-audit..." -ForegroundColor Yellow
scp -r "$projectRoot\article-audit-system\*" "${User}@${VpsIp}:/opt/empire/article-audit/"
scp "$projectRoot\server\article-audit\Dockerfile" "${User}@${VpsIp}:/opt/empire/article-audit/"

# Upload shared config
Write-Host "[4/5] Uploading shared config..." -ForegroundColor Yellow
scp -r "$projectRoot\config\*" "${User}@${VpsIp}:/opt/empire/config/"

# Upload n8n workflows
Write-Host "[5/5] Uploading n8n workflows..." -ForegroundColor Yellow
ssh "${User}@${VpsIp}" "mkdir -p /opt/empire/n8n-workflows"
scp -r "$projectRoot\article-audit-system\n8n-workflows\*" "${User}@${VpsIp}:/opt/empire/n8n-workflows/"
if (Test-Path "$projectRoot\n8n") {
    scp -r "$projectRoot\n8n\*" "${User}@${VpsIp}:/opt/empire/n8n-workflows/"
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Upload Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "SSH in and start:" -ForegroundColor White
Write-Host "  ssh ${User}@${VpsIp}" -ForegroundColor Gray
Write-Host "  cd /opt/empire && docker compose up -d" -ForegroundColor Gray
Write-Host ""
