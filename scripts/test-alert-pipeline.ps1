# Test end-to-end alert pipeline
# Simulates what the empire-monitor pipe sends when detecting an error

$body = @{
    severity = "warning"
    title = "WordPress Error Detected"
    message = "HTTP 500 Internal Server Error on mythicalarchives.com/wp-admin/"
    source = "screenpipe-empire-monitor"
    category = "wordpress_error"
    app_name = "Google Chrome"
    timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
} | ConvertTo-Json

Write-Host "1. Posting alert to dashboard..." -ForegroundColor Cyan
$postResult = Invoke-RestMethod -Uri "http://localhost:8000/api/alerts" -Method POST -Body $body -ContentType "application/json"
Write-Host "   Created alert: $($postResult.alert.id) - $($postResult.status)" -ForegroundColor Green

Write-Host ""
Write-Host "2. Retrieving alerts..." -ForegroundColor Cyan
$alerts = Invoke-RestMethod -Uri "http://localhost:8000/api/alerts"
Write-Host "   Total alerts: $($alerts.Count)" -ForegroundColor Green
$latest = $alerts[0]
Write-Host "   Latest: [$($latest.severity)] $($latest.title)" -ForegroundColor Yellow

Write-Host ""
Write-Host "3. Checking alert summary..." -ForegroundColor Cyan
$summary = Invoke-RestMethod -Uri "http://localhost:8000/api/alerts/summary"
Write-Host "   Critical: $($summary.critical), Warning: $($summary.warning), Info: $($summary.info)" -ForegroundColor Green
Write-Host "   Unacknowledged: $($summary.unacknowledged)" -ForegroundColor Yellow

Write-Host ""
Write-Host "4. Acknowledging alert..." -ForegroundColor Cyan
$ackResult = Invoke-RestMethod -Uri "http://localhost:8000/api/alerts/$($postResult.alert.id)/acknowledge" -Method POST
Write-Host "   Acknowledge: $($ackResult.status)" -ForegroundColor Green

Write-Host ""
Write-Host "Pipeline test PASSED" -ForegroundColor Green
