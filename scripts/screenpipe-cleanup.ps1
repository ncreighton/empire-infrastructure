# Screenpipe Data Cleanup Script
# Removes screen recording data older than 30 days
# Schedule: weekly via Task Scheduler or run manually
# Run: powershell -ExecutionPolicy Bypass -File scripts\screenpipe-cleanup.ps1

$dataDir = "$env:USERPROFILE\.screenpipe"
$retentionDays = 30
$logPath = "$env:LOCALAPPDATA\EmpireArchitect\screenpipe-cleanup.log"

# Ensure log directory exists
$logDir = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Starting screenpipe cleanup (retention: $retentionDays days)"

if (-not (Test-Path $dataDir)) {
    Add-Content -Path $logPath -Value "[$timestamp] Data directory not found: $dataDir"
    exit 0
}

$cutoff = (Get-Date).AddDays(-$retentionDays)
$totalFreed = 0
$filesRemoved = 0

# Clean old video chunks (.mp4 files)
$videoDir = Join-Path $dataDir "data"
if (Test-Path $videoDir) {
    Get-ChildItem -Path $videoDir -Recurse -File -Include "*.mp4" | Where-Object {
        $_.LastWriteTime -lt $cutoff
    } | ForEach-Object {
        $totalFreed += $_.Length
        $filesRemoved++
        Remove-Item $_.FullName -Force
    }
}

# Clean old screenshots/frames
Get-ChildItem -Path $dataDir -Recurse -File -Include "*.png","*.jpg","*.jpeg" | Where-Object {
    $_.LastWriteTime -lt $cutoff
} | ForEach-Object {
    $totalFreed += $_.Length
    $filesRemoved++
    Remove-Item $_.FullName -Force
}

# Clean empty directories left behind
Get-ChildItem -Path $dataDir -Recurse -Directory | Where-Object {
    (Get-ChildItem $_.FullName -Force | Measure-Object).Count -eq 0
} | Remove-Item -Force -ErrorAction SilentlyContinue

$freedMB = [math]::Round($totalFreed / 1MB, 1)
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Cleanup complete: removed $filesRemoved files, freed ${freedMB}MB"

# Report current data size
$currentSize = (Get-ChildItem -Path $dataDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
$currentMB = [math]::Round($currentSize / 1MB, 1)
Add-Content -Path $logPath -Value "[$timestamp] Current data size: ${currentMB}MB"
