# Daily Printables publish — runs daily_publish.py
$ErrorActionPreference = "SilentlyContinue"

$projectDir = "D:\Claude Code Projects\printables-empire"
$python = "D:\Python314\python.exe"
$logFile = "$projectDir\data\daily_publish.log"

# Ensure data dir exists
New-Item -ItemType Directory -Path "$projectDir\data" -Force | Out-Null

# Add timestamp separator to log
Add-Content -Path $logFile -Value "`n--- $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ---"

# Run the daily publish script
Set-Location $projectDir
& $python "$projectDir\daily_publish.py" 2>&1 | Out-File -Append -FilePath $logFile -Encoding utf8
