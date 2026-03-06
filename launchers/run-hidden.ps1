# Universal Hidden Python Runner
# Routes output to log files instead of console windows
# Usage: powershell -WindowStyle Hidden -File run-hidden.ps1 -Script "path.py" -ScriptArgs "args" -WorkDir "dir"

param(
    [Parameter(Mandatory=$true)]
    [string]$Script,
    [string]$ScriptArgs = "",
    [string]$WorkDir = "",
    [string]$LogDir = "D:\EmpireLogs"
)

$ErrorActionPreference = "SilentlyContinue"

# Ensure log directory exists
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Generate log file name from script name
$scriptName = [System.IO.Path]::GetFileNameWithoutExtension($Script)
$logFile = Join-Path $LogDir "$scriptName.log"

# Set working directory if specified
if ($WorkDir -and (Test-Path $WorkDir)) {
    Set-Location $WorkDir
}

# Find Python
$python = "D:\Python314\pythonw.exe"
if (-not (Test-Path $python)) {
    $python = "pythonw"
}

# Build and run command
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] START: $Script $ScriptArgs" | Out-File -Append -FilePath $logFile -Encoding utf8

if ($ScriptArgs) {
    & $python $Script $ScriptArgs.Split(" ") 2>&1 | Out-File -Append -FilePath $logFile -Encoding utf8
} else {
    & $python $Script 2>&1 | Out-File -Append -FilePath $logFile -Encoding utf8
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] END: exit code $LASTEXITCODE" | Out-File -Append -FilePath $logFile -Encoding utf8
