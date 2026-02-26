$adb = "C:\Users\ncreighton\AppData\Local\Android\Sdk\platform-tools\adb.exe"
$envFile = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) ".env"
$device = "100.79.124.62:34647"
if (Test-Path $envFile) {
    $match = Get-Content $envFile | Where-Object { $_ -match '^OPENCLAW_ADB_DEVICE=' }
    if ($match) { $device = ($match -split '=',2)[1].Trim() }
}
$outFile = "D:\Claude Code Projects\openclaw-empire\ui_fresh.xml"

Write-Host "Dumping UI..."
& $adb -s $device shell uiautomator dump /sdcard/ui_new.xml
Start-Sleep 2
Write-Host "Pulling file..."
& $adb -s $device pull /sdcard/ui_new.xml $outFile
Write-Host "Done. File exists: $(Test-Path $outFile)"
