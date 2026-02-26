$adb = "C:\Users\ncreighton\AppData\Local\Android\Sdk\platform-tools\adb.exe"
$envFile = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) ".env"
$dev = "100.79.124.62:34647"
if (Test-Path $envFile) {
    $match = Get-Content $envFile | Where-Object { $_ -match '^OPENCLAW_ADB_DEVICE=' }
    if ($match) { $dev = ($match -split '=',2)[1].Trim() }
}

# Check "Founder at Nick Creighton Digital Ventures"
& $adb -s $dev shell input tap 50 1596
Start-Sleep 1

# Check "Founder at AIinActionHub.com & AIDiscoveryDigest.com"
& $adb -s $dev shell input tap 50 1691
Start-Sleep 1

# Check "Independent Technology Builder at Self-employed"
& $adb -s $dev shell input tap 50 1786
Start-Sleep 2

# Screenshot
& $adb -s $dev shell screencap -p /sdcard/screen.png
& $adb -s $dev pull /sdcard/screen.png "D:\Claude Code Projects\openclaw-empire\screen.png"
Write-Host "Done checking boxes"
