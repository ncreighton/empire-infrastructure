param(
    [Parameter(Mandatory=$true)]
    [string]$Text,
    [string]$Device = $( if (Test-Path "$PSScriptRoot\..\.env") { (Get-Content "$PSScriptRoot\..\.env" | Where-Object { $_ -match '^OPENCLAW_ADB_DEVICE=' } | ForEach-Object { ($_ -split '=',2)[1].Trim() }) } else { "100.79.124.62:34647" } ),
    [int]$WordDelayMs = 1500
)

$adb = "C:\Users\ncreighton\AppData\Local\Android\Sdk\platform-tools\adb.exe"

# Split text into individual words
$words = $Text -split '\s+'

for ($i = 0; $i -lt $words.Count; $i++) {
    $word = $words[$i]

    # Add space before word (except first word)
    if ($i -gt 0) {
        & $adb -s $Device shell input keyevent KEYCODE_SPACE 2>$null
        Start-Sleep -Milliseconds 300
    }

    # Type each character individually using keyevent
    foreach ($char in $word.ToCharArray()) {
        switch -Regex ($char) {
            '[a-z]' {
                $keycode = 29 + ([int][char]$char - [int][char]'a')
                & $adb -s $Device shell input keyevent $keycode 2>$null
            }
            '[A-Z]' {
                # Shift + letter
                $keycode = 29 + ([int][char]$char - [int][char]'A')
                & $adb -s $Device shell "input keyevent --longpress KEYCODE_SHIFT_LEFT $keycode" 2>$null
            }
            '[0-9]' {
                $keycode = 7 + ([int][char]$char - [int][char]'0')
                & $adb -s $Device shell input keyevent $keycode 2>$null
            }
            '\.' { & $adb -s $Device shell input keyevent 56 2>$null }
            ',' { & $adb -s $Device shell input keyevent 55 2>$null }
            '-' { & $adb -s $Device shell input keyevent 69 2>$null }
            '\(' { & $adb -s $Device shell input text "(" 2>$null }
            '\)' { & $adb -s $Device shell input text ")" 2>$null }
            '/' { & $adb -s $Device shell input keyevent 76 2>$null }
            ':' { & $adb -s $Device shell input text ":" 2>$null }
            ';' { & $adb -s $Device shell input keyevent 74 2>$null }
            "'" { & $adb -s $Device shell input text "'" 2>$null }
            '@' { & $adb -s $Device shell input keyevent 77 2>$null }
            '\+' { & $adb -s $Device shell input keyevent 81 2>$null }
            '&' { & $adb -s $Device shell input text "\&" 2>$null }
            default { & $adb -s $Device shell input text "$char" 2>$null }
        }
        Start-Sleep -Milliseconds 100
    }

    $pct = [math]::Round(($i + 1) / $words.Count * 100)
    $wordNum = $i + 1
    $total = $words.Count
    Write-Host "  [$pct%] $wordNum/$total words typed..."

    Start-Sleep -Milliseconds $WordDelayMs
}

Write-Host "`nDone!"
