' Screenpipe — OCR + audio recording
' Launches directly (no PowerShell = no window flash)
Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")
Dim bunPath, screenpipeBin
bunPath = objShell.ExpandEnvironmentStrings("%USERPROFILE%") & "\.bun\bin"
screenpipeBin = objShell.ExpandEnvironmentStrings("%USERPROFILE%") & "\screenpipe\bin\screenpipe.exe"
' Add bun to PATH and launch screenpipe
objShell.Run "cmd /c ""set PATH=" & bunPath & ";%PATH% && """ & screenpipeBin & """ --enable-pipe-manager --fps 1 --video-chunk-duration 120 --ignored-windows Bitwarden --ignored-windows 1Password --ignored-windows KeePass --ignored-windows Private --ignored-windows Incognito""", 0, False
