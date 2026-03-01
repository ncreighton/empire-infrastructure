' Universal Hidden Runner — Wraps any command to run invisibly
' Usage: wscript.exe run-hidden.vbs "full command line"
'
' Examples:
'   wscript.exe run-hidden.vbs "python script.py --arg"
'   wscript.exe run-hidden.vbs "powershell -File script.ps1"
'
Set objShell = CreateObject("WScript.Shell")
Dim cmd
cmd = ""
Dim i
For i = 0 To WScript.Arguments.Count - 1
    If cmd <> "" Then cmd = cmd & " "
    cmd = cmd & WScript.Arguments(i)
Next
If cmd <> "" Then
    objShell.Run cmd, 0, False
End If
