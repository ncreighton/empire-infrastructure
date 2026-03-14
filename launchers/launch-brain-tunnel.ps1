# Brain MCP SSH Reverse Tunnel — forwards VPS:8200 → Windows:8200
# Run via Task Scheduler at logon (use run-hidden.vbs wrapper)

$logFile = "D:\Claude Code Projects\EMPIRE-BRAIN\logs\brain-tunnel.log"

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $msg" | Out-File -Append $logFile
}

Write-Log "Starting Brain MCP tunnel..."

while ($true) {
    # Check if tunnel is already running
    $existing = Get-Process ssh -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -match "8200:localhost:8200"
    }

    if (-not $existing) {
        Write-Log "Establishing SSH reverse tunnel (VPS:8200 -> localhost:8200)..."

        # Start tunnel (blocks until it dies)
        & ssh -R 8200:localhost:8200 empire@217.216.84.245 -N `
            -o ServerAliveInterval=60 `
            -o ServerAliveCountMax=3 `
            -o ExitOnForwardFailure=yes `
            -o StrictHostKeyChecking=no `
            -o ConnectTimeout=10 2>&1 | Out-File -Append $logFile

        Write-Log "Tunnel disconnected. Reconnecting in 10 seconds..."
    } else {
        Write-Log "Tunnel already running (PID: $($existing.Id))"
    }

    Start-Sleep -Seconds 10
}
