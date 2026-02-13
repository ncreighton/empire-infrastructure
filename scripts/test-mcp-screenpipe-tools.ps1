# Test Empire MCP Server Screenpipe Tools
# Simulates JSON-RPC calls to the MCP server

$serverPath = "D:\Claude Code Projects\empire-mcp-server\server_enhanced.py"

Write-Host "Testing Empire MCP Server Screenpipe Tools" -ForegroundColor Cyan
Write-Host "=" * 50

# Test 1: empire_monitor_check
Write-Host "`n1. empire_monitor_check" -ForegroundColor Yellow
$input1 = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
$input2 = '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"empire_monitor_check","arguments":{}}}'
$result1 = ($input1 + "`n" + $input2) | python $serverPath 2>$null | Select-Object -Last 1
$parsed1 = $result1 | ConvertFrom-Json -ErrorAction SilentlyContinue
if ($parsed1.result.content[0].text) {
    $data = $parsed1.result.content[0].text | ConvertFrom-Json
    Write-Host "   Status: $($data.status)" -ForegroundColor Green
} else {
    Write-Host "   Raw: $result1" -ForegroundColor Gray
}

# Test 2: empire_screen_search
Write-Host "`n2. empire_screen_search (query: 'Claude')" -ForegroundColor Yellow
$input3 = '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"empire_screen_search","arguments":{"query":"Claude","limit":3}}}'
$result2 = ($input1 + "`n" + $input3) | python $serverPath 2>$null | Select-Object -Last 1
$parsed2 = $result2 | ConvertFrom-Json -ErrorAction SilentlyContinue
if ($parsed2.result.content[0].text) {
    $data2 = $parsed2.result.content[0].text | ConvertFrom-Json
    Write-Host "   Results: $($data2.total)" -ForegroundColor Green
    if ($data2.results -and $data2.results.Count -gt 0) {
        Write-Host "   First: [$($data2.results[0].type)] $($data2.results[0].app_name) - $($data2.results[0].text.Substring(0, [Math]::Min(80, $data2.results[0].text.Length)))..." -ForegroundColor Gray
    }
} else {
    Write-Host "   Raw: $result2" -ForegroundColor Gray
}

# Test 3: empire_screen_timeline
Write-Host "`n3. empire_screen_timeline (limit: 5)" -ForegroundColor Yellow
$input4 = '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"empire_screen_timeline","arguments":{"limit":5}}}'
$result3 = ($input1 + "`n" + $input4) | python $serverPath 2>$null | Select-Object -Last 1
$parsed3 = $result3 | ConvertFrom-Json -ErrorAction SilentlyContinue
if ($parsed3.result.content[0].text) {
    $data3 = $parsed3.result.content[0].text | ConvertFrom-Json
    $count = if ($data3.timeline) { $data3.timeline.Count } else { 0 }
    Write-Host "   Timeline entries: $count" -ForegroundColor Green
    if ($data3.timeline -and $data3.timeline.Count -gt 0) {
        foreach ($entry in $data3.timeline[0..2]) {
            Write-Host "   $($entry.timestamp) | $($entry.app_name) | $($entry.window_name)" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "   Raw: $result3" -ForegroundColor Gray
}

Write-Host "`nAll MCP Screenpipe tools tested." -ForegroundColor Green
