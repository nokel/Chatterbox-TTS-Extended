# Discord voice bot. Start run_tts_server.ps1 (and LM Studio's local server) first.
# Right-click -> "Run with PowerShell", or: .\run_bot.ps1
$ErrorActionPreference = "Stop"
$python = "$PSScriptRoot\..\.venv-amd\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Environment not found. Run install.ps1 in the parent folder first." -ForegroundColor Red
} else {
    $env:PYTHONUTF8 = "1"
    & $python "$PSScriptRoot\bot.py" --config "$PSScriptRoot\config.json"
}
if (-not [Console]::IsInputRedirected) {
    try { Read-Host "Press Enter to close" | Out-Null } catch {}
}
