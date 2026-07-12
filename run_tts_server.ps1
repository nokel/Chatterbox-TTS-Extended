# Headless Chatterbox TTS API server (for the Discord bot and other clients).
# Right-click -> "Run with PowerShell", or: .\run_tts_server.ps1
$ErrorActionPreference = "Stop"
$python = "$PSScriptRoot\.venv-amd\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Environment not found. Run install.ps1 first." -ForegroundColor Red
} else {
    $env:PYTHONUTF8 = "1"
    & $python "$PSScriptRoot\tts_server.py"
}
# Keep the window open when launched by double-click/right-click so the last
# messages stay readable (skipped for scripted runs with redirected input).
if (-not [Console]::IsInputRedirected) {
    try { Read-Host "Press Enter to close" | Out-Null } catch {}
}
