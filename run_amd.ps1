# Launches Chatterbox-TTS-Extended.
# Right-click -> "Run with PowerShell", or: .\run_amd.ps1
$ErrorActionPreference = "Stop"
$python = "$PSScriptRoot\.venv-amd\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Environment not found. Run install.ps1 first." -ForegroundColor Red
} else {
    # Refresh PATH from the registry so tools installed after this shell was
    # opened (e.g. ffmpeg via winget) are found without a new terminal.
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
    $env:PYTHONUTF8 = "1"
    & $python "$PSScriptRoot\Chatter.py"
}
# Keep the window open when launched by double-click/right-click so the last
# messages stay readable (skipped for scripted runs with redirected input).
if (-not [Console]::IsInputRedirected) {
    try { Read-Host "Press Enter to close" | Out-Null } catch {}
}
