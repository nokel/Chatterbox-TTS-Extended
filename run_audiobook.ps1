# Audiobook reader. Start run_tts_server.ps1 first (and LM Studio's local
# server if you want to run character analysis).
# Right-click -> "Run with PowerShell", or: .\run_audiobook.ps1 [book.pdf]
param([string]$Pdf = "")
$ErrorActionPreference = "Stop"
$python = "$PSScriptRoot\.venv-amd\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Environment not found. Run install.ps1 first." -ForegroundColor Red
} else {
    $env:PYTHONUTF8 = "1"
    if ($Pdf) {
        & $python "$PSScriptRoot\audiobook\reader.py" $Pdf
    } else {
        & $python "$PSScriptRoot\audiobook\reader.py"
    }
}
if (-not [Console]::IsInputRedirected) {
    try { Read-Host "Press Enter to close" | Out-Null } catch {}
}
