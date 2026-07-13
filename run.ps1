# Launches Chatterbox-TTS-Extended (any hardware branch).
# Right-click -> "Run with PowerShell", or from a terminal:
#   .\run.ps1 [--auto] [--host 0.0.0.0] [--port 7860] [--share]
# --auto opens the web client in your browser automatically; without it,
# browse to http://localhost:7860 once the server is up (a few seconds).
# All options are passed through to Chatter.py.
$ErrorActionPreference = "Stop"

# Everything the app prints is also transcribed to run.log next to the script,
# so crashes can be reported/diagnosed after the window is closed. Overwritten
# on each launch so the log always matches the latest run.
$LogFile = "$PSScriptRoot\run.log"
try {
    Start-Transcript -Path $LogFile -Force | Out-Null
} catch {
    $LogFile = $null   # transcription unavailable (e.g. blocked by policy); run without it
}

$python = "$PSScriptRoot\.venv-amd\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Environment not found. Run install.ps1 first." -ForegroundColor Red
} else {
    # Refresh PATH from the registry so tools installed after this shell was
    # opened (e.g. ffmpeg by install.ps1) are found without a new terminal.
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
    $env:PYTHONUTF8 = "1"
    & $python "$PSScriptRoot\Chatter.py" @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "The app exited with an error (code $LASTEXITCODE) - see the messages above." -ForegroundColor Red
        if ($LogFile) {
            Write-Host "A full log of this run was saved to: $LogFile" -ForegroundColor Red
        }
    }
}
try { Stop-Transcript | Out-Null } catch {}
# Keep the window open when launched by double-click/right-click so the last
# messages stay readable (skipped for scripted runs with redirected input).
if (-not [Console]::IsInputRedirected) {
    try { Read-Host "Press Enter to close" | Out-Null } catch {}
}
