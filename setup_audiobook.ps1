# setup_audiobook.ps1 — point SumatraPDF at this Chatterbox install so the
# Read Aloud button reads with the Chatterbox audiobook engine.
#
# Auto-detects this folder + its venv python + the patched SumatraPDF build,
# then writes an [Audiobook] section into SumatraPDF-settings.txt with
# UseChatterbox = true. Right-click -> Run with PowerShell, or:
#   .\setup_audiobook.ps1 [-SettingsPath <SumatraPDF-settings.txt>] [-Windows]
#
#   -Windows   set UseChatterbox = false (revert Read Aloud to Windows TTS)
param(
    [string]$SettingsPath = "",
    [switch]$Windows
)
$ErrorActionPreference = "Stop"

$chatterboxDir = $PSScriptRoot
$python = Join-Path $chatterboxDir ".venv-amd\Scripts\pythonw.exe"
if (-not (Test-Path $python)) { $python = "" }

# --- locate SumatraPDF-settings.txt -----------------------------------------
function Find-Settings {
    $c = @(
        (Join-Path (Split-Path $PSScriptRoot -Parent) "sumatrapdf\out\dbg64\SumatraPDF-settings.txt"),
        (Join-Path (Split-Path $PSScriptRoot -Parent) "sumatrapdf\out\rel64\SumatraPDF-settings.txt"),
        (Join-Path $env:LOCALAPPDATA "SumatraPDF\SumatraPDF-settings.txt"),
        (Join-Path $env:APPDATA "SumatraPDF\SumatraPDF-settings.txt")
    )
    foreach ($p in $c) { if (Test-Path $p) { return $p } }
    # portable build folder exists but no settings yet: create one next to the exe
    $exeDir = Join-Path (Split-Path $PSScriptRoot -Parent) "sumatrapdf\out\dbg64"
    if (Test-Path $exeDir) { return (Join-Path $exeDir "SumatraPDF-settings.txt") }
    return $null
}
if (-not $SettingsPath) { $SettingsPath = Find-Settings }
if (-not $SettingsPath) {
    Write-Host "Could not find SumatraPDF-settings.txt. Run SumatraPDF once, or pass -SettingsPath." -ForegroundColor Red
    if (-not [Console]::IsInputRedirected) { try { Read-Host "Press Enter" | Out-Null } catch {} }
    exit 1
}

$use = if ($Windows) { "false" } else { "true" }
# SumatraPDF settings store paths with single backslashes (verbatim)
$block = @"
Audiobook [
	UseChatterbox = $use
	ChatterboxDir = $chatterboxDir
	PythonExe = $python
	TtsServerPort = 7861
	LmStudioUrl = http://127.0.0.1:11434
	NarratorVoice =
]
"@

if (Test-Path $SettingsPath) {
    $text = Get-Content $SettingsPath -Raw
    Copy-Item $SettingsPath ($SettingsPath + ".bak") -Force
    # replace an existing Audiobook [ ... ] block, else append
    $pattern = "(?ms)^Audiobook \[.*?^\]\r?\n?"
    if ($text -match $pattern) {
        $text = [regex]::Replace($text, $pattern, $block + "`r`n")
    } else {
        $text = $text.TrimEnd() + "`r`n`r`n" + $block + "`r`n"
    }
    Set-Content -Path $SettingsPath -Value $text -Encoding UTF8
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path $SettingsPath) | Out-Null
    Set-Content -Path $SettingsPath -Value $block -Encoding UTF8
}

Write-Host "Configured SumatraPDF audiobook Read Aloud:" -ForegroundColor Green
Write-Host "  settings:    $SettingsPath"
Write-Host "  UseChatterbox = $use"
Write-Host "  ChatterboxDir = $chatterboxDir"
Write-Host "  PythonExe     = $(if ($python) { $python } else { '(default venv)' })"
Write-Host ""
Write-Host "Restart SumatraPDF, open a PDF, and click Read Aloud."
if (-not [Console]::IsInputRedirected) { try { Read-Host "Press Enter to close" | Out-Null } catch {} }
