# Builds the patched SumatraPDF fork (../sumatrapdf) that the audiobook
# reader drives for in-window highlighting.
# Right-click -> "Run with PowerShell", or: .\build_sumatra.ps1
$ErrorActionPreference = "Stop"
$repo = Join-Path (Split-Path $PSScriptRoot -Parent) "sumatrapdf"
$msbuild = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\msbuild.exe"
if (-not (Test-Path $repo)) {
    Write-Host "SumatraPDF fork not found at $repo" -ForegroundColor Red
} elseif (-not (Test-Path $msbuild)) {
    Write-Host "VS 2022 Build Tools not found. Install with:" -ForegroundColor Red
    Write-Host '  winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
} else {
    Push-Location $repo
    try {
        & $msbuild vs2022\SumatraPDF.sln /t:SumatraPDF-dll /p:Configuration=Debug /p:Platform=x64 /m /v:m /nologo
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`nBuilt: $repo\out\dbg64\SumatraPDF-dll.exe" -ForegroundColor Green
        } else {
            Write-Host "`nBuild failed (exit $LASTEXITCODE)" -ForegroundColor Red
        }
    } finally {
        Pop-Location
    }
}
if (-not [Console]::IsInputRedirected) {
    try { Read-Host "Press Enter to close" | Out-Null } catch {}
}
