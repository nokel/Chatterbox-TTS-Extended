# Universal installer for Chatterbox-TTS-Extended on Windows.
#
# 1. Detects the GPU vendor (override with -Hardware amd|nvidia|intel|cpu)
# 2. Probes the toolchain first with the SMALLEST builds of the three
#    hardware-sensitive runtimes (CPU torch ~200 MB, PyPI ctranslate2,
#    PyPI onnxruntime ~15 MB) and runs real computations on them, so a
#    broken Python/pip/network/VC-runtime fails in the first minutes
#    instead of after a 15 GB download.
# 3. Installs the hardware-specific stack:
#      amd    ROCm SDK + ROCm PyTorch (repo.radeon.com) + OpenNMT's ROCm
#             CTranslate2 wheel + onnxruntime-webgpu
#      nvidia CUDA PyTorch (download.pytorch.org/whl/cu128) + PyPI
#             CTranslate2 (its Windows wheel is CUDA-capable) +
#             onnxruntime-gpu
#      intel  CPU PyTorch + PyPI CTranslate2 + onnxruntime-openvino with
#             the paired openvino runtime (Intel GPU/NPU via OpenVINO)
#      cpu    keeps the probe builds; everything runs on the CPU
# 4. Installs the regular packages from requirements.txt, then verifies the
#    result per branch (verify.py).
#
# Usage:  right-click -> "Run with PowerShell", or from a terminal:
#         .\install.ps1 [-Hardware amd|nvidia|intel|cpu]
# Every step is checked; the script stops at the first failure instead of
# pretending everything worked. Re-runs skip everything already satisfied.

param(
    [ValidateSet("auto", "amd", "nvidia", "intel", "cpu")]
    [string]$Hardware = "auto"
)

$ErrorActionPreference = "Stop"
$RocmRel = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"

function Pause-IfInteractive {
    # Keep the window open for double-click/right-click launches so the last
    # messages stay readable (skipped for scripted runs with redirected input).
    if (-not [Console]::IsInputRedirected) {
        try { Read-Host "Press Enter to close" | Out-Null } catch {}
    }
}

function Fail($msg) {
    Write-Host ""
    Write-Host "INSTALL FAILED: $msg" -ForegroundColor Red
    Pause-IfInteractive
    exit 1
}

function Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

# --- 0. Detect hardware ---------------------------------------------------------
$gpuNames = ((Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
              Select-Object -ExpandProperty Name) -join "; ")
if ($Hardware -eq "auto") {
    # Priority: a discrete NVIDIA/AMD card usually sits next to an integrated
    # Intel/AMD GPU, so the discrete vendors win the tie.
    if     ($gpuNames -match "NVIDIA|GeForce|Quadro|RTX|Tesla") { $Hardware = "nvidia" }
    elseif ($gpuNames -match "AMD|Radeon")                      { $Hardware = "amd" }
    elseif ($gpuNames -match "Intel|Arc\b|Iris|UHD")            { $Hardware = "intel" }
    else                                                        { $Hardware = "cpu" }
}
Step "Graphics hardware: $(if ($gpuNames) { $gpuNames } else { '(none reported)' })"
Write-Host "    Install branch: $Hardware  (override with -Hardware amd|nvidia|intel|cpu)"

# --- 1. Find (or install) Python 3.12 ----------------------------------------
# AMD's ROCm PyTorch wheels only support Python 3.12; the other branches use
# it too so every setup is identical.
Step "Looking for Python 3.12..."
cmd /c "py -3.12 --version >nul 2>&1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.12 not found; attempting automatic install via winget..." -ForegroundColor Yellow
    try {
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent
    } catch {
        Fail ("winget is not available. Install Python 3.12 manually from " +
              "https://www.python.org/downloads/ then re-run this script.")
    }
    cmd /c "py -3.12 --version >nul 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Fail ("Python 3.12 is still not visible to the 'py' launcher. " +
              "If winget just installed it, open a NEW terminal and re-run this script. " +
              "Otherwise install it manually from https://www.python.org/downloads/")
    }
}
cmd /c "py -3.12 --version"

# --- 2. Create virtual environment -------------------------------------------
# The venv keeps its historical name .venv-amd so existing run scripts and
# configs keep working regardless of branch.
$freshVenv = $false
if (-not (Test-Path "$PSScriptRoot\.venv-amd\Scripts\python.exe")) {
    Step "Creating virtual environment .venv-amd ..."
    py -3.12 -m venv "$PSScriptRoot\.venv-amd"
    if ($LASTEXITCODE -ne 0) { Fail "could not create the virtual environment" }
    $freshVenv = $true
}
$python = "$PSScriptRoot\.venv-amd\Scripts\python.exe"
if ($freshVenv) {
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed" }
}

# Version of a package already installed in .venv-amd, or $null if absent.
function Get-PyPkgVersion([string]$name) {
    $code = "import importlib.metadata as m`ntry:`n    print(m.version('$name'))`nexcept Exception:`n    pass"
    $out = & $python -c $code
    if ($out) { return "$out".Trim() }
    return $null
}

# --- 3. Toolchain probe with the smallest builds --------------------------------
# Smallest known-good builds of the three hardware-sensitive runtimes:
#   torch        CPU wheel from download.pytorch.org/whl/cpu  (~200 MB vs 3-4 GB GPU builds)
#   ctranslate2  PyPI wheel (CPU everywhere; CUDA-capable on NVIDIA)
#   onnxruntime  PyPI wheel (CPU-only, ~15 MB)
# They are replaced by the branch-specific builds afterwards; on the cpu and
# intel branches torch/ctranslate2 stay as-is (they ARE the right builds).
# Skipped when a torch is already installed (i.e. any previous install).
if ($null -eq (Get-PyPkgVersion "torch")) {
    Step "Probing the toolchain with minimal CPU builds (fail-fast phase)..."
    & $python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { Fail "probe: CPU torch install failed" }
    & $python -m pip install ctranslate2 onnxruntime
    if ($LASTEXITCODE -ne 0) { Fail "probe: ctranslate2/onnxruntime install failed" }
    & $python "$PSScriptRoot\verify.py" probe
    if ($LASTEXITCODE -ne 0) {
        Fail "the minimal builds do not work on this machine - fix the errors above before the large downloads"
    }
} else {
    Step "Toolchain probe skipped (torch already installed)."
}

# --- 4. Hardware-specific stack --------------------------------------------------
switch ($Hardware) {

    "amd" {
        # 4a. ROCm SDK runtime
        $RocmVer = "7.2.1"
        $rocmSatisfied = $true
        foreach ($pkg in @("rocm", "rocm-sdk-core", "rocm-sdk-devel", "rocm-sdk-libraries-custom")) {
            if ((Get-PyPkgVersion $pkg) -ne $RocmVer) { $rocmSatisfied = $false }
        }
        if ($rocmSatisfied) {
            Step "ROCm $RocmVer SDK already installed - skipping."
        } else {
            Step "Installing ROCm $RocmVer SDK (large download, be patient)..."
            & $python -m pip install --no-cache-dir `
                "$RocmRel/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
                "$RocmRel/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
                "$RocmRel/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" `
                "$RocmRel/rocm-7.2.1.tar.gz"
            if ($LASTEXITCODE -ne 0) { Fail "ROCm SDK install failed" }
        }

        # 4b. PyTorch built for ROCm (replaces the probe's CPU torch)
        $TorchVer = "2.9.1+rocm7.2.1"
        if ((Get-PyPkgVersion "torch") -eq $TorchVer -and (Get-PyPkgVersion "torchaudio") -eq $TorchVer) {
            Step "PyTorch $TorchVer already installed - skipping."
        } else {
            Step "Installing PyTorch 2.9.1 + ROCm 7.2.1 ..."
            & $python -m pip install --no-cache-dir `
                "$RocmRel/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" `
                "$RocmRel/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
            if ($LASTEXITCODE -ne 0) { Fail "PyTorch (ROCm) install failed" }
        }

        # 4c. CTranslate2 ROCm build (replaces the probe's PyPI build)
        $Ct2Version = "4.8.1"
        $ct2Stamp = "$PSScriptRoot\.venv-amd\.ctranslate2-rocm.stamp"
        if ((Get-PyPkgVersion "ctranslate2") -eq $Ct2Version -and
                (Test-Path $ct2Stamp) -and ((Get-Content $ct2Stamp) -eq $Ct2Version)) {
            Step "CTranslate2 $Ct2Version ROCm build already installed - skipping."
        } else {
            $ct2zip = "$env:TEMP\ctranslate2-rocm-windows-$Ct2Version.zip"
            $ct2dir = "$env:TEMP\ctranslate2-rocm-windows-$Ct2Version"
            Step "Installing CTranslate2 $Ct2Version ROCm build (GPU-accelerated faster-whisper)..."
            Invoke-WebRequest -Uri "https://github.com/OpenNMT/CTranslate2/releases/download/v$Ct2Version/rocm-python-wheels-Windows.zip" -OutFile $ct2zip
            Expand-Archive -Path $ct2zip -DestinationPath $ct2dir -Force
            $ct2wheel = Get-ChildItem -Path $ct2dir -Recurse -Filter "ctranslate2-$Ct2Version-cp312-cp312-win_amd64.whl" | Select-Object -First 1
            if ($null -eq $ct2wheel) { Fail "could not find the cp312 CTranslate2 ROCm wheel in the release archive" }
            & $python -m pip install --force-reinstall --no-deps "$($ct2wheel.FullName)"
            if ($LASTEXITCODE -ne 0) { Fail "CTranslate2 ROCm wheel install failed" }
            Remove-Item $ct2zip -Force -Confirm:$false
            Remove-Item $ct2dir -Recurse -Force -Confirm:$false
            Set-Content -Path $ct2Stamp -Value $Ct2Version -Encoding ascii
        }

        # 4d. ONNX Runtime: WebGPU build (DirectX 12; implements the
        # GroupQueryAttention op Chatterbox needs, unlike DirectML)
        if (Get-PyPkgVersion "onnxruntime-webgpu") {
            Step "onnxruntime-webgpu already installed - skipping."
        } else {
            Step "Installing onnxruntime-webgpu (replaces the probe's CPU build)..."
            & $python -m pip uninstall -y onnxruntime 2>$null | Out-Null
            & $python -m pip install onnxruntime-webgpu
            if ($LASTEXITCODE -ne 0) { Fail "onnxruntime-webgpu install failed" }
        }
    }

    "nvidia" {
        # 4a. PyTorch built for CUDA (replaces the probe's CPU torch)
        if ("$(Get-PyPkgVersion 'torch')" -like "*+cu*") {
            Step "CUDA PyTorch already installed - skipping."
        } else {
            Step "Installing PyTorch (CUDA 12.8 wheels)..."
            & $python -m pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128
            if ($LASTEXITCODE -ne 0) { Fail "PyTorch (CUDA) install failed" }
        }

        # 4b. CTranslate2: the PyPI wheel already includes CUDA support on
        # Windows (needs cuDNN 9 at runtime; verify.py reports if missing).
        if ($null -eq (Get-PyPkgVersion "ctranslate2")) {
            & $python -m pip install ctranslate2
            if ($LASTEXITCODE -ne 0) { Fail "ctranslate2 install failed" }
        }

        # 4c. ONNX Runtime: CUDA build
        if (Get-PyPkgVersion "onnxruntime-gpu") {
            Step "onnxruntime-gpu already installed - skipping."
        } else {
            Step "Installing onnxruntime-gpu (replaces the probe's CPU build)..."
            & $python -m pip uninstall -y onnxruntime 2>$null | Out-Null
            & $python -m pip install onnxruntime-gpu
            if ($LASTEXITCODE -ne 0) { Fail "onnxruntime-gpu install failed" }
        }
    }

    "intel" {
        # 4a. torch stays the CPU build (Intel GPU/NPU acceleration comes
        # through OpenVINO's ONNX engine, not through torch); make sure
        # torchaudio matches.
        if ($null -eq (Get-PyPkgVersion "torchaudio")) {
            Step "Installing torchaudio (CPU)..."
            & $python -m pip install torchaudio --index-url https://download.pytorch.org/whl/cpu
            if ($LASTEXITCODE -ne 0) { Fail "torchaudio install failed" }
        }

        # 4b. ONNX Runtime: OpenVINO build + paired OpenVINO runtime.
        # The openvino version MUST pair with the onnxruntime-openvino
        # release (1.24.x <-> 2025.4.*); a mismatch loads but silently
        # falls back to CPU (the app detects and reports it).
        if (Get-PyPkgVersion "onnxruntime-openvino") {
            Step "onnxruntime-openvino already installed - skipping."
        } else {
            Step "Installing onnxruntime-openvino + openvino 2025.4 (replaces the probe's CPU build)..."
            & $python -m pip uninstall -y onnxruntime 2>$null | Out-Null
            & $python -m pip install onnxruntime-openvino "openvino==2025.4.*"
            if ($LASTEXITCODE -ne 0) { Fail "onnxruntime-openvino install failed" }
        }
    }

    "cpu" {
        # The probe builds ARE the final builds; just add torchaudio.
        if ($null -eq (Get-PyPkgVersion "torchaudio")) {
            Step "Installing torchaudio (CPU)..."
            & $python -m pip install torchaudio --index-url https://download.pytorch.org/whl/cpu
            if ($LASTEXITCODE -ne 0) { Fail "torchaudio install failed" }
        }
        if ($null -eq (Get-PyPkgVersion "onnxruntime")) {
            & $python -m pip install onnxruntime
            if ($LASTEXITCODE -ne 0) { Fail "onnxruntime install failed" }
        }
    }
}

# --- 5. Install the rest of the app's dependencies -----------------------------
# Skipped entirely when requirements.txt is unchanged since the last
# successful install (the stamp stores the file's hash).
$reqFile = "$PSScriptRoot\requirements.txt"
$reqStamp = "$PSScriptRoot\.venv-amd\.requirements-amd.stamp"
$reqHash = (Get-FileHash $reqFile -Algorithm SHA256).Hash
if ((Test-Path $reqStamp) -and ((Get-Content $reqStamp) -eq $reqHash)) {
    Step "Application dependencies already installed (requirements.txt unchanged) - skipping."
} else {
    Step "Installing application dependencies..."
    & $python -m pip install -r $reqFile
    if ($LASTEXITCODE -ne 0) {
        Fail "application dependency install failed (see pip output above) - nothing after this point was installed"
    }
    Set-Content -Path $reqStamp -Value $reqHash -Encoding ascii
}

# --- 6. Kokoro instant-TTS engine (installed without deps so its declared
# onnxruntime dependency cannot clobber the branch's onnxruntime build) --------
if (Get-PyPkgVersion "kokoro-onnx") {
    Step "kokoro-onnx already installed - skipping."
} else {
    Step "Installing kokoro-onnx (instant TTS engine)..."
    & $python -m pip install --no-deps kokoro-onnx
    if ($LASTEXITCODE -ne 0) { Fail "kokoro-onnx install failed" }
}

# --- 7. ffmpeg (needed for mp3/flac export and openai-whisper) -----------------
function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

Step "Checking for ffmpeg..."
Refresh-Path
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($null -eq $ffmpeg) {
    Write-Host "ffmpeg not found; attempting install via winget..." -ForegroundColor Yellow
    try {
        winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements --silent
    } catch {}
    Refresh-Path
    $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($null -eq $ffmpeg) {
        Write-Host "WARNING: ffmpeg is not on PATH even after install + PATH refresh." -ForegroundColor Yellow
        Write-Host "mp3/flac export and the openai-whisper backend need it; wav output works without it." -ForegroundColor Yellow
    }
}
if ($null -ne $ffmpeg) {
    Write-Host "ffmpeg found: $($ffmpeg.Source)"
}

# --- 8. Verify everything actually works ---------------------------------------
Step "Verifying installation ($Hardware branch)..."
& $python "$PSScriptRoot\verify.py" $Hardware
if ($LASTEXITCODE -ne 0) { Fail "verification failed - see the checks above" }

Write-Host ""
Write-Host "Install complete ($Hardware). Start the app with:  run_amd.ps1" -ForegroundColor Green
Pause-IfInteractive
