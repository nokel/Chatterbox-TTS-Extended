# Universal installer for Chatterbox-TTS-Extended on Windows.

param(
    [ValidateSet("auto", "amd", "nvidia", "intel", "cpu")]
    [string]$Hardware = "auto"
)

$ErrorActionPreference = "Stop"
$RocmRel = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"


$LogFile = "$PSScriptRoot\install.log"
try {
    Start-Transcript -Path $LogFile -Force | Out-Null
} catch {
    $LogFile = $null   # transcription unavailable (e.g. blocked by policy); continue without it
}

function Pause-IfInteractive {
    if (-not [Console]::IsInputRedirected) {
        try { Read-Host "Press Enter to close" | Out-Null } catch {}
    }
}

function Fail($msg) {
    Write-Host ""
    Write-Host "INSTALL FAILED: $msg" -ForegroundColor Red
    if ($script:LogFile) {
        Write-Host "A full log of this run was saved to: $script:LogFile" -ForegroundColor Red
    }
    try { Stop-Transcript | Out-Null } catch {}
    Pause-IfInteractive
    exit 1
}

trap {
    Write-Host ""
    Write-Host "INSTALL FAILED (unexpected error): $_" -ForegroundColor Red
    if ($_.InvocationInfo -and $_.InvocationInfo.PositionMessage) {
        Write-Host $_.InvocationInfo.PositionMessage -ForegroundColor Red
    }
    if ($script:LogFile) {
        Write-Host "A full log of this run was saved to: $script:LogFile" -ForegroundColor Red
    }
    try { Stop-Transcript | Out-Null } catch {}
    Pause-IfInteractive
    exit 1
}

function Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Download-File([string]$Url, [string]$OutFile, [string]$What) {
    Write-Host "Downloading $What" -ForegroundColor Yellow
    Write-Host "    from $Url"
    [Net.ServicePointManager]::SecurityProtocol = `
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $req = [System.Net.HttpWebRequest]::Create($Url)
    $req.UserAgent = "Chatterbox-TTS-Extended-installer"
    $resp = $req.GetResponse()
    try {
        $total = $resp.ContentLength
        $in = $resp.GetResponseStream()
        $out = [System.IO.File]::Create($OutFile)
        try {
            $buf = New-Object byte[] (1MB)
            $done = [long]0
            $nextMilestone = 25
            $ui = [System.Diagnostics.Stopwatch]::StartNew()
            while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
                $out.Write($buf, 0, $n)
                $done += $n
                if ($ui.ElapsedMilliseconds -ge 200) {   # throttled; per-read redraws are what made IWR slow
                    if ($total -gt 0) {
                        Write-Progress -Activity "Downloading $What" `
                            -Status ("{0:n0} of {1:n0} MB" -f ($done/1MB), ($total/1MB)) `
                            -PercentComplete ([int](100 * $done / $total))
                    } else {
                        Write-Progress -Activity "Downloading $What" -Status ("{0:n0} MB" -f ($done/1MB))
                    }
                    $ui.Restart()
                }
                if ($total -gt 0 -and (100 * $done / $total) -ge $nextMilestone) {
                    Write-Host ("    {0,3}%  ({1:n0} of {2:n0} MB)" -f $nextMilestone, ($done/1MB), ($total/1MB))
                    $nextMilestone += 25
                }
            }
        } finally { $out.Dispose(); $in.Dispose() }
    } finally { $resp.Close() }
    Write-Progress -Activity "Downloading $What" -Completed
}

function Install-Python312 {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing Python 3.12 via winget..." -ForegroundColor Yellow
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "winget install failed (exit code $LASTEXITCODE); falling back to the python.org installer..." -ForegroundColor Yellow
    } else {
        Write-Host "winget is not available; downloading the installer from python.org instead..." -ForegroundColor Yellow
    }

    $url = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
    $exe = "$env:TEMP\python-3.12.10-amd64.exe"
    try {
        Download-File $url $exe "Python 3.12.10"
    } catch {
        Fail ("could not download the Python 3.12 installer from $url ($($_.Exception.Message)). " +
              "Install Python 3.12 manually from https://www.python.org/downloads/ then re-run this script.")
    }
    
    Write-Host "Running the Python 3.12.10 installer (silent, per-user)..." -ForegroundColor Yellow
    $proc = Start-Process -FilePath $exe `
        -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_launcher=1" `
        -Wait -PassThru
    Remove-Item $exe -Force -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        Fail ("the Python installer exited with code $($proc.ExitCode). " +
              "Install Python 3.12 manually from https://www.python.org/downloads/ then re-run this script.")
    }
}

function Install-FFmpegStatic {
    
    $url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    $zip = "$env:TEMP\ffmpeg-release-essentials.zip"
    $dest = "$PSScriptRoot\ffmpeg"
    try {
        Download-File $url $zip "ffmpeg (static build)"
    } catch {
        Write-Host "WARNING: could not download ffmpeg ($($_.Exception.Message))." -ForegroundColor Yellow
        return
    }
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Write-Host "Extracting to $dest ..." -ForegroundColor Yellow
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    Remove-Item $zip -Force
    $exe = Get-ChildItem -Path $dest -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    if ($null -eq $exe) {
        Write-Host "WARNING: ffmpeg.exe not found in the downloaded archive." -ForegroundColor Yellow
        return
    }
    $binDir = $exe.DirectoryName
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (($userPath -split ";") -notcontains $binDir) {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$binDir", "User")
    }
    $env:Path = "$env:Path;$binDir"
    Write-Host "ffmpeg installed to $binDir and added to the user PATH." -ForegroundColor Yellow
}


Get-ChildItem -Path $PSScriptRoot -Filter *.ps1 -Recurse -ErrorAction SilentlyContinue |
    Unblock-File -ErrorAction SilentlyContinue


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


Step "Looking for Python 3.12..."
cmd /c "py -3.12 --version >nul 2>&1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.12 not found; attempting automatic install..." -ForegroundColor Yellow
    Install-Python312
    Refresh-Path
    cmd /c "py -3.12 --version >nul 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Fail ("Python 3.12 was installed but is still not visible to the 'py' launcher. " +
              "Open a NEW terminal and re-run this script. " +
              "If that doesn't help, install it manually from https://www.python.org/downloads/")
    }
}
cmd /c "py -3.12 --version"


$freshVenv = $false
$python = "$PSScriptRoot\.venv-amd\Scripts\python.exe"
if (Test-Path $python) {
    & $python -c "pass" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Step "Existing .venv-amd is broken (its base Python is gone; e.g. the folder was copied from another machine) - deleting and recreating it..."
        Remove-Item "$PSScriptRoot\.venv-amd" -Recurse -Force
    }
}
if (-not (Test-Path $python)) {
    Step "Creating virtual environment .venv-amd ..."
    py -3.12 -m venv "$PSScriptRoot\.venv-amd"
    if ($LASTEXITCODE -ne 0) { Fail "could not create the virtual environment" }
    $freshVenv = $true
}
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
            Download-File "https://github.com/OpenNMT/CTranslate2/releases/download/v$Ct2Version/rocm-python-wheels-Windows.zip" $ct2zip "CTranslate2 $Ct2Version ROCm build"
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
        # 4a. PyTorch built for CUDA (replaces the probe's CPU torch).
        # The probe's CPU torch already satisfies "torch" as far as pip is
        # concerned, so it must be uninstalled first - otherwise pip reports
        # "Requirement already satisfied", leaves the CPU build in place and
        # only adds torchaudio, and GPU verification fails (seen in the field).
        if ("$(Get-PyPkgVersion 'torch')" -like "*+cu*") {
            Step "CUDA PyTorch already installed - skipping."
        } else {
            Step "Installing PyTorch (CUDA 12.8 wheels; replaces the probe's CPU build)..."
            & $python -m pip uninstall -y torch torchaudio 2>$null | Out-Null
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

# Skipped entirely when requirements.txt is unchanged since the last
# successful install (the stamp stores the file's hash).
$reqFile = "$PSScriptRoot\requirements.txt"
$reqStamp = "$PSScriptRoot\.venv-amd\.requirements-amd.stamp"
$reqHash = (Get-FileHash $reqFile -Algorithm SHA256).Hash
$reqRan = $false
if ((Test-Path $reqStamp) -and ((Get-Content $reqStamp) -eq $reqHash)) {
    Step "Application dependencies already installed (requirements.txt unchanged) - skipping."
} else {
    Step "Installing application dependencies..."
    & $python -m pip install -r $reqFile
    if ($LASTEXITCODE -ne 0) {
        Fail "application dependency install failed (see pip output above) - nothing after this point was installed"
    }
    Set-Content -Path $reqStamp -Value $reqHash -Encoding ascii
    $reqRan = $true
}


$branchOrt = @{ amd = "onnxruntime-webgpu"; nvidia = "onnxruntime-gpu"; intel = "onnxruntime-openvino" }[$Hardware]
if ($branchOrt) {
    if ($null -eq (Get-PyPkgVersion "onnxruntime")) {
        # Also repairs venvs where an earlier installer version uninstalled
        # the plain package (the source of the ERROR spam).
        Step "Installing the plain onnxruntime package pip expects (silero-vad dependency)..."
        & $python -m pip install onnxruntime
        if ($LASTEXITCODE -ne 0) { Fail "onnxruntime install failed" }
        $reqRan = $true   # its CPU files clobbered the branch build; reassert below
    }
    if ($reqRan) {
        Step "Re-asserting $branchOrt on top of the plain onnxruntime files..."
        & $python -m pip install --force-reinstall --no-deps $branchOrt
        if ($LASTEXITCODE -ne 0) { Fail "$branchOrt reinstall failed" }
    }
}


if (Get-PyPkgVersion "kokoro-onnx") {
    Step "kokoro-onnx already installed - skipping."
} else {
    Step "Installing kokoro-onnx (instant TTS engine)..."
    & $python -m pip install --no-deps kokoro-onnx
    if ($LASTEXITCODE -ne 0) { Fail "kokoro-onnx install failed" }
}

Step "Checking for ffmpeg..."
Refresh-Path
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($null -eq $ffmpeg) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "ffmpeg not found; attempting install via winget..." -ForegroundColor Yellow
        try {
            winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements --silent
        } catch {}
        Refresh-Path
        $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    }
    if ($null -eq $ffmpeg) {
        Write-Host "ffmpeg not found; downloading a static build (no winget needed)..." -ForegroundColor Yellow
        Install-FFmpegStatic
        $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    }
    if ($null -eq $ffmpeg) {
        Write-Host "WARNING: ffmpeg could not be installed automatically and is not on PATH." -ForegroundColor Yellow
        Write-Host "mp3/flac export and the openai-whisper backend need it; wav output works without it." -ForegroundColor Yellow
        Write-Host "You can install it later from https://www.gyan.dev/ffmpeg/builds/ (add its bin folder to PATH)." -ForegroundColor Yellow
    }
}
if ($null -ne $ffmpeg) {
    Write-Host "ffmpeg found: $($ffmpeg.Source)"
}

Step "Verifying installation ($Hardware branch)..."
& $python "$PSScriptRoot\verify.py" $Hardware
if ($LASTEXITCODE -ne 0) { Fail "verification failed - see the checks above" }

Write-Host ""
Write-Host "Install complete ($Hardware). Start the app with:  ./run.ps1 --auto" -ForegroundColor Green
Write-Host "The webclient will be accessible within a few seconds by typing in localhost:7860" -ForegroundColor Green
Write-Host "(--auto opens it in your browser for you)" -ForegroundColor Green
try { Stop-Transcript | Out-Null } catch {}

# Skipped for scripted runs with redirected input, same as Pause-IfInteractive.
if (-not [Console]::IsInputRedirected) {
    Write-Host ""
    do {
        try {
            $answer = (Read-Host "Run the app now? [y] yes / [n] no, close / [a] yes + open the browser automatically").Trim().ToLower()
        } catch { $answer = "n" }
        if ($answer -eq "") { $answer = "n" }   # plain Enter = close
    } until ($answer -match '^(y|n|a)$')
    if ($answer -eq "y") {
        & "$PSScriptRoot\run.ps1"
    } elseif ($answer -eq "a") {
        & "$PSScriptRoot\run.ps1" "--auto"
    }
    # "n": fall through and let the window close.
}
