<#
Builds the same portable folder as build_portable.ps1, but assembled from what is
already on this machine instead of from the internet:

  * dependencies are copied out of the local venv's site-packages (only the closure
    of requirements.txt -- torch/transformers and other dev-only extras are skipped)
  * the model ships from the local models-ct2\ folder (no Hugging Face download,
    no re-conversion)
  * app code is copied from the repo working tree, not from an embedded copy
  * ffmpeg.exe is taken from PATH (or -FfmpegExe)

The one piece that cannot come from the working tree is the Python embeddable
runtime -- the venv has no standalone interpreter to ship. It is downloaded once
and cached in vendor\, so every later build is fully offline. Pass -Offline to fail
instead of downloading.

Use this for iterating on a deployment build; use build_portable.ps1 when you need a
single self-contained script to run on a clean machine with no repo checkout.

Usage:
    .\build_local.ps1
    .\build_local.ps1 -OutDir C:\deploy\stt -Zip
    .\build_local.ps1 -Offline
#>

param(
    [string]$OutDir = "dist\vietnamese-stt-server-local",
    [string]$VenvPython = "venv\Scripts\python.exe",
    [string]$ModelDir = "models-ct2",
    [string]$VendorDir = "vendor",
    # Defaults to the venv interpreter's own version: the copied wheels are built
    # against that ABI, so the shipped runtime has to be the same X.Y.
    [string]$PythonVersion = "",
    [string]$FfmpegExe = "",
    [switch]$Offline,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path $VenvPython)) {
    Write-Error "venv python not found at '$VenvPython'. Create the venv and 'pip install -r requirements.txt' first."
}
if (-not (Test-Path (Join-Path $ModelDir "model.bin"))) {
    Write-Error "No converted model at '$ModelDir\model.bin'. Run download_model.py then convert_ct2.py first (or use build_portable.ps1, which does both)."
}

if (-not $PythonVersion) {
    $PythonVersion = (& $VenvPython -c "import platform;print(platform.python_version())").Trim()
}
$VerTag = ($PythonVersion.Split(".")[0..1] -join "")   # e.g. "313"

# ---------------------------------------------------------------------------
# Output skeleton + app source straight from the working tree
# ---------------------------------------------------------------------------

if (Test-Path $OutDir) {
    Write-Host "Removing existing $OutDir ..."
    Remove-Item -Recurse -Force $OutDir
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

Write-Host "Copying app source from working tree..."
Copy-Item "main.py", "download_model.py", "requirements.txt" $OutDir
Copy-Item "static" (Join-Path $OutDir "static") -Recurse

# ---------------------------------------------------------------------------
# Python embeddable runtime (cached in vendor\ after the first build)
# ---------------------------------------------------------------------------

New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
$EmbedZip = Join-Path (Resolve-Path $VendorDir) "python-$PythonVersion-embed-amd64.zip"
if (-not (Test-Path $EmbedZip)) {
    if ($Offline) {
        Write-Error "Missing $EmbedZip and -Offline was given. Download python-$PythonVersion-embed-amd64.zip from python.org into $VendorDir\ and re-run."
    }
    Write-Host "Fetching Python $PythonVersion embeddable distribution (cached for later builds)..."
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" -OutFile $EmbedZip
} else {
    Write-Host "Using cached Python runtime: $EmbedZip"
}

$PyDir = Join-Path $OutDir "python"
Expand-Archive -Path $EmbedZip -DestinationPath $PyDir -Force
$PyExe = Join-Path $PyDir "python.exe"

# Embeddable distributions ship with site-packages imports disabled. Enable site and
# add Lib\site-packages explicitly, since nothing here runs pip to create it for us.
$PthFile = Join-Path $PyDir "python$VerTag._pth"
$Pth = (Get-Content $PthFile) -replace '^#import site$', 'import site'
if ($Pth -notcontains 'Lib\site-packages') { $Pth += 'Lib\site-packages' }
Set-Content -Path $PthFile -Value $Pth -Encoding ASCII

# ctranslate2/onnxruntime need the MSVC runtime; the embeddable zip ships vcruntime140.dll
# but not always its _1 companion, and the target machine may have no redist installed.
$Vc1 = Join-Path $env:SystemRoot "System32\vcruntime140_1.dll"
if ((-not (Test-Path (Join-Path $PyDir "vcruntime140_1.dll"))) -and (Test-Path $Vc1)) {
    Copy-Item $Vc1 $PyDir
}

# ---------------------------------------------------------------------------
# Dependencies out of the local venv
# ---------------------------------------------------------------------------

Write-Host "Copying dependency closure from $VenvPython ..."
& $VenvPython collect_deps.py requirements.txt (Join-Path $PyDir "Lib\site-packages")
if ($LASTEXITCODE -ne 0) { Write-Error "Dependency collection failed." }

# ---------------------------------------------------------------------------
# Model + ffmpeg
# ---------------------------------------------------------------------------

Write-Host "Copying model from $ModelDir ..."
Copy-Item $ModelDir (Join-Path $OutDir "models-ct2") -Recurse

if (-not $FfmpegExe) {
    $Found = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if (-not $Found) {
        Write-Error "ffmpeg.exe not found on PATH. Pass -FfmpegExe C:\path\to\ffmpeg.exe."
    }
    $FfmpegExe = $Found.Source
}
Write-Host "Bundling ffmpeg from $FfmpegExe ..."
$BinDir = Join-Path $OutDir "bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
Copy-Item $FfmpegExe (Join-Path $BinDir "ffmpeg.exe")

# ---------------------------------------------------------------------------
# Launcher (same contract as build_portable.ps1)
# ---------------------------------------------------------------------------

$ConfigIni = @'
; Edit these values, then restart run.bat to apply them.
HOST=0.0.0.0
PORT=8000

; Force "cuda" or "cpu", or leave as "auto" to use CUDA when available. GPU mode
; requires a compatible NVIDIA driver plus CUDA/cuDNN available on this machine --
; the shipped build itself has no CUDA runtime bundled.
DEVICE=auto

; On a multi-GPU machine, pin to one GPU (e.g. "0" or "1") to avoid contention with
; other processes already loaded onto a busier GPU. Leave blank to let CUDA pick.
CUDA_VISIBLE_DEVICES=
'@
Set-Content -Path (Join-Path $OutDir "config.ini") -Value $ConfigIni -Encoding ASCII

$RunBat = @'
@echo off
setlocal
cd /d "%~dp0"
for /f "usebackq eol=; tokens=1,2 delims==" %%A in ("config.ini") do (
    if not "%%A"=="" set "%%A=%%B"
)
title Vietnamese STT Server (port %PORT%)
set MODEL_DIR=%~dp0models-ct2
set FFMPEG_BIN=%~dp0bin\ffmpeg.exe
"%~dp0python\python.exe" -m uvicorn main:app --host %HOST% --port %PORT%
'@
Set-Content -Path (Join-Path $OutDir "run.bat") -Value $RunBat -Encoding ASCII

# ---------------------------------------------------------------------------
# Smoke test: the shipped interpreter must be able to import the whole stack
# ---------------------------------------------------------------------------

Write-Host "Verifying the bundled runtime..."
& $PyExe -c "import ctranslate2, faster_whisper, fastapi, uvicorn, soundfile, numpy; print('imports OK')"
if ($LASTEXITCODE -ne 0) { Write-Error "The bundled runtime failed to import the app's dependencies." }

$SizeMb = [math]::Round(((Get-ChildItem $OutDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host ""
Write-Host "Done. Local build at: $OutDir ($SizeMb MB)"

if ($Zip) {
    $ZipPath = "$OutDir.zip"
    if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
    Write-Host "Zipping to $ZipPath ..."
    Compress-Archive -Path (Join-Path $OutDir "*") -DestinationPath $ZipPath
    Write-Host "Zipped: $ZipPath"
}

Write-Host "Run 'run.bat' from that folder to start the server."
