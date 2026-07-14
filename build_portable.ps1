<#
Builds a self-contained, portable folder for running the STT server on a Windows
server without Docker or a pre-installed Python/ffmpeg. Downloads the official Python
embeddable distribution (a standalone runtime, not a venv -- no dependency on any
Python already present on the target machine), bootstraps pip into it, installs all
deps (including CUDA-enabled torch), downloads a static ffmpeg build, and copies in
the app code + model weights.

Usage (from repo root):
    .\build_portable.ps1
    .\build_portable.ps1 -OutDir C:\deploy\vietnamese-stt-server

Requires: internet access on the build machine (to fetch the embeddable Python, pip,
and ffmpeg), and model weights already downloaded into .\models (run
`python download_model.py` first if you haven't).
#>

param(
    [string]$OutDir = "dist\vietnamese-stt-server-portable",
    [string]$PythonVersion = "3.13.12",
    # BtbN/FFmpeg-Builds release asset, pinned to the n8.1 release build (static, GPL),
    # not the rolling nightly "master" build, so the ffmpeg version stays predictable.
    [string]$FfmpegAssetUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-8.1.zip"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

if (-not (Test-Path (Join-Path $RepoRoot "models\config.json"))) {
    Write-Error "models\config.json not found. Run 'python download_model.py' first."
}

if (Test-Path $OutDir) {
    Write-Host "Removing existing $OutDir ..."
    Remove-Item -Recurse -Force $OutDir
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

$PyDir = Join-Path $OutDir "python"
New-Item -ItemType Directory -Force -Path $PyDir | Out-Null

Write-Host "Downloading Python $PythonVersion embeddable distribution..."
$EmbedZipUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$EmbedZipPath = Join-Path $env:TEMP "python-$PythonVersion-embed-amd64.zip"
Invoke-WebRequest -Uri $EmbedZipUrl -OutFile $EmbedZipPath
Expand-Archive -Path $EmbedZipPath -DestinationPath $PyDir -Force
Remove-Item $EmbedZipPath

$PyExe = Join-Path $PyDir "python.exe"
$VerTag = ($PythonVersion.Split(".")[0..1] -join "")  # e.g. "313"

# Embeddable distributions ship with site-packages imports disabled and no pip.
# Uncomment "import site" in the ._pth file so installed packages are importable.
$PthFile = Join-Path $PyDir "python$VerTag._pth"
(Get-Content $PthFile) -replace '^#import site$', 'import site' | Set-Content $PthFile

Write-Host "Bootstrapping pip..."
$GetPipPath = Join-Path $env:TEMP "get-pip.py"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPipPath
& $PyExe $GetPipPath --no-warn-script-location
Remove-Item $GetPipPath

Write-Host "Installing torch (CUDA 12.8 build)..."
& $PyExe -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing remaining requirements..."
& $PyExe -m pip install --no-cache-dir -r (Join-Path $RepoRoot "requirements.txt")

Write-Host "Copying app code and model weights..."
Copy-Item (Join-Path $RepoRoot "main.py") $OutDir
Copy-Item (Join-Path $RepoRoot "download_model.py") $OutDir
Copy-Item -Recurse (Join-Path $RepoRoot "static") (Join-Path $OutDir "static")
Copy-Item -Recurse (Join-Path $RepoRoot "models") (Join-Path $OutDir "models")

Write-Host "Downloading static ffmpeg build..."
$BinDir = Join-Path $OutDir "bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$FfmpegZipPath = Join-Path $env:TEMP "ffmpeg-portable-build.zip"
$FfmpegExtractDir = Join-Path $env:TEMP "ffmpeg-portable-extract"
Invoke-WebRequest -Uri $FfmpegAssetUrl -OutFile $FfmpegZipPath
if (Test-Path $FfmpegExtractDir) { Remove-Item -Recurse -Force $FfmpegExtractDir }
Expand-Archive -Path $FfmpegZipPath -DestinationPath $FfmpegExtractDir -Force
$FfmpegExe = Get-ChildItem -Path $FfmpegExtractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
if (-not $FfmpegExe) {
    Write-Error "ffmpeg.exe not found inside downloaded archive from $FfmpegAssetUrl"
}
Copy-Item $FfmpegExe.FullName (Join-Path $BinDir "ffmpeg.exe")
Remove-Item $FfmpegZipPath
Remove-Item -Recurse -Force $FfmpegExtractDir

$ConfigBat = @'
@echo off
REM Edit these values, then restart run.bat to apply them.
set HOST=0.0.0.0
set PORT=8000

REM On a multi-GPU machine, pin to one GPU (e.g. "0" or "1") to avoid contention with
REM other processes already loaded onto a busier GPU. Leave blank to let CUDA pick.
set CUDA_VISIBLE_DEVICES=
'@
Set-Content -Path (Join-Path $OutDir "config.bat") -Value $ConfigBat -Encoding ASCII

$RunBat = @'
@echo off
setlocal
cd /d "%~dp0"
call "%~dp0config.bat"
title Vietnamese STT Server (port %PORT%)
set MODEL_DIR=%~dp0models
set FFMPEG_BIN=%~dp0bin\ffmpeg.exe
"%~dp0python\python.exe" -m uvicorn main:app --host %HOST% --port %PORT%
'@
Set-Content -Path (Join-Path $OutDir "run.bat") -Value $RunBat -Encoding ASCII

Write-Host ""
Write-Host "Done. Portable app built at: $OutDir"
Write-Host ""
Write-Host "Zip the entire '$OutDir' folder, copy it to the target server, unzip, and"
Write-Host "run run.bat. No Python, ffmpeg, or Docker install needed on the target machine."
