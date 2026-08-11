<#
Builds a self-contained, portable folder for running the STT server on a Windows
server without Docker or a pre-installed Python/ffmpeg. Downloads the official Python
embeddable distribution (a standalone runtime, not a venv -- no dependency on any
Python already present on the target machine), bootstraps pip into it, installs all
deps (including CUDA-enabled torch), downloads a static ffmpeg build and the model
weights, and writes out the app code (embedded in this script -- no repo checkout
needed).

This script is fully self-contained: copy just this one file to the target machine
and run it there, or run it on a dev machine and copy the resulting output folder.

Usage:
    .\build_portable.ps1
    .\build_portable.ps1 -OutDir C:\deploy\vietnamese-stt-server

Requires: internet access on the machine running this script (to fetch the embeddable
Python, pip, ffmpeg, and the model weights from Hugging Face).
#>

param(
    [string]$OutDir = "dist\vietnamese-stt-server-portable",
    [string]$PythonVersion = "3.13.12",
    # BtbN/FFmpeg-Builds release asset, pinned to the n8.1 release build (static, GPL),
    # not the rolling nightly "master" build, so the ffmpeg version stays predictable.
    [string]$FfmpegAssetUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-8.1.zip"
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Embedded app source (kept in sync with main.py / download_model.py /
# requirements.txt / static/index.html in this repo).
# ---------------------------------------------------------------------------

$MainPy = @'
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from download_model import REPO_ID

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
SAMPLE_RATE = 16000
STREAM_CHUNK_SECONDS = 3.0
SILENCE_RMS_THRESHOLD = 0.01
AUTO_STOP_SILENCE_SECONDS = float(os.environ.get("AUTO_STOP_SILENCE_SECONDS", "2.0"))

# Resolve ffmpeg: explicit override, then PATH, then a copy bundled alongside the app
# (used by the portable Windows package, which ships its own ffmpeg.exe).
FFMPEG_BIN = (
    os.environ.get("FFMPEG_BIN")
    or shutil.which("ffmpeg")
    or str(Path(__file__).parent / "bin" / "ffmpeg.exe")
)


def is_silent(audio: "np.ndarray") -> bool:
    return float(np.sqrt(np.mean(np.square(audio)))) < SILENCE_RMS_THRESHOLD

logger = logging.getLogger("uvicorn.error")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if device == "cpu":
        logger.warning(
            "torch.cuda.is_available() is False - running on CPU. "
            "If a GPU was expected, check the driver's CUDA ceiling (nvidia-smi) "
            "against the torch build installed (see docs/torch-cuda-version.md)."
        )

    model_state["processor"] = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device).eval()
    if device == "cpu":
        # fp16 has no real hardware acceleration on CPU in PyTorch and is often
        # slower than fp32 there (unlike on CUDA, where fp16 is a speedup).
        model = model.float()
    model_state["model"] = model
    yield
    model_state.clear()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


def load_audio(raw_bytes: bytes) -> "list[float]":
    """Decode arbitrary audio bytes to 16kHz mono PCM via ffmpeg."""
    src = tempfile.NamedTemporaryFile(suffix=".input", delete=False)
    dst_path = src.name + ".wav"
    try:
        src.write(raw_bytes)
        src.close()

        result = subprocess.run(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                src.name,
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-f",
                "wav",
                dst_path,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=400, detail="Could not decode audio file")

        audio, _ = sf.read(dst_path, dtype="float32")
        return audio
    finally:
        Path(src.name).unlink(missing_ok=True)
        Path(dst_path).unlink(missing_ok=True)


def transcribe_array(audio: "np.ndarray") -> str:
    processor = model_state["processor"]
    model = model_state["model"]

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=next(model.parameters()).dtype)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="vi", task="transcribe")

    text = processor.batch_decode(
        predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    raw_bytes = await file.read()
    audio = load_audio(raw_bytes)
    return {"text": transcribe_array(audio)}


@app.websocket("/ws/transcribe")
async def transcribe_stream(websocket: WebSocket):
    """Stream raw PCM16LE mono 16kHz audio; receive partial transcripts as it arrives.

    Send a text message "end" (or just close the socket) to flush the final chunk.
    """
    await websocket.accept()
    chunk_samples = int(STREAM_CHUNK_SECONDS * SAMPLE_RATE)
    buffer = np.empty(0, dtype=np.float32)
    speech_detected = False
    silence_seconds = 0.0

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

            if "bytes" in message and message["bytes"] is not None:
                pcm16 = np.frombuffer(message["bytes"], dtype=np.int16)
                chunk = pcm16.astype(np.float32) / 32768.0
                buffer = np.concatenate([buffer, chunk])

                if len(chunk) > 0:
                    if is_silent(chunk):
                        if speech_detected:
                            silence_seconds += len(chunk) / SAMPLE_RATE
                    else:
                        speech_detected = True
                        silence_seconds = 0.0

                if len(buffer) >= chunk_samples:
                    if is_silent(buffer):
                        text = ""
                    else:
                        text = transcribe_array(buffer)
                    buffer = np.empty(0, dtype=np.float32)
                    if text:
                        await websocket.send_json({"text": text, "final": False})

                if speech_detected and silence_seconds >= AUTO_STOP_SILENCE_SECONDS:
                    if len(buffer) > 0 and not is_silent(buffer):
                        text = transcribe_array(buffer)
                        if text:
                            await websocket.send_json({"text": text, "final": True})
                    await websocket.send_json({"event": "auto_stop"})
                    await websocket.close()
                    return

            elif message.get("text") == "end":
                if len(buffer) > 0 and not is_silent(buffer):
                    text = transcribe_array(buffer)
                    if text:
                        await websocket.send_json({"text": text, "final": True})
                await websocket.close()
                return
    except WebSocketDisconnect:
        return


@app.get("/health")
async def health():
    return {"status": "ok", "device": device, "model": REPO_ID}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
'@

$DownloadModelPy = @'
"""Fetch PhoWhisper-small's files (config, tokenizer, and weights) into models/."""
from huggingface_hub import snapshot_download

REPO_ID = "vinai/PhoWhisper-small"

if __name__ == "__main__":
    snapshot_download(
        repo_id=REPO_ID,
        local_dir="models",
        allow_patterns=["*.json", "*.txt", "vocab.*", "merges.txt", "*.model", "pytorch_model.bin"],
    )

    print("Done. Model files are now in models/")
'@

$RequirementsTxt = @'
fastapi
uvicorn[standard]
python-multipart
transformers
accelerate
huggingface_hub
soundfile
numpy
'@

$IndexHtml = @'
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<title>PhoWhisper Streaming Test</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  button { font-size: 16px; padding: 8px 20px; margin-right: 8px; }
  #status { color: #666; margin: 12px 0; }
  #transcript { border: 1px solid #ccc; border-radius: 6px; padding: 12px; min-height: 120px; white-space: pre-wrap; }
  .partial { color: #999; }
</style>
</head>
<body>
  <h1>PhoWhisper Streaming Test</h1>
  <button id="start">Start</button>
  <button id="stop" disabled>Stop</button>
  <div id="status">idle</div>
  <div id="transcript"></div>

<script>
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");

let ws, audioCtx, source, processor, stream;
let finalText = "";

function floatTo16kPCM16(float32, inputRate) {
  // downsample to 16kHz via linear interpolation, then convert to int16
  const ratio = inputRate / 16000;
  const outLength = Math.floor(float32.length / ratio);
  const out = new Int16Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIdx = i * ratio;
    const i0 = Math.floor(srcIdx);
    const i1 = Math.min(i0 + 1, float32.length - 1);
    const frac = srcIdx - i0;
    const sample = float32[i0] * (1 - frac) + float32[i1] * frac;
    const clamped = Math.max(-1, Math.min(1, sample));
    out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  return out;
}

startBtn.onclick = async () => {
  startBtn.disabled = true;
  stopBtn.disabled = false;
  finalText = "";
  transcriptEl.textContent = "";
  statusEl.textContent = "requesting mic...";

  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new AudioContext();
  source = audioCtx.createMediaStreamSource(stream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/transcribe`);
  ws.onopen = () => (statusEl.textContent = "streaming...");
  ws.onclose = () => (statusEl.textContent = "closed");
  ws.onerror = (e) => (statusEl.textContent = "error: " + e.message);
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.event === "auto_stop") {
      stopRecording({ sendEnd: false });
      statusEl.textContent = "stopped (silence detected)";
      return;
    }
    finalText += (finalText ? " " : "") + msg.text;
    transcriptEl.textContent = finalText;
  };

  processor.onaudioprocess = (e) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const pcm16 = floatTo16kPCM16(input, audioCtx.sampleRate);
    ws.send(pcm16.buffer);
  };

  source.connect(processor);
  processor.connect(audioCtx.destination);
};

function stopRecording({ sendEnd }) {
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "stopping...";

  processor && processor.disconnect();
  source && source.disconnect();
  stream && stream.getTracks().forEach((t) => t.stop());
  audioCtx && audioCtx.close();

  if (sendEnd && ws && ws.readyState === WebSocket.OPEN) {
    ws.send("end");
  }
}

stopBtn.onclick = () => stopRecording({ sendEnd: true });
</script>
</body>
</html>
'@

# ---------------------------------------------------------------------------

if (Test-Path $OutDir) {
    Write-Host "Removing existing $OutDir ..."
    Remove-Item -Recurse -Force $OutDir
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

Write-Host "Writing app source..."
Set-Content -Path (Join-Path $OutDir "main.py") -Value $MainPy -Encoding UTF8
Set-Content -Path (Join-Path $OutDir "download_model.py") -Value $DownloadModelPy -Encoding UTF8
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "static") | Out-Null
Set-Content -Path (Join-Path $OutDir "static\index.html") -Value $IndexHtml -Encoding UTF8
$RequirementsPath = Join-Path $OutDir "requirements.txt"
Set-Content -Path $RequirementsPath -Value $RequirementsTxt -Encoding UTF8

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
& $PyExe -m pip install --no-cache-dir -r $RequirementsPath

Write-Host "Downloading model weights..."
Push-Location $OutDir
try {
    & $PyExe download_model.py
} finally {
    Pop-Location
}

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
Write-Host "Run 'run.bat' from that folder to start the server, or zip the whole"
Write-Host "folder and copy it to another machine (no internet needed there)."
