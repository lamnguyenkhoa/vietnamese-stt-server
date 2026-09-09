<#
Builds a self-contained, portable folder for running the STT server on a Windows
server without a pre-installed Python/ffmpeg. Downloads the official Python
embeddable distribution (a standalone runtime, not a venv -- no dependency on any
Python already present on the target machine), bootstraps pip into it, installs all
deps, downloads and converts the model weights, downloads a static ffmpeg build, and
writes out the app code (embedded in this script -- no repo checkout needed).

The app runs on faster-whisper (CTranslate2), not PyTorch, so inference itself never
touches torch and the model ships int8-quantized (~240MB for small, instead of
~460MB+ fp16/fp32).

The output is CPU-only by default and both halves of that are reversible on the
target machine, without rebuilding:

  * enable_gpu.bat  -- downloads the CUDA runtime DLLs CTranslate2 needs (cuBLAS +
    cuDNN 9, ~1.5GB) into cuda\, after which DEVICE=auto picks up the GPU. Only the
    NVIDIA driver is required there, no CUDA toolkit. Pass -IncludeCuda at build time
    to bundle them up front instead (e.g. when the target machine has no internet).
  * switch_model.bat medium  -- re-downloads and re-converts to another PhoWhisper
    checkpoint (tiny/base/small/medium/large, or any HF repo id). This is why CPU
    torch + transformers stay installed in the output (~1GB): they are the converter.
    Pass -StripTorch to remove them for a smaller, fixed-model package.

This script is fully self-contained: copy just this one file to the target machine
and run it there, or run it on a dev machine and copy the resulting output folder.

Usage:
    .\build_portable.ps1
    .\build_portable.ps1 -OutDir C:\deploy\vietnamese-stt-server
    .\build_portable.ps1 -Model medium -IncludeCuda

Requires: internet access on the machine running this script (to fetch the embeddable
Python, pip, ffmpeg, and the model weights from Hugging Face).
#>

param(
    [string]$OutDir = "dist\vietnamese-stt-server-portable",
    [string]$PythonVersion = "3.13.12",
    # BtbN/FFmpeg-Builds release asset, pinned to the n8.1 release build (static, GPL),
    # not the rolling nightly "master" build, so the ffmpeg version stays predictable.
    [string]$FfmpegAssetUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-8.1.zip",
    # Checkpoint to ship. A PhoWhisper size shortcut (tiny/base/small/medium/large)
    # or any Hugging Face repo id. Changeable after the build with switch_model.bat.
    [string]$Model = "small",
    # Bundle the CUDA runtime DLLs (cuBLAS + cuDNN, ~1.5GB) into cuda\ at build time.
    # Off by default: the output is CPU-only, and enable_gpu.bat fetches them on the
    # target machine instead. Use this when that machine has no internet access.
    [switch]$IncludeCuda,
    # Uninstall torch/transformers after the model conversion, for a ~1GB smaller
    # output. Trade-off: switch_model.bat can no longer convert a different
    # checkpoint, so the shipped model becomes fixed.
    [switch]$StripTorch
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
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

# The portable Windows build ships the CUDA DLLs that CTranslate2 needs (cuBLAS, and
# the cuDNN 9 sublibraries) in a "cuda" folder next to the app. They are loaded
# lazily by name at first inference, so the folder has to be on the DLL search path
# before then -- register it here rather than relying on the launcher's PATH.
CUDA_DLL_DIR = Path(os.environ.get("CUDA_DLL_DIR") or Path(__file__).parent / "cuda")
if sys.platform == "win32" and CUDA_DLL_DIR.is_dir():
    os.add_dll_directory(str(CUDA_DLL_DIR))
    os.environ["PATH"] = str(CUDA_DLL_DIR) + os.pathsep + os.environ.get("PATH", "")

import ctranslate2
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from download_model import current_repo_id

MODEL_DIR = os.environ.get("MODEL_DIR", "models-ct2")
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

device = "cpu"  # placeholder; resolve_device() sets the real value in lifespan()
model_state = {}


def resolve_device() -> str:
    """Pick cuda/cpu from the DEVICE env var (set via --device or directly), or auto-detect."""
    requested = os.environ.get("DEVICE", "auto").lower()
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError(f"Invalid DEVICE={requested!r}; expected 'auto', 'cuda', or 'cpu'")
    cuda_available = ctranslate2.get_cuda_device_count() > 0
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        logger.warning("DEVICE=cuda requested but CUDA is not available; falling back to CPU.")
        return "cpu"
    return requested


def resolve_compute_type(resolved_device: str) -> str:
    """Pick a CTranslate2 compute type: int8 on CPU (fast + small), float16 on GPU
    (CUDA has real fp16 hardware acceleration, unlike CPU). Override with COMPUTE_TYPE."""
    requested = os.environ.get("COMPUTE_TYPE")
    if requested:
        return requested
    return "float16" if resolved_device == "cuda" else "int8"


def _load_model(target_device: str) -> "WhisperModel":
    model = WhisperModel(
        MODEL_DIR, device=target_device, compute_type=resolve_compute_type(target_device)
    )
    # cuBLAS/cuDNN are loaded lazily on first inference, not at model construction --
    # a missing/unreachable CUDA runtime only raises here. Force that to happen now,
    # during startup, rather than on a user's first request.
    list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="vi")[0])
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global device
    device = resolve_device()
    if device == "cpu":
        logger.warning(
            "Running on CPU. If a GPU was expected, check the driver's CUDA ceiling "
            "(nvidia-smi) against the ctranslate2 build installed."
        )

    try:
        model_state["model"] = _load_model(device)
    except Exception:
        # get_cuda_device_count() only confirms a CUDA-capable GPU + driver exist, not
        # that cuBLAS/cuDNN are actually loadable (e.g. missing on the host, or not on
        # PATH) -- that failure only surfaces here, at model load. Fall back to CPU
        # rather than crash the whole server on startup.
        if device != "cuda":
            raise
        logger.exception("Failed to load model on cuda; falling back to CPU.")
        device = "cpu"
        model_state["model"] = _load_model(device)
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
    model = model_state["model"]
    # beam_size=1 (greedy) matches the decoding this PhoWhisper checkpoint was
    # actually used with before this migration (transformers' plain .generate(), which
    # defaults to greedy). faster-whisper's own default, beam_size=5, was measurably
    # worse on this fine-tune -- e.g. "áo đỏ" (red shirt) misdecoded as "áo đảo" on a
    # real test clip, an error greedy decoding doesn't make.
    segments, _ = model.transcribe(audio, language="vi", task="transcribe", beam_size=1)
    return " ".join(segment.text.strip() for segment in segments).strip()


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
    return {"status": "ok", "device": device, "model": current_repo_id()}


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        help="Force cuda/cpu, or auto-detect (default; same as $DEVICE)",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8123")))
    args = parser.parse_args()

    if args.device:
        os.environ["DEVICE"] = args.device

    uvicorn.run(app, host=args.host, port=args.port)
'@

$DownloadModelPy = @'
"""Fetch a PhoWhisper checkpoint's raw HF files (config, tokenizer, weights) into models/.

Which checkpoint gets fetched is not hardcoded: it comes from $MODEL_REPO, else from
model_id.txt (written by switch_model.py), else the default below -- so an already
built install can be moved between small/medium without editing any source.

This is a staging download only -- the app itself runs on the CTranslate2 format
produced by convert_ct2.py from these files, not on this directory directly.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "vinai/PhoWhisper-small"
MODEL_ID_FILE = BASE_DIR / "model_id.txt"

# Short names so callers can say "medium" instead of the full repo path. Any other
# value is passed through to Hugging Face as-is.
ALIASES = {
    "tiny": "vinai/PhoWhisper-tiny",
    "base": "vinai/PhoWhisper-base",
    "small": "vinai/PhoWhisper-small",
    "medium": "vinai/PhoWhisper-medium",
    "large": "vinai/PhoWhisper-large",
}

# Weights only in .bin form: the *.safetensors siblings would double the download.
ALLOW_PATTERNS = ["*.json", "*.txt", "vocab.*", "merges.txt", "*.model", "pytorch_model.bin"]


def resolve_repo_id(name: str) -> str:
    name = (name or "").strip()
    return ALIASES.get(name.lower(), name)


def current_repo_id() -> str:
    """The checkpoint this install is currently set to serve."""
    env = resolve_repo_id(os.environ.get("MODEL_REPO", ""))
    if env:
        return env
    try:
        recorded = resolve_repo_id(MODEL_ID_FILE.read_text(encoding="utf-8"))
    except OSError:
        recorded = ""
    return recorded or DEFAULT_REPO_ID


def download(repo_id: str | None = None) -> str:
    repo_id = resolve_repo_id(repo_id) if repo_id else current_repo_id()
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(BASE_DIR / "models"),
        allow_patterns=ALLOW_PATTERNS,
    )
    return repo_id


REPO_ID = current_repo_id()

if __name__ == "__main__":
    fetched = download(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Done. {fetched} files are now in models/. Run convert_ct2.py next.")
'@

$ConvertCt2Py = @'
"""Convert models/ (raw HF PhoWhisper weights, from download_model.py) into
CTranslate2 format in models-ct2/, quantized to int8. This is what main.py actually
loads at runtime via faster-whisper.

Requires transformers and torch installed -- only for this conversion, not at
runtime. The portable build keeps them installed so switch_model.py can re-run this
against a different checkpoint later; uninstall them for a slimmer environment if you
never intend to switch models.
"""
from pathlib import Path

from ctranslate2.converters import TransformersConverter

BASE_DIR = Path(__file__).resolve().parent

# Auxiliary files the HF repo doesn't put in a single weights blob: tokenizer,
# generation defaults, and PhoWhisper's Vietnamese text normalizer.
COPY_FILES = [
    "tokenizer.json",
    "preprocessor_config.json",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]


def convert(model_dir: str | Path | None = None, out_dir: str | Path | None = None) -> Path:
    model_dir = Path(model_dir) if model_dir else BASE_DIR / "models"
    out_dir = Path(out_dir) if out_dir else BASE_DIR / "models-ct2"
    converter = TransformersConverter(str(model_dir), copy_files=COPY_FILES)
    converter.convert(str(out_dir), quantization="int8", force=True)
    return out_dir


if __name__ == "__main__":
    print(f"Done. CTranslate2 model is now in {convert()}")
'@

$SwitchModelPy = @'
"""Switch which PhoWhisper checkpoint this install serves, e.g. small <-> medium.

    python switch_model.py medium
    python switch_model.py vinai/PhoWhisper-large

Downloads the raw HF weights, re-converts them to CTranslate2 int8 in models-ct2/,
and records the choice in model_id.txt so main.py reports it and later runs of
download_model.py stay on it. Restart the server afterwards to load the new model.

Needs torch + transformers installed (the portable build ships them for exactly
this); the conversion is CPU-only and does not need a GPU.
"""
import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# The portable build runs an embeddable Python, whose python3xx._pth puts the
# interpreter in isolated mode: sys.path comes only from that file, and the script's
# own directory is NOT added. Put it back before importing the sibling modules.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import convert_ct2  # noqa: E402
import download_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "model",
        nargs="?",
        help=f"Checkpoint to switch to: {', '.join(download_model.ALIASES)}, or any HF repo id",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the downloaded HF weights in models/ instead of deleting them after conversion",
    )
    args = parser.parse_args()

    if not args.model:
        print(f"Current model: {download_model.current_repo_id()}")
        print(f"Available shortcuts: {', '.join(download_model.ALIASES)}")
        return 0

    repo_id = download_model.resolve_repo_id(args.model)
    print(f"Switching to {repo_id} ...")

    try:
        download_model.download(repo_id)
    except Exception as exc:
        # A typo'd repo id fails here, before models-ct2/ is touched, so the currently
        # working model is left intact.
        print(f"Download failed ({exc}). Model unchanged.", file=sys.stderr)
        return 1

    print("Converting to CTranslate2 int8 ...")
    convert_ct2.convert()

    (BASE_DIR / "model_id.txt").write_text(repo_id + "\n", encoding="utf-8")

    if not args.keep_raw:
        shutil.rmtree(BASE_DIR / "models", ignore_errors=True)

    print(f"Done. Now serving {repo_id}. Restart the server to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@

$EnableGpuPy = @'
"""Download the CUDA runtime DLLs CTranslate2 needs for GPU mode into the cuda folder.

Portable builds are CPU-only by default; run this once, on a machine with an NVIDIA
GPU and internet access, to add GPU support (~1.5GB). Only the NVIDIA driver has to
be installed there -- these wheels carry the whole CUDA 12 / cuDNN 9 runtime, so no
CUDA toolkit is needed. Afterwards set DEVICE=auto (or cuda) in config.ini and
restart run.bat.
"""
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CUDA_DIR = BASE_DIR / "cuda"
# Installed with --target into a scratch dir rather than into the embedded Python, so
# the wheels stay out of pip's package list and only the DLLs are kept.
STAGING_DIR = BASE_DIR / ".cuda-wheels"
PACKAGES = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]

if __name__ == "__main__":
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "--target", str(STAGING_DIR), *PACKAGES]
        )
        CUDA_DIR.mkdir(exist_ok=True)
        copied = 0
        for dll in STAGING_DIR.rglob("*.dll"):
            shutil.copy2(dll, CUDA_DIR / dll.name)
            copied += 1
    finally:
        shutil.rmtree(STAGING_DIR, ignore_errors=True)

    total_mb = sum(f.stat().st_size for f in CUDA_DIR.glob("*.dll")) / (1024 * 1024)
    print(f"Copied {copied} DLLs into {CUDA_DIR} ({total_mb:,.0f} MB total).")
    print("Set DEVICE=auto (or cuda) in config.ini, then restart run.bat.")
'@

$SwitchModelBat = @'
@echo off
setlocal
cd /d "%~dp0"
"%~dp0python\python.exe" switch_model.py %*
'@

$EnableGpuBat = @'
@echo off
setlocal
cd /d "%~dp0"
"%~dp0python\python.exe" enable_gpu.py
'@

$RequirementsTxt = @'
fastapi
uvicorn[standard]
python-multipart
faster-whisper
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

<hr />

<h1>PhoWhisper Record-Then-Transcribe Test</h1>
<p>Records locally; transcribes once (via POST /transcribe) only after you stop -- either by clicking Stop or after silence auto-stops it. No streaming.</p>
<button id="start2">Start</button>
<button id="stop2" disabled>Stop</button>
<div id="status2">idle</div>
<div id="transcript2"></div>

<script>
const startBtn2 = document.getElementById("start2");
const stopBtn2 = document.getElementById("stop2");
const statusEl2 = document.getElementById("status2");
const transcriptEl2 = document.getElementById("transcript2");

const VAD_SILENCE_RMS_THRESHOLD = 0.01;
const VAD_AUTO_STOP_SILENCE_SECONDS = 2.0;

let mediaRecorder2, recordedChunks2, stream2, audioCtx2, source2, vadProcessor2;
let speechDetected2 = false;
let silenceSeconds2 = 0;

function rms(float32) {
  let sum = 0;
  for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
  return Math.sqrt(sum / float32.length);
}

startBtn2.onclick = async () => {
  startBtn2.disabled = true;
  stopBtn2.disabled = false;
  transcriptEl2.textContent = "";
  statusEl2.textContent = "requesting mic...";
  speechDetected2 = false;
  silenceSeconds2 = 0;

  stream2 = await navigator.mediaDevices.getUserMedia({ audio: true });

  recordedChunks2 = [];
  mediaRecorder2 = new MediaRecorder(stream2);
  mediaRecorder2.ondataavailable = (e) => {
    if (e.data.size > 0) recordedChunks2.push(e.data);
  };
  mediaRecorder2.onstop = async () => {
    statusEl2.textContent = "transcribing...";
    const blob = new Blob(recordedChunks2, { type: mediaRecorder2.mimeType });
    const formData = new FormData();
    formData.append("file", blob, "recording.webm");
    try {
      const res = await fetch("/transcribe", { method: "POST", body: formData });
      const data = await res.json();
      transcriptEl2.textContent = data.text || "(no speech detected)";
      statusEl2.textContent = "done";
    } catch (err) {
      statusEl2.textContent = "error: " + err.message;
    }
  };
  mediaRecorder2.start();

  // Local silence detection only decides *when* to stop recording; the audio
  // itself is buffered client-side and sent as a single file once stopped.
  audioCtx2 = new AudioContext();
  source2 = audioCtx2.createMediaStreamSource(stream2);
  vadProcessor2 = audioCtx2.createScriptProcessor(4096, 1, 1);
  vadProcessor2.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    const chunkDuration = input.length / audioCtx2.sampleRate;
    if (rms(input) < VAD_SILENCE_RMS_THRESHOLD) {
      if (speechDetected2) {
        silenceSeconds2 += chunkDuration;
        if (silenceSeconds2 >= VAD_AUTO_STOP_SILENCE_SECONDS) {
          statusEl2.textContent = "stopped (silence detected)";
          stopRecording2();
        }
      }
    } else {
      speechDetected2 = true;
      silenceSeconds2 = 0;
    }
  };
  source2.connect(vadProcessor2);
  vadProcessor2.connect(audioCtx2.destination);

  statusEl2.textContent = "recording...";
};

function stopRecording2() {
  if (!mediaRecorder2 || mediaRecorder2.state === "inactive") return;
  startBtn2.disabled = false;
  stopBtn2.disabled = true;

  vadProcessor2 && vadProcessor2.disconnect();
  source2 && source2.disconnect();
  audioCtx2 && audioCtx2.close();
  stream2 && stream2.getTracks().forEach((t) => t.stop());

  mediaRecorder2.stop();
}

stopBtn2.onclick = () => {
  statusEl2.textContent = "stopping...";
  stopRecording2();
};
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
Set-Content -Path (Join-Path $OutDir "convert_ct2.py") -Value $ConvertCt2Py -Encoding UTF8
Set-Content -Path (Join-Path $OutDir "switch_model.py") -Value $SwitchModelPy -Encoding UTF8
Set-Content -Path (Join-Path $OutDir "enable_gpu.py") -Value $EnableGpuPy -Encoding UTF8
Set-Content -Path (Join-Path $OutDir "switch_model.bat") -Value $SwitchModelBat -Encoding ASCII
Set-Content -Path (Join-Path $OutDir "enable_gpu.bat") -Value $EnableGpuBat -Encoding ASCII
# The checkpoint the app serves, read back by download_model.current_repo_id() and
# rewritten by switch_model.py. Size shortcuts ("medium") resolve the same as full
# repo ids, so $Model can be stored verbatim.
Set-Content -Path (Join-Path $OutDir "model_id.txt") -Value $Model -Encoding ASCII
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
# The ._pth also puts Python in isolated mode -- sys.path is exactly what this file
# lists, and a script's own directory is not added -- so append ".." (resolved
# relative to the ._pth, i.e. the app folder) to make main.py / download_model.py /
# convert_ct2.py importable from any working directory.
$PthFile = Join-Path $PyDir "python$VerTag._pth"
(Get-Content $PthFile) -replace '^#import site$', 'import site' | Set-Content $PthFile
Add-Content -Path $PthFile -Value ".."

Write-Host "Bootstrapping pip..."
$GetPipPath = Join-Path $env:TEMP "get-pip.py"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPipPath
& $PyExe $GetPipPath --no-warn-script-location
Remove-Item $GetPipPath

Write-Host "Installing requirements (faster-whisper, fastapi, etc.)..."
& $PyExe -m pip install --no-cache-dir -r $RequirementsPath

# torch + transformers are the model converter: CTranslate2 reads the Hugging Face
# checkpoint through them. CPU torch is all that is needed even for a GPU deployment
# -- it only reads the weights, it never runs them -- and they stay installed in the
# output so switch_model.bat can convert a different checkpoint on the target machine
# later. -StripTorch removes them again for a smaller, fixed-model package.
if ($StripTorch) {
    # Snapshot installed packages first, so torch/transformers' transitive deps (e.g.
    # sympy/networkx/mpmath) can be removed too by diffing against this baseline --
    # `pip uninstall torch transformers` alone leaves them behind as dead weight.
    $BaselinePackages = (& $PyExe -m pip freeze) | ForEach-Object { ($_ -split "==")[0].ToLower() }
}

Write-Host "Installing the model converter (CPU torch + transformers)..."
& $PyExe -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
& $PyExe -m pip install --no-cache-dir transformers

Write-Host "Downloading model weights ($Model)..."
Push-Location $OutDir
try {
    & $PyExe download_model.py
} finally {
    Pop-Location
}

# Convert to CTranslate2 int8 format (~240MB for small, vs ~923MB for the raw fp32
# weights).
Write-Host "Converting model weights to CTranslate2 int8 format..."
Push-Location $OutDir
try {
    & $PyExe convert_ct2.py
} finally {
    Pop-Location
}

if ($StripTorch) {
    Write-Host "Removing conversion-only packages (torch, transformers, and their deps)..."
    & $PyExe -m pip uninstall -y torch transformers
    $CurrentPackages = (& $PyExe -m pip freeze) | ForEach-Object { ($_ -split "==")[0] }
    $Orphans = $CurrentPackages | Where-Object { $BaselinePackages -notcontains $_.ToLower() }
    if ($Orphans) {
        Write-Host "Also removing leftover transitive deps: $($Orphans -join ', ')"
        & $PyExe -m pip uninstall -y @Orphans
    }
    Remove-Item (Join-Path $OutDir "switch_model.py") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $OutDir "switch_model.bat") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $OutDir "convert_ct2.py") -ErrorAction SilentlyContinue
}

# The raw HF weights are only converter input; models-ct2\ is what the app loads.
Remove-Item (Join-Path $OutDir "models") -Recurse -Force -ErrorAction SilentlyContinue

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

if (-not $IncludeCuda) {
    Write-Host "Skipping the CUDA runtime bundle; this build is CPU-only."
    Write-Host "Run enable_gpu.bat on the target machine to add GPU support later."
} else {
    # The ctranslate2 wheel ships ctranslate2.dll and cudnn64_9.dll, but none of the
    # CUDA libraries those load by name at first inference (cublas64_12.dll, the cuDNN
    # 9 sublibraries) -- so GPU mode dies with "Library cublas64_12.dll is not found"
    # on a machine with no CUDA toolkit installed. Ship them alongside the app.
    # Installed with --target into a temp dir rather than into the embedded Python, so
    # they stay out of pip's package list and only the DLLs land in the output. This
    # is the build-time twin of enable_gpu.py, for targets with no internet access.
    Write-Host "Downloading CUDA runtime DLLs (cuBLAS + cuDNN) for GPU support..."
    $CudaDir = Join-Path $OutDir "cuda"
    New-Item -ItemType Directory -Force -Path $CudaDir | Out-Null
    $NvidiaTmp = Join-Path $env:TEMP "vstt-nvidia-wheels"
    if (Test-Path $NvidiaTmp) { Remove-Item -Recurse -Force $NvidiaTmp }
    & $PyExe -m pip install --no-cache-dir --target $NvidiaTmp nvidia-cublas-cu12 nvidia-cudnn-cu12
    Get-ChildItem -Path $NvidiaTmp -Recurse -Filter "*.dll" |
        ForEach-Object { Copy-Item $_.FullName (Join-Path $CudaDir $_.Name) -Force }
    Remove-Item -Recurse -Force $NvidiaTmp
    $CudaSize = (Get-ChildItem $CudaDir | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ("Bundled CUDA runtime: {0:N0} MB in cuda\" -f $CudaSize)
}

$ConfigIni = @'
; Edit these values, then restart run.bat to apply them.
HOST=0.0.0.0
PORT=8123

; Force "cuda" or "cpu", or leave as "auto" to use CUDA when available. GPU mode
; needs an NVIDIA driver new enough for CUDA 12, plus the cuBLAS/cuDNN runtime in the
; cuda\ folder -- if that folder is missing or empty, this package was built CPU-only:
; run enable_gpu.bat once (needs internet) to download it. No CUDA toolkit install is
; required on this machine either way.
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
set PATH=%~dp0cuda;%PATH%
"%~dp0python\python.exe" -m uvicorn main:app --host %HOST% --port %PORT%
'@
Set-Content -Path (Join-Path $OutDir "run.bat") -Value $RunBat -Encoding ASCII

Write-Host ""
Write-Host "Done. Portable app built at: $OutDir"
Write-Host ""
Write-Host "Run 'run.bat' from that folder to start the server, or zip the whole"
Write-Host "folder and copy it to another machine (no internet needed there)."
Write-Host ""
if (-not $IncludeCuda) {
    Write-Host "GPU support:   enable_gpu.bat        (downloads ~1.5GB of CUDA DLLs)"
}
if (-not $StripTorch) {
    Write-Host "Change model:  switch_model.bat medium   (currently: $Model)"
}
