#!/usr/bin/env bash
# Builds a self-contained, portable folder for running the STT server on a Linux
# server without a pre-installed Python/ffmpeg. Downloads a standalone CPython build
# (python-build-standalone -- a real, independent interpreter, not a venv, so it
# doesn't depend on whatever Python the target machine has, or lacks), installs all
# deps into it, downloads and converts the model weights, downloads a static ffmpeg
# build, and writes out the app code (embedded in this script -- no repo checkout
# needed).
#
# This is the Linux counterpart of build_portable.ps1; see that file's header for the
# full design rationale (CTranslate2 not PyTorch at runtime, CPU-only by default,
# GPU/model switchable after the build without a rebuild). Both scripts produce the
# same folder layout: src/ for the Python modules, everything else (launchers,
# config.ini, model_id.txt, python/, bin/, cuda/, models-ct2/, static/) in the root.
#
# This script is fully self-contained: copy just this one file to the target machine
# and run it there, or run it on a dev machine and copy the resulting output folder.
# Run it ON Linux (or WSL with a native Linux filesystem -- extraction relies on
# symlinks, which a Windows-mounted path can't create).
#
# Usage:
#   ./build_portable.sh
#   ./build_portable.sh --out-dir /opt/vietnamese-stt-server
#   ./build_portable.sh --model medium --include-cuda
#
# Requires: internet access on the machine running this script (to fetch the
# standalone Python build, pip packages, ffmpeg, and the model weights from Hugging
# Face).

set -euo pipefail

OUT_DIR="dist/vietnamese-stt-server-portable-linux"
PY_MINOR="3.13"                     # major.minor only; the exact patch comes from
                                     # whatever python-build-standalone last released
PYTHON_BUILD_URL=""                 # override: a full install_only tarball URL
FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
MODEL="medium"
INCLUDE_CUDA=0
STRIP_TORCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --python-minor) PY_MINOR="$2"; shift 2 ;;
    --python-build-url) PYTHON_BUILD_URL="$2"; shift 2 ;;
    --ffmpeg-url) FFMPEG_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --include-cuda) INCLUDE_CUDA=1; shift ;;
    --strip-torch) STRIP_TORCH=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Embedded app source: verbatim copies of src/*.py, requirements.txt and
# static/index.html from this repo, written into the output as src/*.py plus the app
# root files, so the shipped folder has the same layout as the checkout.
# ---------------------------------------------------------------------------

echo "Removing existing $OUT_DIR ..."
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/src"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
SRC_DIR="$OUT_DIR/src"

echo "Writing app source..."
cat > "$SRC_DIR/paths.py" <<'PYEOF_APP'
"""Filesystem layout shared by every module in src/.

The Python code lives in src/, while everything a user or the launcher touches sits
one level up in the app folder: config.ini, the .bat wrappers, models-ct2/, cuda/,
bin/, static/. Paths are resolved from this file rather than the working directory,
so the scripts behave the same whether run from the app folder, from src/, or by
run.bat.
"""
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
APP_DIR = SRC_DIR.parent

STATIC_DIR = APP_DIR / "static"
BIN_DIR = APP_DIR / "bin"
CUDA_DIR = APP_DIR / "cuda"
# Raw Hugging Face weights (converter input; deleted after a successful conversion).
MODELS_DIR = APP_DIR / "models"
# CTranslate2 model the server actually loads.
MODEL_CT2_DIR = APP_DIR / "models-ct2"
# Which checkpoint this install serves; written by switch_model.py.
MODEL_ID_FILE = APP_DIR / "model_id.txt"
PYEOF_APP

cat > "$SRC_DIR/main.py" <<'PYEOF_APP'
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import paths

# The portable Windows build ships the CUDA DLLs that CTranslate2 needs (cuBLAS, and
# the cuDNN 9 sublibraries) in a "cuda" folder in the app root. They are loaded
# lazily by name at first inference, so the folder has to be on the DLL search path
# before then -- register it here rather than relying on the launcher's PATH.
CUDA_DLL_DIR = Path(os.environ.get("CUDA_DLL_DIR") or paths.CUDA_DIR)
if CUDA_DLL_DIR.is_dir():
    if sys.platform == "win32":
        os.add_dll_directory(str(CUDA_DLL_DIR))
        os.environ["PATH"] = str(CUDA_DLL_DIR) + os.pathsep + os.environ.get("PATH", "")
    else:
        # glibc's dynamic linker re-reads LD_LIBRARY_PATH on every dlopen(), not just
        # at process start, so setting it here (before ctranslate2 is imported and
        # dlopen's cuBLAS/cuDNN by name) still takes effect -- same trick the nvidia-*
        # wheels' own loaders use.
        os.environ["LD_LIBRARY_PATH"] = (
            str(CUDA_DLL_DIR) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )

import ctranslate2
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from download_model import current_repo_id

MODEL_DIR = os.environ.get("MODEL_DIR") or str(paths.MODEL_CT2_DIR)
SAMPLE_RATE = 16000

# Resolve ffmpeg: explicit override, then PATH, then a copy bundled alongside the app
# (used by the portable Windows package, which ships its own ffmpeg.exe).
FFMPEG_BIN = (
    os.environ.get("FFMPEG_BIN")
    or shutil.which("ffmpeg")
    or str(paths.BIN_DIR / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"))
)


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
app.mount("/static", StaticFiles(directory=str(paths.STATIC_DIR)), name="static")


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
PYEOF_APP

cat > "$SRC_DIR/download_model.py" <<'PYEOF_APP'
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

import paths

DEFAULT_REPO_ID = "vinai/PhoWhisper-medium"
MODEL_ID_FILE = paths.MODEL_ID_FILE

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
        local_dir=str(paths.MODELS_DIR),
        allow_patterns=ALLOW_PATTERNS,
    )
    return repo_id


REPO_ID = current_repo_id()

if __name__ == "__main__":
    fetched = download(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Done. {fetched} files are now in models/. Run convert_ct2.py next.")
PYEOF_APP

cat > "$SRC_DIR/convert_ct2.py" <<'PYEOF_APP'
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

import paths

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
    model_dir = Path(model_dir) if model_dir else paths.MODELS_DIR
    out_dir = Path(out_dir) if out_dir else paths.MODEL_CT2_DIR
    converter = TransformersConverter(str(model_dir), copy_files=COPY_FILES)
    converter.convert(str(out_dir), quantization="int8", force=True)
    return out_dir


if __name__ == "__main__":
    print(f"Done. CTranslate2 model is now in {convert()}")
PYEOF_APP

cat > "$SRC_DIR/switch_model.py" <<'PYEOF_APP'
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

SRC_DIR = Path(__file__).resolve().parent

# The portable build runs an embeddable Python, whose python3xx._pth puts the
# interpreter in isolated mode: sys.path comes only from that file, and the script's
# own directory is NOT added. Put it back before importing the sibling modules.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import convert_ct2  # noqa: E402
import download_model  # noqa: E402
import paths  # noqa: E402


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

    paths.MODEL_ID_FILE.write_text(repo_id + "\n", encoding="utf-8")

    if not args.keep_raw:
        shutil.rmtree(paths.MODELS_DIR, ignore_errors=True)

    print(f"Done. Now serving {repo_id}. Restart the server to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF_APP

cat > "$SRC_DIR/enable_gpu.py" <<'PYEOF_APP'
"""Download the CUDA runtime libraries CTranslate2 needs for GPU mode into cuda/.

Portable builds are CPU-only by default; run this once, on a machine with an NVIDIA
GPU and internet access, to add GPU support (~1.5GB). Only the NVIDIA driver has to
be installed there -- these wheels carry the whole CUDA 12 / cuDNN 9 runtime, so no
CUDA toolkit is needed. Afterwards set DEVICE=auto (or cuda) in config.ini and
restart the server (run.bat on Windows, run.sh on Linux).
"""
import shutil
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import paths  # noqa: E402

CUDA_DIR = paths.CUDA_DIR
# Installed with --target into a scratch dir rather than into the embedded Python, so
# the wheels stay out of pip's package list and only the DLLs are kept.
STAGING_DIR = paths.APP_DIR / ".cuda-wheels"
PACKAGES = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]

if __name__ == "__main__":
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "--target", str(STAGING_DIR), *PACKAGES]
        )
        # Windows wheels ship .dll files; Linux wheels ship versioned .so files
        # (libcublas.so.12, ...) plus a couple of plain .so symlinks.
        pattern = "*.dll" if sys.platform == "win32" else "*.so*"
        CUDA_DIR.mkdir(exist_ok=True)
        copied = 0
        for lib in STAGING_DIR.rglob(pattern):
            if lib.is_file():
                shutil.copy2(lib, CUDA_DIR / lib.name)
                copied += 1
    finally:
        shutil.rmtree(STAGING_DIR, ignore_errors=True)

    total_mb = sum(f.stat().st_size for f in CUDA_DIR.glob(pattern)) / (1024 * 1024)
    print(f"Copied {copied} DLLs into {CUDA_DIR} ({total_mb:,.0f} MB total).")
    print("Set DEVICE=auto (or cuda) in config.ini, then restart the server.")
PYEOF_APP

mkdir -p "$OUT_DIR/static"
cat > "$OUT_DIR/static/index.html" <<'HTMLEOF_APP'
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<title>PhoWhisper Transcribe Test</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  button { font-size: 16px; padding: 8px 20px; margin-right: 8px; }
  #status { color: #666; margin: 12px 0; }
  #transcript { border: 1px solid #ccc; border-radius: 6px; padding: 12px; min-height: 120px; white-space: pre-wrap; }
</style>
</head>
<body>
<h1>PhoWhisper Transcribe Test</h1>
<p>Records locally; transcribes once (via POST /transcribe) only after you stop — either by clicking Stop or after silence auto-stops it.</p>
<button id="start">Start</button>
<button id="stop" disabled>Stop</button>
<div id="status">idle</div>
<div id="transcript"></div>

<script>
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");

const VAD_SILENCE_RMS_THRESHOLD = 0.01;
const VAD_AUTO_STOP_SILENCE_SECONDS = 2.0;

let mediaRecorder, recordedChunks, stream, audioCtx, source, vadProcessor;
let speechDetected = false;
let silenceSeconds = 0;

function rms(float32) {
  let sum = 0;
  for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
  return Math.sqrt(sum / float32.length);
}

startBtn.onclick = async () => {
  startBtn.disabled = true;
  stopBtn.disabled = false;
  transcriptEl.textContent = "";
  statusEl.textContent = "requesting mic...";
  speechDetected = false;
  silenceSeconds = 0;

  stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = async () => {
    statusEl.textContent = "transcribing...";
    const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
    const formData = new FormData();
    formData.append("file", blob, "recording.webm");
    try {
      const res = await fetch("/transcribe", { method: "POST", body: formData });
      const data = await res.json();
      transcriptEl.textContent = data.text || "(no speech detected)";
      statusEl.textContent = "done";
    } catch (err) {
      statusEl.textContent = "error: " + err.message;
    }
  };
  mediaRecorder.start();

  // Local silence detection only decides *when* to stop recording; the audio
  // itself is buffered client-side and sent as a single file once stopped.
  audioCtx = new AudioContext();
  source = audioCtx.createMediaStreamSource(stream);
  vadProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
  vadProcessor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    const chunkDuration = input.length / audioCtx.sampleRate;
    if (rms(input) < VAD_SILENCE_RMS_THRESHOLD) {
      if (speechDetected) {
        silenceSeconds += chunkDuration;
        if (silenceSeconds >= VAD_AUTO_STOP_SILENCE_SECONDS) {
          statusEl.textContent = "stopped (silence detected)";
          stopRecording();
        }
      }
    } else {
      speechDetected = true;
      silenceSeconds = 0;
    }
  };
  source.connect(vadProcessor);
  vadProcessor.connect(audioCtx.destination);

  statusEl.textContent = "recording...";
};

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;
  startBtn.disabled = false;
  stopBtn.disabled = true;

  vadProcessor && vadProcessor.disconnect();
  source && source.disconnect();
  audioCtx && audioCtx.close();
  stream && stream.getTracks().forEach((t) => t.stop());

  mediaRecorder.stop();
}

stopBtn.onclick = () => {
  statusEl.textContent = "stopping...";
  stopRecording();
};
</script>
</body>
</html>
HTMLEOF_APP

cat > "$OUT_DIR/requirements.txt" <<'PYEOF_APP'
fastapi
uvicorn[standard]
python-multipart
faster-whisper
huggingface_hub
soundfile
numpy
PYEOF_APP

# The checkpoint the app serves, read back by download_model.current_repo_id() and
# rewritten by switch_model.py. Size shortcuts ("medium") resolve the same as full
# repo ids, so $MODEL can be stored verbatim.
echo "$MODEL" > "$OUT_DIR/model_id.txt"

# ---------------------------------------------------------------------------
# Standalone Python runtime (python-build-standalone -- what uv/rye use for the same
# purpose: a real interpreter with no dependency on the target machine's system
# Python, unlike a venv).
# ---------------------------------------------------------------------------

if [[ -z "$PYTHON_BUILD_URL" ]]; then
  echo "Resolving latest python-build-standalone release for Python $PY_MINOR ..."
  # GitHub URL-encodes the "+" in the asset filename (cpython-X.Y.Z+DATE-...) as
  # %2B in browser_download_url, so match that instead of a literal "+".
  ASSET_URL=$(curl -fsSL "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest" \
    | grep -o "\"browser_download_url\": *\"[^\"]*cpython-$PY_MINOR\.[0-9]*%2B[0-9]*-x86_64-unknown-linux-gnu-install_only\.tar\.gz\"" \
    | head -1 \
    | sed -E 's/.*"(https[^"]+)"/\1/')
  if [[ -z "$ASSET_URL" ]]; then
    echo "Could not find a python-build-standalone asset for Python $PY_MINOR." >&2
    echo "Pass --python-build-url with a direct .tar.gz URL from" >&2
    echo "https://github.com/astral-sh/python-build-standalone/releases instead." >&2
    exit 1
  fi
else
  ASSET_URL="$PYTHON_BUILD_URL"
fi

echo "Downloading standalone Python from $ASSET_URL ..."
PY_TARBALL=$(mktemp)
curl -fsSL "$ASSET_URL" -o "$PY_TARBALL"
tar xzf "$PY_TARBALL" -C "$OUT_DIR"   # extracts to $OUT_DIR/python/
rm -f "$PY_TARBALL"

PY_EXE="$OUT_DIR/python/bin/python3"
"$PY_EXE" -c "print('interpreter OK:', __import__('sys').version)"

echo "Upgrading pip..."
"$PY_EXE" -m pip install --no-cache-dir --upgrade pip

echo "Installing requirements (faster-whisper, fastapi, etc.)..."
"$PY_EXE" -m pip install --no-cache-dir -r "$OUT_DIR/requirements.txt"

# torch + transformers are the model converter: CTranslate2 reads the Hugging Face
# checkpoint through them. CPU torch is all that is needed even for a GPU deployment
# -- it only reads the weights, it never runs them -- and they stay installed in the
# output so switch_model.sh can convert a different checkpoint on the target machine
# later. --strip-torch removes them again for a smaller, fixed-model package.
if [[ "$STRIP_TORCH" -eq 1 ]]; then
  # Snapshot installed packages first, so torch/transformers' transitive deps (e.g.
  # sympy/networkx/mpmath) can be removed too by diffing against this baseline --
  # `pip uninstall torch transformers` alone leaves them behind as dead weight.
  BASELINE_PACKAGES=$(mktemp)
  "$PY_EXE" -m pip freeze | cut -d= -f1 | tr 'A-Z' 'a-z' | sort > "$BASELINE_PACKAGES"
fi

echo "Installing the model converter (CPU torch + transformers)..."
"$PY_EXE" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
"$PY_EXE" -m pip install --no-cache-dir transformers

echo "Downloading model weights ($MODEL)..."
MODEL_REPO="$MODEL" "$PY_EXE" "$SRC_DIR/download_model.py"

# Convert to CTranslate2 int8 format (~240MB for small, vs ~923MB for the raw fp32
# weights).
echo "Converting model weights to CTranslate2 int8 format..."
"$PY_EXE" "$SRC_DIR/convert_ct2.py"

if [[ "$STRIP_TORCH" -eq 1 ]]; then
  echo "Removing conversion-only packages (torch, transformers, and their deps)..."
  "$PY_EXE" -m pip uninstall -y torch transformers
  CURRENT_PACKAGES=$(mktemp)
  "$PY_EXE" -m pip freeze | cut -d= -f1 | tr 'A-Z' 'a-z' | sort > "$CURRENT_PACKAGES"
  ORPHANS=$(comm -23 "$CURRENT_PACKAGES" "$BASELINE_PACKAGES" || true)
  if [[ -n "$ORPHANS" ]]; then
    echo "Also removing leftover transitive deps: $(echo "$ORPHANS" | tr '\n' ' ')"
    "$PY_EXE" -m pip uninstall -y $ORPHANS
  fi
  rm -f "$BASELINE_PACKAGES" "$CURRENT_PACKAGES"
  rm -f "$SRC_DIR/switch_model.py" "$OUT_DIR/switch_model.sh"
  rm -f "$SRC_DIR/convert_ct2.py"
fi

# The raw HF weights are only converter input; models-ct2/ is what the app loads.
rm -rf "$OUT_DIR/models"

# ---------------------------------------------------------------------------
# Static ffmpeg
# ---------------------------------------------------------------------------

echo "Downloading static ffmpeg build..."
mkdir -p "$OUT_DIR/bin"
FFMPEG_TMP=$(mktemp -d)
curl -fsSL "$FFMPEG_URL" -o "$FFMPEG_TMP/ffmpeg.tar.xz"
tar xf "$FFMPEG_TMP/ffmpeg.tar.xz" -C "$FFMPEG_TMP"
FFMPEG_BINARY=$(find "$FFMPEG_TMP" -type f -name ffmpeg | head -1)
if [[ -z "$FFMPEG_BINARY" ]]; then
  echo "ffmpeg binary not found inside downloaded archive from $FFMPEG_URL" >&2
  exit 1
fi
cp "$FFMPEG_BINARY" "$OUT_DIR/bin/ffmpeg"
chmod +x "$OUT_DIR/bin/ffmpeg"
rm -rf "$FFMPEG_TMP"

# ---------------------------------------------------------------------------
# GPU runtime (cuBLAS + cuDNN), same trade-off as build_portable.ps1: off by
# default (the build is CPU-only), --include-cuda bundles it now, or run
# enable_gpu.sh on the target machine later.
# ---------------------------------------------------------------------------

if [[ "$INCLUDE_CUDA" -eq 0 ]]; then
  echo "Skipping the CUDA runtime bundle; this build is CPU-only."
  echo "Run enable_gpu.sh on the target machine to add GPU support later."
else
  echo "Downloading CUDA runtime libraries (cuBLAS + cuDNN) for GPU support..."
  "$PY_EXE" "$SRC_DIR/enable_gpu.py"
fi

# ---------------------------------------------------------------------------
# Launcher scripts
# ---------------------------------------------------------------------------

cat > "$OUT_DIR/config.ini" <<'CFGEOF'
; Edit these values, then restart run.sh to apply them.
HOST=0.0.0.0
PORT=8000

; Force "cuda" or "cpu", or leave as "auto" to use CUDA when available. GPU mode
; needs an NVIDIA driver new enough for CUDA 12, plus the cuBLAS/cuDNN runtime in the
; cuda/ folder -- if that folder is missing or empty, this package was built CPU-only:
; run enable_gpu.sh once (needs internet) to download it. No CUDA toolkit install is
; required on this machine either way.
DEVICE=auto

; On a multi-GPU machine, pin to one GPU (e.g. "0" or "1") to avoid contention with
; other processes already loaded onto a busier GPU. Leave blank to let CUDA pick.
CUDA_VISIBLE_DEVICES=
CFGEOF

cat > "$OUT_DIR/run.sh" <<'RUNEOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

set -a
# eol=; in the Windows build's config.ini reader means the same thing here: lines
# starting with ";" are comments, blank lines are skipped.
source <(grep -v '^\s*;' config.ini | grep -v '^\s*$')
set +a

export MODEL_DIR="$PWD/models-ct2"
export FFMPEG_BIN="$PWD/bin/ffmpeg"
export CUDA_DLL_DIR="$PWD/cuda"
export LD_LIBRARY_PATH="$PWD/cuda:${LD_LIBRARY_PATH:-}"

exec ./python/bin/python3 -m uvicorn main:app --app-dir src --host "$HOST" --port "$PORT"
RUNEOF
chmod +x "$OUT_DIR/run.sh"

cat > "$OUT_DIR/switch_model.sh" <<'SWITCHEOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
exec ./python/bin/python3 src/switch_model.py "$@"
SWITCHEOF
chmod +x "$OUT_DIR/switch_model.sh"

cat > "$OUT_DIR/enable_gpu.sh" <<'ENABLEEOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
exec ./python/bin/python3 src/enable_gpu.py
ENABLEEOF
chmod +x "$OUT_DIR/enable_gpu.sh"

SIZE_MB=$(du -sm "$OUT_DIR" | cut -f1)
echo ""
echo "Done. Portable app built at: $OUT_DIR ($SIZE_MB MB)"
echo ""
echo "Run './run.sh' from that folder to start the server, or tar the whole folder"
echo "and copy it to another machine (no internet needed there)."
echo ""
if [[ "$INCLUDE_CUDA" -eq 0 ]]; then
  echo "GPU support:   ./enable_gpu.sh        (downloads ~1.5GB of CUDA libraries)"
fi
if [[ "$STRIP_TORCH" -eq 0 ]]; then
  echo "Change model:  ./switch_model.sh medium   (currently: $MODEL)"
fi
