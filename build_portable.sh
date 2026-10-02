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
import asyncio
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
from faster_whisper.vad import get_speech_timestamps
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from download_model import current_repo_id

MODEL_DIR = os.environ.get("MODEL_DIR") or str(paths.MODEL_CT2_DIR)
SAMPLE_RATE = 16000
# Live preview (/ws/stream): re-transcribe the in-progress window whenever at least
# STREAM_UPDATE_SECONDS of new audio has arrived, and commit the window once it reaches
# STREAM_WINDOW_SECONDS so each pass stays short. Preview accuracy is secondary --
# clients are expected to POST the whole recording to /transcribe for the final text.
STREAM_UPDATE_SECONDS = float(os.environ.get("STREAM_UPDATE_SECONDS", "1.0"))
STREAM_WINDOW_SECONDS = float(os.environ.get("STREAM_WINDOW_SECONDS", "8.0"))

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
    # Whisper never outputs "nothing": fed silence or room noise it hallucinates fluent
    # Vietnamese sentences. Gate on Silero VAD (bundled with faster-whisper) and return
    # "" when it hears no speech. Only a gate, though -- passing vad_filter=True to
    # transcribe() instead also splices the audio down to the speech regions, which
    # measurably hurt accuracy on real clips.
    if not get_speech_timestamps(audio):
        return ""
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
    # Run decode + inference off the event loop so live-preview sockets stay responsive.
    audio = await asyncio.to_thread(load_audio, raw_bytes)
    return {"text": await asyncio.to_thread(transcribe_array, audio)}


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    """Live preview transcription, meant to run alongside a client-side recording.

    The client sends raw PCM16LE mono 16kHz audio as binary frames and closes the socket
    when done. The server replies with {"text": "..."} -- the full preview so far, which
    replaces the previous one. Only one inference runs at a time per socket; audio that
    arrives meanwhile is picked up by the next pass, so a slow device just updates less
    often instead of falling behind.
    """
    await websocket.accept()
    update_samples = int(STREAM_UPDATE_SECONDS * SAMPLE_RATE)
    window_samples = int(STREAM_WINDOW_SECONDS * SAMPLE_RATE)
    window = np.empty(0, dtype=np.float32)  # audio since the last commit
    committed: "list[str]" = []
    audio_arrived = asyncio.Event()

    async def preview_loop():
        nonlocal window
        transcribed_len = 0
        while True:
            await audio_arrived.wait()
            audio_arrived.clear()
            if len(window) - transcribed_len < update_samples:
                continue
            snapshot = window
            transcribed_len = len(snapshot)
            text = await asyncio.to_thread(transcribe_array, snapshot)
            if len(snapshot) >= window_samples:
                if text:
                    committed.append(text)
                # The receiver only ever appends, so everything past the snapshot is new.
                window = window[len(snapshot):]
                transcribed_len = 0
                text = ""
            await websocket.send_json({"text": " ".join(committed + [text]).strip()})

    preview_task = asyncio.create_task(preview_loop())
    try:
        while not preview_task.done():
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes"):
                pcm16 = np.frombuffer(message["bytes"], dtype=np.int16)
                window = np.concatenate([window, pcm16.astype(np.float32) / 32768.0])
                audio_arrived.set()
    except WebSocketDisconnect:
        pass
    finally:
        preview_task.cancel()


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
    # HTTPS: browsers only allow the web page's mic access on https:// (or localhost).
    # make_cert.py creates a self-signed pair.
    parser.add_argument("--ssl-certfile", default=os.environ.get("SSL_CERTFILE") or None)
    parser.add_argument("--ssl-keyfile", default=os.environ.get("SSL_KEYFILE") or None)
    args = parser.parse_args()

    if args.device:
        os.environ["DEVICE"] = args.device

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
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

cat > "$SRC_DIR/make_cert.py" <<'PYEOF_APP'
"""Create a self-signed HTTPS certificate for the server and enable it in config.ini.

Browsers only allow microphone access on https:// pages (or http://localhost), so the
web page can't record when the server is reached over plain http by IP. This writes
cert.pem/key.pem to the app folder, valid for localhost, this machine's hostname, its
IPv4 addresses, and any extra hostnames/IPs given on the command line:

    python make_cert.py                      # auto-detected names/IPs only
    python make_cert.py 203.0.113.5 stt.lan  # plus these

Browsers will warn once that the certificate is self-signed; accept it to continue.
Re-run after the machine's IP changes.
"""
import datetime
import ipaddress
import re
import socket
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

import paths

CERT_FILE = paths.APP_DIR / "cert.pem"
KEY_FILE = paths.APP_DIR / "key.pem"
CONFIG_FILE = paths.APP_DIR / "config.ini"


def local_ipv4s() -> "set[str]":
    ips = {"127.0.0.1"}
    try:
        ips.update(info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except socket.gaierror:
        pass
    # The address of the interface that would route outward (no packet is sent).
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return ips


def build_san(extra: "list[str]") -> "tuple[list[str], list[str]]":
    dns_names = {"localhost", socket.gethostname()}
    ips = local_ipv4s()
    for name in extra:
        try:
            ips.add(str(ipaddress.ip_address(name)))
        except ValueError:
            dns_names.add(name)
    return sorted(dns_names), sorted(ips, key=ipaddress.ip_address)


def enable_in_config() -> bool:
    """Point SSL_CERTFILE/SSL_KEYFILE in config.ini at the new files. Returns False if
    there is no config.ini (dev checkout) or it lacks those keys."""
    if not CONFIG_FILE.is_file():
        return False
    text = CONFIG_FILE.read_text(encoding="utf-8")
    new_text = re.sub(r"(?m)^SSL_CERTFILE=.*$", f"SSL_CERTFILE={CERT_FILE.name}", text)
    new_text = re.sub(r"(?m)^SSL_KEYFILE=.*$", f"SSL_KEYFILE={KEY_FILE.name}", new_text)
    if "SSL_CERTFILE=" not in new_text or "SSL_KEYFILE=" not in new_text:
        return False
    CONFIG_FILE.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    dns_names, ips = build_san(sys.argv[1:])

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Vietnamese STT Server")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(n) for n in dns_names]
                + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    KEY_FILE.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Wrote {CERT_FILE} and {KEY_FILE}, valid for:")
    for entry in dns_names + ips:
        print(f"  {entry}")
    if enable_in_config():
        print(f"Enabled HTTPS in {CONFIG_FILE.name}; restart the server to apply.")
    else:
        print(f"Start the server with: --ssl-certfile {CERT_FILE.name} --ssl-keyfile {KEY_FILE.name}")
    print("Then open https://<one of the addresses above>:<port>/static/index.html")


if __name__ == "__main__":
    main()
PYEOF_APP

mkdir -p "$OUT_DIR/static"
cat > "$OUT_DIR/static/index.html" <<'HTMLEOF_APP'
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<title>PhoWhisper Transcribe Test</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; }
  button { font-size: 16px; padding: 8px 20px; margin-right: 8px; }
  label { margin-left: 8px; }
  #status { color: #666; margin: 12px 0; }
  #transcript { border: 1px solid #ccc; border-radius: 6px; padding: 12px; min-height: 120px; white-space: pre-wrap; }
  #transcript.preview { color: #888; font-style: italic; }
  #dev { margin-top: 48px; border-top: 1px solid #ddd; padding-top: 8px; line-height: 1.5; }
  #dev table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
  #dev th, #dev td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }
  #dev th { background: #f5f5f5; }
  #dev code { font-family: ui-monospace, Consolas, monospace; font-size: 13px; background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }
  #dev pre { background: #f6f8fa; border: 1px solid #ddd; border-radius: 6px; padding: 12px; overflow-x: auto; margin: 0; }
  #dev pre code { background: none; padding: 0; }
  #dev .code { position: relative; margin: 12px 0; }
  #dev .copy { position: absolute; top: 6px; right: 6px; font-size: 12px; padding: 2px 10px; margin: 0; }
</style>
</head>
<body>
<h1>PhoWhisper Transcribe Test</h1>
<p>Records locally; transcribes once (via POST /transcribe) only after you stop — either by clicking Stop or after silence auto-stops it.
With live preview on, audio is also streamed to /ws/stream while you speak, and the rough preview (grey) is replaced by the whole-recording result when it arrives.</p>
<button id="start">Start</button>
<button id="stop" disabled>Stop</button>
<label><input type="checkbox" id="preview" checked /> Live preview</label>
<div id="status">idle</div>
<div id="transcript"></div>

<section id="dev">
<h2>Developer guide</h2>
<p>The server offers two ways to transcribe, designed to be used <strong>together</strong>:</p>
<table>
<thead><tr><th>Endpoint</th><th>What it gives you</th><th>Accuracy</th><th>When</th></tr></thead>
<tbody>
<tr><td><code>WS /ws/stream</code></td><td>Rough text that grows while the user speaks</td><td>Lower</td><td>During recording</td></tr>
<tr><td><code>POST /transcribe</code></td><td>Final text for the whole recording</td><td>Best</td><td>After recording stops</td></tr>
</tbody>
</table>
<p>The preview is for responsiveness: words appear as the user talks. Its accuracy is lower because it only ever sees short windows of audio.
The final result from <code>/transcribe</code> sees the whole recording, so it should <strong>always replace</strong> the preview.
If you don't need the preview, skip the WebSocket and just POST the recording.</p>
<p>The examples below already use this server's address: <code>{{HTTP}}://{{HOST}}</code>.
Calling from a web page on another origin? This server sends no CORS headers, so serve your page from here or call it from your backend.</p>

<h3>1. Open the preview socket when recording starts</h3>
<p>Connect to <code>{{WS}}://{{HOST}}/ws/stream</code>.</p>
<p><strong>Send</strong> binary frames of raw audio: 16-bit signed PCM, little-endian, mono, 16,000 Hz, no header (not a WAV file).
Any chunk size works; 50–250 ms per frame is a good range.</p>
<p><strong>Receive</strong> JSON text messages, roughly once per second while there's new audio:</p>
<div class="code"><button class="copy" type="button">Copy</button><pre><code class="json">{"text": "tìm người mặc áo đỏ"}</code></pre></div>
<p>Each message is the <strong>whole preview so far</strong>, not just new words. Replace what you display; don't append.
During silence the text may stay the same or be <code>""</code>.</p>

<h3>2. Record the full audio at the same time</h3>
<p>Keep your own copy of the recording, in any format ffmpeg can decode (WAV, MP3, WebM/Opus from <code>MediaRecorder</code>, and so on).
Do not rebuild it from the preview stream. Use the original recording, ideally at the device's native quality.</p>

<h3>3. When recording stops</h3>
<ol>
<li><strong>Close the WebSocket.</strong> The server stops preview work for this session.</li>
<li><strong>Ignore any preview messages that arrive after this point.</strong> One may still be in flight and must not overwrite the final text.</li>
<li><strong>POST the recording</strong> to <code>/transcribe</code> as multipart form data with the field name <code>file</code>. The response is <code>{"text": "..."}</code>.</li>
<li><strong>Replace the preview</strong> with that text. <code>""</code> means no speech was detected.</li>
</ol>
<div class="code"><button class="copy" type="button">Copy</button><pre><code class="bash">curl -X POST {{HTTP}}://{{HOST}}/transcribe -F "file=@recording.mp3"
# self-signed HTTPS: add  --cacert cert.pem  (or -k to skip verification)</code></pre></div>

<h3>Example: browser (JavaScript)</h3>
<p>The essentials only. This page's own source (view source) is the full version, with auto-stop on silence and error handling.
Browsers allow microphone access only on <code>https://</code> or <code>localhost</code>.</p>
<div class="code"><button class="copy" type="button">Copy</button><pre><code class="js">let ws, recorder, chunks, audioCtx, processor, finished;

async function start(onPreview) {
  finished = false;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  // 1. Live preview socket
  ws = new WebSocket("{{WS}}://{{HOST}}/ws/stream");
  ws.onmessage = (e) =&gt; { if (!finished) onPreview(JSON.parse(e.data).text); };

  // 2. Full recording for the final pass
  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) =&gt; chunks.push(e.data);
  recorder.start();

  // Feed the preview: browser audio is Float32 at 44.1/48 kHz -&gt; PCM16 at 16 kHz
  audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(stream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (e) =&gt; {
    if (ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const ratio = audioCtx.sampleRate / 16000;
    const pcm = new Int16Array(Math.floor(input.length / ratio));
    for (let i = 0; i &lt; pcm.length; i++) {
      const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
      pcm[i] = s * 0x7fff;
    }
    ws.send(pcm.buffer);
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);
}

function stop() {
  // 3. Stop the preview first, and ignore late preview messages
  finished = true;
  ws.close();
  processor.disconnect();
  audioCtx.close();

  return new Promise((resolve) =&gt; {
    recorder.onstop = async () =&gt; {
      recorder.stream.getTracks().forEach((t) =&gt; t.stop());
      const form = new FormData();
      form.append("file", new Blob(chunks, { type: recorder.mimeType }), "recording.webm");
      const res = await fetch("{{HTTP}}://{{HOST}}/transcribe", { method: "POST", body: form });
      resolve((await res.json()).text); // 4. Final text: replaces the preview
    };
    recorder.stop();
  });
}

// Usage
// await start((text) =&gt; (box.textContent = text));   // grey preview
// box.textContent = await stop();                    // final text</code></pre></div>

<h3>Example: Python</h3>
<p>Streams a file instead of a microphone. The protocol is the same for a microphone: send PCM chunks as they're captured, then POST the full audio.
Needs <code>pip install httpx websockets</code> and ffmpeg on PATH.</p>
<div class="code"><button class="copy" type="button">Copy</button><pre><code class="python">import asyncio, json, ssl, subprocess
import httpx, websockets

WS_URL = "{{WS}}://{{HOST}}/ws/stream"
HTTP_URL = "{{HTTP}}://{{HOST}}/transcribe"
AUDIO = "recording.mp3"
# Server on HTTPS with a self-signed certificate (make_cert)? Point this at its cert.pem.
CAFILE = None
ssl_ctx = ssl.create_default_context(cafile=CAFILE) if CAFILE else None

async def main():
    # Decode to raw PCM16LE mono 16 kHz, the format /ws/stream expects.
    pcm = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", AUDIO, "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout

    async with websockets.connect(WS_URL, ssl=ssl_ctx) as ws:
        async def show_preview():
            async for msg in ws:
                print("preview:", json.loads(msg)["text"])
        preview = asyncio.create_task(show_preview())

        chunk = 3200                                # 100 ms of audio
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i:i + chunk])
            await asyncio.sleep(0.1)                # simulate real time
        preview.cancel()                            # stop listening, then close

    async with httpx.AsyncClient(timeout=300, verify=ssl_ctx or True) as client:
        with open(AUDIO, "rb") as f:
            res = await client.post(HTTP_URL, files={"file": f})
    print("final:", res.json()["text"])

asyncio.run(main())</code></pre></div>

<h3>Tuning and behavior</h3>
<p>Set these as server environment variables. In a portable build, add them as new lines in <code>config.ini</code>.</p>
<table>
<thead><tr><th>Setting</th><th>Default</th><th>Effect</th></tr></thead>
<tbody>
<tr><td><code>STREAM_UPDATE_SECONDS</code></td><td>1.0</td><td>The preview reruns once at least this much new audio has arrived. Lower = more frequent updates, more GPU/CPU load.</td></tr>
<tr><td><code>STREAM_WINDOW_SECONDS</code></td><td>8.0</td><td>The preview transcribes at most this much audio per pass, then locks that text in and starts a new window. Longer = better preview accuracy, slower passes.</td></tr>
</tbody>
</table>
<ul>
<li><strong>Silence returns <code>""</code>.</strong> Both endpoints run voice activity detection first, so silence or background noise produces an empty result, not invented text.</li>
<li><strong>Slow machines don't fall behind.</strong> Each socket runs at most one preview pass at a time. Audio that arrives during a pass is included in the next one, so updates just come less often.</li>
<li><strong>The preview and the final pass share the model.</strong> A preview pass still running at stop time can delay the final result by up to one pass. This is why you close the socket before POSTing.</li>
<li><strong>Preview text can change.</strong> The current window is re-transcribed as more audio arrives, so the last few words may be revised.</li>
</ul>
<p>Other endpoints: <code>GET /health</code> (device and model in use) and the interactive API docs at <a href="/docs">/docs</a>.</p>
</section>

<script>
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const previewCheckbox = document.getElementById("preview");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");

const VAD_SILENCE_RMS_THRESHOLD = 0.01;
const VAD_AUTO_STOP_SILENCE_SECONDS = 2.0;
const STREAM_SAMPLE_RATE = 16000;

let mediaRecorder, recordedChunks, stream, audioCtx, source, vadProcessor, previewSocket;
let speechDetected = false;
let silenceSeconds = 0;

function rms(float32) {
  let sum = 0;
  for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
  return Math.sqrt(sum / float32.length);
}

// Average-downsample to 16kHz and convert to PCM16, the format /ws/stream expects.
// Crude, but the preview is only for show.
function toPcm16(float32, inRate) {
  const ratio = inRate / STREAM_SAMPLE_RATE;
  const out = new Int16Array(Math.floor(float32.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), float32.length);
    let sum = 0;
    for (let j = start; j < end; j++) sum += float32[j];
    const s = Math.max(-1, Math.min(1, sum / Math.max(1, end - start)));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function openPreviewSocket() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/stream`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => {
    // Ignore anything arriving after Stop -- the final result owns the transcript then.
    if (ws !== previewSocket) return;
    const data = JSON.parse(e.data);
    transcriptEl.textContent = data.text;
  };
  return ws;
}

function closePreviewSocket() {
  if (previewSocket) previewSocket.close();
  previewSocket = null;
}

startBtn.onclick = async () => {
  startBtn.disabled = true;
  stopBtn.disabled = false;
  transcriptEl.textContent = "";
  transcriptEl.classList.toggle("preview", previewCheckbox.checked);
  statusEl.textContent = "requesting mic...";
  speechDetected = false;
  silenceSeconds = 0;

  try {
    // navigator.mediaDevices only exists in a secure context: https:// or localhost.
    if (!navigator.mediaDevices) throw new Error("microphone needs https:// (or localhost); this page is " + location.origin);
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    statusEl.textContent = "mic error: " + err.message;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    return;
  }

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
      transcriptEl.classList.remove("preview");
      statusEl.textContent = "done";
    } catch (err) {
      statusEl.textContent = "error: " + err.message;
    }
  };
  mediaRecorder.start();

  if (previewCheckbox.checked) previewSocket = openPreviewSocket();

  // Local silence detection decides *when* to stop recording; the same audio is also
  // streamed for the live preview. The final transcript always comes from the whole
  // recording, buffered client-side and sent as a single file once stopped.
  audioCtx = new AudioContext();
  source = audioCtx.createMediaStreamSource(stream);
  vadProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
  vadProcessor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    const chunkDuration = input.length / audioCtx.sampleRate;
    if (previewSocket && previewSocket.readyState === WebSocket.OPEN) {
      previewSocket.send(toPcm16(input, audioCtx.sampleRate).buffer);
    }
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

  closePreviewSocket();
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

// Developer guide: fill this server's address into the examples, and wire Copy buttons.
const httpScheme = location.protocol === "https:" ? "https" : "http";
const wsScheme = httpScheme === "https" ? "wss" : "ws";
document.querySelectorAll("#dev code").forEach((el) => {
  el.textContent = el.textContent
    .replaceAll("{{HOST}}", location.host)
    .replaceAll("{{HTTP}}", httpScheme)
    .replaceAll("{{WS}}", wsScheme);
});
document.querySelectorAll("#dev .copy").forEach((btn) => {
  btn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(btn.nextElementSibling.textContent);
      btn.textContent = "Copied";
    } catch {
      btn.textContent = "Select & copy manually";
    }
    setTimeout(() => (btn.textContent = "Copy"), 1500);
  };
});
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
cryptography
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

; HTTPS. Browsers only let the web page use the microphone over https:// (or on
; localhost). Run ./make_cert.sh once: it creates a self-signed cert.pem/key.pem and fills
; these in (browsers warn about the self-signed certificate once; accept it). Leave
; blank for plain http. Paths are relative to this folder.
SSL_CERTFILE=
SSL_KEYFILE=
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

SSL_ARGS=()
if [[ -n "${SSL_CERTFILE:-}" ]]; then
  SSL_ARGS=(--ssl-certfile "$SSL_CERTFILE" --ssl-keyfile "${SSL_KEYFILE:?SSL_KEYFILE must be set along with SSL_CERTFILE}")
fi

exec ./python/bin/python3 -m uvicorn main:app --app-dir src --host "$HOST" --port "$PORT" ${SSL_ARGS[@]+"${SSL_ARGS[@]}"}
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

cat > "$OUT_DIR/make_cert.sh" <<'CERTEOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
exec ./python/bin/python3 src/make_cert.py "$@"
CERTEOF
chmod +x "$OUT_DIR/make_cert.sh"

SIZE_MB=$(du -sm "$OUT_DIR" | cut -f1)
echo ""
echo "Done. Portable app built at: $OUT_DIR ($SIZE_MB MB)"
echo ""
echo "Run './run.sh' from that folder to start the server, or tar the whole folder"
echo "and copy it to another machine (no internet needed there)."
echo ""
echo "HTTPS (needed for the web page's mic when not on localhost):  ./make_cert.sh"
if [[ "$INCLUDE_CUDA" -eq 0 ]]; then
  echo "GPU support:   ./enable_gpu.sh        (downloads ~1.5GB of CUDA libraries)"
fi
if [[ "$STRIP_TORCH" -eq 0 ]]; then
  echo "Change model:  ./switch_model.sh medium   (currently: $MODEL)"
fi
