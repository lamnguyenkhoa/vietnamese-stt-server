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
