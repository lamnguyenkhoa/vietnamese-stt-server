import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import ctranslate2
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from download_model import REPO_ID

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
    return {"status": "ok", "device": device, "model": REPO_ID}


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
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    if args.device:
        os.environ["DEVICE"] = args.device

    uvicorn.run(app, host=args.host, port=args.port)
