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

MODEL_DIR = os.environ.get("MODEL_DIR", REPO_ID)
SAMPLE_RATE = 16000
STREAM_CHUNK_SECONDS = 3.0
SILENCE_RMS_THRESHOLD = 0.01

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

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

            if "bytes" in message and message["bytes"] is not None:
                pcm16 = np.frombuffer(message["bytes"], dtype=np.int16)
                buffer = np.concatenate([buffer, pcm16.astype(np.float32) / 32768.0])

                if len(buffer) >= chunk_samples:
                    if is_silent(buffer):
                        text = ""
                    else:
                        text = transcribe_array(buffer)
                    buffer = np.empty(0, dtype=np.float32)
                    if text:
                        await websocket.send_json({"text": text, "final": False})

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
