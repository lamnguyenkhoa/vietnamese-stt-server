import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from download_model import REPO_ID

MODEL_DIR = os.environ.get("MODEL_DIR", REPO_ID)
SAMPLE_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["processor"] = WhisperProcessor.from_pretrained(MODEL_DIR)
    model_state["model"] = (
        WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
        .to(device)
        .eval()
    )
    yield
    model_state.clear()


app = FastAPI(lifespan=lifespan)


def load_audio(raw_bytes: bytes) -> "list[float]":
    """Decode arbitrary audio bytes to 16kHz mono PCM via ffmpeg."""
    src = tempfile.NamedTemporaryFile(suffix=".input", delete=False)
    dst_path = src.name + ".wav"
    try:
        src.write(raw_bytes)
        src.close()

        result = subprocess.run(
            [
                "ffmpeg",
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


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    raw_bytes = await file.read()
    audio = load_audio(raw_bytes)

    processor = model_state["processor"]
    model = model_state["model"]

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="vi", task="transcribe")

    text = processor.batch_decode(
        predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return {"text": text.strip()}


@app.get("/health")
async def health():
    return {"status": "ok", "device": device, "model": REPO_ID}
