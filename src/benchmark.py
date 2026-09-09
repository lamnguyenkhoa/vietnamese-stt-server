import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import ctranslate2
from faster_whisper import WhisperModel

from main import load_audio

MODEL_DIR = os.environ.get("MODEL_DIR", "models-ct2")
AUDIO_PATH = "data/ao-do-quan-den.mp3"

device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
compute_type = "float16" if device == "cuda" else "int8"


def transcribe(model, audio):
    start = time.perf_counter()
    segments, _ = model.transcribe(audio, language="vi", task="transcribe", beam_size=1)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    elapsed = time.perf_counter() - start
    return text, elapsed


def main():
    print(f"Device: {device} (compute_type={compute_type})")

    t0 = time.perf_counter()
    model = WhisperModel(MODEL_DIR, device=device, compute_type=compute_type)
    load_time = time.perf_counter() - t0
    print(f"Model load time: {load_time:.2f}s")

    with open(AUDIO_PATH, "rb") as f:
        raw_bytes = f.read()
    audio = load_audio(raw_bytes)

    for run in (1, 2):
        text, elapsed = transcribe(model, audio)
        label = "warm-up" if run == 1 else "second run"
        print(f"Run {run} ({label}): {elapsed:.2f}s -> {text!r}")


if __name__ == "__main__":
    main()
