import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from main import SAMPLE_RATE, load_audio

MODEL_DIR = os.environ.get("MODEL_DIR", "vinai/PhoWhisper-medium")
AUDIO_PATH = "data/ao-do-quan-den.mp3"

device = "cuda" if torch.cuda.is_available() else "cpu"


def transcribe(processor, model, audio):
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    elapsed = time.perf_counter() - start

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip(), elapsed


def main():
    print(f"Device: {device}")

    t0 = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(device).eval()
    load_time = time.perf_counter() - t0
    print(f"Model load time: {load_time:.2f}s")

    with open(AUDIO_PATH, "rb") as f:
        raw_bytes = f.read()
    audio = load_audio(raw_bytes)

    for run in (1, 2):
        text, elapsed = transcribe(processor, model, audio)
        label = "warm-up" if run == 1 else "second run"
        print(f"Run {run} ({label}): {elapsed:.2f}s -> {text!r}")


if __name__ == "__main__":
    main()
