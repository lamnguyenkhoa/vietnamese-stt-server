# Vietnamese STT Server

A FastAPI server that transcribes audio to text using [PhoWhisper-medium](https://huggingface.co/vinai/PhoWhisper-medium).

## Prerequisites

- Docker (with NVIDIA Container Toolkit if you want GPU acceleration)
- Model weights in `models/` (see below)

### Getting the model weights

Fetch the model files (config, tokenizer, and `pytorch_model.bin`, ~3GB) into `models/`:

```bash
python download_model.py
```

The build then copies `models/` into the image, so the resulting image is self-contained — no volume mount needed at runtime.

## Build the image

```bash
docker build -t khoalamphilong/vietnamese-stt-server .
```

## Run the image

GPU (requires NVIDIA Container Toolkit):

```bash
docker run --gpus all -p 8000:8000 vietnamese-stt-server
```

CPU only:

```bash
docker run -p 8000:8000 vietnamese-stt-server
```

## Run the server without Docker

```bash
python -m venv venv
./venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

Then start it with the provided script (uses local `models/` by default):

```bash
./run.sh
```

Or directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Requires `ffmpeg` and `libsndfile` installed on the host.

Then you can go to localhost:8000/docs to test it.

## API

- `POST /transcribe` — multipart file upload (`file`), returns `{"text": "..."}`
- `WS /ws/transcribe` — streaming transcription over a WebSocket (see below)
- `GET /health` — returns `{"status": "ok", "device": "cuda" | "cpu", "model": "..."}`

Example:

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@sample.wav"
```

### Streaming transcription (`/ws/transcribe`)

Whisper isn't a natively streaming model, so this endpoint buffers incoming audio and
transcribes it in fixed-size chunks (`STREAM_CHUNK_SECONDS`, default 3s) rather than
returning individual tokens as they're spoken. Expect a few seconds of latency per
result, and occasionally a word getting split across two chunks.

**Protocol:**

1. Open a WebSocket connection to `ws://<host>:8000/ws/transcribe`.
2. Stream raw audio as binary frames — **16-bit signed little-endian PCM, mono,
   16000 Hz** (no container/codec — do not send WAV/MP3/Opus bytes directly). If you're
   capturing from a browser mic, you'll need to downsample/convert to this format
   client-side first (see `static/index.html` for a working example).
3. Every time the server has buffered `STREAM_CHUNK_SECONDS` worth of audio, it runs
   inference on that chunk and sends back a JSON text frame:
   ```json
   {"text": "...", "final": false}
   ```
   Near-silent chunks (below `SILENCE_RMS_THRESHOLD`) are skipped rather than
   transcribed, to avoid Whisper hallucinating text from silence.
4. When you're done speaking, send a text frame with the literal string `"end"`. The
   server transcribes whatever's left in the buffer, sends a final message:
   ```json
   {"text": "...", "final": true}
   ```
   and closes the socket. (Simply closing the connection without sending `"end"`
   also works, but you lose the last partial chunk.)

**Try it in a browser:** start the server and open
`http://localhost:8000/static/index.html` — it captures your mic, streams audio to
`/ws/transcribe`, and renders the transcript live.

**Minimal Python client** (streaming from a WAV file for testing):

```python
import asyncio
import websockets
import soundfile as sf
import numpy as np

async def main():
    audio, sr = sf.read("sample.wav", dtype="float32")
    assert sr == 16000, "resample to 16kHz first"
    pcm16 = (audio * 32767).astype(np.int16).tobytes()

    async with websockets.connect("ws://localhost:8000/ws/transcribe") as ws:
        chunk_size = 4096
        for i in range(0, len(pcm16), chunk_size):
            await ws.send(pcm16[i : i + chunk_size])
        await ws.send("end")

        async for message in ws:
            print(message)

asyncio.run(main())
```

## Configuration

| Env var | Default | Description |
|-|-|
| `MODEL_DIR` | `vinai/PhoWhisper-medium` | Local path or HF repo ID for model weights |
