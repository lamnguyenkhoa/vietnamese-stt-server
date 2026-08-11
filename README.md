# Vietnamese STT Server

A FastAPI server that transcribes audio to text using [PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small).

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

This builds for your host architecture. amd64 gets a CUDA 12.8 PyTorch build; arm64 gets
a CUDA 13.0 build (PyTorch's `cu128` index has no arm64 wheels, but `cu130` does).

The arm64 GPU build only works if the target machine's driver supports CUDA 13.0+
(check with `nvidia-smi` — see [docs/torch-cuda-version.md](docs/torch-cuda-version.md)
for how to read the "CUDA Version" ceiling). If your arm64 target's driver only covers
12.x, edit the `arm64` branch in the [Dockerfile](Dockerfile) to `pip install torch`
(no index URL) for a CPU-only build instead.

### Building for arm64 / multi-arch

Requires [Docker Buildx](https://docs.docker.com/build/buildx/) (bundled with modern
Docker Desktop; on Linux you may need `docker buildx create --use` once).

Build an arm64 image while on an amd64 host (or vice versa) via emulation:

```bash
docker buildx build --platform linux/arm64 -t khoalamphilong/vietnamese-stt-server:arm64 --load .
```

Or build and push a single multi-arch manifest covering both architectures at once
(requires pushing to a registry — `--load` doesn't support multi-platform images):

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t khoalamphilong/vietnamese-stt-server:latest --push .
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

## Portable Windows deployment (no Docker)

For Windows servers where Docker isn't available/allowed, [build_portable.ps1](build_portable.ps1)
packages the app into a self-contained folder: a standalone Python (the official
embeddable distribution, not a venv — no dependency on any Python already installed on
the target machine), torch and all deps, a static ffmpeg build, plus the app code
and model weights. Nothing needs to be pre-installed on the target server.

By default this installs CPU-only torch, keeping the output folder small (a few
hundred MB) and easy to copy between machines. Pass `-Cuda` to install CUDA 12.8 torch
instead for GPU acceleration — this adds ~4GB to the output.

The script is fully self-contained — the app source (`main.py`, `download_model.py`,
`requirements.txt`, `static/index.html`) is embedded directly in it, so it does **not**
need a repo checkout or pre-downloaded model weights. You can copy just this one file
anywhere and run it there.

Run this from **PowerShell** (not Git Bash/WSL — the script uses PowerShell syntax):

```powershell
.\build_portable.ps1
.\build_portable.ps1 -Cuda   # GPU acceleration instead of CPU-only (~4GB larger)
```

Run it directly **on the target server** if that machine has internet access — no need
to build on a dev machine and copy a zip over. It downloads the Python embeddable zip,
pip, model weights from Hugging Face, and a static ffmpeg build from
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds). If the target server has no
internet access, run it on a dev machine instead, then zip and copy the output folder
over.

This produces `dist\vietnamese-stt-server-portable\`. Run `run.bat` from that folder (or
copy the whole folder to another machine first). It sets `MODEL_DIR` and `FFMPEG_BIN` to
point at the bundled copies and starts uvicorn — same API as the Docker deployment.

To change the host/port after building (e.g. on the target server, no rebuild needed),
edit `config.bat` in the output folder:

```bat
set HOST=0.0.0.0
set PORT=8000
set CUDA_VISIBLE_DEVICES=
```

`CUDA_VISIBLE_DEVICES` is useful on a multi-GPU machine shared with other processes:
set it to `0` or `1` to pin the server to a specific, less-contended GPU. If you see
`torch.AcceleratorError: CUDA-capable device(s) is/are busy or unavailable` at startup
on a machine with several other CUDA processes already running, check `nvidia-smi` for
which GPU has fewer processes/contexts and pin to that one.

`run.bat` sources it on every start.

With `-Cuda`, the build script installs the `cu128` torch build (matching the amd64
Docker image). If the target server's driver has a different CUDA ceiling, edit the
`--index-url` in `build_portable.ps1` — see
[docs/torch-cuda-version.md](docs/torch-cuda-version.md).

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
| `MODEL_DIR` | `vinai/PhoWhisper-small` | Local path or HF repo ID for model weights |
