# Vietnamese STT Server

A FastAPI server that transcribes audio to text using [PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small).

The app runs on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2),
not PyTorch — no multi-GB torch/CUDA install, and the model ships int8-quantized
(~240MB instead of ~1GB+ in fp32/fp16). CPU inference is fast enough to use directly;
GPU acceleration is opt-in where available (see below).

## Prerequisites

- Python 3.11+ (or use the portable Windows build below, which bundles its own)
- `ffmpeg` and `libsndfile` installed on the host
- Model weights, converted to CTranslate2 format (see below)

### Getting the model weights

Fetch PhoWhisper-small's raw HF files into `models/`, then convert them to the
CTranslate2 int8 format the app actually runs on:

```bash
python download_model.py
python convert_ct2.py
```

`convert_ct2.py` needs `transformers` and `torch` installed (only for this one-time
conversion — `pip install transformers torch`; CPU-only torch is fine even for a GPU
deployment, since it's only used to read the weights, not run them). Neither is needed
at runtime. This produces `models-ct2/` (~240MB).

## Run the server

```bash
python -m venv venv
./venv/Scripts/activate      # Windows
pip install -r requirements.txt
python download_model.py
python convert_ct2.py        # needs `pip install transformers torch` first
```

Then start it:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then you can go to localhost:8000/docs to test it.

**GPU note:** `faster-whisper`'s CTranslate2 backend doesn't bundle its own CUDA
runtime the way PyTorch's pip wheels do, so GPU acceleration requires a compatible
NVIDIA driver plus cuBLAS/cuDNN already available on the host (see
[docs/torch-cuda-version.md](docs/torch-cuda-version.md)). `DEVICE=auto` (the default)
falls back to CPU automatically if that's not the case.

## Portable Windows deployment

[build_portable.ps1](build_portable.ps1) packages the app into a self-contained
folder: a standalone Python (the official embeddable distribution, not a venv — no
dependency on any Python already installed on the target machine), faster-whisper and
all deps, a static ffmpeg build, plus the app code and the int8-quantized model.
Nothing needs to be pre-installed on the target server. Since there's no torch runtime
dependency, the output folder is small (~700MB, mostly Python + the model) and easy to
copy between machines.

The script is fully self-contained — the app source (`main.py`, `download_model.py`,
`convert_ct2.py`, `requirements.txt`, `static/index.html`) is embedded directly in it,
so it does **not** need a repo checkout or pre-downloaded model weights. You can copy
just this one file anywhere and run it there. It does temporarily install CPU torch +
transformers mid-build to do the one-time model conversion, then uninstalls them
before finishing — they never end up in the shipped output.

Run this from **PowerShell** (not Git Bash/WSL — the script uses PowerShell syntax):

```powershell
.\build_portable.ps1
```

Run it directly **on the target server** if that machine has internet access — no need
to build on a dev machine and copy a zip over. It downloads the Python embeddable zip,
pip, model weights from Hugging Face, and a static ffmpeg build from
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds). If the target server has no
internet access, run it on a dev machine instead, then zip and copy the output folder
over.

This produces `dist\vietnamese-stt-server-portable\`. Run `run.bat` from that folder (or
copy the whole folder to another machine first). It sets `MODEL_DIR` and `FFMPEG_BIN` to
point at the bundled copies and starts uvicorn.

To change the host/port after building (e.g. on the target server, no rebuild needed),
edit `config.ini` in the output folder:

```ini
HOST=0.0.0.0
PORT=8000
CUDA_VISIBLE_DEVICES=
```

By default `DEVICE=auto` in `config.ini`, which uses GPU only if the target machine
already has a compatible NVIDIA driver and CUDA/cuDNN runtime available — the portable
build itself doesn't bundle a CUDA runtime (see the GPU note above: CTranslate2's pip
wheel doesn't ship one). CPU (int8) is what this build is optimized for.

`CUDA_VISIBLE_DEVICES` is useful on a multi-GPU machine shared with other processes:
set it to `0` or `1` to pin the server to a specific, less-contended GPU.

`run.bat` reads `config.ini` on every start.

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
| `MODEL_DIR` | `models-ct2` | Path to the CTranslate2-format model directory (see `convert_ct2.py`) |
| `DEVICE` | `auto` | `cuda`, `cpu`, or `auto` to use GPU when available |
| `COMPUTE_TYPE` | `int8` on CPU, `float16` on GPU | CTranslate2 compute type, e.g. `int8`, `int8_float16`, `float16`, `float32` |
