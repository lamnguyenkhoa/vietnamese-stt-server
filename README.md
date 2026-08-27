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
uvicorn main:app --host 0.0.0.0 --port 8123
```

Then you can go to localhost:8123/docs to test it.

**GPU note:** `faster-whisper`'s CTranslate2 backend doesn't bundle its own CUDA
runtime the way PyTorch's pip wheels do, so GPU acceleration needs cuBLAS (CUDA 12)
and cuDNN 9 available to the process on top of a compatible NVIDIA driver (see
[docs/torch-cuda-version.md](docs/torch-cuda-version.md)). For a dev checkout, either
install the CUDA toolkit or just `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
into the venv; the portable build ships them (see below). `DEVICE=auto` (the default)
falls back to CPU automatically if the runtime can't be loaded.

## Portable Windows deployment

[build_portable.ps1](build_portable.ps1) packages the app into a self-contained
folder: a standalone Python (the official embeddable distribution, not a venv — no
dependency on any Python already installed on the target machine), faster-whisper and
all deps, a static ffmpeg build, plus the app code and the int8-quantized model.
Nothing needs to be pre-installed on the target server. Inference never touches torch,
so nothing multi-GB is required to *run* the server; the default build is CPU-only and
keeps CPU torch installed purely as the model converter (see
[Changing the model](#changing-the-model-after-a-build) below).

The script is fully self-contained — the app source (`main.py`, `download_model.py`,
`convert_ct2.py`, `switch_model.py`, `enable_gpu.py`, `requirements.txt`,
`static/index.html`) is embedded directly in it,
so it does **not** need a repo checkout or pre-downloaded model weights. You can copy
just this one file anywhere and run it there.

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
PORT=8123
CUDA_VISIBLE_DEVICES=
```

### Adding GPU support after a build

The build is **CPU-only by default** — CPU int8 is what this package is best optimized
for, and the CUDA runtime is ~1.5GB. To enable the GPU on the target machine, run
`enable_gpu.bat` from the output folder once (needs internet). It downloads the CUDA
runtime CTranslate2 needs (cuBLAS + cuDNN 9) into the `cuda\` folder, which `run.bat`
already puts on the DLL search path, so the only other requirement is an NVIDIA driver
new enough for CUDA 12 — no CUDA toolkit install. `DEVICE=auto` then picks up the GPU
on the next start, and still falls back to CPU if the runtime can't be loaded.

If the target machine has no internet access, build with `-IncludeCuda` instead to
bundle the same DLLs up front.

### Changing the model after a build

The shipped checkpoint is `vinai/PhoWhisper-small`. Pick a different one at build time
with `-Model medium`, or swap it later from the output folder:

```powershell
.\switch_model.bat medium              # tiny | base | small | medium | large
.\switch_model.bat vinai/PhoWhisper-large
.\switch_model.bat                     # prints the current model
```

That re-downloads the raw weights from Hugging Face, re-converts them to CTranslate2
int8 in `models-ct2\`, and records the choice in `model_id.txt` (which `/health`
reports). Restart `run.bat` afterwards. This is why CPU torch + transformers stay in
the output — the CTranslate2 converter reads HF checkpoints through them, on CPU only.
Build with `-StripTorch` to drop them (~1GB smaller) if the model never needs to
change; `switch_model.bat` is then omitted from the output.

`CUDA_VISIBLE_DEVICES` is useful on a multi-GPU machine shared with other processes:
set it to `0` or `1` to pin the server to a specific, less-contended GPU.

`run.bat` reads `config.ini` on every start.

### Building from the local checkout (offline)

[build_local.ps1](build_local.ps1) produces the same output folder, but assembled from
what is already on this machine instead of from the internet — useful for iterating on
a deploy, or for building on a machine with no (or slow) internet:

```powershell
.\build_local.ps1            # -> dist\vietnamese-stt-server-local\
.\build_local.ps1 -Zip       # also writes dist\vietnamese-stt-server-local.zip
.\build_local.ps1 -Offline   # fail instead of downloading anything
```

It takes the app code from the working tree, the model from your local `models-ct2\`
(no Hugging Face download, no re-conversion), `ffmpeg.exe` from `PATH` (override with
`-FfmpegExe`), and the dependencies out of `venv\Lib\site-packages` — so the build is
seconds, not minutes, and ships exactly the package versions you tested against.
[collect_deps.py](collect_deps.py) resolves the dependency closure of
`requirements.txt` from the venv's installed metadata, so dev-only extras that also
live in the venv (torch, transformers, …) are left out of the shipped folder.

The one thing that can't come from the working tree is the standalone Python runtime —
a venv has no interpreter to ship. The embeddable distribution is downloaded once and
cached in `vendor\`, after which every build (and `-Offline`) works with no network.
The shipped interpreter must match the venv's Python X.Y, since the copied wheels are
built against that ABI; the script defaults to the venv's exact version.

The build finishes by importing the whole stack with the bundled interpreter, so a
missing dependency fails the build rather than the first run on the server.

## API

- `POST /transcribe` — multipart file upload (`file`), returns `{"text": "..."}`
- `WS /ws/transcribe` — streaming transcription over a WebSocket (see below)
- `GET /health` — returns `{"status": "ok", "device": "cuda" | "cpu", "model": "..."}`

Example:

```bash
curl -X POST http://localhost:8123/transcribe -F "file=@sample.wav"
```

### Streaming transcription (`/ws/transcribe`)

Whisper isn't a natively streaming model, so this endpoint buffers incoming audio and
transcribes it in fixed-size chunks (`STREAM_CHUNK_SECONDS`, default 3s) rather than
returning individual tokens as they're spoken. Expect a few seconds of latency per
result, and occasionally a word getting split across two chunks.

**Protocol:**

1. Open a WebSocket connection to `ws://<host>:8123/ws/transcribe`.
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
`http://localhost:8123/static/index.html` — it captures your mic, streams audio to
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

    async with websockets.connect("ws://localhost:8123/ws/transcribe") as ws:
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
