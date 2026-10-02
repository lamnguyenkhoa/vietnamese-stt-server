# Vietnamese STT Server

A FastAPI server that transcribes audio to text using [PhoWhisper](https://huggingface.co/vinai)
(medium by default — swap it with [switch_model](#changing-the-model-after-a-build), tiny/base/small/large also work).

The app runs on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2),
not PyTorch — no multi-GB torch/CUDA install at runtime, and the model ships
int8-quantized (~770MB for medium, ~240MB for small, vs 2x+ that in fp32/fp16). CPU
inference is fast enough to use directly; GPU acceleration is opt-in where available
(see below).

## Layout

Python modules live in `src/`; everything a user opens — the launchers, `config.ini`,
`requirements.txt` — sits in the folder above them, and so do the data directories
(`models-ct2/`, `static/`, and in a portable build `python/`, `bin/`, `cuda/`). Paths
are resolved from [src/paths.py](src/paths.py), not from the working directory, so the
scripts behave the same wherever you run them from.

## Prerequisites

- Python 3.11+ (or use the portable Windows build below, which bundles its own)
- `ffmpeg` and `libsndfile` installed on the host
- Model weights, converted to CTranslate2 format (see below)

### Getting the model weights

Fetch PhoWhisper-medium's raw HF files into `models/` (set `MODEL_REPO` to fetch a
different size/checkpoint instead — see [download_model.py](src/download_model.py)),
then convert them to the CTranslate2 int8 format the app actually runs on:

```bash
python src/download_model.py
python src/convert_ct2.py
```

`src/convert_ct2.py` needs `transformers` and `torch` installed (only for this one-time
conversion — `pip install transformers torch`; CPU-only torch is fine even for a GPU
deployment, since it's only used to read the weights, not run them). Neither is needed
at runtime. This produces `models-ct2/` (~770MB for medium, ~240MB for small).

## Run the server

```bash
python -m venv venv
./venv/Scripts/activate      # Windows
pip install -r requirements.txt
python src/download_model.py
python src/convert_ct2.py    # needs `pip install transformers torch` first
```

Then start it:

```bash
uvicorn main:app --app-dir src --host 0.0.0.0 --port 8123
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

The script is fully self-contained — the app source (everything in `src/`, plus
`requirements.txt` and `static/index.html`) is embedded directly in it, so it does
**not** need a repo checkout or pre-downloaded model weights. You can copy just this
one file anywhere and run it there.

Run this from **PowerShell** (not Git Bash/WSL — the script uses PowerShell syntax):

```powershell
.\build_portable.ps1
.\build_portable.ps1 -OutDir C:\deploy\vietnamese-stt-server
.\build_portable.ps1 -Model medium -IncludeCuda
```

Run it directly **on the target server** if that machine has internet access — no need
to build on a dev machine and copy a zip over. It downloads the Python embeddable zip,
pip, model weights from Hugging Face, and a static ffmpeg build from
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds). If the target server has no
internet access, run it on a dev machine instead, then zip and copy the output folder
over.

This produces `dist\vietnamese-stt-server-portable\`, laid out so the only things in
its root are the ones you might want to open:

```
run.bat            start the server
switch_model.bat   change the checkpoint
enable_gpu.bat     add GPU support
config.ini         host/port/device
model_id.txt       which checkpoint is installed
src\                the Python modules
python\ bin\ cuda\ models-ct2\ static\
```

Run `run.bat` from that folder (or copy the whole folder to another machine first). It
sets `MODEL_DIR` and `FFMPEG_BIN` to point at the bundled copies and starts uvicorn.

To change the host/port after building (e.g. on the target server, no rebuild needed),
edit `config.ini` in the output folder:

```ini
HOST=0.0.0.0
PORT=8000
CUDA_VISIBLE_DEVICES=
```

`CUDA_VISIBLE_DEVICES` is useful on a multi-GPU machine shared with other processes:
set it to `0` or `1` to pin the server to a specific, less-contended GPU.

`run.bat` reads `config.ini` on every start.

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

The shipped checkpoint is `vinai/PhoWhisper-medium` (set with `-Model` at build time;
`tiny`/`base`/`small`/`large` also work, as does any HF repo id). Swap it later from
the output folder:

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

### Building from the local checkout (offline)

[build_local.ps1](build_local.ps1) produces the same output folder, but assembled from
what is already on this machine instead of from the internet — useful for iterating on
a deploy, or for building on a machine with no (or slow) internet:

```powershell
.\build_local.ps1            # -> dist\vietnamese-stt-server-local\
.\build_local.ps1 -Zip       # also writes dist\vietnamese-stt-server-local.zip
.\build_local.ps1 -Offline   # fail instead of downloading anything
```

It takes the app code from the working tree (`src\` modules), the model from your
local `models-ct2\` (no Hugging Face download, no re-conversion), `ffmpeg.exe` from
`PATH` (override with `-FfmpegExe`), and the dependencies out of `venv\Lib\site-packages`
— so the build is seconds, not minutes, and ships exactly the package versions you
tested against. [src/collect_deps.py](src/collect_deps.py) resolves the dependency
closure of `requirements.txt` from the venv's installed metadata, so dev-only extras
that also live in the venv (torch, transformers, …) are left out of the shipped folder.

The one thing that can't come from the working tree is the standalone Python runtime —
a venv has no interpreter to ship. The embeddable distribution is downloaded once and
cached in `vendor\`, after which every build (and `-Offline`) works with no network.
The shipped interpreter must match the venv's Python X.Y, since the copied wheels are
built against that ABI; the script defaults to the venv's exact version.

The build finishes by importing the whole stack with the bundled interpreter, so a
missing dependency fails the build rather than the first run on the server.

## Portable Linux deployment

[build_portable.sh](build_portable.sh) is the Linux counterpart of
`build_portable.ps1`: same self-contained design (the app source is embedded in the
script, so you can copy just this one file to a server with no repo checkout), same
CPU-only-by-default / switchable GPU and model, same output layout (`src/` for the
Python modules, everything else in the root). Run it on Linux itself (or WSL with a
native Linux filesystem -- it extracts an archive containing symlinks, which a
Windows-mounted path can't create).

Instead of the Windows embeddable zip, it downloads a standalone CPython build from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone) (the
same project `uv`/`rye` use) -- a real interpreter with no dependency on whatever
Python the target machine has or lacks. ffmpeg comes from a static build at
[johnvansickle.com](https://johnvansickle.com/ffmpeg/), and GPU support from the same
`nvidia-cublas-cu12`/`nvidia-cudnn-cu12` wheels as the Windows build, just with `.so`
files and `LD_LIBRARY_PATH` instead of `.dll` files and the DLL search path.

```bash
./build_portable.sh
./build_portable.sh --out-dir /opt/vietnamese-stt-server
./build_portable.sh --model medium --include-cuda
```

This produces `dist/vietnamese-stt-server-portable-linux/`, laid out the same way as
the Windows build:

```
run.sh              start the server
switch_model.sh      change the checkpoint
enable_gpu.sh        add GPU support
config.ini           host/port/device
model_id.txt         which checkpoint is installed
src/                 the Python modules
python/ bin/ cuda/ models-ct2/ static/
```

`--include-cuda` and `--strip-torch` mirror `-IncludeCuda`/`-StripTorch` on the
Windows script (see [Adding GPU support](#adding-gpu-support-after-a-build) and
[Changing the model](#changing-the-model-after-a-build) above -- the same trade-offs
apply here; the default checkpoint is likewise `vinai/PhoWhisper-medium`).

There's no Linux equivalent of `build_local.ps1` yet; contributions welcome.

## API

- `POST /transcribe` — multipart file upload (`file`), returns `{"text": "..."}`
- `WS /ws/stream` — live *preview* transcription. Send raw PCM16LE mono 16kHz audio as
  binary frames; receive `{"text": "..."}` (the whole preview so far, replacing the
  previous one) about once a second. Close the socket when done. The preview is
  produced from short windows of audio, so it's less accurate. Use it for instant
  feedback while the user speaks, then POST the full recording to `/transcribe` and
  replace the preview with that result.
- `GET /health` — returns `{"status": "ok", "device": "cuda" | "cpu", "model": "..."}`

Example:

```bash
curl -X POST http://localhost:8123/transcribe -F "file=@sample.wav"
```

**Try it in a browser:** start the server and open
`http://localhost:8123/static/index.html` — it records from your mic, stops on a
click or after ~2s of silence, and posts the whole recording to `/transcribe`. With
"Live preview" checked, it also streams to `/ws/stream` while you speak, showing a grey
preview that the final `/transcribe` result replaces.

`/transcribe` decodes the audio with ffmpeg, so any container
or codec ffmpeg understands works (WAV, MP3, Opus, the browser's WebM, …).

## Configuration

| Env var | Default | Description |
|-|-|
| `MODEL_DIR` | `models-ct2` | Path to the CTranslate2-format model directory (see `src/convert_ct2.py`) |
| `MODEL_REPO` | `vinai/PhoWhisper-medium` | Checkpoint `src/download_model.py` fetches; a size shortcut (`tiny`/`base`/`small`/`medium`/`large`) or any HF repo id. Overrides `model_id.txt` |
| `DEVICE` | `auto` | `cuda`, `cpu`, or `auto` to use GPU when available |
| `COMPUTE_TYPE` | `int8` on CPU, `float16` on GPU | CTranslate2 compute type, e.g. `int8`, `int8_float16`, `float16`, `float32` |
| `STREAM_UPDATE_SECONDS` | `1.0` | `/ws/stream`: minimum amount of new audio before the preview is re-run |
| `STREAM_WINDOW_SECONDS` | `8.0` | `/ws/stream`: the in-progress window is committed and a new one starts once it reaches this length |
