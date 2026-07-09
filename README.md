# Vietnamese STT Server

A FastAPI server that transcribes audio to text using [PhoWhisper-medium](https://huggingface.co/vinai/PhoWhisper-medium).

## Prerequisites

- Docker (with NVIDIA Container Toolkit if you want GPU acceleration)

Model weights are downloaded automatically during the image build (`docker build` runs `download_model.py`), so the resulting image is self-contained — no manual download or volume mount needed.

## Build the image

```bash
docker build -t vietnamese-stt-server .
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

Fetch the model weights into `models/`:

```bash
python download_model.py
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
- `GET /health` — returns `{"status": "ok", "device": "cuda" | "cpu"}`

Example:

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@sample.wav"
```

## Configuration

| Env var | Default | Description |
|-|-|
| `MODEL_DIR` | `vinai/PhoWhisper-medium` | Local path or HF repo ID for model weights |
