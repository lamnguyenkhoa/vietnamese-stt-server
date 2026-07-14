FROM python:3.11-slim

WORKDIR /app

# System deps needed by soundfile (libsndfile) and audio handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch first. amd64 gets the CUDA 12.8 build (required for RTX 50-series /
# Blackwell sm_120 support; torch>=2.7.0 is the first release with Blackwell kernels).
# arm64 gets the CUDA 13.0 build (PyTorch's cu128 index has no arm64 wheels; cu130 does).
# NOTE: cu130 wheels need a driver whose CUDA ceiling (`nvidia-smi`) is >= 13.0 — see
# docs/torch-cuda-version.md. If your arm64 target's driver only covers 12.x, this
# build won't run on GPU; fall back to a plain `pip install torch` (CPU-only) instead.
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130; \
    else \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128; \
    fi

# Remaining deps from requirements.txt (fastapi, transformers, etc.)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
