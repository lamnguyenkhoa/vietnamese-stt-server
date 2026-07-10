FROM python:3.11-slim

WORKDIR /app

# System deps needed by soundfile (libsndfile) and audio handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch built against CUDA 12.8 first (required for RTX 50-series / Blackwell sm_120 support).
# torch>=2.7.0 is the first release with Blackwell kernels.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# Remaining deps from requirements.txt (fastapi, transformers, etc.)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
