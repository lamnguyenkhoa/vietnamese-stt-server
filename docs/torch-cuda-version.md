# GPU Setup (CTranslate2 / faster-whisper)

The app runs on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), backed by
CTranslate2, not PyTorch. This changes the GPU story from the old torch-based build:

- **No CUDA-version index selection needed.** Torch published separate wheel indexes
  per CUDA version (`cu118`, `cu121`, ... `cu128`) that had to match the target
  driver's ceiling. CTranslate2 ships a single pip wheel for all supported CUDA
  versions — nothing to pick or edit here.
- **But the host needs its own CUDA/cuDNN.** Torch's pip wheels bundle the CUDA
  runtime libraries they need. CTranslate2's wheel does not — it expects
  cuBLAS/cuDNN to already be available wherever it runs. A plain Python install or the
  Windows portable build (see [README.md](../README.md)) has neither by default.

## Enabling GPU acceleration

Pick one:

1. **`nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip packages**: install them and set
   `LD_LIBRARY_PATH` to point at their install location before starting uvicorn. This
   is faster-whisper's documented approach — see their
   [GPU setup docs](https://github.com/SYSTRAN/faster-whisper#gpu) for the exact
   packages/versions and the `LD_LIBRARY_PATH` snippet.
2. **System-installed CUDA Toolkit + cuDNN**: install them directly on the target
   machine, matching a CUDA/cuDNN version your NVIDIA driver supports (check the
   ceiling with `nvidia-smi`).

Neither has been verified against a live GPU in this repo — test before relying on
GPU acceleration in production. `DEVICE=auto` (the default) falls back to CPU
automatically if CUDA isn't usable, so a misconfigured GPU setup fails soft, not hard.

## CPU is the default, and it's fast

Unlike torch, CTranslate2 has real int8 kernels on CPU, so `DEVICE=cpu` with the
default `COMPUTE_TYPE=int8` is a genuinely fast, small deployment target — this is
what the portable Windows build is optimized for out of the box.
