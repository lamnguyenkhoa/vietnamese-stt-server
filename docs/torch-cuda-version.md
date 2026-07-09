# Picking/Downgrading the Torch CUDA Version

The [Dockerfile](../Dockerfile) currently installs torch from the `cu128` index (needed for RTX 50-series / Blackwell GPUs, torch >= 2.7.0). If deploying to a machine with an older GPU/driver, follow these steps.

## 1. Check the target's max supported CUDA version

On the target machine, run:

```
nvidia-smi
```

The "CUDA Version" shown in the header is the *maximum* CUDA runtime that driver supports (not what's installed — just the ceiling).

## 2. Match a torch build to that ceiling

PyTorch publishes separate wheel indexes per CUDA version. Pick the highest one <= the target's ceiling:

| Target driver's CUDA Version | Use index-url |
|-|-|
| 12.8+ | `https://download.pytorch.org/whl/cu128` |
| 12.6+ | `https://download.pytorch.org/whl/cu126` |
| 12.4+ | `https://download.pytorch.org/whl/cu124` |
| 12.1+ | `https://download.pytorch.org/whl/cu121` |
| 11.8+ | `https://download.pytorch.org/whl/cu118` |
| no NVIDIA GPU | `https://download.pytorch.org/whl/cpu` |

Full matrix: https://pytorch.org/get-started/previous-versions/ — cross-check torch version vs. CUDA version there before picking, since not every torch release ships every cu index.

## 3. Edit the Dockerfile

Change the install line in [Dockerfile](../Dockerfile):

```dockerfile
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
```

(swap `cu121` for whichever matches step 2; drop the CUDA index entirely and use `.../whl/cpu` for CPU-only deploys).

## 4. GPU architecture caveat

Older torch/CUDA builds drop support for newer GPU architectures — e.g. `cu121` wheels don't include Blackwell (RTX 50-series) kernels at all. This only matters if one image needs to run on both an RTX 50-series box and an older one; if the older-CUDA target also has an older GPU, there's no conflict.

## 5. Rebuild and verify

```
docker build -t vietnamese-stt .
docker run --gpus all vietnamese-stt python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If it prints `False` or errors with "no kernel image," the torch/CUDA build doesn't match the driver or GPU — recheck steps 1-2.

## Host prerequisites (any CUDA version)

1. NVIDIA driver version high enough to cover the CUDA version chosen in step 2.
2. NVIDIA Container Toolkit installed (`nvidia-ctk` / `nvidia-docker2`) so `--gpus all` is available to Docker.
