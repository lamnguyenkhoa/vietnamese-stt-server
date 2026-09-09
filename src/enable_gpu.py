"""Download the CUDA runtime libraries CTranslate2 needs for GPU mode into cuda/.

Portable builds are CPU-only by default; run this once, on a machine with an NVIDIA
GPU and internet access, to add GPU support (~1.5GB). Only the NVIDIA driver has to
be installed there -- these wheels carry the whole CUDA 12 / cuDNN 9 runtime, so no
CUDA toolkit is needed. Afterwards set DEVICE=auto (or cuda) in config.ini and
restart the server (run.bat on Windows, run.sh on Linux).
"""
import shutil
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import paths  # noqa: E402

CUDA_DIR = paths.CUDA_DIR
# Installed with --target into a scratch dir rather than into the embedded Python, so
# the wheels stay out of pip's package list and only the DLLs are kept.
STAGING_DIR = paths.APP_DIR / ".cuda-wheels"
PACKAGES = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]

if __name__ == "__main__":
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "--target", str(STAGING_DIR), *PACKAGES]
        )
        # Windows wheels ship .dll files; Linux wheels ship versioned .so files
        # (libcublas.so.12, ...) plus a couple of plain .so symlinks.
        pattern = "*.dll" if sys.platform == "win32" else "*.so*"
        CUDA_DIR.mkdir(exist_ok=True)
        copied = 0
        for lib in STAGING_DIR.rglob(pattern):
            if lib.is_file():
                shutil.copy2(lib, CUDA_DIR / lib.name)
                copied += 1
    finally:
        shutil.rmtree(STAGING_DIR, ignore_errors=True)

    total_mb = sum(f.stat().st_size for f in CUDA_DIR.glob(pattern)) / (1024 * 1024)
    print(f"Copied {copied} DLLs into {CUDA_DIR} ({total_mb:,.0f} MB total).")
    print("Set DEVICE=auto (or cuda) in config.ini, then restart the server.")
