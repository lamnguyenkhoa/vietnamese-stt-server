"""Filesystem layout shared by every module in src/.

The Python code lives in src/, while everything a user or the launcher touches sits
one level up in the app folder: config.ini, the .bat wrappers, models-ct2/, cuda/,
bin/, static/. Paths are resolved from this file rather than the working directory,
so the scripts behave the same whether run from the app folder, from src/, or by
run.bat.
"""
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
APP_DIR = SRC_DIR.parent

STATIC_DIR = APP_DIR / "static"
BIN_DIR = APP_DIR / "bin"
CUDA_DIR = APP_DIR / "cuda"
# Raw Hugging Face weights (converter input; deleted after a successful conversion).
MODELS_DIR = APP_DIR / "models"
# CTranslate2 model the server actually loads.
MODEL_CT2_DIR = APP_DIR / "models-ct2"
# Which checkpoint this install serves; written by switch_model.py.
MODEL_ID_FILE = APP_DIR / "model_id.txt"
