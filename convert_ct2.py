"""Convert models/ (raw HF PhoWhisper weights, from download_model.py) into
CTranslate2 format in models-ct2/, quantized to int8. This is what main.py actually
loads at runtime via faster-whisper.

Requires transformers and torch installed -- only for this conversion, not at
runtime. The portable build keeps them installed so switch_model.py can re-run this
against a different checkpoint later; uninstall them for a slimmer environment if you
never intend to switch models.
"""
from pathlib import Path

from ctranslate2.converters import TransformersConverter

BASE_DIR = Path(__file__).resolve().parent

# Auxiliary files the HF repo doesn't put in a single weights blob: tokenizer,
# generation defaults, and PhoWhisper's Vietnamese text normalizer.
COPY_FILES = [
    "tokenizer.json",
    "preprocessor_config.json",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]


def convert(model_dir: str | Path | None = None, out_dir: str | Path | None = None) -> Path:
    model_dir = Path(model_dir) if model_dir else BASE_DIR / "models"
    out_dir = Path(out_dir) if out_dir else BASE_DIR / "models-ct2"
    converter = TransformersConverter(str(model_dir), copy_files=COPY_FILES)
    converter.convert(str(out_dir), quantization="int8", force=True)
    return out_dir


if __name__ == "__main__":
    print(f"Done. CTranslate2 model is now in {convert()}")
