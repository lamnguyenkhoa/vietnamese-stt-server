"""Fetch a PhoWhisper checkpoint's raw HF files (config, tokenizer, weights) into models/.

Which checkpoint gets fetched is not hardcoded: it comes from $MODEL_REPO, else from
model_id.txt (written by switch_model.py), else the default below -- so an already
built install can be moved between small/medium without editing any source.

This is a staging download only -- the app itself runs on the CTranslate2 format
produced by convert_ct2.py from these files, not on this directory directly.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

import paths

DEFAULT_REPO_ID = "vinai/PhoWhisper-medium"
MODEL_ID_FILE = paths.MODEL_ID_FILE

# Short names so callers can say "medium" instead of the full repo path. Any other
# value is passed through to Hugging Face as-is.
ALIASES = {
    "tiny": "vinai/PhoWhisper-tiny",
    "base": "vinai/PhoWhisper-base",
    "small": "vinai/PhoWhisper-small",
    "medium": "vinai/PhoWhisper-medium",
    "large": "vinai/PhoWhisper-large",
}

# Weights only in .bin form: the *.safetensors siblings would double the download.
ALLOW_PATTERNS = ["*.json", "*.txt", "vocab.*", "merges.txt", "*.model", "pytorch_model.bin"]


def resolve_repo_id(name: str) -> str:
    name = (name or "").strip()
    return ALIASES.get(name.lower(), name)


def current_repo_id() -> str:
    """The checkpoint this install is currently set to serve."""
    env = resolve_repo_id(os.environ.get("MODEL_REPO", ""))
    if env:
        return env
    try:
        recorded = resolve_repo_id(MODEL_ID_FILE.read_text(encoding="utf-8"))
    except OSError:
        recorded = ""
    return recorded or DEFAULT_REPO_ID


def download(repo_id: str | None = None) -> str:
    repo_id = resolve_repo_id(repo_id) if repo_id else current_repo_id()
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(paths.MODELS_DIR),
        allow_patterns=ALLOW_PATTERNS,
    )
    return repo_id


REPO_ID = current_repo_id()

if __name__ == "__main__":
    fetched = download(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Done. {fetched} files are now in models/. Run convert_ct2.py next.")
