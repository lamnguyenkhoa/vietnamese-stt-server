"""Fetch PhoWhisper-small's raw HF files (config, tokenizer, and weights) into models/.

This is a staging download only -- the app itself runs on the CTranslate2 format
produced by convert_ct2.py from these files, not on this directory directly.
"""
from huggingface_hub import snapshot_download

REPO_ID = "vinai/PhoWhisper-small"

if __name__ == "__main__":
    snapshot_download(
        repo_id=REPO_ID,
        local_dir="models",
        allow_patterns=["*.json", "*.txt", "vocab.*", "merges.txt", "*.model", "pytorch_model.bin"],
    )

    print("Done. Model files are now in models/. Run convert_ct2.py next.")
