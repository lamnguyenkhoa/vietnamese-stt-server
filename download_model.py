"""Fetch PhoWhisper-small's files (config, tokenizer, and weights) into models/."""
from huggingface_hub import snapshot_download

REPO_ID = "vinai/PhoWhisper-small"

if __name__ == "__main__":
    snapshot_download(
        repo_id=REPO_ID,
        local_dir="models",
        allow_patterns=["*.json", "*.txt", "vocab.*", "merges.txt", "*.model", "pytorch_model.bin"],
    )

    print("Done. Model files are now in models/")
