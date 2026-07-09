"""Fetch PhoWhisper-medium's files (config, tokenizer, and weights) into models/."""
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vinai/PhoWhisper-medium",
    local_dir="models",
    allow_patterns=["*.json", "*.txt", "vocab.*", "merges.txt", "*.model", "pytorch_model.bin"],
)

print("Done. Model files are now in models/")
