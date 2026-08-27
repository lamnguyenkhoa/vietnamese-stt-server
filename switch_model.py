"""Switch which PhoWhisper checkpoint this install serves, e.g. small <-> medium.

    python switch_model.py medium
    python switch_model.py vinai/PhoWhisper-large

Downloads the raw HF weights, re-converts them to CTranslate2 int8 in models-ct2/,
and records the choice in model_id.txt so main.py reports it and later runs of
download_model.py stay on it. Restart the server afterwards to load the new model.

Needs torch + transformers installed (the portable build ships them for exactly
this); the conversion is CPU-only and does not need a GPU.
"""
import argparse
import shutil
import sys
from pathlib import Path

import convert_ct2
import download_model

BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "model",
        nargs="?",
        help=f"Checkpoint to switch to: {', '.join(download_model.ALIASES)}, or any HF repo id",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the downloaded HF weights in models/ instead of deleting them after conversion",
    )
    args = parser.parse_args()

    if not args.model:
        print(f"Current model: {download_model.current_repo_id()}")
        print(f"Available shortcuts: {', '.join(download_model.ALIASES)}")
        return 0

    repo_id = download_model.resolve_repo_id(args.model)
    print(f"Switching to {repo_id} ...")

    try:
        download_model.download(repo_id)
    except Exception as exc:
        # A typo'd repo id fails here, before models-ct2/ is touched, so the currently
        # working model is left intact.
        print(f"Download failed ({exc}). Model unchanged.", file=sys.stderr)
        return 1

    print("Converting to CTranslate2 int8 ...")
    convert_ct2.convert()

    (BASE_DIR / "model_id.txt").write_text(repo_id + "\n", encoding="utf-8")

    if not args.keep_raw:
        shutil.rmtree(BASE_DIR / "models", ignore_errors=True)

    print(f"Done. Now serving {repo_id}. Restart the server to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
