"""Convert models/ (raw HF PhoWhisper weights, from download_model.py) into
CTranslate2 format in models-ct2/, quantized to int8. This is what main.py actually
loads at runtime via faster-whisper.

Requires transformers and torch installed -- only for this one-time conversion, not
at runtime. Safe to uninstall them afterward (`pip uninstall torch transformers`) if
you want a slimmer environment.
"""
from ctranslate2.converters import TransformersConverter

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

if __name__ == "__main__":
    converter = TransformersConverter("models", copy_files=COPY_FILES)
    converter.convert("models-ct2", quantization="int8", force=True)

    print("Done. CTranslate2 model is now in models-ct2/")
