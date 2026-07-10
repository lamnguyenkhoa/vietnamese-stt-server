"""Convert models/pytorch_model.bin to fp16 in place to roughly halve its size."""
import torch
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("models", torch_dtype=torch.float32)
model.half()
model.save_pretrained("models")

print("Done. models/pytorch_model.bin is now fp16.")
