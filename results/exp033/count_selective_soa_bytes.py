#!/usr/bin/env python3
"""Inventory PTQ1_0 tensors whose reduction dimension is 17,408."""

from pathlib import Path

from gguf import GGUFReader


MODEL = Path("models/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
reader = GGUFReader(MODEL)
selected = [
    tensor
    for tensor in reader.tensors
    if tensor.tensor_type.name == "PTQ1_0"
    and len(tensor.shape) >= 2
    and int(tensor.shape[0]) == 17_408
]
total = sum(int(tensor.n_bytes) for tensor in selected)

print(f"model={MODEL}")
print(f"PTQ1_0 tensors with K=17408: {len(selected)}")
for tensor in selected:
    print(f"{tensor.name}\t{list(map(int, tensor.shape))}\t{int(tensor.n_bytes)}")
print(f"sidecar_bytes={total}")
print(f"sidecar_MiB={total / 1_048_576:.2f}")
print(f"baseline_peak_MiB=6805")
print(f"projected_peak_MiB={6805 + total / 1_048_576:.2f}")
