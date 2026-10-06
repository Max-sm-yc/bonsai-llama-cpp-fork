#!/usr/bin/env python3
"""One gated baseline/candidate screening pair for ROWS 1, 2, and 8."""
import json
import shutil
import subprocess
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/exp010"
MODEL = "PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
variants = [(2, False), (8, True)]  # True means candidate process starts the pair.

def run(label: str, binary: str, name: str) -> None:
    before = set((ROOT / "results/raw").glob("*"))
    cmd = [
        "python3", "benchmark/run.py", "--model", MODEL, "--modes", "decode",
        "--contexts", "512", "4096", "--decode-tokens", "128",
        "--repetitions", "3", "--cooldown-temp-c", "60",
        "--binary", binary, "--output", f"results/exp010/screen/{name}_isolated.json",
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str((ROOT / binary).parent)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    after = set((ROOT / "results/raw").glob("*"))
    raw_out = OUT / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    for src in sorted(after - before):
        shutil.copy2(src, raw_out / src.name)

for rows, candidate_first in variants:
    base = "results/exp010/builds/baseline/llama-bench"
    candidate = f"results/exp010/builds/rows{rows}/llama-bench"
    sequence = [(f"rows{rows}", candidate), ("baseline", base)]
    if not candidate_first:
        sequence.reverse()
    for label, binary in sequence:
        run(label, binary, f"rows{rows}_{label}")
