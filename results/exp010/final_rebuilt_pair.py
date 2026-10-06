#!/usr/bin/env python3
"""Verify the freshly rebuilt source-default ROWS=1 library against baseline."""
from pathlib import Path
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MODEL = "PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
variants = [
    ("rows1_rebuilt", "build/bin/llama-bench", "build/bin"),
    ("baseline", "results/exp010/builds/baseline/llama-bench", "results/exp010/builds/baseline"),
]

for label, binary, library_dir in variants:
    before = set((ROOT / "results/raw").glob("*"))
    command = [
        "python3", "benchmark/run.py", "--model", MODEL, "--modes", "decode",
        "--contexts", "512", "4096", "--decode-tokens", "128",
        "--repetitions", "7", "--cooldown-temp-c", "60",
        "--binary", binary,
        "--output", f"results/exp010/final_rebuilt_pair/{label}.json",
    ]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(ROOT / library_dir)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    raw_dir = ROOT / "results/exp010/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    after = set((ROOT / "results/raw").glob("*"))
    for raw in sorted(after - before):
        shutil.copy2(raw, raw_dir / raw.name)
