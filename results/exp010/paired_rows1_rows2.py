#!/usr/bin/env python3
"""Two alternating 7-repetition pairs comparing isolated ROWS=1 and ROWS=2 builds."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/exp010"
MODEL = "PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
variants = {
    "rows1": "results/exp010/builds/rows1/llama-bench",
    "rows2": "results/exp010/builds/rows2/llama-bench",
}
pairs = [
    [("rows1", variants["rows1"]), ("rows2", variants["rows2"])],
    [("rows2", variants["rows2"]), ("rows1", variants["rows1"])],
]

for pair_n, sequence in enumerate(pairs, 1):
    for label, binary in sequence:
        before = set((ROOT / "results/raw").glob("*"))
        cmd = [
            "python3", "benchmark/run.py", "--model", MODEL, "--modes", "decode",
            "--contexts", "512", "4096", "--decode-tokens", "128",
            "--repetitions", "7", "--cooldown-temp-c", "60",
            "--binary", binary,
            "--output", f"results/exp010/rows1_vs_rows2/pair{pair_n}_{label}.json",
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str((ROOT / binary).parent)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        after = set((ROOT / "results/raw").glob("*"))
        raw_out = OUT / "raw"
        raw_out.mkdir(parents=True, exist_ok=True)
        for src in sorted(after - before):
            shutil.copy2(src, raw_out / src.name)
