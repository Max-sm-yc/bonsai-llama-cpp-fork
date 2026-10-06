#!/usr/bin/env python3
"""One final ROWS=1 source-default build vs archived baseline pair."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/exp010"
MODEL = "PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
sequence = [
    ("rows1_current", "build/bin/llama-bench", "build/bin"),
    ("baseline", "results/exp010/builds/baseline/llama-bench", "results/exp010/builds/baseline"),
]

for label, binary, libdir in sequence:
    before = set((ROOT / "results/raw").glob("*"))
    cmd = [
        "python3", "benchmark/run.py", "--model", MODEL, "--modes", "decode",
        "--contexts", "512", "4096", "--decode-tokens", "128",
        "--repetitions", "7", "--cooldown-temp-c", "60",
        "--binary", binary,
        "--output", f"results/exp010/final_build_check/{label}.json",
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ROOT / libdir)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    after = set((ROOT / "results/raw").glob("*"))
    raw_out = OUT / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    for src in sorted(after - before):
        shutil.copy2(src, raw_out / src.name)
