#!/usr/bin/env python3
"""Paired PTQ1_0 prefill A/B for Exp083; always restores candidate library."""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/exp083/raw"
TARGET = ROOT / "build/bin/libggml-cuda.so.0.21.0"
CONTROL = RAW / "libggml-cuda-clean-control.so"
CANDIDATE_SNAPSHOT = RAW / "libggml-cuda-exp083-candidate-tested.so"
MODEL = ROOT / "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
LOG = RAW / "prefill_ab_driver.log"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def install(path: Path) -> None:
    shutil.copy2(path, TARGET)
    if sha(TARGET) != sha(path):
        raise RuntimeError(f"library swap verification failed for {path}")

def main() -> int:
    if not CONTROL.is_file() or not TARGET.is_file() or not MODEL.is_file():
        raise RuntimeError("control library, candidate build, or PTQ1_0 model is missing")
    initial = sha(TARGET)
    if initial != "0f847737c969f5656f90fde2fc1e94a5dee60e39561572b28332a36f16a26b72":
        raise RuntimeError(f"unexpected candidate library hash before A/B: {initial}")
    shutil.copy2(TARGET, CANDIDATE_SNAPSHOT)
    candidate_hash = sha(CANDIDATE_SNAPSHOT)
    control_hash = sha(CONTROL)
    orders = {
        512: (("control", "candidate"), ("candidate", "control")),
        4096: (("candidate", "control"), ("control", "candidate")),
    }
    lines = [f"candidate_sha256={candidate_hash}", f"control_sha256={control_hash}"]
    try:
        for context, pair_orders in orders.items():
            for pair_id, order in enumerate(pair_orders, 1):
                for arm in order:
                    lib = CONTROL if arm == "control" else CANDIDATE_SNAPSHOT
                    install(lib)
                    output = RAW / f"prefill_ctx{context}_pair{pair_id}_{arm}.json"
                    cmd = [
                        sys.executable, "benchmark/run.py",
                        "--model", f"PTQ1_0={MODEL}",
                        "--binary", "build/bin/llama-bench",
                        "--output", str(output),
                        "--modes", "prefill", "--contexts", str(context),
                        "--repetitions", "7", "--batch-size", "2048",
                        "--ubatch-size", "512", "--cpu-threads", "8",
                        "--kv-type", "f16", "--cooldown-temp-c", "60",
                    ]
                    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
                    lines.append(f"\n[{context} pair {pair_id} {arm}] exit={proc.returncode} library_sha256={sha(TARGET)}")
                    lines.append("STDERR:\n" + proc.stderr)
                    lines.append("STDOUT:\n" + proc.stdout)
                    if proc.returncode:
                        raise RuntimeError(f"benchmark failed for context={context}, pair={pair_id}, arm={arm}")
    finally:
        install(CANDIDATE_SNAPSHOT)
        lines.append(f"\nrestored_candidate_sha256={sha(TARGET)}")
        LOG.write_text("\n".join(lines) + "\n")
    print(LOG)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"prefill A/B failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
