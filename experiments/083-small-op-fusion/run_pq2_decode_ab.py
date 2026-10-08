#!/usr/bin/env python3
"""Paired PQ2_0 decode A/B for Exp083; always restores candidate library."""
from __future__ import annotations
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/exp083/raw"
TARGET = ROOT / "build/bin/libggml-cuda.so.0.21.0"
CONTROL = RAW / "libggml-cuda-clean-control.so"
CANDIDATE = RAW / "libggml-cuda-exp083-candidate-tested.so"
MODEL = ROOT / "models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
LOG = RAW / "pq2_decode_ab_driver.log"

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
    if not all(path.is_file() for path in (CONTROL, CANDIDATE, TARGET, MODEL)):
        raise RuntimeError("required library or PQ2_0 model is missing")
    initial = sha(TARGET)
    candidate_hash, control_hash = sha(CANDIDATE), sha(CONTROL)
    if initial != candidate_hash:
        raise RuntimeError(f"unexpected current candidate library hash: {initial}")
    orders = {
        512: (("control", "candidate"), ("candidate", "control")),
        4096: (("candidate", "control"), ("control", "candidate")),
    }
    lines = [f"candidate_sha256={candidate_hash}", f"control_sha256={control_hash}"]
    try:
        for context, pair_orders in orders.items():
            for pair_id, order in enumerate(pair_orders, 1):
                for arm in order:
                    library = CONTROL if arm == "control" else CANDIDATE
                    install(library)
                    output = RAW / f"pq2_decode_ctx{context}_pair{pair_id}_{arm}.json"
                    command = [
                        sys.executable, "benchmark/run.py",
                        "--model", f"PQ2_0={MODEL}",
                        "--binary", "build/bin/llama-bench",
                        "--output", str(output),
                        "--modes", "decode", "--contexts", str(context),
                        "--decode-tokens", "128", "--repetitions", "7",
                        "--batch-size", "2048", "--ubatch-size", "512",
                        "--cpu-threads", "8", "--kv-type", "f16",
                        "--cooldown-temp-c", "60",
                    ]
                    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
                    lines.extend((
                        f"\n[{context} pair {pair_id} {arm}] exit={proc.returncode} library_sha256={sha(TARGET)}",
                        "STDERR:\n" + proc.stderr,
                        "STDOUT:\n" + proc.stdout,
                    ))
                    if proc.returncode:
                        raise RuntimeError(f"benchmark failed for context={context}, pair={pair_id}, arm={arm}")
    finally:
        install(CANDIDATE)
        lines.append(f"\nrestored_candidate_sha256={sha(TARGET)}")
        LOG.write_text("\n".join(lines) + "\n")
    print(LOG)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PQ2_0 A/B failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
