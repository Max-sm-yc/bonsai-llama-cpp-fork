#!/usr/bin/env python3
"""Direct PTQ1_0 comparison between frozen project baseline and Exp083 best."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/exp083/raw"
REF_BIN = Path("/tmp/bonsai2-reference/build/bin/llama-bench")
CANDIDATE_BIN = ROOT / "build/bin/llama-bench"
MODEL = ROOT / "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
LOG = RAW / "final_reference_ab_65c_driver.log"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    for path in (REF_BIN, CANDIDATE_BIN, MODEL):
        if not path.is_file():
            raise RuntimeError(f"required file missing: {path}")
    reference_ldd = subprocess.run(["ldd", str(REF_BIN)], text=True, capture_output=True, check=True).stdout
    candidate_ldd = subprocess.run(["ldd", str(CANDIDATE_BIN)], text=True, capture_output=True, check=True).stdout
    if "/tmp/bonsai2-reference/build/bin/libggml-cuda" not in reference_ldd:
        raise RuntimeError("frozen baseline executable did not resolve its isolated CUDA library")
    if str(ROOT / "build/bin/libggml-cuda") not in candidate_ldd:
        raise RuntimeError("candidate executable did not resolve its project-local CUDA library")
    lines = [
        f"reference_binary_sha256={sha(REF_BIN)}",
        f"reference_cuda_library_sha256={sha(REF_BIN.parent / 'libggml-cuda.so.0.21.0')}",
        f"candidate_binary_sha256={sha(CANDIDATE_BIN)}",
        f"candidate_cuda_library_sha256={sha(CANDIDATE_BIN.parent / 'libggml-cuda.so.0.21.0')}",
        "reference_ldd:\n" + reference_ldd,
        "candidate_ldd:\n" + candidate_ldd,
    ]
    orders = {
        512: (("reference", "candidate"), ("candidate", "reference")),
        4096: (("candidate", "reference"), ("reference", "candidate")),
    }
    binaries = {"reference": REF_BIN, "candidate": CANDIDATE_BIN}
    try:
        for context, pairs in orders.items():
            for pair_id, order in enumerate(pairs, 1):
                for arm in order:
                    output = RAW / f"final65_ptq1_decode_ctx{context}_pair{pair_id}_{arm}.json"
                    # The machine idles around 61 C; use the same 65 C start gate for both binaries.
                    command = [
                        sys.executable, "benchmark/run.py",
                        "--model", f"PTQ1_0={MODEL}",
                        "--binary", str(binaries[arm]),
                        "--output", str(output),
                        "--modes", "decode", "--contexts", str(context),
                        "--decode-tokens", "128", "--repetitions", "7",
                        "--batch-size", "2048", "--ubatch-size", "512",
                        "--cpu-threads", "8", "--kv-type", "f16",
                        "--cooldown-temp-c", "65",
                    ]
                    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
                    lines.extend((
                        f"\n[{context} pair {pair_id} {arm}] exit={proc.returncode}",
                        "STDERR:\n" + proc.stderr,
                        "STDOUT:\n" + proc.stdout,
                    ))
                    if proc.returncode:
                        raise RuntimeError(f"benchmark failed for context={context}, pair={pair_id}, arm={arm}")
    finally:
        LOG.write_text("\n".join(lines) + "\n")
    print(LOG)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"final reference A/B failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
