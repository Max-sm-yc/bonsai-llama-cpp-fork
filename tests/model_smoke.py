#!/usr/bin/env python3
"""Load each GGUF on CUDA and generate a deterministic fixed-length completion."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "PQ2_0=models/Ternary-Bonsai-2-27B-PQ2_0.gguf",
]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=None, metavar="FORMAT=PATH")
    parser.add_argument("--binary", default="build/bin/llama-cli")
    parser.add_argument("--output", default="results/baseline_smoke.json")
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    binary = resolve_path(args.binary)
    if not binary.is_file():
        parser.error(f"llama-cli not found: {binary}")
    specs = args.model if args.model is not None else DEFAULT_MODELS
    results = []
    for spec in specs:
        if "=" not in spec:
            parser.error(f"invalid --model {spec!r}; expected FORMAT=PATH")
        label, path = spec.split("=", 1)
        model = resolve_path(path)
        if not model.is_file():
            parser.error(f"model file not found: {model}")
        command = [
            str(binary), "-m", str(model), "-ngl", "99", "-fa", "on",
            "-c", "512", "-t", "8", "-p", "The RTX 3080 is a graphics card that",
            "-n", str(args.tokens), "--seed", "42", "--temp", "0",
            "--ignore-eos", "--no-display-prompt", "--simple-io", "--single-turn",
        ]
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_dir = resolve_path("results/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
        stdout_path = raw_dir / f"{stamp}_{safe_label}_smoke.stdout.txt"
        stderr_path = raw_dir / f"{stamp}_{safe_label}_smoke.stderr.log"
        stdout_path.write_text(proc.stdout)
        stderr_path.write_text(proc.stderr)
        completion = proc.stdout.strip()
        if proc.returncode != 0 or not completion:
            raise RuntimeError(f"CUDA smoke inference failed for {label}; see {stderr_path}")
        results.append({
            "format": label,
            "model": str(model.relative_to(ROOT)) if model.is_relative_to(ROOT) else str(model),
            "command": command,
            "exit_code": proc.returncode,
            "completion": completion,
            "stdout_path": str(stdout_path.relative_to(ROOT)),
            "stderr_path": str(stderr_path.relative_to(ROOT)),
        })
        print(f"{label}: generated a non-empty {args.tokens}-token completion")
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware": "NVIDIA GeForce RTX 3080, sm_86",
        "configuration": {"context": 512, "tokens": args.tokens, "seed": 42, "temperature": 0.0},
        "results": results,
    }, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"model smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1)
