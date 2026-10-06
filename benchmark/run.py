#!/usr/bin/env python3
"""Run repeatable llama-bench workloads and sample whole-GPU memory use."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "PQ2_0=models/Ternary-Bonsai-2-27B-PQ2_0.gguf",
]
GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm",
    "--format=csv,noheader,nounits",
]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_number(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def gpu_sample() -> dict[str, float | None] | None:
    try:
        proc = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=3, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    fields = [part.strip() for part in proc.stdout.strip().split(",")]
    if len(fields) != 5:
        return None
    names = ("memory_used_mib", "utilization_percent", "temperature_c", "power_w", "sm_clock_mhz")
    return dict(zip(names, (parse_number(field) for field in fields)))


def wait_for_cool_gpu(max_temp_c: float) -> dict[str, float | None]:
    """Wait until the GPU is idle and below the requested start temperature."""
    started = time.monotonic()
    last_notice = 0.0
    while True:
        sample = gpu_sample()
        if sample is None:
            raise RuntimeError("cannot read GPU telemetry while waiting for cooldown")
        temperature = sample["temperature_c"]
        utilization = sample["utilization_percent"]
        if temperature is not None and utilization is not None and temperature <= max_temp_c and utilization <= 5:
            waited = time.monotonic() - started
            print(f"GPU ready at {temperature:.0f} C and {utilization:.0f}% utilization after {waited:.0f}s", file=sys.stderr, flush=True)
            return sample
        now = time.monotonic()
        if now - last_notice >= 30:
            temp_text = "unknown" if temperature is None else f"{temperature:.0f} C"
            util_text = "unknown" if utilization is None else f"{utilization:.0f}%"
            print(f"waiting for GPU cooldown: {temp_text}, utilization {util_text}", file=sys.stderr, flush=True)
            last_notice = now
        time.sleep(5)


def make_command(args: argparse.Namespace, model_path: Path, mode: str) -> list[str]:
    command = [
        str(resolve_path(args.binary)),
        "-m", str(model_path),
        "-ngl", "99",
        "-fa", "on",
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-ctk", args.kv_type,
        "-ctv", args.kv_type,
        "-t", str(args.cpu_threads),
        "-r", str(args.repetitions),
        "-o", "json",
    ]
    depths = ",".join(str(value) for value in args.contexts)
    if mode == "prefill":
        command += ["-p", depths, "-n", "0", "-d", "0"]
    elif mode == "decode":
        command += ["-p", "0", "-n", str(args.decode_tokens), "-d", depths]
    else:
        command += ["-p", "0", "-n", "0"]
        for context in args.contexts:
            command += ["-pg", f"{context},{args.decode_tokens}"]
    return command


def summarize_row(row: dict) -> dict:
    samples = row.get("samples_ts", [])
    sample_ns = row.get("samples_ns", [])
    result = {
        "test": row.get("test"),
        "prompt_tokens": row.get("n_prompt"),
        "decode_tokens": row.get("n_gen"),
        "prefilled_context_tokens": row.get("n_depth"),
        "context_length_tokens": row.get("n_depth") or row.get("n_prompt"),
        "average_tokens_per_second": row.get("avg_ts"),
        "stddev_tokens_per_second": row.get("stddev_ts"),
        "mean_latency_ms": row.get("avg_ns", 0) / 1_000_000,
        "stddev_latency_ms": row.get("stddev_ns", 0) / 1_000_000,
        "sample_tokens_per_second": samples,
        "sample_latency_ns": sample_ns,
        "runtime_row": row,
    }
    if samples:
        result["median_tokens_per_second"] = statistics.median(samples)
        result["min_tokens_per_second"] = min(samples)
        result["max_tokens_per_second"] = max(samples)
    return result


def run_one(args: argparse.Namespace, label: str, model_path: Path, mode: str, stamp: str) -> dict:
    command = make_command(args, model_path, mode)
    before = gpu_sample()
    peak = before
    samples: list[dict[str, float | None]] = []
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while proc.poll() is None:
        sample = gpu_sample()
        if sample is not None:
            samples.append(sample)
            if peak is None or (sample["memory_used_mib"] or 0) > (peak["memory_used_mib"] or 0):
                peak = sample
        time.sleep(0.25)
    stdout, stderr = proc.communicate()
    raw_dir = resolve_path("results/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    (raw_dir / f"{stamp}_{safe_label}_{mode}.stdout.json").write_text(stdout)
    (raw_dir / f"{stamp}_{safe_label}_{mode}.stderr.log").write_text(stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed for {label}/{mode}; see results/raw/{stamp}_{safe_label}_{mode}.stderr.log")
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"llama-bench emitted invalid JSON for {label}/{mode}: {error}") from error
    if not rows:
        raise RuntimeError(f"llama-bench produced no rows for {label}/{mode}")
    idle_mem = before.get("memory_used_mib") if before else None
    peak_mem = peak.get("memory_used_mib") if peak else None
    return {
        "format": label,
        "model": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
        "mode": mode,
        "command": command,
        "exit_code": proc.returncode,
        "gpu_memory_before": before,
        "gpu_memory_peak": peak,
        "gpu_memory_increase_mib": peak_mem - idle_mem if peak_mem is not None and idle_mem is not None else None,
        "gpu_samples": samples,
        "results": [summarize_row(row) for row in rows],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=None, metavar="FORMAT=PATH", help="repeat to select formats; defaults to both project GGUF files")
    parser.add_argument("--binary", default="build/bin/llama-bench")
    parser.add_argument("--output", default="results/latest.json")
    parser.add_argument("--modes", nargs="+", choices=("prefill", "decode", "combined"), default=("prefill", "decode", "combined"))
    parser.add_argument("--contexts", nargs="+", type=int, default=(128, 512, 2048, 4096))
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--kv-type", default="f16")
    parser.add_argument("--cooldown-temp-c", type=float, default=None, help="wait for <= this GPU temperature and <=5%% utilization before each format")
    args = parser.parse_args()
    if any(context <= 0 for context in args.contexts):
        parser.error("context sizes must be positive")
    if args.repetitions < 2 or args.decode_tokens <= 0:
        parser.error("use at least two repetitions and a positive decode length")
    specs = args.model if args.model is not None else DEFAULT_MODELS
    args.models = []
    for spec in specs:
        if "=" not in spec:
            parser.error(f"invalid --model {spec!r}; expected FORMAT=PATH")
        label, path = spec.split("=", 1)
        model_path = resolve_path(path)
        if not model_path.is_file():
            parser.error(f"model file not found: {model_path}")
        args.models.append((label, model_path))
    if not resolve_path(args.binary).is_file():
        parser.error(f"llama-bench not found: {resolve_path(args.binary)}")
    return args


def main() -> int:
    args = parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs = []
    for label, model_path in args.models:
        if args.cooldown_temp_c is not None:
            ready_sample = wait_for_cool_gpu(args.cooldown_temp_c)
            print(f"[{label}] cooldown gate sample: {ready_sample}", file=sys.stderr, flush=True)
        for mode in args.modes:
            print(f"[{label}] {mode}: contexts={args.contexts}, repetitions={args.repetitions}", file=sys.stderr, flush=True)
            runs.append(run_one(args, label, model_path, mode, stamp))
    result = {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware": {"gpu": "NVIDIA GeForce RTX 3080", "compute_capability": "8.6", "vram_mib": 10240},
        "configuration": {
            "contexts": args.contexts,
            "decode_tokens": args.decode_tokens,
            "repetitions": args.repetitions,
            "warmup": "llama-bench default warmups enabled",
            "n_gpu_layers": 99,
            "flash_attention": "on",
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "kv_type": args.kv_type,
            "cpu_threads": args.cpu_threads,
            "cooldown_temperature_c": args.cooldown_temp_c,
            "timing_scope": "llama-bench excludes tokenization and sampling; combined mode includes prompt evaluation and model token generation",
        },
        "runs": runs,
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1)
