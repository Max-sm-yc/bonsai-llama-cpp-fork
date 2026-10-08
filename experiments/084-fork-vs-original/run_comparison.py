#!/usr/bin/env python3
"""Run a paired RTX 3080 comparison of the research fork and frozen baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results/exp084"
FORMATS = {
    "PTQ1_0": ROOT / "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "PQ2_0": ROOT / "models/Ternary-Bonsai-2-27B-PQ2_0.gguf",
}
WORKLOADS = [
    ("prefill", 512),
    ("prefill", 4096),
    ("decode", 512),
    ("decode", 2048),
    ("decode", 4096),
    ("combined", 512),
    ("combined", 4096),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_isolated_binary(binary: Path, expected_library: Path) -> str:
    if not binary.is_file():
        raise RuntimeError(f"missing llama-bench binary: {binary}")
    deps = subprocess.run(["ldd", str(binary)], check=True, capture_output=True, text=True).stdout
    if str(expected_library) not in deps:
        raise RuntimeError(f"{binary} does not resolve its isolated CUDA library {expected_library}")
    return deps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-binary", type=Path, default=Path("/tmp/bonsai2-reference/build/bin/llama-bench"))
    parser.add_argument("--candidate-binary", type=Path, default=ROOT / "build/bin/llama-bench")
    parser.add_argument("--cooldown-temp-c", type=float, default=65.0)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true", help="resume a partial summary and skip complete workload pairs")
    parser.add_argument("--rerun", action="append", default=[], metavar="MODE:CONTEXT", help="replace a completed workload, for example decode:4096")
    parser.add_argument("--rerun-pair", action="append", default=[], metavar="MODE:CONTEXT:PAIR", help="replace one paired order, for example combined:4096:2")
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("use at least two repetitions")
    args.reference_binary = args.reference_binary.resolve()
    args.candidate_binary = args.candidate_binary.resolve()
    args.rerun_workloads = set()
    for value in args.rerun:
        try:
            mode, context = value.split(":", 1)
            context = int(context)
        except ValueError:
            parser.error(f"invalid --rerun value {value!r}; expected MODE:CONTEXT")
        args.rerun_workloads.add((mode, context))
    args.rerun_pairs = set()
    for value in args.rerun_pair:
        try:
            mode, context, pair = value.split(":", 2)
            context, pair = int(context), int(pair)
            if pair not in (1, 2):
                raise ValueError
        except ValueError:
            parser.error(f"invalid --rerun-pair value {value!r}; expected MODE:CONTEXT:1|2")
        args.rerun_pairs.add(((mode, context), pair))
    return args


def main() -> int:
    args = parse_args()
    for model in FORMATS.values():
        if not model.is_file():
            raise RuntimeError(f"missing benchmark model: {model}")

    OUTPUT_DIR.joinpath("raw").mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / ("driver_resume.log" if args.resume else "driver.log")
    summary_path = OUTPUT_DIR / "summary.json"
    source = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    reference_deps = check_isolated_binary(args.reference_binary, args.reference_binary.parent / "libggml-cuda.so")
    candidate_deps = check_isolated_binary(args.candidate_binary, args.candidate_binary.parent / "libggml-cuda.so")
    arms = {
        "original": args.reference_binary,
        "fork": args.candidate_binary,
    }
    binary_info = {
        name: {
            "path": str(binary),
            "sha256": sha256(binary),
            "cuda_library_path": str(binary.parent / "libggml-cuda.so.0.21.0"),
            "cuda_library_sha256": sha256(binary.parent / "libggml-cuda.so.0.21.0"),
            "ldd": reference_deps if name == "original" else candidate_deps,
        }
        for name, binary in arms.items()
    }
    records = []
    completed_workloads = set()
    completed_pairs = {}
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text())
        previous_gate = previous.get("configuration", {}).get("cooldown_temperature_c", args.cooldown_temp_c)
        grouped = {}
        for record in previous.get("records", []):
            grouped.setdefault((record["mode"], record["context"]), []).append(record)
        for key, group in grouped.items():
            if key in args.rerun_workloads:
                continue
            pairs = {}
            for record in group:
                pairs.setdefault(record["pair"], []).append(record)
            complete = {
                pair: pair_records for pair, pair_records in pairs.items()
                if len(pair_records) == 4 and (key, pair) not in args.rerun_pairs
            }
            for pair_records in complete.values():
                for record in pair_records:
                    record.setdefault("cooldown_temp_c", record.get("cooldown_temperature_c", previous_gate))
                    if "start_temperature_c" not in record:
                        run_file = ROOT / record["source_file"]
                        run_data = json.loads(run_file.read_text())
                        model_run = next(run for run in run_data["runs"] if run["format"] == record["format"])
                        record["start_temperature_c"] = model_run["gpu_memory_before"]["temperature_c"]
                    if isinstance(record["cooldown_temp_c"], list):
                        record["cooldown_temp_c"] = float(record["cooldown_temp_c"][0])
                    if record["cooldown_temp_c"] is None:
                        record["cooldown_temp_c"] = float(previous_gate)
                    records.append(record)
            completed_pairs[key] = set(complete)
            if len(complete) == 2:
                completed_workloads.add(key)
    transcript = [
        f"timestamp_utc={dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"candidate_checkout={source}",
        "candidate_code_commit=62b4b4ce0c2809272b9d69d09f3359abd7111848",
        "reference_project_commit=2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c",
        f"cooldown_temp_c={args.cooldown_temp_c}",
        f"repetitions={args.repetitions}",
        f"decode_tokens={args.decode_tokens}",
        json.dumps(binary_info, indent=2),
    ]

    def save_summary(status: str) -> None:
        gates_by_mode = {}
        for mode in sorted({record["mode"] for record in records}):
            gates_by_mode[mode] = sorted({record["cooldown_temp_c"] for record in records if record["mode"] == mode})
        summary_path.write_text(json.dumps({
            "schema_version": 1,
            "status": status,
            "candidate_checkout": source,
            "candidate_code_commit": "62b4b4ce0c2809272b9d69d09f3359abd7111848",
            "reference_project_commit": "2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c",
            "configuration": {
                "gpu": "NVIDIA GeForce RTX 3080, sm_86",
                "contexts": sorted({depth for _, depth in WORKLOADS}),
                "decode_tokens": args.decode_tokens,
                "repetitions_per_run": args.repetitions,
                "reversed_order_pairs": 2,
                "cooldown_temperature_c_by_mode": gates_by_mode,
                "batch_size": 2048,
                "ubatch_size": 512,
                "gpu_layers": 99,
                "flash_attention": True,
                "kv_cache": "f16",
                "cpu_threads": 8,
            },
            "binaries": binary_info,
            "records": records,
        }, indent=2) + "\n")

    save_summary("running")

    try:
        for mode, context in WORKLOADS:
            if (mode, context) in completed_workloads:
                print(f"[{mode} context={context}] already complete; skipping", flush=True)
                continue
            for pair, order in enumerate((("original", "fork"), ("fork", "original")), start=1):
                if pair in completed_pairs.get((mode, context), set()):
                    print(f"[{mode} context={context} pair={pair}] already complete; skipping", flush=True)
                    continue
                for arm in order:
                    stem = f"{mode}_ctx{context}_pair{pair}_{arm}"
                    output = OUTPUT_DIR / "raw" / f"{stem}.json"
                    command = [
                        sys.executable,
                        str(ROOT / "benchmark/run.py"),
                        "--binary", str(arms[arm]),
                        "--output", str(output),
                        "--modes", mode,
                        "--contexts", str(context),
                        "--decode-tokens", str(args.decode_tokens),
                        "--repetitions", str(args.repetitions),
                        "--cooldown-temp-c", str(args.cooldown_temp_c),
                    ]
                    for format_name, model in FORMATS.items():
                        command.extend(("--model", f"{format_name}={model}"))
                    print(f"[{mode} context={context} pair={pair} arm={arm}] starting", flush=True)
                    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                    transcript.extend((
                        f"\n[{stem}] exit={proc.returncode}",
                        "COMMAND: " + subprocess.list2cmdline(command),
                        "STDERR:\n" + proc.stderr,
                        "STDOUT:\n" + proc.stdout,
                    ))
                    log_path.write_text("\n".join(transcript) + "\n")
                    if proc.returncode:
                        raise RuntimeError(f"benchmark failed for {stem}; see {log_path}")
                    run_data = json.loads(output.read_text())
                    for run in run_data["runs"]:
                        for row in run["results"]:
                            records.append({
                                "mode": mode,
                                "context": context,
                                "format": run["format"],
                                "pair": pair,
                                "arm": arm,
                                "median_tokens_per_second": row["median_tokens_per_second"],
                                "mean_tokens_per_second": row["average_tokens_per_second"],
                                "stddev_tokens_per_second": row["stddev_tokens_per_second"],
                                "min_tokens_per_second": row["min_tokens_per_second"],
                                "max_tokens_per_second": row["max_tokens_per_second"],
                                "median_latency_ms": statistics.median(row["sample_latency_ns"]) / 1_000_000,
                                "peak_gpu_memory_mib": run["gpu_memory_peak"]["memory_used_mib"],
                                "cooldown_temp_c": run_data["configuration"]["cooldown_temperature_c"],
                                "start_temperature_c": run["gpu_memory_before"]["temperature_c"],
                                "source_file": str(output.relative_to(ROOT)),
                            })
                    save_summary("running")
            print(f"[{mode} context={context}] both reversed-order pairs complete", flush=True)
    finally:
        log_path.write_text("\n".join(transcript) + "\n")

    save_summary("complete")
    print(summary_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        raise SystemExit(1)
