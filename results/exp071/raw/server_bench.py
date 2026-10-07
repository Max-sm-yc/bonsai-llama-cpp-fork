#!/usr/bin/env python3
"""Repeat fixed-token decode requests against one llama-server instance."""
import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/exp071/raw"
SERVER = ROOT / "build/bin/llama-server"
PROMPTS = [
    "Write a Python function to sort a list and explain its complexity. ",
    "Explain why GPU matrix multiplication is useful for language model inference. ",
    "Review this C++ function for correctness: int divide(int x) { return 10 / x; } ",
]
NATURAL_SOURCES = [
    ("report", ["experiments/052-pq2-steady-profile/REPORT.md",
                "experiments/070-adaptive-ubatch/REPORT.md",
                "research/STATE.md"],
     "Read the technical reports below and explain their benchmark methods and findings.\n\n"),
    ("model_source", ["src/models/qwen35.cpp"],
     "Review this Qwen3.5 model graph implementation and explain its data flow.\n\n"),
    ("spec_source", ["common/speculative.cpp"],
     "Explain how this speculative decoding implementation proposes tokens and verifies them.\n\n"),
]


def request(url, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def gpu_gate():
    for _ in range(720):
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True).strip()
        temp, util = [float(x.strip()) for x in raw.split(",")]
        if temp <= 60 and util <= 5:
            return {"temperature_c": temp, "gpu_utilization_pct": util}
        time.sleep(5)
    raise RuntimeError("GPU did not reach <=60 C and <=5% utilization within one hour")


def wait_server(proc, url, log_path):
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited {proc.returncode}; see {log_path}")
        try:
            request(url + "/health", timeout=2)
            return
        except (OSError, urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise RuntimeError(f"llama-server did not become healthy; see {log_path}")


def natural_prompts():
    prompts = []
    sources = []
    for name, paths, instruction in NATURAL_SOURCES:
        pieces = [instruction]
        for rel in paths:
            source = ROOT / rel
            content = source.read_text(errors="replace")
            sources.append({"name": name, "path": rel,
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "characters": len(content)})
            pieces.append(f"\n\n--- {rel} ---\n\n{content}")
        prompts.append("".join(pieces)[:40000])
    if any(len(prompt) < 12000 for prompt in prompts):
        raise RuntimeError("natural source text is too short to form a 4096-token context")
    (OUT / "natural-prompt-sources.json").write_text(
        json.dumps(sources, indent=2) + "\n")
    return prompts


def tokenize_seeds(url, prompt_set):
    prompts = PROMPTS if prompt_set == "cycled" else natural_prompts()
    seeds = []
    for prompt in prompts:
        result = request(url + "/tokenize", {
            "content": prompt, "add_special": True, "parse_special": False,
        })
        tokens = result["tokens"]
        min_tokens = 4096 if prompt_set == "natural" else 2
        if len(tokens) < min_tokens:
            raise RuntimeError(f"tokenizer returned only {len(tokens)} tokens")
        seeds.append(tokens[:4096] if prompt_set == "natural" else tokens)
    seed_file = OUT / f"prompt-seeds-{prompt_set}.json"
    if seed_file.exists():
        saved = json.loads(seed_file.read_text())
        if saved["token_ids"] != seeds:
            raise RuntimeError("tokenizer output differs from the first arm")
    else:
        seed_file.write_text(json.dumps({"prompt_set": prompt_set,
                                         "source_files": [x[1] for x in NATURAL_SOURCES] if prompt_set == "natural" else [],
                                         "token_ids": seeds}, indent=2) + "\n")
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["PTQ1_0", "PQ2_0", "BUNDLE_TARGET", "PQ2_0_MTP"], required=True)
    parser.add_argument("--context", type=int, choices=[512, 4096], required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--prompt-set", choices=["natural", "cycled"], default="natural")
    parser.add_argument("--focus-prompt-index", type=int, choices=[0, 1, 2])
    parser.add_argument("--trace-acceptance", action="store_true")
    args = parser.parse_args()

    model_name = {
        "PTQ1_0": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        "PQ2_0": "Ternary-Bonsai-2-27B-PQ2_0.gguf",
        "BUNDLE_TARGET": "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf",
        "PQ2_0_MTP": "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf",
    }[args.arm]
    model = ROOT / "models" / model_name
    n_ctx = 4608
    url = f"http://127.0.0.1:{args.port}"
    stem = f"{args.prompt_set}_r{args.round}_{args.arm.lower()}_ctx{args.context}"
    if args.trace_acceptance:
        stem += "_trace"
    log_path = OUT / f"{stem}.server.log"
    telemetry_path = OUT / f"{stem}.gpu.csv"
    result_path = OUT / f"{stem}.json"
    cmd = [str(SERVER), "-m", str(model), "-ngl", "99", "-fa", "on",
           "-b", "2048", "-ub", "512", "-ctk", "f16", "-ctv", "f16",
           "-t", "8", "-c", str(n_ctx), "-np", "1", "--host", "127.0.0.1",
           "--port", str(args.port), "--no-webui"]
    if args.arm == "PQ2_0_MTP":
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    if args.trace_acceptance:
        cmd += ["-lv", "4"]

    gate = gpu_gate()
    with log_path.open("w") as log:
        env = os.environ.copy()
        if args.trace_acceptance:
            env["LLAMA_TRACE"] = "1"
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        telemetry = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits", "--loop-ms=200",
        ], stdout=telemetry_path.open("w"), stderr=subprocess.DEVNULL,
           start_new_session=True)
        try:
            wait_server(proc, url, log_path)
            seeds = tokenize_seeds(url, args.prompt_set)
            rows = []
            for prompt_idx, seed in enumerate(seeds):
                if args.focus_prompt_index is not None and prompt_idx != args.focus_prompt_index:
                    continue
                for rep in range(args.repetitions):
                    prompt_tokens = (seed[:args.context] if args.prompt_set == "natural" else
                                     [seed[0]] + [seed[1 + (i % (len(seed) - 1))]
                                                  for i in range(args.context - 1)])
                    start = time.perf_counter()
                    result = request(url + "/completion", {
                        "prompt": prompt_tokens,
                        "n_predict": 128,
                        "temperature": 0,
                        "top_k": 1,
                        "seed": 42,
                        "cache_prompt": False,
                        "ignore_eos": True,
                        "n_probs": 5 if args.trace_acceptance else 0,
                    })
                    wall_s = time.perf_counter() - start
                    timing = result.get("timings", {})
                    if timing.get("prompt_n") != args.context:
                        raise RuntimeError(f"prompt_n={timing.get('prompt_n')} expected {args.context}")
                    if timing.get("predicted_n") != 128:
                        raise RuntimeError(f"predicted_n={timing.get('predicted_n')} expected 128")
                    rows.append({
                        "prompt_index": prompt_idx,
                        "repetition": rep,
                        "prompt_n": timing.get("prompt_n"),
                        "prompt_ms": timing.get("prompt_ms"),
                        "prompt_per_second": timing.get("prompt_per_second"),
                        "predicted_n": timing.get("predicted_n"),
                        "predicted_ms": timing.get("predicted_ms"),
                        "predicted_per_second": timing.get("predicted_per_second"),
                        "draft_n": timing.get("draft_n", 0),
                        "draft_n_accepted": timing.get("draft_n_accepted", 0),
                        "wall_seconds": wall_s,
                        "content": result.get("content", ""),
                    })
            data = {
                "arm": args.arm,
                "prompt_set": args.prompt_set,
                "model": str(model),
                "context_tokens": args.context,
                "server_context": n_ctx,
                "repetitions_per_prompt": args.repetitions,
                "prompt_source_files": [x[1] for x in NATURAL_SOURCES] if args.prompt_set == "natural" else [],
                "start_gate": gate,
                "command": cmd,
                "rows": rows,
            }
            result_path.write_text(json.dumps(data, indent=2) + "\n")
        finally:
            try:
                request(url + "/shutdown", {}, timeout=3)
            except Exception:
                pass
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            try:
                os.killpg(telemetry.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            telemetry.wait(timeout=10)
    print(json.dumps({"result": str(result_path), "log": str(log_path), "gpu": str(telemetry_path)}))


if __name__ == "__main__":
    main()
