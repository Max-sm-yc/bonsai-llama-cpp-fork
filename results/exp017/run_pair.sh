#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
for VARIANT in candidate control; do
  LD_LIBRARY_PATH="$ROOT/results/exp017/builds/$VARIANT" \
    python3 "$ROOT/benchmark/run.py" \
      --binary "results/exp017/builds/$VARIANT/llama-bench" \
      --output "results/exp017/raw/${VARIANT}_pair1.json" \
      --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
      --modes decode --contexts 512 4096 --decode-tokens 128 --repetitions 7 \
      --batch-size 2048 --ubatch-size 512 --cpu-threads 8 --kv-type f16 \
      --cooldown-temp-c 60
 done
