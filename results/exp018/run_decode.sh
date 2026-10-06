#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
for VARIANT in candidate control; do
  for PAIR in 1 2; do
    LD_LIBRARY_PATH="$ROOT/results/exp018/$VARIANT" \
      python3 "$ROOT/benchmark/run.py" \
        --binary "$ROOT/build/bin/llama-bench" \
        --output "$ROOT/results/exp018/raw/${VARIANT}_pair${PAIR}.json" \
        --model "PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
        --modes decode --contexts 512 4096 --decode-tokens 128 --repetitions 7 \
        --batch-size 2048 --ubatch-size 512 --cpu-threads 8 --kv-type f16 \
        --cooldown-temp-c 60
  done
done
