#!/usr/bin/env bash
set -euo pipefail
BASE=/home/maxsun/autonomous_projects/bonsai2-rtx3080/build/bin/llama-bench
CAND=/home/maxsun/autonomous_projects/.worktrees/exp060-concat-cache-fusion/build/bin/llama-bench
MODEL=/home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
run() {
  local label="$1" binary="$2" ctx="$3"
  python3 benchmark/run.py --binary "$binary" --model "PTQ1_0=$MODEL" --modes decode --contexts "$ctx" --decode-tokens 128 --repetitions 7 --cooldown-temp-c 60 --output "results/exp060/${label}.json"
}
run pair1_base_ctx512 "$BASE" 512
run pair1_cand_ctx512 "$CAND" 512
run pair1_base_ctx4096 "$BASE" 4096
run pair1_cand_ctx4096 "$CAND" 4096
run pair2_cand_ctx512 "$CAND" 512
run pair2_base_ctx512 "$BASE" 512
run pair2_cand_ctx4096 "$CAND" 4096
run pair2_base_ctx4096 "$BASE" 4096
