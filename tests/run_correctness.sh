#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cmake --build build --parallel 4 --target \
  test-quantize-fns test-ptq1_0-element-map test-ptq1_0-cuda-dot \
  test-pq2-row-shapes test-backend-ops

ctest --test-dir build --output-on-failure -R \
  'test-quantize-fns|test-ptq1_0-element-map|test-ptq1_0-cuda-dot|test-pq2-row-shapes'

# Compare representative PTQ1_0/PQ2_0 GGML matvec and small-batch matrix
# products on CUDA against the CPU backend. Includes odd row tails and model K sizes.
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT \
  -p '^type_a=(ptq1_0|pq2_0),type_b=f32,m=(67|70),n=(1|2|4|8),k=(1024|5120|6144|17408)'

python3 tests/model_smoke.py --output results/baseline_smoke.json
