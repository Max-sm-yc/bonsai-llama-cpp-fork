#!/bin/bash
set -euo pipefail
for k in 40 136; do
  for nt in 64 128 256 512; do
    results/exp042/cta_width_screen "$k" "$nt" 100 > "results/exp042/screen_k${k}_t${nt}.txt"
  done
done
