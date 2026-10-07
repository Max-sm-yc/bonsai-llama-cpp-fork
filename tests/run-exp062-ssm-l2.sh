#!/usr/bin/env bash
set -euo pipefail

binary=$1
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

for shape in model fallback; do
    GGML_CUDA_DISABLE_SSM_L2_FUSION=1 "$binary" "$shape" "$tmpdir/$shape.generic"
    "$binary" "$shape" "$tmpdir/$shape.fused"
    cmp "$tmpdir/$shape.generic" "$tmpdir/$shape.fused"
done
