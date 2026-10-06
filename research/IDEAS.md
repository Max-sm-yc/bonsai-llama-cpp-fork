# Candidate hypotheses

Ranked from the RTX 3080 baseline trace; results must be judged by controlled end-to-end decode and combined throughput.

1. PTQ1_0 GEMV dominates the trace. With prefetch, 2/4/8 warp-count changes, and a direct constant-memory LUT exhausted, test exact bit-sliced/integer extraction for the base-3 trits in `vec_dot_ptq1_0_q8_1_multi` (`ggml/src/ggml-cuda/vecdotq.cuh`). Benchmark the full block dot with `qh` and production activation layout, preserve DP4A accumulation and bias correction, then require order-balanced integrated decode confirmation.
2. Determine whether PTQ1_0's lower payload is converting into DRAM traffic reduction or whether instruction/occupancy limits dominate. NCU counters are unavailable to this user; use static resources, controlled size sweeps, and end-to-end evidence without changing system counter permissions.
3. Test whether fused Hadamard-to-Q8_1 activation preparation for PQ2_0 recovers its extra transform/quantization kernels; then revisit launch gaps, RMSNorm, and gated-delta recurrent attention after a matvec improvement changes the ranking.
