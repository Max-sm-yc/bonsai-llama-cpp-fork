# Candidate hypotheses

Ranked from the RTX 3080 baseline trace; results must be judged by controlled end-to-end decode and combined throughput.

1. Resolve experiment 001 with a paired same-build prefetch on/off comparison at contexts 512/4096 for decode and combined workloads. Isolate workload/context starts, alternate variant order, and retain GPU thermal/clock telemetry.
2. PTQ1_0 GEMV dominates the trace. After measurement is repeatable, explore sm_86-specialized work partitioning, ternary unpack scheduling, and register use in `ggml/src/ggml-cuda/mmvq.cu` and `vecdotq.cuh`.
3. Determine whether PTQ1_0's lower payload is converting into DRAM traffic reduction or whether instruction/occupancy limits dominate. NCU counters are unavailable to this user; use static resources, controlled size sweeps, and end-to-end evidence without changing system counter permissions.
4. Test whether fused Hadamard-to-Q8_1 activation preparation for PQ2_0 recovers its extra transform/quantization kernels; then revisit launch gaps, RMSNorm, and gated-delta recurrent attention after a matvec improvement changes the ranking.
