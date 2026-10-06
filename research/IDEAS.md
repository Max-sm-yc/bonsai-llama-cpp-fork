# Candidate hypotheses

Ranked from the RTX 3080 baseline trace; results must be judged by controlled end-to-end decode and combined throughput.

1. PTQ1_0 GEMV dominates the trace. Test an exact 2-bit side representation against the current 1.75-bit base-3 format to measure the traffic-versus-unpack tradeoff. Prototype the full block dot on sm_86, include `qh` and production activation layout, verify the <10 GiB model path, and require order-balanced end-to-end confirmation. If reformatting loses, test bit-sliced extraction in the original base-3 path; do not repeat the slow direct constant-memory LUT.
2. Determine whether PTQ1_0's lower payload is converting into DRAM traffic reduction or whether instruction/occupancy limits dominate. NCU counters are unavailable to this user; use static resources, controlled size sweeps, and end-to-end evidence without changing system counter permissions.
3. Test whether fused Hadamard-to-Q8_1 activation preparation for PQ2_0 recovers its extra transform/quantization kernels; then revisit launch gaps, RMSNorm, and gated-delta recurrent attention after a matvec improvement changes the ranking.
