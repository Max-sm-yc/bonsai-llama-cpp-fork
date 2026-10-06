# Candidate hypotheses

Ranked from the RTX 3080 baseline trace; results must be judged by controlled end-to-end decode and combined throughput.

1. PTQ1_0 GEMV dominates the trace. Tune the active RTX 3080 `mul_mat_vec_ptq1_0_pt` kernel and its planar-transposed Q8 layout; prior warp-count tuning edited a bypassed path, and SOA side-code/floor screens did not cover the active sm_86 layout.
2. Determine whether PTQ1_0's lower payload is converting into DRAM traffic reduction or whether instruction/occupancy limits dominate. NCU counters are unavailable to this user; use static resources, controlled size sweeps, and end-to-end evidence without changing system counter permissions.
3. Test whether fused Hadamard-to-Q8_1 activation preparation for PQ2_0 recovers its extra transform/quantization kernels; then revisit launch gaps, RMSNorm, and gated-delta recurrent attention after a matvec improvement changes the ranking.
