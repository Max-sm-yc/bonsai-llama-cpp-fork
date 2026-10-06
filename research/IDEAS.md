# Candidate hypotheses

Ranked after experiment 013 and the ROWS=1 re-profile. Judge every candidate by controlled end-to-end decode and combined throughput.

1. If continuing multiwarp reductions, specialize 4 warps/output row only for large-K shapes (>64 blocks), where lanes can cover the 136-block projection with little imbalance. Fixed one- and two-warp row splits lost 1.8–3.8% in experiments 015/017.
2. Revisit Hadamard-to-Q8_1 fusion for PQ2_0 after one distinct PTQ1 reduction experiment; then inspect recurrent attention, RMSNorm, and launch gaps if their share rises after PTQ1 improvements.
3. Consider another same-format trit-unpack design only if it differs materially from the tested constant LUT, floor-difference, pairwise `x*9`, and exact 2-bit side approaches; the active planar variants lost their focused comparisons.

Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, and controlled size/workload sweeps.
