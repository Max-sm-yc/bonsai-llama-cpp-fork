# Candidate hypotheses

Ranked after experiment 012 and the ROWS=1 re-profile. Judge every candidate by controlled end-to-end decode and combined throughput.

1. Tune rows-per-CTA and shared-memory reduction geometry in the active planar `mul_mat_vec_ptq1_0_pt` path while holding the verified ROWS=1 item mapping. Its host heuristic caps the row tile at 16 and scores fill efficiency, but context-shape occupancy/E2E impact is untested.
2. Consider another same-format trit-unpack design only if it differs materially from the tested constant LUT, floor-difference, and pairwise `x*9` methods; the exact 2-bit side codes and pairwise decoder both lost in the active planar dot.
3. Revisit Hadamard-to-Q8_1 fusion for PQ2_0; then inspect recurrent attention, RMSNorm, and launch gaps if their share rises after PTQ1 improvements.

Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, and controlled size/workload sweeps.
