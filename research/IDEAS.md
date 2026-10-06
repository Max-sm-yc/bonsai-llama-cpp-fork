# Candidate hypotheses

Ranked after experiment 013 and the ROWS=1 re-profile. Judge every candidate by controlled end-to-end decode and combined throughput.

1. Test cooperative 2/4-warp-per-output-row reductions for active planar `mul_mat_vec_ptq1_0_pt`: recover K-block parallelism while combining each warp's K partials before the CTA reduction. Experiment 015 showed the one-warp-per-row, serial K-loop design loses 3.3–3.8%; do not repeat it.
2. Revisit Hadamard-to-Q8_1 fusion for PQ2_0 after one distinct PTQ1 reduction experiment; then inspect recurrent attention, RMSNorm, and launch gaps if their share rises after PTQ1 improvements.
3. Consider another same-format trit-unpack design only if it differs materially from the tested constant LUT, floor-difference, pairwise `x*9`, and exact 2-bit side approaches; the active planar variants lost their focused comparisons.

Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, and controlled size/workload sweeps.
