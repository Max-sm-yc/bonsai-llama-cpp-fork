# Candidate hypotheses

Ranked after experiment 022. Use controlled end-to-end decode as the decision metric.

1. Challenge the dominant PTQ1_0 GEMV with a materially different exact sm_86 dataflow. Prior fixed and shape-gated multiwarp row reductions, side-code formats, and pairwise trit decoding did not win; do not repeat those variants without a concrete architectural change.
2. Explore RMSNorm only with measured shape data and a different implementation idea than the rejected global 256-thread geometry. Standard RMSNorm plus learned weight is already fused, and per-branch fusion must preserve the Q/K/V shared normalized activation.
3. Consider activation fusion or other targeted work on PQ2_0 after higher-value PTQ1_0 paths.

Experiment 021 screened GDN columns-per-warp 1/2/4/8 for the active S_v=128 scalar raw-gate specialization. No mapping change had a repeatable end-to-end gain. Do not repeat this sweep without a new kernel design or dispatch premise. The default graph already fuses recurrent-state gather into GDN; disabling that rewrite adds one GET_ROWS launch per observed GDN call. Cache-copy fusion was inspected in source but not independently toggled.

Experiments 019–020 exhausted the current per-branch RMSNorm→FWHT/Q8_1 fusion and FWHT CTA-width sweeps. Do not retry without a coordinated consumer design or a concrete kernel change.

Nsight Compute counters are unavailable (ERR_NVGPUCTRPERM); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, and controlled workload sweeps.

Experiment 022 corrected the RMSNorm family accounting: the 1024-thread signature is 3.94% of the mixed trace, but adding 10,400 calls / 24.43 ms from the 256-thread signature makes the family 5.21%. A global switch of the fused-weight `ncols >= 1024` branch from 1024 to 256 threads regressed decode by about 3.3%; do not repeat without a shape-specific or algorithmic reason.
