# Candidate hypotheses

Ranked after experiment 021. Use controlled end-to-end decode as the decision metric.

1. Profile standalone RMSNorm on sm_86 and investigate geometry changes or a safe fusion with a single-consumer neighbor. Standard RMSNorm plus learned weight is already fused, and per-branch fusion must preserve the Q/K/V shared normalized activation.
2. Challenge the dominant PTQ1_0 GEMV with a materially different decode design. Prior fixed and shape-gated multiwarp row reductions, side-code formats, and pairwise trit decoding did not win; do not repeat those variants without a concrete architectural change.
3. Consider activation fusion or other targeted work on PQ2_0 after higher-value PTQ1_0 paths.

Experiment 021 screened GDN columns-per-warp 1/2/4/8 for the active S_v=128 scalar raw-gate specialization. No mapping change had a repeatable end-to-end gain. Do not repeat this sweep without a new kernel design or dispatch premise. The default graph already fuses recurrent-state gather into GDN; disabling that rewrite adds one GET_ROWS launch per observed GDN call. Cache-copy fusion was inspected in source but not independently toggled.

Experiments 019–020 exhausted the current per-branch RMSNorm→FWHT/Q8_1 fusion and FWHT CTA-width sweeps. Do not retry without a coordinated consumer design or a concrete kernel change.

Nsight Compute counters are unavailable (ERR_NVGPUCTRPERM); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, and controlled workload sweeps.
