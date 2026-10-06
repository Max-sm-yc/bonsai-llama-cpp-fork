# Candidate hypotheses

Ranked after experiment 023. Use controlled end-to-end decode as the decision metric.

1. Test a group-aware warp mapping that partitions the active packed PTQ1_0 base-3 recurrence itself. Experiment 023's four-lane scalar-extraction prototype was exact but 3.82x slower than a scalar reference and did not measure the active recurrence; do not reuse its decoder or interpret it as a production-path result.
2. Explore RMSNorm only with measured shape data and a different implementation idea than the rejected global 256-thread geometry. Standard RMSNorm plus learned weight is already fused, and per-branch fusion must preserve the Q/K/V shared normalized activation.
3. Consider activation fusion or other targeted work on PQ2_0 after higher-value PTQ1_0 paths.

Preserve ROWS=1. Avoid fixed/shape-gated multiwarp row reductions, side-code formats, and pairwise trit decoding without a materially new premise.

Experiment 021 screened GDN columns-per-warp 1/2/4/8 for the active S_v=128 scalar raw-gate specialization. No mapping change had a repeatable end-to-end gain. Do not repeat this sweep without a new kernel design or dispatch premise. The default graph already fuses recurrent-state gather into GDN; disabling that rewrite adds one GET_ROWS launch per observed GDN call. Cache-copy fusion was inspected in source but not independently toggled.

Experiments 019–020 exhausted the current per-branch RMSNorm→FWHT/Q8_1 fusion and FWHT CTA-width sweeps. Do not retry without a coordinated consumer design or a concrete kernel change.

Nsight Compute counters are unavailable (ERR_NVGPUCTRPERM); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, and controlled workload sweeps.

Experiment 022 corrected the RMSNorm family accounting: the 1024-thread signature is 3.94% of the mixed trace, but adding 10,400 calls / 24.43 ms from the 256-thread signature makes the family 5.21%. A global switch of the fused-weight `ncols >= 1024` branch from 1024 to 256 threads regressed decode by about 3.3%; do not repeat without a shape-specific or algorithmic reason.

Experiment 023 screened four lanes per block using independent scalar ternary extraction. It is bitwise correct but 3.82x slower; this does not answer whether lane cooperation helps when applied to the production packed recurrence.
