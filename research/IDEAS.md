# Candidate hypotheses

Ranked against the current ROWS=1 implementation. Controlled batch-1 model decode is the decision metric; isolated kernel gains are screening evidence only.

1. Compare default global caching with `.cg` L1-bypass or `.cs` streaming policy for aligned packed PTQ1_0 weight-word loads in the active plain GEMV. Weights stream once per token while activation planes repeat across output rows; test whether limiting weight residency protects useful activation L1 data. Preserve the 28-byte layout and arithmetic; require an active-kernel screen and matched end-to-end decode if promising.
2. Consider a different packed-weight staging/dataflow for the 28-byte PTQ1_0 block, with correctness and instruction/resource checks before integration.
3. Fused-gate PTQ1_0 accounts for 15.5% (298.6 ms / 5,161 launches), but its active SASS already shares the nine 128-bit Q8_1 activation loads with the ungated dot. Only pursue paired-dot scheduling if the audit identifies redundant non-load work. PQ2_0 decode remains another lower-ranked path.

Preserve ROWS=1. Avoid repeating row-tile geometry, warp-per-row splits, the scalar decoder, two-bit side encodings, pairwise trit decode, the eight-lane production recurrence, or 2/4-item source unrolling without a materially new premise. Experiment 024's isolated 1.49x recurrence gain regressed model decode about 81.5%; experiment 025's source strip mining did not change static resources and lost about 0.4–1.4% end-to-end.

The active PTQ1_0 GEMV family is still the largest measured cost: 60.4% (1.166 s) of the post-ROWS=1 mixed context-512 Nsight Systems trace. The plain specialization contributes 30.6% (590.8 ms / 31,219 launches); the fused-gate and fused no-gate variants contribute 15.5% (298.6 ms / 5,161 launches) and 14.3% (276.3 ms / 10,192 launches). These are setup/decode trace totals, not decode-only attribution. Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, focused CUDA-event tests, and matched end-to-end workload sweeps.

Experiments 001/002's `mmvq.cu` prefetch toggle does not reach the dedicated one-column PTQ1_0 kernel used by the RTX 3080 decode path. Experiment 026's gated specialization has the same nine 128-bit activation loads as ungated; do not pursue duplicate-load elimination without codegen evidence.

Experiment 021 confirmed graph gather fusion is active: the 16-token trace had 864 GDN calls and no GET_ROWS; disabling fusion added exactly 864 GET_ROWS calls. Do not repeat the column-per-warp sweep without a new kernel design. Experiments 019–020 exhausted the current per-branch RMSNorm→FWHT/Q8_1 fusion and FWHT CTA-width sweeps. Experiment 022 showed a global switch to 256-thread fused-weight RMSNorm regresses decode about 3.3%; do not retry without measured shape-specific evidence.
