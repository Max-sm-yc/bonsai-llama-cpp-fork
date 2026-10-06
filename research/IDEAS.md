# Candidate hypotheses

Ranked against the current ROWS=1 implementation. Controlled batch-1 model decode is the decision metric; isolated kernel gains are screening evidence only.

1. Audit the active fused-gate PTQ1_0 specialization for duplicated planar Q8_1 activation loads across its main and gate weight dots. The specialization accounts for 15.5% (298.6 ms) of the post-ROWS=1 mixed context-512 kernel trace. Inspect SASS first; implement shared activation loads only if the compiler has not already reused them. Preserve each weight recurrence, output partial, and fold order; compare static resource use and full-model decode.
2. If the gate-path audit shows existing load reuse, audit the `has_fusion=true, has_gate=false` specialization (14.3%, 276.3 ms) for a concrete unshared activation or address-generation cost before coding.
3. Consider targeted PQ2_0 activation fusion after higher-value PTQ1_0 work.

Preserve ROWS=1. Avoid repeating row-tile geometry, warp-per-row splits, the scalar decoder, two-bit side encodings, pairwise trit decode, the eight-lane production recurrence, or 2/4-item source unrolling without a materially new premise. Experiment 024's isolated 1.49x recurrence gain regressed model decode about 81.5%; experiment 025's source strip mining did not change static resources and lost about 0.4–1.4% end-to-end.

The active PTQ1_0 GEMV family is still the largest measured cost: 60.4% (1.166 s) of the post-ROWS=1 mixed context-512 Nsight Systems trace. The plain specialization contributes 30.6% (590.8 ms / 31,219 launches); the fused-gate and fused no-gate variants contribute 15.5% (298.6 ms / 5,161 launches) and 14.3% (276.3 ms / 10,192 launches). These are setup/decode trace totals, not decode-only attribution. Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, focused CUDA-event tests, and matched end-to-end workload sweeps.

Experiment 021 confirmed graph gather fusion is active: the 16-token trace had 864 GDN calls and no GET_ROWS; disabling fusion added exactly 864 GET_ROWS calls. Do not repeat the column-per-warp sweep without a new kernel design. Experiments 019–020 exhausted the current per-branch RMSNorm→FWHT/Q8_1 fusion and FWHT CTA-width sweeps. Experiment 022 showed a global switch to 256-thread fused-weight RMSNorm regresses decode about 3.3%; do not retry without measured shape-specific evidence.
