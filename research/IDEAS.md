# Candidate hypotheses

Ranked against the current ROWS=1 implementation. Controlled batch-1 model decode is the decision metric; isolated kernel gains are screening evidence only.

1. Test explicit 2/4-item software pipelining of independent K-block work in the existing 128-thread ROWS=1 PTQ1_0 GEMV. Keep each K-block's recurrence serial and exact; preserve work-item addresses, output partial slots, and ordered final reduction. The goal is to expose memory/load latency and decode-chain instruction-level parallelism without changing lane ownership or adding warp communication. Compare register use, spills, a kernel-level screen, and model decode.
2. Audit the `has_gate` PTQ1_0 specialization for redundant activation loads. Proceed only if generated code shows non-shared loads; a fused main/gate weight calculation should load the same planar Q8_1 fragments once while maintaining exact weight recurrences and each output's existing fold order.
3. Consider targeted PQ2_0 activation fusion after higher-value PTQ1_0 work.

Preserve ROWS=1. Avoid repeating row-tile geometry, warp-per-row splits, the scalar decoder, two-bit side encodings, pairwise trit decode, or the eight-lane production recurrence mapping without a materially new premise. Experiment 024's isolated 1.49x recurrence gain regressed model decode by about 81.5% at both tested contexts after integration.

The active PTQ1_0 GEMV family is still the largest measured cost: 60.4% (1.166 s) of the post-ROWS=1 mixed context-512 Nsight Systems trace, with 31,219 plain launches and 15,353 fused-specialization launches. It accounts for 60.4% in a setup/decode trace, not a decode-only attribution. Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, focused CUDA-event tests, and matched end-to-end workload sweeps.

Experiment 021 confirmed graph gather fusion is active: the 16-token trace had 864 GDN calls and no GET_ROWS; disabling fusion added exactly 864 GET_ROWS calls. Do not repeat the column-per-warp sweep without a new kernel design. Experiments 019–020 exhausted the current per-branch RMSNorm→FWHT/Q8_1 fusion and FWHT CTA-width sweeps. Experiment 022 showed a global switch to 256-thread fused-weight RMSNorm regresses decode about 3.3%; do not retry without measured shape-specific evidence.
