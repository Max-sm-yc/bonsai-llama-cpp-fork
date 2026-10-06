# Research state

## Current best

- **PTQ1_0, active sm_86 planar GEMV with ROWS=1.** Code commit `9fa97200e68fd798ef027470c8e420172a0ac719`; reference runtime commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`. Rebuilt source-default medians: 82.22 tok/s at context 512 and 79.70 at 4096 (7 reps, 128 decode tokens, F16 KV, FA on, 99 GPU layers, 8 CPU threads); matched archived ROWS=4 medians: 78.00/75.66, or +5.42%/+5.34%. Peak whole-GPU memory 6,805 MiB. Prefill not remeasured; reference medians are 1,378/1,331 tok/s at 512/4096.
- Correctness: final source-default ROWS=1 passed 4 CTests, 96 CUDA-vs-CPU PTQ1_0/PQ2_0 matmul cases, and both fixed 32-token model smokes. See experiment 010 report and raw results.

## Bottlenecks

1. PTQ1_0 batch-1 `mul_mat_vec_ptq1_0_pt` remains dominant at 60.4% (1.166 s) in the post-ROWS=1 mixed context-512 trace.
2. PTQ1_0 GEMM: 12.6% (242.8 ms); gated delta attention 4.6% (88.0 ms); fused FWHT/Q8_1 4.4% (85.1 ms); RMSNorm 3.9% (76.2 ms).
3. Nsight Compute counters remain blocked by `ERR_NVGPUCTRPERM`; do not alter system-wide driver permissions.

## Successful optimizations

- Experiment 010: ROWS=1 lowers the active specialization from 108 to 76 registers/thread (no spills) and raises paired median decode by 4.9–5.8% vs ROWS=4; the manager's rebuilt A/B measured +5.42%/+5.34% at contexts 512/4096. ROWS=1 edges ROWS=2 by 0.65–0.73% in two direct pairs.

## Failed or exhausted approaches

- Experiments 001/002: disabling PTQ1_0 L2 prefetch tied within noise or varied with run order.
- Experiment 003 changed a generic path bypassed by sm_86 batch-1 dispatch. Experiments 004–009 either targeted SOA rather than this planar kernel or failed/slowed their focused test; see reports.
- Experiment 010 ROWS=8 lost 16–17%. Its first archived-binary screens and follow-ups loaded the same `build/bin` library due absolute RUNPATH; treat those timings as invalid. Corrected per-library runs are marked `_isolated` and verified with `ldd`/`LD_DEBUG`.
- Experiment 011: exact planar 2-bit side codes lost to base-3 by 3.02–7.01% (scalar) and 3.10–4.68% (packed-byte expansion) at 16K/65K blocks, before conversion; no model integration. Manager rebuilt and independently rechecked exact outputs.
- Experiment 012: pairwise `x*9` trit decode became exact after correcting an activation-word index, but lost 1.97–24.59% in full-block dot screens at 65K/16K/128 blocks; no model integration.
- Experiment 013: CTA row-tile caps 4/8/24/32 and a larger shared-memory target did not improve decode; the best-looking cap-8 candidate tied ROWS=1 within 0.11% / -0.01% at contexts 512/4096. Larger tiles lost, with cap-32/8192 notably slower. Exact ROWS=1 source and library restoration were independently hash-verified.
- Experiment 014: inspected a warp-per-row reduction idea but stopped before implementing or measuring it. This is no performance evidence; the kernel hypothesis remains unresolved.
- Experiment 015: implemented four warps/CTA, one warp/output-row with register K accumulation; it passed selected correctness/model checks but lost 3.27–3.79% in two reversed-order decode pairs. Source/library restoration hashes match the ROWS=1 control.
- Experiment 016: the multiwarp-per-row follow-up ended before implementation; no performance or correctness evidence. The hypothesis is still open.
- Experiment 017: two warps/output row (four warps/CTA) passed selected correctness and smoke checks but lost 2.60% at context 512 and 1.82% at 4096 in a seven-rep pair. The candidate lane split is uneven for common 40-block K rows; source and active library were independently restored.
- Experiment 018: shape-gated four warps/output row for K>64 passed correctness but showed no repeatable decode gain (512: -0.32%/-0.69%; 4096: +0.77%/-0.58% across reversed pairs). Exact ROWS=1 source/library hashes were restored.
- Experiment 019: per-branch RMSNorm→FWHT/Q8_1 fusion was not implemented; the attention-normalized activation feeds Q/K/V projection branches, so this route cannot discard the shared normalized tensor. No performance evidence; see report.

## Important discoveries

- RTX 3080/sm_86 selects planar-transposed Q8_1 and dedicated `mul_mat_vec_ptq1_0_pt` for plain batch-1 PTQ1_0. ROWS=1 changes the one-column work mapping only; other column counts retain the existing schedule.
- Median decode gains repeat, but context-4096 samples have intermittent slow tails in both ROWS=1 and ROWS=4 builds. Keep means/ranges with medians; do not hide outliers.
- PTQ1_0 remains faster than PQ2_0 by 32–54% in the controlled reference format comparison; prefill is nearly tied. Both models fit in VRAM.
- CUDA Graphs are already active (127 graph launches in the context-512 mixed trace); prioritize measured device work/fusion over generic launch-overhead changes.
- The FWHT→Q8_1 path is already one kernel, and model attention normalization has Q/K/V fanout. Do not retry per-branch RMS fusion without a coordinated consumer design and measured evidence that it can avoid extra reductions.

## Next candidates

1. Optimize the already-fused PTQ1_0 FWHT/Q8_1 activation kernel (4.4% of mixed trace) independently; test actual sm_86 PT layout and judge by full decode.
2. Investigate the gated-delta network kernel (4.6%) or its remaining data movement on the Bonsai hybrid attention path.
3. Consider the measured RMSNorm path (3.9%) independently, then revisit PQ2_0 activation fusion.
