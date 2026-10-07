# Research state

## Current best

- **PTQ1_0, sm_86 ROWS=1 GEMV plus coordinated RMS/FWHT/Q8 preparation.** Code commit `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`; reference runtime commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`. Same-binary, two order-reversed 7-repetition comparisons: enabled median-of-run-medians 83.35 tok/s at context 512 and 80.35 at 4096 versus disabled 81.99/79.13 (+1.65%/+1.55%). Peak whole-GPU memory 6,803 MiB enabled / 6,805 disabled. Prefill was not remeasured; reference medians remain 1,378/1,331 tok/s at 512/4096.
- Correctness: rebuilt main checkout passed `tests/run_correctness.sh`: selected CTests 4/4, CUDA-vs-CPU PTQ1_0/PQ2_0 backend cases 96/96, and both fixed-seed 32-token model smokes. The PTQ1_0 normalized completion exactly matches the previous default. Main CUDA library SHA-256 `4b4adb58e3d26cb8694aebf0843112b981de66ac54441760ec8290bb2b021dcf`.

## Bottlenecks

1. PTQ1_0 batch-1 `mul_mat_vec_ptq1_0_pt` remains dominant at 61.2% (1.166 s) in the post-Exp036 mixed context-512 trace.
2. PTQ1_0 GEMM: 12.7% (242.6 ms).
3. Coordinated activation prep (standard plus fused RMS/FWHT/Q8): 5.2% (98.6 ms); gated delta attention: 4.6% (88.2 ms).
4. Remaining RMSNorm family: 3.0% (56.8 ms). Nsight Compute counters remain blocked by `ERR_NVGPUCTRPERM`; do not alter system-wide driver permissions.

## Successful optimizations

- Experiment 010: ROWS=1 lowers the active specialization from 108 to 76 registers/thread (no spills) and raises paired median decode by 4.9–5.8% vs ROWS=4; the manager's rebuilt A/B measured +5.42%/+5.34% at contexts 512/4096. ROWS=1 edges ROWS=2 by 0.65–0.73% in two direct pairs.
- Experiment 036: exact-shape, use-count-guarded coordinated RMS→weighted multiply→sign→FWHT/Q8 preparation. Same-binary reversed-order decode pairs improved +1.65%/+1.55% at contexts 512/4096; combined RMSNorm plus activation-preparation time fell 30.4 ms in the mixed profile. Default on; `GGML_CUDA_RMS_FWHT_Q8=0` disables it. See report and raw A/B under `experiments/036-coordinated-qkv-prep/` and `results/exp036/`.

## Failed or exhausted approaches

- Experiments 001/002: dispatch audit found their `mmvq.cu` prefetch edit is bypassed by the dedicated sm_86 batch-1 planar kernel; decode data are no-op comparisons, not prefetch evidence. Exp003 edited the same generic dispatch region and is likewise a no-op for target decode.
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
- Experiment 020: FWHT/Q8_1 NT=128 lost about 0.9–1.0%; NT=512 tied after reversed-order pairs (+0.15%/+0.06%). Preserve NT=256; A/B correctness and samples are in the report.

- Experiment 021: GDN columns-per-warp 1/2/4/8 had no repeatable decode winner; column 1 fell from +0.52/+0.41% to +0.10/+0.11% in reversed order, while the 2/8 screens were 0.05–0.32% below the later control without reversed ordering. See report and samples.
- Experiment 022: changing the active fused-weight RMSNorm branch from 1024 to 256 threads lost 3.32%/3.25% decode at contexts 512/4096. Fixed-seed model smokes matched; the broad standalone correctness suite was interrupted during an unnecessary full rebuild after the clear regression. Keep the 1024-thread branch.
- Experiment 023: a four-lane-per-block screen with scalar trit extraction was exact but 3.82x slower than its scalar reference. It does not test partitioning the production packed recurrence and has no E2E candidate result.
- Experiment 024: the production packed recurrence's cooperative-eight-lane screen was 1.49x faster in isolation (16,384 blocks), exact, and sanitizer-clean; integrating it into the active GEMV regressed decode 81.47% at context 512 and 81.50% at 4096. Reverted and hash-restored; do not repeat this mapping. See report 024.
- Experiment 025: explicit 2/4-item strip mining preserved correctness but lost 0.42–0.52% at context 512 and 0.42–1.36% at 4096; the active specialization stayed at 76 registers/thread with no stack/local storage. No evidence of useful ILP from source unrolling alone. Reverted and hash-restored.
- Experiment 026: SASS audit found nine 128-bit activation loads in both gated and ungated fused GEMV variants; compiler already reuses the activation vectors. No paired-helper candidate was built.
- Experiment 027: active-kernel current-block prefetch has no useful lookahead; 28-byte weight-block spacing complicates vector loads. Static challenge ended before a candidate/event test, so alternate dataflows remain open.
- Experiment 028: per-thread next-item lookahead was address-safe and emitted active `CCTL.E.PF1/PF2`, but one trace per variant showed 2.3% (distance 1) and 7.4% (distance 2) more plain-kernel time. Reverted at the focused screen; no E2E claim.
- Experiment 029: `.cg` reached the active plain GEMV's six packed-word loads and retained 74 registers/no spills, but only one candidate trace was captured (4,115 launches; 177.969 ms). No matched control, correctness comparison, or E2E A/B exists; classify as inconclusive. Source was restored and rebuilt; see report for the distinct binary-hash recovery note.
- Experiment 031: padding PTQ1_0 blocks from 28 to 32 bytes enabled the intended two 128-bit loads and exact outputs, but lost 3.37% at 16K blocks and 21.39% at 65K; reject this layout and its +14.29% payload cost. No runtime integration.
- Experiment 033: a full CUDA SoA conversion has no safe GEMV-only hook: multi-column/MMQ/vector-dot/utility readers and generic transfer callbacks assume AoS. A distinct selective sidecar for the 64 K=17,408 `ffn_down` tensors would cost 1,190 MiB and projects to 7,995 MiB total; not measured.
- Experiments 034–035: a buffer-owned selective SoA sidecar was implemented in isolation and reached the actual 64 long-K `ffn_down` GEMVs correctly, but repeated decode was tied/noisy and it cost +1,280 MiB peak GPU memory plus ~294 ms load repacking. Keep AoS; do not repeat without a new traffic premise.

## Important discoveries

- RTX 3080/sm_86 selects planar-transposed Q8_1 and dedicated `mul_mat_vec_ptq1_0_pt` for plain batch-1 PTQ1_0. ROWS=1 changes the one-column work mapping only; other column counts retain the existing schedule.
- Median decode gains repeat, but context-4096 samples have intermittent slow tails in both ROWS=1 and ROWS=4 builds. Keep means/ranges with medians; do not hide outliers.
- PTQ1_0 remains faster than PQ2_0 by 32–54% in the controlled reference format comparison; prefill is nearly tied. Both models fit in VRAM.
- CUDA Graphs are already active (127 graph launches in the context-512 mixed trace); prioritize measured device work/fusion over generic launch-overhead changes.
- Full-attention Q/K/V share a memoized activation transform. Exp036 added a guarded coordinated RMS/weight/sign/FWHT/Q8 path; five CTAs repeat a 5120-wide RMS reduction. It reduces combined norm/preparation time in the mixed trace and improves same-binary decode medians modestly. Keep the shape/use-count guards and disable override.
- The active GDN trace is S_v=128, scalar-gate (`KDA=false`), raw-gate (`RAW=true`); on sm_86 it already uses four columns per warp, fused cache/gather paths, and CUDA Graphs.
- The RMSNorm profile has 16,770 calls / 76.18 ms for `<1024,true,false>` and 10,400 / 24.43 ms for `<256,true,false>`; Nsight recorded CTA sizes but not `ncols`. Global reduction to 256 threads on the fused-weight `ncols >= 1024` path regressed full-model decode, so retain its current geometry.
- Experiment 023's scalar extraction cost dominated its cooperative microbenchmark; 024 then tested the actual packed recurrence and showed that its isolated 1.49x screen did not translate to production decode.
- Experiment 025's work-list strip mining changed neither static register usage nor model throughput favorably; do not retry source unrolling alone without disassembly or kernel-counter evidence.
- Experiments 001/002 changed only the generic PTQ1 prefetch, while the RTX 3080 batch-1 decode takes the dedicated kernel. Experiment 026's gated/ungated SASS each has nine 128-bit activation loads, so do not pursue load reuse there without a new codegen premise.
- The dedicated GEMV assigns `(row group,K block)` work items with per-thread loop stride 128. Prefetching that thread's next work item is a distinct lookahead candidate; current-block prefetch is not.
- Experiment 032: a same-footprint seven-plane SoA layout was exact and sanitizer-clean. Repeated 2,048-row screens tied at 40 K blocks (-0.22% manager rerun) and won at 136 blocks (-7.82%); this is block-dot evidence only, with no production/runtime result.
- The work-list lookahead adds index/address instructions before the dot and did not repay that overhead in the first actual-kernel trace. PTQ1 packed `qs` words remain naturally 4-byte aligned even though 28-byte block starts are not 16-byte aligned. Exp029's `.cg` modifier emitted `LDG.E.STRONG.GPU` for these loads while the nine activation vector loads remained cached; its profile lacks a contemporaneous control and says nothing about speed.
- Exp029 showed that interrupted broad Ninja rebuilds can remove the linked CUDA library and leave missing objects. The source-default library has now been rebuilt and passed a model smoke, but its byte hash differs from the archived best library; preserve the active library and use isolated candidate relinks for future cache-policy tests.

- Experiment 021 confirmed graph gather fusion is active: the 16-token trace had 864 GDN calls and no GET_ROWS; disabling fusion added exactly 864 GET_ROWS calls. Cache-copy fusion was source-audited, not toggled.

## Next candidates

1. Fresh challenge of the active sm_86 PTQ1_0 GEMV dataflow. It remains 61.2% of mixed-trace kernel time after Exp036; seek a materially different packed-weight/activation mapping and screen it on the actual ROWS=1 specialization before model A/B.
2. Revisit PQ2_0 or prompt-side work only if a format-specific or workload-level change has a stronger expected decode impact than the active PTQ1_0 GEMV.

- Experiment 030: matched default/`.cg`/`.cs` screen completed with three actual-kernel traces per arm (485 target launches each). `.cg` was +128.4% target-kernel time; `.cs` was -0.30% in the kernel screen but lost 0.9–1.1% end-to-end median throughput in the reversed-order 7-rep comparison at contexts 512/4096. Reverted; production source, active source-default library, and backup hashes are intact. No exact correctness comparison was completed, so no candidate was retained. See report 030 and `results/exp030/`.
