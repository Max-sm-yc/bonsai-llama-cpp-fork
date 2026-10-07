# Research state

## Current best

- PTQ1_0 on RTX 3080/sm_86: ROWS=1 planar batch-1 GEMV plus Exp036 coordinated QKV RMS/FWHT/Q8 prep. Production code commit `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`; reference runtime `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`.
- Same-binary, two reversed-order pairs of seven decode runs: 83.35 tok/s at context 512 and 80.35 at 4096; Exp036 disabled control 81.99/79.13 (+1.65%/+1.55%). Direct Exp041 reference/current A/B: +6.82%/+5.76% decode at contexts 512/4096, with 6,803 MiB versus 6,805 MiB peak GPU memory. Prefill measured 1,291.9/1,377.4/1,355.5/1,332.3 tok/s at contexts 128/512/2048/4096 and matches the frozen reference within 0.09%.
- Correctness: `tests/run_correctness.sh` passed selected CTests 5/5, backend CUDA-vs-CPU cases 96/96, and fixed-seed 32-token PTQ1_0/PQ2_0 smokes. Current main CUDA library SHA-256 `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`.

## Bottlenecks

1. Active PTQ1_0 batch-1 GEMV (`mul_mat_vec_ptq1_0_pt`): 9.014 ms/token at context 512 and 9.022 ms/token at 4096, 75.9%/73.8% of summed steady-state graph kernel duration. Its plain, fused-gate, and fused non-gate specializations cost ~4.57, ~2.31, and ~2.14 ms/token.
2. QKV activation preparation: 0.752 ms/token (6.2–6.3%); GDN: 0.500 ms/token (4.1–4.2%, no distinct candidate after Exp049); BF16 `mul_mat_vec_f<__nv_bfloat16,float,1,256,false,false>`: ~0.311 ms/token (~2.6%). Remaining RMSNorm: 0.364 ms/token; attention rises from 0.236 ms/token at context 512 to 0.588 ms at 4096.
3. The earlier context-512 post-Exp036 trace's 1.166 s / 61.2% GEMV share is a mixed setup/decode denominator. Exp047 directly grouped 255 graph replays with 1,432 nodes/replay; it contains no quantized GEMM graph nodes.
4. Nsight Compute counters fail with `ERR_NVGPUCTRPERM`; do not alter system-wide permissions. See [Exp047](../experiments/047-steady-decode-profile/REPORT.md) for filters, reproducibility, and variability.

## Successful optimizations

- Exp010: ROWS=1 lowered active specialization registers from 108 to 76/thread (no spills); matched decode improved +5.42%/+5.34% at contexts 512/4096 versus ROWS=4. ROWS=2 was 0.65–0.73% slower.
- Exp036: shape/use-count-guarded coordinated RMS→weight/sign→FWHT/Q8 prep reduced combined norm/prep trace time by 30.4 ms and improved same-binary decode +1.65%/+1.55%. Default on; `GGML_CUDA_RMS_FWHT_Q8=0` disables it.

## Failed or exhausted approaches

- Dispatch, decoder, and GEMV scheduling: generic-path edits in 001–003 do not reach the active sm_86 batch-1 path. LUT/floor, two-bit, pairwise, fixed-point, row-tile, warp/multiwarp reduction, recurrence distribution, and source strip-mining attempts (004–025, 039) were invalid for the target or exact but slower/no better. Exp024's isolated 1.49x cooperative recurrence screen became an 81.5% model regression.
- Prefetch/cache/load/layout variants: next-item hints and `.cs` lost E2E; `.cg` lost its kernel screen; padding to 32B added 14.3% payload and lost. Exp032's same-footprint SoA block screen gained 7.8% at 136 blocks but tied at 40; the runtime sidecar in 034–035 had no repeatable decode gain, added 1,280 MiB peak VRAM, and cost ~294 ms load time. Synchronous shared staging (037) lost 9.9–12.9%; warp-register transpose (038) lost 2.42–2.75x. The measured sm_86 `cp.async` next-work-item pipeline (040) emitted real async transfers but lost 10.4%/23.2% at 40/136 blocks. CTA widths 64/128/256/512 (042) lowered active-kernel registers at 256/512 but did not win both K shapes; a 256-at-K40/128-otherwise dispatch regressed matched decode 0.60%/0.68%.
- Register pressure/occupancy: Exp045 held 128-thread ROWS=1 geometry fixed and asked for 6/7/8 resident CTAs on `ncols==1`. Active plain/gated register use fell to 72/77, 68/72, and 60/63 with no active spills, but all three short reversed model screens regressed at contexts 512/4096 (-2.5% to -5.1%). Keep the four-CTA bound.
- Inconclusive only: 014/016/027 ended before candidate measurement; 029 had one unmatched `.cg` trace. See `research/EXPERIMENTS.md` and linked reports before revisiting any idea.

## Important architectural discoveries

- RTX 3080 batch-1 PTQ1_0 uses planar-transposed Q8_1 and dedicated `mul_mat_vec_ptq1_0_pt`; generic `mmvq.cu` tunings do not apply. It processes flattened `(row group,K block)` work items with per-thread stride 128 and 128-thread CTAs.
- PTQ1_0 reference decode was 32–54% faster than PQ2_0 in the initial format comparison; prefill was nearly tied. Exp041 later measured current versus frozen-reference decode directly (+6.82%/+5.76% at contexts 512/4096) and prefill within 0.09%. Do not use the original matrix as a speedup denominator.
- CUDA Graphs are active; the old context-512 trace had 127 graph launches but mixed setup/decode. Exp047 groups 255 direct `-p 0` token replays at 1,432 nodes each, with no prompt-side quantized GEMM nodes. Q/K/V share memoized activation transforms; keep Exp036 guards.
- Exp052 compared the formats under the same runtime: each replay has 361 GEMV nodes in both formats, but PQ2_0 uses `mul_mat_vec_q<type 142>` and averages 10.71 ms versus PTQ1_0's dedicated planar `mul_mat_vec_ptq1_0_pt` at 9.02 ms. PQ2_0 also has 441 extra nodes/replay and ~0.59 ms more activation-prep plus standalone RMSNorm time. The PTQ1_0 model payload is 17.5% smaller. Systems timing cannot distinguish weight traffic from decoder/instruction/occupancy effects; Nsight Compute counters remain unavailable. See [Exp052](../experiments/052-pq2-steady-profile/REPORT.md).
- Repeated context-4096 decode samples have slow tails in both arms. Preserve all repetitions/ranges and use medians. Verify candidate library paths with `ldd`/`LD_DEBUG`; earlier absolute RUNPATHs caused false A/Bs.

## Latest research result

- Exp052 measured matched PTQ1_0/PQ2_0 decode in two reversed-order pairs: +19.0% at context 512 and +34.7% at 4096 by median of run medians; long-context samples have slow tails. Node-level captures had 31 full replays per format/context. PTQ1_0's GEMV family costs 9.02–9.03 ms/token versus PQ2_0 at 10.71 ms with identical per-signature launch counts; PTQ1_0 also avoids 441 graph nodes and ~0.59 ms/token of separate Q8/RMS work. No production source changed. Nsight Compute did not provide counters; no bandwidth or integer-pipe conclusion is claimed. See `experiments/052-pq2-steady-profile/REPORT.md` and `results/exp052/`.

- Exp053 tested the active sm_86 Ampere FlashAttention configuration. The valid 64/64 single-stage tile passed correctness but regressed focused attention timing by 17.2% at context 512 and 11.3% at 4096; its 4096 Stream-K fixup rose from 0.0359 to 0.0920 ms/token. The 96/96 option failed a compile-time loop invariant. Source restored; no model A/B was run. Close tile-size sweeps unless a new premise reduces Stream-K fixup cost. See `experiments/053-flash-attention-longctx/REPORT.md`.

- Exp054 found Q+gate is already one `wq` projection; K/V share `cur` and 2048-wide output but need separate outputs and K-only norm/RoPE. The active GEMV gate path is not generic paired-output support, and profiles cannot attribute GEMV launches to Q/K/V. No code or measurements. The concrete follow-up is a dedicated K/V output-pair path with matched decode A/B. See `experiments/054-grouped-attention-projections/REPORT.md`.

- Exp055 confirmed K/V are adjacent same-activation graph matmuls but PTQ dispatch is below the scheduler and consumes a prepared planar Q8 activation; current second-dot fusion is a nonlinear gate, not a second output. No candidate was built. Next, screen a direct paired-output kernel against two launches on the same prepared Q8 input before graph integration. See `experiments/055-ptq1-kv-pair-gemv/REPORT.md`.

- Exp050 audited the active BF16 matvec family: Exp047 reports 24,672 kernel instances over the capture (not CTAs), with 255 graph launches. Model-specific `ssm_alpha`/`ssm_beta` projections are 48x5120, giving 48 CTAs per applicable batch-1 kernel launch; compact replay artifacts do not resolve per-replay signature counts. Existing paired loads, FP32 accumulation, warp reduction and shared inter-warp fold leave no distinct low-cost sm_86 candidate; no implementation/build or E2E run. See [Exp050](../experiments/050-bf16-matvec/REPORT.md).

- Exp049 audited active sm_86 GDN code/SASS and found no distinct safe candidate: q/k reuse, contiguous state access, required warp reductions, and adjacent graph fusions are already present. No code or measurements; see `experiments/049-gdn-design/REPORT.md`.
- Exp048's cooperative one-launch QKV RMS-sharing variant was byte-exact and sanitizer-clean, but graph replay was ~25% slower; reverted before E2E. Current best unchanged; see `experiments/048-qkv-rms-sharing/REPORT.md`.
- Exp051's ping-pong shared-buffer FWHT cut active stage barriers and improved isolated graph-call medians 1.22%/5.91% for N=1024/NT=256 and NT=1024. Direct bytes matched baseline and Compute Sanitizer found no errors, but reversed-order model A/B was flat (+0.07% at context 512, -0.04% at 4096). Source was restored; current best unchanged. See `experiments/051-fwht-barriers/REPORT.md`.
- Exp047 directly measured one-token graph replay: PTQ1_0 GEMV is 9.01/9.02 ms/token (74–76%); QKV prep is 0.752 ms, GDN 0.500 ms, and the BF16 matvec specialization is ~0.311 ms/token at context 512. See `experiments/047-steady-decode-profile/REPORT.md`.

## Next candidates

**Format profile:** Use Exp052's paired graph signatures as the baseline for any format-specific decoder work. The measurements show both a per-call GEMV difference and fewer PTQ1_0 activation-prep nodes, but do not identify the GEMV hardware bottleneck; collect permitted hardware counters before making a traffic-versus-integer-throughput claim.

1. Screen a dedicated paired-output PTQ1_0 GEMV using the existing prepared planar Q8 activation against two sequential active GEMVs. If the kernel screen wins, integrate graph fusion with separate K/V outputs and exact fallback, then run two reversed-order full-model decode pairs.
2. Revisit PTQ1_0 GEMV only when a genuinely new dataflow or codegen premise appears; Exp046 and prior screens close the obvious decoder, staging, geometry, and scheduling families.
3. Revisit GDN only if profiling/codegen exposes redundant state traffic, a removable launch, or synchronization-free gate sharing; Exp049 found none in the current kernel.
4. Revisit BF16 matvec only if a future design can raise row-level CTA parallelism without an expensive cross-CTA K reduction; Exp050 found 48 CTAs per applicable 48-row launch and no surviving low-cost dataflow candidate. Per-replay family invocation counts remain unavailable in the compact profile artifacts.
