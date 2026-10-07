# Research state

## Current best

- PTQ1_0 on RTX 3080/sm_86: ROWS=1 planar batch-1 GEMV plus Exp036 coordinated QKV RMS/FWHT/Q8 prep. Production code commit `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`; reference runtime `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`.
- Same-binary, two reversed-order pairs of seven decode runs: 83.35 tok/s at context 512 and 80.35 at 4096; Exp036 disabled control 81.99/79.13 (+1.65%/+1.55%). Direct Exp041 reference/current A/B: +6.82%/+5.76% decode at contexts 512/4096, with 6,803 MiB versus 6,805 MiB peak GPU memory. Prefill measured 1,291.9/1,377.4/1,355.5/1,332.3 tok/s at contexts 128/512/2048/4096 and matches the frozen reference within 0.09%.
- Correctness: `tests/run_correctness.sh` passed selected CTests 5/5, backend CUDA-vs-CPU cases 96/96, and fixed-seed 32-token PTQ1_0/PQ2_0 smokes. Current main CUDA library SHA-256 `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`.

## Bottlenecks

1. Active PTQ1_0 batch-1 GEMV (`mul_mat_vec_ptq1_0_pt`): 9.014 ms/token at context 512 and 9.022 ms/token at 4096, 75.9%/73.8% of summed steady-state graph kernel duration. Its plain, fused-gate, and fused non-gate specializations cost ~4.57, ~2.31, and ~2.14 ms/token.
2. QKV activation preparation: 0.752 ms/token (6.2–6.3%); GDN: 0.500 ms/token (4.1–4.2%); remaining standalone RMSNorm: 0.364 ms/token (~3%). Attention rises from 0.236 ms/token at context 512 to 0.588 ms at 4096.
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
- Repeated context-4096 decode samples have slow tails in both arms. Preserve all repetitions/ranges and use medians. Verify candidate library paths with `ldd`/`LD_DEBUG`; earlier absolute RUNPATHs caused false A/Bs.

## Latest research result

- Exp047 separated direct one-token CUDA graph replay from non-graph setup on the actual RTX 3080. PTQ1_0 GEMV remains decisively first at 9.01/9.02 ms/token for contexts 512/4096. The largest named secondary family is coordinated QKV activation preparation at 0.752 ms/token, followed by GDN at 0.500 ms. The prior 61.2% mixed trace share is qualified, not decode-only. See `experiments/047-steady-decode-profile/REPORT.md`.

- Exp046 challenged the active PTQ1_0 batch-1 GEMV mapping and found no distinct candidate outside already-screened decoder, lane-mapping, staging, and scheduling families. No implementation or new performance/correctness measurements were produced; decision: NO CANDIDATE / REVERT. Current best and production path are unchanged. See `experiments/046-ptq1-dataflow-challenge/REPORT.md`.

## Next candidates

1. Profile-driven follow-up after Exp047: when a materially new GEMV premise appears, evaluate it against the measured 9.01/9.02 ms/token decode cost. With no new GEMV mapping from Exp046, first isolate coordinated QKV activation preparation (0.752 ms/token) and test a specific way to reduce its device work or launches; GDN is next at 0.500 ms/token. Avoid implementation without a concrete premise.
