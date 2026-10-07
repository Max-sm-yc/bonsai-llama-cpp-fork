# Research state

## Current best

- PTQ1_0 on RTX 3080/sm_86: ROWS=1 planar batch-1 GEMV plus Exp036 coordinated QKV RMS/FWHT/Q8 prep. Production code commit `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`; reference runtime `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`.
- Same-binary, two reversed-order pairs of seven decode runs: 83.35 tok/s at context 512 and 80.35 at 4096; Exp036 disabled control 81.99/79.13 (+1.65%/+1.55%). Direct Exp041 reference/current A/B: +6.82%/+5.76% decode at contexts 512/4096, with 6,803 MiB versus 6,805 MiB peak GPU memory. Prefill measured 1,291.9/1,377.4/1,355.5/1,332.3 tok/s at contexts 128/512/2048/4096 and matches the frozen reference within 0.09%.
- Correctness: `tests/run_correctness.sh` passed selected CTests 5/5, backend CUDA-vs-CPU cases 96/96, and fixed-seed 32-token PTQ1_0/PQ2_0 smokes. Current main CUDA library SHA-256 `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`.

## Bottlenecks

1. Active PTQ1_0 batch-1 `mul_mat_vec_ptq1_0_pt`: 61.2%, 1.166 s of the mixed context-512 post-Exp036 trace; still the highest-value target.
2. PTQ1_0 GEMM: 12.7%, 242.6 ms.
3. Activation prep: 5.2%, 98.6 ms; GDN: 4.6%, 88.2 ms; remaining RMSNorm: 3.0%, 56.8 ms.
4. These are mixed setup/decode Nsight Systems totals. Nsight Compute counters fail with `ERR_NVGPUCTRPERM`; do not alter system-wide permissions.

## Successful optimizations

- Exp010: ROWS=1 lowered active specialization registers from 108 to 76/thread (no spills); matched decode improved +5.42%/+5.34% at contexts 512/4096 versus ROWS=4. ROWS=2 was 0.65–0.73% slower.
- Exp036: shape/use-count-guarded coordinated RMS→weight/sign→FWHT/Q8 prep reduced combined norm/prep trace time by 30.4 ms and improved same-binary decode +1.65%/+1.55%. Default on; `GGML_CUDA_RMS_FWHT_Q8=0` disables it.

## Failed or exhausted approaches

- Dispatch, decoder, and GEMV scheduling: generic-path edits in 001–003 do not reach the active sm_86 batch-1 path. LUT/floor, two-bit, pairwise, fixed-point, row-tile, warp/multiwarp reduction, recurrence distribution, and source strip-mining attempts (004–025, 039) were invalid for the target or exact but slower/no better. Exp024's isolated 1.49x cooperative recurrence screen became an 81.5% model regression.
- Prefetch/cache/load/layout variants: next-item hints and `.cs` lost E2E; `.cg` lost its kernel screen; padding to 32B added 14.3% payload and lost. Exp032's same-footprint SoA block screen gained 7.8% at 136 blocks but tied at 40; the runtime sidecar in 034–035 had no repeatable decode gain, added 1,280 MiB peak VRAM, and cost ~294 ms load time. Synchronous shared staging (037) lost 9.9–12.9%; warp-register transpose (038) lost 2.42–2.75x. The measured sm_86 `cp.async` next-work-item pipeline (040) emitted real async transfers but lost 10.4%/23.2% at 40/136 blocks. CTA widths 64/128/256/512 (042) lowered active-kernel registers at 256/512 but did not win both K shapes; a 256-at-K40/128-otherwise dispatch regressed matched decode 0.60%/0.68%.
- Inconclusive only: 014/016/027 ended before candidate measurement; 029 had one unmatched `.cg` trace. See `research/EXPERIMENTS.md` and linked reports before revisiting any idea.

## Important architectural discoveries

- RTX 3080 batch-1 PTQ1_0 uses planar-transposed Q8_1 and dedicated `mul_mat_vec_ptq1_0_pt`; generic `mmvq.cu` tunings do not apply. It processes flattened `(row group,K block)` work items with per-thread stride 128 and 128-thread CTAs.
- PTQ1_0 reference decode was 32–54% faster than PQ2_0 in the initial format comparison; prefill was nearly tied. Both files fit in VRAM. Current versus frozen-reference total A/B is still needed; original matrix is not apples-to-apples.
- CUDA Graphs are active (127 graph launches in the context-512 trace). Focus on device work and measured fusion. Q/K/V share memoized activation transforms; keep Exp036 guards. Nsight timings include setup and decode, not decode-only attribution.
- Repeated context-4096 decode samples have slow tails in both arms. Preserve all repetitions/ranges and use medians. Verify candidate library paths with `ldd`/`LD_DEBUG`; earlier absolute RUNPATHs caused false A/Bs.

## Next candidates

1. Sweep active 128-thread PTQ1_0 `__launch_bounds__` minimum-CTA constraints while holding ROWS=1 geometry fixed. The plain/gated kernels use 76/98 registers/thread, permitting about six/five 128-thread CTAs per SM by register capacity; Exp042 changed width and register use together. Test occupancy without spills, then exactness and matched E2E decode. Exp043/044 closed gate-stream and full activation-staging ideas; retain current production meanwhile.
