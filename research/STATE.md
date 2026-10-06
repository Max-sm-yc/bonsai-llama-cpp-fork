# Research state

## Current best

- **PTQ1_0, active sm_86 planar GEMV with ROWS=1.** Project commit `9fa97200e68fd798ef027470c8e420172a0ac719`; reference runtime commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`. Rebuilt source-default medians: 82.22 tok/s at context 512 and 79.70 at 4096 (7 reps, 128 decode tokens, F16 KV, FA on, 99 GPU layers, 8 CPU threads); matched archived ROWS=4 medians: 78.00/75.66, or +5.42%/+5.34%. Peak whole-GPU memory 6,805 MiB. Prefill not remeasured; reference medians are 1,378/1,331 tok/s at 512/4096.
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

## Important discoveries

- RTX 3080/sm_86 selects planar-transposed Q8_1 and dedicated `mul_mat_vec_ptq1_0_pt` for plain batch-1 PTQ1_0. ROWS=1 changes the one-column work mapping only; other column counts retain the existing schedule.
- Median decode gains repeat, but context-4096 samples have intermittent slow tails in both ROWS=1 and ROWS=4 builds. Keep means/ranges with medians; do not hide outliers.
- PTQ1_0 remains faster than PQ2_0 by 32–54% in the controlled reference format comparison; prefill is nearly tied. Both models fit in VRAM.

## Next candidates

1. Test a PTQ1 2-bit side-code/unpack design against the active planar sm_86 path; prior side-code tests used SOA and do not decide this layout. Benchmark actual model decode, including any repacking and VRAM cost.
2. Explore a distinct active-planar trit unpack/reduction mapping; keep end-to-end PTQ1_0 decode as the decision metric.
3. Revisit PQ2_0 activation fusion after a few further PTQ1 experiments.
