# Research state

## Current best

- Code commit: 62b4b4ce0c2809272b9d69d09f3359abd7111848; runtime: PrismML 6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17.
- Recommended format: PTQ1_0, batch-1 decode. Fresh two-pair, seven-repetition frozen-reference A/B (<=65 C gate): ctx512 77.690->84.234 tok/s (+8.42%); ctx4096 75.549->81.388 (+7.73%). Median latency 1647.6->1519.6 ms / 1694.3->1572.7 ms. Peak memory 6581->6579 / 6805->6803 MiB.
- Same-build Exp083 incremental check: PTQ1_0 84.492/82.090 tok/s at ctx512/4096 (+0.23%/+0.19%); PQ2_0 70.937/69.182 (+0.22%/+0.14%). PTQ1_0 prefill 1394.35/1350.34 tok/s at prompts 512/4096 (+0.32%/+0.53%).
- Config: 99 GPU layers, Flash Attention, F16 KV, batch/ubatch 2048/512, 8 CPU threads, 128 generated tokens, 7 repetitions/run. Correctness: selected CTests 7/7, CUDA backend cases 96/96, both format smoke runs pass, Exp083 direct test 6/6. Peak VRAM remains below 8 GiB.
- Final tables and all qualification details: FINAL_RESULTS.md. Exp083 artifacts: results/exp083/raw/.

## Current bottlenecks

1. PTQ1_0 planar batch-1 GEMV: about 9.0 ms/token, roughly 75% of summed decode kernel time.
2. QKV activation prep: 0.75 ms; GDN: 0.50 ms; remaining RMSNorm: 0.36 ms; attention: 0.23/0.58 ms at contexts 512/4096.
3. Remaining 48 linear-attention final_output layout copies: about 0.082 ms/token at context 512. The small-op ceiling is below 0.8%.
4. Nsight Compute counters remain blocked by ERR_NVGPUCTRPERM; current synthetic bandwidth estimates do not prove GEMV is bandwidth-bound.

## Successful optimizations

- Exp010: ROWS=1 scheduling for the active PTQ1_0 planar GEMV, +5.42%/+5.34% vs matched ROWS=4.
- Exp036: guarded coordinated QKV RMS/FWHT/Q8 prep, +1.65%/+1.55% vs same-binary disabled path.
- Exp060: guarded recurrent concat/cache-tail fusion; +0.95%/+0.90% in its matched decode pairs.
- Exp062: guarded recurrent SSM/SiLU/L2 fusion; flat at ctx512, +0.231% at ctx4096; 24 nodes removed/replay.
- Exp083: guarded strided Q-gate CONT/SIGMOID/MUL fusion; 16 nodes and 16 copies removed/replay; small positive E2E results for PTQ1_0 and PQ2_0. Final code commit is 62b4b4c.

## Failed or exhausted approaches

- Active PTQ1_0 GEMV: LUT, floor-difference, direct 2-bit, pairwise decode, warp transpose, shared staging, cp.async, prefetch/cache/layout, and simple Tensor Core variants were slower, no-op on the active path, or unsuitable at batch one. Owner-count changes conflict with exact four-stream FP32 accumulation.
- L2 persistence regressed decode; FlashAttention one-stage split regressed 12.8%/13.7%; adaptive ubatch improved some prompt cases but regressed ctx4096 decode about 1.8%.
- PQ2_0+MTP did not meet correctness and long-context performance requirements; not promoted. PQ2_0 decode remains slower than PTQ1_0.

## Architectural discoveries

- sm_86 batch-1 PTQ1_0 uses planar-transposed Q8_1 and dedicated mul_mat_vec_ptq1_0_pt; generic mmvq tunings do not reach it.
- The active GEMV feeds decoded raw 0/1/2 trits to DP4A and subtracts the exact activation sum for the signed digit-minus-one correction.
- PTQ1_0 prefill already uses signed-int8 Tensor Core MMQ; the batch-one Tensor Core alternative expands weights and wastes most output columns.
- PTQ1_0 was substantially faster than PQ2_0 in the original matched decode format matrix; prefill was nearly tied. See PROFILE.md and Exp052.

## Next research directions

1. Map the remaining 48 final_output CONT copies and their real consumers; retain only if matched E2E decode improves.
2. Keep PTQ1_0 GEMV as the major target, but require a new exact dataflow premise or hardware-measured bottleneck before another kernel experiment.
3. Re-profile after any material gain and revisit QKV prep, GDN, and attention in the new ranking.
