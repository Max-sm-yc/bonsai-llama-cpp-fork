# Current results — research in progress

This is the current verified snapshot, not the end of the campaign. The fastest correct implementation remains PTQ1_0 with ROWS=1 planar GEMV and the coordinated QKV RMS/FWHT/Q8 preparation path. Experiments 037–040 screened alternatives without changing production code.

## Hardware and software

- GPU: NVIDIA GeForce RTX 3080, GA102, compute capability 8.6, 10 GiB VRAM. Host: Intel Core i7-10700K, 31 GiB RAM, Fedora Linux 43 Workstation, kernel 7.1.8.
- Driver 580.178.04; CUDA compatibility 13.0; Toolkit/NVCC 13.2.86; GCC 15.3.1; CMake 3.31.11; Ninja 1.13.1; Nsight Systems 2025.6.3.541; Nsight Compute 2026.1.1.0. Full environment is in [ENVIRONMENT.md](ENVIRONMENT.md).
- Runtime/reference: PrismML `https://github.com/PrismML-Eng/llama.cpp`, branch `prism`, upstream commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`; project baseline commit `2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c`.
- Current best production code commit: `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`; the current tree also includes research records and the direct Exp036 correctness test. Latest code-library SHA-256: `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`.
- Model repository revision `b072e1d3b35a0a630cece372c2127528e0994386`. PTQ1_0 file: 5,946,648,928 bytes, SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`. PQ2_0 file: 7,206,168,928 bytes, SHA-256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`. See [SETUP.md](SETUP.md).

## Benchmark method

The reproducible baseline uses seven `llama-bench` repetitions per row with default warmups, a GPU start gate at or below 60°C and 5% utilization, contexts 128/512/2048/4096, and decode length 128. GPU layers 99, Flash Attention on, batch/microbatch 2048/512, F16 KV, and 8 CPU threads are held fixed. Raw samples, standard deviations, latency, and memory telemetry are recorded in JSON under `results/`. `llama-bench` excludes tokenization and sampling.

Exp041 directly compared the frozen project baseline with the current build under isolated matched conditions. Each mode used two reversed-order pairs of seven repetitions, with a fresh ≤60°C / ≤5%-utilization gate before every arm. The original prefill-first matrix remains format reference data; use Exp041 for the current-versus-reference cumulative gain.

## Original PTQ1_0 and PQ2_0 reference results

Median tokens/s from the original prefill-first baseline matrix:

| Format | Prefill at 512 | Prefill at 4096 | Decode at context 512 | Decode at context 4096 | Combined at prompt 4096 | Peak whole-GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| PTQ1_0 | 1378.14 | 1331.25 | 46.09 | 39.21 | 426.16 | 6805 MiB |
| PQ2_0 | 1364.88 | 1326.92 | 31.11 | 25.53 | 346.71 | 7949 MiB |

PTQ1_0 was 53.6% faster than PQ2_0 in the original context-4096 decode matrix, while prefill was nearly tied. The matrix warmed the GPU before decode, so those values are reference records, not the denominator for later isolated A/B results. Both files fit within the RTX 3080's available VRAM.

## Current best results

The current best is PTQ1_0, sm_86 planar batch-1 GEMV with ROWS=1, plus Exp036's guarded coordinated attention RMSNorm/weight/sign/FWHT/Q8 preparation. The Exp036 same-binary A/B measured:

| Context | Current best median tok/s | Same-binary disabled-path median | Improvement | Peak memory, enabled/disabled |
|---:|---:|---:|---:|---:|
| 512 | 83.3458 | 81.9939 | +1.65% | 6803 / 6805 MiB |
| 4096 | 80.3522 | 79.1268 | +1.55% | 6803 / 6805 MiB |

Each value is the median of two reversed-order seven-repetition run medians. Context-4096 runs had slow tails in both arms; all samples are retained in `results/exp036/`. The incremental Exp036 improvement is verified. Exp010's separate matched ROWS=1-vs-ROWS=4 test measured +5.42%/+5.34% at contexts 512/4096. These stage-wise results are not combined into a single total percentage because they came from different paired campaigns; Exp041 measures the cumulative current-versus-reference change directly.

### Direct frozen-reference comparison

The Exp041 comparison used the original project baseline commit `2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c` and current production code `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`; both use PrismML runtime source commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`. Each build had an isolated RUNPATH and `ldd` resolution. The following values are medians of two reversed-order run medians (14 samples per arm/context):

| Workload | Frozen reference | Current best | Change |
|---|---:|---:|---:|
| Decode, context 512 | 78.106 tok/s | 83.433 tok/s | +6.82% |
| Decode, context 4096 | 75.533 tok/s | 79.882 tok/s | +5.76% |
| Combined, prompt 512 + 128 generated | 314.372 tok/s | 332.196 tok/s | +5.67% |
| Combined, prompt 4096 + 128 generated | 860.947 tok/s | 877.786 tok/s | +1.96% |
| Prefill at 128 tokens | 1292.995 tok/s | 1291.875 tok/s | -0.09% |
| Prefill at 512 tokens | 1377.400 tok/s | 1377.395 tok/s | -0.00% |
| Prefill at 2048 tokens | 1355.280 tok/s | 1355.505 tok/s | +0.02% |
| Prefill at 4096 tokens | 1332.110 tok/s | 1332.260 tok/s | +0.01% |

Decode and combined peak memory was 6,805 MiB for the reference and 6,803 MiB for current. The full sample arrays, per-run telemetry, build hashes, and exact options are in [the Exp041 report](experiments/041-reference-current-ab/REPORT.md) and `results/reference_ab/`. Context-4096 decode included rare slow-tail samples in both builds; no fastest-run selection was used.

The matched comparison verifies +6.82%/+5.76% total decode improvement. This is a directly measured campaign result, not a product of the separate Exp010 and Exp036 deltas. Current prefill is unchanged within measurement noise. PQ2_0 has not been optimized; its best verified values remain the original baseline above.

### Performance progression across retained changes

| Code commit | Format / change | Decode at context 512 / 4096 | Matched result | Prefill at 4096 | Decision |
|---|---|---:|---:|---:|---|
| `2a6ac56` | PTQ1_0 reference runtime | 46.09 / 39.21 tok/s | Original matrix; not comparable with isolated A/B | 1331.25 tok/s | Baseline |
| `2a6ac56` | PQ2_0 reference runtime | 31.11 / 25.53 tok/s | Original matrix; not optimized | 1326.92 tok/s | Baseline |
| `9fa9720` | PTQ1_0 ROWS=1 planar GEMV | 82.22 / 79.70 tok/s | +5.42% / +5.34% vs matched ROWS=4 | Not remeasured | Keep |
| `c6cdaa5` | ROWS=1 + coordinated QKV preparation | 83.35 / 80.35 tok/s | +1.65% / +1.55% vs same-binary disabled path | Not remeasured | Current best |
| `c6cdaa5` | Same current build vs frozen project baseline | 83.43 / 79.88 tok/s | +6.82% / +5.76% vs matched baseline | 1332.26 tok/s | Current verified |

## Correctness and profile

- `bash tests/run_correctness.sh`: selected CTests 5/5, including the direct fused RMS/FWHT/Q8 GPU reference test; CUDA-vs-CPU PTQ1_0/PQ2_0 backend cases 96/96; fixed-seed 32-token model smokes for both formats passed. The fused-kernel test's max error was 0.54 stored Q8 scale for one and three rows, with exact block sums. The PTQ1_0 normalized model completion matched the previous reference exactly.
- The post-Exp036 Nsight Systems context-512 trace still ranks PTQ1_0 batch-1 GEMV first at 1.166 s / 61.22% of mixed-workload GPU kernel time. PTQ1_0 GEMM is 12.7%; activation prep and remaining RMSNorm are 5.2%; GDN is 4.6%. The trace is setup plus decode, not decode-only attribution.
- Nsight Compute hardware counters are blocked by `ERR_NVGPUCTRPERM`; no system-wide permission change was made. See [PROFILE.md](PROFILE.md).

## Retained optimizations, failed experiments, and next work

- Retained: Exp010's ROWS=1 scheduling for the active sm_86 planar GEMV; Exp036's exact-shape/use-count-guarded coordinated QKV preparation, default on and disableable with `GGML_CUDA_RMS_FWHT_Q8=0`.
- Exp037 tested CTA-local shared-memory staging of contiguous 28-byte AoS blocks. It matched codes/outputs and passed memcheck, but the production-equivalent work-plus-fold screen lost 9.85% at 40 blocks and 12.86% at 136 blocks. It was rejected before model integration.
- Exp038 tested a warp-register transpose with contiguous packed loads. It was exact, but the four-lane-per-warp dot mapping and seven shuffles lost 142%/175% at 40/136 blocks. It was rejected before model integration.
- Exp039 tested fixed-point floor-difference trit extraction in the active planar work-plus-fold path. Exhaustive byte/qh and full-row checks were exact, but the candidate lost 2.62% at 40 blocks and 4.28% at 136 blocks; it emitted 490 SASS instructions versus 338 for recurrence work. It was rejected before model integration.
- Exp040 tested `cp.async` copies of the next per-thread K-block while decoding the current item. SASS confirmed async copies and exactness passed, but work-plus-fold lost 10.40% at 40 blocks and 23.16% at 136. It was rejected before model integration.
- Other important negative results: 2-bit side encodings, pairwise radix-3 decode, warp/multiwarp reductions, cooperative trit recurrence, strip mining, cache modifiers, next-item prefetch, padded 32-byte blocks, and the selective SoA sidecar did not improve the active decode path. The compact index and report links are in [research/EXPERIMENTS.md](research/EXPERIMENTS.md).
- Architectural finding: active decode uses a dedicated planar Q8_1 / PTQ1_0 GEMV; generic `mmvq.cu` tuning does not reach it. CUDA Graphs are active, and the current PTQ1_0 GEMV remains the dominant measured family.
- Next: autotune the active GEMV CTA width (64/128/256/512 threads) together with compatible row-tile geometry; require active-planar correctness, a focused gain, and matched E2E decode improvement.
