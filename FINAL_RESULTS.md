# Final results — research in progress

The reference baseline is verified; the optimization campaign has not selected a final optimized commit yet. Values below establish the comparison point and will be extended as experiments are accepted.

## Hardware and software

- Hardware: NVIDIA GeForce RTX 3080 10 GiB, GA102, compute capability 8.6; Intel Core i7-10700K; Fedora 43 Workstation.
- Driver 580.178.04; CUDA driver compatibility 13.0; CUDA Toolkit/NVCC 13.2.86; Nsight Systems 2025.6.3.541; Nsight Compute 2026.1.1.0.
- Reference runtime: PrismML `https://github.com/PrismML-Eng/llama.cpp`, branch `prism`, commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`.
- Model repository: `https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf`, revision `b072e1d3b35a0a630cece372c2127528e0994386`.
- Files: `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5,946,648,928 bytes; SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`); `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,206,168,928 bytes; SHA-256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`).
- Current implementation source is unchanged reference code. Project baseline commit: `2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c`; final optimized commit: pending.

## Method

Seven llama-bench repetitions per row with default warm-up. Paired format runs start when the GPU is at or below 60°C and 5% utilization. Contexts 128/512/2048/4096; decode length 128; batch 2048, microbatch 512; batch-1 decode; F16 KV; Flash Attention on; 99 GPU layers; 8 CPU threads. The raw JSON stores each sample and sampled whole-GPU memory. Combined throughput includes prompt evaluation and 128 autoregressive tokens but excludes tokenization and sampling. Full details are in [BASELINE.md](BASELINE.md).

## Reference performance

Median tokens/s from the controlled baseline:

| Format | Prefill at 4096 | Decode at context 128 | Decode at context 4096 | Combined at prompt 4096 | Peak whole-GPU memory |
|---|---:|---:|---:|---:|---:|
| PTQ1_0 | 1331.3 | 46.88 | 39.21 | 426.16 | 6805 MiB |
| PQ2_0 | 1326.9 | 35.53 | 25.53 | 346.71 | 7949 MiB |

PTQ1_0 is 53.6% faster in decode at context 4096 under these conditions; prefill is nearly tied. This is the measured RTX 3080 result, independent of cross-GPU model-card claims.

## Correctness and profiling

- Upstream quantization/layout/shape tests: 4/4 pass.
- Random CUDA-vs-CPU PTQ1_0/PQ2_0 `MUL_MAT` checks: 96/96 pass, upstream NMSE threshold 5e-4.
- Both actual model files load on CUDA and produce greedy 32-token completions.
- Nsight Systems identifies ternary GEMV as 61.8% (PTQ1_0) and 62.6% (PQ2_0) of the mixed context-512 trace's GPU kernel time. Full measured ranking and profiler limitations are documented in [PROFILE.md](PROFILE.md).
- Nsight Compute hardware counters were denied by `ERR_NVGPUCTRPERM`; no system-wide permission changes were made.

## Progression by verified implementation

| Code commit | Format | Decode tok/s at context 4096 | Prefill tok/s at 4096 | Correctness | Decision |
|---|---|---:|---:|---|---|
| `2a6ac56` (PrismML `6bfcd79a`) | PTQ1_0 | 39.21 | 1331.3 | Pass | Baseline |
| `2a6ac56` (PrismML `6bfcd79a`) | PQ2_0 | 25.53 | 1326.9 | Pass | Baseline |
| Optimization commits | — | — | — | — | None accepted yet |

## Retained changes, failures, and next work

- No performance optimization has been accepted; retained speedup over the verified baseline is currently 0%. Upstream CUDA PTQ1_0/PQ2_0 GEMV and existing PTQ1_0 Hadamard/Q8_1 fusion remain the current best implementation.
- The first format benchmark began PTQ1_0 cool and PQ2_0 hot and is excluded. Experiments 001/002 disabled explicit PTQ1_0 GEMV L2 prefetch; paired 128-token workloads tied within 0.04%, while longer tails varied with process order and clocks. The source was restored.
- Experiment 003 compared two and eight warps with the four-warp batch-1 PTQ1_0 baseline. Decode changes remained within +/-0.05% at contexts 512 and 4096, so warp-count tuning was rejected.
- Experiment 004's exact constant-memory LUT matched 65,536 focused dot outputs but took 5.89x longer than multiply/byte-permute decoding; it was not integrated. Its harness excluded `qh` and production activation layout.
- Experiment 005's 2-bit side prototype wrote 128 per-weight bytes into a 32-byte packed-code field, overrunning adjacent records, and also used the wrong base-3 element order. Its timing is invalid.
- Experiment 006 fixed packing and element mapping and passed exact code/dot checks, but its Q8 activation address did not match production SOA_ISUM. Its measured 18.4–18.8% slowdown is inconclusive; no runtime integration or E2E test was run. The proposed correct code expands blocks from 28 to 34 bytes (+21.43%).
- Experiment 007 used the production SOA mapping and DP4A block arithmetic; packed 2-bit was 3.44–3.50% slower at 65,536 blocks and 5.02% slower at 16,384 blocks. With 21.43% more payload, this route was rejected before runtime integration.
- Next, test a bit-sliced decoder on the original base-3 format. The direct constant-memory LUT was 5.89x slower in its focused test. Check the 10 GiB VRAM limit before any future runtime route.
- Final bottleneck ranking, total speedup, and future work remain pending while the optimization campaign continues.
