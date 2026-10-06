# Final results — research in progress

The optimization campaign is active. Experiment 010 is the current verified best; this file records the reference results and the first retained optimization. Continue updating it as experiments are accepted.

## Hardware and software

- Hardware: NVIDIA GeForce RTX 3080 10 GiB, GA102, compute capability 8.6; Intel Core i7-10700K; Fedora 43 Workstation.
- Driver 580.178.04; CUDA driver compatibility 13.0; CUDA Toolkit/NVCC 13.2.86; Nsight Systems 2025.6.3.541; Nsight Compute 2026.1.1.0.
- Reference runtime: PrismML `https://github.com/PrismML-Eng/llama.cpp`, branch `prism`, commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`.
- Model repository: `https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf`, revision `b072e1d3b35a0a630cece372c2127528e0994386`.
- Files: `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5,946,648,928 bytes; SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`); `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,206,168,928 bytes; SHA-256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`).
- Current best project commit: `9fa97200e68fd798ef027470c8e420172a0ac719` (PTQ1_0 ROWS=1); project reference baseline commit: `2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c`.

## Method

Seven llama-bench repetitions per row with default warm-up. Paired format runs start when the GPU is at or below 60°C and 5% utilization. Contexts 128/512/2048/4096; decode length 128; batch 2048, microbatch 512; batch-1 decode; F16 KV; Flash Attention on; 99 GPU layers; 8 CPU threads. The raw JSON stores each sample and sampled whole-GPU memory. Combined throughput includes prompt evaluation and 128 autoregressive tokens but excludes tokenization and sampling. Full details are in [BASELINE.md](BASELINE.md).

## Original reference performance

Median tokens/s from the controlled baseline:

| Format | Prefill at 4096 | Decode at context 128 | Decode at context 4096 | Combined at prompt 4096 | Peak whole-GPU memory |
|---|---:|---:|---:|---:|---:|
| PTQ1_0 | 1331.3 | 46.88 | 39.21 | 426.16 | 6805 MiB |
| PQ2_0 | 1326.9 | 35.53 | 25.53 | 346.71 | 7949 MiB |

PTQ1_0 is 53.6% faster than PQ2_0 in the original context-4096 decode matrix; prefill is nearly tied. That matrix ran prefill before decode, heated the GPU, and is not an apples-to-apples speedup denominator for the later isolated decode experiment. The per-format data remain a reference record, not the matched optimization comparison.

## Current best: PTQ1_0 ROWS=1

The active sm_86 planar GEMV now uses one row per item for batch-1 decode. In the manager's rebuilt source-default A/B, the seven-repetition medians were 82.22 tok/s at context 512 and 79.70 tok/s at context 4096. The paired archived ROWS=4 build measured 78.00 and 75.66 tok/s under the same isolated decode command, respectively: **+5.42% at context 512 and +5.34% at context 4096**. The improvement was also reproduced in two earlier reversed-order paired rounds. See [experiment 010](experiments/010-ptq1-planar-rows/REPORT.md) and its [raw paired measurements](results/exp010/final_rebuilt_pair/).

Both paired builds peaked at 6,805 MiB whole-GPU memory. The timing uses 128 generated tokens, batch-1 decode, F16 KV, Flash Attention, 99 GPU layers, batch/microbatch 2048/512, eight CPU threads, and a start gate of at most 60°C. The exact start temperatures in the manager's final pair differed (51°C candidate, 59°C control); earlier reversed-order pairs started at 58–60°C and showed the same direction. Keep this qualification with the point estimate.

Optimized prefill was not remeasured; ROWS=1 only changes the one-column matvec schedule. The reference PTQ1_0 prefill medians remain 1,378.1 tok/s at prompt 512 and 1,331.3 tok/s at 4096 as reference values, not new measurements. PQ2_0 has not been optimized and retains its original benchmark results.

## Correctness and profiling

- Upstream quantization/layout/shape tests: 4/4 pass.
- Random CUDA-vs-CPU PTQ1_0/PQ2_0 `MUL_MAT` checks: 96/96 pass, upstream NMSE threshold 5e-4.
- Both actual model files load on CUDA and produce greedy 32-token completions.
- Nsight Systems identifies ternary GEMV as 60.4% of GPU kernel time after ROWS=1 (down from 61.8% in the PTQ1_0 reference trace); the three active GEMV variants now total 1.166 s versus 1.253 s. Full ranking and profiler limitations are in [PROFILE.md](PROFILE.md).
- Nsight Compute hardware counters were denied by `ERR_NVGPUCTRPERM`; no system-wide permission changes were made.

## Progression by verified implementation

| Code commit | Format / schedule | Decode tok/s at context 512 / 4096 | Matched decode delta | Prefill tok/s at 4096 | Correctness | Decision |
|---|---|---:|---:|---:|---|---|
| `2a6ac56` (PrismML `6bfcd79a`) | PTQ1_0 reference matrix | 46.09 / 39.21 | Original matrix; hot decode is not comparable to isolated pairs | 1331.3 | Pass | Baseline |
| `2a6ac56` (PrismML `6bfcd79a`) | PQ2_0 reference matrix | 31.11 / 25.53 | Reference only; not optimized | 1326.9 | Pass | Baseline |
| `9fa9720` | PTQ1_0 ROWS=1 | 82.22 / 79.70 | +5.42% / +5.34% vs matched isolated ROWS=4 at 78.00 / 75.66 | Not remeasured | Pass | Keep |

The absolute reference-matrix decode figures are shown to preserve the original record. Due to heating before those decode rows, use the isolated ROWS=1-vs-ROWS=4 pair for the optimization percentage.

## Retained changes, failures, and next work

- Retained optimization: one-row scheduling in the active PTQ1_0 planar batch-1 GEMV, for +5.3% median decode at context 4096 against the matched isolated ROWS=4 control. Current PQ2_0 remains the unmodified reference implementation.
- The first format benchmark began PTQ1_0 cool and PQ2_0 hot and is excluded. Experiments 001/002 disabled explicit PTQ1_0 GEMV L2 prefetch; paired 128-token workloads tied within 0.04%, while longer tails varied with process order and clocks. The source was restored.
- Experiment 003 changed the generic PTQ1_0 warp-count selection, but a dispatch audit found that sm_86 batch-1 inference bypasses it for the dedicated planar PT kernel. Its decode measurements compared the same active kernel and do not inform kernel geometry.
- Experiment 004's exact constant-memory LUT matched 65,536 focused dot outputs but took 5.89x longer than multiply/byte-permute decoding; it was not integrated. Its harness excluded `qh` and production activation layout.
- Experiment 005's 2-bit side prototype wrote 128 per-weight bytes into a 32-byte packed-code field, overrunning adjacent records, and also used the wrong base-3 element order. Its timing is invalid.
- Experiment 006 fixed packing and element mapping and passed exact code/dot checks, but its Q8 activation address did not match production SOA_ISUM. Its measured 18.4–18.8% slowdown is inconclusive; no runtime integration or E2E test was run. The proposed correct code expands blocks from 28 to 34 bytes (+21.43%).
- Experiment 007 used the SOA_ISUM mapping and DP4A block arithmetic; packed 2-bit was 3.44–3.50% slower at 65,536 blocks and 5.02% slower at 16,384 blocks. RTX 3080 batch-1 uses planar PT instead, so this rejects the tested SOA side dot but does not decide active-kernel performance.
- Experiment 008's direct-floor trit identity passed host exhaustive checks, but the first packed CUDA implementation failed full-block correctness because its `qh` path did not interleave the two trit streams. No timing or runtime change was made.
- Experiment 009 fixed the `qh` interleave and passed exhaustive device, full-block, and sanitizer checks; the floor-difference decoder was 1.24–7.04% slower in the SOA block harness. The active planar sm_86 kernel was not tested.
- Experiments 001–009 tested prefetch, generic geometry, alternate decoders, and side-code variants; they either addressed an inactive SOA path, failed exactness, or lost their focused timings. Experiment 011's exact 2-bit codes lost 3.02–7.01% before conversion on the active planar mapping. Experiment 012's exact pairwise radix-3 decoder lost 1.97–24.59%. Details are indexed in [research/EXPERIMENTS.md](research/EXPERIMENTS.md).
- Experiment 013 swept active planar CTA row-tile caps and shared-memory targets. Cap 8 was effectively tied with ROWS=1 (+0.11%/-0.01% at contexts 512/4096), while larger tiles regressed. The source and rebuilt CUDA library were restored and hash-verified; this does not change the best result.
- Experiment 014 recorded a warp-per-row reduction hypothesis but did not implement or measure a candidate. It leaves no performance result and does not change the best.
- Experiment 015 implemented the warp-per-output-row register reduction. It passed selected correctness and model checks but lost 3.3–3.8% in two reversed-order pairs; the exact ROWS=1 source and library were restored and hash-verified.
- Experiment 016 proposed a cooperative multiwarp row reduction but did not implement or measure it; the hypothesis remains open.
- Experiment 017 implemented two warps per row and passed selected correctness and model checks, but decode fell by 2.60% at context 512 and 1.82% at 4096. The source and active library were hash-verified after restoration.
- Experiment 018 specialized four warps per row for K>64 and passed selected correctness/model checks. Context-512 medians lost 0.32%/0.69% in reversed pairs; context 4096 changed sign (+0.77%/-0.58%). No repeatable gain; ROWS=1 was restored.
- Experiment 019 audited RMSNorm→FWHT/Q8_1 fusion. The attention norm output feeds Q/K/V projection paths, so one projection's fused transform cannot eliminate the shared result. No implementation or benchmark was made; a coordinated multi-output design would need a separate cost analysis.
- After ROWS=1, the active planar GEMV still accounts for 60.4% of profiled kernel time. Fixed and shape-gated warp-per-row reductions have not improved it. Next investigate the measured FWHT/Q8_1 kernel (4.4%) and gated-delta kernel (4.6%) independently. CUDA Graphs are already active in the baseline trace.
- Nsight Compute counters remain unavailable. Future claims should rely on isolated repeated end-to-end runs and available Nsight Systems timing.
