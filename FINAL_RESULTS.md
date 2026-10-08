# Final results

The final verified code candidate is **62b4b4ce0c2809272b9d69d09f3359abd7111848**, based on the unmodified PrismML runtime at **6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17**. It retains the strongest correct implementation found during the campaign. PTQ1_0 remains the best format for batch-1 decode on this RTX 3080.

A fresh, paired comparison against the frozen project reference measured **+8.42% decode throughput at context 512** and **+7.73% at context 4096**, with peak memory unchanged within 2 MiB. These gains come from one coherent current implementation; they are measured directly and are not the sum of stage-wise experiment deltas.

## Hardware and software

- GPU: NVIDIA GeForce RTX 3080, GA102, compute capability 8.6, 10 GiB VRAM; CUDA reports 9,867 MiB usable.
- Host: Intel Core i7-10700K, 31 GiB RAM, Fedora Linux 43 Workstation, kernel 7.1.8.
- Driver 580.178.04; CUDA compatibility 13.0; Toolkit/NVCC 13.2.86; GCC 15.3.1; CMake 3.31.11; Ninja 1.13.1; Nsight Systems 2025.6.3.541; Nsight Compute 2026.1.1.0. See [ENVIRONMENT.md](ENVIRONMENT.md).
- Reference runtime: [PrismML llama.cpp](https://github.com/PrismML-Eng/llama.cpp), branch prism, upstream commit 6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17.
- Frozen project reference: commit 2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c.
- Final optimized code: commit 62b4b4ce0c2809272b9d69d09f3359abd7111848. Candidate CUDA library SHA-256: 14471383ade09fbfb6153970f340119bb91e7bbe27a85bb9c4a2ebbfa2c20f0d.
- [Model repository](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) revision b072e1d3b35a0a630cece372c2127528e0994386:
  - PTQ1_0: 5,946,648,928 bytes, SHA-256 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3.
  - PQ2_0: 7,206,168,928 bytes, SHA-256 3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1.

## Benchmark method

All results are from the actual RTX 3080. llama-bench was warmed up and used seven repetitions per process. The model was fully offloaded with 99 GPU layers, Flash Attention enabled, F16 K/V cache, batch/microbatch 2048/512, and 8 CPU threads. Batch-1 decode generated 128 tokens at the stated existing context. Reported decode latency is the median per-run median of the seven measured sample latencies; throughput is the median of the two reversed-order run medians per arm. Tokenization and sampling are excluded.

The final frozen-reference comparison ran two reversed-order pairs per context and gated every arm at GPU temperature <=65 C and utilization <=5%. The normal <=60 C gate remained at a 61 C idle floor during the first attempt, so that incomplete refresh was excluded. The completed 65 C-gated comparison used the same gate, configuration, model, and alternating order for both isolated binaries; ldd confirmed each executable loaded libraries from its own build. All eight final run JSON files, telemetry, and the driver transcript are in results/exp083/raw/.

The original baseline matrix and most optimization A/B tests used a <=60 C start gate; their own comparisons remain paired within their campaign. Do not directly mix throughput values across campaigns with different start gates.

## Original PTQ1_0 and PQ2_0 reference matrix

Median tokens/s from the original prefill-first seven-repetition baseline matrix:

| Format | Prefill at 512 | Prefill at 4096 | Decode at context 512 | Decode at context 4096 | Peak whole-GPU memory |
|---|---:|---:|---:|---:|---:|
| PTQ1_0 | 1,378.14 | 1,331.25 | 46.09 | 39.21 | 6,805 MiB |
| PQ2_0 | 1,364.88 | 1,326.92 | 31.11 | 25.53 | 7,949 MiB |

In that matched format matrix, PTQ1_0 decoded 48.2% faster at context 512 and 53.6% faster at 4096 while prefill was nearly tied. The original matrix ran prefill before decode and warmed the GPU. Its decode numbers are useful format references, but they are not the denominator for the final cumulative speedup.

## Final PTQ1_0 decode against the frozen reference

| Existing context | Frozen reference | Final candidate | Throughput change | Median latency: reference -> candidate | Peak memory: reference / candidate |
|---:|---:|---:|---:|---:|---:|
| 512 | 77.690 tok/s | 84.234 tok/s | **+8.42%** | 1,647.6 -> 1,519.6 ms | 6,581 / 6,579 MiB |
| 4096 | 75.549 tok/s | 81.388 tok/s | **+7.73%** | 1,694.3 -> 1,572.7 ms | 6,805 / 6,803 MiB |

Each value is based on two reversed-order run pairs with seven repetitions per run. The two run medians favored the candidate in every pair: 77.779/77.601->84.224/84.245 tok/s at context 512 and 75.503/75.596->81.273/81.503 at 4096. Across the 14 samples per arm, the min-max and sample standard deviation were 76.836-77.933 / 0.298 tok/s (reference) and 80.539-84.369 / 1.113 (candidate) at context 512; at context 4096 they were 72.352-75.637 / 1.102 and 63.075-81.902 / 4.974. One candidate context-4096 slow-tail sample drives much of that spread; it is retained, and no fastest-run selection is used. The sample arrays, run medians, latencies, and GPU telemetry are in results/exp083/raw/final65_ptq1_decode_ctx*.json.

## Final candidate format results

The following are latest verified absolute measurements for the final code. PTQ1_0 decode values above are from the direct frozen-reference comparison. The incremental PTQ1_0 prefill and PQ2_0 decode values below come from paired Exp083 comparisons against the immediately preceding implementation; they are not presented as fresh direct comparisons against the frozen project reference.

| Format / workload | Context | Final candidate tok/s | Immediate-parent tok/s | Exp083 change | Peak memory |
|---|---:|---:|---:|---:|---:|
| PTQ1_0 prefill | 512 | 1,394.35 | 1,389.98 | +0.32% | 6,569 MiB |
| PTQ1_0 prefill | 4096 | 1,350.34 | 1,343.21 | +0.53% | 6,775 MiB |
| PQ2_0 decode | 512 | 70.937 | 70.783 | +0.22% | 7,723 / 7,725 MiB |
| PQ2_0 decode | 4096 | 69.182 | 69.083 | +0.14% | 7,947 / 7,949 MiB |

PQ2_0 has not received a format-specific production optimization. The shared graph fusion in Exp083 applies to both formats. Its final observed decode is slower and uses about 1.1 GiB more peak VRAM than PTQ1_0 in the original format comparison, so PTQ1_0 is the recommended format for this GPU and workload.

## Performance progression across retained changes

Stage-wise experiments used their own matched controls and sometimes different thermal gates. These rows show measured effects, not factors that can be multiplied to estimate a total. The final row is the direct cumulative comparison above.

| Code commit / milestone | Change | Measured result |
|---|---|---|
| 2a6ac56 | Frozen PTQ1_0 reference | Original matrix: 46.09 / 39.21 tok/s at contexts 512 / 4096 |
| 9fa9720 | Exp010, ROWS=1 planar GEMV | 82.22 / 79.70 tok/s; +5.42% / +5.34% vs matched ROWS=4 |
| c6cdaa5 | Exp036, coordinated QKV RMS/FWHT/Q8 preparation | 83.35 / 80.35 tok/s; +1.65% / +1.55% vs same-binary disabled path |
| 4cb2072 | Exp060, recurrent concat/cache-tail fusion | +0.95% / +0.90% in its reversed-order decode pairs |
| ffb0ef3 | Exp062, recurrent SSM/SiLU/L2 fusion | 84.41 / 81.88 tok/s; +0.037% at 512 (flat) / +0.231% at 4096 vs control |
| 62b4b4c | Exp083, full-attention strided gate fusion | 84.49 / 82.09 tok/s; +0.23% / +0.19% vs immediate parent in the <=60 C incremental campaign |
| 62b4b4c | Fresh direct frozen-reference comparison | 84.23 / 81.39 tok/s; +8.42% / +7.73% vs same-session reference under the <=65 C gate |

## Correctness

- The final selected CUDA CTest set passed **7/7**, including the new Exp083 strided-view test.
- CUDA-versus-CPU PTQ1_0/PQ2_0 backend cases passed **96/96** under the campaign's 5e-4 NMSE criterion.
- Fixed-seed 32-token PTQ1_0 and PQ2_0 CUDA model smokes passed. PTQ1_0 normalized completion text matched the reference. The saved baseline smoke result retains its original SHA-256, 2a502e2d53f1ccb88c6944e4f84dc0f27f6b5ca8ea9e95e324084a0c20f7454d.
- The new CUDA fusion test covered sequence lengths 1, 2, 128, 512, and 4096 plus a contiguous-source fallback. All six cases passed against a host scalar reference at 2e-6 tolerance; maximum absolute error was 1.1920929e-7.
- See [tests/README.md](tests/README.md), [tests/run_correctness.sh](tests/run_correctness.sh), and the [Exp083 report](experiments/083-small-op-fusion/REPORT.md).

## Bottlenecks and research summary

The steady-state decode profile ranks the active PTQ1_0 batch-1 GEMV first: about **9.0 ms/token**, approximately **75% of summed kernel time** at contexts 512 and 4096. Other measured families are QKV activation preparation at about 0.75 ms, GDN at 0.50 ms, remaining RMSNorm at 0.36 ms, and attention at about 0.23/0.58 ms. The 48 remaining linear-attention final_output layout copies cost about 0.082 ms/token at context 512. Nsight Compute hardware counters remain unavailable with ERR_NVGPUCTRPERM; payload-equivalent and synthetic bandwidth figures do not establish actual GEMV DRAM throughput.

The campaign covered the active PTQ1_0 GEMV encoding and scheduling, memory/cache behavior, tensor-core alternatives, model graph fusions, attention, MTP, and prefill scheduling. Successful production changes were ROWS=1 planar GEMV scheduling, coordinated QKV activation preparation, recurrent concat/cache fusion, recurrent SSM/L2 fusion, and the Exp083 Q-gate fusion. The last fusion removed 16 graph nodes and 16 cpy_scalar calls/token, but its end-to-end gain is necessarily small while GEMV dominates.

Important negative findings:

- LUT, floor-difference, direct 2-bit, pairwise, warp transpose, shared staging, and cp.async trit-decoder/GEMV variants were exact in some cases but slower or unsuitable for the active path.
- Generic prefetch and GEMV geometry edits did not reach the dedicated sm_86 PTQ1_0 batch-1 planar kernel. Cache modifiers, padding, and sidecar layouts did not produce a repeatable model gain; maximum L2 persistence regressed decode.
- An int8 Tensor Core GEMV mapping expands weights and wastes most output columns at batch one; PTQ1_0 prefill already uses an sm_86 int8 Tensor Core MMQ path.
- The lower-shared-memory FlashAttention split doubled the long-context grid but regressed attention by 12.8% at context 4096 and 13.7% at 512.
- Adaptive prompt ubatch selection improved long prefill/combined cases but regressed context-4096 batch-1 decode by about 1.8%, so the default remains ubatch 512.
- The PQ2_0+MTP bundle did not meet correctness and long-context performance requirements; it was not promoted.

See the experiment index and individual reports for measured negative results.

## Future experiments

1. Map the remaining 48 final_output CONT copies to their actual consumers. This is the best small-op opportunity; the estimated ceiling is below 0.8% at context 512, so require matched decode improvement before keeping a fusion.
2. Keep PTQ1_0 GEMV as the main research target. Previous screens exhausted straightforward decoder substitutions, byte staging, simple tensor-core expansion, and owner-count reductions that preserve the exact four-stream FP32 order. A worthwhile next attempt needs a new dataflow or hardware-measured bottleneck premise.
3. Revisit actual memory-versus-integer-pipeline balance when Nsight Compute counters are available without system-wide permission changes. Current synthetic bandwidth estimates are not proof of GEMV's limiting resource.
4. Re-profile after any larger gain; then re-rank QKV preparation, GDN, attention, and remaining copies.

## Reproduction

- Run the baseline matrix: python3 benchmark/run.py --output results/latest.json.
- Run selected correctness checks: bash tests/run_correctness.sh.
- Reproduce the final direct PTQ1_0 comparison: python3 experiments/083-small-op-fusion/run_final_reference_ab.py.
- Read [SETUP.md](SETUP.md), [ENVIRONMENT.md](ENVIRONMENT.md), [BASELINE.md](BASELINE.md), [PROFILE.md](PROFILE.md), [research/STATE.md](research/STATE.md), and the [Exp083 report](experiments/083-small-op-fusion/REPORT.md).
