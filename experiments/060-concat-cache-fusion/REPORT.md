# Experiment 060: recurrent concat/cache fusion

## Hypothesis and decision

The standard one-token Qwen3.5 recurrent graph materializes a 4×10,240 F32 concat for `SSM_CONV`, then copies its last three time rows into the recurrent cache. One CUDA launch can materialize the same concat and write those cache bytes, removing the separate CPY node while preserving all intervening recurrent-state operations. **KEEP, integrated in code commit `4cb2072`:** exact CUDA output/cache tests passed, the real graph lost 48 nodes per replay, and two reversed-order PTQ1_0 A/B pairs showed repeatable decode gains of 0.95% at context 512 and 0.90% at 4096. Independent manager A/B and final main-build decode checks also passed.

## Implementation and guards

`ggml_cuda_try_fuse` matches only a F32 CONCAT with shape `[4,10240,1,1]`, dim 0, contiguous F32 inputs `[3,10240]` and `[1,10240]`, and use count 2. Its next three graph entries must be exactly the source VIEW, destination VIEW, and CPY. The source view must be the concat's `[3,10240,1,1]` tail at byte offset 4 with strides `[4,16,163840,163840]`; the destination must be a contiguous `[30720,1,1,1]` view at offset 0. The relevant nodes cannot be graph outputs. It checks the view use counts, source pointer offset, and pairwise non-overlap of both inputs, concat output, and cache destination. Rollback or multi-sequence shapes/offsets fall back. The matcher returns 3; executor semantics add that to the node index and then advance once, skipping precisely the two VIEWs and CPY.

The new kernel writes the full concat (`history[0..2]`, then current value per channel) and the exact cache order (`history[1]`, `history[2]`, current value per channel) in one launch. SSM_CONV/SiLU and recurrent-state operations remain unchanged.

## Correctness and activation

- `tests/test-exp060-concat-cache.cpp` compares both complete outputs bit-for-bit on CUDA: three repeated updates at the model shape and two updates at 1,024 channels to exercise fallback. CTest `test-exp060-concat-cache` passed.
- `tests/run_correctness.sh` passed selected CTests 5/5, CUDA backend comparisons 96/96, and PTQ1_0/PQ2_0 fixed-seed model smokes. Separate 32-token PTQ1_0 baseline/candidate smokes produced identical completions after stripping build ID and timing lines.
- The isolated Release build used CUDA, CUDA graphs, FlashAttention, and `CMAKE_CUDA_ARCHITECTURES=86`. `ldd` confirms each benchmark/CLI resolved `libggml-cuda.so.0` from its own build directory; see `results/exp060/raw/ldd_{base,candidate}.txt`.
- Real graph captures activated the new signature at 48 calls/replay. Only the CONCAT and CPY signature counts changed; every other kernel signature count was identical.

## CUDA graph captures

Each Nsight Systems capture contains 31 complete one-token graph replays. Times are mean summed kernel duration per replay.

| Context | Arm | Nodes/replay | CONCAT or fused calls/time | CPY calls/time | Total kernel ms/replay |
|---:|---|---:|---:|---:|---:|
| 512 | Baseline | 1,432 | 48 / 0.099598 ms | 112 / 0.208986 ms | 11.844936 |
| 512 | Candidate | 1,384 | fused 48 / 0.080944 ms | 64 / 0.111674 ms | 11.726621 |
| 4096 | Baseline | 1,432 | 48 / 0.099511 ms | 112 / 0.208968 ms | 12.196336 |
| 4096 | Candidate | 1,384 | fused 48 / 0.080699 ms | 64 / 0.111408 ms | 12.079838 |

The fusion removes exactly one CPY node for each of the 48 recurrent blocks. Total replay kernel time fell 0.118315 ms (1.00%) at context 512 and 0.116498 ms (0.96%) at 4096. The untouched 64 CPY calls include other recurrent-state mutations.

## Matched PTQ1_0 decode A/B

Two order-reversed pairs used seven repetitions per run, 128 generated tokens, and identical runtime/model settings (99 GPU layers, FlashAttention, batch 2048, ubatch 512, F16 KV, 8 CPU threads). Every run passed the ≤60°C/≤5% GPU start gate. All sample values and telemetry are retained in `results/exp060/pair*.json`; this table reports each run's median across its seven samples. Latency is the median measured sample latency.

| Context | Pair | Baseline tok/s (median latency ms) | Candidate tok/s (median latency ms) | Change |
|---:|---:|---:|---:|---:|
| 512 | 1 | 83.6922 (1529.413) | 84.3560 (1517.379) | +0.79% |
| 512 | 2, reversed | 83.2450 (1537.629) | 84.1744 (1520.653) | +1.12% |
| 4096 | 1 | 80.9863 (1580.514) | 81.6728 (1567.230) | +0.85% |
| 4096 | 2, reversed | 80.9360 (1581.496) | 81.7019 (1566.672) | +0.95% |

The median of the two run medians improved from 83.4686 to 84.2652 tok/s (+0.954%) at context 512 and from 80.9612 to 81.6874 tok/s (+0.897%) at 4096. Corresponding 128-token latency from those throughput medians fell 1533.51→1519.01 ms and 1581.01→1566.95 ms. Both reversed pairs agree in direction at each context. Measured peak GPU memory was identical between arms: 6,579 MiB at context 512 and 6,803 MiB at 4096. The full per-run GPU telemetry is preserved.

## Analysis and follow-ups

The measured local gain is real and repeatable but small, as expected from the 48× concat/cache-copy site compared with the ~12 ms replay. The final main-tree Release build succeeded, the focused integrated CUDA test passed 1/1, and a final main-build model run reproduced 84.5944 tok/s at context 512 and 81.9324 tok/s at 4096 (seven samples each). The isolated candidate suite also passed its selected CTests, backend comparisons, and fixed-seed PTQ1_0/PQ2_0 model smokes. Follow up by checking fallback in rollback and multi-sequence workloads.

## Independent manager verification

The manager independently reran one seven-repetition baseline/candidate pair per context with the same benchmark settings and ≤60°C/≤5% utilization start gate. The binaries resolved their own `libggml-cuda.so.0` libraries. Their recorded build IDs were `44d3e5f` and `195b416`; a source diff across those commits is empty for the CUDA/runtime paths, so the only tested runtime-code difference is this candidate. The candidate binary included the final conservative singleton-dimension and F32-CPY guards.

| Context | Baseline median / mean tok/s | Candidate median / mean tok/s | Median delta | Mean latency ms, baseline → candidate | Peak MiB |
|---:|---:|---:|---:|---:|---:|
| 512 | 83.2259 / 83.1039 | 83.6770 / 83.5766 | +0.54% | 1540.265 → 1531.544 | 6,579 both |
| 4096 | 80.8924 / 80.7566 | 81.6427 / 81.5240 | +0.93% | 1585.031 → 1570.111 | 6,803 both |

The manager pair supports the direction of both reversed-order experimenter pairs. Raw JSON and stdout are retained in `results/exp060/manager-verify/` and `results/raw/`.

The final integrated main-tree build was then measured directly, again with seven repetitions and a fresh idle/cooldown gate per context. At context 512 it recorded 84.5944 tok/s median (84.4283 mean, 0.3540 stddev), 1516.102 ms mean latency, and 6,579 MiB peak. At context 4096 it recorded 81.9324 tok/s median (81.8168 mean, 0.3227 stddev), 1564.492 ms mean latency, and 6,803 MiB peak. These standalone final-build checks confirm the merged binary; the matched A/B above remains the optimization-effect estimate. The integrated main CUDA library SHA-256 is `8e54b08629aafaae0e835a4c630367fe8126e85f9df64199f34f0db2e2798c94`; `ldd` and hash records are in `results/exp060/raw/ldd_integrated_main.txt` and `integrated_main_sha256.txt`. Results are `results/exp060/main-final_ctx512.json` and `main-final_ctx4096.json`.

Artifacts: `results/exp060/raw/` contains graph traces, exports, summaries, focused correctness log, model-smoke comparison, loader paths, and A/B start-gate log. The experimenter made no commit; the manager retained the verified implementation in commit `4cb2072`.
