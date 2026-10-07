# Exp079: PTQ1_0 exact L2 reuse and persisting-policy screen

## HYPOTHESIS

The active PTQ1_0 batch-1 GEMV reads a selected K/V matrix again at the next token after a very large intervening weight traversal. A graph-capturable L2 persisting access policy might retain one or a few of the 1,146,880-byte matrices and reduce cold replay latency even though the full PTQ reuse distance is about 5.598 GB.

## IMPLEMENTATION

This experiment used isolated worktree `.worktrees/exp079-ptq1-l2-persistence` based on commit `44132a0`. The manager checkout and production best were not modified. The selected model is `/home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf` (SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`).

A narrow Exp079-only hook in `mmvq.cu` invokes the production `mul_mat_vec_ptq1_0_pt_launch<1>` specialization for the real `[5120,1024]` GGUF tensors, with K=5120, 1024 output rows, and 40 PTQ blocks per row. The hook is compiled in the isolated CUDA library. It reads the real packed model tensor bytes and a deterministic nonzero planar Q8_1 activation (8 quant planes plus the scale/sum plane, 5,760 bytes). A CPU reference using `dequantize_row_ptq1_0` matched all 1,024 GPU outputs exactly (0 mismatches, max absolute error 0).

The microbenchmark captured exact production kernel launches into CUDA Graphs. It tested 1, 4, and 32 contiguous real matrices (1.14688, 4.58752, and 36.70016 MB), 10 warmup graph replays, then 25 warm and 25 cold replay samples per condition. Cold samples ran a 32 MiB eviction kernel before the start event; event timing enclosed only the graph replay. A policy condition attached `cudaAccessPolicyWindow` to each graph kernel node with hit ratio 1.0, persisting hits, and streaming misses. The four and 32 tensor windows were valid contiguous spans in one device allocation. The CUDA API accepted all graph-node attributes; the graph had exactly 1/4/32 kernel nodes for each set.

The first E2E candidate reserved the device maximum persisting-L2 limit, 3,604,480 bytes (3.44 MiB), and applied the policy to the first eligible K/V PTQ1_0 graph node only. This was selected because the device L2 is 5,242,880 bytes (5 MiB), and the one tensor fits within the persisting reservation. API header checks confirmed that graph node access policy is a hint without a performance guarantee; the experiment therefore relies only on measured latency, not assumed residency. The separate exact-size arm reserved only the selected tensor window.

Build command:

```sh
cmake -S . -B results/exp079/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=OFF
cmake --build results/exp079/build --target llama-bench -j 8
cmake --build results/exp079/build --target llama-cli -j 8
```

The isolated maximum-reservation CUDA library SHA-256 is retained at `results/exp079/raw/libggml-cuda-max-reservation.so` (`c6aa742c51fcaf746ff22b86b6eb9b1315d1f35084e91b497990bea16db3ccf3`). The later exact-size candidate library SHA-256 is `d7f192e33bcec0a9fa0d35e51088daa3f0629863403bd0689f972789287ad99c`; its final `mmvq-ptq1_0.cuh` SHA-256 is `eda448bad948998e76f023215fe16dbcf8e7fc8d9920c3fc23ba79cc311e0466`. Its source and build were produced from commit `44132a0`; the baseline PTQ1_0 kernel header SHA-256 was `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`. The complete isolated harness is in `results/exp079/raw/replay.cu`, and the model tensor names and sizes are in `results/exp079/raw/selected_tensors.json`.

## RESULT

### Microbenchmark

Median CUDA-event time per captured graph sequence (25 samples each, microseconds):

| Matrix set | Policy | Warm | Forced cold |
|---|---|---:|---:|
| 1 matrix (1.147 MB) | default | 5.120 | 7.168 |
| 1 matrix (1.147 MB) | persisting | 5.472 | 5.440 |
| 4 matrices (4.588 MB) | default | 17.376 | 21.504 |
| 4 matrices (4.588 MB) | persisting | 17.408 | 21.504 |
| 32 matrices (36.700 MB) | default | 154.624 | 156.672 |
| 32 matrices (36.700 MB) | persisting | 154.624 | 155.648 |

The one-matrix cold-policy median was 1.728 µs lower than cold default, while its warm median was 0.352 µs slower. The 4-matrix and 32-matrix sets had no meaningful policy gain. Raw samples are `results/exp079/raw/events.csv`; conditions and exact source are represented by `results/exp079/raw/run.log` and `results/exp079/raw/selected_tensors.json`.

### End-to-end decode

The maximum-reservation candidate applied an access window to one selected 1.14688 MB K/V tensor. Two reversed-order A/B pairs used a fixed PTQ1_0 model, context 512 or 4096, 128 generated tokens, seven `llama-bench` repetitions per arm, and a 60 C / 5% GPU-utilization start gate before every run. Candidate policy-attachment markers are preserved in the stderr logs. Peak VRAM was 6,579 MiB at context 512 and 6,803 MiB at context 4096.

| Context | Pair 1 baseline / policy | Pair 2 baseline / policy | Median baseline | Median policy | Change |
|---|---:|---:|---:|---:|---:|
| 512 | 84.123 / 83.445 tok/s | 84.217 / 83.390 tok/s | 84.170 | 83.418 | -0.894% |
| 4096 | 81.781 / 81.051 tok/s | 81.714 / 81.051 tok/s | 81.747 | 81.051 | -0.852% |

Raw JSON and telemetry are in `results/exp079/ptq1-decode-paired.json`; arm stdout and stderr are in `results/exp079/raw/ctx*_pair*`.

## CORRECTNESS

A deterministic 32-token fixed completion smoke used `tests/model_smoke.py` with identical model, seed, context, and decode parameters. After removing the timing footer, baseline and candidate response strings were identical (119 characters). The captures are `results/exp079/fixed-smoke-baseline.json` and `results/exp079/fixed-smoke-candidate.json`; normalized exact response parity is recorded in `results/exp079/fixed-smoke-compare.json`, and raw outputs are retained under `results/raw/`. The direct production-kernel microbenchmark also matched the CPU reference exactly for every output row.

## MICROBENCHMARK DETAILS

Start telemetry for the maximum-reservation event run was 52 C / 0% utilization / 173 MiB. The 32 MiB eviction area is 6.4 times device L2 and the eviction kernel was outside the timed interval. At 1.147 MB, one selected weight fits both L2 and the maximum persisting reservation. Four matrices exceed the 3.44 MiB reservation, and all 32 exceed both the reservation and L2. Nsight Compute counters remain unavailable (`ERR_NVGPUCTRPERM`); no cache-hit or DRAM-byte count is claimed.

## END-TO-END IMPACT

The maximum-reservation access policy regressed paired decode by about 0.9%; exact-size reservation measured -0.159% at context 512 and flat (-0.0003%) at context 4096. Both stayed below the 10,240 MiB device limit. No production speedup was observed, so no policy was retained in the production tree.

## ANALYSIS

The exact production kernel is graph-capturable with `cudaAccessPolicyWindow`, and forcing an eviction showed a repeatable cold-replay improvement for one small tensor. Warm one-matrix latency did not improve. The benefit disappeared for four and 32 selected matrices, while the real model A/B regressed even though only one K/V tensor received the policy. This indicates the forced-cold result does not translate to model decode; full-model traffic, policy reservation, cache replacement, and launch-level behavior are not captured by the small isolated replay.

The maximum reservation was much larger than the one 1.147 MB policy window. A separate exact-size reservation follow-up then set `cudaLimitPersistingL2CacheSize` to only the 1,146,880-byte selected window. It reduced the model regression substantially but remained flat or slightly negative, so it also did not qualify for promotion.

## EXACT-SIZE RESERVATION FOLLOW-UP

After the maximum-reservation candidate was rejected, a separate candidate set `cudaLimitPersistingL2CacheSize` to exactly the selected 1,146,880-byte access window instead of the device maximum 3,604,480 bytes. It attached the same persisting/streaming `cudaAccessPolicyWindow` to only the first eligible `[5120,1024]` K/V graph node. The updated candidate was rebuilt independently, passed the same exact CPU output check and fixed 32-token response comparison, and used the same 2 reversed A/B pairs per context, 7 repetitions, 128 decode tokens, and ≤60 C / ≤5% start gate.

The exact-size microbenchmark retained the cold single-matrix effect but did not improve warm replay. Median graph-sequence time (µs): one matrix default/persisting was 5.120/5.312 warm and 7.168/6.144 cold; four matrices were 17.408/17.408 warm and 22.528/22.528 cold; 32 matrices were 155.296/155.456 warm and 157.696/157.696 cold. All 300 event samples are in `results/exp079/raw/events-exact-reservation.csv`; the correctness and graph-node report is `results/exp079/raw/run-exact-reservation.log`. The run started at 60 C / 0% utilization.

| Context | Pair 1 baseline / policy | Pair 2 baseline / policy | Median baseline | Median policy | Change |
|---|---:|---:|---:|---:|---:|
| 512 | 84.345 / 84.128 tok/s | 84.184 / 84.133 tok/s | 84.265 | 84.130 | -0.159% |
| 4096 | 81.760 / 81.774 tok/s | 81.765 / 81.750 tok/s | 81.762 | 81.762 | -0.0003% |

Per-arm gate samples, telemetry, stdout, and stderr are in `results/exp079/ptq1-decode-exact-reservation-paired.json` and matching `results/exp079/raw/ctx*_exact` files. Peak VRAM was 6,579/6,803 MiB. Exact-size fixed output parity is in `results/exp079/fixed-smoke-exact-compare.json`.

The smaller reservation avoids most of the maximum-reservation loss at context 512, but both contexts remain flat or slightly negative. No repeatable end-to-end gain was measured. This follow-up is also rejected.

## FINAL DECISION

**REVERT both policy candidates.** The maximum-reservation policy regressed paired decode by about 0.9% at both contexts. The exact-size reservation reduced that regression to a small ctx512 loss and a flat ctx4096 result, but did not produce a repeatable gain. The single-matrix forced-cold replay effect does not translate to faster model decode. Keep the current production implementation and current-best result unchanged; no model speedup is claimed.

## FOLLOW-UPS

- Retain the selected weight bytes, planar activation, both sets of 300 event samples, source/build hashes, correctness logs, E2E JSON, and telemetry in `results/exp079/`.
- Do not promote either access-policy variant without repeatable paired E2E improvement and exact output parity.

## IMPORTANT DISCOVERIES

- RTX 3080 L2 is 5 MiB; the device maximum persisting-L2 reservation queried on this setup was 3,604,480 bytes.
- The 32 active K/V matrices are each 1,146,880 bytes; a set of four is about 4.38 MiB and all 32 total 36.7 MB.
- Exact graph-node policy attachment is supported for the production PTQ1_0 kernel on this CUDA 13.2 / sm_86 build.
- One-matrix forced-cold latency improved with persistence, but warm latency did not, and model decode regressed by 0.85–0.89%.
- No source change was merged; the production baseline remains unchanged.
