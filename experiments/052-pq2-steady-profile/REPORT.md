# Experiment 052: matched PTQ1_0 and PQ2_0 steady decode profile

## Objective and decision

Compare the current PTQ1_0 and PQ2_0 model formats using the same production runtime and RTX 3080, then separate steady-state graph replay time into active quantized GEMV, activation preparation, RMSNorm, and other families. No source, build configuration, or production binary was changed.

PTQ1_0 leads in both the matched decode benchmark and the graph profile. The active GEMV family explains most of the traced difference: the PTQ1_0 dedicated planar kernel takes about 9.02 ms per token, while PQ2_0's generic type-142 matvec takes about 10.71 ms, with the same 361 GEMV graph nodes per replay. PTQ1_0 also has 441 fewer graph nodes per token and saves about 0.60 ms in activation preparation plus standalone RMSNorm, consistent with its fused planar Q8 preparation path. Its GGUF weight file is 17.5% smaller. Nsight Systems does not establish whether the GEMV advantage comes from lower memory traffic, different integer work, occupancy, or their combination.

## Setup and artifacts

- GPU: NVIDIA GeForce RTX 3080, sm_86, 10,240 MiB; driver 580.178.04 (CUDA compatibility 13.0); CUDA toolkit 13.2.86; Nsight Systems 2025.6.3.541.
- Research HEAD: `8e9c9303b0ebb59ecbe78e463a36d7e493ac5fbb`. Production source remains commit `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`.
- `build/bin/llama-bench` SHA-256: `81187ab3fc4aeda74f92b08ca21ad774d74d1418fb2467d278b41dfe8dcdab13`.
- `build/bin/libggml-cuda.so.0` SHA-256: `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642` (matches the expected library).
- `ldd build/bin/llama-bench` resolves `libggml-cuda.so.0` to the current build directory and CUDA libraries from `/usr/local/cuda/lib64`.
- PTQ1_0 GGUF: `models/Ternary-Bonsai-2-27B-PTQ1_0.gguf`, SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`, 5,935,527,936 bytes.
- PQ2_0 GGUF: `models/Ternary-Bonsai-2-27B-PQ2_0.gguf`, SHA-256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`, 7,195,047,936 bytes. The model has the same 26,895,998,464 parameters; its payload file is 17.5% larger than PTQ1_0.
- Both benchmark arms loaded and decoded successfully. Peak whole-GPU memory was 6,803 MiB for PTQ1_0 and 7,949 MiB for PQ2_0. No new independent token-equivalence test was run; existing project correctness records are unchanged.

## Matched decode benchmark

Ran two order-reversed rounds. Each arm used the same `build/bin/llama-bench`, contexts 512 and 4096, 128 generated tokens, seven repetitions, default warmups, `-ngl 99`, Flash Attention, batch/ubatch 2048/512, eight CPU threads, and F16 K/V. A fresh gate required <=60 C and <=5% GPU utilization before each format. Round 1 ran PTQ1_0 then PQ2_0; round 2 ran PQ2_0 then PTQ1_0. The gate samples were PTQ/PQ 52/59 C in round 1 and PQ/PTQ 59/60 C in round 2; all were at 0% utilization.

Commands:

```bash
python3 benchmark/run.py \
  --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --model PQ2_0=models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --modes decode --contexts 512 4096 --decode-tokens 128 \
  --repetitions 7 --cooldown-temp-c 60 --output results/exp052/matched_decode.json
python3 benchmark/run.py \
  --model PQ2_0=models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --modes decode --contexts 512 4096 --decode-tokens 128 \
  --repetitions 7 --cooldown-temp-c 60 \
  --output results/exp052/matched_decode_reverse.json
```

| Context | PTQ1_0 run medians (tok/s) | PQ2_0 run medians (tok/s) | Median-of-run-medians change |
|---:|---:|---:|---:|
| 512 | 83.6354, 83.2879 | 70.0899, 70.1841 | PTQ1_0 +19.0% |
| 4096 | 81.0677, 79.0127 | 61.7282, 57.1057 | PTQ1_0 +34.7% |

The context-4096 samples have slow tails in both formats. PTQ1_0 ranges were 80.27–81.10 and 67.55–80.63 tok/s; PQ2_0 ranges were 51.50–68.26 and 44.24–68.26. Keep the full samples in the JSON files; the long-context percentage is less stable than the context-512 result. Peak memory readings above are absolute whole-GPU use.

## Decode-only CUDA graph profile

The profile used node-level graph replay tracing. Initial `-n 128 -r 2` captures produced Nsight Systems exit code 139 and incomplete activity for some arms, so those traces are retained under `results/exp052/raw/incomplete-long-attempts/` and excluded. The final matched profile uses `-n 16 -r 2`; all four `nsys profile` processes completed with code 0 and recorded 31 complete one-token replays per format/context. Every replay has 1,432 PTQ1_0 graph nodes or 1,873 PQ2_0 graph nodes, with a stable node count across all 31 replays. Start gates for the final four captures were 57 C (PTQ1_0/512), 59 C (PQ2_0/512), 59 C (PTQ1_0/4096), and 59 C (PQ2_0/4096), each at 0% utilization and 173 MiB whole-GPU use.

Common command template (the four exact model/context commands and gate samples are also saved in `results/exp052/raw/profile_setup.json`):

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --cuda-memory-usage=true --force-overwrite=true \
  --output results/exp052/raw/{format}_ctx{context}_short \
  build/bin/llama-bench -m models/Ternary-Bonsai-2-27B-{format}.gguf \
  -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 \
  -r 2 -o json -p 0 -n 16 -d {context}
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  --force-overwrite=true --force-export=true \
  --output results/exp052/raw/{format}_ctx{context}_short.stats.csv \
  results/exp052/raw/{format}_ctx{context}_short.nsys-rep
nsys export --type sqlite --force-overwrite=true \
  --output results/exp052/raw/{format}_ctx{context}_short.sqlite \
  results/exp052/raw/{format}_ctx{context}_short.nsys-rep
python3 results/exp052/analyze.py \
  results/exp052/raw/{format}_ctx{context}_short.sqlite
```

Per-replay family times are mean summed CUDA kernel duration; share is of summed device kernel time. Replay span is the earliest-to-latest graph node GPU timestamp. Quantized GEMM had zero graph-node instances in all four decode captures.

| Format/context | Nodes and kernel instances per replay | Mean GPU span (range) ms | GEMV ms/share | Activation prep ms/share | RMSNorm ms/share | GDN ms | Attention ms | Other ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PTQ1_0 / 512 | 1,432 / 1,432 | 11.911 (11.887–11.923) | 9.022 / 75.9% | 0.760 / 6.4% | 0.366 / 3.1% | 0.502 | 0.237 | 1.009 |
| PQ2_0 / 512 | 1,873 / 1,873 | 14.249 (14.043–14.299) | 10.714 / 75.5% | 0.976 / 6.9% | 0.736 / 5.2% | 0.510 | 0.228 | 1.034 |
| PTQ1_0 / 4096 | 1,432 / 1,432 | 12.262 (12.231–12.276) | 9.030 / 73.7% | 0.759 / 6.2% | 0.366 / 3.0% | 0.502 | 0.589 | 1.003 |
| PQ2_0 / 4096 | 1,873 / 1,873 | 14.602 (14.443–14.638) | 10.706 / 73.6% | 0.976 / 6.7% | 0.737 / 5.1% | 0.510 | 0.585 | 1.040 |

Family replay counts are constant per token: GEMV 361, activation preparation 258 PTQ1_0 / 619 PQ2_0, RMSNorm 129 / 209, GDN 144, attention 32, and Other 508. The 441-node PQ2_0 increase is exactly 361 extra separate Q8 quantization nodes plus 80 extra standalone RMSNorm nodes. The common activation-transform work remains represented in each format; PTQ1_0's `fwht_rms_quantize_q8_1` and planar Q8 path combine work that PQ2_0 expresses in separate kernels.

The three GEMV signatures have the same per-replay launch counts in both models:

| Active signature | Count/replay | PTQ1_0 mean ms (512 / 4096) | PQ2_0 mean ms (512 / 4096) |
|---|---:|---:|---:|
| Plain: PTQ `mul_mat_vec_ptq1_0_pt<1,1,false,false>`; PQ2 `mul_mat_vec_q<type 142,1,false,false,false,false,false>` | 242 | 4.572 / 4.573 | 5.485 / 5.476 |
| Fused specialization A: PTQ `<1,1,true,false>`; PQ2 `<type 142,1,true,false,false,false,false>` | 79 | 2.139 / 2.143 | 2.489 / 2.491 |
| Fused specialization B: PTQ `<1,1,true,true>`; PQ2 `<type 142,1,true,true,false,false,false>` | 40 | 2.311 / 2.313 | 2.740 / 2.740 |

Combined GEMV is 9.02–9.03 ms for PTQ1_0 and 10.71 ms for PQ2_0, an 18.7% higher PQ2_0 graph-node duration. GEMV contributes roughly 1.68–1.69 ms/token of the 2.30–2.34 ms summed-kernel gap. PQ2_0's additional activation-prep and RMSNorm work contributes about 0.59 ms/token. GDN, attention, and the `Other` bucket are otherwise similar; the `Other` bucket is 1.00–1.04 ms/token in both formats.

## Interpretation and limits

The evidence points to two supported sources of PTQ1_0's decode lead:

1. Its dedicated `mul_mat_vec_ptq1_0_pt` implementation runs faster than the same-count generic `mul_mat_vec_q<type 142>` GEMV family in all three active signatures. The smaller PTQ1_0 model payload is consistent with less weight data, but these profiles cannot separate payload traffic from different decode instructions, layout, or other kernel behavior.
2. PTQ1_0's activation path uses fewer graph nodes, including fused RMS/FWHT/Q8 preparation. This saves about 0.59 ms/token in activation prep plus standalone RMSNorm. Most of the graph time gap remains in GEMV; general non-GEMV overhead is not the main explanation.

Nsight Compute counters were not collected. Exp047 records `ERR_NVGPUCTRPERM`; no permission or driver setting was changed. Nsight Systems timing alone does not establish whether the GEMV is limited by memory bandwidth, integer throughput, occupancy, or a mixture. The benchmark has two reversed-order pairs, but only seven repetitions per run; context-4096 tails are substantial. The profile is one gated capture per format/context with 31 replays, sufficient for a stable node-level family comparison but not a repeated independent profile study.

## Decision

Measurement only. Preserve the production source, binary, and `BEST_RESULTS.json`; this experiment does not establish a new verified best. Keep the active PTQ1_0 specialized GEMV and fused activation path as observations for future work, not as a new optimization proposal. Current production and research worktree changes are limited to the requested experiment/report artifacts and compact research notes; no commit was made.
