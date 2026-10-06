# Baseline profile

Captured on the actual RTX 3080 from the reference build with Nsight Systems 2025.6.3. Each run used `llama-bench -p 0 -n 64 -d 512 -r 2`, the same CUDA/offload/Flash Attention/KV settings as the benchmark, and `--cuda-graph-trace=node`. The trace includes the default warm-up, context setup, and two 64-token measured decode repetitions. It is a mixed setup/decode trace, so percentages rank the full captured workload rather than decode-only time.

Raw reports: `results/profile/ptq1_decode512.nsys-rep` and `results/profile/pq2_decode512.nsys-rep`. The compact `cuda_gpu_kern_sum` and `cuda_api_sum` exports are adjacent CSV files. Recreate each trace with:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --cuda-memory-usage=true --force-overwrite=true \
  --output results/profile/ptq1_decode512 \
  build/bin/llama-bench -m models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 \
  -r 2 -o json -p 0 -n 64 -d 512

# Repeat with PQ2_0.gguf and output prefix pq2_decode512.
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum \
  --format csv --force-overwrite=true \
  --output results/profile/ptq1_decode512.stats.csv \
  results/profile/ptq1_decode512.nsys-rep
```

## Measured kernel ranking

| Rank | PTQ1_0 | PQ2_0 | Interpretation |
|---:|---|---|---|
| 1 | Three `mul_mat_vec_ptq1_0_pt` variants: 61.8% of GPU kernel time (1.253 s total) | Three `mul_mat_vec_q<type 142>` variants: 62.6% (1.393 s) | Quantized matrix-vector products dominate both traces. On PTQ1_0, the three variants account for 31,219, 10,192, and 5,161 launches. |
| 2 | `mul_mat_q<type 143>`: 12.1% (245.9 ms) | `mul_mat_q<type 142>`: 11.1% (247.8 ms) | Quantized matrix-matrix work, primarily prompt processing in this mixed trace. |
| 3 | RMSNorm family: 5.00% (101.4 ms) | FWHT plus Q8_1 quantization: 6.4% combined (142.3 ms) | RMSNorm includes both 1024-thread and 256-thread weighted signatures. |
| 4 | Gated delta network: 4.4% (88.9 ms); fused FWHT/Q8_1 quantization: 4.2% (85.6 ms) | RMSNorm family: 4.80% (106.7 ms); gated delta network: 4.0% (89.6 ms) | Recurrent attention and activation preparation remain visible secondary costs. |
| 5 | Flash attention: 1.3% (26.5 ms) | Flash attention: 1.1% (25.1 ms) | Attention is not a leading cost in this context-512 trace. |

The CUDA API summary lists 127 graph launches for each profile. `cudaStreamSynchronize` and asynchronous-copy API durations include time waiting on device work, so they overlap kernel time and must not be added to the kernel totals as independent bottlenecks.

## Findings

- PTQ1_0 uses a dedicated `mul_mat_vec_ptq1_0_pt` path. Two of three variants with matching launch counts are about 13–16% faster per launch than their PQ2_0 counterparts; the third is about 2.5% slower. Across the three variants, PTQ1_0 spends about 140 ms less in matvec kernels in this trace. Its weight payload is 17.6% smaller. This is consistent with weight traffic being important; without hardware counters, the trace cannot prove whether the limiting factor is memory bandwidth, instruction throughput, or occupancy.
- Source dispatch audit for this target: sm_86 selects the planar-transposed `GGML_CUDA_Q8_1_PT` activation layout, then plain one-column PTQ1_0 `MUL_MAT` enters `mul_mat_vec_ptq1_0_pt` with 128-thread CTAs. This bypasses the generic one-column warp-count selection. Experiments 003 and 007–009 must therefore not be interpreted as direct tuning of the active RTX 3080 kernel; see their manager audits/reports.
- The top PTQ1_0 variants use 106–126 registers per thread in the sm_86 cubin according to `cuobjdump --dump-resource-usage`; the profiled variants have no local-memory stack spills. Register pressure is a plausible tuning dimension, not yet a measured occupancy bottleneck. See `results/profile/mmvq_resource_usage.txt`.
- The PTQ1_0 path has a fused Hadamard-to-Q8_1 kernel. PQ2_0 uses separate Hadamard and activation-quantization kernels in this graph. Their combined trace cost is about 56 ms higher; extending compatible fusion to PQ2_0 is a secondary candidate.
- Prompt processing was about 1.3k tokens/s for both formats in the controlled baseline. Its quantized `mul_mat_q` contribution is meaningful but smaller than the decode-focused matvec family.
- Nsys traces show near-saturated GPU utilization. The controlled benchmark's temperature reached 89°C, and individual repetition speed shifted during a run. Use the cooldown gate and seven-sample median for subsequent comparisons.

## Profiling limitation

Nsight Compute 2026.1.1 attached to the real workload but hardware-counter collection failed with `ERR_NVGPUCTRPERM` for this user. No system-wide driver or counter-permission setting was changed. Nsight Systems tracing, `nvidia-smi` telemetry, source inspection, and static cubin resource usage remain available. Memory-throughput, integer-pipe utilization, and measured occupancy counters remain unresolved.

## Bottleneck priority

1. PTQ1_0 batch-1 ternary GEMV: unpack/dot path dominates the trace and the end-to-end decode metric.
2. Fused gate/up GEMV variants in the same kernel family: they have substantial measured cumulative time and distinct compile-time specializations.
3. Prompt-side ternary GEMM and activation quantization/Hadamard scheduling.
4. Gated-delta recurrent attention and RMSNorm fusion/launch overhead.

The next experiment should focus on the active PTQ1_0 `mmvq-ptq1_0.cuh` path on sm_86. A microbenchmark-only win is insufficient; compare controlled model decode and combined workloads and run the CUDA-vs-CPU correctness suite.

## Post-ROWS=1 profile

Re-profiled the final source-default ROWS=1 build on 2026-10-06 with Nsight Systems 2025.6.3.541, after the matched decode A/B and a 59°C/0%-utilization idle gate. The command, model, batch/offload/KV options, context, warm-up, and measured 64-token repetitions match the reference trace. This is still a mixed setup/decode trace, not a decode-only kernel timer. The trace reports 73.09 tok/s under profiler instrumentation; use the non-profiled benchmark for throughput.

Raw profile: `results/profile/ptq1_decode512_rows1.nsys-rep`; CSV exports: `results/profile/ptq1_decode512_rows1.stats.csv_cuda_gpu_kern_sum.csv` and `_cuda_api_sum.csv`. Recreate with the command above, substituting output prefix `ptq1_decode512_rows1` and the checked-out ROWS=1 build.

| Rank | ROWS=1 kernel family | Time | Reference trace | Interpretation |
|---:|---|---:|---:|---|
| 1 | Three `mul_mat_vec_ptq1_0_pt` variants (ROWS=1) | 60.4%, 1.166 s total | 61.8%, 1.253 s | Still dominates; combined time fell about 87.5 ms (7.0%) in the same mixed trace. |
| 2 | `mul_mat_q<(ggml_type)143>` | 12.6%, 242.8 ms | 12.1%, 245.9 ms | Prompt-side matrix work is essentially unchanged. |
| 3 | RMSNorm family | 5.21%, 100.61 ms | 5.00%, 101.39 ms | Includes 1024-thread and 256-thread weighted signatures; the 3.9% entry in the original kernel grouping counted only the 1024-thread signature. |
| 4 | Gated delta network | 4.6%, 88.0 ms | 4.4%, 88.9 ms | Recurrent attention remains a secondary cost. |
| 5 | Fused FWHT/Q8_1 quantization | 4.4%, 85.1 ms | 4.2%, 85.6 ms | Activation preparation is essentially unchanged. |
| 6 | Flash attention | 1.3%, 25.5 ms | 1.3%, 26.5 ms | Not a leading context-512 cost. |

The three GEMV specializations retain the same launch counts (31,219 / 10,192 / 5,161) but now instantiate one row/item instead of four. Their individual totals are 590.8, 298.6, and 276.3 ms. The corresponding reference totals were 616.1, 330.1, and 307.0 ms. The one-row mapping therefore improved all three measured variants in this trace, while total kernel-time share remains about 60%; the active GEMV is still the highest-value optimization target.

The CUDA API summary still has 127 graph-launch calls. Synchronization and async-copy API durations overlap device work and are not additive bottleneck totals. Nsight Compute permissions remain unavailable, so this profile does not distinguish bandwidth, integer throughput, and occupancy limits.

## RMSNorm family accounting and geometry screen (experiment 022)

The post-ROWS=1 CSV contains 16,770 calls / 76.18 ms for `rms_norm_f32<1024,true,false>` and 10,400 calls / 24.43 ms for `rms_norm_f32<256,true,false>`. Together they account for 100.61 ms, or 5.21% of the 1.933 s summed kernel time. The former alone is 3.94%, matching the previously reported 3.9% row; the 256-thread signature had been omitted from that family total. The baseline CSV has the same call counts and 101.39 ms combined family time. In the PQ2_0 CSV, the two signatures account for 81.21 ms / 16,770 calls and 25.46 ms / 10,400 calls (106.67 ms total, or 4.80% of 2.224 s).

The fused-weight dispatch uses the 1024-thread specialization for `ncols >= 1024`, and the 256-thread specialization below that threshold. A full-model decode screen changed only that branch to 256 threads; it lost 3.32% at context 512 and 3.25% at 4096 (7 repetitions, 128 tokens), so the original 1024-thread dispatch remains active. Both fixed-seed 32-token PTQ1_0/PQ2_0 model outputs matched after removing only build and timing text. The broader standalone CTest/CUDA-vs-CPU suite did not complete because its script initiated a 393-target rebuild after the clear end-to-end regression; no correctness-suite pass is claimed for this rejected candidate. See `experiments/022-rmsnorm-sm86/REPORT.md`.
