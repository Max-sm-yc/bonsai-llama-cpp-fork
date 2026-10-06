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
| 3 | Gated delta network: 4.4% (88.9 ms); fused FWHT/Q8_1 quantization: 4.2% (85.6 ms) | Gated delta network: 4.0% (89.6 ms); FWHT plus Q8_1 quantization: 6.4% combined (142.3 ms) | Recurrent attention and activation preparation are the next measured costs. |
| 4 | RMSNorm: 3.8% (76.8 ms) | RMSNorm: 3.7% (81.2 ms) | Repeated norm work is visible but well below ternary matvec. |
| 5 | Flash attention: 1.3% (26.5 ms) | Flash attention: 1.1% (25.1 ms) | Attention is not a leading cost in this context-512 trace. |

The CUDA API summary lists 127 graph launches for each profile. `cudaStreamSynchronize` and asynchronous-copy API durations include time waiting on device work, so they overlap kernel time and must not be added to the kernel totals as independent bottlenecks.

## Findings

- PTQ1_0 uses a dedicated `mul_mat_vec_ptq1_0_pt` path. Two of three variants with matching launch counts are about 13–16% faster per launch than their PQ2_0 counterparts; the third is about 2.5% slower. Across the three variants, PTQ1_0 spends about 140 ms less in matvec kernels in this trace. Its weight payload is 17.6% smaller. This is consistent with weight traffic being important; without hardware counters, the trace cannot prove whether the limiting factor is memory bandwidth, instruction throughput, or occupancy.
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

The next experiment should focus on the PTQ1_0 `mmvq.cu`/`vecdotq.cuh` path on sm_86. A microbenchmark-only win is insufficient; compare the complete controlled decode and combined workloads and run the CUDA-vs-CPU correctness suite.
