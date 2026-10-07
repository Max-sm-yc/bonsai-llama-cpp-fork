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

## Post-Exp036 coordinated RMS/FWHT/Q8 profile

After promoting Exp036 commit `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`, I repeated the same context-512 Nsight Systems command and workload on the actual RTX 3080: `llama-bench -p 0 -n 64 -d 512 -r 2`, with node-level CUDA graph tracing. This is again a mixed setup/decode trace. Nsight-instrumented throughput was 74.81 tok/s; use the non-profiled paired A/B for the performance claim. Raw report and CSV exports are `results/profile/ptq1_decode512_rmsfwht.nsys-rep` and `results/profile/ptq1_decode512_rmsfwht.stats.csv_cuda_gpu_kern_sum.csv` / `_cuda_api_sum.csv`.

| Kernel family | Post-ROWS=1 | Post-Exp036 | Interpretation |
|---|---:|---:|---|
| Three active PTQ1_0 GEMV variants | 1,165.7 ms / 60.32%, 46,572 launches | 1,165.5 ms / 61.22%, 46,572 launches | Still the dominant target; absolute traced time is effectively unchanged. |
| PTQ1_0 quantized GEMM | 242.8 ms / 12.56% | 242.6 ms / 12.74% | Prompt-side work unchanged. |
| RMSNorm family | 100.6 ms / 5.21%, 27,170 launches | 56.8 ms / 2.98%, 16,849 launches | The coordinated fusion eliminates about 43.8 ms from this family in the mixed trace. |
| Standard FWHT/Q8_1 | 85.1 ms / 4.41%, 33,156 launches | 59.3 ms / 3.12%, 22,835 launches | Fewer transform-to-Q8 launches. |
| New coordinated RMS/FWHT/Q8 kernel | — | 39.3 ms / 2.06%, 10,321 launches | Five 1024-element CTAs per row; included separately from the remaining standard FWHT calls. |
| Gated delta network | 88.0 ms / 4.56%, 6,240 launches | 88.2 ms / 4.63%, 6,240 launches | Unchanged secondary cost. |

The measured activation-preparation kernels (`fwht_quantize_q8_1` plus `fwht_rms_quantize_q8_1`) total 98.6 ms in the post-Exp036 mixed trace, versus 85.1 ms for the old fused-FWHT row alone; the new candidate kernel is a separate signature, so comparing only that old row would be misleading. Together, RMSNorm plus activation preparation fall from 185.7 to 155.4 ms (-30.4 ms) across these comparable traces. Overall summed kernel time fell from 1.933 s to 1.904 s; the GEMV absolute total stayed flat. Nsight Compute remains blocked by `ERR_NVGPUCTRPERM`.

At the time, the next experiment was a fresh challenge to the active `mul_mat_vec_ptq1_0_pt` dataflow; Exp046 later found no new mapping beyond screened families. The steady-state follow-up and current secondary-family ranking are in Exp047 below.

## Steady-state graph replay profile (Exp047)

On 2026-10-07, profiled the current production PTQ1_0 build on the actual RTX 3080 at contexts 512 and 4096. Driver 580.178.04 reports CUDA compatibility 13.0; the installed CUDA toolkit is 13.2.86 (`nvcc`). The binary SHA-256 was `81187ab3fc4aeda74f92b08ca21ad774d74d1418fb2467d278b41dfe8dcdab13`; `build/bin/libggml-cuda.so.0` was `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`. `ldd build/bin/llama-bench` resolved that CUDA library from the current build directory. The GPU start gates were 50°C/0% utilization (512) and 52°C/0% (4096).

Exact context-512 command (use prefix `results/profile/exp047_ctx4096` and `-d 4096` for context 4096):

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --cuda-memory-usage=true --force-overwrite=true \
  --output results/profile/exp047_ctx512 \
  build/bin/llama-bench -m models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 \
  -r 2 -o json -p 0 -n 128 -d 512
```

For each report, exported standard summaries with `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv --force-overwrite=true --output results/profile/exp047_ctx512.stats.csv results/profile/exp047_ctx512.nsys-rep`. Exported SQLite to a temporary file with `nsys export --type sqlite --output /tmp/exp047_ctx512.sqlite results/profile/exp047_ctx512.nsys-rep`, then ran `experiments/047-steady-decode-profile/analyze_profile.py /tmp/exp047_ctx512.sqlite --prefix results/profile/exp047_ctx512 --context 512`. The context-4096 invocation substitutes its matching prefix and context. The parser groups `CUPTI_ACTIVITY_KIND_KERNEL` rows by `correlationId` matching the `cudaGraphLaunch` runtime call; it filters to non-null graph ID/node records. Every one of 255 graph launch calls has a matching replay of 1,432 kernels and 1,432 unique nodes. This gives 255 directly observed one-token replays from `-n 128 -r 2`; non-graph initialization, model load, graph creation/capture, and host setup are excluded from per-token kernel-family sums. The report does not identify a replay as warm-up versus a measured repetition, so the summary includes all replays and reports per-replay variability. Quantized GEMM has zero graph-node instances, consistent with `-p 0` decode.

| Kernel family | Context 512 | Context 4096 |
|---|---:|---:|
| PTQ1_0 GEMV (all three specializations) | 9.014 ms/token (75.94%) | 9.022 ms/token (73.79%) |
| Other kernels | 1.004 ms/token (8.46%) | 1.000 ms/token (8.17%) |
| QKV activation preparation | 0.752 ms/token (6.34%) | 0.753 ms/token (6.16%) |
| GDN | 0.500 ms/token (4.21%) | 0.500 ms/token (4.09%) |
| RMSNorm not fused into preparation | 0.364 ms/token (3.06%) | 0.364 ms/token (2.98%) |
| Attention | 0.236 ms/token (1.99%) | 0.588 ms/token (4.81%) |

These are sums of individual CUDA kernel durations per replay and shares of that summed device-kernel time, not whole-process time. Replay GPU span averages 11.886 ms at context 512 and 12.239 ms at 4096. Per-replay family timing standard deviation is 0.0007–0.0045 ms for the named families; see `results/profile/exp047_ctx512.replay.json` and `exp047_ctx4096.replay.json` for ranges, exact definitions, and timing detail. Raw Nsight reports and stats CSVs use the `exp047_ctx{512,4096}` prefix under `results/profile/`.

This resolves the earlier mixed trace: the post-Exp036 1.166 s / 61.2% GEMV result included setup/decode and is not a steady-state share. Direct graph replay puts GEMV at about 74–76% of decode kernel time. GEMV remains the primary optimization target if a new implementation premise exists. Exp046 found no distinct mapping; the next measurable secondary opportunity is QKV activation preparation at 0.752 ms/token, then GDN at 0.500 ms/token. Nsight Compute remains unavailable (`ERR_NVGPUCTRPERM`), so this ranking does not establish memory bandwidth or instruction throughput limits. Nsight-instrumented throughput is not comparable to the best non-profiled decode result.

## Matched PTQ1_0 versus PQ2_0 steady-state profile (Exp052)

Using the same production runtime and RTX 3080, two reversed-order seven-repetition decode benchmark rounds measured PTQ1_0 ahead of PQ2_0 by 19.0% at context 512 and 34.7% at 4096 (median-of-run-medians; 4096 has slow tails). Peak whole-GPU use was 6,803 MiB for PTQ1_0 and 7,949 MiB for PQ2_0. The full benchmark configuration and samples are in `experiments/052-pq2-steady-profile/REPORT.md` and `results/exp052/`.

Matched node-level graph captures included 31 complete token replays per format/context. PTQ1_0 had 1,432 nodes per replay; PQ2_0 had 1,873. Both issued 361 GEMV kernels, but PTQ1_0's dedicated planar `mul_mat_vec_ptq1_0_pt` family took 9.02–9.03 ms/token while PQ2_0's generic type-142 `mul_mat_vec_q` family took 10.71 ms. PQ2_0 also spent about 0.59 ms/token more in activation preparation and standalone RMSNorm, mostly from 361 separate Q8 quantization nodes and 80 additional standalone RMSNorm nodes. GEMV remains about 74–76% of summed PTQ1_0 decode kernel time and the dominant format-level difference. The PTQ1_0 GGUF file is 17.5% smaller; this is consistent with lower weight traffic but does not establish the GEMV hardware limit. Nsight Compute counters remain unavailable (`ERR_NVGPUCTRPERM`), so these Nsight Systems traces cannot distinguish memory bandwidth from integer instruction throughput, occupancy, or layout effects.

## Active sm_86 FlashAttention tile screen (Exp053)

At context 4096, the active PTQ1_0 attention path uses `flash_attn_ext_f16<256,256,1,8,...>` plus Stream-K fixup: 0.5511 ms/token main attention and 0.0359 ms fixup, 0.5869 ms total. At context 512 the corresponding totals are 0.2007 + 0.0344 = 0.2351 ms/token. On RTX 3080 the dispatch selects the Ampere helper with `(ncols1,ncols2)=(1,8)`; Turing-helper test captures were excluded as no-op profiles.

The tested 64/64 single-stage Ampere tile reduced shared K/V storage from 67,584 to 17,408 bytes but increased attention-family time to 0.2754 ms/token at context 512 (+17.2%) and 0.6533 ms at context 4096 (+11.3%). At 4096, its fixup grew to 0.0920 ms/token. A 96/96 configuration did not compile because of a static loop-size invariant. The candidate passed selected correctness checks but failed the focused performance gate, so no end-to-end model A/B was run and the production source stayed unchanged. Per-replay timing, commands, and logs: `experiments/053-flash-attention-longctx/REPORT.md` and `results/exp053/`. GEMV remains the primary decode bottleneck at roughly 74% of summed kernel time; reopen attention tile work only with a design that addresses Stream-K fixup cost.
