# Experiment 047: steady-state decode profile

## Objective and decision

Profile the current PTQ1_0 implementation's actual one-token CUDA graph replay on the RTX 3080 and use device kernel time per replay to rank implementation targets. No production source was changed and no optimization was attempted.

PTQ1_0 batch-1 GEMV remains the clear primary target: its three active specializations consume 9.01 ms per token at context 512 and 9.02 ms at 4096, about 75.9% and 73.8% of summed kernel duration. The largest named secondary family is QKV activation preparation at 0.752 ms/token (6.2–6.3%), followed by GDN at 0.500 ms/token (4.1–4.2%). Attention increases from 0.236 ms at context 512 to 0.588 ms at 4096. Keep GEMV first in priority when a materially new premise exists; among secondary families, profile-driven follow-up should target activation-prep fusion or cost reduction. Exp046 found no new GEMV mapping, so do not repeat its exhausted variants.

## Hardware, build, and workload

- GPU: NVIDIA GeForce RTX 3080, compute capability 8.6, 10,240 MiB; driver 580.178.04 (CUDA compatibility 13.0), CUDA toolkit 13.2.86 (`nvcc`).
- Host: Intel Core i7-10700K. Nsight Systems 2025.6.3.541.
- Manager HEAD: `bc4720ef5761556dda49cffc00cced3d9541de89`; production implementation commit remains `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`.
- `build/bin/llama-bench` SHA-256: `81187ab3fc4aeda74f92b08ca21ad774d74d1418fb2467d278b41dfe8dcdab13`.
- `build/bin/libggml-cuda.so.0` SHA-256: `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`.
- `ldd build/bin/llama-bench` resolves `libggml-cuda.so.0` to `/home/maxsun/autonomous_projects/bonsai2-rtx3080/build/bin/libggml-cuda.so.0`; CUDA runtime and CUDA libraries resolve from `/usr/local/cuda/lib64`.
- Each profile began below the required gate: context 512 at 50°C / 0% GPU utilization; context 4096 at 52°C / 0%. Model was the current PTQ1_0 GGUF with 99 GPU layers, Flash Attention enabled, batch 2048 / ubatch 512, F16 K/V, 8 CPU threads, prompt 0, 128 generated tokens, and two repetitions.

Profiler-instrumented throughput was 75.35 tok/s (512) and 73.63 tok/s (4096); these are not performance claims. The prior non-profiled best results remain 83.34575 and 80.35215 tok/s, respectively.

## Capture and analysis method

Exact command for context 512 (substitute output prefix `results/profile/exp047_ctx4096` and `-d 4096` for the other capture):

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --cuda-memory-usage=true --force-overwrite=true \
  --output results/profile/exp047_ctx512 \
  build/bin/llama-bench -m models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 \
  -r 2 -o json -p 0 -n 128 -d 512
```

The `.nsys-rep` reports are retained at `results/profile/exp047_ctx512.nsys-rep` and `exp047_ctx4096.nsys-rep`. Summary exports were made with:

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  --force-overwrite=true --output results/profile/exp047_ctx512.stats.csv \
  results/profile/exp047_ctx512.nsys-rep
nsys export --type sqlite --output /tmp/exp047_ctx512.sqlite results/profile/exp047_ctx512.nsys-rep
python3 experiments/047-steady-decode-profile/analyze_profile.py \
  /tmp/exp047_ctx512.sqlite --prefix results/profile/exp047_ctx512 --context 512
```

Repeat with `ctx4096` and context 4096. `analyze_profile.py` associates each `cudaGraphLaunch` runtime correlation ID with its GPU kernel records sharing that ID and a non-null graph ID/node ID. It checks that every launch has 1,432 kernel instances and 1,432 distinct graph nodes, then emits per-replay CSV and compact JSON. Family times are the sum of the CUDA kernel durations inside each replay, not host API time. Variability is across all 255 replay records. Graph span is reported separately; these are GPU timings under Nsight, not wall-time throughput.

The capture has exactly 255 graph-launch calls and 255 matching graph replay groups for the 2 × 128-token benchmark. Every group has the same 1,432 kernel nodes. Quantized GEMM has no graph-node instances; this confirms prompt-side GEMM/setup does not contaminate the decode-node family totals. Non-graph initialization, model load, graph construction/capture, and host setup are excluded from the per-replay sums. The trace does not mark each graph replay as warm-up versus one of the two benchmark repetitions; because all 255 have the same complete node set, any one warm-up replay included contributes only one of the 255 samples and is included in the variability summary.

Nsight Systems `cuda_gpu_kern_sum` and `cuda_api_sum` exports are retained alongside the report captures. API synchronization durations overlap device work and are not added. Nsight Compute counters remain unavailable with `ERR_NVGPUCTRPERM`; permissions were not changed.

## Per-token kernel family results

Values are mean summed kernel duration across graph replays, with per-token sample standard deviation and min–max range. Family shares divide by total summed kernel duration across the same replay set. Sub-millisecond differences are descriptive; the capture is one profile run per context, not a repeated independent profile experiment.

| Family | Context 512 ms/token (SD; range) | Share | Context 4096 ms/token (SD; range) | Share |
|---|---:|---:|---:|---:|
| PTQ1_0 GEMV, three specializations | 9.0138 (0.0044; 9.0003–9.0227) | 75.94% | 9.0222 (0.0033; 9.0085–9.0308) | 73.79% |
| Other kernels | 1.0044 (0.0030; 0.9968–1.0086) | 8.46% | 0.9995 (0.0017; 0.9919–1.0016) | 8.17% |
| QKV activation prep | 0.7524 (0.0022; 0.7467–0.7552) | 6.34% | 0.7528 (0.0011; 0.7468–0.7542) | 6.16% |
| GDN | 0.4998 (0.0017; 0.4946–0.5037) | 4.21% | 0.5003 (0.0013; 0.4951–0.5035) | 4.09% |
| RMSNorm | 0.3637 (0.0011; 0.3609–0.3656) | 3.06% | 0.3644 (0.0007; 0.3609–0.3657) | 2.98% |
| Attention | 0.2358 (0.0007; 0.2339–0.2378) | 1.99% | 0.5883 (0.0009; 0.5858–0.5910) | 4.81% |
| Quantized GEMM | 0 | 0 | 0 | 0 |

For these graph groups, the mean GPU replay span was 11.886 ms (SD 0.012 ms, range 11.857–11.901) at context 512 and 12.239 ms (SD 0.006 ms, range 12.205–12.248) at 4096. Family times sum close to the replay span; the reported family ranking reflects measured device work. PTQ1_0 GEMV per token separates into the plain specialization (4.568 ms at 512 / 4.569 ms at 4096), fused-gate (2.310 / 2.312 ms), and fused non-gate (2.136 / 2.141 ms).

The activation-prep family is `fwht_quantize_q8_1` plus `fwht_rms_quantize_q8_1`; the latter includes its fused RMS operation and is counted here, not under standalone RMSNorm. GDN includes `gated_delta_net_cuda`, `ssm_conv_f32`, and `l2_norm_f32`. RMSNorm is the remaining `rms_norm_f32` signatures. Attention is the `flash_attn` family including its stream-fixup kernel. The `Other` bucket is not one optimization target: its largest signatures are BF16 `mul_mat_vec_f` at ~0.308 ms/token, scalar copies at ~0.211 ms, concatenation at ~0.100 ms, SiLU at ~0.096 ms, row gather at ~0.094 ms, and add broadcast at ~0.086 ms (context 512); these are similar at 4096.

## Comparison with prior mixed trace

The post-Exp036 context-512 mixed trace attributed 1.166 s / 61.2% of its total summed kernel time to PTQ1_0 GEMV, with 46,572 GEMV launches. This experiment attributes 9.014 ms per direct graph replay to the same three specializations, totaling 75.9% of decode replay kernel time. The difference in percentage is expected: this trace removes initialization/setup and prompt-side work from the denominator. Multiplying the direct decode mean by 127 replays gives about 1.145 s, close to the earlier mixed trace's 1.166 s despite its extra work. The old 61.2% remains valid only as a mixed-profile share; it should not be read as steady-state decode share.

## Reproducibility limits and next experiment

The profile used one capture per context. Within-capture per-replay variability is small, but this does not estimate between-run thermal or profiler variability. Temperature and utilization met the start gates; end temperature need not be treated as the benchmark control because the reported performance metric is kernel duration during the capture. No compute counters were collected, so these timings rank kernel-family time but do not establish whether a family is bandwidth, instruction, occupancy, or launch limited.

Keep PTQ1_0 GEMV as the top optimization target only when a materially new codegen/dataflow idea appears. Since Exp046 found no distinct candidate after the prior screens, the next implementable family to investigate from this profile is coordinated QKV activation preparation (0.752 ms/token), followed by GDN (0.500 ms/token); first isolate activation-prep kernels in a focused repeated measurement and identify a specific fusion/launch or device-work reduction opportunity before implementation. Do not treat the old mixed trace as evidence that GDN or RMSNorm outranks decode GEMV.
