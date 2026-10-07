# Exp074: PTQ1_0 GEMV traffic and read ceiling

## HYPOTHESIS

The active batch-1 PTQ1_0 GEMV might be close to RTX 3080 sustained read bandwidth, making traffic reduction more promising than further instruction scheduling. This is an empirical comparison of model tensor bytes and replay time against a standalone streaming ceiling; it does not measure DRAM bytes for the GEMV.

## IMPLEMENTATION

Measurement only. Created detached worktree `/home/maxsun/autonomous_projects/bonsai2-rtx3080/.worktrees/exp074-gemv-throughput-ceiling` at requested base `8695c426f805a7575bc592ee1fed41829856d58b`. No production source or build was changed. Hardware software: driver 580.178.04, CUDA toolkit 13.2.86, Nsight Systems 2025.6.3.541, and Nsight Compute 2026.1.1.0. The active `mmvq-ptq1_0.cuh` hash is `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`, matching the Exp073 source audit; `mmvq.cu` is `e889b1543656cd2e5c0c6151f11bfdd7f641e88af441449d92064ac902b9483d`.

Verified the local model at `models/Ternary-Bonsai-2-27B-PTQ1_0.gguf`: 5,946,648,928 bytes, SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`. GGUF parsing found 851 tensors and 5,935,527,936 tensor payload bytes, beginning at data offset 11,120,992. Of these:

- 401 PTQ1_0 type-143 GEMV weight tensors, including `output.weight` and excluding `token_embd.weight`: **5,599,641,600 bytes**.
- Token embedding lookup tensor: 278,118,400 bytes; not a GEMV weight.
- Type-30 SSM alpha/beta projection weights: 47,185,920 bytes; these use separate BF16 matvec kernels.
- Other type-0 recurrent, norm, and small tensors: 10,582,016 bytes, mostly small/cache-friendly weights.

The active GEMV payload is therefore separated from total file size and from the non-GEMV payload. Fused gate launches can consume two PTQ tensors; tensor count is not replay launch count.

Commands used (from the isolated worktree):

```bash
git worktree add --detach .worktrees/exp074-gemv-throughput-ceiling 8695c426f805a7575bc592ee1fed41829856d58b
python3 experiments/074-gemv-throughput-ceiling/summarize_gguf.py /home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf > results/exp074/raw/ptq1_tensor_manifest.json
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,clocks.sm,clocks.mem --format=csv,noheader
nvcc -O3 -std=c++17 -arch=sm_86 experiments/074-gemv-throughput-ceiling/bw_ceiling.cu -o results/exp074/raw/bw_ceiling -Xcompiler -O3
./results/exp074/raw/bw_ceiling 5599641600
```

The before-run gate reading was 49 C, 0% utilization, 173 MiB used. The recorded after-run sample was 53 C, 95% utilization, 2010 MHz SM / 9251 MHz memory clocks. See `results/exp074/raw/gpu_{before,during,after}.csv`. The final binary SHA-256 is in `benchmark_binary_hash.txt`; source hashes are in `source_hashes.txt` and `bench_source_hash.txt`.

## RESULT

The current Exp062 candidate graph's retained Nsight Systems node-replay data gives PTQ1_0 GEMV-family mean durations of 9.006462 ms/token at context 512 (62 replays pooled from two 31-replay captures; pooled SD 0.003525 ms, range 8.998960–9.015731) and 9.021207 ms/token at context 4096 (31 replays; SD 0.003165 ms, range 9.013836–9.026833). The profile summaries, SQLite exports, and captures are `results/exp062/raw/candidate{,2}_ctx512.{profile.json,sqlite,nsys-rep}` and `results/exp062/raw/candidate_ctx4096.{profile.json,sqlite,nsys-rep}`. The PTQ1_0 GEMV source hash matches the active source in this worktree.

Dividing the 5.599642 GB PTQ GEMV tensor payload by those durations gives a **payload-equivalent** 621.7 GB/s and 620.7 GB/s. These are ratios, not measured DRAM throughput.

Current model file size corrects the Exp052 report's PTQ1_0 “file size” entry: its 5,935,527,936-byte value equals the tensor payload sum, not the file size. Adding the GGUF data offset (11,120,992) gives the verified 5,946,648,928-byte file size. The SHA-256 is unchanged.

## CORRECTNESS

No production candidate was implemented, and no model outputs or tensor arithmetic were changed. The read microbenchmark uses initialized synthetic bytes and checksums to preserve loads; it is not a GEMV correctness test. No new model correctness or sanitizer claim is made.

## MICROBENCHMARK

A native sm_86 CUDA event microbenchmark used a 5,599,641,600-byte working set, larger than the RTX 3080's reported 5 MiB L2. It had two warmups and nine recorded samples per mode. Each CTA reduced a checksum to one output to avoid per-thread atomics. Results use decimal GB/s and the weight working-set byte count:

| Read pattern | Median | SD | Range | Throughput |
|---|---:|---:|---:|---:|
| Contiguous `uint4` streaming | 7.7244 ms | 0.0009 ms | 7.7230–7.7257 ms | 724.93 GB/s |
| Sequential 28-byte records (seven aligned 32-bit loads/record) | 7.7257 ms | 0.0009 ms | 7.7250–7.7280 ms | 724.81 GB/s |
| 28-byte records plus reused 9-plane, 136-block planar Q8 context | 7.7279 ms | 0.0010 ms | 7.7261–7.7301 ms | 724.60 GB/s |

The planar test adds nine 16-byte activation-plane reads per record index, reusing a 19,584-byte activation context. It approximates the active layout's planar activation access while keeping the large record stream outside cache. Raw event samples are in `results/exp074/raw/bw_ceiling.stdout.txt`; the harness is `experiments/074-gemv-throughput-ceiling/bw_ceiling.cu`. This is a high-throughput synthetic read ceiling for these simple access patterns, not a guarantee that the GEMV can attain the same rate.

The manager independently reran the same binary from the main worktree after a 48 C / 0% utilization / 173 MiB idle reading. Medians were 724.80, 724.69, and 724.42 GB/s for the three modes, within 0.03% of the experimenter run. The raw output is `results/exp074/raw/manager_recheck.txt`.

## END-TO-END IMPACT

No end-to-end benchmark was run. The current best remains the established paired result, 84.407 tok/s at context 512 and 81.885 tok/s at 4096; active GEMV remains about 9.0 ms/token and 74.7–77.1% of graph kernel time.

## ANALYSIS

If each active PTQ weight byte were fetched from device memory once per token, its replay duration corresponds to about 86% of the synthetic 725 GB/s ceiling. That makes a bandwidth-near regime plausible and makes traffic reduction a reasonable hypothesis to investigate. However, the ratio combines a static GGUF tensor payload with a summed kernel duration. It does not account for cache hits, actual memory transaction amplification, or time the kernel spends decoding ternary values and reducing products. The event microbenchmark has no production kernel's instruction mix or access scheduling.

Nsight Systems establishes the GEMV family duration, and GGUF metadata establishes candidate PTQ tensor payload. Neither provides DRAM bytes or counters. Nsight Compute remains unavailable with `ERR_NVGPUCTRPERM`; no system permission was changed. Consequently this study cannot establish GEMV DRAM bandwidth, percentage of peak bandwidth, or whether memory is the binding bottleneck. The measured synthetic ceiling and payload-equivalent ratio only bound plausibility.

The Exp052 size mismatch is reconciled: the reported number was exactly the GGUF tensor payload sum; adding the tensor-data offset yields the current actual file size. No evidence of a model-file/hash mismatch was found.

## DECISION

**Measurement only; no production candidate.** The result supports treating traffic reduction as a live hypothesis, but does not justify declaring the GEMV bandwidth-bound or changing the kernel. Preserve the current production source, build, and best result.

## FOLLOW-UPS

Next bounded experiment: isolate one representative active PTQ1_0 GEMV shape in a dedicated replay harness with known resident weights and compare cold versus warm replay and working-set sizes around/beyond L2. In parallel, seek counter access through an already authorized profiling setup; do not change system-wide permissions. This would help separate cache-residency effects from the compute/dataflow floor before another kernel rewrite.

## IMPORTANT DISCOVERIES

- Exact type-143 PTQ batch-1 GEMV payload is 5,599,641,600 bytes across 401 tensors, rather than the 5.9466 GB whole GGUF file.
- The model token embedding (278 MB) is not GEMV traffic; 47.2 MB of type-30 BF16 SSM alpha/beta weights use separate matvecs; small type-0 weights total 10.6 MB.
- The actual PTQ GEMV payload divided by replay duration is ~620 GB/s, about 86% of a 725 GB/s synthetic read ceiling. This is a plausibility ratio, not DRAM measurement.
- The Exp052 file-size figure is the tensor payload sum; payload plus GGUF tensor-data offset matches today's file size and SHA.
