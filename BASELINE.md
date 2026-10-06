# Reference baseline

Status: these reference measurements were verified on the RTX 3080 using unmodified PrismML runtime commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`. The current optimized project commit is `9fa97200e68fd798ef027470c8e420172a0ac719`; see the matched isolated comparison below.

## Conditions

- GPU: NVIDIA GeForce RTX 3080, GA102, sm_86, 10 GiB; CUDA reports 9867 MiB usable.
- Runtime build: Release, CUDA enabled, `CMAKE_CUDA_ARCHITECTURES=86`.
- Model: Ternary Bonsai 2 27B at repository revision `b072e1d3b35a0a630cece372c2127528e0994386`; file SHA-256 values are in [SETUP.md](SETUP.md).
- Seven measured repetitions per row, with llama-bench's default warm-up enabled. Throughput below is the median of the seven raw repetitions; all individual samples, arithmetic means, standard deviations, commands, and GPU samples are in [results/baseline.json](results/baseline.json).
- The harness waited before each format until GPU temperature was at most 60°C and utilization at most 5%. It then ran prefill, decode, and combined workloads in that order. GPU temperature reached 89°C during the complete matrix. Both formats used identical contexts, batch sizes, KV types, GPU offload, Flash Attention, thread count, and repetition count.
- Batch 1 decode generated 128 tokens at contexts 128, 512, 2048, and 4096. Prefill used the same prompt lengths. Combined throughput measures prompt evaluation plus 128 generated tokens; llama-bench excludes tokenization and sampling.
- Batch 2048, microbatch 512, 8 CPU threads, F16 K/V cache, 99 GPU layers, Flash Attention on. Peak whole-GPU use includes the desktop: 6805 MiB PTQ1_0 and 7949 MiB PQ2_0, with 173 MiB idle. These are sampled peaks.

## Prefill throughput

Median tokens/s over seven repetitions:

| Prompt tokens | PTQ1_0 | PQ2_0 |
|---:|---:|---:|
| 128 | 1292.3 | 1296.8 |
| 512 | 1378.1 | 1364.9 |
| 2048 | 1355.3 | 1351.6 |
| 4096 | 1331.3 | 1326.9 |

## Batch-1 decode throughput

Median tokens/s over seven repetitions:

| Existing context | PTQ1_0 | PQ2_0 | PTQ1_0 lead |
|---:|---:|---:|---:|
| 128 | 46.88 | 35.53 | 31.9% |
| 512 | 46.09 | 31.11 | 48.2% |
| 2048 | 42.82 | 29.26 | 46.3% |
| 4096 | 39.21 | 25.53 | 53.6% |

The first one or two timed repetitions were faster than the remaining repetitions in several decode rows. The seven-sample median tracks the stable middle of each row without selecting the fastest run. Raw distributions and standard deviations are retained for later comparisons. The 60°C gate applies only once per format, not before each mode; later modes inherit GPU heating from earlier ones. Decode-first screenings therefore cannot be compared with this prefill-first matrix, and a follow-up paired experiment should isolate each mode/context and record the thermal state for each run.

## Combined prompt and generation throughput

Median total model tokens/s for each prompt length plus 128 generated tokens:

| Prompt tokens | PTQ1_0 | PQ2_0 |
|---:|---:|---:|
| 128 | 85.54 | 59.37 |
| 512 | 236.08 | 128.71 |
| 2048 | 436.15 | 276.34 |
| 4096 | 426.16 | 346.71 |

These are reference results, not optimization gains. The separate first pass in [results/baseline_initial_uncontrolled.json](results/baseline_initial_uncontrolled.json) began formats at different GPU temperatures and is kept only as a diagnostic; use `results/baseline.json` for apples-to-apples comparisons.

## Correctness

- Upstream quantization, packed-layout, and row-shape suite: 4/4 tests passed.
- CUDA-vs-CPU `MUL_MAT` reference checks on PTQ1_0 and PQ2_0, with odd row tails and K sizes 1024, 5120, 6144, and 17408: 96/96 passed within the upstream `5e-4` NMSE bound.
- Actual CUDA model smoke inference: both GGUF files loaded and produced non-empty 32-token completions using greedy decoding. See [results/baseline_smoke.json](results/baseline_smoke.json).
- Re-run using `tests/run_correctness.sh`.

## Matched isolated decode comparison after optimization

The original matrix ran prefill before decode and warmed the GPU; its decode figures above are reference data and should not be used as a speedup denominator. Experiment 010 compared the rebuilt ROWS=1 source-default implementation against an archived ROWS=4 baseline build using isolated batch-1 decode, seven repetitions, 128 generated tokens, and a start gate of at most 60°C. The measured medians were:

| Existing context | ROWS=1 PTQ1_0 | ROWS=4 PTQ1_0 control | Median delta |
|---:|---:|---:|---:|
| 512 | 82.22 tok/s | 78.00 tok/s | +5.42% |
| 4096 | 79.70 tok/s | 75.66 tok/s | +5.34% |

Both builds peaked at 6,805 MiB. The manager's final pair started at 51°C for ROWS=1 and 59°C for ROWS=4; two earlier reversed-order pairs starting at 58–60°C also favored ROWS=1. Full samples, per-run telemetry, and correctness evidence are in [experiment 010](experiments/010-ptq1-planar-rows/REPORT.md).
