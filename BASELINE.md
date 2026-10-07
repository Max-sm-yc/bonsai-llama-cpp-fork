# Reference baseline

Status: these reference measurements were verified on the RTX 3080 using unmodified PrismML runtime commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`. The current optimized production-code commit is `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`; the newer commits add research records only.

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

The first one or two timed repetitions were faster than the remaining repetitions in several decode rows. The seven-sample median tracks the stable middle of each row without selecting the fastest run. Raw distributions and standard deviations are retained. The 60°C gate applies only once per format, not before each mode; later modes inherit GPU heating from earlier ones. Experiment 041 adds an isolated, reversed-order reference/current comparison under a fresh start gate before every arm.

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

## Matched frozen-reference versus current-best comparison (experiment 041)

The baseline project commit `2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c` was freshly built alongside current production code `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`, both on runtime commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`. Both used Release, CUDA enabled for sm_86, CUDA graphs, and matching CMake settings. Each binary's RUNPATH and `ldd` were checked; each loaded libraries from its own build. Binary/library hashes are in `results/reference_ab/build_info.txt`.

Every mode used two reversed-order pairs, seven repetitions per run, and a fresh ≤60°C / ≤5% GPU-utilization gate before each arm. Decode used 128 generated tokens at contexts 512/4096; prefill used 128/512/2048/4096; combined used prompts 512/4096 plus 128 generated tokens. Batch 2048, microbatch 512, F16 KV, 99 GPU layers, Flash Attention, and 8 CPU threads were fixed. All raw samples and telemetry are in `results/reference_ab/`.

| Workload | Frozen reference | Current best | Change |
|---|---:|---:|---:|
| Decode, context 512 | 78.106 tok/s | 83.433 tok/s | +6.82% |
| Decode, context 4096 | 75.533 tok/s | 79.882 tok/s | +5.76% |
| Combined, prompt 512 + 128 generated | 314.372 tok/s | 332.196 tok/s | +5.67% |
| Combined, prompt 4096 + 128 generated | 860.947 tok/s | 877.786 tok/s | +1.96% |
| Prefill, 128 tokens | 1292.995 tok/s | 1291.875 tok/s | -0.09% |
| Prefill, 512 tokens | 1377.400 tok/s | 1377.395 tok/s | -0.00% |
| Prefill, 2048 tokens | 1355.280 tok/s | 1355.505 tok/s | +0.02% |
| Prefill, 4096 tokens | 1332.110 tok/s | 1332.260 tok/s | +0.01% |

Decode and combined peaks were 6,805 MiB for the reference and 6,803 MiB current. These direct results, not the thermally loaded original baseline matrix above, are the cumulative current-versus-reference comparison. See [experiment 041](experiments/041-reference-current-ab/REPORT.md) and `results/reference_ab/summary.json`.
