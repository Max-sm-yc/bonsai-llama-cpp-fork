# Experiment 041: matched frozen-reference versus current-best comparison

## HYPOTHESIS

The ROWS=1 GEMV and coordinated QKV preparation gains should persist against the original reference runtime when both builds use the same benchmark harness, model files, cooldown gate, and reversed run order.

## IMPLEMENTATION

No source was changed. Built baseline project commit `2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c` in `/tmp/bonsai2-reference` and compared it with current production code commit `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`. Both use upstream runtime commit `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`, Release, CUDA on, `CMAKE_CUDA_ARCHITECTURES=86`, CUDA graphs on, and identical CMake feature settings. `ldd` and RUNPATH were checked; each executable loaded its own build's CUDA and ggml libraries. Build hashes and the full run order are in `results/reference_ab/README.md` and `summary.json`.

Each mode used two order-reversed pairs: reference→current, then current→reference. Each arm/run had seven llama-bench repetitions with default warmups and a fresh start gate at ≤60°C and ≤5% GPU utilization. Decode used contexts 512/4096 and 128 generated tokens; prefill used 128/512/2048/4096; combined used prompts 512/4096 plus 128 generated tokens. Batch 2048, microbatch 512, F16 KV, Flash Attention, 99 GPU layers, and 8 CPU threads were fixed. The benchmark excludes tokenization and sampling.

## RESULT

The current best is faster than the frozen reference on batch-1 decode and combined workloads. The median is the median of two seven-repetition run medians; all fourteen samples per arm/context are preserved.

| Workload | Reference | Current best | Change |
|---|---:|---:|---:|
| Decode, context 512 | 78.106 tok/s | 83.433 tok/s | +6.82% |
| Decode, context 4096 | 75.533 tok/s | 79.882 tok/s | +5.76% |
| Combined, prompt 512 + 128 decode | 314.372 tok/s | 332.196 tok/s | +5.67% |
| Combined, prompt 4096 + 128 decode | 860.947 tok/s | 877.786 tok/s | +1.96% |
| Prefill, 128 tokens | 1292.995 tok/s | 1291.875 tok/s | -0.09% |
| Prefill, 512 tokens | 1377.400 tok/s | 1377.395 tok/s | -0.00% |
| Prefill, 2048 tokens | 1355.280 tok/s | 1355.505 tok/s | +0.02% |
| Prefill, 4096 tokens | 1332.110 tok/s | 1332.260 tok/s | +0.01% |

Peak whole-GPU memory was 6,805 MiB for the reference and 6,803 MiB for the current build. Model loading completed successfully for every run.

## CORRECTNESS

The project baseline had already passed the quantization and packed-layout tests, 96/96 CUDA-vs-CPU PTQ1_0/PQ2_0 matmul cases, and fixed-seed model smokes. The current best separately passed `tests/run_correctness.sh` with selected CTests 5/5, backend cases 96/96, and both model smokes. This measurement made no code change. `llama-bench` completed every requested model/workload row for both builds.

## MICROBENCHMARK

Not run. This is a model-level comparison; the current active PTQ1_0 GEMV remains the measured profiling bottleneck.

## END-TO-END IMPACT

The matched decode improvement is +6.82% at context 512 and +5.76% at 4096. Prefill is effectively unchanged. Combined improvement is larger at 512 (+5.67%) than at 4096 (+1.96%) because the long prompt dominates the latter workload.

Context-4096 decode contained slow-tail samples in both builds: reference range 70.07–76.17 tok/s; current range 64.27–80.99. The run-median comparison is stable in direction, but retain the tails in future comparisons.

## ANALYSIS

The controlled direct reference comparison verifies that the cumulative decode gain is real and avoids using the thermally loaded initial matrix as a denominator. The +5.76–6.82% total improvement is consistent with, but is not calculated by multiplying, the separately measured Exp010 and Exp036 gains. Prefill results show that those changes do not materially affect prompt throughput.

## DECISION

**KEEP current best.** No code change was made. Use this run as the matched frozen-reference denominator in final project reporting.

## FOLLOW-UPS

Continue optimizing the active batch-1 PTQ1_0 GEMV. The next untested geometry dimension is CTA thread count with row-tile tuning; previous row-per-item, row-tile, and warp-split experiments kept the production 128-thread CTA.

## IMPORTANT DISCOVERIES

- Direct reference-to-current decode is +6.82% at context 512 and +5.76% at 4096 under matched, reversed-order runs.
- Prefill medians match within 0.09% across all four tested lengths.
- Combined throughput improves +5.67% at prompt 512 and +1.96% at prompt 4096.
- Current peak memory is 2 MiB below the frozen reference build; both remain well within RTX 3080 VRAM.
