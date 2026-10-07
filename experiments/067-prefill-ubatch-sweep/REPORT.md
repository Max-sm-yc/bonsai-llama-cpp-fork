# Experiment 067: global prefill ubatch sweep

## HYPOTHESIS

A larger global `ubatch_size` may improve prompt processing on the RTX 3080 while keeping logical batch size `-b 2048`, model behavior, peak VRAM below 10 GiB, and batch-1 decode behavior intact. The same ubatch must work across prompts 128, 512, 2,048, and 4,096; a context-specific dispatch is out of scope.

## IMPLEMENTATION

This is a runtime-configuration screen only. No source or arithmetic changed. The detached worktree is based on manager documentation HEAD `2059b5e81aa67e4c4de4628b310aaeb819e1535e`; current production code remains the Exp062 best at `ffb0ef37690b902829ea1158b02b14517ed93c2b`.

The same PTQ1_0 GGUF was used in all five three-repetition screens (`ub=128, 256, 512, 1024, 2048`). Common options were `-ngl 99 -fa on -b 2048 -ctk f16 -ctv f16 -t 8`, default llama-bench warmups, contexts 128/512/2048/4096, and a <=60 C / <=5% utilization start gate. Each screen used the established `benchmark/run.py` harness. These are screening runs; there was no seven-repetition confirmation because no candidate was globally consistent.

The `llama-bench` executable and libraries came from the clean Exp066 baseline build, whose source is the same production baseline. Its embedded RUNPATH points at the Exp066 build directory, so benchmark processes explicitly set `LD_LIBRARY_PATH=$PWD/build/bin`. `results/exp067/llama-bench-ldd.txt` records that every llama/GGML library resolved to this worktree under that environment; CUDA/system libraries resolve from `/usr/local/cuda` and the OS. No source or runtime file was changed to enable this. The benchmark JSON records each command, raw sample, timing summary, and sampled GPU telemetry.

## RESULT

Larger ubatches win only at the two long prompts while slowing both short prompts. The best-looking long-prompt setting, `ub=2048`, was +1.85% at 2,048 tokens and +1.79% at 4,096, but -1.33% at 128 and -1.49% at 512 against the same-session `ub=512` control. `ub=1024` was within +0.2% at the long prompts but slower at 128/512. `ub=128` and `ub=256` regressed at every context except a noise-level short-prompt comparison. No single global setting is a consistent improvement.

All measured peaks stayed below 10 GiB. `ub=2048` used 8,363 MiB whole-GPU peak, leaving about 1,877 MiB below the 10,240 MiB device limit.

## CORRECTNESS

The canonical fixed-seed smoke ran on the same PTQ1_0 model, prompt, seed, context, logical batch, and other runtime settings with `ub=512` and `ub=2048`. Both produced the same non-empty 32-token completion text; the only difference in the captured CLI output was the prompt throughput timing line. The captured outputs and stderr are in `results/exp067/canonical-smoke-ub*.{stdout.txt,stderr.log}`. This is a deterministic smoke comparison, not a numerical-logit test.

An initial interactive CLI attempt was stopped and excluded after it returned at stdin EOF before the bounded completion finished. The final one-shot commands used `--single-turn`, fixed `-n 32`, and the project smoke flags; only the canonical-smoke files above support the correctness statement.

No batch-1 decode A/B or PQ2_0 comparison was run: those checks are conditional on having a viable global setting, and no setting passed the global screen. The retained decode configuration remains `ub=512`.

## MICROBENCHMARK

Three samples per context and setting are shown below. “Mean ± SD” and range use llama-bench's three tok/s samples; latency is its mean operation latency. Percentage deltas compare medians against the same-session `ub=512` screen. Peak VRAM is the maximum sampled whole-GPU usage for the setting.

| ubatch | Prompt | Median tok/s | Mean ± SD tok/s | Range tok/s | Mean latency | Peak VRAM | Δ median vs ub512 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1,303.46 | 1,256.48 ± 83.91 | 1,159.61–1,306.39 | 102.19 ms | 6,403 MiB | -0.56% |
| 128 | 512 | 1,285.54 | 1,282.11 ± 6.74 | 1,274.35–1,286.44 | 399.35 ms | 6,403 MiB | -7.90% |
| 128 | 2,048 | 1,263.59 | 1,262.20 ± 2.60 | 1,259.20–1,263.81 | 1,622.57 ms | 6,403 MiB | -8.61% |
| 128 | 4,096 | 1,243.08 | 1,243.11 ± 1.11 | 1,242.01–1,244.23 | 3,294.97 ms | 6,403 MiB | -9.04% |
| 256 | 128 | 1,297.26 | 1,251.70 ± 79.87 | 1,159.48–1,298.36 | 102.55 ms | 6,531 MiB | -1.03% |
| 256 | 512 | 1,326.45 | 1,326.88 ± 6.06 | 1,321.05–1,333.15 | 385.87 ms | 6,531 MiB | -4.97% |
| 256 | 2,048 | 1,323.45 | 1,323.01 ± 2.27 | 1,320.55–1,325.03 | 1,547.99 ms | 6,531 MiB | -4.28% |
| 256 | 4,096 | 1,304.39 | 1,304.08 ± 1.13 | 1,302.82–1,305.01 | 3,140.93 ms | 6,531 MiB | -4.55% |
| 512 | 128 | 1,310.81 | 1,265.67 ± 82.01 | 1,171.00–1,315.19 | 101.43 ms | 6,793 MiB | 0.00% |
| 512 | 512 | 1,395.87 | 1,389.22 ± 16.09 | 1,370.87–1,400.92 | 368.59 ms | 6,793 MiB | 0.00% |
| 512 | 2,048 | 1,382.69 | 1,383.08 ± 0.70 | 1,382.66–1,383.88 | 1,480.75 ms | 6,793 MiB | 0.00% |
| 512 | 4,096 | 1,366.58 | 1,366.51 ± 1.04 | 1,365.44–1,367.51 | 2,997.42 ms | 6,793 MiB | 0.00% |
| 1,024 | 128 | 1,297.16 | 1,249.37 ± 85.08 | 1,151.15–1,299.81 | 102.78 ms | 7,313 MiB | -1.04% |
| 1,024 | 512 | 1,371.34 | 1,368.58 ± 15.93 | 1,351.45–1,382.94 | 374.15 ms | 7,313 MiB | -1.76% |
| 1,024 | 2,048 | 1,385.00 | 1,386.08 ± 2.53 | 1,384.27–1,388.97 | 1,477.55 ms | 7,313 MiB | +0.17% |
| 1,024 | 4,096 | 1,368.26 | 1,368.47 ± 1.27 | 1,367.32–1,369.83 | 2,993.12 ms | 7,313 MiB | +0.12% |
| 2,048 | 128 | 1,293.36 | 1,246.58 ± 86.99 | 1,146.21–1,300.18 | 103.03 ms | 8,363 MiB | -1.33% |
| 2,048 | 512 | 1,375.02 | 1,375.41 ± 13.61 | 1,362.00–1,389.22 | 372.28 ms | 8,363 MiB | -1.49% |
| 2,048 | 2,048 | 1,408.30 | 1,407.20 ± 2.22 | 1,404.64–1,408.66 | 1,455.38 ms | 8,363 MiB | +1.85% |
| 2,048 | 4,096 | 1,391.06 | 1,390.81 ± 1.39 | 1,389.31–1,392.06 | 2,945.04 ms | 8,363 MiB | +1.79% |

The 128-token row had one slow sample in every arm, increasing the three-run mean and SD. Medians and full ranges are retained; these short screens are not treated as confirmation data.

## END-TO-END IMPACT

No ubatch change is retained, so production prefill and primary decode configuration remain unchanged. `ub=2048` is not adopted for an end-to-end claim: its prefill gain is limited to longer prompt shapes and it harms shorter ones. No instrumented profile or speedup claim is involved.

## ANALYSIS

The observed tradeoff is consistent with more prompt work per physical batch: increasing ubatch improves the two long prompts modestly and raises scratch-memory use, but does not preserve the short-prompt timings. `ub=2048` satisfies the measured VRAM bound but fails the global no-material-harm criterion. `ub=1024` has much more headroom and near-flat long-prompt speed but still gives up 1.0–1.8% on the 128/512 screens. Smaller values reduce memory but lose throughput increasingly as prompt length grows.

The three-sample screen supports selection against these candidates; it is not a 7-run confirmation and cannot establish sub-percent differences reliably. No candidate showed a global advantage large and consistent enough to advance under the prescribed rule.

## DECISION

**REVERT / retain baseline `-ub 512`.** No global runtime parameter change is selected. Logical batch size stays `-b 2048`; code and arithmetic remain unchanged. This is a configuration-screen result, not a code optimization.

## FOLLOW-UPS

- Keep `-ub 512` as the global PTQ1_0 setting on this RTX 3080.
- Reopen only if a workload-level requirement justifies long-prompt-only tuning; such context-specific dispatch was excluded from this experiment.

## IMPORTANT DISCOVERIES

- `ub=2048` peaked at 8,363 MiB, below the 10 GiB limit, and improved the 2,048/4,096 prompt medians by about 1.8% while reducing the 128/512 medians by about 1.3–1.5%.
- The canonical fixed-seed smoke completion text was identical for `ub=512` and `ub=2048`.
- No candidate satisfied the global consistency gate, so decode and PQ2_0 checks were not triggered and the primary decode setup remains unchanged.
