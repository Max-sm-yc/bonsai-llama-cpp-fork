# Optimization log

## Baseline establishment

- Built the unchanged PrismML runtime for sm_86, verified both model files and smoke runs, and passed upstream numerical tests plus 96 CUDA-vs-CPU ternary matmul cases.
- Measured both formats with seven repetitions at contexts 128, 512, 2048, and 4096 under a matched 60°C idle start gate. PTQ1_0 is the faster decode baseline; see `BASELINE.md`.
- Nsight Systems ranks PTQ1_0 GEMV as the first optimization target (61.8% of traced GPU kernel time). Nsight Compute counters are unavailable due `ERR_NVGPUCTRPERM`; no system setting was changed.
- No optimization has been accepted yet. PTQ1_0 GEMV remains the highest-value target; use paired isolated measurements to resolve experiment 001 before testing alternative work partitioning or unpack schedules.

## Experiment 001: PTQ1_0 L2 prefetch

- Disabling the GEMV's explicit next-block L2 prefetch passed correctness but did not establish a robust speedup. The decode-first three-repetition screen started at 54°C and overstated speed relative to the reference matrix, which runs prefill first. The seven-repetition full matrix's medians were modestly higher at several points, but sample spreads were broad and no same-build prefetch-on control was collected.
- Reverted the source change. The full candidate matrix and report are preserved in `results/exp001_no_prefetch_full.json` and `experiments/001-ptq1-sm86-gemv/REPORT.md`; the effect is inconclusive.
- After restoring the baseline source and rebuilding, manager reran `tests/run_correctness.sh`: 4/4 CTests, 96/96 CUDA-vs-CPU ternary matmul cases, and both actual-model smoke runs passed. Ninja again warned of a truncated log and recovered by rebuilding broadly.
- Follow-up: paired same-build A/B runs isolated by workload/context, with alternating order and start temperature/clock telemetry, before tuning the GEMV further.

## Experiment 002: paired PTQ1_0 L2 prefetch A/B

- Built prefetch-on/off variants from the same PrismML source revision and measured isolated decode and combined workloads with alternating process order, a per-process temperature/utilization gate, and GPU telemetry. At 128 generated tokens, the four workloads were tied within 0.04% median.
- A longer 512-token, context-4096 follow-up initially showed a faster third sample with prefetch-on. In two seven-repetition reversed-order pairs, the later-sample winner switched with process order; SM clock samples ranged from 270 to 1980 MHz. No robust prefetch effect was demonstrated.
- Kept the baseline prefetch-on source. Candidate correctness passed: 4/4 upstream tests, 96/96 CUDA-vs-CPU ternary matmul cases, and both CUDA model smoke runs. The paired runner and raw results are preserved in `benchmark/prefetch_ab.py` and `results/exp002/`.
- Follow-up: pursue a different PTQ1_0 GEMV work-partition/unpack hypothesis; use paired runs and retain results only when the end-to-end gain repeats across process orders and warmed samples.
