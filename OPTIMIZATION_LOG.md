# Optimization log

## Baseline establishment

- Built the unchanged PrismML runtime for sm_86, verified both model files and smoke runs, and passed upstream numerical tests plus 96 CUDA-vs-CPU ternary matmul cases.
- Measured both formats with seven repetitions at contexts 128, 512, 2048, and 4096 under a matched 60°C idle start gate. PTQ1_0 is the faster decode baseline; see `BASELINE.md`.
- Nsight Systems ranks PTQ1_0 GEMV as the first optimization target (61.8% of traced GPU kernel time). Nsight Compute counters are unavailable due `ERR_NVGPUCTRPERM`; no system setting was changed.
- No optimization has been accepted yet. The first bounded experiment should investigate the PTQ1_0 sm_86 batch-1 GEMV implementation, then re-profile if it yields an end-to-end gain.
