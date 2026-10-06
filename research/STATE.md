# Research state

## Current best

- Project baseline commit `2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c` (unchanged PrismML source `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`); PTQ1_0. Median decode: 46.88/46.09/42.82/39.21 tok/s at contexts 128/512/2048/4096. Prefill: 1292/1378/1355/1331 tok/s. Peak whole-GPU use 6805 MiB. See `results/baseline.json`; 7 repetitions, 60°C format start gate, then prefill/decode/combined in sequence.
- Correctness on the restored baseline, rerun after experiment 001: 4/4 upstream tests, 96/96 CUDA-vs-CPU ternary matmul cases, and both CUDA model smoke runs pass.

## Bottlenecks

1. Ternary batch-1 GEMV kernels: 61.8% of PTQ1_0 and 62.6% of PQ2_0 GPU kernel time in the context-512 Nsight Systems trace.
2. Ternary GEMM: 12.1% PTQ1_0, 11.1% PQ2_0.
3. Recurrent gated-delta attention, Hadamard/Q8_1 preparation, and RMSNorm: about 4% each.

## Successful optimizations

- None yet. Upstream reference paths are the baseline.

## Failed or exhausted approaches

- The first format comparison started PTQ1_0 cool and PQ2_0 hot; it is retained as `results/baseline_initial_uncontrolled.json` but excluded from decisions.
- Experiment 001 disabled PTQ1_0's L2 prefetch. The apparent decode-first speedup vanished under the baseline mode order; full-run medians trended higher but sample spreads were broad and there was no paired same-build control. Source was reverted; performance effect remains inconclusive. See `experiments/001-ptq1-sm86-gemv/REPORT.md`.
- Nsight Compute counters are blocked by `ERR_NVGPUCTRPERM`; do not change system-wide driver permissions. Nsight Systems and static cubin resource reports are available.

## Important discoveries

- Under matched conditions, PTQ1_0 decode is faster than PQ2_0 on this RTX 3080 by 32–54%, while prefill is nearly tied. This differs from the model card's broad Ampere result, which has no RTX 3080 row.
- PTQ1_0's three dominant specialized GEMV variants take about 140 ms less total in the Nsight trace than PQ2_0's corresponding variants. Two variants are 13–16% faster per launch; one is 2.5% slower. The 17.6% lower PTQ1_0 payload is consistent with a weight-traffic advantage, but NCU bandwidth/instruction counters are unavailable.
- The hot PTQ1_0 GEMV variants compile to 106–126 registers/thread with no local spills. PTQ1_0 also fuses Hadamard and Q8_1 quantization; PQ2_0 uses separate kernels.
- The benchmark's temperature gate runs once per format, not once per workload. Its initial timed decode samples can be around 73–77 tok/s, then settle in the mid-30s to 40s after the GPU heats. Do not compare results across different mode orders; isolate workload/context and capture thermal/clock state for kernel decisions.
- Both files fit at 4096 context with F16 KV. Peak whole-GPU memory is 6805 MiB PTQ1_0 and 7949 MiB PQ2_0.

## Next candidates

1. Run a paired same-build PTQ1_0 prefetch on/off experiment with isolated 512/4096 decode and combined runs, alternating order and matching idle temperature; this resolves experiment 001 and establishes measurement repeatability.
2. Tune the sm_86 PTQ1_0 decode GEMV geometry/unpack path using the improved paired protocol; require end-to-end gains and numerical tests.
3. Measure whether broader Hadamard/Q8_1 fusion benefits PQ2_0; keep secondary to the faster PTQ1_0 decode path.
