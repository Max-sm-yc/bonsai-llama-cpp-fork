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
- Experiments 001/002 disabled PTQ1_0's L2 prefetch; the paired 128-token workloads tied within 0.04%, and long-run tail effects reversed with process order and broad clock variation. Experiment 003 changed the one-column GEMV CTA warps from four to two/eight: deltas were +0.001%/-0.047% for two warps and -0.025%/+0.026% for eight at contexts 512/4096. No robust gain; prefetch and four-warp geometry restored. See reports 001–003.
- Experiment 004 tested a constant-memory lookup table for PTQ1_0 `qs` trit expansion. It matched the multiply decoder but took 5.89x longer in a focused 120-trit dot kernel; no production change or E2E run. The test excluded `qh` and production activation layout, so it rejects this direct LUT design only.
- Nsight Compute counters are blocked by `ERR_NVGPUCTRPERM`; do not change system-wide driver permissions. Nsight Systems and static cubin resource reports are available.

## Important discoveries

- Under matched conditions, PTQ1_0 decode is faster than PQ2_0 on this RTX 3080 by 32–54%, while prefill is nearly tied. This differs from the model card's broad Ampere result, which has no RTX 3080 row.
- PTQ1_0's three dominant specialized GEMV variants take about 140 ms less total in the Nsight trace than PQ2_0's corresponding variants. Two variants are 13–16% faster per launch; one is 2.5% slower. The 17.6% lower PTQ1_0 payload is consistent with a weight-traffic advantage, but NCU bandwidth/instruction counters are unavailable.
- The hot PTQ1_0 GEMV variants compile to 106–126 registers/thread with no local spills. PTQ1_0 also fuses Hadamard and Q8_1 quantization; PQ2_0 uses separate kernels.
- The baseline matrix temperature gate runs once per format, not once per workload. Isolated decode processes at <=62 C start measured about 76–78 tok/s, unlike the hotter baseline matrix. In 512-token, 4096-context runs, process tails varied strongly with pair order and SM clocks; compare matched isolated runs and capture per-process telemetry.
- The default four-warp batch-1 PTQ1_0 GEMV CTA launch is unchanged by two- or eight-warp alternatives on the tested 512/4096 decode contexts; warp count alone is not a useful tuning lever.
- Both files fit at 4096 context with F16 KV. Peak whole-GPU memory is 6805 MiB PTQ1_0 and 7949 MiB PQ2_0.

## Next candidates

1. Test an exact 2-bit side representation for PTQ1_0 on sm_86: measure whether simpler per-weight decode offsets its larger weight traffic. Start with the full block dot, include `qh` and production activation layout, account for the added VRAM, then integrate only if decode improves within 10 GiB.
2. If reformatting loses, test a bit-sliced decoder directly on the original base-3 layout; the direct constant-memory LUT was 5.89x slower.
2. Measure whether broader Hadamard/Q8_1 fusion benefits PQ2_0; keep secondary to the faster PTQ1_0 decode path.
