# Optimization log

## Baseline establishment

- Built the unchanged PrismML runtime for sm_86, verified both model files and smoke runs, and passed upstream numerical tests plus 96 CUDA-vs-CPU ternary matmul cases.
- Measured both formats with seven repetitions at contexts 128, 512, 2048, and 4096 under a matched 60°C idle start gate. PTQ1_0 is the faster decode baseline; see `BASELINE.md`.
- Nsight Systems ranks PTQ1_0 GEMV as the first optimization target (61.8% of traced GPU kernel time). Nsight Compute counters are unavailable due `ERR_NVGPUCTRPERM`; no system setting was changed.
- No optimization has been accepted yet. PTQ1_0 GEMV remains the highest-value target. Paired follow-up in experiment 002 found no L2-prefetch gain, experiment 003 found no warp-count gain, and experiment 004 rejected a direct constant-memory decoder LUT; next compare an exact 2-bit side representation's simpler decode with its extra weight traffic.

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

## Experiment 003: PTQ1_0 batch-1 GEMV warp geometry

- Compared the four-warp baseline with two and eight warps on the actual RTX 3080. The two-warp screen changed medians by +0.001% at context 512 and -0.047% at 4096; five paired eight-warp runs changed them by -0.025% and +0.026%. The 8-warp paired sign varied by context and pair, within overlapping sample ranges.
- Both candidates passed CUDA-vs-CPU coverage (96/96 cases) and PTQ1_0/PQ2_0 CUDA smoke inference. No end-to-end gain was demonstrated; restored the four-warp source and baseline build. See `experiments/003-ptq1-gemv-geometry/REPORT.md` and `results/exp003/`.

## Experiment 004: PTQ1_0 constant-memory trit LUT

- Compared the production-style repeated multiply/byte-permute decoder with an exact 256-by-5 constant-memory LUT in a CUDA-event microbenchmark of 120 `qs` trits and DP4A dot/bias correction. They matched 65,536 generated block-dot outputs, but the LUT took 0.130949 ms/launch versus 0.022250 ms for multiply (5.89x slower).
- Rejected without production integration or model benchmarking. The focused harness excluded `qh` and the exact warp-transposed activation layout; it rejects the direct LUT arrangement, not all alternate trit decoders. See `experiments/004-ptq1-trit-decoder/REPORT.md`.

## Experiment 005: PTQ1_0 2-bit side representation

- The prototype did not implement packed conversion: it wrote one byte per trit into a 32-byte field intended for 2-bit codes, overrunning side records, and its base-3 accessor also used the wrong element order. Its 65,531/65,536 dot mismatches invalidate the timing, so the experiment gives no performance conclusion.
- No production source, binaries, or model files changed; no end-to-end measurement was made. The next screen must implement bounded 2-bit packing, match all decoded elements against the CPU dequantizer, and only then time the complete production block dot. See `experiments/005-ptq1-2bit-side/REPORT.md`.

## Experiment 006: corrected PTQ1_0 2-bit packing screen

- Fixed packed storage and the canonical element map; host/GPU checks covered 8,388,608 codes and exact synthetic full-block dots, and Compute Sanitizer found no device memory errors.
- Manager review found activation indexing still differed from production SOA_ISUM: the screen derived `sub` from the block index rather than the weight element. The reported 18.4–18.8% slowdown is inconclusive and must not guide integration. No production source or model changed and no E2E run occurred. See `experiments/006-ptq1-2bit-corrected/REPORT.md`.

## Experiment 007: PTQ1_0 2-bit side codes with production SOA_ISUM dots

- Corrected the activation map and used the same full-block DP4A reduction for base-3 and packed codes. Every packed code and output matched; Compute Sanitizer found no errors.
- Across three 65,536-block invocations, the 2-bit path was 3.44–3.50% slower; it was 5.02% slower at 16,384 blocks. With a 21.43% larger payload and no kernel win, rejected runtime integration without E2E testing. See `experiments/007-ptq1-side-production-dot/REPORT.md`.

## Experiment 008: PTQ1_0 parallel floor-difference decoder

- The identity `d[k]=floor(x*3^(k+1)/256)-3*floor(x*3^k/256)` passed exhaustive host checks for all 256 bytes and five digits.
- Its first CUDA block implementation failed correctness in `qh`, where two encoded byte streams must be interleaved into each four-element DP4A word. No timing or runtime change was made. The bounded follow-up is a device exhaustive check and corrected `qh` packing before timing. See `experiments/008-ptq1-parallel-trit-decode/REPORT.md`.
