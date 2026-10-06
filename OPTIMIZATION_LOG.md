# Optimization log

## Baseline establishment

- Built the unchanged PrismML runtime for sm_86, verified both model files and smoke runs, and passed upstream numerical tests plus 96 CUDA-vs-CPU ternary matmul cases.
- Measured both formats with seven repetitions at contexts 128, 512, 2048, and 4096 under a matched 60°C idle start gate. PTQ1_0 is the faster decode baseline; see `BASELINE.md`.
- Nsight Systems ranks PTQ1_0 GEMV as the first optimization target (61.8% of traced GPU kernel time). Nsight Compute counters are unavailable due `ERR_NVGPUCTRPERM`; no system setting was changed.
- Experiments 001–009 did not produce a retained end-to-end optimization. Experiment 010 is the first verified speedup; its active-path row schedule and measured effect are recorded below.

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

## Experiment 003: generic PTQ1_0 GEMV warp-count override (no-op on sm_86)

- The decode values tied at contexts 512 and 4096, and the candidate passed correctness, but a later source dispatch audit found that sm_86 batch-1 PTQ1_0 uses the dedicated `mul_mat_vec_ptq1_0_pt` kernel. The edited generic `calc_nwarps` path was bypassed, so all compared binaries ran the same active kernel.
- Keep the no-op comparison as a dispatch control; do not treat it as evidence on active GEMV geometry. The active planar-transposed PT kernel still needs tuning. See the manager audit in `experiments/003-ptq1-gemv-geometry/REPORT.md`.

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
- Manager source audit later established the harness uses SOA_ISUM, while the target RTX 3080 selects planar PT activations and the dedicated PTQ1 kernel. Its measured slowdown rejects the SOA dot only; it does not decide the side representation in the active sm_86 path.

## Experiment 008: PTQ1_0 parallel floor-difference decoder

- The identity `d[k]=floor(x*3^(k+1)/256)-3*floor(x*3^k/256)` passed exhaustive host checks for all 256 bytes and five digits.
- Its first CUDA block implementation failed correctness in `qh`, where two encoded byte streams must be interleaved into each four-element DP4A word. No timing or runtime change was made. The bounded follow-up is a device exhaustive check and corrected `qh` packing before timing. See `experiments/008-ptq1-parallel-trit-decode/REPORT.md`.

## Experiment 009: PTQ1_0 floor decoder with corrected qh interleave

- Device exhaustive checks covered every source byte and digit and every pair of `qh` bytes; the full production-layout block harness matched its recurrence and independent host reference exactly, and Compute Sanitizer found no errors.
- The corrected decoder was slower than the production recurrence in all three paired RTX 3080 screens: 7.04% at 1,024 blocks, 4.11% at 16,384, and 1.24% at 65,536. Rejected without runtime integration or E2E testing. Keep the production decoder. See `experiments/009-ptq1-floor-decoder-qh/REPORT.md`.

## Experiment 010: active PTQ1_0 planar GEMV row scheduling

- Changed the one-column work item from four rows to one in the dedicated sm_86 `mul_mat_vec_ptq1_0_pt` kernel. The specialization dropped from 108 to 76 registers/thread with no spills; the final source-default build passed correctness.
- Corrected isolated A/Bs show ROWS=1 faster than ROWS=4 by 4.9–5.8% in paired medians, and faster than ROWS=2 by 0.65–0.73% in both direct pair orders. The manager's fresh rebuilt A/B measured +5.42% at context 512 and +5.34% at context 4096. ROWS=8 regressed 16–17%.
- Some context-4096 repetitions have severe low-throughput tails in both row1 and baseline binaries. Preserve medians and means/ranges together; there is not yet a causal diagnosis. The first archived-binary timing set is invalid because of absolute RUNPATH leakage and is explicitly excluded in the report and `results/exp010/README.md`.
- Rebuilt and rechecked the final source-default library independently. A fresh matched seven-repetition pair measured +5.4%/+5.3% medians at contexts 512/4096; all sample ranges were tight. Four CTests, 96 CUDA-vs-CPU cases, and both model smokes passed.
- Re-profiled with Nsight Systems: the three PTQ1_0 GEMV variants fell from 1.253 s to 1.166 s combined in the mixed setup/decode trace and remain 60.4% of GPU kernel time. Retained ROWS=1; investigate a 2-bit code specialized to the actual planar path, because prior 2-bit screens were SOA-only.

## Experiment 011: exact 2-bit side codes on the active planar path

- Repacked canonical PTQ1_0 blocks into 34-byte 2-bit side blocks and compared scalar extraction and packed-byte DP4A expansion against the active planar base-3 dot. All trits, full block outputs, and independent CPU references matched exactly; manager rebuilt the harness and independently rechecked 16,384 blocks.
- Both decoders were slower at 16K and 65K K-blocks with disjoint sample ranges. At 65K, scalar extraction lost 3.02% and packed expansion lost 4.68%; this screen excluded conversion, favoring the side representation. No runtime/model candidate was integrated.
- Reject the 2-bit side representation on this sm_86 planar kernel. The 34-byte blocks add 21.43% weight payload, with no decode benefit to cover that cost. Continue with a different base-3 unpack/reduction mapping. Full data: `experiments/011-ptq1-planar-2bit/REPORT.md`.

## Experiment 012: pairwise radix-3 decode on the active planar path

- Replaced two serial multiply-by-three remainder steps with one packed `x*9`, then split its quotient into two ordered trits. Exact byte/position, repeated-remainder, both qs stream mapping, exhaustive qh interleave, and full-block CPU-reference gates all passed after fixing a redundant activation-word shift. The manager independently rebuilt and checked a 127-block boundary.
- The corrected candidate remains slower in all full-block screens: +24.59% at 128 blocks, +9.82% at 16,384, and +1.97% at 65,536, with disjoint candidate/control ranges. Registers and spills match the base-3 kernel (40/0).
- Reject this pairwise decoder; no production changes or model benchmark were made. The next active-path question is rows-per-CTA/shared-memory reduction geometry, distinct from ROWS=1's per-item mapping. See `experiments/012-ptq1-pairwise-trits/REPORT.md`.
