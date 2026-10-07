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
- Manager dispatch audit (2026-10-06): the edited prefetch lies in generic `mul_mat_vec_q`; RTX 3080 one-column planar PTQ1_0 returns through `mul_mat_vec_ptq1_0_pt` first. The decode comparison is a no-op for the target path; preserve raw timings as protocol diagnostics only.

## Experiment 002: paired PTQ1_0 L2 prefetch A/B

- Built prefetch-on/off variants from the same PrismML source revision and measured isolated decode and combined workloads with alternating process order, a per-process temperature/utilization gate, and GPU telemetry. At 128 generated tokens, the four workloads were tied within 0.04% median.
- A longer 512-token, context-4096 follow-up initially showed a faster third sample with prefetch-on. In two seven-repetition reversed-order pairs, the later-sample winner switched with process order; SM clock samples ranged from 270 to 1980 MHz. No robust prefetch effect was demonstrated.
- Kept the baseline prefetch-on source. Candidate correctness passed: 4/4 upstream tests, 96/96 CUDA-vs-CPU ternary matmul cases, and both CUDA model smoke runs. The paired runner and raw results are preserved in `benchmark/prefetch_ab.py` and `results/exp002/`.
- Follow-up: pursue a different PTQ1_0 GEMV work-partition/unpack hypothesis; use paired runs and retain results only when the end-to-end gain repeats across process orders and warmed samples.
- Manager dispatch audit (2026-10-06): both decode variants use the same dedicated one-column planar kernel, so the paired result does not measure the prefetch edit. The generic-path effect on other shapes remains unmeasured.

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

## Experiment 013: active planar CTA row-tile geometry

- Swept CTA caps 4/8/24/32 and a larger 8192-float target while keeping ROWS=1 and the dot/reduction math fixed. The best screen (cap 8) tied the control in a seven-repetition pair: +0.11% at context 512 and -0.01% at 4096. Cap 32 at the larger target regressed decode materially.
- Matched Nsight Systems traces showed a 0.56% reduction in the three planar variants' aggregate time for cap 8, but no end-to-end benefit. Cap 8 increases CTA counts substantially on major projections; the existing chooser's utilization heuristic remains the best measured tradeoff.
- Reverted the geometry changes. The manager independently verified the production source and CUDA library against the saved control SHA-256 and removed the temporary copied build. Keep the cap-16/4096-float production heuristic. See `experiments/013-planar-cta-rows/REPORT.md`.
- Next: test a genuinely different warp/register reduction strategy that removes per-K-block shared partial traffic, and require exactness plus matched end-to-end decode improvement before retaining it.

## Experiment 014: warp-per-row reduction proposal (not implemented)

- The experimenter confirmed active dispatch and documented a warp-per-row/register-reduction hypothesis, but ended before making a candidate. No code was built, correctness was not run, and no performance conclusion follows.
- Source and active CUDA library hashes matched the current best. Keep the optimization frontier open; the next experiment must implement the candidate and benchmark it rather than repeat analysis only. See `experiments/014-planar-warp-reduction/REPORT.md`.

## Experiment 015: direct warp-per-row reduction

- Implemented an sm_86 compile-time one-column path with four warps per CTA, one output row per warp, local K-block accumulation, and a final warp sum. It removed dynamic shared partials and the CTA barrier; other column counts retained the existing kernel.
- The candidate passed 4/4 selected CTests, 96/96 CUDA-vs-CPU PTQ1_0/PQ2_0 matmul cases, and both 32-token model smokes. Generated text matched the ROWS=1 reference after ignoring the build identifier. The full correctness script's build began recompiling 394 missing outputs and was stopped at 129; its selected tests were then executed directly against the candidate library.
- Two reversed-order seven-repetition decode pairs consistently lost: -3.27/-3.27% at contexts 512/4096 in pair 1 and -3.79/-3.72% in pair 2. Peak VRAM was effectively unchanged (6803 vs 6805 MiB). Reject this mapping; serial per-lane K work and only four output rows per CTA likely cost more than the shared partial/barrier savings.
- Restored and hash-verified the ROWS=1 source/library. Next test should combine K work across two/four warps per output row, reducing per-K shared partial traffic while restoring more K parallelism. See `experiments/015-warp-reduction-impl/REPORT.md`.

## Experiment 016: cooperative multiwarp reduction proposal (not implemented)

- The follow-up agent inspected active dispatch and fusion, then stopped before implementing a candidate. No tests, profiling, or measurements were run; this is not evidence against the idea.
- Keep the proposal open, but begin with a small compileable two-warps-per-row/4-warp-CTA path and iterate from build and timing feedback. See `experiments/016-multiwarp-row-reduce/REPORT.md`.

## Experiment 017: two warps per PTQ1_0 output row

- Implemented a one-column-only 2-warps/row, 4-warps/CTA mapping. Each pair of warps split K-blocks, reduced locally, and wrote two sums per row for the final combine. Candidate compiled for sm_86 and passed 4 CTests, 96/96 CUDA-vs-CPU cases, and fixed 32-token PTQ1_0/PQ2_0 smokes with reference-matching generated text.
- One seven-repetition end-to-end pair lost 2.60% at context 512 and 1.82% at 4096. Ranges were clearly separated at 512; no second pair was justified after the candidate lost at both contexts. The 40-block projections split unevenly (32+8 lanes); larger-K projections did not offset the combine overhead model-wide.
- Reverted. The manager independently checked source and current-library restoration against saved hashes and removed temporary binaries. A shape-aware four-warps/row design for K>64 remains a distinct possible test; do not repeat the fixed one-/two-warp mappings. See `experiments/017-two-warps-per-row/REPORT.md`.

## Experiment 018: shape-gated four-warps-per-row reduction for large K

- Routed one CTA per output row through a four-warp reduction only for one-column PTQ1_0 shapes with more than 64 K blocks; K<=64 retained ROWS=1. At 136 blocks, 128 lanes covered K nearly in parallel and combined just four warp sums. Candidate passed 4 CTests, 96 CUDA-vs-CPU matmul cases, and both fixed model smokes.
- Two reversed-order seven-repetition decode pairs showed no repeatable gain. Context 512 lost 0.32% and 0.69%; context 4096 changed from +0.77% to -0.58%, with long-context tails in both variants. Reject the specialization and restore ROWS=1.
- Manager verified source and library restoration hashes; temporary build copies were removed. This exhausts the current shared-reduction row-layout sweep. Next focus is a different measured path: potentially fuse RMSNorm and FWHT/Q8_1 activation preparation (3.9% + 4.4% in the mixed profile). See `experiments/018-largek-fourwarp-row/REPORT.md`.

## Profiling note: graph launches already used

- The post-ROWS=1 Nsight Systems trace records 127 CUDA graph-launch API calls in its mixed context-512 workload. The runtime already uses CUDA Graphs, so a generic graph-capture optimization is not the next candidate.

## Experiment 019: RMSNorm plus FWHT/Q8_1 fusion audit

- Audited the current CUDA fusion chain and model graph before coding. Hadamard transform and Q8_1 quantization are already one kernel, and standard RMSNorm plus learned-weight multiply are already fused separately.
- On the Qwen3.5 attention path, `attn_norm` feeds `build_layer_attn`, which uses the activation independently for Q, K, and V projection construction. One projection's FWHT/Q8_1 kernel cannot eliminate that shared norm output. A coordinated multi-branch design would need to preserve/reuse the full-row RMS scale and was outside this candidate.
- No code, tests, benchmarks, or binaries changed. Manager verified the graph fanout and the source/library SHA-256 values in the report. This is not a performance result; pursue the existing FWHT/Q8_1 and gated-delta kernels independently.

## Experiment 020: PTQ1_0 FWHT/Q8_1 CTA width on sm_86

- Kept the fused N=1024 transform and PT Q8_1 stores fixed while comparing NT=128 and NT=512 against NT=256. NT=128 regressed 0.9–1.0% at contexts 512/4096. NT=512 initially looked slightly faster, but reversed-order medians were only +0.15% at 512 and +0.06% at 4096, with substantial long-context tails in both arms. Reject both alternatives.
- NT=128, NT=256, and NT=512 each passed 4 selected CTests, 96/96 CUDA-vs-CPU PTQ1_0/PQ2_0 matmul cases, and fixed-seed 32-token model smokes; candidate outputs matched NT=256. Peak GPU memory stayed 6,805 MiB. No separate kernel microbenchmark was available.
- `LD_DEBUG` confirms each A/B process loads the intended copied `libggml-cuda.so.0`. It also records a later plugin-style probe of `build/bin/libggml-cuda.so` with missing `ggml_backend_score/init` entrypoints. This build has `GGML_BACKEND_DL=OFF`; the linked CUDA backend is registered directly under `GGML_USE_CUDA` in `ggml-backend-reg.cpp`, and the benchmark results identify CUDA. The source-default active library was rebuilt separately after restoring the exact source; its binary hash differs from the explicit A/B NT=256 control hash, so no byte-identity claim is made.
- Source is restored at hash `6027c6ab…`; active library hash is `708eceba…`, versus A/B control library hash `23bd3d12…`. No production change is retained. See `experiments/020-ptq1-fwht-q8-ampere/REPORT.md` and `results/exp020/`.

## Experiment 021: active GDN columns per warp on sm_86

- Screened columns-per-warp 1/2/4/8 for the active S_v=128 scalar raw-gate kernel, preserving the remaining GDN mapping and gather/cache behavior. Column 1 gained 0.52%/0.41% over control in the first pair but only 0.10%/0.11% after reversing order. The 2/8 screens were 0.05–0.32% below the later control, but ran before that control and were not reversed-order pairs. Context-4096 samples had slow tails in both arms.
- Nsight Systems confirmed default recurrent-state gather fusion (864 GDN calls and zero GET_ROWS in a 16-token trace); disabling it added 864 GET_ROWS calls. The existing GDN cache-copy matcher was source-audited and left unchanged.
- The restored default passed 4/4 selected CTests, 96/96 CUDA-vs-CPU ternary matmul cases, and PTQ1_0/PQ2_0 model smokes. Fixed-seed completions matched after excluding only timing text. No source or production binary change was retained; see experiments/021-gdn-column-grouping/REPORT.md.

## Experiment 022: fused-weight RMSNorm CTA geometry on sm_86

- Auditing the post-ROWS=1 trace showed the previously listed 3.9% RMSNorm share covered only the 1024-thread signature. Adding 10,400 calls / 24.43 ms from the 256-thread signature gives a family total of 100.61 ms, or 5.21% of summed kernel time.
- Replacing the fused-weight `ncols >= 1024` path's 1024-thread CTA with 256 threads reduced full-model PTQ1_0 decode by 3.32% at context 512 and 3.25% at 4096 (seven repetitions, 128 tokens). Reject the global geometry change and retain the 1024-thread path.
- Candidate PTQ1_0/PQ2_0 fixed-seed 32-token completions matched the saved controls after normalizing only build/timing text. The standalone correctness script's broad rebuild was stopped after the clear performance regression; its selected CTests and CUDA-vs-CPU matmul suite were not completed for this rejected candidate. Source, active library, and baseline smoke hashes were independently checked against the saved controls. See `experiments/022-rmsnorm-sm86/REPORT.md`.

## Experiment 023: warp-cooperative PTQ1_0 block screen

- A four-lane-per-block map assigned one lane to each 32-weight activation sub-block, then used a four-lane shuffle reduction. On a 16,384-block CUDA screen it was bitwise exact, but took 0.013312 ms/launch versus 0.003490 ms for the scalar-reference kernel (3.82x slower).
- Both screen kernels used test-only scalar per-element trit extraction rather than the active packed base-3 recurrence. No production code changed and no candidate E2E or model correctness run exists; mark this screen inconclusive for the production dataflow. Keep ROWS=1 and the planar activation layout.
- A follow-up is justified only for a group-aware mapping that partitions the active packed recurrence itself, with a focused implementation-equivalent screen before full model integration. See `experiments/023-ptq1-warp-cooperative/REPORT.md`.
- The manager independently rebuilt and reran the scalar screen at 53°C/0% utilization: the 16,384-block output was still bitwise exact, and the medians were 0.003511 vs 0.013312 ms (3.79x slower). This confirms the prototype regression, not the production recurrence hypothesis.

## Experiment 024: warp-cooperative packed PTQ1_0 recurrence

- After fixing the standalone harness's aligned packed-word load, the production-style packed recurrence screen was exact and sanitizer-clean across 16,384 blocks. Cooperative eight-lane decoding took 0.006459 ms versus 0.009615 ms for serial (1.49x faster) and used 32 registers versus 40 without spills.
- Integrated into the active ROWS=1 PTQ1_0 GEMV, the candidate passed 4 selected CTests, 96/96 CUDA-vs-CPU matmul cases, and both 32-token model smokes, but regressed seven-repetition decode by 81.47% at context 512 and 81.50% at 4096. The isolated recurrence screen did not capture the production schedule's communication and parallelism costs.
- Reverted. Manager restored the exact archived ROWS=1 CUDA library and independently verified source/library hashes and PTQ1_0/PQ2_0 model smokes. See `experiments/024-packed-trit-warp/REPORT.md` and `results/exp024/`.
- Next PTQ1_0 candidate: test explicit 2/4-item K-block software pipelining without changing lane ownership or output fold order; measure register pressure and end-to-end decode.

## Experiment 025: PTQ1_0 K-block work-list strip mining

- Compile-time groups of two and four independent work-list items per lane preserved the existing row/K-block ownership, serial block-dot recurrence, partial-buffer slots, and reduction order. Both variants passed 4 selected CTests, 96/96 CUDA-vs-CPU matmul cases, and fixed-seed PTQ1_0/PQ2_0 model smokes with baseline-matching normalized completions.
- Decode medians lost 0.43%/0.42% (items2) and 0.52%/1.36% (items4) at contexts 512/4096. The four-item long-context run had a 49.84 tok/s outlier. All variants reported 76 registers/thread and zero stack/local usage; no useful ILP effect was observed.
- Reverted to the exact source-default ROWS=1 source and archived baseline CUDA library hashes. Do not retry source unrolling alone without disassembly or counter evidence. The next audit is the active fused-gate path's activation reuse. See `experiments/025-ptq1-kblock-ilp/REPORT.md`.

## Experiment 026: gated PTQ1_0 activation-load reuse audit

- Compared the active `<1,1,true,true>` and `<1,1,true,false>` sm_86 SASS bodies. Each has nine 128-bit activation loads; the extra gated `LDG`s are scalar weight-stream loads, not a second activation vector set. Stop the paired-helper implementation path because its load-reuse premise is absent.
- Manager independently re-counted the instructions and verified source/library hashes. No candidate was built and no correctness run was needed. The seven-repetition fresh control was 81.5795/79.0749 tok/s at contexts 512/4096; this refreshes the control but does not change current best. See `experiments/026-ptq1-gate-activation-reuse/REPORT.md` and `results/exp026/`.
- Follow-up: challenge the active plain PTQ1_0 GEMV with a materially different dataflow, while avoiding already-tested row/warp mappings, decoder variants, and source-only unrolling.

## Experiment 027: active plain PTQ1_0 dataflow challenge

- Static source/SASS inspection considered current-block `.L1` prefetch and packed-weight vector/cache-policy changes. Current-block prefetch did not create lookahead, while successive weights are separated by a 28-byte block stride. No candidate reached an implementation-equivalent event screen, so no performance conclusion was drawn.
- Manager reviewed the report and independently verified source/library hashes remain the current production baseline. Keep this as an inconclusive screening result, not evidence that all lookahead or staging approaches fail. See `experiments/027-ptq1-dataflow-challenge/REPORT.md` and `results/exp027/`.
- Follow-up: screen a prefetch for each thread's next `(row group,K block)` work item, since the active 128-thread work list advances each thread by 128 items.

## Experiment 028: PTQ1_0 work-list lookahead prefetch

- Tested `.L1` distance 1/2 and `.L2` distance 1 for each thread's future work item in the actual active plain specialization. Address checks passed for 4,575 shape/pitch combinations and 21.6M future indices; SASS emitted the expected `CCTL.E.PF1/PF2` hints.
- In one 16-token Nsight Systems trace per arm (485 target launches each), plain-kernel totals were 9.577 ms control, 9.799 ms L1-D1 (+2.3%), 10.281 ms L1-D2 (+7.4%), and 9.793 ms L2-D1 (+2.3%). Manager independently reproduced the totals, medians, selected library paths, and address-check result. The screen was negative; no correctness suite or end-to-end candidate A/B was run, and no candidate is retained.
- Next, test load-cache policy directly on the active packed weight reads. This avoids adding future-index decode and tests whether streaming weights currently displace the reusable activation planes.

## Experiment 029: PTQ1_0 packed-weight cache policy

- Tested `.cg` on the six aligned packed `qs` u32 reads in the active plain planar GEMV. SASS emitted `LDG.E.STRONG.GPU`; activation vector loads were unchanged and resources remained 74 registers/thread with no spills. `.cs` was not built.
- Captured one 16-token actual-model profile (4,115 active plain launches, 177.969 ms aggregate) but no matched default control, so this is code-path evidence only. No candidate correctness comparison or end-to-end A/B was performed; decision is inconclusive and the candidate is reverted.
- Interrupted broad rebuilds removed the active CUDA library and some objects. The source-default tree was rebuilt successfully and passed a fixed-seed 32-token PTQ1_0 smoke. Its active-kernel resource record matches the archived control, but the rebuilt `.so` SHA differs from the archived binary and byte identity is not claimed. See `experiments/029-ptq1-weight-cache-policy/REPORT.md` and `results/exp029/`.
- Follow-up: compare default/`.cg`/`.cs` with identical actual-kernel traces and isolated candidate relinks, preserving the working source-default library.

## Experiment 030: matched PTQ1_0 packed-weight cache-policy comparison

- Compared default, `.cg`, and `.cs` on only the six packed `qs` loads, with three rotated Nsight Systems traces per arm. All arms invoked the active plain GEMV exactly 485 times per trace. `.cg` took 21.8838 ms median aggregate versus 9.5801 ms default (+128.4%); `.cs` was 9.5510 ms (-0.30%). SASS emitted `LDG.E.STRONG.GPU` for `.cg` and `LDG.E.EF` for `.cs`.
- Two reversed-order, seven-repetition end-to-end comparisons at contexts 512/4096 rejected `.cs`: the matched 60 C pass was -1.10%/-0.91% median decode, with the same direction in the first pass. Both arms had long-context tails; all samples and peaks (6805 MiB) are saved.
- A fixed-seed completion compare did not complete, so numerical equivalence was not checked. Neither candidate is retained; this is sufficient for rejection but not for a correctness claim. Source and active source-default library remain restored and intact. Temporary candidate binaries were removed after hashes, SASS, traces, and exact patch were preserved. See `experiments/030-ptq1-cache-policy-ab/REPORT.md` and `results/exp030/`.
- Do not retry these cache modifiers on this active plain specialization without new cache-traffic evidence or a different data-reuse premise. Next investigate a different packed-weight staging/dataflow with a correctness-first event screen.
