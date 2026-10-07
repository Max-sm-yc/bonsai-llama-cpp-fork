# Experiment 037: CTA-local PTQ1_0 weight staging

## HYPOTHESIS

The active sm_86 ROWS=1 GEMV assigns adjacent lanes consecutive 28-byte AoS blocks. Its packed `qs` word loads therefore stride 28 bytes across lanes. A CTA can first read one contiguous tile of 128 blocks in coalesced word rounds, stage the tile in shared memory, then run the existing packed dot on the staged blocks. This keeps the model's AoS representation and avoids Exp035's persistent sidecar and repack cost.

## IMPLEMENTATION

Implemented a focused CUDA screen in [`staging_screen.cu`](../../results/exp037/staging_screen.cu), based on the Exp032 exact packed-dot harness. It compares direct AoS loads, the existing seven-plane SoA reference, and the staged AoS path. The staged path copies each 128-block CTA tile as seven contiguous global word rounds into a 3,584-byte shared array, synchronizes, executes the same packed recurrence/DP4A/scale correction, writes per-block partials, and uses the same four-accumulator row fold as the control. The benchmark times work plus fold. Shapes are 40 blocks/row (K=5120) and 136 blocks/row (K=17408), 2,048 rows, 128-thread CTAs, nine rotated-order samples; each sample averages 100 launches at 40 blocks and 80 at 136 blocks.

Built on RTX 3080 / sm_86 with `nvcc -O3 -arch=sm_86 --ptxas-options=-v`. The source hash is `9ccad63769e7f9b2d66074c8f4e5b8d271c99d3264eaed1f1248e3484da2635f`. The screen harness, raw timings, resources, SASS, and sanitizer log are under [`results/exp037/`](../../results/exp037/). No production source or binary was changed; the manager checkout remained at HEAD `b50a4b222779f97c738319dd4aaecd82c3cbbe45`.

## RESULT

**REVERT.** CTA staging lost against direct AoS at both shapes. Median work-plus-fold time was 9.25696 µs direct versus 10.16832 µs staged at 40 blocks (+9.85%), and 25.07520 µs versus 28.30080 µs at 136 blocks (+12.86%). All nine staged samples were slower than the direct samples at each shape. The SoA reference measured 9.23264 µs at 40 blocks and 23.10400 µs at 136 blocks; its result is context only and it is not the staged candidate.

## CORRECTNESS

All packed codes matched the independent host decoder: 10,485,760 codes at 40 blocks and 35,651,584 at 136 blocks, with zero mismatches. Direct AoS, SoA, and staged row outputs matched bitwise. All outputs also matched the independent host output reference exactly (zero mismatches, max absolute error 0). Compute Sanitizer memcheck on 257 rows at 40 blocks reported zero errors. See the raw screen files and [`memcheck_40.txt`](../../results/exp037/memcheck_40.txt).

## MICROBENCHMARK

CUDA event milliseconds for the paired work-plus-fold launches; nine samples per arm:

| Blocks per row | Direct AoS median (range), µs | CTA staged median (range), µs | Staged delta |
|---:|---:|---:|---:|
| 40 | 9.25696 (9.24672–9.31712) | 10.16832 (10.15712–10.16832) | +9.85% |
| 136 | 25.07520 (25.01120–25.27360) | 28.30080 (28.26240–28.38960) | +12.86% |

Ptxas reports 40 registers/thread, no stack or spills, and 3,584 bytes shared memory for `work_staged`; `work_aos` uses 40 registers, no stack/spills, and no shared memory. SASS confirms the intended coalesced global loads (`LDG.E`), shared stores/loads (`STS`/`LDS`), and a CTA barrier (`BAR.SYNC`). This proves the copy is generated as designed, but its added work costs more than it saves in this screen. Full resource output is [`resources.txt`](../../results/exp037/resources.txt); SASS is [`staging_screen.sass`](../../results/exp037/staging_screen.sass).

## END-TO-END IMPACT

Not measured. The candidate lost the focused screen at both common and long-K shapes, so it did not advance to model integration or decode A/B. The project reference remains 83.35 tok/s at context 512 and 80.35 tok/s at context 4096.

## ANALYSIS

The staging scheme did coalesce the input stream, but each CTA also performed seven rounds of shared stores and reads plus synchronization and indexing. That overhead raised work-plus-fold time by about 10–13%, with the loss larger for 136 blocks per row. The result rejects this particular CTA-local staging design without needing unavailable NCU counters. Persistent SoA remains a separate path and is not evaluated by this decision.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** Reject CTA-local shared-memory staging for this kernel. Keep the current ROWS=1 production implementation. No production changes were made.

## FOLLOW-UPS

Do not integrate this staging scheme. A future dataflow challenge needs to remove or amortize the shared copy and barrier while preserving the direct path's packed-dot parallelism; any new mapping should pass the same 40/136-block focused screen before runtime work.

## IMPORTANT DISCOVERIES

- A contiguous 128-block AoS tile can be copied through seven coalesced global word rounds and consumed by the packed dot from shared memory, including tiles that cross row boundaries in the flattened work list.
- The staged path compiled without spills but added a barrier and shared-memory traffic; its focused penalty was repeatable at both shapes.
- The independent host code and output checks, plus sanitizer, passed exactly. No model benchmark was needed to reject this candidate.
