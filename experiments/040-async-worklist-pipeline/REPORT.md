# Experiment 040: sm_86 `cp.async` next-work-item pipeline

## HYPOTHESIS

Can an sm_86 `cp.async` pipeline hide the global-memory latency of a thread's next PTQ1_0 K-block work item while that thread decodes and dots its current item? The experiment uses the active planar-transposed activation indexing and the ROWS=1 flattened `(row, K block)` schedule, with 40 blocks/row as a short-row control and 136 as the long-row case.

## IMPLEMENTATION

The standalone focused harness is `results/exp040/async_screen.cu`. Both arms use the same 128-thread CTA shape, 16-row CTA tile, per-thread item stride 128, PT activation-plane indexing, block decoder/dot, and separate row fold. The control synchronously loads the current item's seven 32-bit packed weight words. The candidate issues seven 4-byte `cp.async.ca.shared.global` operations for the next item into a per-thread shared slot, commits the group, and computes the current item before waiting and consuming the copied item. Each CTA owns 16 rows, giving 640 work items (five items/thread) at K=40 and 2,176 (17 items/thread) at K=136, matching the active row-tile selection for these shapes. The separate fold kernel is included in event timing.

This is a focused scheduling screen; it does not edit production CUDA source or build/runtime libraries. The first draft harness mistakenly let each CTA stride across the entire flattened workload. That draft was discarded. Final commands and samples below use the corrected production-like CTA row tile; the retained source and binary are the corrected versions.

Build and screen commands:

```sh
nvcc -O3 -arch=sm_86 results/exp040/async_screen.cu -o results/exp040/async_screen
results/exp040/async_screen 40 4096 100 > results/exp040/screen_40.txt
results/exp040/async_screen 136 2048 60 > results/exp040/screen_136.txt
nvcc -O3 -arch=sm_86 -Xptxas=-v -cubin results/exp040/async_screen.cu -o results/exp040/async_screen.cubin 2> results/exp040/resources.txt
cuobjdump --dump-sass results/exp040/async_screen.cubin > results/exp040/async_screen.sass
```

The source was prepared from main HEAD `afb781800d526a35cf9aad36d45329e5e3ab7671` in isolated worktree `/tmp/exp040-async-worklist`; production source/build remained untouched. Device: RTX 3080, sm_86; driver 580.178.04; CUDA toolkit 13.2.86. Final SHA-256: source `1cdfc7b6495e3b8c1707ea92c149e39063f4fcf94e6220011470e7a0aad59b74`, executable `a7f4a37083e259b5a6016d94dc6caab5d7c388dd82117ba787fc8c3c59a039c7`, cubin `7f8077a8cde532d03c2762a730d276c2dee80fc16bd0257c30cd8820872cf825`.

## RESULT

**REVERT.** The candidate is slower at both shapes: +10.40% at 40 blocks and +23.16% at 136 blocks. It does not qualify for model integration or E2E decode testing.

## CORRECTNESS

The screen emits and verifies every decoded device code against the independent host decoder. At 40 blocks, all 20,971,520 codes matched; at 136, all 35,651,584 matched. The candidate's complete row outputs matched the synchronous control bitwise, and both matched the independent host reference: zero mismatches, maximum absolute error 0 at both shapes. Raw correctness output is retained in `results/exp040/screen_40.txt` and `screen_136.txt`.

## MICROBENCHMARK

Nine CUDA-event samples per arm, rotated order, each sample repeats work-plus-fold 100 times at K=40 and 60 times at K=136. Values are milliseconds per work-plus-fold pair; raw samples are in the screen text files.

| K blocks/row | Synchronous median (range), ms | `cp.async` median (range), ms | Candidate delta |
|---:|---:|---:|---:|
| 40 | 0.013895 (0.013865–0.014027) | 0.015340 (0.015328–0.015410) | +10.40% |
| 136 | 0.027853 (0.027802–0.028092) | 0.034303 (0.034259–0.035022) | +23.16% |

Both candidate ranges are wholly above the corresponding control ranges. `ptxas` reports 40 registers/thread, zero stack/spills, one barrier, and 3,584 bytes shared memory for each work kernel (shared storage is otherwise unused in control). SASS for the candidate contains seven `LDGSTS.E` instructions (the sm_86 lowering of `cp.async`), `LDGDEPBAR`, and a CTA synchronization before shared values are consumed. The intended asynchronous transfer therefore exists in generated code; it does not translate into useful latency hiding here.

## END-TO-END IMPACT

Not run: the focused screen regressed at both shapes. Current reference remains 83.3458 tok/s at context 512 and 80.3522 tok/s at 4096. Production source and library were not changed.

## ANALYSIS

The schedule exposes the target overlap: the next work item's packed weight copies are issued before current-item decode/dot completes. The benefit is outweighed by seven narrow async transactions per next item, shared-memory address/staging work, waiting, and CTA synchronization. For K=40, each thread has only five items and little pipeline depth; at K=136, the additional item count gives more opportunities but also more repeated staging and synchronization, worsening the measured delta. The exact active PT activation mapping and fold are retained, and code generation confirms actual async operations, so this rejects this per-thread seven-word pipeline rather than merely a compiler-elided hint.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** Do not integrate. Keep the production ROWS=1 path unchanged.

## FOLLOW-UPS

No follow-up for this per-thread AoS weight-copy pipeline. A future overlap proposal needs to amortize staging/synchronization across more useful data and must first demonstrate focused work-plus-fold gains in the exact mapping.

## IMPORTANT DISCOVERIES

- sm_86 emitted the expected `LDGSTS` async copies, dependency barrier, and synchronization; the mechanism was not optimized away.
- The candidate retained the control's register count and introduced 3.5 KiB shared memory plus synchronization.
- Correct async overlap still lost by 10.40% at 40 K blocks and 23.16% at 136; no model result is claimed.
