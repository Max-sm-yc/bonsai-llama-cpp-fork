# Experiment 014: planar warp reduction

## HYPOTHESIS

A warp-per-output-row reduction could replace one shared FP32 partial per K block and the CTA-wide barrier in the active batch-1 planar PTQ1_0 GEMV. Each lane would process K blocks in a strided loop, then reduce within the warp.

## IMPLEMENTATION

The required research state, experiments 010–013, and current PTQ1_0 kernel/dispatch were inspected. The active dedicated kernel is selected for plain 2D, K-aligned PTQ1_0 calls with 1–8 output columns; batch-1 uses ROWS=1. This experiment did not reach an implementation. The proposed mapping would trade more serial K work per lane and fewer output rows per CTA against the eliminated partial array and barrier. No source or build change was made.

## RESULT

No candidate was built or measured.

## CORRECTNESS

Not run: no candidate implementation exists.

## MICROBENCHMARK

Not run.

## END-TO-END IMPACT

Not run.

## ANALYSIS

The warp-per-row mapping is materially different and directly targets the current shared-memory partial exchange. It also lowers CTA row concurrency substantially (four rows per 128-thread CTA) and increases each lane's serial K work, so its outcome cannot be inferred from the saved traffic alone. A candidate and measured decode pair are required to decide.

## DECISION

**INCONCLUSIVE / NO CHANGE.** The source and current best build remain untouched.

## FOLLOW-UPS

Implement the warp-per-row mapping as a one-column-only specialization, retain ROWS=1 block-dot work, and benchmark isolated candidate/control binaries. Run candidate correctness and both model smoke tests before any keep decision.

## IMPORTANT DISCOVERIES

- The one-column batch path uses the dedicated planar-transposed kernel, ROWS=1 block-dot work, a shared FP32 partial per K block, and a CTA-wide barrier before reduction.
- Multi-column calls share this kernel family but are explicitly out of scope for a one-column reduction change.
