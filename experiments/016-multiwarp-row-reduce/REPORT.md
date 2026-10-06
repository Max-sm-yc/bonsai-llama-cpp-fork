# Experiment 016: cooperative multiwarp row reduction proposal

## HYPOTHESIS

Two or four warps per output row could restore K-block parallelism lost by experiment 015 while reducing shared partial stores from one per K block to one per warp.

## IMPLEMENTATION

The experimenter read the required state and reports and confirmed the active batch-1 dispatch and fusion route. It noted that the proposed path would need changes to the CTA tile chooser, shared-memory sizing, and specialized epilogue, then ended without editing source. No candidate was built.

## RESULT

Not measured. There is no evidence for or against this mapping.

## CORRECTNESS

Not run because no implementation was produced. Production source and build were not changed.

## MICROBENCHMARK

Not run.

## END-TO-END IMPACT

Not measured.

## ANALYSIS

The work needed to implement the variant is part of the experiment, not a technical blocker. Experiment 015 only rejects one warp per row with serial per-lane K loops; it does not decide a cooperative multiwarp design.

## DECISION

**INCONCLUSIVE / NOT IMPLEMENTED.** The multiwarp hypothesis remains open.

## FOLLOW-UPS

Start with a minimal compile-time candidate for 2 warps per row and 4 warps per CTA (two output rows/CTA). Each warp should cover a disjoint K-block range, reduce its local accumulator, and write one compact row/warp sum for the final CTA combination. Compile this first, then test a 4-warps-per-row variant if justified by the first measurements.

## IMPORTANT DISCOVERIES

- No implementation or benchmark result was obtained in this attempt.
- The one-warp-per-row result from experiment 015 remains a measured regression, but does not resolve the multiwarp design.
