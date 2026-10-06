# Experiment 027: PTQ1_0 batch-1 dataflow challenge

## HYPOTHESIS

A distinct sm_86 dataflow in the active plain `<ncols=1, ROWS=1, has_fusion=false, has_gate=false>` PTQ1_0 planar GEMV may improve batch-1 decode beyond ROWS=1. The required initial audit covered the active source, prior experiments 010–018 and 023–026, available sm_86 SASS evidence, and live GPU state.

## IMPLEMENTATION

No candidate was implemented. I considered explicit `.L1` prefetch for each current weight block as a memory-access change. Inspection showed it would prefetch the same block immediately before its existing packed dot, leaving no meaningful lookahead; it would add an instruction without demonstrated overlap. Applying a cache modifier or vector load to packed weights is constrained by the 28-byte block stride and unaligned addresses of successive blocks. The SASS/load audit from experiment 026 also rules out duplicated activation vector loads as a premise.

The active source and library were left untouched. Inspection details and observed hashes are in [inspection.txt](../../results/exp027/inspection.txt). There is no candidate patch.

## RESULT

**INCONCLUSIVE.** The inspection did not identify a sufficiently supported candidate for the required implementation-equivalent CUDA-event screen. No event screen, model integration, or end-to-end A/B was run. This does not establish that all alternate dataflows are unprofitable; it records that the considered prefetch/cache candidates lacked a defensible premise from the available evidence.

## CORRECTNESS

No code changed, so no candidate correctness run applies. Production source hash remains `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`; active library hash remains `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`.

## MICROBENCHMARK

Not run. There was no candidate. The only relevant SASS evidence is the experiment 026 audit: both active fused and ungated block-dot bodies contain nine `LDG.E.128` loads, consistent with the nine planar activation vectors. Nsight Compute remains unavailable as recorded in the project state.

## END-TO-END IMPACT

Not measured. The fresh control from experiment 026 remains 81.5795 / 79.0749 tok/s at contexts 512 / 4096. Since there is no candidate, these numbers are a reference only and do not constitute an A/B result.

## ANALYSIS

The obvious activation-load reuse path was already ruled out by SASS. A same-item weight prefetch offers little overlap because it precedes the existing dot load/decode chain; an actual lookahead requires redesigning the work-list issue schedule and was not supported by current codegen evidence. Packed weight vectorization also has to accommodate the 28-byte stride. Implementing a speculative candidate anyway would not meet the stated requirement to screen an implementation-equivalent operation before integration.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**INCONCLUSIVE.** No candidate survived source/SASS inspection to a CUDA-event screen. Production source and active library were not modified and remain at the requested hashes. No commit was made.

## FOLLOW-UPS

A future challenge should first establish a concrete memory or dependency premise from active plain-specialization SASS or a permitted kernel counter/timer. Candidate directions might include a work-list lookahead that provably overlaps weight fetches, or a packed-weight staging representation that handles the 28-byte stride without excessive copy instructions. Any such design still needs an implementation-equivalent event screen, correctness checks, and matched seven-repetition decode before retention.

## IMPORTANT DISCOVERIES

- The active source maps one thread to a row/K-block dot and uses a serial packed recurrence; each work item gathers planar activation vectors through aligned `int4` loads.
- The activation vector traffic is already shared by compiler codegen in fused and ungated specializations (nine 128-bit loads each), so duplicate-load elimination has no support.
- Current-block prefetch is not a real lookahead strategy, and packed-weight alignment makes broad vector-load/cache-policy edits nontrivial.
- No source or binary modifications occurred; observed hashes match the control.

## MANAGER VERIFICATION

The manager reviewed the report and inspection notes on 2026-10-06 and independently recomputed both production SHA-256 values. No source, active library, or candidate patch changed. This is a bounded negative inspection result, not a performance result; the next experiment tests the explicit work-list lookahead suggested here.
