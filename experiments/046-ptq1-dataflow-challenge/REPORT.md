# Experiment 046: PTQ1_0 dataflow challenge

## HYPOTHESIS

A materially different mapping of the active sm_86 batch-1 PTQ1_0 GEMV might reduce the dependent ternary decode work while retaining the compact 28-byte PTQ1_0 block representation. The challenge began by questioning the one-thread-per-(row,K-block) assignment, rather than assuming it was the right starting point.

## DESIGN CHALLENGE AND SOURCE AUDIT

The active path is `mul_mat_vec_ptq1_0_pt<1,1,...>` in `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`; `mmvq.cu` dispatches this dedicated kernel for the planar-transposed Q8_1 batch-1 path. Each lane owns one `(row,K-block)` item, and the inner helper holds packed byte remainders in 16-bit lanes, advances them with multiply-by-three, extracts four trits using `__byte_perm`, and feeds the existing DP4A accumulators. Independent groups and the main/gate accumulators are already interleaved by the generated code.

I considered whether a new lane mapping could compute the four 32-element Q8 sub-blocks cooperatively. That is the same basic split tested in Exp023 and the production recurrence distribution tested in Exp024; the latter was exact but regressed model decode by about 81.5%. Assigning multiple K items per lane is covered by Exp025 and lost. A different arithmetic expression for direct trit extraction falls into the floor-difference/fixed-point or pairwise families (Exp009/039 and Exp012), which were exact but slower. Replacing the byte stream with lookup or a side encoding repeats Exp004/011. The remaining obvious dataflow directions—weight staging, activation staging, transpose, async prefetch, wider CTA scheduling, and register limits—were measured and rejected in Exp037–040, 042, 044, and 045. Exp032/035 also tested a row-block layout permutation/sidecar and found no end-to-end win.

The active SASS evidence retained in `results/exp043/sass_schedule_excerpt.txt` shows alternating independent DP4A accumulator chains. Source review likewise confirms independent `qs` groups are processed in parallel before each group’s recurrence step; changing loop spelling would not introduce independent arithmetic absent from the compiler schedule. I found no distinct, concrete mapping with a credible chance to reduce total work while retaining the same compact weights and established row-reduction behavior.

## IMPLEMENTATION AND MEASUREMENTS

No candidate was implemented or built. This was an early-stop design challenge: every coherent remaining proposal identified in the audit was a repeat of a measured mapping/arithmetic family, and no genuinely distinct proposal remained to screen. Therefore there are no candidate focused CUDA-event measurements, active candidate resource counts, spills, candidate SASS, load-behavior measurements, correctness/tolerance results, or E2E impact. No model benchmark or correctness suite was run. The existing active SASS excerpt is retained as context only and is not a new measurement.

No production source or build was touched. A detached worktree was created at `/tmp/exp046-ptq1-dataflow` from manager HEAD `c43323558b3f34b5e43ba72715d5de6c84c3ceef`. No candidate binary exists, so candidate `ldd` verification does not apply. Current active baseline hashes remain as recorded in `research/STATE.md`.

## DECISION

**NO CANDIDATE / REVERT.** No implementation was created, so there is nothing to copy into production. Preserve the existing ROWS=1, 128-thread, four-CTA-bound path and current best result. This decision does not assert that PTQ1_0 decode cannot be improved; it records that the bounded fresh challenge found no unexhausted candidate worth an empirical screen.

## FOLLOW-UPS

Reopen this area only with a new primitive or representation that changes the measured work, such as a compact format-level code transform with a demonstrated memory/quality/storage case, or a device instruction/codegen change that is not equivalent to the existing recurrence, direct arithmetic, or cooperative lane variants. Start with an exact active-layout screen at K=40 and K=136 before any model A/B.

## IMPORTANT DISCOVERIES

- The active decoder already uses packed 16-bit-lane recurrence state and byte permutation to expose multiple trits per step; a scalar-per-thread description understates the intra-thread parallelism already present.
- Generated fused-gate SASS interleaves the independent DP4A chains, so explicit source-level stream interleaving has no unaddressed schedule premise under the current toolchain.
- The principal apparently new option—cooperatively mapping each activation sub-block—substantially overlaps prior tests and the exact production recurrence candidate that caused a large end-to-end regression.
- No new performance, correctness, resource, or library measurement was produced in this experiment.

## ISOLATION

Manager checkout and production build were not modified. No commit was made. `research/BEST_RESULTS.json` is unchanged because there is no new independently verified result.
