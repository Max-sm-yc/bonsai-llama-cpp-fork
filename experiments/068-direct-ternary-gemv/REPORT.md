# Experiment 068: direct ternary dot challenge for active PTQ1_0 GEMV

## HYPOTHESIS

The active PTQ1_0 planar GEMV might be spending substantial time materializing signed-byte ternary weights before applying the Q8 activation dot. Feeding packed ternary codes directly into the dot could remove that work in the dominant batch-1 decode kernel.

## IMPLEMENTATION

No candidate was implemented because source inspection disproved the premise. At manager HEAD `661966965935edc19243a54dc6404027491edff7`, `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh::ptq1_0_pt_block_dot` decodes each packed `qs` group into raw digits 0, 1, or 2 in registers and immediately passes those packed digits to `ggml_cuda_dp4a_us`. It does not materialize signed-byte weights. The signed value `digit - 1` is handled algebraically by subtracting the exact activation sum (`isum`) at the 32-element fold. The `qh` path follows the same direct-dot pattern.

A distinct positive/negative one-hot mask formulation was considered. It requires class extraction from the same base-3 stream followed by byte masking/packing and dot work, with no evident instruction reduction versus the existing packed-lane recurrence and DP4A path. That is an unsupported decoder rewrite, not a grounded direct-dot opportunity. Prior exact active-path alternatives (2-bit side encoding, pairwise radix-3 and fixed-point floor differences) all lost their focused screens; see the related reports and `results/exp068/source_audit.txt`.

An isolated worktree was created at `/home/maxsun/autonomous_projects/.worktrees/exp068-direct-ternary-gemv`, detached at manager HEAD. The source hash is recorded in the audit artifact. The worktree has no candidate changes. No build was necessary because there was no candidate to compile.

## RESULT

The requested optimization already exists in the active dataflow. The failed premise is that the current GEMV first materializes signed-byte weights. The kernel instead fuses packed digit extraction, DP4A accumulation and the signed-weight activation-sum correction. No distinct alternative had a credible, evidence-based path to lower total work, so this challenge ends with no candidate.

## CORRECTNESS

No candidate was built or tested. Existing production arithmetic and behavior are unchanged. Therefore Exp068 claims no new correctness result and no Compute Sanitizer result. Exactness evidence for adjacent decoder formulations is documented in Exp011, Exp012 and Exp039.

## MICROBENCHMARK

Not run: there is no candidate kernel to time. No CUDA event samples, resource report, SASS comparison or candidate build hash exists. Production performance baselines remain those in `research/STATE.md` and Exp047.

## END-TO-END IMPACT

Not run. No candidate qualified for integration or model A/B. This report makes no end-to-end performance claim.

## ANALYSIS

For a ternary digit `d` in `{0,1,2}`, the signed weight is `d - 1`. The active kernel computes `dot(d, activation) - sum(activation)` for each 32-element sub-block, then applies the Q8 scale. This is algebraically the signed ternary dot and removes any need to store or load an intermediate signed-byte vector. The direct-dot question therefore identifies existing behavior rather than an optimization gap.

The next plausible class-mask formulation must still decode the base-3 representation, and adds byte-lane selection and masking before the dot. The current stream is already packed to four digits per DP4A operand, with independent groups and fused-gate accumulator chains scheduled in the active path. Prior exact alternative decoders lost on the active layout. Without a new primitive or code-generation opportunity, implementing the mask route would repeat the exhausted decoder work without a measured premise.

## DECISION

**NO CANDIDATE / REVERT.** Preserve the unchanged ROWS=1, 128-thread, four-CTA-bound active kernel. No source was modified, no build or benchmark was run, and there is nothing to integrate.

## FOLLOW-UPS

Reopen only if a new primitive, compiler/code-generation change, or representation demonstrates lower work while retaining direct packed-code accumulation. Begin with an exact active-layout screen at model-relevant K shapes before runtime integration.

## IMPORTANT DISCOVERIES

- The active GEMV already feeds packed raw ternary digits directly to DP4A; it never materializes the signed-byte weight vector.
- The `isum` subtraction implements the `digit - 1` signed-weight bias exactly inside the Q8 fold.
- The active `qh` tail also uses direct packed-digit accumulation.
- The premise failed at source inspection. No timing or correctness claims were added.
