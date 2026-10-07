# Experiment 078: PTQ1_0 architecture challenge

## HYPOTHESIS

The active PTQ1_0 batch-1 GEMV accounts for roughly three quarters of steady decode graph time. A different exact representation might reduce its recurring work even though the current kernel already keeps packed ternary digits in registers and feeds DP4A directly. This challenge derived the operation from source first, then evaluated a persistent signed-bit-plane representation as a new dataflow.

## IMPLEMENTATION

Created isolated worktree `.worktrees/exp078-ptq1-architecture-challenge` from `8e9229e4f8e61cf86e2905854f80cf86977c78e2`. No candidate was implemented: the representation cost model did not produce a plausible net-win case. Source-derived cost and hash evidence are in [`architecture_audit.md`](../../results/exp078/architecture_audit.md) and [`source_hashes.txt`](../../results/exp078/source_hashes.txt).

Each active work item handles one row and one 128-weight block. The path loads 28 packed weight bytes and nine planar activation vectors (144 bytes), decodes base-3 digits in registers, issues the exact integer DP4A products and applies `digit-1` by subtracting the subgroup activation sum. Prior profile evidence measures 9.014/9.022 ms per token of GEMV-family replay at contexts 512/4096. Exp074's approximately 621 GB/s value is payload-equivalent, not measured DRAM throughput.

The alternative would store positive and negative bit masks for each 128-weight block. That takes 32 bytes instead of 28, increasing the 5.600 GB active weight payload by approximately 0.800 GB (+14.3%). It still must select arbitrary signed Q8 values and reduce the selected values; it cannot use popcount. The active path already performs the useful 128 products and has no signed-weight materialization. The added weight traffic and selection/reduction work outweigh the speculative benefit of removing recurrence instructions. Reconstructing masks inside the kernel would retain the recurrence and add work. No other unexhausted exact mapping survived review of the listed prior experiments.

## RESULT

No candidate qualified for implementation. There is no build, focused kernel timing, or model A/B result. This is a negative architecture result, not a claim that the active kernel cannot be improved. The source challenge found no concrete alternative with a grounded cost advantage on sm_86.

## CORRECTNESS

No candidate arithmetic was implemented, so candidate-vs-production/reference checks, sanitizer tests, and `tests/run_correctness.sh` were not run. The current arithmetic and production code remain unchanged.

## MICROBENCHMARK

Not run because there was no candidate. Prior measurements used as context are from Exp047/062/074 and are not new Exp078 samples. No Nsight Compute permission changes were attempted; counters remain unavailable with `ERR_NVGPUCTRPERM`.

## END-TO-END IMPACT

Not run because no candidate passed architecture screening. No performance claim is made. Current-best PTQ1_0 results remain 84.407 tok/s at context 512 and 81.885 tok/s at context 4096.

## ANALYSIS

The direct packed ternary dot is already the key representation advantage of this kernel. An alternate encoding must reduce recurring per-weight work enough to repay conversion/storage and avoid adding operations to arbitrary signed-int8 activation products. The tested bit-plane concept increases persistent bytes and still needs mask selection/reduction. The alternatives in Exp001–025, 039–046, 068, and 073–076 cover the remaining obvious decoder, lane-mapping, Tensor Core, cache, staging, and scheduling choices; the focused reports supply negative evidence against repeating them. The full source-derived operation and cost calculation are retained in the audit artifact.

## DECISION

**NO CANDIDATE; retain current implementation.** No production code or best result was changed. Manager checkout was not modified.

## FOLLOW-UPS

Reopen when a primitive or model representation can directly reduce arbitrary signed activation products without increasing the active weight payload or materializing expanded weights, or when new hardware counters identify a measured bottleneck that changes the cost model.

## IMPORTANT DISCOVERIES

- Current GEMV consumes packed base-3 digits directly through DP4A and implements the signed bias with exact subgroup activation sums.
- Positive/negative bit planes require 14.3% more active weight storage and still need arbitrary signed-byte selection and reduction.
- Existing payload-equivalent rate and synthetic read ceiling do not establish actual GEMV DRAM bandwidth.
- No exact implementation or performance candidate is supported by the current sm_86 cost model.
