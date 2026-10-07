# Exp073: Fresh sm_86 PTQ1_0 batch-1 GEMV challenge

## HYPOTHESIS

An Ampere integer Tensor Core tile across output rows might amortize packed-weight decoding and activation reads better than the active direct DP4A path. I also screened a bit-sliced positive/negative mask formulation. The target is the planar PTQ1_0 batch-1 GEMV on RTX 3080/sm_86, which contributes about 74–77% of steady decode graph kernel time.

## IMPLEMENTATION

No kernel was implemented. A feasibility audit derived the exact int8 Tensor Core dataflow and counted its work. The detailed derivation, source hashes, facts, and cost estimates are in [design_audit.txt](../../results/exp073/design_audit.txt).

Tensor Core candidate shape: stage a 16-row tile of signed ternary weights expanded to int8, replicate each activation across the eight N columns, and issue signed-int8 `m16n8k16` MMA instructions. Keep each Q8 32-element subgroup's int32 sum separate, then apply its scale and existing fold. This can express the integer dot exactly, but it changes row/K scheduling and requires a new FP32 reduction equivalence check.

The isolated worktree is based on `b74746760ea0fefd9ca11a866695bca99f7bb15a` at `/home/maxsun/autonomous_projects/.worktrees/exp073-sm86-gemv-alternative`; its branch is `exp073-sm86-gemv-alternative`. Production source hashes match the base. No source or runtime library was changed.

## RESULT

No candidate passed the feasibility screen. An `m16n8k16` tile would do 2,048 products per instruction but only 256 are useful for the one-column output; the seven replicated columns make 87.5% of its products redundant. A 128-element block across 16 rows takes eight MMA instructions, computes 16,384 products for 2,048 useful products, and first requires expansion/staging of 2,048 signed weight bytes. The current kernel retains packed digits in registers and uses DP4A directly. With no evidence that Tensor Core throughput offsets that expansion, shared-memory traffic, synchronization, redundant columns, and row/K grid changes, the expected net gain is not sufficiently grounded to implement.

The bit-sliced mask route also does not survive: PTQ1_0 codes are base-3 packed, and Q8 activations are arbitrary signed int8, so decoding masks still requires the base-3 work and the weighted sum cannot be replaced by a binary popcount. It adds selection/reduction work to the existing direct dot.

Baseline diagnostic remains 9.014 ms/token at context 512 and 9.022 ms/token at context 4096; these are prior measurements from research state, not new Exp073 measurements. No before/after candidate samples exist.

## CORRECTNESS

No candidate was built, so no candidate arithmetic/output check, sanitizer run, selected correctness suite, or fixed-seed model comparison applies. The Tensor Core equation can represent exact subgroup integer sums, but equivalence of the subsequent floating-point reduction remains unproven and would require testing at K=40/136 and model shapes before performance measurement.

## MICROBENCHMARK

Not run. There are no candidate event samples, resource counts, SASS, or library hashes. No Nsight Compute permission change was attempted. Existing baseline profile remains the production diagnostic; the audit does not infer memory or integer-pipe saturation from it.

## END-TO-END IMPACT

Not measured. No candidate qualified for paired decode. The established best result remains Exp062 (`ffb0ef37690b902829ea1158b02b14517ed93c2b`), with paired-current medians 84.407 tok/s at context 512 and 81.885 tok/s at 4096.

## ANALYSIS

Tensor Cores offer a genuinely different primitive, but batch-1 GEMV is an awkward fit for the required `m16n8` output tile. Replicating the activation column preserves the operation but wastes seven output columns. Expanding ternary weights to signed bytes also forfeits the active kernel's central property: packed codes are decoded in registers and fed directly into DP4A without a materialized weight vector. In addition, the 16-row tile replaces independent row/K-block work with a grouped tile and requires maintaining four independently scaled Q8 subgroups and the existing output reduction behavior. Without a concrete compiler/resource result or a tested alternative MMA shape that avoids these costs, the nominal instruction-count advantage alone is insufficient evidence for a likely win.

The mask formulation fares worse because Q8 values are not binary; bit masks do not turn arbitrary signed-byte weighted sums into popcount operations. The audit therefore closes this specific Tensor Core/bit-sliced screen, while leaving future work open to a primitive or representation with lower exact total work and no expansion penalty.

## DECISION

**REVERT / NO CANDIDATE.** No code was changed. The isolated branch contains only this report and its audit artifact; nothing was committed to the main tree.

## FOLLOW-UPS

Reopen only if a new Ampere MMA shape or data representation can avoid redundant output columns and signed-weight expansion, or a prototype demonstrates enough throughput margin to repay both. Any candidate must first prove exact subgroup sums and model-shape reduction behavior at K=40/136, then pass sanitizer/correctness and resource/SASS inspection before paired E2E decode.

## IMPORTANT DISCOVERIES

- The active GEMV already feeds raw packed ternary digits directly to DP4A and corrects `digit-1` with the exact activation sum.
- Signed-int8 `m16n8k16` is expressible but computes eight columns for batch-1 GEMV; replicated activations make seven columns redundant.
- A 16-row Tensor Core tile needs an expanded signed-byte weight tile and changes the existing row/K-block work grid.
- Q8 activations are arbitrary signed bytes, so ternary bit slices do not enable a simple popcount replacement.
- No candidate correctness, performance, SASS, or end-to-end claims were made.
