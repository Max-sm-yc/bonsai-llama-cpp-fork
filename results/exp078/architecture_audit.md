# Exp078 architecture audit

Base commit: `8e9229e4f8e61cf86e2905854f80cf86977c78e2`  
Target: RTX 3080, sm_86  
Candidate changes: none

## Source-derived operation

The active batch-1 path is `mul_mat_vec_ptq1_0_pt<1,1,...>` in `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`, selected for planar Q8_1 in `mmvq.cu`. Each independent lane item is one output row and one K block (128 weights). For each PTQ block, the implementation loads 28 bytes of packed weights and nine 16-byte planar activation pieces (eight quant planes plus four `(d,s)` pairs). It expands base-3 packed digits through five recurrence steps, keeps raw digits packed in registers, accumulates them with DP4A, and subtracts each subgroup's exact activation sum so that `digit-1` is implemented without signed-byte materialization. Four integer subgroup results are folded into FP32 in the existing order. K blocks are parallelized across work items; partials are reduced per row through shared memory.

This describes 144 activation bytes and 28 packed weight bytes requested per row/K-block item before cache reuse/coalescing effects, and 128 useful integer products. Across 401 model GEMV tensors, prior GGUF parsing counted 5,599,641,600 weight bytes. Prior steady graph measurements were 9.014/9.022 ms per token at contexts 512/4096. Their payload/time ratio (~621 GB/s) is payload-equivalent only; Exp074's synthetic larger-than-L2 stream ceiling was ~725 GB/s and does not identify actual GEMV DRAM traffic.

## New alternative cost model: persistent signed bit planes

Alternative considered: convert each 128-trit block once at model load into positive and negative 128-bit masks, then compute `sum(a_i for positive) - sum(a_i for negative)` using the planar activation. This would replace per-use base-3 recurrence with mask tests and reductions, a genuinely different representation and repeated-use dataflow.

Cost screen: two 128-bit masks require 32 bytes/block versus 28 bytes for PTQ1_0 (14.3% more packed-weight payload; scale metadata is shared). The 5.600 GB active PTQ payload would grow by approximately 0.800 GB. Each output still needs to classify and select arbitrary signed int8 activation values, then reduce positive and negative contributions. A direct implementation needs either activation masking/packing plus two sums, or lane predicates and reductions; unlike binary activations this cannot become popcount. The current path already has the 128 activation values and performs a single packed DP4A stream using four subgroup corrections. The representation therefore raises compulsory weight traffic where weight reads are already substantial, while not reducing the 128 useful multiplies and adding selection/reduction work. A model-load transform would also require new storage/lifecycle integration and under 10 GiB accounting. With no credible instruction or bandwidth saving to offset those costs, a kernel prototype is not justified.

The tempting hybrid of reconstructing positive/negative masks on the fly is worse: it retains base-3 decode and adds mask extraction. The alternative only has a plausible case if an architecture supplies a fast bit-mask gather/reduction over arbitrary signed bytes or an encoding with materially fewer stored bits and direct dot instructions; sm_86 does not provide that primitive.

## Evidence review

- Required research state, ideas, and experiment index were read. Exp001–025 rows were reviewed for generic dispatch, decoder, row/warp mappings, and recurrence outcomes; reports 004/009/011/012/015/017/018/023/024/025 document exact-but-slower variants or large model regressions.
- Exp039: floor-difference decoding exact but 2.62%/4.28% slower at K=40/136.
- Exp043 retained SASS shows the independent fused main/gate DP4A chains already interleaved.
- Exp046 repeated the broad architecture challenge and found no candidate.
- Exp068 confirmed direct packed-digit DP4A and exact `isum` correction are already active.
- Exp073 ruled out batch-1 signed-int8 Tensor Cores (7/8 output columns redundant plus expansion/staging) and bit-sliced popcount.
- Exp074 distinguishes payload-equivalent replay rate from measured DRAM bandwidth.
- Exp075 finds no whole-matrix next-token L2 retention premise; reuse distance is not a hit-rate measurement.
- Exp076 covers prefill MMQ, not this batch-1 decode path; it confirms the prefill Tensor Core path is already active.

No Nsight Compute permissions were changed. Its counters remain unavailable (`ERR_NVGPUCTRPERM`). No source, build, tests, benchmark settings, or manager checkout were changed.

## Source hashes

See `results/exp078/source_hashes.txt` for hashes of the active kernel, dispatch, and prior SASS evidence.
