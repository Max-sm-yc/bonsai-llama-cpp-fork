# Exp063: audit the remaining recurrent SSM/L2 sites

## HYPOTHESIS

The 24 `SSM_CONV + SiLU → L2_NORM` pairs not matched by Exp062 might use a different live QK view offset or shape that supports a separate exact fusion. Any such path would require a narrow dispatch and the same generic fallback.

## IMPLEMENTATION

Used an isolated worktree at code commit `ffb0ef37690b902829ea1158b02b14517ed93c2b`. Added temporary scheduler metadata instrumentation to capture tensor shapes, strides, view offsets, live uses, pins, data addresses, and alias ranges, then removed it. Production source has no diff.

The captures used the PTQ1_0 model with `-p 0 -n 1 -r 1`, contexts 512 and 4096, 99 GPU layers, FlashAttention on, `b=2048`, `ub=512`, F16 KV, and 8 CPU threads. Each capture began at 50 C / 0% GPU utilization. Repeated scheduler constructions were deduplicated by layer name.

## RESULT

All 48 sites at both contexts have identical operator shapes and strides:

| Tensor | Shape | Byte strides | Uses | Flags / pinned |
| --- | --- | --- | ---: | --- |
| SSM_CONV output | `[10240,1,1,1]` | `[4,40960,40960,40960]` | 1 | 16 (COMPUTE), not OUTPUT-pinned |
| SiLU output | `[10240,1,1,1]` | `[4,40960,40960,40960]` | 2 | 16, not OUTPUT-pinned |
| QK input view | `[128,32,1,1]` | `[4,512,40960,40960]` | 1 | 16, not OUTPUT-pinned |
| L2 output | `[128,32,1,1]` | `[4,512,16384,16384]` | 2 | 16, not OUTPUT-pinned |
| SSM input 0 | `[4,10240,1,1]` | `[4,16,163840,163840]` | 2 | 16, not OUTPUT-pinned |

The QK view is a zero-offset view of SiLU in every case. The only split is storage aliasing:

| Classification | Count | Layers | L2 destination relative to SSM input 0 |
| --- | ---: | --- | --- |
| Exp062-compatible | 24 | 1, 4, 6, 9, 12, 14, 17, 20, 22, 25, 28, 30, 33, 36, 38, 41, 44, 46, 49, 52, 54, 57, 60, 62 | Disjoint |
| Rejected by Exp062 guard | 24 | 0, 2, 5, 8, 10, 13, 16, 18, 21, 24, 26, 29, 32, 34, 37, 40, 42, 45, 48, 50, 53, 56, 58, 61 | Exact same base pointer; L2 writes bytes `[0,16384)` within the 160 KiB SSM input |

For the rejected class, an L2 store can overwrite input still being read by another SSM CTA. The guard correctly checks live address ranges, rather than relying on tensor names or view offsets.

## CORRECTNESS

Exp063 created no candidate, so it has no new output comparison or tolerance claim. The temporary instrumentation did not change dispatch or arithmetic and was removed. Exp062 remains the only relevant kernel correctness evidence: its exact model and fallback comparisons were byte-identical, its integrated CTest passed 1/1, and its broader CUDA backend checks passed 96/96. See [Exp062](../062-repeated-smallop-fusion/REPORT.md).

## MICROBENCHMARK

No Exp063 kernel microbenchmark was justified because no safe new dispatch was identified. Exp062's prior graph capture measured 1,384→1,360 nodes/replay and summed kernel time 11.728813→11.696529 ms at context 512, and 12.102607→12.072064 ms at context 4096. These are Exp062 results, not Exp063 measurements.

## END-TO-END IMPACT

No Exp063 A/B was run because production code did not change. Exp062's prior four-pair PTQ1_0 medians were +0.037% at context 512 and +0.231% at context 4096; they remain the current best comparison.

## ANALYSIS

The remaining sites do not represent a distinct view geometry. They differ only because the scheduler reuses SSM input storage for the L2 destination. The fused SSM+SiLU/L2 kernel would expose concurrent global reads and writes across CTAs. The existing disjoint-range guard is necessary for correctness.

## DECISION

**NO CODE CHANGE; KEEP Exp062.** No additional performance or correctness claim is made.

## FOLLOW-UPS

A cooperative grid barrier after all SSM input reads could potentially order the aliased stores. Before implementing that path, verify cooperative launch support under CUDA Graph capture and confirm the entire grid can be resident on sm_86. Keep the current alias guard unless a race-free ordering is proven.

## IMPORTANT DISCOVERIES

- All 48 recurrent SSM/L2 sites at contexts 512 and 4096 have the same zero-offset `[128,32]` QK view and shapes.
- Exactly 24 L2 outputs are disjoint from SSM input; the other 24 alias the first 16 KiB of SSM input.
- Graph metadata shows alias safety, not view shape, is the constraint on expanding Exp062.

## ARTIFACTS

- `results/exp063/raw/site_classification_ctx512.json` and `site_classification_ctx4096.json`: all 48 detailed site records for each context.
- `results/exp063/raw/site_classification.tsv`: compact site index.
- `results/exp063/raw/metadata_capture*.stderr` and matching JSON: scheduler captures and run metadata.
- `results/exp063/raw/exp062_graph_adjacency.json` and `exp062_metadata_ctx512.json`: copied Exp062 context.
