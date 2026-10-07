# Experiment 048: QKV RMS reduction sharing

## HYPOTHESIS

Exp036's `fwht_rms_quantize_q8_1` launches five CTAs for a 5120-wide row, one per 1024-wide FWHT tile. Every CTA independently reduces the whole row to compute the same RMS scale. Splitting RMS work across those CTAs and sharing the result could remove repeated reduction work while preserving transform parallelism.

## IMPLEMENTATION

The candidate was implemented only in detached worktree `/tmp/exp048-qkv-rms-sharing`. It uses five CTAs per exact 5120-element row. Each CTA reduces its 1024-element tile and writes one partial sum into four otherwise-temporary Q8 bytes in its tile's output region. A cooperative grid barrier publishes the partials; each CTA reads the five partials, stores the resulting sum in shared memory, and a second cooperative grid barrier ensures all CTAs have finished reading the markers before any CTA overwrites output bytes. Each CTA then runs its original 1024-wide FWHT/Q8 tile. The exact shape guard also requires `ne0 == 5120` and five x-grid CTAs. The launcher queries cooperative support and occupancy and only launches cooperatively when all grid CTAs fit the computed residency limit; otherwise it falls back to the original kernel. `GGML_CUDA_RMS_SHARE=0` selected the existing implementation for control measurements.

The output scratch lifetime is bounded to this kernel. The Q8 output is fully overwritten before return, and the existing graph path detects when its output overlaps the RMS input (`x`) and redirects to a held pool allocation. No separate scratch allocation or launch was added. The Exp036 graph/use-count guards and `GGML_CUDA_RMS_FWHT_Q8=0` disable override were untouched. The candidate launch was captured and replayed as CUDA graph nodes successfully.

## REQUIRED AUDIT

The production starting point was HEAD `aad463854d153e43eb4a0a3ab64dde6e245cf460`; the source tree and active build outside this detached worktree were untouched. Direct numerical coverage is `tests/test-fwht-rms-q8.cpp`: one and three rows at width 5120, PT Q8 layout, dequantized error, stored sums, and unsupported shape/layout checks. The existing host-reference limits are max error/scale 0.75 and exact stored sums.

Exp019 rejected single-branch RMS fusion because the normalized activation is shared across Q/K/V. Exp020's FWHT CTA-width sweep was not repeated. Exp047's graph profile contains 255 one-token replays with 1,432 nodes/replay. At context 512, the current fused kernel appears 20,561 times (3,746 ns mean, 3,712 ns median); the QKV-preparation family totals 0.7524 ms/token. At 4096 the family is 0.7528 ms/token. Compact raw profile exports are in `results/exp048/exp047_*`.

## CORRECTNESS

Control and candidate produced byte-identical packed PT output for one and three rows: 0 differing bytes out of 5,760 and 17,280 bytes, respectively. Both also passed the independent host reference: max error/scale 0.539778 (one row) and 0.542589 (three rows); stored block sums were exact. `compute-sanitizer` memcheck reported 0 errors and racecheck reported 0 hazards/errors/warnings. Candidate CUDA graph capture and replay passed for both shapes.

## MICROBENCHMARK

Timings use repeated CUDA events on the actual 5120-wide kernel launch. Graph timings capture 100 actual kernel nodes and replay the captured graph five times per sample. Values are medians over five samples; ranges are retained in `results/exp048/*_timing.txt`.

| Rows | Control direct event | Candidate direct event | Control graph replay per node | Candidate graph replay per node |
|---:|---:|---:|---:|---:|
| 1 | 4.543 μs (4.542–4.547) | 5.509 μs (5.499–5.519) | 3.845 μs (3.844–3.848) | 4.798 μs (4.798–4.800) |
| 3 | 4.588 μs (4.586–4.588) | 5.540 μs (5.538–5.548) | 3.871 μs (3.869–3.873) | 4.808 μs (4.808–4.811) |

The graph replay path is slower by about 25% in both row cases. At 80.6 such instances per token, the one-row graph delta is about +0.077 ms/token for this kernel family. The repeated host-launch event is also slower by 21–22%; the cooperative launch and occupancy-query overhead is visible there, but the graph replay result independently establishes a kernel-level regression.

## ANALYSIS

On sm_86, `cuobjdump --dump-resource-usage` reports 24 registers/thread, 4,224 bytes shared, and no stack/local spills for the current kernel; the cooperative candidate reports 25 registers/thread with the same shared memory and no spills. The active transform geometry remains five 1024-thread CTAs per row. The candidate adds two grid-wide barriers (partial publication and marker-read completion) and reads five partial values per CTA. Its compact SASS excerpt shows the global barrier/fence sequences and partial loads in `results/exp048/kernel_sass_excerpt.txt`; resource output is in `results/exp048/kernel_resources.txt`. It removes four fifths of the redundant RMS input reduction, but the two grid barriers and cooperative launch dominate the measured kernel time. Candidate test linkage was verified with `ldd` to resolve to `/tmp/exp048-qkv-rms-sharing/build-exp048/bin/libggml-cuda.so.0` (SHA-256 `c9651e3e9a7cfa2fe6fb7ff00b0704a3447f0b2144aee45e0084a2655f93f51d`). The active manager library remains SHA-256 `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`.

## RESULT

Cooperative cross-CTA RMS sharing is correct for the tested one- and three-row cases and can be captured in a CUDA Graph, but it loses on the target kernel. The two barriers needed to safely publish/read output-backed partials cost more than the removed repeated RMS work.

## END-TO-END IMPACT

No model smoke or end-to-end decode A/B was run. The focused graph-replay screen regressed consistently by about 25%, so the candidate was rejected before model integration. The production source and active CUDA library remain unchanged.

## DECISION

**REVERT.** Exact packed output, numerical tolerances, sanitizer checks, and CUDA graph capture all pass, but focused graph replay is consistently about 25% slower. No full correctness suite, model smoke, or end-to-end decode benchmark was run because the focused screen rejected the candidate. All CUDA and test-source edits were reverted in the isolated worktree; there is no production source or binary change. Current best remains unchanged.

## FOLLOW-UPS

Keep Exp036 unchanged. GDN at about 0.500 ms/token is the next measured secondary family. Reopen RMS sharing only with a different synchronization design that removes the cross-grid barrier cost while retaining exact output and the five-way FWHT parallelism.

## IMPORTANT DISCOVERIES

- Reusing output bytes as partial-sum scratch requires two cooperative grid barriers: one to publish all partials and another to ensure every CTA has read them before any CTA overwrites its output tile.
- Cooperative graph capture works for this guarded shape, but graph replay is about 25% slower than the ordinary launch path despite byte-exact output and no sanitizer findings.
