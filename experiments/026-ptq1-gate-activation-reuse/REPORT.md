# Experiment 026: PTQ1_0 gated activation-load reuse audit

## HYPOTHESIS

The active sm_86 `<ncols=1, ROWS=1, has_fusion=true, has_gate=true>` GEMV calls `ptq1_0_pt_block_dot` for main and gate weights. If the compiler emitted duplicate planar Q8_1 activation loads, a paired dot helper might reuse those fragments and reduce global traffic.

## IMPLEMENTATION

No code change was made. I extracted the embedded sm_86 SASS from the current source-default `build/bin/libggml-cuda.so.0` with `cuobjdump --dump-sass`, then isolated the exact `<1,1,true,true>` and `<1,1,true,false>` function bodies. Extracted SASS and concise global-load listings are in `results/exp026/`.

The relevant source calls and shared `ycol` activation pointers are in `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`; the active function computes the main dot and then the gate dot. The benchmark control JSON and captured `llama-bench` stdout are `results/exp026/control.json` and `results/exp026/control.stdout.json`.

## RESULT

**No duplicated activation vector loads were found.** The gated function contains 32 total `LDG` instructions and the ungated function 20, but both contain exactly nine `LDG.E.128` instructions. The extra gated scalar loads are consistent with reading the second set of PTQ1_0 weight metadata/packed data; they do not indicate another set of 128-bit Q8_1 activation-fragment loads. The activation vector load count therefore does not double for the gate dot. This matches compiler reuse/hoisting of the shared activation fragments across the two inlined dot computations.

The current behavior fails the bounded optimization precondition, so the paired-helper implementation path was stopped. No candidate build, microbenchmark, or candidate correctness run was made.

## CORRECTNESS

No production code was changed or built. Correctness tests were not run. The unchanged active library and source match the recorded baseline hashes below.

## MICROBENCHMARK

Not run; there is no candidate. SASS evidence: `<1,1,true,true>` has 32 total / 9 128-bit `LDG`; `<1,1,true,false>` has 20 total / 9 128-bit `LDG`. Full isolated function bodies and extracted load lists are under `results/exp026/`.

## END-TO-END IMPACT

A fresh seven-repetition control was run before any candidate build using the requested command and settings. Medians were 81.5795 tok/s at context 512 and 79.0749 tok/s at 4096. No candidate comparison was run because the activation-load duplication hypothesis was not supported by SASS.

## ANALYSIS

The raw `LDG` total alone is misleading because gating adds weight-stream loads. The relevant comparison is the vector activation traffic: each specialization emits nine 128-bit global-load instructions. The gated specialization has no second nine-instruction activation vector-load set. With the compiler already sharing these fragments, a paired helper has no demonstrated activation-traffic reduction to pursue in this experiment.

## DECISION

**REVERT / stop implementation path.** No source or binary changes were made, so no restore operation was needed. Source SHA-256: `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`. Active library SHA-256: `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`. Runtime commit remains `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`; no commit was made.

## FOLLOW-UPS

Proceed to a different measured bottleneck. Revisit this path only if a future compiler/toolchain build disassembly demonstrates duplicate activation vector loads.

## IMPORTANT DISCOVERIES

- The source calls the shared block-dot routine twice, but the active gated SASS does not emit twice the 128-bit activation loads.
- `<1,1,true,true>` adds 12 total `LDG` instructions over `<1,1,true,false>`, while the number of 128-bit `LDG.E.128` instructions remains nine in both kernels; raw instruction totals must be separated by data stream before inferring duplicate activation traffic.
- Control medians were 81.5795 / 79.0749 tok/s at contexts 512 / 4096, with 6,805 MiB peak GPU memory.

## MANAGER VERIFICATION

On 2026-10-06, the manager independently counted nine `LDG.E.128` instructions in each saved specialization, rechecked both source/library hashes above, and confirmed that `build/bin/libggml-cuda.so` and `.so.0` are byte-identical to the recorded active library. The full SASS and per-specialization load listings are preserved under `results/exp026/`. No implementation or correctness claim is made because no candidate was built.
