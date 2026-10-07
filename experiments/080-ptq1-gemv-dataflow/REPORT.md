# Experiment 080: PTQ1_0 GEMV dataflow challenge

## HYPOTHESIS

The one-column sm_86 `mul_mat_vec_ptq1_0_pt` still accounts for about 9.014/9.022 ms per token at contexts 512/4096. A new decomposition might eliminate or overlap per-block FP32 shared partials and their reduction while preserving compact weights, enough parallelism, and exact outputs.

## IMPLEMENTATION

I created isolated worktree `/home/maxsun/autonomous_projects/.worktrees/exp080-ptq1-gemv-dataflow` from manager commit `f4c8740d3307648626ad708671221c93ecab2083`. Required research state, idea list, current-best record, and reports Exp010, Exp024–046, Exp047, Exp052, Exp055–056, and Exp068, Exp073–079 were inspected before choosing a design.

The current kernel was traced from dispatch through the active `<1,1,false,false>` specialization, and its `mmvq.cu.o` was compiled for sm_86 from unchanged source. No candidate implementation survived the source, ordering, and resource challenge: the obvious partial-removal mapping serializes the four exact FP32 accumulation streams per row; spreading a stream across owners changes FP32 addition association; row/warp/CTA-width/reduction and alternate arithmetic families repeat measured experiments. The detailed path and work/order audit is in [`source_dispatch_audit.md`](../../results/exp080/raw/source_dispatch_audit.md).

Initial source hashes and commit are in [`source_hashes_baseline.txt`](../../results/exp080/raw/source_hashes_baseline.txt) and [`commit_initial.txt`](../../results/exp080/raw/commit_initial.txt). The worktree's production sources remain unchanged from the best implementation; `git diff --exit-code` against the current source passed. The only untracked files are this report, `results/exp080/`, and the isolated build output.

Reference build command and settings:

```sh
cmake -S . -B results/exp080/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=OFF
cmake --build results/exp080/build --target llama-bench -j 8
```

The full 441-step target build succeeded. The binary's `ldd` resolves the experiment-local CUDA, GGML, and llama libraries. Exact hashes:

- Active production header `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`: `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`.
- Dispatch source `ggml/src/ggml-cuda/mmvq.cu`: `e889b1543656cd2e5c0c6151f11bfdd7f641e88af441449d92064ac902b9483d`.
- Reference `llama-bench`: `aadddd4d72d48a61c7c8953880d3b5036b0b53e1542cb0353d1f18ffe84ecfc3`.
- Reference `libggml-cuda.so.0.21.0`: `7b94bcc9fabf068ecb3e60ddf0c84591855a615164e58ff5171a8958de56834a`.
- Compiled active `mmvq.cu.o`: `328b17791c01a240bf569b8e4bf03ae26030ed064909d662abfb5f70d98c6431`.

The complete source hash list, build config, commands, evidence hashes, and loader paths are in `results/exp080/raw/`.

## RESULT

**NO CANDIDATE.** The exact-order constraint limits straightforward reduction of the shared partial array to four sequential K-block streams per row (10 dots/stream at K=40; 34 at K=136). The production mapping instead exposes all block dots across 128-thread CTAs, then sums partials in its fixed four-stream order. Any alternative that parallelizes each stream further changes the FP32 addition grouping and would need numerical equivalence evidence; the already measured warp/reduction approaches offer no untested dataflow premise. No modified kernel was built or timed.

## CORRECTNESS

No candidate arithmetic was implemented, so there is no new candidate-vs-production or independent-reference result and no candidate path to sanitize. `tests/run_correctness.sh` was not run because no candidate was integrated. Production sources were not changed. The reference binary built successfully; this is not a new correctness claim.

## MICROBENCHMARK

No candidate microbenchmark was run. The isolated baseline sm_86 CUDA object and active SASS/resource evidence were generated from the reference build. For `<1,1,false,false>`, ptxas reports 76 registers/thread, 0 stack/spills, and 0 static shared bytes; the launcher supplies dynamic shared memory. Extracted active SASS contains 32 `IDP.4A` instructions, nine 128-bit activation loads, 217 shared loads, one shared store site, and a CTA barrier. This confirms the active packed-dot and shared-partial path; it does not provide candidate timing or occupancy measurements. Raw compact evidence is in `results/exp080/raw/active-kernel.sass.txt` and `baseline-resource-usage.txt`.

## END-TO-END IMPACT

Not measured because no candidate qualified for integration. The current-best standard-workload results remain 84.407 tok/s at context 512 and 81.885 tok/s at context 4096, with 6,579/6,803 MiB peak VRAM, as recorded in `research/STATE.md`. These are prior results, not Exp080 measurements. No arm-level rates or candidate peak VRAM exist.

## ANALYSIS

For the baseline, the one-column host selector maps each work item to one row and one 128-weight PTQ block. The CTA uses three rows at K=40 (120 independent block dots) and sixteen rows at K=136 (2,176 block dots over 17 full CTA waves). Every dot writes one FP32 partial to dynamic shared memory. After a CTA barrier, each row's epilogue accumulates indices `k mod 4` sequentially and folds `(s0+s1)+(s2+s3)`.

A four-owner-per-row design would preserve the four accumulation sequences only by making each owner process 10 or 34 block dots serially. That cuts the per-row dot parallelism substantially. Parallelizing those same sequence terms changes the FP32 operation association and is not an exact-output candidate without new evidence. Mapping one row per CTA or changing warp/CTA reduction geometry repeats prior row/CTA/reduction families; prior screens also rejected shared staging, async prefetch, activation reuse, alternate decoder formats, and cache policy. Exp078 already rejected persistent bit planes on payload growth and arbitrary-Q8 selection cost. Nsight Compute remains unavailable (`ERR_NVGPUCTRPERM`); permissions were not changed or retried.

## DECISION

**NO CANDIDATE.** Retain the current PTQ1_0 implementation and current-best result. No source change or commit was made; no cherry-pick or manager-checkout modification occurred.

## FOLLOW-UPS

Reopen only with an sm_86 primitive or compact weight representation that changes the per-weight work without adding payload/expansion traffic, or with an exact method that reduces the four ordered FP32 streams while retaining parallel K-block execution. A future candidate must first pass exact K=40/K=136 and tail tests before model correctness or E2E screening.

## IMPORTANT DISCOVERIES

- The actual one-column model path reaches the dedicated PTQ1_0 planar kernel only for the guarded plain 2D, no-ids, one-channel/one-sample shape; batch-1 selects `<1,1,...>`.
- Baseline generated code directly decodes compact trits into DP4A and has no spills; the reduction uses dynamic shared memory and one CTA barrier.
- Exact output order creates four sequential per-row streams. Reducing their owners to eliminate partials also eliminates most independent K-block work per row; distributing each stream changes FP32 association.
- The current 128-thread mapping already fills CTA work at K=136 and reaches 94% work-item coverage at K=40 with three rows/CTA. Obvious geometry alternatives are prior measured families.
- No source, correctness, microbenchmark, or E2E candidate change was produced; current best is unchanged.
