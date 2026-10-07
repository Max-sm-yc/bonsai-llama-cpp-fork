# Experiment 055: paired K/V PTQ1_0 GEMV

## HYPOTHESIS

The standard-attention K and V projections in Qwen3.5 use the same normalized `cur` activation and both produce 2048 values for this model (8 KV heads × 256 elements). A dedicated one-launch paired-output PTQ1_0 GEMV could remove one of two separate decode GEMV launches while retaining distinct K and V buffers. K normalization/RoPE and V cache consumers must remain unchanged. Q+gate is already produced by `wq` and is outside this experiment.

## IMPLEMENTATION

No candidate implementation was made. The audit confirms the graph order is K `MUL_MAT`, V `MUL_MAT`, followed by K normalization. The active PTQ1_0 planar GEMV in `mmvq-ptq1_0.cuh` takes one weight pointer and one output pointer; its optional second dot product is specifically a nonlinear gate path and is not a valid V output mechanism.

A scheduler-level pair is feasible only if it also owns activation preparation. The active `ggml_cuda_mul_mat_vec_q` path selects PTQ1_0's Q8 layout, allocates/aliases its quantized activation representation, and dispatches the one-output kernel. The graph scheduler's fusion hook runs above that path. Skipping both graph nodes without introducing an equivalent shared Q8 preparation plus a true two-output PTQ dispatch would either read the wrong activation layout, duplicate launches, or alter output semantics. No shortcut was accepted as a candidate. This isolated worktree is based on manager HEAD `3416f373441f42d17d3c5c0c1aa8e7dbfc88971b`; no production files were changed. Source hashes: `mmvq-ptq1_0.cuh` `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`, `mmvq.cu` `e889b1543656cd2e5c0c6151f11bfdd7f641e88af441449d92064ac902b9483d`, `ggml-cuda.cu` `aee803e29b853a70d3e5274606cad32cdefc93d72578b1824df259dd2ba86351`, `qwen35.cpp` `056ae5e71776e1cf54d7d3eb48f34eeafa3a6d7eae2f9130044585b67c06625a`.

## RESULT

No candidate binary was built. The requested separate-output paired dispatch was not implemented, so correctness and performance comparisons could not be run. The audit narrows the required design: graph matching must prove exact PTQ1_0 matrices, same contiguous batch-1 activation, matching 2048-row output dimensions, adjacent compatible graph nodes, and non-overlapping separate output storage; dispatch must prepare the existing PT layout once and write both destinations. All other shapes/types and unsupported consumers must fall back.

## CORRECTNESS

Not run: there is no candidate. Baseline correctness records remain those in Exp053/Exp054. No numerical tolerance is claimed. Required future coverage includes direct paired GEMV checks across representative K block counts and output rows, including boundary rows, then selected CTests, CUDA-vs-CPU backend ops, and normalized fixed-seed PTQ1_0 output comparison.

## MICROBENCHMARK

Not run. No paired kernel or candidate model graph exists. The reference steady-state profile from Exp047/Exp054 measures 361 PTQ1_0 GEMV launches per token and 9.014/9.022 ms/token total GEMV at contexts 512/4096, but it cannot attribute launches to K and V individually.

## END-TO-END IMPACT

Not measured. The verified baseline remains 83.35 tok/s at context 512 and 80.35 tok/s at 4096 with approximately 6,803 MiB peak GPU use. There is no candidate throughput or memory result.

## ANALYSIS

K and V are adjacent graph matmuls with identical activation source and compatible dimensions, but the low-level dispatch does not expose a paired-output interface. The existing gate specialization accumulates a second matrix only to feed a GLU epilogue into the first result; it cannot preserve an independently addressable V output. A useful candidate needs a new kernel signature and a dispatch path that coordinates Q8 activation layout/preparation, graph-node skipping, and both output allocations. Reusing the current fusion hook without those changes is unsafe. The experiment therefore did not claim a launch saving or extrapolate from the total GEMV count.

The worktree has no pre-existing build directory. The RTX 3080 was available at 47 C / 0% utilization / 173 MiB use during the audit, but no CUDA build was attempted because there was no implementation to build. No benchmark start gate or binary/library SHA comparison was applicable.

## DECISION

**INCONCLUSIVE.** No candidate source diff, build, correctness result, microbenchmark, or end-to-end A/B exists. The isolated worktree production source remains unchanged and no commit was made.

## FOLLOW-UPS

- Prototype a true paired-output planar GEMV together with a paired dispatch that shares PT-layout Q8 preparation and preserves two output buffers.
- Add exact graph and shape guards with ordinary per-node fallback before testing any candidate.
- Add direct numerical tests for K block-count/row boundaries before model smoke or performance runs.
- Attribute named K/V projection calls in a model graph capture to estimate the per-token launch opportunity; existing family signatures do not identify projection names.

## IMPORTANT DISCOVERIES

- Qwen3.5 standard-attention K and V graph nodes are adjacent and consume the same `cur`; their graph order leaves K normalization after both projections.
- The concrete model shape is K=5120 and output rows=2048 for both projections.
- Active PTQ1_0 batch-1 planar GEMV is selected below the graph scheduler and currently accepts one output matrix plus an optional gate matrix.
- The gate matrix is not semantically interchangeable with V: the kernel folds it through a GLU operation instead of writing a second output tensor.
- Any one-launch candidate must coordinate the existing PT-layout activation preparation and paired output writes; a graph-only skip is not sufficient.

## EXACT COMMANDS AND RAW ARTIFACTS

Worktree creation:

```bash
git worktree add /home/maxsun/autonomous_projects/.worktrees/exp055-ptq1-kv-pair \
  -b exp055-ptq1-kv-pair 3416f373441f42d17d3c5c0c1aa8e7dbfc88971b
```

Source audit:

```bash
sed -n '327,380p' src/models/qwen35.cpp
sed -n '301,560p' ggml/src/ggml-cuda/mmvq-ptq1_0.cuh
sed -n '1160,1240p' ggml/src/ggml-cuda/mmvq.cu
sed -n '3600,3680p' ggml/src/ggml-cuda/ggml-cuda.cu
```

No candidate build, model correctness, benchmark, or loader command was run. `results/exp055/raw/` exists but contains no candidate artifacts. Reference profiles and baseline measurements are documented in `results/exp054/raw/` and `experiments/047-steady-decode-profile/REPORT.md`.
