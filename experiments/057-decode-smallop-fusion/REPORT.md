# Experiment 057: decode small-op fusion screen

## Objective and decision

Map the five high-frequency context-512 “Other” signatures from Exp053 to Qwen3.5 PTQ1_0 graph work, then screen one low-risk fusion. The measured 0.584 ms/token total is an upper bound only.

The selected candidate was to bypass the recurrent convolution-input concat and read its two inputs directly in the existing SSM-convolution-plus-SiLU kernel. The standalone CUDA graph case passed, but the production Qwen graph places a required recurrent-cache CPY between CONCAT and SSM_CONV. The candidate matcher therefore did not fire in model decode. The capture retained 48 concat kernels and 1,432 nodes/replay, with no graph-time benefit. **Reject this matcher and keep source restored.** No full model A/B was warranted.

## Hardware and baseline

RTX 3080 (sm_86), CUDA 13.2.86, driver 580.178.04, host i7-10700K. The isolated worktree was based on manager HEAD `a75b61031fdc302d5d6470d675e4fbef4f832d41`; production source was restored before finishing. Main checkout remained untouched.

The manager baseline executable and CUDA library were SHA-256 `81187ab3fc4aeda74f92b08ca21ad774d74d1418fb2467d278b41dfe8dcdab13` and `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`. The candidate loader trace resolved `libggml-cuda.so.0` from the isolated `build/bin`; CUDA runtime/cuBLAS came from `/usr/local/cuda/lib64`. Candidate test library before source restoration was SHA-256 `61457a53629758a3cf248340996c47d7f94a3c3b9233f46779c257772a1a954f` during the initial diagnostic capture. The final restored source rebuild produced `build/bin/libggml-cuda.so.0` SHA-256 `a17c411718bc13da55b13cc1333ad5d0434e1d09683a478016d460508b9dbec1`.

## Graph mapping evidence

The PTQ1_0 GGUF metadata gives 64 Qwen3.5 blocks, hidden width 5120, convolution width 4, SSM state size 128, group count 16, time-step rank 48, and inner width 6144. Thus a recurrent convolution input has 10,240 channels (`6144 + 2*16*128`), with a 3-by-10,240 cached history and one new 10,240-wide QKV row.

| Measured signature (ctx 512) | Source graph operation and chain | Mapping / limits |
|---|---|---|
| `cpy_scalar<&cpy_1_scalar<float,float>>` — 112 calls, 0.209 ms | F32 `GGML_OP_CPY` cache writes built in `build_conv_state` and `build_recurrent_attn`. Conv-state write copies the last 3×10,240 history/input values to the recurrent cache; GDN writes its new state or rollback snapshots to the SSM cache. | These are state mutations, so removing or delaying them needs matching cache-write semantics. The aggregate signature does not separate the distinct CPY sites. |
| `concat_cont<unsigned int>` — 48 calls, 0.100 ms | `build_conv_state`: `CONCAT(conv_states, transpose(qkv_mixed), dim=0)` creates `[4,10240,1,1]`, consumed by SSM_CONV and then SiLU; output views split Q/K/V for GDN. | 48 calls match the recurrent-attention blocks. Although the input-copy kernel is contiguous, `build_conv_state` also expands a CPY of a view of this concat into the forward graph before returning. This required cache write separates CONCAT from SSM_CONV in the actual node order. |
| `unary_gated_op_kernel<&op_silu,float>` — 72 calls, 0.096 ms | CUDA `UNARY(SILU) → MUL` fusion (`ggml_cuda_op_unary_mul`); the model uses SiLU gating in feed-forward projections. | The kernel writes the gated projection used by the down projection. Qwen has 64 blocks; the retained profile signature does not label its extra eight sites or tensor shapes. |
| `k_get_rows_float<float,float>` — 50 calls, 0.093 ms | `GGML_OP_GET_ROWS` F32 gather. Qwen’s recurrent state builder gathers selected convolution-state rows; the last-layer narrow-output path gathers both attention output and residual (`qwen35.cpp`, lines 212–218). | The two last-layer 5120-wide gathers are followed by residual ADD. The remaining calls include recurrent state-row gathers. Signature-only traces do not expose per-node dimensions or consumer IDs. |
| `k_bin_bcast<&op_add,float,float,float,...>` — 49 calls, 0.086 ms | F32 `GGML_OP_ADD` broadcast kernel; Qwen uses it for residual and recurrent gate/state arithmetic. | One source-visible chain worth mapping next is the two last-layer GET_ROWS results followed by the hidden-width residual ADD. A fused gather-plus-add would need to preserve the selected row and ordering for both operands. |

Source references: `src/models/qwen35.cpp` lines 205–219, 476–503, and `src/models/delta-net-base.cpp` lines 468–501 and 535–568. Architecture values and derived dimensions are archived at `results/exp057/raw/model_architecture.json`. Raw signatures are from Exp053’s `results/exp053/raw/base_ctx512.profile.json` and `base_ctx4096.profile.json`.

## Implementation and correctness

The temporary patch added a CUDA kernel that read the convolution history and new QKV row directly, plus a scheduler matcher for CONCAT→SSM_CONV→SiLU. It was guarded to one-token F32 decode with the supported convolution shapes. A focused backend-op case exercised a direct CONCAT→SSM_CONV→SiLU graph with history `[3,1024]`, current row `[1,1024]`, and weights `[4,1024]`.

- Selected CTests passed 5/5: `test-quantize-fns`, `test-ptq1_0-element-map`, `test-ptq1_0-cuda-dot`, `test-pq2-row-shapes`, and `test-fwht-rms-q8`.
- New CUDA backend-op case passed 1/1 on CUDA0; CPU was skipped by the operation-test invocation. Command: `build/bin/test-backend-ops test -b CUDA0 -o SSM_CONV_CONCAT_SILU -p '.*'`.
- Fixed-seed 32-token PTQ1_0 smoke completed on both binaries. Generated bodies matched after removing build ID and prompt/generation timing lines. JSON outputs are `results/exp057/raw/model_smoke_base.json` and `model_smoke_candidate.json`; loader logs are under `results/raw/20261007T095132Z_PTQ1_0_smoke.stderr.log` and `20261007T095135Z_PTQ1_0_smoke.stderr.log`.
- The candidate source patch is preserved at `results/exp057/raw/concat_conv_attempt.patch`. Source files in the worktree match production after rejection.

## Focused graph screen

Captures used the same `llama-bench -p 0 -n 16 -r 2` node-level Nsight Systems method as Exp053. The command template for each arm/context was:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --cuda-memory-usage=true --force-overwrite=true --output results/exp057/raw/{arm}_ctx{context} \
  {ABSOLUTE_LLAMA_BENCH} \
  -m /home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 \
  -r 2 -o json -p 0 -n 16 -d {context}
```

`base_ctx512` and `base_ctx4096` used the manager binary; `cand_ctx512`, `cand_ctx4096`, and `cand3_ctx512` used the isolated candidate binary. The GPU start gate was met (47°C/0% at the first paired capture; subsequent captures remained under 60°C and 5%). Each valid report contains 31 complete replay groups.

| Context | Arm | Nodes/replay | CONCAT calls/time | SSM_CONV calls/time | Summed kernel time |
|---:|---|---:|---:|---:|---:|
| 512 | baseline | 1432 | 48 / 0.099527 ms | 48 / 0.073924 ms | 11.839205 ms |
| 512 | candidate (`cand3`) | 1432 | 48 / 0.099501 ms | 48 / 0.074002 ms | 11.847991 ms |
| 4096 | baseline | 1432 | 48 / 0.099680 ms | 48 / 0.074098 ms | 12.197669 ms |
| 4096 | candidate | 1432 | 48 / 0.099500 ms | 48 / 0.074150 ms | 12.198074 ms |

These are no-op captures: the unchanged node and call counts show the direct matcher never replaced the production graph chain. The small total-time differences are noise and are not an optimization result. `cand2_ctx512` is retained but invalid: an early matcher probe called `ggml_get_unary_op` on a non-unary node and aborted; it was fixed before `cand3` and before the 4096 final captures. `cand_debug_ctx512` and `diag.*` preserve the one-shot diagnostic attempt. No full-model A/B was run.

## Analysis and follow-ups

The selected producer-consumer chain was not adjacent in the expanded graph because cache mutation is explicitly expanded between building the concat and constructing the convolution. Fusing the concat away safely would require changing the cache-copy path as part of the same operation chain, which is outside this small screen. The proposed implementation is rejected, and its source was restored.

The next graph-local candidate is the final-layer pair `GET_ROWS(attention)` + `GET_ROWS(residual)` + `ADD` (hidden width 5120), which could combine two indexed reads and their residual sum. Its maximum benefit is limited to those last-layer nodes; measure the exact graph adjacency and a dedicated CUDA-vs-CPU case before integrating. Do not infer a 0.584 ms/token gain from the five-signature sum.

## Exact build and analysis commands

```bash
mkdir -p .cuda-tmp
TMPDIR="$PWD/.cuda-tmp" cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_COMPRESSION_MODE=size -DGGML_CUDA_NCCL=ON \
  -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_CUDA_FORCE_MMQ=OFF
TMPDIR="$PWD/.cuda-tmp" cmake --build build --target llama-bench llama-cli -j 8

nsys export --type sqlite --force-overwrite=true --output results/exp057/raw/{name}.sqlite \
  results/exp057/raw/{name}.nsys-rep
python3 results/exp052/analyze.py results/exp057/raw/{name}.sqlite
```

The successful CUDA build used `mkdir -p .cuda-tmp` before building. The initial build invocation without that directory failed at the first nvcc intermediate-file write; it was rerun successfully with the required worktree-local TMPDIR. The current source worktree has no source diff; only this report and `results/exp057/` artifacts remain untracked. No commit was created.
