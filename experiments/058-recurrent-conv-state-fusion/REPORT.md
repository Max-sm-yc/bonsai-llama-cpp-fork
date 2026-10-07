# Experiment 058: recurrent convolution and cache state fusion

## Hypothesis and decision

A one-token recurrent decode kernel could replace the Qwen3.5 `CONCAT → CPY(cache update) → SSM_CONV → SiLU` chain if one dispatch writes both the convolution-plus-SiLU output and the exact recurrent cache destination. This experiment stopped at the implementation safety gate. **INCONCLUSIVE; no candidate source was built and no performance claim is made.** The worktree source remains unchanged from manager HEAD `8a3303e`.

The direct `CONCAT → SSM_CONV → SiLU` matcher from Exp057 cannot apply: the expanded graph inserts a required CPY. A more capable matcher is plausible, but needs an explicit multi-output/side-effect contract and graph validation beyond the current fusion helper, which reasons about a contiguous operator range and designated outputs.

## Required source and graph mapping

`llm_build_delta_net_base::build_conv_state` constructs `conv_states = build_rs(...)`, reshapes history to `[d_conv-1, conv_channels, n_seqs]`, transposes the new QKV row, and creates `ggml_concat(..., dim=0)`. For the one-token model path, the Qwen3.5 metadata recorded by Exp057 gives `d_conv=4`, `conv_channels=10,240`, and a one-token input: the concatenated logical shape is `[4, 10,240, 1, 1]` (F32 in the measured graph).

The `n_rs_seq == 0` path creates a source view beginning at concat row `s_idx=1`, covering the last three rows, and a cache destination view at byte offset `(0 * mem_size + kv_head) * row_size`, where `row_size = ggml_row_size(F32, 3 * 10,240)`. The CPY therefore writes the newest three-row history for the active sequence/slot. For rollback (`n_rs_seq != 0`), the graph emits `K=n_rs_seq+1` CPYs to distinct slots; this path must remain unfused unless separately proven safe.

The concat has two graph-level consumers: the SSM_CONV input and the view used by the CPY. SSM_CONV is followed by SiLU and existing CUDA dispatch already fuses those two into one convolution kernel. The current scheduler's `ggml_cuda_can_fuse` checks contiguous node patterns; dispatch starts at SSM_CONV, not CONCAT. Exp057's node-level profile observed 48 concat and 48 SSM_CONV launches per replay, consistent with the 48 recurrent blocks, and 1,432 total nodes/replay. CPY's profile signature was aggregated across recurrent conv and GDN cache writes, so it does not identify the conv CPY instance or its exact node position/use count.

This gives the necessary arithmetic and intended cache range, but not enough evidence to safely move the mutation to concat dispatch. In particular, the retained profile exports do not preserve the expanded graph's per-node view metadata, cache allocation/alias ranges, direct and view use counts, or whether any graph output pins an intermediate. A matcher would have to prove these for every invocation and reject rollback/split sequences; broadening the existing matcher based only on shapes would be unsafe. No instrumentation or candidate matcher was added in this bounded screen.

## Existing baseline evidence

The closest valid node-level captures are Exp057's baseline Nsight Systems captures, made with the same PTQ1_0 model and one-token decode graph-replay method. They are retained under `results/exp057/raw/` in this worktree. Exp057 observed:

| Context | Graph nodes/replay | CONCAT calls / mean time | SSM_CONV calls / mean time |
|---:|---:|---:|---:|
| 512 | 1,432 | 48 / 0.099527 ms | 48 / 0.073924 ms |
| 4096 | 1,432 | 48 / 0.099680 ms | 48 / 0.074098 ms |

These are baseline measurements only. There is no Exp058 candidate capture, candidate node count, focused graph timing, correctness case, CTest run, or model smoke. Because no candidate could be justified from available graph evidence, the required implementation and subsequent gates were not reached. No E2E A/B was run.

## Correctness and state semantics

No modified CUDA code exists to compare. The existing unfused order is semantically significant: the CPY commits the rolling history into recurrent state before later recurrent graph operations consume cache state. A valid fused kernel would need to reproduce both the concatenation's convolution input and the CPY's strided destination write, including repeated one-token updates, while proving that destination writes cannot overlap still-needed input data. It must also preserve the multi-CPY rollback path and non-decode sequence paths by falling back unchanged.

A future implementation should first add graph instrumentation exposing the exact concat/CPY/SSM_CONV/SiLU node indices, tensor strides, data pointers/allocation ranges, graph output flags, direct/view use counts, `n_rs_seq`, `n_seqs`, `n_tokens`, and cache-view byte ranges. Only then add an explicitly guarded multi-output matcher and a CUDA test that compares both conv output and cache bytes over repeated updates and boundary-supported shapes. The graph replay gate should require one fewer concat and one fewer CPY node per recurrent block while retaining the fused SSM_CONV result, followed by matched E2E only if focused device time improves beyond noise.

## Environment and source status

The experiment worktree is `/home/maxsun/autonomous_projects/.worktrees/exp058-recurrent-state-fusion`, based on `8a3303e`; it is isolated from the main checkout. RTX 3080 gate at inspection was 47 C and 0% utilization. No source files were modified, no build or measurements were performed, and no commit was created. The main checkout and current best remain untouched.
