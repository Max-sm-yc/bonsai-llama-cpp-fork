# Experiment 059: recurrent graph metadata and fusion feasibility

## Objective and decision

Collect runtime metadata for Qwen3.5's `CONCAT → CPY(cache update) → SSM_CONV → SiLU` chain on the RTX 3080 and decide whether a guarded two-output fusion is safe. Runtime metadata was collected, including the actual device pointers after allocation. The candidate was not implemented: the CUDA graph places unrelated recurrent state operations between the cache CPY and SSM_CONV, so the current contiguous fusion/skip mechanism cannot remove both target nodes without also skipping unrelated graph work or adding a graph-local side-effect protocol. **INCONCLUSIVE; no candidate correctness, node-count, or timing gate was reached.**

No production source was changed and no commit was created. Temporary scheduler and CUDA matcher logging was removed after capture. The manager checkout remains unchanged at `39c2a882767758983b5a29b7d278815295dafde6`.

## Runtime capture

The existing production executable initially printed only the CUDA device banner with `GGML_SCHED_DEBUG=2`; llama-bench's `-v` option is also needed to enable GGML debug log output. The production run used `-p 0 -n 1 -d 512`, PTQ1_0, 99 GPU layers, Flash Attention, batch 2048, ubatch 512, F16 KV, and 8 CPU threads. Concise scheduler rows are in `results/exp059/raw/production_ctx512.recurrent_nodes.txt`; the original verbose trace is `production_ctx512_verbose.stderr.log`.

The scheduler listing omits view nodes and only reports tensor sizes/use counts. A temporary isolated-worktree probe logged view metadata and backend ranges, then a second temporary probe logged tensors at `ggml_cuda_can_fuse`, after CUDA allocation assigned data pointers. Exact relevant rows, including duplicate graph constructions, are in `instrumented_chain_ctx512.stderr.log` and `cuda_probe_ctx512.stderr.log`. The one-token decode graph is identified by `conv_input-0` shape `[4,10240,1,1]`; the command also caused larger setup/prompt graph constructions, which are not used as one-token evidence.

## Standard one-token metadata

For the matching standard decode graph, the expanded node order is CONCAT `conv_input-0` (#22 in the CUDA probe graph), VIEW `conv_state_last-0` (#23), VIEW `conv_state_update-0` (#24), CPY `conv_state_update-0 (copy of conv_state_last-0)` (#25), followed by unrelated recurrent state work before SSM_CONV `conv_output_raw-0` (#33). The next SSM_CONV outputs are followed by the existing SiLU fusion.

| Tensor | Shape / strides | Runtime address and relevant bounds | Flags / use |
|---|---|---|---|
| `conv_input-0` CONCAT | F32 `[4,10240,1,1]`; `nb=[4,16,163840,163840]` | `0x7f2b75ce2880`, 163,840-byte contiguous output range | `0x10` COMPUTE only; use=2 |
| `conv_state_last-0` VIEW | F32 `[3,10240,1,1]`; `nb=[4,16,163840,163840]` | `conv_input` base + 4 bytes; strided source span ends at `conv_input` base + 163,840 bytes | `0x10` COMPUTE only; use=1 |
| `conv_state_update-0` VIEW / CPY destination | F32 `[30720,1,1,1]`; `nb=[4,122880,122880,122880]` | `0x7f2bb6000000` to `0x7f2bb601e000` (120 KiB); view offset 0 from runtime cache view `cache_r_l0` | VIEW `0x10`; CPY use=0 |
| `conv_output_raw-0` SSM_CONV | F32 `[10240,1,1,1]`; `nb=[4,40960,40960,40960]` | `0x7f2b748e2880` to +40 KiB | `0x10` COMPUTE only; use=1 |

The concat, cache destination, and conv output ranges are disjoint in the captured standard decode graph. The graph nodes have no `GGML_TENSOR_FLAG_OUTPUT` bit. The source view starts one F32 item (4 bytes) into the concat and retains its non-contiguous strides; it is not a contiguous 120 KiB source. Runtime `cache_r_l0` has shape `[30720,1,1,1]` and the target view offset is zero. This matches the builder's standard `n_rs_seq==0` branch and its one CPY. Rollback uses `K=n_rs_seq+1` CPYs and must remain on fallback.

The data addresses above come from the actual CUDA fusion eligibility path, not inferred pointer arithmetic. The scheduler-only pass itself runs before temporary concat storage has been allocated and reports `data=NULL` for CONCAT; this is why the second probe was necessary. The backend buffer range is broader than the individual tensor allocations, so individual ranges above are calculated from the runtime tensor address, byte strides, and logical dimensions. No overlap is present among these relevant tensors.

## Why no candidate was attempted

The CUDA graph's expanded order is not one contiguous fusion range. After the cache CPY, the graph performs separate SSM state scaling/gather/copy operations before the convolution node. A matcher beginning at CONCAT and skipping through the SiLU node would skip those unrelated state operations. The existing `ggml_cuda_try_fuse` mechanism can consume one contiguous node interval and return a skip count; it has no graph-local facility for executing the conv and cache side effects early, then suppressing only the later SSM_CONV/SiLU nodes while retaining intervening state updates. Executing at the later SSM_CONV cannot remove the earlier cache CPY. Executing at CONCAT without a node suppression protocol would leave duplicate CPY/SSM work, so it would not be the requested two-node removal.

This is an implementation/graph-execution safety obstacle, not missing runtime artifacts or unknown address aliasing. A follow-up would need a graph-local multi-output rewrite/skip contract that identifies the single CPY and SSM_CONV/SiLU consumers, preserves all intervening nodes, and explicitly rejects graph-output pins and multi-CPY rollback graphs. Only after that can the multi-output kernel be tested against both conv output and cache bytes over repeated updates.

## Validation and performance

The isolated worktree was configured Release with CUDA, CUDA FlashAttention, CUDA graphs, and `CMAKE_CUDA_ARCHITECTURES=86`; exact configure/build logs are in `results/exp059/raw/`. The instrumented CUDA library SHA-256 was `9546065804029b4f9742be4aed4142a69d77bb30fa38fb9640beed5e7de40e92`, and `ldd` resolved the isolated executable to its own `build/bin/libggml-cuda.so.0`. The source probes were then reverted; final worktree source has no diff. No candidate kernel was built, so no correctness test, model smoke, Nsight node-count comparison, focused timing, or E2E A/B was run. Existing best results are unchanged.

## Decision

**INCONCLUSIVE.** Runtime shape, strides, use counts, cache destination, output pins, and relevant non-aliasing bounds are established. Fusion cannot be accepted or timed until the CUDA graph executor can skip the target nodes without skipping unrelated work between them. Do not update best results.
