# Experiment 034: selective PTQ1_0 long-K SoA sidecar

## HYPOTHESIS

A seven-plane, no-padding SoA copy of only PTQ1_0 weights with K=17,408 could accelerate the batch-1 ROWS=1 GEMV while canonical AoS allocations continue to serve prefill, MMQ, multi-column MMVQ, conversion/dequantization, vector-dot, and get-rows. Exp032 found an exact 7.82% long-row block-dot improvement. Keeping the canonical weights plus these copies projects a 7,995 MiB peak from the 6,805 MiB baseline.

## IMPLEMENTATION

No candidate source edits or library builds were made. The code experiment used an isolated worktree at code commit `9fa97200e68fd798ef027470c8e420172a0ac719` (`/tmp/bonsai-exp034`, branch `exp034-selective-soa`); the manager checkout and source-default library were preserved. Source-path inspection is recorded in `results/exp034/source_path_audit.txt`.

The GEMV switch in `mmvq-ptq1_0.cuh` receives a raw weight pointer and integer dimensions/strides. It does not receive the tensor identity or the owning CUDA buffer context. Persistent sidecars need an owner tied to the backing CUDA allocation: `ggml_backend_cuda_buffer_context` is destroyed in `ggml_backend_cuda_buffer_free_buffer`, but the GEMV path has no registration/cleanup link to that context. A pointer-keyed cache without this cleanup can retain stale entries after buffer free and pointer reuse. Per-invocation conversion would need to repack the full 19.5 MB tensor repeatedly during decode and would add a temporary allocation and conversion kernel on the active path. The sound route is to thread an explicit sidecar owner/registry through the op dispatcher and free it with the buffer context; that requires crossing buffer lifetime and op dispatch, so this bounded GEMV-sidecar prototype stopped before introducing an unsafe cache.

The exact proposed runtime selector is type PTQ1_0, K=17,408, M=5,120, contiguous AoS row stride 136 blocks, one output column, one channel/sample, no ids, and no fusion. The required GGUF inventory has exactly 64 matching PTQ1_0 tensors, each 19,496,960 bytes. Inventory source and complete output are preserved from Exp033 in `results/exp033/count_selective_soa_bytes.py` and `results/exp033/selective_soa_bytes.txt`.

## RESULT

No sound runtime candidate was produced. This is a source ownership/lifetime integration blocker for the bounded implementation, not evidence that a context-owned sidecar design cannot work.

## CORRECTNESS

No new runtime correctness gates were run because no candidate kernel or storage path was built. Canonical AoS behavior was unchanged. Exp032's standalone SoA dot checks and sanitizer result remain prior-art evidence only; they do not validate runtime integration.

## MICROBENCHMARK

No new microbenchmark was run. Exp032 manager rerun remains the only relevant measurement: exact output, -0.22% at 40 blocks/row and -7.82% at 136 blocks/row.

## END-TO-END

Not run. There was no candidate library for fixed-seed model output comparisons or matched prefill/decode A/B at contexts 512 and 4096. Current best baseline remains 82.2218 tok/s at 512 and 79.6967 tok/s at 4096 under the documented 7 x 128-token setup.

## VRAM / LOAD COST

No allocation or load-time was measured. The inventory reports 1,247,805,440 sidecar bytes (1,190.00 MiB) for exactly 64 tensors, yielding a projected total peak of 7,995 MiB against the 10,240 MiB card. This is an estimate only; AoS remains the current runtime allocation and actual peak was not sampled.

## ANALYSIS

The shape/type registry avoids model-name dispatch and the local GGUF proves the intended 64 tensors. However, eligibility alone is insufficient for persistent lookup: the active hook sees only `vx` and launch dimensions. To safely cache a transformed allocation, an explicit owner lifetime must reach the hook, or another context-owned sidecar registry must be introduced. Existing buffer free already has the right ownership boundary, but the GEMV hook cannot currently register its copy there. Keeping AoS is correct for every existing consumer; only the intended batch-1 dispatch would change after the registry and SoA reader exist.

## DECISION

**STOP BEFORE CANDIDATE / INCONCLUSIVE.** Do not integrate a pointer-keyed cache without buffer-owned cleanup. Preserve current ROWS=1 source/default library. A follow-up can be sound if it first adds a buffer-context-owned sidecar registry, then verifies ownership and exact eligibility before measuring conversion/load cost and runtime.

## FOLLOW-UPS

- Add an explicit CUDA buffer sidecar owner/registry with cleanup in `ggml_backend_cuda_buffer_free_buffer`.
- Pass that owner and an exact shape/type eligibility record to the PTQ1 dispatch, or register sidecars at load time.
- Convert once into the 7-plane representation and dispatch only the batch-1, one-column, ROWS=1 plain GEMV. Keep all other readers on AoS.
- Then test selected CTests, CUDA-vs-CPU PTQ1_0/PQ2_0 cases, multi-column/prefill, fixed-seed output, measured peak/load time, and matched E2E A/B.

## IMPORTANT DISCOVERIES

- The exact GGUF inventory has 64 PTQ1_0 tensors of shape `[17408, 5120]`; this is sufficient to define a shape/type selector without model-name shortcuts.
- The dedicated PTQ1 dispatch does not receive `ggml_tensor` or its owning CUDA buffer context.
- CUDA buffer destruction is an explicit lifecycle boundary, but no sidecar cleanup hook is connected to it from the GEMV path.
- Source-default source and library were preserved; no model, correctness, or performance claim was added.

## Manager source-path follow-up

The public `ggml_cuda_mul_mat_vec_q` entry point does have `src0` before it calls `mul_mat_vec_q_switch_type`; the tensor identity is discarded only when the raw-pointer type switch is called. The buffer context and `free_buffer` owner are available in `ggml-cuda.cu`. This makes a narrow safe prototype plausible: add a helper that looks up a sidecar from `src0`'s owning CUDA buffer, thread its pointer through the PTQ1 type switch, and release sidecar allocations with that buffer context. Register only after a complete tensor upload. This is an implementation path to test, not evidence that loader callbacks always provide full uploads or that graphs/lifetimes are safe.
