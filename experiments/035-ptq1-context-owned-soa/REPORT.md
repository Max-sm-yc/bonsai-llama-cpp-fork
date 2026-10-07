# Experiment 035: buffer-owned PTQ1_0 selective SoA

## Decision

**REVERT; no candidate.** The selective sidecar path is now proven correct and reaches the actual batch-1 PTQ1_0 GEMV, but it does not produce a repeatable decode gain. It increases measured peak GPU memory by about 1.25 GiB and adds about 294 ms of model-load repacking. Prefill is about 2% slower in the measured pair. The active checkout and source-default CUDA library remain the baseline.

The candidate source and runtime were built and exercised only in `/tmp/bonsai-exp035` on branch `exp035-ptq1-context-owned-soa`; they are uncommitted and isolated. Raw logs, benchmark JSON, test logs, dispatch proof, and resource output are preserved in [`results/exp035/`](../../results/exp035/). The candidate source remains isolated in `/tmp/bonsai-exp035`; it is not part of the production checkout.

## Scope and implementation

The experiment targeted the 64 PTQ1_0 `blk.*.ffn_down.weight` tensors with shape `[17408, 5120]`. The source retained canonical AoS storage and all existing consumers. A CUDA buffer-context registry owned a second, seven-word-plane copy for those exact tensors; entries were freed with the owning CUDA buffer. Repack/registration happened only after a complete upload. Partial uploads, async setter mutation, 2D setter, destination copies, and buffer clear invalidated the entry. Host repacking used the upload source synchronously, and a diagnostic D2H comparison of the first target upload matched every byte against the host upload source and CUDA tensor.

The selector was threaded through the CUDA mat-vec dispatch and accepted only PTQ1_0, K=17408, M=5120, contiguous rows, one column, no ids, and either no fusion payload or `x_bias`-only fusion. Gate, gate bias, and scaling remained excluded. The kernel used a separate compile-time SoA specialization for the primary dot while retaining the ordinary x-bias epilogue. Prefill, multi-column, MMQ, conversion, dequantization, and get-rows continued to read AoS.

Dispatch instrumentation on the target model observed eight `has_soa=1` launches with `fusion=1`, `x_bias=1`, `ncols=1`, and `rows=5120`; the registry returned a non-null sidecar for each. The model buffer destructor reported 64 entries and 1,247,805,440 bytes. Three final load logs measured sidecar build/repack at 292.985, 294.047, and 303.005 ms. The model graph used these pointers while the owning model buffer remained live; buffer destruction freed them on the owning CUDA device.

Two correctness defects were found and fixed during the prototype. First, the original selector incorrectly treated the fusion-argument object pointer as proof of fusion; model calls carry an allocated fusion object with only `x_bias` populated. Second, plane 6 contains `qh[0:2]` in its low bytes and scale `d` in its high half. The SoA decoder must reconstruct qh as `(word & 0xff) | ((word & 0xff00) << 8)` to match the AoS `qh[0] | (qh[1] << 16)` layout, while extracting `d` from `word >> 16`. The first x-bias SoA smoke emitted malformed text; the corrected path passed exact completion comparison.

## Correctness

- Selected CTests: `test-quantize-fns`, `test-ptq1_0-element-map`, `test-ptq1_0-cuda-dot`, and `test-pq2-row-shapes` passed 4/4.
- CUDA `MUL_MAT` backend-ops filtered to PTQ1_0 and PQ2_0 passed 393/393 cases. This includes the standard multi-column AoS route.
- Fixed-seed PTQ1_0 CLI smoke, control versus corrected sidecar-enabled candidate, generated the exact same normalized completion: `We need to respond to user: "A short test: explain a triangle." Need final answer short. Explain triangle. Keep concise.`
- Fixed-seed PQ2_0 CLI smoke, control versus candidate-enabled build, generated the same normalized completion: `We need to respond to user: "A short test: explain a triangle." Need final answer short. Explain triangle. Keep concise.`
- The first diagnostic launch trace established that the `x_bias`-only specialization was reached. The later deterministic smoke exercised that corrected path with tracing disabled.

Logs: [`ctest_final.log`](../../results/exp035/ctest_final.log), [`backend_ops_ptq1_pq2_final.log`](../../results/exp035/backend_ops_ptq1_pq2_final.log), the PTQ1_0/PQ2_0 control and candidate smoke logs, and the dispatch trace, all under [`results/exp035/`](../../results/exp035/). Repeated and reversed-order decode JSON/sample logs are preserved there as `repeat_*` and `reverse_*`; the initial full-workload runs are `final_control.json` and `final_candidate.json`.

## End-to-end performance and memory

Runs used the fixed project llama-bench workload, 7 repetitions, 128 decode tokens, contexts 512/4096, batch/ubatch 2048/512, 8 CPU threads, F16 KV, Flash Attention on, and 99 GPU layers. Each process began after the runner's idle and <=60 C cooldown gate. The clean decode-only pair was repeated in both process orders. Values below are per-process medians; the full seven samples are in the raw JSON files.

| Process order | Context | AoS control tok/s | SoA candidate tok/s | Delta |
|---|---:|---:|---:|---:|
| control → candidate | 512 | 81.6815 | 81.6241 | -0.07% |
| control → candidate | 4096 | 78.4412 | 77.7962 | -0.82% |
| candidate → control | 512 | 81.6843 | 81.6453 | -0.05% |
| candidate → control | 4096 | 77.9794 | 78.3425 | +0.47% |

Across both pairs, the median-of-pair-medians is 81.6829 tok/s for control and 81.6347 for candidate at 512, and 78.2103 versus 78.0694 at 4096. This is effectively a tie at 512 and a small, inconsistent loss at 4096. Both 4096 sets contain large low outliers, so the apparent +0.47% in the reverse pair is not a repeatable gain.

In the initial full prefill/decode run (control prefill→decode, candidate decode→prefill), prefill medians were 1393.45 versus 1365.03 tok/s at 512 (-2.04%) and 1357.17 versus 1327.74 at 4096 (-2.17%). Because the process order differed, treat these prefill numbers as directional only. Measured peak GPU memory was 6,805 MiB for control and 8,085 MiB for candidate, a +1,280 MiB increase for the 1,190 MiB sidecar payload. The device reports 9,867 MiB total VRAM, so the candidate fit but left substantially less headroom.

The initial enabled/disabled A/B from before the x-bias selector fix was invalid for kernel performance: the host treated any non-null fusion-argument object as active fusion, while model dispatch passed an empty-or-x-bias fusion payload. That run registered sidecars but did not activate the SoA specialization. Its measurements are retained as diagnostic artifacts and are not used in this decision.

The candidate library resource dump showed no stack or local-memory use for the relevant kernels. The batch-1 AoS specialization used 76 registers (74 with x-bias); the SoA variants used 78 registers (77 with x-bias). The SoA choice is a host dispatch decision and `use_soa` is a compile-time kernel parameter, so the AoS kernel body has no runtime null-sidecar test. The measured AoS control and candidate decode results likewise show no material default-path slowdown beyond run noise; the separately added host lookup exists only in this rejected prototype.

## Conclusion and follow-up

The concrete ownership route was viable: it registered exactly the intended matrices, survived model decode and graph execution, cleaned up with the CUDA buffer, and preserved exact output after correcting qh lane expansion. It is not worthwhile as a runtime experiment result: long-row speed did not translate into a repeatable end-to-end decode improvement, while memory and load-time costs are clear and prefill trends slower. Revert the candidate source/runtime and retain canonical AoS as the active best.
