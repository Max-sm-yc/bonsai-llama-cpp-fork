# Exp064: cooperative SSM/L2 fusion for aliased outputs

## Hypothesis

The 24 remaining one-token Qwen3.5 `SSM_CONV + SiLU → L2_NORM` pairs use the same zero-offset `[128,32]` QK view as Exp062, but the L2 output aliases bytes `[0,16384)` of the SSM input. A cooperative grid barrier after all CTAs finish the SSM reads could make a single fused launch safe and remove one more repeated kernel node at each site.

## Feasibility gate

The isolated worktree was created at code commit `ffb0ef37690b902829ea1158b02b14517ed93c2b` (the current Exp062 implementation). The target is an NVIDIA GeForce RTX 3080, compute capability 8.6, with CUDA Toolkit 13.2.86 (`nvcc`) and driver 580.178.04. The driver reports CUDA 13.0 in `nvidia-smi`.

On this device, `cudaDevAttrCooperativeLaunch=1` and there are 68 SMs. A 128-thread CUDA probe measured 12 resident CTAs/SM (816 total), 38 registers/thread, 0 static shared bytes, and 0 local bytes. The production candidate kernel was queried separately through `cudaOccupancyMaxActiveBlocksPerMultiprocessor`: it measured 12 CTAs/SM with 38 registers/thread, 0 static shared bytes, 16 dynamic shared bytes, and 0 local bytes. The intended 80-CTA grid therefore fits within the 816-CTA residency bound. The exact production query output is in `results/exp064/raw/occupancy_test.log`; the standalone API probe and source are `coop_probe.log` and `coop_probe.cu`.

A standalone `cudaLaunchCooperativeKernel` launch captured into a CUDA Graph, instantiated, and replayed successfully as one node. The focused production-kernel test also captured and replayed the alias path: the generic SSM and L2 operations captured as two nodes; the cooperative fused operation captured as one. Its `cudaGraphGetNodes` diagnostic is in `results/exp064/raw/graph_capture_alias_{generic,candidate}.log`. The matching Nsight Systems focused captures show graphId records for the replayed nodes. This validates both cooperative launch and graph capture/replay on the active CUDA stack. The implementation uses `cooperative_groups::this_grid().sync()`; it does not use a spinning global barrier.

CUDA’s Cooperative Groups guide requires a cooperative launch for grid synchronization and documents the grid-wide memory visibility guarantee of `grid.sync()`. The Runtime API documents that the cooperative grid must not exceed occupancy times SM count and permits cooperative launches in CUDA Graphs when MPS is not in use: [CUDA Programming Guide, Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), [CUDA Runtime API, `cudaLaunchCooperativeKernel`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html).

## Candidate and guards

The candidate kept Exp062’s ordinary launch for the 24 disjoint sites and added a separate `GGML_CUDA_DISABLE_SSM_L2_ALIAS_FUSION=1` control. Alias fusion required the same model shapes, F32 types, strides, zero-offset view, use counts, and unpinned outputs as Exp062, plus an exact `l2->data == ssm->src[0]->data` alias. It also checked that the L2 byte range fit within SSM input and did not overlap the SiLU output or convolution-weight range. Any other graph or layout used generic fallback. The candidate diff, including the focused test extension, is preserved at `results/exp064/raw/candidate.patch`.

For aliased sites, each CTA computes its SSM+SiLU values into registers. All CTAs then cross the grid barrier before any CTA stores. CTAs write the full SiLU output, reduce their own QK group, and write normalized values into the aliased input range. The ordinary disjoint path retains Exp062’s launch and ordering.

## Correctness

`tests/run-exp062-ssm-l2.sh` passed with the Exp062 model and fallback cases, and with a new forced-alias model-shape case. The alias test sets L2 output to exactly the SSM-input base, compares the generic alias path against the cooperative path with `cmp`, and includes the complete SiLU output, normalized QK values, and post-normalization aliased input bytes. It repeats the graph three times so it exercises direct evaluation, graph capture, and graph replay. The control only sets `GGML_CUDA_DISABLE_SSM_L2_ALIAS_FUSION=1`; Exp062 remains on.

Compute Sanitizer reported zero errors for generic and cooperative `synccheck`, zero racecheck hazards, and zero memcheck errors. Logs are `synccheck_{generic,candidate}.log`, `racecheck_candidate.log`, and `memcheck_candidate.log`. A one-shot PTQ1_0 model smoke completed four generated tokens at 54.0 tok/s; the log is `cli_graph_candidate.log`. The smoke is supplementary and is not used as the correctness proof.

The broader `tests/run_correctness.sh` suite was not feasible from the isolated build state. The first full CUDA build hit the shared `/tmp` quota during unrelated translation units. I reused the code-commit-matching Exp063 build outputs, recompiled the two modified CUDA translation units, manually linked the isolated CUDA library, and verified library resolution with `ldd`. A direct selected-CTest attempt confirmed the copied build did not contain the selected test executables, so those CTests reported Not Run. The focused test and sanitizer runs above did execute against the candidate library.

## Focused graph and kernel screen

The production alias test’s captured graph had two nodes for the generic SSM and L2 path and one cooperative node for the candidate. This is an exact one-site graph result. Exp063 observed 24 alias sites per model decode graph; replacing each two-node pair with one node would remove 24 nodes if all such sites are captured in the same model graph.

The isolated Nsight Systems graph replay profile (`results/exp064/raw/focused_profile_{generic,candidate}.nsys-rep` and matching SQLite exports) measured the graphId events as follows:

| Path | Nodes at this site | Mean kernel time in graph replay |
|---|---:|---:|
| Generic SSM + L2 | 2 | 1.696 + 1.472 = 3.168 µs |
| Cooperative fused | 1 | 3.408 µs |

The cooperative kernel was 0.240 µs (7.6%) slower per site in this focused graph replay, despite removing one graph node. The focused graph capture logs and SQLite exports preserve node identities and every measured kernel event. The separate full-model nsys profile included `--cuda-graph-trace=node`, but CUDA 13.2’s exported kernel rows had null graphId/graphNodeId, so I do not use that trace to claim a full-model node or per-replay timing delta. It did confirm the candidate specialization was dispatched in the PTQ1_0 model path and recorded its 38-register/16-byte-dynamic-shared resource use.

The first full rebuild attempt failed because NVCC ran out of the shared `/tmp` quota. The isolated candidate library was then made by reusing prebuilt objects from the same code commit, compiling the modified SSM and scheduler translation units, and linking in the Exp064 worktree. Its SHA-256 during A/B was `943d750304aff8169e59e7f581efdb7e27286651793eb0c6d0e97296a9e3ed90`. The exact A/B loader paths are in `results/exp064/raw/ldd_candidate.txt`.

## End-to-end PTQ1_0 decode

The same `llama-bench` binary and candidate library were used for both arms. The control set only `GGML_CUDA_DISABLE_SSM_L2_ALIAS_FUSION=1`, preserving Exp062’s disjoint-site fusion. Each arm used `-p 0 -n 128 -r 7`, 99 GPU layers, FlashAttention on, batch 2048, ubatch 512, F16 K/V, and 8 CPU threads with default identical warmup. Each run began at GPU utilization 0% and temperature at or below 60 °C. Pair 1 ran generic then candidate; pair 2 reversed the order. All seven samples, per-run standard deviations, start gates, GPU memory samples, and peak VRAM are preserved in `results/exp064/raw/pair*_*.json` and `.meta.json`.

| Context | Pair order | Generic avg tok/s | Candidate avg tok/s | Change |
|---:|---|---:|---:|---:|
| 512 | generic → candidate | 84.695 | 84.572 | −0.146% |
| 512 | candidate → generic | 84.304 | 84.401 | +0.115% |
| 4096 | generic → candidate | 81.789 | 81.784 | −0.006% |
| 4096 | candidate → generic | 81.735 | 81.789 | +0.067% |

Median of the two per-arm run averages changed by −0.016% at context 512 and +0.030% at context 4096. The directions disagree within each context and are negligible relative to run spread. Peak VRAM was 6,579 MiB at context 512 and 6,803 MiB at 4096 in both arms. `results/exp064/raw/ab_summary.json` contains the computed summaries and sample ranges.

## Analysis and decision

The residency and graph-safety premise is valid, and the exact alias path passes focused byte comparisons and synchronization checks. However, the barrier makes each fused site slower than the two generic kernels in the focused graph replay, and the model A/B shows no repeatable decode gain. The added cross-CTA synchronization and cooperative launch are not justified by the one-node reduction.

**Decision: REVERT the candidate and KEEP Exp062 as the best verified code.** The Exp064 worktree source is restored to the requested Exp062 commit; the candidate patch and all logs/profiles remain in `results/exp064/` for review.

## Manager verification

The manager independently recomputed the paired model changes from `ab_summary.json`; the two run orders disagree at each context and the median changes are −0.016% (512) and +0.030% (4096). The focused SQLite exports contain the captured graph kernels: the generic path has two replayed operations per capture (SSM 1.856/1.536 µs and L2 1.472/1.472 µs, mean combined 3.168 µs); the cooperative path has one kernel per capture (3.488/3.328 µs, mean 3.408 µs), a 7.6% regression. Capture logs independently confirm 2 versus 1 graph nodes. The report’s sanitizer summaries and exact alias test logs are present. The isolated source is restored at `ffb0ef37690b902829ea1158b02b14517ed93c2b`; the main production checkout remains unchanged.

## Follow-ups and discoveries

- Exact runtime residency of the production kernel is 12 CTAs/SM, so the 80-block cooperative launch is safely co-resident with an 816-block capacity on this RTX 3080.
- The barrier provides the required all-read-before-any-write order for the alias, and CUDA Graph capture/replay works for the production cooperative kernel.
- The extra synchronization costs more than the eliminated small L2 launch at each site. Revisit only if a new layout or kernel design can retain a grid-wide read/write phase boundary at lower cost.
- Keep the existing Exp062 disjoint-range path and its independent disable control unchanged.
