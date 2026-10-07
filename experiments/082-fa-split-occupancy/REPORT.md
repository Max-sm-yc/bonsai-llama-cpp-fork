# Experiment 082: Ampere FlashAttention split occupancy

## Hypothesis

The active sm_86 F16 decode specialization, `flash_attn_ext_f16<256,256,1,8,...>`, uses a two-stage K/V pipeline with 67,584 B of K/V shared storage (67,728 B total dynamic shared storage including the mask). That footprint limits the main kernel to one CTA per SM on the RTX 3080. A one-stage configuration with the same KV batch and 128-wide K/V tiles should nearly halve dynamic shared storage, permit two resident CTAs per SM, and make more long-context Stream-K partitions available. The added partial-output and online-softmax fixup work must be included in the comparison.

## Implementation

Ran in the isolated worktree `/home/maxsun/autonomous_projects/.worktrees/exp082-fa-split-occupancy`, based at manager commit `39f6b1e7bbe305dbbb068053e29298025276e26c`. The supplied hash had one extra trailing `2`; the checkout's actual `HEAD` is recorded above. The manager checkout was not changed.

The only candidate source change was the active Ampere `(DKQ,DV,ncols)=(256,256,8)` entry: `nstages_target` 2→1. Thread count, `nbatch_fa=64`, K/V tile widths 128, combine width 128, GQA grouping, and attention math were unchanged. The candidate uses 33,792 B for shared K/V plus 144 B for the mask, 33,936 B total dynamic shared storage.

`cuobjdump --dump-resource-usage` reports 184 registers/thread and 16 B stack for the control kernel, versus 195 registers/thread and 16 B stack for the candidate. Nsight Systems reports dynamic shared storage of 67,728 B and 33,936 B, respectively. The runtime occupancy calculation in `launch_fattn` therefore raises the available CTA cap: on long-context graph launches the observed grid rises from 68 to 136 CTAs. At context 512, the measured grid remains 48 CTAs because the KV work does not fill the larger cap. The graph contains 16 calls to this attention signature per replay.

The control was the unchanged current-best build in the manager checkout. Its CUDA library hash is `7b0c851cd4c1ff7800dfe88aaf1e102922a5712274f7f70bc320532452b991a1`; candidate library hash is `88355f4de9695c29b91af15ea7dce16648a1792933c6e4b7cb89a238a8b0e522`. `ldd` and `LD_DEBUG=libs` confirmed each executable loaded its expected absolute library path. Hashes and loader traces are in `results/exp082/raw/`.

## Focused benchmark

Used the same `llama-bench` settings for control and candidate: PTQ1_0 model, 99 GPU layers, FlashAttention on, batch 2048, ubatch 512, F16 K/V, 8 CPU threads, `-p 0 -n 16`, contexts 512 and 4096, two warmup repetitions, and CUDA Graph node tracing. Each capture contains 31 graph replays and 1,360 graph nodes per replay. GPU start checks were within the requested gate (50–53 C, 0% utilization). Raw captures, exported SQLite, and per-replay summary JSON are under `results/exp082/raw/`.

Times below are mean summed duration per replay. Main plus fixup is the total attention cost. The ranges are minima and maxima across the 31 replays.

| Context | Variant | Main attention | Stream-K fixup | Total attention | Graph grid X |
|---:|---|---:|---:|---:|---:|
| 512 | Control | 0.196410 ms (0.194594–0.197763) | 0.034482 ms (0.034176–0.034657) | 0.230892 ms | 48 |
| 512 | One-stage | 0.225433 ms (0.224290–0.226274) | 0.037158 ms (0.037024–0.037344) | 0.262591 ms | 48 |
| 4096 | Control | 0.546593 ms (0.544933–0.548101) | 0.036184 ms (0.035967–0.036448) | 0.582777 ms | 68 |
| 4096 | One-stage | 0.600891 ms (0.598501–0.603590) | 0.056649 ms (0.056384–0.056993) | 0.657541 ms | 136 |

The candidate regressed total attention by 13.7% at context 512 and 12.8% at context 4096. At 4096, the doubled launch grid verifies that the lower footprint does permit two CTAs per SM on the long-context attention work. That added split capacity did not offset a 9.9% main-kernel slowdown plus a 56.6% fixup slowdown. At 512, the available KV partitions did not use the higher occupancy cap, yet the main kernel still slowed by 14.8% and fixup by 7.8%.

## Correctness

The candidate passed the CUDA `FLASH_ATTN_EXT` backend suite: 2,994/2,994 cases passed on CUDA0. This includes F16 cases but is not an exact production-model output comparison by itself. A fixed-seed, 32-token PTQ1_0 model smoke at context 512 also succeeded. Its generated completion body matched the current-best smoke exactly. Both loader traces and outputs are retained in `results/exp082/raw/`.

The full `tests/run_correctness.sh` suite was not run because the candidate failed the required focused-performance gate. The all-operation CUDA backend checks in that script target matrix multiplication rather than FlashAttention; the directly relevant FlashAttention suite and model smoke were run instead.

## End-to-end impact and memory

No decode speed A/B was run because focused attention regressed consistently at both contexts. Therefore there are no end-to-end throughput pairs or candidate peak-VRAM result. The profile and smoke runs loaded the full model and completed without allocation failure; start and idle GPU memory were 173 MiB. The candidate does not qualify for a full benchmark under the conditional protocol.

## Decision

**REVERT / NO CANDIDATE.** The experiment proves that halving dynamic shared storage can double the long-context CTA grid on this path, but measured main-plus-fixup latency is worse at both required contexts. Restore the source configuration and keep the current best. No model speed A/B, prefill study, or production integration is warranted.

The temporary source change was saved as `results/exp082/raw/experiment.patch`; the source in the isolated worktree is restored to its base version. Candidate binaries and all raw evidence remain available there for audit. No commit was created and no manager documentation or source was modified.

## Follow-ups and discoveries

- The resource premise is supported: total dynamic shared storage fell 49.9%, and the context-4096 launch grid doubled from 68 to 136 CTAs.
- This occupancy gain is not itself a performance gain. The extra partial softmax states raise fixup cost substantially, and the candidate's main kernel also slows despite the larger grid.
- Keep the two-stage Ampere configuration. Revisit only with a concrete approach that preserves the throughput of the current main kernel while reducing the amount or cost of partial outputs; another shared-memory-only occupancy increase is not justified by this result.

## Commands and artifacts

The control and candidate profile commands were identical except for executable and output prefix:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --cuda-memory-usage=true --force-overwrite=true \
  --output results/exp082/raw/{arm}_ctx{context} \
  {absolute-path-to-llama-bench} \
  -m /home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 \
  -r 2 -o json -p 0 -n 16 -d {context}
nsys export --type sqlite --force-overwrite=true \
  --output results/exp082/raw/{arm}_ctx{context}.sqlite \
  results/exp082/raw/{arm}_ctx{context}.nsys-rep
python3 results/exp052/analyze.py results/exp082/raw/{arm}_ctx{context}.sqlite
```

Contexts were 512 and 4096; control executable was `/home/maxsun/autonomous_projects/bonsai2-rtx3080/build/bin/llama-bench`, and candidate executable was `build/bin/llama-bench`. Candidate build configuration matched the CUDA Release settings from Exp053. Candidate resource evidence came from:

```bash
cuobjdump --dump-resource-usage build/bin/libggml-cuda.so.0
```

Important artifacts include `control_ctx{512,4096}.nsys-rep`, `candidate_ctx{512,4096}.nsys-rep`, matching `.sqlite` and `.profile.json`, `control_resource_usage.txt`, `candidate_resource_usage.txt`, build logs, hash lists, loader traces, `backend_flash_candidate.log`, and both smoke output/log pairs, all under `results/exp082/raw/`.
