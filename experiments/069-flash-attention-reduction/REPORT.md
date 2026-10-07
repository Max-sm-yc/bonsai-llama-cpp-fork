# Experiment 069: FlashAttention Stream-K reduction

## HYPOTHESIS

The active sm_86 `flash_attn_ext_f16<256,256,1,8,...>` path spends about 36 µs/token on 16 Stream-K fixup launches. A different reduction or launch design might save this work while retaining the active attention tile geometry. The screen was bounded to deciding whether source and dispatch behavior support a credible candidate; no reduction design was assumed to be faster.

## IMPLEMENTATION

No candidate was implemented. Source inspection found the active Ampere config for `(DKQ,DV,ncols)=(256,256,8)`: 64 threads, occupancy target 4, `nbatch_fa=128`, K/V widths 128, combine width 128, two pipeline stages, and Q in registers. The active dispatch uses Stream-K. The profile records `ntiles_dst=4` (the fixup grid has x=4 and z=8) and `ntiles_KV=4` at context 512 versus 32 at context 4096. The observed main-kernel grid x is 16 and 68 respectively, so the launch policy gives four and seventeen partial softmax states per output tile. At context 4096, the 68-block cap matches one active block per SM on this 68-SM GPU; the config's occupancy target is not the runtime active-block count.

The main kernel stores each partial output vector and its `(max, rowsum)` metadata. The uniform fixup kernel must rescale and combine those vectors using their maxima and row sums, then normalize. That merge is required whenever a tile has more than one K/V partition. Setting one block per output tile would leave only four x-grid CTAs (each handling its grouped heads) for the 68-SM GPU and make each CTA scan all 512 or 4096 keys. A grid-cooperative merge would require a cooperative launch and a grid-wide barrier in the main kernel; no source-level evidence establishes that such a barrier would cost less than the current 2.2 µs fixup launch, and the earlier Exp064 cooperative launch was slower on a much smaller kernel. Neither route was sufficiently grounded to implement.

No tile geometry or production source was changed. The experiment worktree started at `1b0b3f614ef8d6dce28faa5a85ac92279e9d0fff`. The baseline CUDA library was preserved in the main checkout at `/home/maxsun/autonomous_projects/bonsai2-rtx3080/build/bin/libggml-cuda.so.0`, SHA-256 `860fcca9977ed5ba9f9d81ce7d310481db9e9cefbe14ac30987ceb2867b30f3e`. `ldd` and `LD_DEBUG=libs` confirmed the profiled executable loaded that library; see `results/exp069/raw/ld_debug_baseline.log`. No candidate loader path exists because no candidate was built.

## RESULT

No candidate was justified. Fresh baseline CUDA Graph profiles reproduced the measured reduction cost: 16 main attention calls and 16 uniform fixups per replay at both contexts. The current path's fixup total was 0.03430 ms/token at context 512 and 0.03594 ms/token at context 4096. The captures had 31 replays and 1,360 graph nodes per replay. No end-to-end candidate comparison was warranted.

## CORRECTNESS

No code candidate was built, so there are no candidate correctness claims or correctness tests. The profiled baseline was the verified main-checkout executable and CUDA library; its loaded library SHA-256 is recorded above. Worktree source remains unchanged from the starting commit except for this report, raw profiles, and research documentation.

## MICROBENCHMARK

Fresh profiles used CUDA Graph node tracing at `-p 0 -n 16`, matching Exp053's profile method. Each capture contained 31 graph replays. The table reports mean summed kernel duration per replay, equivalent to ms/token, with min–max across replays.

| Context | Main attention | Uniform fixup | Attention total | Calls per replay | Nodes per replay |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.195781 ms (0.194656–0.196800) | 0.034298 ms (0.034144–0.034432) | 0.230079 ms | 16 + 16 | 1,360 |
| 4096 | 0.545052 ms (0.543268–0.547142) | 0.035936 ms (0.035680–0.036160) | 0.580988 ms | 16 + 16 | 1,360 |

Raw `.nsys-rep`, exported SQLite, and analyzer JSON files are under `results/exp069/raw/`. Current main-kernel times are slightly below Exp053's historical baseline (0.200702/0.551083 ms); the current graph has 72 fewer nodes due to later retained fusions. The fixup times closely reproduce Exp053's 0.034404/0.035861 ms. No before/after candidate timing exists.

## END-TO-END IMPACT

Not measured. There was no candidate that passed a focused gate, so no decode A/B or VRAM measurement was run. The current verified best remains unchanged.

## ANALYSIS

The source audit locates the cost in a necessary cross-CTA online-softmax reduction for the selected partitioning. Both captured grids use uniform partition counts, so the existing optimized fixup is already selected. At context 512 there are four partial states per output tile; at 4096 there are seventeen. The measured 16-fixup launch family totals about 34–36 µs/token.

Avoiding the merge outright means serializing each output tile's entire K/V scan into one CTA and exposing only four x-grid CTAs (each CTA handles grouped heads), a poor fit for a 68-SM device. Fusing the merge into the existing multi-CTA main grid would need a grid-wide synchronization and cooperative launch support. The main kernel currently uses ordinary launch semantics and has no such barrier. Given the modest per-fixup cost and Exp064's measured cooperative-barrier penalty, a barrier-based candidate does not have a grounded performance case without first proving a lower-cost synchronization method. No performance conclusion is claimed for either unimplemented idea.

## DECISION

**NO CANDIDATE; KEEP CURRENT PATH.** This is not a candidate REVERT: no experimental source change was made. Retain the existing active tile geometry and Stream-K fixup. No E2E or correctness work was needed.

## FOLLOW-UPS

- Reopen only with a concrete synchronization or partitioning design that avoids the 16-CTA underoccupancy of a one-partition-per-output launch and has a plausible cost below the current 2.2 µs/fixup launch.
- Preserve the fresh context 512/4096 profiles as the current graph baseline; compare against them when a grounded design exists.

## IMPORTANT DISCOVERIES

- At the active 256-wide, GQA-8 dispatch, context 512 uses 16 x-grid CTAs across 4 output tiles (4 partitions/tile), and context 4096 uses 68 CTAs (17 partitions/tile); both call the optimized uniform reducer.
- The reduction cost persists at both contexts at roughly 34–36 µs/token, despite the main kernel growing from 0.196 to 0.545 ms/token.
- The current graph has 1,360 nodes/replay versus Exp053's 1,432 after subsequent fusion work; its fixup timing remains consistent with Exp053.
- No tile-size screen was repeated, no candidate was built, and no correctness or model A/B result is claimed.

## EXACT COMMANDS AND ARTIFACTS

Read first: `research/STATE.md`, `research/EXPERIMENTS.md`, `research/IDEAS.md`, and `experiments/053-flash-attention-longctx/REPORT.md`.

Source audit: `rg -n 'stream_k_fixup|launch_fattn|nbatch_combine|flash_attn_stream_k' ggml/src/ggml-cuda/fattn-common.cuh ggml/src/ggml-cuda/fattn-mma-f16.cuh`; reviewed the active Ampere config, `launch_fattn` partition policy, main-kernel scratch writes, and uniform fixup implementation.

For each context, after verifying GPU temperature <=60 C and utilization <=5% (49 C/0% before the ctx512 run; 51 C/0% before ctx4096), ran the same profile command with the context and output prefix shown:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --cuda-memory-usage=true --force-overwrite=true \
  --output results/exp069/raw/base_ctx{context} \
  /home/maxsun/autonomous_projects/bonsai2-rtx3080/build/bin/llama-bench \
  -m /home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 \
  -r 2 -o json -p 0 -n 16 -d {context}
nsys export --type sqlite --force-overwrite=true \
  --output results/exp069/raw/base_ctx{context}.sqlite \
  results/exp069/raw/base_ctx{context}.nsys-rep
python3 /home/maxsun/autonomous_projects/bonsai2-rtx3080/results/exp052/analyze.py \
  results/exp069/raw/base_ctx{context}.sqlite
```

The commands were run once with `context=512` and once with `context=4096`. The active main-kernel launch grids observed in the SQLite captures were x=16 and x=68 respectively; the uniform fixup grid was x=4, y=1, z=8 in each case. These raw grid values establish the partition counts above.

`ldd` confirmed the executable loaded the main checkout's CUDA library. `LD_DEBUG=libs .../llama-bench --help` was captured in `results/exp069/raw/ld_debug_baseline.log`. CUDA library SHA-256: `860fcca9977ed5ba9f9d81ce7d310481db9e9cefbe14ac30987ceb2867b30f3e`. Starting source/build commit: `1b0b3f614ef8d6dce28faa5a85ac92279e9d0fff`. No candidate build or code hash exists.
