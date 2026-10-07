# Experiment 042: active PTQ1_0 GEMV CTA-width sweep

## HYPOTHESIS

The active sm_86 batch-1 PTQ1_0 GEMV uses 128-thread CTAs and serializes each thread's K work list. A narrower or wider CTA, together with a compatible row tile, might improve occupancy or distribute long-K work better. The best width could differ for K=40 and K=136 blocks per row.

## IMPLEMENTATION

The production checkout and build were left untouched. All kernel work used an isolated worktree at `d3e7daf` in `/tmp/exp042-cta-width`; the production CUDA library remained SHA-256 `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`.

The width sweep compiled 64, 128, 256, and 512-thread variants of the two CUDA translation units that include `mmvq-ptq1_0.cuh` (`mmvq.cu` and `quantize.cu`). Each library was linked into its own copied runtime directory under `results/exp042/width{64,128,256,512}/`. The header's `PTQ1_0_PT_THREADS` define was made overridable; row-tile selection, partial writes, and folds retained the active ROWS=1 and existing reduction order. At K=40/K=136, selected rows per CTA were:

| CTA threads | K=40 rows/CTA | K=136 rows/CTA |
|---:|---:|---:|
| 64 | 8 | 8 |
| 128 | 16 | 16 |
| 256 | 6 | 15 |
| 512 | 12 | 15 |

These values are the production chooser's utilization maximum under its 16-row/4,096-float shared-memory cap. The selected shared-memory allocations stayed in bounds.

Ptxas reported the active plain ROWS=1 specialization at 76 registers/thread for 64/128 threads, 60 for 256, and 40 for 512. All four had zero stack, shared-memory, and local-memory use. `cuobjdump` PTX confirmed `.maxntid` values of 64, 128, 256, and 512 for the active entry point. The shape-specific candidate emits both `.maxntid 128` and `.maxntid 256`; the active plain ROWS=1 forms use 76 and 60 registers/thread respectively. SASS and full resource dumps are retained under `results/exp042/sass/` and `results/exp042/resources/`. The 512-thread build warned that the existing minimum-four-CTAs-per-SM launch-bound hint exceeds the SM thread limit and was ignored; the 512-thread maximum bound was retained.

Because the width sweep showed a small isolated K=40 win at 256 threads and 128 remained best at K=136, an additional isolated shape-dispatch candidate used 256 threads only for `ncols_x / QK_PTQ1_0 == 40` and 128 otherwise. Its source diff is `results/exp042/shape256_k40.patch`; its CUDA library is in `results/exp042/shape256_k40/`.

The focused screen (`results/exp042/cta_width_screen.cu`) uses the active planar activation indexing, the same packed ternary decode and DP4A/FMA block dot, the production row-tile chooser, and the same four-accumulator row fold. It compares the width candidate against the 128-thread control and an independent host decoder/output reference. Each shape used 2,048 rows and nine rotated-order CUDA-event samples, each averaging 100 work-plus-fold pairs.

Exact build and run commands:

```bash
# Isolated source base
 git worktree add --detach /tmp/exp042-cta-width d3e7daf
# Width-specific focused harness
nvcc -O3 -arch=sm_86 -Xptxas=-v results/exp042/cta_width_screen.cu \
  -o results/exp042/cta_width_screen 2> results/exp042/harness_resources.txt
bash results/exp042/run_micro.sh
# Per-width runtime libraries were built by taking the existing Release/sm_86
# Ninja compile commands for mmvq.cu and quantize.cu, adding
# -DPTQ1_0_PT_THREADS={64,128,256,512}, compiling those two objects from the
# isolated worktree, then relinking the unchanged ggml-cuda object list into
# the matching results/exp042/width{N}/libggml-cuda.so.0.21.0 directory.
# The shape-dispatch candidate used the same process with the retained patch.
```

Before timing, both `readelf -d <binary>` and `ldd <binary>` were checked. The executables retain the build's absolute RUNPATH, so each timing command explicitly set `LD_LIBRARY_PATH` to the arm directory. `ldd` then resolved `libggml-cuda.so.0` and `libggml-base.so.0` to that exact directory. Control and candidate CUDA library SHA-256 values were respectively `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642` and `85989591ba4abce7c77968ee7e2958ad2d2e5a5b28a94ed9e50f5611d755c7b6`.

## RESULT

The global width sweep found no width that improves both shapes. 256 threads improved K=40 by 1.16% in the final screen, with non-overlapping sample ranges, but lost 1.53% at K=136. 128 threads remained the control choice at K=136. The shape-specific 256-at-40/128-otherwise build then lost in the matched model decode comparison and is rejected.

## CORRECTNESS

For all eight width/shape combinations, 2,048 candidate rows matched the independent host output reference bit-for-bit: zero control mismatches, zero candidate mismatches, and zero maximum absolute error. Raw checks and samples are in `results/exp042/screen_k{40,136}_t{64,128,256,512}.txt`.

The shape-dispatch candidate completed fixed-seed 32-token PTQ1_0 and PQ2_0 model smokes. Its normalized PTQ1_0 completion matched the control smoke exactly; both formats generated non-empty completions. This is a smoke result, not a claim that the full correctness suite was run.

## MICROBENCHMARK

CUDA-event milliseconds per work-plus-fold pair; each cell is median (range) of nine rotated-order samples. Delta is candidate versus the same-run 128-thread control median.

| K blocks | CTA threads | rows/CTA | 128 control, ms | candidate, ms | delta |
|---:|---:|---:|---:|---:|---:|
| 40 | 64 | 8 | 0.00991072 (0.00990016–0.00994176) | 0.00993280 (0.00992704–0.00995328) | +0.22% |
| 40 | 128 | 16 | 0.01000416 (0.00999424–0.01002304) | 0.01000064 (0.00999296–0.01001472) | -0.04% |
| 40 | 256 | 6 | 0.01003936 (0.01002176–0.01008736) | 0.00992256 (0.00990208–0.00993280) | -1.16% |
| 40 | 512 | 12 | 0.00996928 (0.00995616–0.00999904) | 0.01033312 (0.01031456–0.01034240) | +3.65% |
| 136 | 64 | 8 | 0.02782880 (0.02777920–0.02803712) | 0.02839552 (0.02833344–0.02850592) | +2.04% |
| 136 | 128 | 16 | 0.02781088 (0.02778112–0.02785888) | 0.02781184 (0.02776096–0.02796544) | +0.00% |
| 136 | 256 | 15 | 0.02786304 (0.02781184–0.02816640) | 0.02828896 (0.02824000–0.02839552) | +1.53% |
| 136 | 512 | 15 | 0.02784256 (0.02780800–0.02807776) | 0.02837376 (0.02833408–0.02839552) | +1.91% |

## END-TO-END IMPACT

Run. The shape-specific candidate and control each ran two order-reversed pairs with seven repetitions at contexts 512 and 4096, 128 decode tokens, and the fresh ≤60°C/≤5% utilization gate before each arm. The exact benchmark commands are retained in `results/exp042/README.md`; raw JSON samples are `control_forward.json`, `candidate_forward.json`, `candidate_reverse.json`, and `control_reverse.json`.

The median of the two run medians regressed by 0.60% at context 512 and 0.68% at context 4096:

| Context | Control run medians, tok/s | Candidate run medians, tok/s | Paired median, control → candidate | Change |
|---:|---:|---:|---:|---:|
| 512 | 83.7078, 83.3152 | 83.0653, 82.9631 | 83.5115 → 83.0142 | -0.60% |
| 4096 | 81.1593, 79.8505 | 80.3848, 79.5367 | 80.5049 → 79.9608 | -0.68% |

All repetitions are preserved. Across all 14 repetitions, context-512 ranged 82.43–83.75 tok/s for control and 82.04–83.15 for candidate. Context-4096 ranged 62.83–81.18 tok/s for control and 56.81–80.58 for candidate, with severe slow tails in both reverse-order runs. Despite those tails, both run-median comparisons favor control. Peak GPU memory was 6,803 MiB in each arm.

## ANALYSIS

CTA width alone reduces per-thread register use at wider sizes, but that change did not improve this workload. The 512-thread form was slower at both K sizes. The 256-thread form was measurably useful only for K=40; its larger row-tile mismatch at K=136 added cost. Shape-specific dispatch preserved the K=136 control performance at the kernel-screen level, but the modest K=40 screen gain did not survive model decode: the E2E result regressed at both contexts. The extra template and dispatch complexity is not justified.

The K=136 tail variability is substantial in the model runs; it limits fine-grained interpretation, but the direction was negative at both contexts and both run medians. No claim of a statistically meaningful regression magnitude is needed to reject a candidate that failed to improve the decision metric.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** Keep the current fixed 128-thread production implementation. No production source, build, or library was changed.

## FOLLOW-UPS

Do not promote a CTA-width change from this sweep. Revisit this dimension only with a new mapping premise that reduces K=136 work-plus-fold time without losing the K=40 gain, or with stronger evidence that the shape-specific dispatch reduces the decode bottleneck.

## IMPORTANT DISCOVERIES

- The emitted active kernel honors `.maxntid` 64/128/256/512, and resource use fell from 76 to 60 to 40 registers/thread as width increased, with no spills in the active plain ROWS=1 specialization.
- CTA width interacts with the row-tile selector: K=40 uses 8/16/6/12 rows per CTA; K=136 uses 8/16/15/15.
- 256 threads was 1.16% faster at K=40 but 1.53% slower at K=136 in the exact work-plus-fold screen.
- Shape-specific dispatch to 256 only at K=40 passed model smokes, but its matched decode medians regressed 0.60%/0.68%; the production 128-thread implementation remains best.
