# Experiment 013: PTQ1_0 planar CTA row tiles

## HYPOTHESIS

The retained ROWS=1 work item could run faster with a different number of rows per CTA. Smaller tiles could expose more CTAs and improve occupancy; larger tiles could lower CTA count and scheduling/reduction overhead. The chooser's current first-full-iteration rule may also leave useful larger tiles untried.

## IMPLEMENTATION

Saved the exact control header as [control-mmvq-ptq1_0.cuh](control-mmvq-ptq1_0.cuh), recorded the evidence HEAD in [control-head.txt](control-head.txt), and temporarily copied the exact source-default `build/bin` as `results/exp013/builds/control_bin` (CUDA library SHA256 `4b7b09cd570d3f2ca289a97806ab7d2cac0c8c0eb40d73f79e3ec2a752825fde`). After the experiment, the tracked source was restored from that header and the active build from the control copy. The manager independently verified that both current files match their saved control hashes, then removed the temporary build copy. No source change remains.

Candidate-only changes, recorded in [candidate.patch](candidate.patch), made the row cap and tie-break override conditional on `ncols_dst == 1`. They left the multi-column cap=16, 4096-float budget, and early exit unchanged. The only runtime mapping change was rows-per-CTA; ROWS=1 per-item mapping, PTQ1_0 dot arithmetic, activation layout, and reduction order stayed fixed. Candidate constants were supplied at compile time. [build_variant.py](build_variant.py) shows the CUDA object rebuild procedure.

[geometry.py](geometry.py) and [geometry.csv](geometry.csv) enumerate caps 4, 8, 12, 16, 24, and 32, plus cap 32 at an 8192-float target. Shape dimensions come from the PTQ1_0 model GGUF tensor metadata: K/M = 5120/10240 (attention QKV), 5120/6144 (attention gate), 5120/17408 (fused FFN gate), 6144/5120 (SSM output), 17408/5120 (FFN down), and 5120/248320 (output). Gated calls double partial-buffer storage. The geometry assertion compares candidate and original selection for ncols 2–8 across the model K sizes; multi-column behavior is unchanged.

Largest selected active candidate geometry was 32 rows × 137 partials × 4 bytes × 2 gated buffers = 35,072 bytes, below the conservative 96 KiB limit and the runtime dispatch guard. Cap 32 at the 8192-float target used 17,536 bytes on the K=17408 ungated FFN-down projection. No measured candidate shape exceeded the hardware per-block limit.

## RESULT

No candidate improved end-to-end decode reproducibly. The best-looking screen, cap 8, was nearly tied in the full seven-repetition comparison. Larger CTA tiles lost, especially cap 32.

Three-repetition screens (tok/s medians; contexts 512 / 4096):

| Candidate | 512 | 4096 | Peak VRAM |
|---|---:|---:|---:|
| cap 4, target 4096 | 81.909 | 79.433 | 6,805 MiB |
| cap 8, target 4096 | 82.492 | 79.933 | 6,805 MiB |
| cap 24, target 4096 | 81.999 | 79.478 | 6,805 MiB |
| cap 32, target 4096 | 80.846 | 78.513 | 6,805 MiB |
| cap 32, target 8192 | 75.950 | 73.882 | 6,805 MiB |

The 3-repetition cap 8 screen looked slightly positive, so I repeated it against the exact control with seven repetitions. It did not hold. Raw JSON distributions and telemetry are in [results/exp013](../../results/exp013/).

## CORRECTNESS

Candidate changes only selected the row tile in the host helper. The ROWS=1 work item, dot code, activation layout, and reduction arithmetic/order were unchanged. Nsight Systems confirmed that the cap 8 model run executed the dedicated `mul_mat_vec_ptq1_0_pt<1,1,...>` specialization for plain, fused-gated, and fused-ungated calls; its smem budget did not force generic fallback. Full correctness and model-smoke suites were not run for the candidates because none survived end-to-end screening. The restored control source has the Experiment 010 correctness coverage.

## MICROBENCHMARK

There was no independent kernel timer. Matched one-repetition Nsight Systems traces were used to verify dispatch and inspect aggregate active-kernel time. Cap 8's three PTQ1_0 planar variants summed to 584.2 ms versus 587.5 ms for control, a 0.56% reduction in this mixed setup/decode trace. Plain GEMV was 293.0 ms / 15,731 calls versus 297.8 ms / 15,731; fused gate was 149.5 ms / 2,601 versus 150.5 ms / 2,601; fused ungated was 141.7 ms / 5,136 versus 139.2 ms / 5,136. This small trace-level difference did not improve full decode. Compact kernel summaries are in [cap8_dispatch.stats.csv_cuda_gpu_kern_sum.csv](../../results/exp013/raw/cap8_dispatch.stats.csv_cuda_gpu_kern_sum.csv) and [control_dispatch.stats.csv_cuda_gpu_kern_sum.csv](../../results/exp013/raw/control_dispatch.stats.csv_cuda_gpu_kern_sum.csv).

## END-TO-END IMPACT

The paired processes used identical isolated settings: RTX 3080/sm_86, 60°C cooldown gate, 128 generated tokens, seven repetitions, contexts 512/4096, F16 KV, Flash Attention, 99 GPU layers, batch/microbatch 2048/512, and eight CPU threads. Each binary had an explicit `LD_LIBRARY_PATH`; the saved [ldd outputs](../../results/exp013/raw/) show each process loading its intended library tree.

| Context | Control: median; mean ± SD; range | cap 8: median; mean ± SD; range | Median delta |
|---:|---:|---:|---:|
| 512 | 81.815; 81.689 ± 0.345; 80.914–81.886 | 81.907; 81.787 ± 0.356; 80.990–82.009 | +0.11% |
| 4096 | 78.719; 75.573 ± 8.226; 57.014–79.292 | 78.709; 76.084 ± 6.752; 60.877–79.410 | -0.01% |

Both full runs peaked at 6,805 MiB. They began at 59°C and 60°C, respectively, and peaked at 71°C and 74°C; both reached 1,830 MHz under load. The context-4096 distributions had one severe low sample each, so means are noisy. The medians show a tie, not a decode gain.

## ANALYSIS

The existing chooser already selects tiles with full 128-thread iteration utilization. Increasing the cap to 24 often changed nothing because smaller tiles already achieved a perfect fill: for K=5120, control stayed at 16 rows; cap 24 also chose 16. Cap 32 chose 32 rows for K=5120 and reduced its CTA count by half, but was slower in the model decode screen. At target 4096, K=17408 remained at 16 rows because that tile already filled all threads; raising the target to 8192 allowed 32 rows there and caused a marked decode regression. Smaller cap 8 selected six rows for K=5120 and eight for K=6144/K=17408, multiplying CTA counts on major projections without a repeatable decode gain.

The profile's small reduction in planar-kernel time was not enough to change model throughput. The 4096-context tails occurred in both control and candidate, consistent with the known long-context variability.

## DECISION

**REVERT.** Restored the exact ROWS=1 production header and control build. Manager audit confirmed byte-identical source and CUDA library (`4b7b09cd570d3f2ca289a97806ab7d2cac0c8c0eb40d73f79e3ec2a752825fde`); the temporary control build copy was removed afterward. No candidate binaries or large Nsight reports are retained. No code commit was made.

## FOLLOW-UPS

Keep the current CTA heuristic. Revisit only if available kernel counters can distinguish occupancy from reduction/scheduling costs; CTA count alone did not predict decode speed here.

## IMPORTANT DISCOVERIES

- Tie-breaking toward larger tiles must be scoped to one-column calls; the original multi-column schedule was preserved and audited for ncols 2–8.
- Larger rows-per-CTA settings can satisfy the smem guard and still lose end-to-end. A larger shared-memory target made K=17408 move from 16 to 32 rows and regressed the decode screen.
- The cap 8 kernel trace improved aggregate planar time by 0.56%, while the seven-repetition decode medians were flat.
