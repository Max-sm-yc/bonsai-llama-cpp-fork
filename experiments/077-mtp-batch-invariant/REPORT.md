# Exp077: Batch-invariant mode for MTP verification on RTX 3080 sm_86

**Decision: REVERT MTP promotion.** With `GGML_CUDA_BATCH_INVARIANT=1`, MTP exactly matches the bundle's target-only token IDs in all six workload cells. The mode also changes target-only output IDs in 21 of 42 samples relative to the same build with the variable unset, and its long-context speed is family-dependent. This experiment does not establish that those target-only changes preserve output quality. Production remains unchanged.

## HYPOTHESIS

The existing `GGML_CUDA_BATCH_INVARIANT=1` path makes per-column work consistent between one-column target-only decoding and the multi-column MTP verifier. If so, it should eliminate the Exp071/072 greedy token divergences. The paired server runs also test whether the mode retains enough MTP decode benefit to be useful at contexts 512 and 4096.

## IMPLEMENTATION

- Used the required isolated worktree at `/home/maxsun/autonomous_projects/.worktrees/exp077-mtp-batch-invariant`, detached at `56b1e3faeec003e86afe8b4dce7e9182f23323a3`. No production source edits were made.
- Built one Release CUDA sm_86 `llama-server` in `results/exp077/build` from this checkout. The CUDA library SHA256 is `8ea0295e1a5d7e56ed12c2a44458212527ce4d2b7caba9f60185552bf2ddc419`. Both environment variants used this same binary and library; only the server environment differed.
- The harness uses the exact Exp071 natural prompt token IDs, with seven repetitions per family/context and 128 generated tokens. It records server timing, generated token IDs/text, acceptance counts, server logs, and 200 ms GPU telemetry. To keep IDs fixed across the newer source checkout, the harness reads the saved Exp071 token IDs rather than retokenizing the source prompts.
- Each server start passed the fresh GPU gate of at most 60 C and at most 5% utilization. Settings were greedy temperature 0/top-k 1/seed 42, cache disabled, EOS ignored, `-ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 -c 4608 -np 1`; the MTP arm adds `--spec-type draft-mtp --spec-draft-n-max 2`.
- Source inspection confirms `ggml_cuda_batch_invariant()` selects the warp-per-row PTQ1_0 GEMV reduction and one-column PTQ1_0 planar activation layout. It also affects other CUDA paths: FlashAttention kernel dispatch and split sizing, plus F16/BF16 MMVF dispatch. Therefore this is a global mode experiment, not an isolated PTQ1_0 GEMV experiment.

## RESULT

The invariant mode passes the exact MTP-versus-target token-ID gate in all six cells (42 of 42 paired sample sequences). The env-off pair reproduces three failing cells from Exp071/072. Target-only output IDs change under the mode in three of six family/context cells (21 of 42 samples). MTP has a pooled server-side gain over invariant target-only of 34.2% at context 512 and 28.0% at 4096, but loses consistently on the reports family at 4096 and has noisy speculative-C++ timings.

## CORRECTNESS

Each parity cell below checks all seven corresponding 128-token sequences. “Target on vs off” compares invariant target-only IDs with target-only IDs from the same binary with the variable unset.

| Context | Prompt family | Env-off MTP vs target | Env-on MTP vs target | Target on vs off |
|---:|---|---|---|---|
| 512 | Reports | Exact, 7/7 | Exact, 7/7 | Exact, 7/7 |
| 512 | Qwen graph C++ | **Mismatch, 0/7** | Exact, 7/7 | Exact, 7/7 |
| 512 | Speculative C++ | Exact, 7/7 | Exact, 7/7 | **Changed, 0/7** |
| 4096 | Reports | **Mismatch, 0/7** | Exact, 7/7 | **Changed, 0/7** |
| 4096 | Qwen graph C++ | **Mismatch, 0/7** | Exact, 7/7 | **Changed, 0/7** |
| 4096 | Speculative C++ | Exact, 7/7 | Exact, 7/7 | Exact, 7/7 |

For all 21 target-only samples in each context, cross-environment changes were reproducible: speculative C++ at ctx512 first diverged at generated ID index 104; reports at ctx4096 at index 17; Qwen graph C++ at ctx4096 at index 94. The other 21 streams matched exactly.

At the Exp072 ctx512 Qwen boundary, default target-only emits token 1167 at generated index 66, while default MTP emits 6195 after the shared prefix and rejected draft token 18912. In invariant mode, both target-only and MTP emit **1167** at index 66, and their complete sampled ID sequences match. This establishes equality of selected target output tokens at that boundary; it does not establish bitwise equality of the full logits tensor.

Compared with the saved Exp071 target-only response text, env-off target-only matches only three of six family/context cells: reports and Qwen graph C++ at context 512, and speculative C++ at context 4096. The other three text streams differ. Exp071 did not retain generated token IDs, so this historical comparison is text-only. The builds also use different runtime revisions (Exp071's pinned runtime `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`, versus this experiment's source commit `56b1e3faeec003e86afe8b4dce7e9182f23323a3`); the text differences are not attributed here. All within-Exp077 comparisons above use token IDs from the same binary.

## MICROBENCHMARK

No isolated kernel microbenchmark was run. The evidence is end-to-end server timing, which is the relevant measure for this mode-level correctness experiment. Full per-server logs and telemetry are retained under `results/exp077/raw/`.

## END-TO-END IMPACT

Rates are server-reported generation tokens/s, median `[min–max]`, sample SD, over seven repetitions per family/context. Paired gain is the median of the seven same-repetition MTP/target rate ratios.

### Context 512

| Family | Env off target | Env off MTP | Env on target | Env on MTP | Env-on paired MTP gain |
|---|---:|---:|---:|---:|---:|
| Reports | 69.54 [69.29–69.60], SD 0.11 | 120.64 [118.66–120.80], SD 0.75 | 68.83 [68.58–68.90], SD 0.12 | 117.73 [116.53–118.24], SD 0.53 | +71.2% |
| Qwen graph C++ | 69.35 [69.14–69.41], SD 0.09 | 85.08 [84.93–85.14], SD 0.08 | 62.38 [56.66–68.59], SD 5.42 | 90.21 [90.13–90.35], SD 0.08 | +44.7% |
| Speculative C++ | 69.08 [68.93–69.13], SD 0.07 | 71.99 [67.30–73.43], SD 2.36 | 65.95 [45.47–66.76], SD 8.10 | 69.50 [51.79–70.17], SD 7.38 | +5.3% |

Invariant mode pooled rate was **85.57 tok/s MTP vs 63.78 target-only (+34.2%)**. Env-off pooled rate was 88.29 vs 69.84 tok/s (+26.4%).

### Context 4096

| Family | Env off target | Env off MTP | Env on target | Env on MTP | Env-on paired MTP gain |
|---|---:|---:|---:|---:|---:|
| Reports | 67.13 [66.12–67.33], SD 0.42 | 62.73 [61.67–63.23], SD 0.50 | 65.16 [63.77–65.34], SD 0.56 | 55.22 [54.04–55.62], SD 0.53 | **-15.2%** |
| Qwen graph C++ | 56.73 [52.18–64.69], SD 4.98 | 66.20 [60.45–69.74], SD 3.24 | 55.59 [42.75–62.88], SD 6.90 | 69.44 [64.18–73.45], SD 3.76 | +24.2% |
| Speculative C++ | 41.59 [22.31–51.49], SD 10.79 | 62.23 [53.69–79.85], SD 9.11 | 33.04 [21.22–47.25], SD 10.80 | 48.49 [44.29–72.44], SD 11.50 | +70.7%* |

Invariant mode pooled rate was **58.61 tok/s MTP vs 45.78 target-only (+28.0%)**. Env-off pooled rate was 63.51 vs 49.67 tok/s (+27.9%). `*` Speculative-C++ ctx4096 has a broad paired ratio range (0.94–2.50), so its median gain is noisy; reports MTP is consistently slower at this context.

MTP acceptance counts (accepted/proposed) were 512 env-off: reports 581/609, Qwen 448/861, speculative C++ 372/1027; 512 env-on: 581/609, 483/791, 364/1036. At 4096 env-off: 302/1167, 366/1025, 455/868; env-on: 266/1239, 433/898, 448/882.

Peak whole-device VRAM was 7,593 MiB for target-only and 8,485 MiB for MTP, below the 10 GiB limit. Full GPU temperature, utilization, and memory samples are saved for every server.

For historical context, Exp071 measured PTQ1_0 at 79.02 / 65.84 tok/s pooled at contexts 512 / 4096 with its server harness. Invariant MTP here is 85.57 / 58.61 tok/s (+8.3% / -11.0% against those values). This is a harness-level reference, not a paired comparison: Exp071 used a different build/library and a PTQ1_0 model, whereas this candidate bundle's target is PQ2_0. It does not show that invariant MTP beats current PTQ1_0 at both contexts.

## ANALYSIS

The token correctness result is clear: with the mode enabled, all sampled MTP outputs match the corresponding target-only outputs. But the mode has broad dispatch effects beyond PTQ1_0. The target model in this experiment is PQ2_0, so the PTQ1_0 GEMV reduction is not the target's matvec path; the mode's impact here can include its FlashAttention and MMVF paths. The observed target-only sequence changes confirm that the mode changes actual standalone outputs for some workloads. The ctx4096 reports family also loses 15.2% in paired MTP speed, despite a positive pooled total.

There is no independent quality evaluation for the changed target-only sequences. Exact MTP parity against the invariant target establishes internal consistency, not that the invariant target's changed output behavior preserves quality. The mode therefore does not meet the no-quality-regression promotion gate.

## DECISION

**REVERT MTP promotion.** The env-on parity result is promising, but target-only outputs differ from default behavior in 21/42 sequences and the long-context speed is not consistently useful by family. Do not enable the mode or promote this MTP bundle as a production result. No production source or runtime setting was changed.

## FOLLOW-UPS

1. If MTP is revisited, evaluate output quality for the invariant target-only changes and keep the same six-cell ID parity gate.
2. Consider isolating the per-column arithmetic behavior from global FlashAttention/MMVF dispatch changes before treating this environment switch as a PTQ1_0-only fix.
3. Keep PTQ1_0 comparisons paired on one build/library if a future candidate reaches the promotion gate.

## IMPORTANT DISCOVERIES

- The existing mode fixes every observed Exp071/072 MTP greedy mismatch in these six cells; ctx512 Qwen's known token changes from 6195 back to target-only 1167 at index 66.
- The same mode changes target-only output IDs in 3/6 cells and 21/42 sample sequences. These are observed sequence changes, not harmless tie noise.
- Invariant MTP has pooled speed gains at both contexts, but ctx4096 reports MTP loses 15.2%; the speculative-C++ ctx4096 estimate has high variance.
- The active bundle target is PQ2_0. The experiment therefore did not isolate PTQ1_0 GEMV as the cause or mechanism; the environment variable controls several CUDA dispatch and reduction choices.
