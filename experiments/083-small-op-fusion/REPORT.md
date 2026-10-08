# Experiment 083: full-attention Q-gate layout fusion

## HYPOTHESIS

The remaining repeated `cpy_scalar` family might contain avoidable layout copies directly adjacent to compute. If a copy only materializes a strided view for a single sigmoid consumer, the view can be read directly inside the following sigmoid-times-attention kernel, removing both a launch and a temporary write/read without changing arithmetic.

## IMPLEMENTATION

Mapped the live Qwen3.5 one-token graph and found 16 full-attention sites with `CONT -> SIGMOID -> MUL`. Added a CUDA graph matcher and a specialized path in the existing sigmoid-gated multiply kernel. The matcher requires the exact `[256,24,sequence]` F32 source view and strides, single-use intermediates, matching operation order, and a distinct output. Other shapes and consumers use the existing generic kernels. Production change: commit `62b4b4c` (`perf(cuda): fuse strided Q gate activation chain`).

## RESULT

The candidate removes 16 graph nodes per token. `cpy_scalar` drops from 64 to 48 calls/token. Reversed-order PTQ1_0 decode comparisons used four pairs of seven-repetition runs per context; PQ2_0 used two pairs. Table values are the medians of paired run medians, candidate versus the immediately preceding implementation, on the same RTX 3080 and benchmark configuration.

| Format / workload | Control | Candidate | Change | Peak GPU memory |
|---|---:|---:|---:|---:|
| PTQ1_0 decode, context 512 | 84.2985 tok/s | 84.4918 tok/s | +0.23% | 6,579 MiB both |
| PTQ1_0 decode, context 4096 | 81.9320 tok/s | 82.0899 tok/s | +0.19% | 6,803 MiB both |
| PQ2_0 decode, context 512 | 70.7831 tok/s | 70.9374 tok/s | +0.22% | 7,725 / 7,723 MiB |
| PQ2_0 decode, context 4096 | 69.0831 tok/s | 69.1824 tok/s | +0.14% | 7,949 / 7,947 MiB |
| PTQ1_0 prefill, prompt 512 | 1,389.98 tok/s | 1,394.35 tok/s | +0.32% | 6,569 MiB |
| PTQ1_0 prefill, prompt 4096 | 1,343.21 tok/s | 1,350.34 tok/s | +0.53% | 6,775 MiB |

PTQ1_0 prefill and PQ2_0 decode used two reversed-order pairs each. The first combined-context PTQ1_0 screen had thermal/frequency drift and is excluded from these claims. A separate final direct comparison against the frozen project reference used two reversed-order pairs/context and a uniform <=65 C gate after the <=60 C gate stopped at the idle floor. It measured 77.690->84.234 tok/s at context 512 (+8.42%) and 75.549->81.388 tok/s at context 4096 (+7.73%), with two 7-repetition runs per arm. Its full results and latency/memory samples are in `results/exp083/raw/final65_ptq1_decode_ctx*.json` and `FINAL_RESULTS.md`.

## CORRECTNESS

- The final selected CUDA CTest set passed 7/7, CUDA-versus-CPU PTQ1_0/PQ2_0 cases passed 96/96, and fixed-seed 32-token model smokes passed for both quantizations. The PTQ1_0 smoke completion matched the saved reference body after normalizing build and timing headers.
- A dedicated CUDA test constructs the exact strided model-shaped chain at sequence lengths 1, 2, 128, 512, and 4096, plus a contiguous-source fallback. All six cases passed against a host scalar sigmoid-times-multiply reference with tolerance `2e-6`; maximum absolute error was `1.1920929e-7` and there were no mismatches.
- The dedicated CUDA test was also run directly against the rebuilt candidate library; the final CTest rerun included it.

## MICROBENCHMARK

Nsight Systems paired graph traces each covered 31 replays/context. At context 512, summed graph-node kernel time fell from 11.726378 to 11.701206 ms/replay (-0.215%). At context 4096 it fell from 12.077187 to 12.067217 ms (-0.083%). The 16 removed copy kernels saved about 0.030 ms/replay; the fused sigmoid kernel grew by about 0.003 ms. Node count changed 1,360->1,344 at both contexts. Profile summaries and raw captures are under `results/exp083/raw/`.

## END-TO-END IMPACT

The improvement is small because the targeted family accounts for little decode time; the active PTQ1_0 batch-one GEMV remains dominant. The result repeats across both quantizations and both tested contexts, with unchanged peak VRAM. The fresh direct frozen-reference comparison measures the cumulative current-code gain without adding experiment deltas.

## ANALYSIS

The matcher removes memory traffic and 16 launches without changing the sigmoid or multiply arithmetic. The fused kernel costs slightly more per site, so net gains are bounded by the approximately 0.03 ms/token copy cost. At context 4096, the copy time is lower and the relative decode gain correspondingly smaller. PTQ1_0 remains faster than PQ2_0 in decode; this fusion does not address their different weight formats or unpacking paths.

## DECISION

**KEEP.** The full-shape guarded fusion passed correctness and showed positive matched end-to-end results in every reported format/context pair. Commit `62b4b4c` is the current production-code candidate.

## FOLLOW-UPS

1. The most promising remaining small-op target is the 48 linear-attention `final_output` `CONT` copies, about 0.082 ms/token at context 512 in the measured graph. Map their consumers and aliasing first; only fuse if the saved copy survives end-to-end benchmarking.
2. The larger opportunity remains `mul_mat_vec_ptq1_0_pt`, approximately 9.0 ms/token and 75% of summed decode kernel time. Revisit only with a new exact dataflow premise; prior decoder, memory-stage, and tensor-core alternatives were slower or unsuitable for batch one.
3. Re-profile after any successful larger change because the secondary bottlenecks and copy ceilings will move.

## IMPORTANT DISCOVERIES

- The 16 full-attention copies were real adjacent graph nodes, not inferred from kernel-family totals; the fusion removed them from captured graphs.
- A safe matcher must preserve the strided view's row/sequence addressing and reject intermediate aliases, extra consumers, or graph outputs.
- Removing low-cost kernels can measurably help, but the end-to-end ceiling is small while PTQ1_0 GEMV dominates.
- PTQ1_0 and PQ2_0 were both measured: this graph fusion helps either format slightly and does not change the decode ranking.
