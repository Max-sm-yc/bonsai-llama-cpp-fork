# Experiment 018: four-warps-per-row for large-K PTQ1_0 GEMV

## HYPOTHESIS

For the active one-column planar PTQ1_0 GEMV, use four warps on one output row only when the row has more than 64 K blocks. At K=17,408 (136 blocks), the 128 lanes cover almost all blocks in parallel and accumulate locally; four warp sums need only a compact shared-memory combine. Keep the production ROWS=1 schedule at K<=64 and preserve all multi-column routes.

## IMPLEMENTATION

Added a temporary large-K, one-column kernel in `mmvq-ptq1_0.cuh`, routed only when `ncols_dst == 1 && Kblocks > 64`. One CTA handles one output row with 128 threads: warp `w` starts at K block `32*w + lane`, and lanes step by 128. At 136 blocks, all lanes process one block and eight lanes process a second. Each warp locally accumulates its products, warp-reduces, then stores one partial. Thread 0 combines four partials. Shared memory is 16 bytes without a gate and 32 bytes with gate fusion; the fused bias/GELU epilogues are retained. The K<=64 and multi-column paths were unchanged.

The existing rows-per-CTA picker fills 128-thread work tiles while keeping its K-block partial buffer under a 16 KiB target. For one column and 136 K blocks, the old schedule chooses 15 rows per CTA: that gives 2,040 items in 16 thread iterations, near-full thread utilization. The candidate instead uses one output row per CTA because all four warps cooperate on that row; it requests at most eight floats of dynamic shared memory. The backend build completed with `CMAKE_CUDA_ARCHITECTURES=86` (sm_86). The only diagnostic was an unused launcher local; it was removed from the saved candidate patch after compilation and had no generated-code effect.

Candidate patch: [candidate.patch](candidate.patch). Temporary candidate library SHA-256: `dfa8f09e9a781ee86dd6bcf21ac19a7acc476eddb93cdcf4b89be9b2b00a8237`. The source-default control library SHA-256 was `72d89c0c69200865b2200ef35b94e14b9a6a52a840c17cb031e987d809207e72`. Candidate and control libraries were used from explicit `LD_LIBRARY_PATH` directories; `ldd` and `LD_DEBUG=libs` evidence is in `results/exp018/raw/`. Temporary library copies were removed after recording hashes.

## CORRECTNESS

All four selected CTests passed. The CUDA-vs-CPU backend-op selection passed 96/96 PTQ1_0/PQ2_0 `MUL_MAT` cases, including K=1024/5120/6144/17408 and widths 1/2/4/8. Both fixed-prompt 32-token CUDA model smokes completed. PTQ1_0 and PQ2_0 generated text matched `results/baseline_smoke.json` after removing the build banner and timing metadata. Logs and JSON are in `results/exp018/raw/`.

## END-TO-END IMPACT

RTX 3080 / sm_86; PTQ1_0; F16 KV; FA on; 99 GPU layers; batch/microbatch 2048/512; eight CPU threads; 128 decode tokens; seven repetitions and default warmups. The benchmark gate was <=60 C and <=5% utilization. Pair 1 ran candidate then control; pair 2 reversed the order. The JSON files include every sample and GPU telemetry.

| Pair | Context | Candidate median; mean ± sample SD; range (tok/s) | ROWS=1 control median; mean ± sample SD; range (tok/s) | Median delta |
|---:|---:|---|---|---:|
| 1 | 512 | 81.5237; 81.4150 ± 0.3239; 80.6828–81.5762 | 81.7832; 81.6107 ± 0.3398; 80.8701–81.8204 | -0.32% |
| 1 | 4096 | 79.0878; 78.9827 ± 0.2698; 78.3750–79.1213 | 78.4843; 75.6739 ± 5.0774; 65.9141–79.2382 | +0.77% |
| 2 | 512 | 81.1908; 81.0464 ± 0.3336; 80.3118–81.2599 | 81.7550; 81.6073 ± 0.3340; 80.8667–81.8231 | -0.69% |
| 2 | 4096 | 78.0643; 75.0221 ± 5.6843; 63.9034–78.7832 | 78.5233; 75.2803 ± 8.0478; 57.1650–79.2745 | -0.58% |

Each process passed the start gate at 53–60 C and 0% GPU utilization. Peak whole-GPU memory was 6,805 MiB for both variants. Peak sampled temperatures were 66/73 C (candidate) and 73/74 C (control), respectively in pairs 1/2. Both variants show long-context slow tails. The 512-context result slightly favors control in both run orders; the 4096-context sign flips across pairs. This is not a reproducible E2E gain.

Raw benchmark samples, telemetry and invocations: `results/exp018/raw/{candidate,control}_pair{1,2}.{json,log}`. The invocation template is `results/exp018/run_decode.sh`.

## DECISION

**REVERT.** Keep the production ROWS=1 mapping. The candidate passed correctness but did not produce a repeatable decode gain. The source header was restored to `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`; the active CUDA library was restored to `72d89c0c69200865b2200ef35b94e14b9a6a52a840c17cb031e987d809207e72`. Both hashes are recorded in `results/exp018/HASHES.txt`.
