# Experiment 017: two warps per row for planar PTQ1_0 GEMV

## HYPOTHESIS

Two warps per output row could recover K parallelism lost by experiment 015 while storing only one local sum per warp in shared memory. With four warps per CTA, the candidate handles two output rows and combines two warp sums per row after one barrier.

## IMPLEMENTATION

Added a compile-time candidate in `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`; the reviewable implementation is [candidate.patch](candidate.patch). The route is restricted to the dedicated one-column `mul_mat_vec_ptq1_0_pt<1,1,...>` specialization. Each warp handles `kbx = warp_in_row*32 + lane + 64*t`, accumulates in registers, warp-reduces, and writes one of four row/warp partials. One CTA barrier precedes the two-row final combine. Gate sums use a second four-float shared buffer, and the existing x-bias, gate-bias, SwiGLU, GeGLU, and SwiGLU-OAI epilogues are preserved. The candidate requests 16 bytes shared memory without a gate and 32 bytes with a gate. Multi-column mapping and dispatch are unchanged.

The candidate and source-default ROWS=1 control libraries are isolated in `results/exp017/builds/{candidate,control}`. The candidate CUDA library SHA-256 is `ea2930d93c5d54f5c4b775b0b7df7388d2270a1977e4c8260fdf650b2aeba9bb`; the control SHA-256 is `72d89c0c69200865b2200ef35b94e14b9a6a52a840c17cb031e987d809207e72`. Candidate `mmvq.cu` compiled for `compute_86` / `sm_86`. Ninja's build log was truncated and caused full-target rebuilds; the affected object was compiled with its generated nvcc command and the CUDA library linked from the resulting object. `ldd` and `LD_DEBUG=libs` logs confirm the candidate test process loaded the isolated candidate library.

## RESULT

The candidate lost in both tested contexts in its seven-repetition pair against the contemporaneous ROWS=1 control. The measured median decode deltas were -2.60% at context 512 and -1.82% at context 4096. No second reversed-order pair was warranted after the first pair showed a consistent loss at both contexts.

## CORRECTNESS

The four selected CTests passed: `test-quantize-fns`, `test-ptq1_0-element-map`, `test-ptq1_0-cuda-dot`, and `test-pq2-row-shapes`. The CUDA-vs-CPU backend-op run passed all 96 PTQ1_0/PQ2_0 matrix cases, covering K=1024/5120/6144/17408 and output widths 1/2/4/8. Both fixed-prompt 32-token CUDA model smokes completed on the candidate library. Their completions match `results/baseline_smoke.json` after removing build-banner and timing metadata.

Raw outputs and loader evidence are in [results/exp017/raw](../../results/exp017/raw/).

## MICROBENCHMARK

No independent device-event microbenchmark was needed because the prescribed full decode candidate benchmark produced a clear loss. Benchmark commands, all seven samples per context, GPU telemetry, and process logs are retained in `results/exp017/raw/candidate_pair1.json`, `control_pair1.json`, and their `.log` files.

## END-TO-END IMPACT

Both processes used the RTX 3080 / sm_86, PTQ1_0, F16 KV, Flash Attention, 99 GPU layers, batch/microbatch 2048/512, eight CPU threads, 128 generated tokens, seven repetitions, and default warmups. The cooldown gate was <=60 C and <=5% utilization. Each binary used an explicit `LD_LIBRARY_PATH`; loader checks confirmed library identity.

| Context | Candidate: mean; median; sample SD; range (tok/s) | ROWS=1 control: mean; median; sample SD; range (tok/s) | Median delta |
|---:|---|---|---:|
| 512 | 79.6976; 79.7669; 0.2674; 79.1066–79.9055 | 81.7872; 81.8994; 0.3437; 81.0221–82.0072 | -2.60% |
| 4096 | 77.3202; 77.4338; 0.2433; 76.7787–77.4500 | 78.7003; 78.8722; 0.7370; 77.4482–79.3719 | -1.82% |

Peak whole-GPU memory was 6,803 MiB for the candidate and 6,805 MiB for control. Candidate telemetry sampled 51–69 C and up to 1,995 MHz / 100% utilization; control sampled 60–74 C and up to 1,995 MHz / 100% utilization. Both passed the start gate.

## ANALYSIS

The mapping restores a second warp of K work, but the split is uneven for common model projections with 40 K blocks: one warp has 32 active lanes and the other only 8. Both warps still pay for a warp reduction, shared stores, and a CTA barrier. For larger K rows the split is more even, but the model-wide decode result shows those benefits did not offset the added combine path. The repeated loss at both contexts is outside each run's sample spread.

## DECISION

**REVERT.** The candidate source was removed and `build/bin/libggml-cuda.so.0.21.0` was restored from the source-default control library. The manager independently confirmed that the active source matches the ROWS=1 header hash and the active library matches the saved control library hash `72d89c0c69200865b2200ef35b94e14b9a6a52a840c17cb031e987d809207e72`. After verifying the candidate hash, temporary candidate/control build copies were removed. No experiment code was committed.

## FOLLOW-UPS

Do not promote this two-warp split. Any later multiwarp candidate should account for the uneven lane occupancy when K has fewer than 64 blocks and should first show a focused kernel win before another full model benchmark.

## IMPORTANT DISCOVERIES

- The candidate compiled for sm_86 and preserved one-column fused gate/bias behavior in model smoke output.
- All 96 backend matrix cases passed, including unchanged multi-column widths.
- Two warps per row regressed decode by 1.8–2.6% despite reducing shared partial storage to four sums per matrix and using only one CTA barrier.
