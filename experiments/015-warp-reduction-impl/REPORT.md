# Experiment 015: direct warp reduction for planar PTQ1_0 GEMV

## HYPOTHESIS

For one-column batch-1 PTQ1_0 GEMV, assigning one warp to each output row lets each lane accumulate its K-block dot products in registers and finish with a warp reduction. This should remove the current FP32 shared partial writes and CTA barrier.

## IMPLEMENTATION

Implemented a compile-time candidate in `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`. With `PTQ1_0_PT_WARP_REDUCE=1`, only the `ncols == 1 && ROWS == 1` kernel specialization takes the new path. Four warps own four output rows per CTA; each lane visits K blocks `lane, lane+32, ...`, accumulates locally, and participates in `warp_reduce_sum`. The path supports the existing one-column bias and gate fusion epilogues. It requests zero dynamic shared memory. Other column counts keep the original shared-partial kernel and all dispatch eligibility checks remain unchanged.

The sm_86 CUDA translation unit compiled successfully with the candidate define, and the isolated candidate library loaded from its explicit `LD_LIBRARY_PATH` directory (`ldd` and `LD_DEBUG=libs` evidence is in `results/exp015/raw/loader_version.log`). Candidate/control libraries and executables were removed after testing. The tracked source and production library were restored from the ROWS=1 control copies and SHA-256 checked against those copies: source `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`; library `4b7b09cd570d3f2ca289a97806ab7d2cac0c8c0eb40d73f79e3ec2a752825fde`.

## RESULT

The warp-per-row candidate lost in both reversed-order end-to-end pairs. Its medians were 3.27–3.79% below the ROWS=1 control. It was rejected; there is no candidate change left in production.

## CORRECTNESS

The candidate passed all four selected CTests and 96/96 CUDA-vs-CPU PTQ1_0/PQ2_0 matrix cases. Logs are `results/exp015/raw/ctest.log` and `backend_ops.log`. `tests/run_correctness.sh` was started with the candidate library in `LD_LIBRARY_PATH`, but its build step began rebuilding 394 missing outputs and was stopped at 129/394 to avoid waiting through unrelated full rebuild work. Its test and model-smoke commands were then run directly against the candidate library.

Both PTQ1_0 and PQ2_0 fixed-prompt 32-token CUDA smokes completed successfully. Their generated completion text matched the ROWS=1 reference after disregarding build banner and timing metadata; neither changed a generated token. See `results/exp015/raw/model_smoke_warp.json` and the moved raw stdout captures.

## MICROBENCHMARK

No separate device-event kernel timer was used. The prescribed focused decode benchmark served as the early rejection screen; it showed a clear loss, so the candidate did not qualify for additional profiling.

## END-TO-END IMPACT

Each process used the RTX 3080 (sm_86), PTQ1_0 model, F16 KV, Flash Attention on, 99 GPU layers, batch/microbatch 2048/512, eight CPU threads, 128 decode tokens, seven repetitions, default warmups, and a <=60 C / <=5% idle start gate. Pair 1 ran candidate then control; pair 2 ran control then candidate. The raw JSONs include every sample and GPU telemetry.

| Pair | Context | Warp candidate: median; mean ± sample SD; range (tok/s) | ROWS=1 control: median; mean ± sample SD; range (tok/s) | Median delta |
|---:|---:|---|---|---:|
| 1 | 512 | 79.2462; 79.1019 ± 0.3845; 78.2371–79.2969 | 81.9228; 81.7538 ± 0.3390; 81.0133–81.9603 | -3.27% |
| 1 | 4096 | 76.7615; 76.6681 ± 0.2589; 76.0827–76.7989 | 79.3558; 78.9342 ± 0.7173; 77.4469–79.3997 | -3.27% |
| 2 | 512 | 78.6551; 78.5668 ± 0.2773; 77.9421–78.7273 | 81.7528; 81.6244 ± 0.3476; 80.8376–81.7822 | -3.79% |
| 2 | 4096 | 75.5642; 72.8935 ± 4.4528; 65.0080–76.3456 | 78.4872; 75.3010 ± 4.4752; 69.9211–79.1893 | -3.72% |

Peak whole-GPU memory was 6,803 MiB for the candidate and 6,805 MiB for control. Candidate telemetry ranged 50–77 C and 1,980–2,010 MHz under load; control ranged 59–76 C and 1,980–1,995 MHz under load. GPU utilization reached 100% during both. The long-context tail varied in both builds, but the 512-context loss repeated tightly in both orders and the long-context median loss agreed.

## ANALYSIS

This mapping removed the shared partial stage, but lanes now perform serial K-block loops and each CTA handles four rows at a time. The result suggests the removed shared writes and barrier do not offset the less parallel K work and lower row scheduling flexibility on this GPU. Both contexts and both run orders favor the existing ROWS=1 schedule by a margin well outside sample spread. The new reduction association preserved the deterministic smoke completion and passed the CUDA-vs-CPU arithmetic checks.

## DECISION

**REVERT.** Retain the verified ROWS=1 production source and library. Do not promote the warp reduction.

## FOLLOW-UPS

No immediate follow-up on this exact warp-per-row mapping. A future reduction experiment should preserve more K-block parallelism per row or use a device-event harness to isolate its kernel costs before full decode screening.

## IMPORTANT DISCOVERIES

- A four-warp, one-row-per-warp design was straightforward to route only through the dedicated one-column path, including fusion support.
- Avoiding shared partials did not improve decode: the candidate lost 3.3–3.8% at both tested contexts.
- The existing workspace lacked many intermediate build outputs; the full correctness script's build phase attempted 394 targets. Running its selected CTests, backend-op cases, and smoke commands directly with the candidate library completed the requested coverage without building unrelated targets.
