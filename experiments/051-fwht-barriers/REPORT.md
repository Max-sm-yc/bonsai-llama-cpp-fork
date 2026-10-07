# Experiment 051: one-barrier cross-warp FWHT stages

## HYPOTHESIS

The cross-warp stages in `ggml_cuda_fwht_block_butterfly` stored registers to shared memory, synchronized, loaded a partner, computed, then synchronized a second time before the next stage could overwrite shared memory. Alternating two shared-memory banks may let the next stage's publication barrier also wait for all previous-stage reads, cutting cross-warp barriers in half without changing arithmetic or tile parallelism.

The active paths are N=1024/NT=256 in `fwht_quantize_q8_1` and N=1024/NT=1024 in Exp036 `fwht_rms_quantize_q8_1`. Exp047 groups these under the 0.752 ms/token QKV activation-preparation family. Exp020's NT sweep was not repeated.

## IMPLEMENTATION

In detached worktree `/home/maxsun/autonomous_projects/.worktrees/exp051-fwht-barriers` at base commit `21269b75c363b935cb98a9d4bc2d89df0ed35bc4`, the helper alternated two N-float banks for N<=4096. For each cross-warp stage, threads write their current register values into the active bank, execute one `__syncthreads`, read their partners from that bank, compute, and toggle banks. The next stage writes only after each thread has completed its previous-stage partner read; its barrier cannot complete until every thread has arrived, so no previous read remains when a bank is reused. The just-written bank differs from the bank read by the current stage. No shared data is consumed after the final stage, so a trailing barrier is unnecessary. Every participating thread follows the same unbranched barrier sequence.

This proof applies to every cross-warp stage for the helper's power-of-two N, NT-divides-N, NT-multiple-of-warp-size shapes. It also applies when NT equals one warp (there are no cross-warp stages). The first build exposed a generic N=8192 instantiation in the standalone FWHT path; doubling its shared allocation would exceed sm_86's 48 KiB per-block limit. N>4096 therefore retains the original one-bank/two-barrier loop, with N-float storage. All three helper callers use a compile-time conditional allocation. This keeps other template instantiations valid.

Configured with `cmake -S . -B build-exp051 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_TESTS=ON`, then built `llama-bench` and `test-fwht-rms-q8` using `TMPDIR="$PWD/build-exp051/tmp" cmake --build build-exp051 --target llama-bench test-fwht-rms-q8 -j 4`. The default `/tmp` user quota was near exhaustion, so CUDA compiler temporary files were directed into the worktree. Candidate library SHA-256: `3f3e724887ba9abf4f9e6c4aafd9b5f26079286872fea8526164ea3c5abc446d`. Production control library SHA-256: `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`.

## RESULT

**REVERT.** The candidate repeatedly improved isolated graph timing for both active kernels, but two reversed-order matched decode pairs did not show a repeatable end-to-end gain. Production remains unchanged and the three experimental source edits have been restored to the base commit.

## CORRECTNESS

- Candidate `build-exp051/bin/test-fwht-rms-q8` passed the Exp036 direct reference for one and three 5120-wide rows. Maximum dequantized error was 0.539778 and 0.542589 stored scales; block sums were exact. Output covers the active N=1024/NT=1024 kernel.
- `results/exp051/fwht_quant_test.cpp` directly exercised the generic N=64, 128, 256, 512, 1024, and 2048 PT quantizer shapes (NT=64, 128, then 256). Candidate output was byte-identical to the production control at every size, including scales and sums; saved bytes and SHA-256 comparisons are under `results/exp051/{candidate_bytes,control_bytes}/`. Host-reference dequantization error was below 0.75 stored scale at each size and sums were exact. A few host quantized values differ at rounding boundaries because the independent host butterfly accumulates in a different floating-point order; candidate and control bytes still match exactly.
- Compute Sanitizer memcheck and racecheck passed for both direct test executables: zero errors and zero hazards/warnings. Logs are `sanitizer_{rms,generic}_{memcheck,racecheck}.txt`.

## MICROBENCHMARK

On the RTX 3080, a CUDA graph held 64 actual wrapper calls per replay. Nine CUDA-event samples per executable run were reduced to a median and range per call. Six repeated candidate/control pairs alternated order; all six favored the candidate for both shapes.

| Kernel | Control median across run medians (range) | Candidate median across run medians (range) | Change |
|---|---:|---:|---:|
| `fwht_quantize_q8_1`, N=1024/NT=256 | 2.4275 μs (2.4160–2.4320) | 2.3980 μs (2.3840–2.4160) | 1.22% faster |
| `fwht_rms_quantize_q8_1`, N=1024/NT=1024 | 3.8720 μs (3.8660–3.8855) | 3.6430 μs (3.6320–3.6580) | 5.91% faster |

Each underlying run and its nine-sample ranges are retained in `results/exp051/repeat{1..6}_{candidate,control}.txt`; harness and initial runs are also retained there. Candidate/control processes used the candidate-built harness. `LD_DEBUG=libs` records confirm control loaded the production library and candidate loaded the isolated candidate library. GPU start temperature/utilization for the repeated timing sequence was 47°C/0%.

`cuobjdump` disassembly confirms the candidate emitted three `BAR.SYNC` instructions for N=1024/NT=256 and six for N=1024/NT=1024 (the latter includes one RMS-reduction barrier plus five FWHT stages). The original control emits six and eleven respectively. Candidate resource usage is 30 registers and 8192 shared bytes versus control 30 registers and 4096 bytes for generic PT N=1024/NT=256; RMS is 22 registers and 8320 shared bytes versus control 24 registers and 4224 bytes. No stack or local spills were reported. Relevant function SASS and full resource listings are in `results/exp051/*_nt*.sass.txt` and `*_resources.txt`.

## END-TO-END IMPACT

Ran `benchmark/run.py` twice in forward order (control→candidate, then candidate→control), with both decode contexts in each invocation: PTQ1_0, 128 generated tokens, seven repetitions, `-ngl 99`, Flash Attention on, batch/ubatch 2048/512, eight CPU threads, F16 K/V. The model was the existing PTQ1_0 GGUF in the manager workspace; both arms used the same `build-exp051/bin/llama-bench`, with only `LD_LIBRARY_PATH` changed. `LD_DEBUG=libs` verified each library. Commands, exact per-repetition samples, telemetry, peak memory, and stdout are retained in `results/exp051/e2e_{control,candidate}_pair{1,2}.*` and `results/exp051/raw/`.

| Pair order | Context | Control median (samples tok/s) | Candidate median (samples tok/s) | Pair change |
|---|---:|---|---|---:|
| Control then candidate | 512 | 83.9063 (82.9520–83.9375) | 83.7498 (82.8958–83.8108) | -0.19% |
| Control then candidate | 4096 | 81.2993 (80.5139–81.3562) | 81.1498 (80.4715–81.2186) | -0.18% |
| Candidate then control | 512 | 83.2668 (82.4851–83.3952) | 83.5453 (82.5936–83.5652) | +0.33% |
| Candidate then control | 4096 | 80.0278 (67.7575–80.8040) | 80.1164 (72.9642–80.8794) | +0.11% |

Median-of-pair medians changed +0.07% at context 512 and -0.04% at 4096. Each arm started at <=60°C and 0% utilization (gates: 46/54°C in pair 1; both 60°C in pair 2, with the second run waiting 50 seconds). Peak VRAM was 6803 MiB for all four runs. Pair 2 had slow-tail repetitions at 4096 for both arms. The two pairs do not establish an end-to-end improvement.

## ANALYSIS

The shared-memory dependency argument holds and sanitizer reported no races. Generated code removes exactly the second barrier at each active cross-warp stage. This shortens the active kernels in isolation, especially the five-stage NT=1024 transform. The extra 4 KiB of shared storage did not cause local spills, but this kernel family contributes only about 0.752 ms/token to decode, and other decode work dominates. The matched model samples were already near the prior model's normal run-to-run variation; paired changes switched sign or stayed within 0.2%.

## DECISION

**REVERT.** Retain the current two-barrier implementation. A clear focused kernel gain alone does not qualify when matched PTQ1_0 decode is flat within noise. No production files or libraries were changed. Fixed-seed PTQ1_0/PQ2_0 CLI smoke was not run because the matched decode did not meet the E2E qualification gate; direct output checks, sanitizer, and actual model decode A/B did run.

## FOLLOW-UPS

Revisit only if a way to remove the added shared-memory footprint or further cut active decode work gives a larger, repeatable whole-model gain. Do not repeat Exp020's CTA-width sweep.

## IMPORTANT DISCOVERIES

- Ping-pong is race-free when the next stage first writes its alternate bank and then synchronizes: arrival at that barrier proves every thread completed the previous stage's read before that old bank is reused.
- Shared-memory doubling would make the N=8192 generic instantiation exceed the sm_86 per-block limit; the high-N fallback is required for compile-time compatibility.
- The active kernels gained 1.23% and 6.29% in focused graph replay, but the QKV-preparation contribution was not large enough to yield a measurable decode gain in two reversed-order pairs.
