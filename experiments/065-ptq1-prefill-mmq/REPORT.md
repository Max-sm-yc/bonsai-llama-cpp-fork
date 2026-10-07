# Experiment 065: PTQ1_0 prefill MMQ schedule on sm_86

## HYPOTHESIS

Prompt-side PTQ1_0 `MUL_MAT` uses type-143 MMQ and has not had a focused schedule screen. The current sm_86 tile appears resource-heavy; a 128-thread, I=64 tile could allow two CTAs per SM and improve prompt throughput while leaving batch-1 decode on its dedicated GEMV path.

## IMPLEMENTATION

The isolated worktree is `/home/maxsun/autonomous_projects/.worktrees/exp065-ptq1-prefill-mmq`, created at manager HEAD `3f8777568413021bc82e1eea8e7d1b9bb8327943`. The production implementation is code commit `ffb0ef37690b902829ea1158b02b14517ed93c2b`; the worktree's CUDA source is byte-identical to it before and after the screen. The manager HEAD adds research records only.

Built Release/CUDA sm_86 in `build-exp065`:

```bash
cmake -S . -B build-exp065 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
TMPDIR="$PWD/tmp-exp065" cmake --build build-exp065 --parallel 8 --target llama-bench
```

The benchmark model was read-only at `/home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf` (SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`). Baseline `llama-bench` SHA-256 is `85693517f0c078c4801e0836c1fe281577b4fdcdd5cca5b303e6ed070d04495b`; baseline `libggml-cuda.so.0.21.0` is `1898e7cc90e2fc6193511a55eb2ce7c8e6ce0b2583d258138dfce3784ff723ad`. Its RUNPATH and `ldd` resolve the GGML/llama libraries from `build-exp065/bin`; CUDA runtime/libraries resolve from `/usr/local/cuda/lib64`. The retained candidate CUDA library SHA-256 is `e2f331904174e389b754555fa19aa7e24bb3fdf56a6e31aca97d61b5c9ef5830`.

Source fingerprints before and after restore are in `results/exp065/base-identities.sha256`. The only candidate edit changed the 16 PTQ1_0 entries in `ggml/src/ggml-cuda/mmq-config-ampere.cuh` from `(threads=256, occupancy=1, I=128)` to `(threads=128, occupancy=2, I=64)`. J and all other formats were unchanged. The edit was reverted after profiling; the worktree has no production-source diff. The candidate library and all traces remain under `results/exp065/`.

Candidate source/library hashes are recorded in `results/exp065/candidate-identities.sha256`; its source hash is `326fa65e66555186579b308f9f2d6018573d55714fd7c915491555c5b36478d8`.

## RESULT

The seven-repetition PTQ1_0 prefill baseline ran through `benchmark/run.py` from a 47 C / 0% utilization gate with the frozen benchmark settings: `-ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8`, default warmups, and prompts 128/512/2048/4096. Whole-GPU peak use was 6,793 MiB.

| Prompt tokens | Median tok/s | Seven-sample range | Exp041 current reference |
|---:|---:|---:|---:|
| 128 | 1,315.80 | 1,161.04–1,317.40 | 1,291.875 |
| 512 | 1,400.21 | 1,370.50–1,410.04 | 1,377.395 |
| 2,048 | 1,383.66 | 1,382.24–1,386.73 | 1,355.505 |
| 4,096 | 1,359.56 | 1,355.89–1,367.20 | 1,332.260 |

The context-128 range contains one slow sample. The other medians are 1.7–2.1% above Exp041; this is a reproduction of the current build, not an optimization delta.

## CORRECTNESS

Both baseline and candidate profiled llama-bench runs completed. The candidate failed its focused timing screen, so no separate correctness suite or fixed-seed output comparison was run, and no correctness claim is made for the candidate. Existing baseline correctness evidence remains the project record in `research/STATE.md`.

## MICROBENCHMARK/PROFILE

On RTX 3080 / sm_86 (driver 580.178.04, CUDA toolkit 13.2.86, Nsight Systems 2025.6.3.541), a two-repetition 4,096-token prompt capture confirmed that type-143 MMQ is the actual prefill bottleneck: 9,528 `mul_mat_q<(ggml_type)143,128,false>` launches took 5.882 s total (65.5% of captured GPU-kernel time), with 617.3 us mean, 667.5 us median, and 55.4–916.8 us range. Its top-level count and timing are not inferred from the older mixed decode trace.

Exact unprofiled baseline command:

```bash
python3 benchmark/run.py \
  --model PTQ1_0=/home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --binary /home/maxsun/autonomous_projects/.worktrees/exp065-ptq1-prefill-mmq/build-exp065/bin/llama-bench \
  --modes prefill --contexts 128 512 2048 4096 --repetitions 7 \
  --cooldown-temp-c 60 --output results/exp065/prefill-baseline.json
```

Profile command template (the raw captures use `profile_prefill_ctx4096`, `profile_candidate_ctx4096`, `profile_base_ctx128`, `profile_candidate_ctx128`, `profile_base_ctx512`, and `profile_candidate_ctx512` output prefixes and substitute the prompt length):

```bash
TMPDIR="$PWD/tmp-exp065" nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --cuda-memory-usage=true --force-overwrite=true \
  --output results/exp065/profile_prefill_ctx4096 \
  build-exp065/bin/llama-bench \
  -m /home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 \
  -r 2 -o json -p 4096 -n 0 -d 0
```

At 128/512, the candidate library was selected using `LD_LIBRARY_PATH=/home/maxsun/autonomous_projects/.worktrees/exp065-ptq1-prefill-mmq/results/exp065/candidate-runtime`; controls used the same binary and build-local baseline library. `ldd` outputs for both loader selections are retained. The candidate run at 4,096 used the candidate library in `build-exp065/bin` before the baseline library was restored.

Each raw `.nsys-rep` has a corresponding SQLite export and `cuda_gpu_kern_sum`/`cuda_api_sum` CSV. The export commands were:

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  --force-overwrite=true --force-export=true \
  --output results/exp065/profile_prefill_ctx4096.stats.csv \
  results/exp065/profile_prefill_ctx4096.nsys-rep
nsys export --type sqlite --force-overwrite=true \
  --output results/exp065/profile_prefill_ctx4096.sqlite \
  results/exp065/profile_prefill_ctx4096.nsys-rep
```

Every base launch used a 256-thread `(32,8,1)` block, 254 registers/thread, and 57,856 B dynamic shared memory. The observed `(gridX,gridY,gridZ)` shapes were `(68,1,1)`, `(192,1,1)`, `(320,1,1)`, `(384,1,1)`, and `(544,1,1)`; J=128 was selected. The `mul_mat_q_stream_k_fixup` family added 73.6 ms over 3,816 launches. Source dispatch is `ggml_cuda_mul_mat_q_switch_type` → PTQ1_0 `mul_mat_q_case` → `mul_mat_q_switch_J` / `launch_mul_mat_q`; PTQ1_0's tile unpack is in `mmq-load-tiles.cuh`, and its dot path is in `mmq-vec-dot.cuh`.

The candidate compiled with 254 registers/thread, 128 threads, and 38,400 B dynamic shared memory, so the register/shared-memory arithmetic permits two CTAs per SM (65,024 of 65,536 registers and 76,800 B shared memory). Nsight Systems observed `(32,4,1)` blocks and no type-143 stream-K fixups at 512/4096. Achieved occupancy was not directly measured; this experiment did not use Nsight Compute counters.

| Prompt | Base type-143 MMQ | Candidate type-143 MMQ | Candidate delta |
|---:|---:|---:|---:|
| 128 | 183.85 ms / 1,191 launches | 212.59 ms / 1,191 launches | +15.6% |
| 512 | 723.67 ms / 1,191 launches | 751.42 ms / 1,191 launches | +3.8% |
| 4,096 | 5.882 s / 9,528 launches | 6.078 s / 9,528 launches | +3.3% |

At prompt 128, candidate/base mean kernel time was 178.5/154.4 us per launch; at 512 it was 630.9/607.6 us. At 4,096 it was 637.9/617.3 us. The 4,096 candidate median was 673.0 us versus 667.5 us base. Despite lower shared memory and removal of fixup launches at the larger prompts, the total type-143 family became slower. Full per-kernel distributions, launch grids, and resource output are retained in the raw CSVs and `ptxas-resources-*.txt`.

Nsight-instrumented llama-bench samples (two repetitions, default warmups; not unprofiled end-to-end results) were: prompt 128, 1,060.81 candidate vs 1,167.75 base tok/s; 512, 1,324.49 vs 1,346.63; 4,096, 1,331.72 vs 1,352.16. These are diagnostic only and include profiler overhead.

## END-TO-END IMPACT

No unprofiled candidate A/B or decode benchmark was run because the candidate lost the focused type-143 timing screen at all three profiled prompt sizes. Therefore there is no candidate end-to-end or decode performance claim. The seven-repetition baseline measurements above remain the only unprofiled workload results. Baseline whole-GPU peak was 6,793 MiB; no candidate whole-GPU peak was sampled outside Nsight.

## ANALYSIS

The initial hypothesis had a concrete source/profile basis: base MMQ used 254 registers/thread and 57.9 KB shared memory, constraining its 256-thread CTA to one resident CTA per SM by either resource. The candidate's 128-thread/I=64 tile lowered shared memory to 38.4 KB and its register budget allows two CTAs per SM. It nevertheless slowed the MMQ family 3.3% at 4,096, 3.8% at 512, and 15.6% at 128. The unprofiled baseline's 6,793 MiB peak and captured model allocation footprints were stable; the candidate showed no resource-driven speed improvement. A prompt-length guarded form is not justified by the short-prompt captures.

## DECISION

**REVERT.** Restored `mmq-config-ampere.cuh` and the build-local CUDA library to the baseline hash. The candidate CUDA library is preserved separately. No manager checkout files were changed.

## FOLLOW-UPS

No follow-up schedule candidate is supported by this screen. Revisit type-143 MMQ only with a different measured dataflow or code-generation premise. The standard benchmark settings and current batch-1 PTQ1_0 GEMV remain unchanged.

## IMPORTANT DISCOVERIES

- Type-143 MMQ, rather than a mixed-trace inference, consumes 65.5% of this 4,096-prompt capture.
- Current prefill MMQ selects J=128 and shows five recurrent row-grid sizes, with 254 registers/thread and 57,856 B dynamic shared memory.
- A smaller I=64 / 128-thread type-143 schedule fits a two-CTA resource budget but is slower at prompt lengths 128, 512, and 4,096; no short-prompt exception emerged.
- Nsight Compute counters were not used, so the two-CTA statement is a resource-budget capacity, not a measured achieved-occupancy result.
