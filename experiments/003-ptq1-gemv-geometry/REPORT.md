# Experiment 003: PTQ1_0 batch-1 GEMV warp geometry

## HYPOTHESIS

The PTQ1_0 batch-1 GEMV launch uses four warps per CTA on this sm_86 GPU. A lower warp count could improve CTA scheduling and expose more independent output rows; a higher count could improve latency hiding for the unpack and dot-product loop. This experiment changes only the warp count for the PTQ1_0, one-column path. The baseline's four-warp geometry remains the default, and no prefetch policy was changed.

## IMPLEMENTATION

Added an overrideable `GGML_CUDA_PTQ1_0_GEMV_NWARPS` selection in `ggml/src/ggml-cuda/mmvq.cu`; default value 4 is the original geometry. Built and retained 2-warp and 8-warp variants, alongside the original 4-warp build. `results/exp003/warp-count.patch` contains the temporary source diff. No arithmetic, ternary unpack, activation layout, reduction order, row scheduling outside the warp-count-derived launch, or PQ2_0 code changed.

Each variant has its own `bin/` directory with `llama-bench`, `llama-cli`, and matching shared libraries under `results/exp003/{baseline,nwarps2,nwarps8}/bin/`. The baseline directory was captured from the verified reference build before edits. Both candidates were compiled from the same HEAD and Release/CUDA sm_86 configuration; the 8-warp CUDA library was linked with only `mmvq.cu` recompiled at the alternate macro value. To reproduce a candidate from source, configure the same build with `-DCMAKE_CUDA_FLAGS=-DGGML_CUDA_PTQ1_0_GEMV_NWARPS=N`, build `llama-bench`, then copy the complete `build/bin/` directory into the corresponding variant directory. The workspace's tracked source and CMake CUDA flags were restored afterward.

The paired runner is `benchmark/exp003_geometry_ab.py`. It alternates baseline/candidate process order, waits for <=5% reported GPU utilization and the configured temperature gate before every process, captures GPU telemetry throughout each run, and stores raw stdout/stderr under `results/exp003/raw/`. One initial 8-warp screen at a 62 C gate was interrupted when the idle GPU would not return below the gate; the two completed process stdout JSON outputs are retained in `results/exp003/aborted_raw/`; small stderr gate messages are locally ignored and are excluded from the analysis.

## RESULT

The 2-warp screen used two alternating pairs, three `llama-bench` samples per process, and a 62 C start gate. It showed no gain:

| Context | Baseline 4-warp median | Candidate 2-warp median | Delta |
|---:|---:|---:|---:|
| 512 | 78.1697 tok/s | 78.1701 tok/s | +0.0005% |
| 4096 | 75.8491 tok/s | 75.8132 tok/s | -0.0474% |

The 8-warp comparison used five alternating pairs, three samples per process, and a 64 C start gate. Across the 15 samples per variant and context:

| Context | Baseline 4-warp median (range) | Candidate 8-warp median (range) | Delta |
|---:|---:|---:|---:|
| 512 | 77.6937 (76.8163–77.8778) tok/s | 77.6744 (76.7865–77.7917) tok/s | -0.0248% |
| 4096 | 75.6488 (74.8659–75.7217) tok/s | 75.6681 (74.8683–75.6961) tok/s | +0.0255% |

The five paired median deltas for 8 warps were -0.0963%, +0.0108%, -0.0452%, -0.0422%, -0.0642% at context 512 and -0.0241%, -0.0206%, +0.0126%, -0.0058%, +0.0508% at context 4096. The sign changes by context and pair. The highest apparent gain is far below the run-to-run spread.

## CORRECTNESS

`tests/run_correctness.sh` passed on the 2-warp build:

- 4/4 selected CTest cases passed: quantization functions, PTQ1_0 element map, PTQ1_0 CUDA dot, and PQ2_0 row shapes.
- 96/96 CUDA-vs-CPU `MUL_MAT` cases passed for PTQ1_0 and PQ2_0, including K sizes 1024, 5120, 6144, and 17408 and batch widths 1, 2, 4, and 8.
- PTQ1_0 and PQ2_0 each produced a non-empty deterministic 32-token CUDA completion.

The same 96-case backend-ops suite and both model smoke runs also passed on the 8-warp binary. Smoke records and the backend-ops log are in `results/exp003/`.

## MICROBENCHMARK

No standalone kernel microbenchmark was run. Nsight Compute counters remain unavailable on this host (`ERR_NVGPUCTRPERM`), and a per-kernel timer is not available in the repository harness. The measurements here are end-to-end `llama-bench` decode results, not a kernel-only speed claim.

## END-TO-END IMPACT

The fixed workload used the PTQ1_0 GGUF, `-ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8`, `-p 0 -n 128 -d CONTEXT`, with contexts 512 and 4096. Each process used three repetitions and llama-bench's default warmup. Baseline and candidate used identical flags and their own isolated binary/library directories. The 8-warp comparison used five pairs per context, alternated process order, and gated every process at 64 C and <=5% utilization. Gate temperature was 64 C in all 20 processes; median cooldown wait was 25.2 seconds (maximum 45.3 seconds). In-process telemetry ranged from 64–77 C, 210–1980 MHz SM clock, and 0–100% GPU utilization.

The measured decode difference was -0.025% at context 512 and +0.026% at context 4096. These opposing, sub-noise deltas are not a sustained decode improvement. No 512-token sustained follow-up was warranted because the early result was not promising.

## ANALYSIS

Changing the CTA warp count from four to two or eight did not materially change batch-1 decode throughput on the RTX 3080. The 2-warp screen is tied at context 512 and marginally slower at 4096; the five-pair 8-warp test slightly favors baseline at context 512 and is effectively tied at 4096. Within a pair, process-to-process and sample variation is larger than either candidate's median delta. The per-process sample ranges overlap almost completely.

This suggests launch warp count alone is not the limiting factor for the dominant PTQ1_0 GEMV time. It does not identify whether unpack instruction throughput, weight traffic, or another part of the dot-product loop is limiting; NCU instruction and bandwidth counters could not be collected. Since only launch geometry changed, there was no new arithmetic or ternary unpack path to validate separately.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** Neither tested geometry showed a repeatable end-to-end gain. Restored `ggml/src/ggml-cuda/mmvq.cu` to its verified reference content, cleared the temporary `CMAKE_CUDA_FLAGS` override, and restored `build/bin/` from the saved baseline build. The candidate binaries and matching libraries remain available for inspection; no changes were committed.

## FOLLOW-UPS

- Keep the 4-warp baseline for PTQ1_0 batch-1 GEMV.
- If revisiting this kernel, use a kernel-specific timing harness or obtain permitted profiling counters before changing unpack scheduling or vectorization. Do not repeat the prefetch on/off question from experiments 001/002.

## IMPORTANT DISCOVERIES

- The verified four-warp launch is not measurably improved by two or eight warps on the RTX 3080 at either tested context.
- The 8-warp paired result changes sign by context, and its largest paired median difference is only about 0.10%.
- The benchmark process start gate can be met consistently at 64 C, while the initial attempt to use 62 C stalled after the GPU's idle temperature rose; incomplete screen logs were retained but excluded.
- The packed PTQ1_0 arithmetic/unpack path stayed unchanged. Both alternate geometries passed numerical backend coverage and PTQ1_0/PQ2_0 real-model smoke inference.

## FINAL SOURCE/BINARY STATE

Both candidate geometry changes were reverted. The checked-out CUDA source matches the verified four-warp baseline, `CMAKE_CUDA_FLAGS` is empty, and the default `build/bin/` matches the saved baseline build. The experimenter made no commit; the manager records the report and measurements in a research-only commit. Candidate binaries remain as local ignored artifacts under `results/exp003/{baseline,nwarps2,nwarps8}/bin/`.

## MANAGER AUDIT (post-exp009 dispatch review)

The RTX 3080 host layout selector routes batch-1 PTQ1_0 to the planar-transposed `GGML_CUDA_Q8_1_PT` layout. The plain one-column matvec dispatcher then invokes the dedicated `mul_mat_vec_ptq1_0_pt` kernel and returns before instantiating the generic `mul_mat_vec_q` path whose `calc_nwarps` value this experiment changed. Thus the 2/4/8-warp binaries used the same active PTQ1_0 batch-1 kernel. The measured decode ties are a useful no-op control, but provide no evidence about warp geometry in the actual RTX 3080 kernel.
