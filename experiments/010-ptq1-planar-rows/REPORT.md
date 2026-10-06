# Experiment 010: PTQ1_0 planar kernel row scheduling

## HYPOTHESIS

The RTX 3080 routes batch-1 PTQ1_0 through the dedicated planar-transposed `mul_mat_vec_ptq1_0_pt` kernel. Its four-row work item may not balance activation reuse, register pressure, and CTA occupancy well. I tested one, two, and eight rows per item against the existing four-row schedule.

## IMPLEMENTATION

Added a one-column `PTQ1_0_PT_ROWS_1` control; kept the existing `ptq1_0_pt_rows_per_cta` heuristic and all dot arithmetic unchanged. Tested exact CUDA sm_86 builds of ROWS=1/2/4/8. The winning source change is in `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`; candidate binaries and libraries used for the comparisons remain locally under [results/exp010/builds](../../results/exp010/builds) and are excluded from Git.

An initial benchmark set was invalid: copied executables retained an absolute `build/bin` RUNPATH, so each process loaded that directory's library rather than its archived variant. Those JSONs remain under `results/exp010/screen` and `results/exp010/followup` as no-op controls and are excluded from every result below. Corrected runs explicitly set `LD_LIBRARY_PATH` per process. `ldd` and `LD_DEBUG=libs` evidence for baseline and all row variants is in [results/exp010/raw](../../results/exp010/raw).

## RESULT

ROWS=1 beat ROWS=4 in both reversed-order seven-repetition process pairs at contexts 512 and 4096. A direct seven-repetition comparison also favored ROWS=1 over ROWS=2 in both pairs. ROWS=8 lost substantially.

| Comparison, pair | Context | Candidate median | Control median | Median delta |
|---|---:|---:|---:|---:|
| ROWS=1 vs 4, pair 1 | 512 | 81.8196 | 77.9198 | +5.00% |
| ROWS=1 vs 4, pair 1 | 4096 | 79.3098 | 74.9338 | +5.84% |
| ROWS=1 vs 4, pair 2 | 512 | 81.7452 | 77.8996 | +4.94% |
| ROWS=1 vs 4, pair 2 | 4096 | 78.7377 | 75.0929 | +4.86% |
| ROWS=1 vs 2, pair 1 | 512 | 81.7766 | 81.2009 | +0.71% |
| ROWS=1 vs 2, pair 1 | 4096 | 78.5319 | 78.0209 | +0.65% |
| ROWS=1 vs 2, pair 2 | 512 | 81.7677 | 81.1773 | +0.73% |
| ROWS=1 vs 2, pair 2 | 4096 | 78.5392 | 77.9710 | +0.73% |

The corrected three-repetition screen produced these medians and ranges (tok/s):

| Rows/item | Context | Candidate median (range) | ROWS=4 median (range) | Delta |
|---:|---:|---:|---:|---:|
| 2 | 512 | 81.2603 (80.3983–81.2912) | 78.0716 (77.1673–78.0937) | +4.08% |
| 2 | 4096 | 78.7907 (78.0946–78.8194) | 75.7962 (75.1207–75.8383) | +3.95% |
| 8 | 512 | 65.0142 (64.3626–65.0282) | 77.9025 (77.1354–77.9345) | -16.54% |
| 8 | 4096 | 63.4206 (62.9413–63.4387) | 75.6351 (75.0822–75.6506) | -16.15% |

The isolated result files are `screen/rows2_baseline_isolated.json`, `rows2_rows2_isolated.json`, `rows8_rows8_isolated.json`, and `rows8_baseline_isolated.json`.

## CORRECTNESS

The final source-default ROWS=1 library passed all four selected CTests, all 96 CUDA-vs-CPU PTQ1_0/PQ2_0 matmul cases, and both 32-token model smoke runs. The final logs are [final_ctest.log](../../results/exp010/final_ctest.log), [final_backend_ops.log](../../results/exp010/final_backend_ops.log), and [final_model_smoke.json](../../results/exp010/final_model_smoke.json). Earlier exact ROWS=2 and ROWS=8 builds passed the same coverage; see their `ctest_rows*`, `backend_ops_rows*`, and `model_smoke_rows*` artifacts.

## MICROBENCHMARK

No standalone kernel timer was used. Static cubin usage was 108 registers/thread for default ROWS=4, 76 for ROWS=1, 92 for ROWS=2, and 128 for ROWS=8; all had zero stack/local spills. The raw report is [cubin_resources.txt](../../results/exp010/raw/cubin_resources.txt). Nsight Compute counters remain unavailable (`ERR_NVGPUCTRPERM`).

## END-TO-END IMPACT

Each row1-vs-baseline process used seven repetitions, contexts 512/4096, 128 generated tokens, the same 60°C/5% idle gate, and captured GPU telemetry. The table below gives every process's sample median, mean ± sample SD, and full range; throughput is in tok/s.

| Pair | Context | ROWS=1: median; mean ± SD; range | ROWS=4: median; mean ± SD; range |
|---:|---:|---:|---:|
| 1 | 512 | 81.8196; 81.7247 ± 0.3507; 80.9755–82.0144 | 77.9198; 77.8080 ± 0.3167; 77.0922–77.9635 |
| 1 | 4096 | 79.3098; 79.1764 ± 0.3252; 78.6844–79.4455 | 74.9338; 73.5848 ± 2.7496; 68.4418–75.6284 |
| 2 | 512 | 81.7452; 81.6151 ± 0.3542; 80.8127–81.7683 | 77.8996; 77.8014 ± 0.3087; 77.1037–77.9575 |
| 2 | 4096 | 78.7377; 72.6902 ± 10.6250; 55.3784–79.2375 | 75.0929; 73.8850 ± 2.9462; 67.5431–75.6095 |

The median improvement repeats in both process orders. Context-4096 tails remain unstable in both builds: ROWS=1 had two slow samples in pair 2, which drove its arithmetic mean (72.69 tok/s) below the baseline mean (73.89 tok/s) and its time-weighted aggregate below baseline for that pair. The median result supports ROWS=1, while the tail variation limits any sustained-throughput claim.

In the direct ROWS=1/ROWS=2 comparison, ROWS=1 also won both contexts in both process orders. The seven-sample ranges were 80.95–81.83 vs 80.40–81.23 tok/s at context 512 in pair 1, and 49.02–79.19 vs 69.54–78.74 at context 4096. In pair 2, ranges were 80.94–81.81 vs 80.33–81.22 at 512, and 67.63–79.26 vs 65.63–78.72 at 4096. The full distributions are in [rows1_vs_rows2](../../results/exp010/rows1_vs_rows2).

The final source-default build also reproduced the ROWS=1 result in a candidate-first seven-repetition pair against the archived baseline. Its 512-context median was 81.9463 tok/s (81.7846 ± 0.3455; 81.0273–81.9951), versus 77.8702 (77.7655 ± 0.2859; 77.1197–77.8994). At context 4096 it was 79.4135 (79.2617 ± 0.2797; 78.6566–79.4247), versus 74.9311 (70.5070 ± 8.6220; 53.6652–75.5569). This final-build check is in [final_build_check](../../results/exp010/final_build_check); `ldd` confirms it loaded the source-default `build/bin` library.

All corrected runs peaked at 6,805 MiB whole-GPU memory, within the 10 GiB limit. The seven-repetition runs started at 58–60°C; telemetry reached 77°C, with loaded SM clocks between 1,830 and 1,995 MHz. No prefill or combined follow-up was run after the decode result selected the row schedule.

## MANAGER VERIFICATION

The manager performed a fresh build of all 394 targets from the checked-out source, then independently reran the correctness path: 4/4 CTests, 96/96 CUDA-vs-CPU matmul cases, and both fixed 32-token model smokes passed. Logs are `results/exp010/manager_correctness_build.log`, `manager_ctest.log`, `manager_backend_ops.log`, and `manager_model_smoke.json`.

After that rebuild, the exact `build/bin` ROWS=1 library was A/B-tested candidate-first against the archived baseline with separate verified `LD_LIBRARY_PATH` settings. Both processes used the same 60°C gate, 7 repetitions, 128 generated tokens, and contexts 512/4096. At context 512, ROWS=1 measured 82.2218 tok/s median (82.1258 ± 0.3182; range 81.41–82.33), versus 77.9966 (77.9098 ± 0.2965; 77.26–78.11), a +5.42% median delta. At context 4096, it measured 79.6967 (79.6061 ± 0.2680; 79.00–79.75), versus 75.6584 (75.5847 ± 0.2350; 75.09–75.78), a +5.34% delta. Both peaked at 6,805 MiB. Data and runner are in `results/exp010/final_rebuilt_pair/` and `final_rebuilt_pair.py`.

The manager also re-profiled the rebuilt kernel with Nsight Systems. The three active GEMV variants remain 60.4% of kernel time (1.166 s combined versus 1.253 s in the reference trace); see `PROFILE.md` and `results/profile/ptq1_decode512_rows1.stats.csv_cuda_gpu_kern_sum.csv`.

## ANALYSIS

Reducing rows per item lowers register use and exposes more independent row work. ROWS=1 improves batch-1 decode by about 4.9–5.8% versus ROWS=4 in the paired medians, and edges ROWS=2 by about 0.7% in a separate reversed-order comparison. ROWS=8 reduces throughput sharply. The ROWS=1 long-context median gain is repeatable, but occasional slow samples in both builds leave sustained tail behavior noisy.

## DECISION

**KEEP ROWS=1.** It is the fastest exact candidate in the corrected paired tests. The tracked CUDA source defaults the one-column PTQ1_0 path to ROWS=1, and `build/bin/libggml-cuda.so` was rebuilt from that source with `CMAKE_CUDA_FLAGS` empty. The manager independently rebuilt the tree and verified the A/B and correctness results; the retained change is commit `9fa97200e68fd798ef027470c8e420172a0ac719`.

## FOLLOW-UPS

Keep the current row tile. Revisit only with permitted kernel counters or a kernel-level timer that can explain the long-context tail variation. Do not use the archived no-op timings for further tuning decisions.

## IMPORTANT DISCOVERIES

- The active RTX 3080 batch-1 path is the dedicated planar-transposed kernel; the one-column row item can be changed without altering activation layout or dot arithmetic.
- ROWS=1 uses 76 registers/thread versus 108 for ROWS=4, with no spills. ROWS=2 uses 92 registers and is close, but slower in both direct pairs; ROWS=8 uses 128 registers and loses substantially.
- The original isolated-binary timing scripts did not override absolute RUNPATH and therefore benchmarked a shared library. `LD_LIBRARY_PATH`, `ldd`, and `LD_DEBUG=libs` now verify every corrected process's library.
- An interrupted cache-flag-triggered Ninja rebuild removed several incomplete object outputs; a single CUDA target recovery build repopulated them. Final `CMAKE_CUDA_FLAGS` is empty, and the active CUDA library was then compiled with the source-default ROWS=1 specialization.
