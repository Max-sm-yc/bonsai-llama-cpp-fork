# Experiment 002: PTQ1_0 prefetch paired A/B

## HYPOTHESIS

Disabling the explicit next-K-block L2 prefetch in the PTQ1_0 batch-1 GEMV may reduce instruction overhead and improve real decode, especially at long context. Experiment 001 did not isolate the variants in paired runs.

## IMPLEMENTATION

The on and off binaries were compiled from the same source revision (`aa9edbb22bf9970f3189331aac0b4681e715b494`) and Release/CUDA configuration (`CMAKE_CUDA_ARCHITECTURES=86`, CUDA enabled). The off variant removes only the existing PTQ1_0 `prefetch.global.L2` conditional in `ggml/src/ggml-cuda/mmvq.cu`; the complete diff is [prefetch-off.patch](../../results/exp002/prefetch-off.patch). No other kernel behavior changed. Both runtime `bin` directories were preserved under `results/exp002/{on,off}/` so each executable loads its matching shared libraries.

Build commands, run with the corresponding source variant checked out:

```bash
cmake --build build --parallel 4 --target llama-bench
```

The on build used the original source. For the off build, the change in `results/exp002/prefetch-off.patch` was applied before the same command. The source was restored afterward and the baseline build was rebuilt.

The reusable paired runner is [prefetch_ab.py](../../benchmark/prefetch_ab.py). It runs one context and mode per process, alternates order (`on/off`, `off/on`, `on/off`), waits before every process for at most 62 C and <=5% GPU utilization, and captures temperature, utilization, power, and SM clock during the process. The first attempted 60 C run stalled after the GPU's idle reading settled at 61 C; it was stopped before completion and its raw process outputs are preserved under `results/exp002/aborted_raw/`. An initial 62 C attempt at 512-token decode was also stopped when the idle sensor stayed above its gate; those raw process outputs are in the same folder. The complete matrix used a consistent 62 C gate for both variants in all pairs. Actual gate readings were 62 C and 0% utilization; median cooldown waits were 30–81 seconds depending on workload.

## RESULT

At 128 generated tokens, the prefetch-off candidate showed no consistent performance advantage. The median of the nine raw samples per variant is effectively tied at all four workloads; paired differences change sign across pairs and stay within 0.11%. A sustained 512-token decode follow-up at context 4096 initially leaned toward prefetch-on in five three-repetition pairs. Two further reversed-order pairs used seven repetitions per process to inspect samples 3–7; the direction reversed with run order. The tail effect is therefore not robust.

| Workload | Prefetch on median (range) tok/s | Prefetch off median (range) tok/s | Off vs on | Paired off/on deltas by pair |
|---|---:|---:|---:|---|
| Decode, context 512 | 77.865 (77.013–77.900) | 77.842 (77.022–77.861) | -0.030% | -0.058%, +0.010%, -0.060% |
| Decode, context 4096 | 75.810 (75.007–75.856) | 75.785 (74.986–75.864) | -0.033% | -0.100%, +0.037%, +0.032% |
| Combined, context 512 | 314.211 (313.971–314.532) | 314.097 (313.874–314.596) | -0.036% | -0.108%, +0.055%, -0.035% |
| Combined, context 4096 | 886.259 (884.998–890.002) | 886.305 (884.965–890.017) | +0.005% | -0.017%, +0.045%, -0.006% |

The isolated decode rates are higher than the earlier full-matrix baseline because decode now starts from a cool, isolated process instead of running after prefill. They are only interpreted as paired on/off comparisons here.

## CORRECTNESS

`tests/run_correctness.sh` passed on the off candidate:

- 4/4 upstream quantization/layout tests passed.
- 96/96 CUDA-vs-CPU PTQ1_0/PQ2_0 `MUL_MAT` cases passed, with the upstream NMSE tolerance of `5e-4`.
- PTQ1_0 and PQ2_0 each loaded on CUDA and produced a non-empty greedy 32-token completion (context 512, seed 42, temperature 0).

The completion record is [correctness_smoke.json](../../results/exp002/correctness_smoke.json). Source was restored to the original prefetch-on baseline after the checks.

## MICROBENCHMARK

No standalone kernel microbenchmark was run: the available harness measures model workloads, and Nsight Compute instruction/cycle counters are unavailable on this host (`ERR_NVGPUCTRPERM`, as recorded in earlier profiling). The paired `llama-bench` process is the end-to-end measurement. The 512-token decode follow-up first produced three samples per process in five pairs at context 4096, then two more pairs at seven samples per process to inspect the sustained tail. The primary 128-token matrix has nine samples per variant/workload in [paired.json](../../results/exp002/paired.json). Follow-up samples and telemetry are in [long_decode_512.json](../../results/exp002/long_decode_512.json), [long_decode_4096_pairs4_5.json](../../results/exp002/long_decode_4096_pairs4_5.json), and [long_decode_4096_pairs6_7_r7.json](../../results/exp002/long_decode_4096_pairs6_7_r7.json). The primary process telemetry reached 100% GPU utilization, 62–79 C, about 31–347 W, and 1110–1980 MHz SM clocks. The seven-repetition tail runs also reached 79 C; their per-process minimum SM clocks ranged from 270–1020 MHz.

The exact 128-token run command was:

```bash
python3 benchmark/prefetch_ab.py --pairs 3 --repetitions 3 \
  --cooldown-temp-c 62 --output results/exp002/paired.json
```

The sustained decode follow-up commands were:

```bash
python3 benchmark/prefetch_ab.py --modes decode --contexts 512 4096 \
  --pairs 3 --repetitions 3 --decode-tokens 512 --cooldown-temp-c 64 \
  --output results/exp002/long_decode_512.json
python3 benchmark/prefetch_ab.py --modes decode --contexts 4096 \
  --pairs 2 --pair-start 4 --repetitions 3 --decode-tokens 512 --cooldown-temp-c 64 \
  --output results/exp002/long_decode_4096_pairs4_5.json
python3 benchmark/prefetch_ab.py --modes decode --contexts 4096 \
  --pairs 2 --pair-start 6 --repetitions 7 --decode-tokens 512 --cooldown-temp-c 64 \
  --output results/exp002/long_decode_4096_pairs6_7_r7.json
```

Workload commands used the same model and flags for both variants:

```text
llama-bench -m models/Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa on \
  -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 -r 3 -o json \
  -p 0 -n 128 -d CONTEXT
```

Combined runs replaced the final workload arguments with `-p 0 -n 0 -pg CONTEXT,128`. Each command ran separately for contexts 512 and 4096. llama-bench's default warmup was enabled.

## END-TO-END IMPACT

At 128 generated tokens, the off candidate changed the median by -0.03% at decode context 512, -0.03% at decode context 4096, -0.04% at combined context 512, and +0.01% at combined context 4096. At 512 generated tokens, context 512 remained tied (off/on +0.01%). Context 4096 had a -0.38%, -0.43%, and -2.30% off/on change in three-repetition pairs 1–3, followed by +1.93% and +0.52% in reversed-order pairs 4–5. Across those 15 raw samples, the median was 73.943 tok/s on and 73.838 tok/s off (-0.14%). In the two seven-repetition pairs, samples 3–7 averaged 61.970 off vs 57.451 on in pair 6 (off then on), but 58.470 on vs 55.489 off in pair 7 (on then off). The first variant in each pair won the tail, and the result reversed with order. Pair 6 started at 62 C for off and 64 C for on; pair 7 started at 64 C for both variants. Both runs peaked at 79 C; observed SM clocks ranged from 270–1980 MHz. No repeatable end-to-end benefit was demonstrated.

## ANALYSIS

The paired protocol removed the large workload-order and thermal-history mismatch from Experiment 001. At 128 tokens, the variants are indistinguishable within tightly grouped samples. In five three-repetition context-4096 runs, sample 3 favored prefetch-on. However, seven-repetition runs showed that the sample 3–7 tail advantage changed sign when order reversed: the first variant in each pair ran faster on the tail. The long-run clocks and temperatures varied substantially, with both variants reaching 79 C. This indicates the apparent tail effect is confounded by process order and thermal/clock history.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**INCONCLUSIVE; REVERT the candidate.** Keep the original prefetch-on baseline source. No candidate patch is proposed for retention; the exact off diff is preserved for review. The 128-token test found no measurable effect. The 512-token context-4096 third-sample pattern did not persist across the seven-repetition, reversed-order tail check, so there is no robust candidate gain to retain.

## FOLLOW-UPS

- Do not sweep prefetch distance or policy unless a standalone kernel measurement can expose a meaningful instruction-level difference.
- Continue with the separate sm_86 GEMV geometry/unpack investigation in the research queue.

## IMPORTANT DISCOVERIES

- A controlled isolated decode run can reach about 76–78 tok/s at contexts 512–4096; this must not be compared directly with the earlier full-matrix decode rows, which ran after prefill and GPU heating.
- At 128 generated tokens, turning prefetch off changed no median by more than 0.04% and pair effects stayed within +/-0.11%.
- At 512 tokens and context 4096, five three-repetition pairs showed a faster third sample with prefetch-on. In two seven-repetition pairs, samples 3–7 instead favored whichever variant ran first; the direction flipped under order reversal.
- Requiring a strict 60 C gate became impractical after the first candidate workload; a 62 C gate consistently matched both variants and all final pairs.

## MANAGER DISPATCH AUDIT (2026-10-06)

The on/off patch changes only the PTQ1_0 prefetch in the generic `mul_mat_vec_q` loop in `mmvq.cu`. At the compared source revision `aa9edbb`, plain one-column PTQ1_0 on RTX 3080 already dispatches to `mul_mat_vec_ptq1_0_pt` before reaching that generic loop. Thus all batch-1 decode A/B arms ran the same active kernel, and their ties or order-dependent differences are a no-op comparison for the target decode path. Keep the captured runs as protocol/noise diagnostics, not evidence about prefetch. The generic-path prefetch effect for other eligible shapes was not tested here.
