# Experiment 045: PTQ1_0 register occupancy bounds

## HYPOTHESIS

The active 128-thread, ROWS=1 planar PTQ1_0 GEMV uses 76 registers/thread for the plain kernel and 98 for the fused gate kernel. Applying a larger compile-time minimum-resident-CTA constraint to only `ncols==1` may lower register allocation enough to increase resident CTAs and hide latency while leaving the CTA geometry and math unchanged.

## IMPLEMENTATION AND ISOLATION

The manager checkout was at `b6464b9cf5eafe7f4931b12ddb5708ea7c498489`; production code was `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`. All candidate source edits and CUDA compilation used detached worktree `/tmp/exp045-ptq1-register-occupancy`. The source change adds `PTQ1_0_PT_MINB_1` and substitutes it only in the `ncols == 1` arm of `__launch_bounds__`; other column counts retain their existing launch bounds. CTA width (128), ROWS=1 mapping, ternary decode, accumulation, model settings, and runtime library objects were unchanged.

The tested minimum-CTA values were 4, 6, 7, and 8. `mmvq.cu` and `quantize.cu` were compiled from the isolated worktree with the selected macro, then linked with the unchanged object set into isolated runtime directories. The exact patch is [`launch_bounds.patch`](../../results/exp045/launch_bounds.patch). The active PTX excerpt shows `.maxntid 128, 1, 1` and `.minnctapersm 8` for the eight-CTA build.

## STATIC RESOURCES

Ptxas registers/thread for the active ROWS=1 entries were:

| Minimum CTAs/SM | Plain `<1,1,false,false>` | Fused no-gate `<1,1,true,false>` | Fused gate `<1,1,true,true>` | Active stack/spills |
|---:|---:|---:|---:|---:|
| 4 (control) | 76 | 74 | 98 | 0 / 0 |
| 6 | 72 | 72 | 77 | 0 / 0 |
| 7 | 68 | 68 | 72 | 0 / 0 |
| 8 | 60 | 60 | 63 | 0 / 0 |

At 128 threads, these changes lower each CTA's nominal register demand from 9,728/12,544 registers (plain/gated control) to 7,680/8,064 at the eight-CTA bound. The compiler honored each minimum CTA request in emitted PTX. Exact `ptxas -v` logs are gzip-compressed under `results/exp045/`; all variants retained the same existing spill sites in unrelated multi-column specializations, while the three active ROWS=1 entries had no stack or spill bytes at any tested bound. The eight-CTA active resource record reports `REG:60` plain and `REG:63` gated, `STACK:0`, `LOCAL:0`. SASS retains the expected 128-bit activation loads and `IDP.4A` instructions; no active local spill loads/stores were emitted. See [`active_ptxas_summary.txt`](../../results/exp045/min8/active_ptxas_summary.txt), [`active_ptx_excerpt.txt`](../../results/exp045/min8/active_ptx_excerpt.txt), and [`sass_excerpt.txt`](../../results/exp045/min8/sass_excerpt.txt).

## SHORT MODEL SCREEN

All arms used the RTX 3080, PTQ1_0, contexts 512/4096, 128 decode tokens, three repetitions per run, and a fresh <=60 C / <=5% utilization gate. Pair 1 ran candidate then control; pair 2 reversed that order. The values below are each run's sample median (full min–max range), in tok/s. Deltas use the median of the two run medians.

| Bound | Context | Control pair medians (ranges) | Candidate pair medians (ranges) | Paired-median delta |
|---:|---:|---|---|---:|
| 6 | 512 | 83.452 (82.469–83.549), 83.266 (82.345–83.292) | 81.488 (80.626–81.546), 80.952 (80.151–81.018) | -2.57% |
| 6 | 4096 | 80.959 (80.187–80.960), 80.804 (80.080–80.841) | 79.110 (78.402–79.144), 78.690 (78.018–78.730) | -2.45% |
| 7 | 512 | 83.258 (82.343–83.299), 83.329 (82.355–83.335) | 79.854 (78.883–79.879), 79.808 (78.966–79.864) | -4.16% |
| 7 | 4096 | 80.900 (80.147–80.908), 80.908 (80.132–80.924) | 77.587 (76.938–77.620), 77.550 (76.921–77.682) | -4.12% |
| 8 | 512 | 83.617 (82.676–83.626), 83.467 (82.386–83.472) | 79.564 (78.666–79.699), 79.052 (78.389–79.093) | -5.07% |
| 8 | 4096 | 81.030 (80.305–81.061), 80.970 (80.261–81.044) | 77.140 (76.487–77.151), 76.662 (75.968–76.692) | -5.06% |

The 6, 7, and 8 CTA constraints all lost in both order-reversed short screens at both contexts. More aggressive register caps made the regression larger, despite raising the requested occupancy. Since every candidate failed this focused model screen, no fixed-seed model smoke, correctness suite, full seven-repetition E2E comparison, or promotion check was run. There is no correctness claim for these runtime variants.

For each bound, four gated benchmark JSON files (two candidate and two control) retain all samples and gate telemetry; compact raw `llama-bench` output is under `results/exp045/raw/`. An attempted min7 reverse-control invocation without `--modes decode` began the default multi-mode run and was interrupted during its combined mode; its partial output is excluded from all measurements. The subsequent correctly scoped reverse arms passed the stated gates.

## DECISION

**REVERT.** Do not change production launch bounds. Lower registers and a higher requested resident-CTA count did not improve decode; all three tested candidates clearly regressed. Keep the current 4-CTA bound.

## COMMANDS, HASHES, AND CHECKS

The source macro was compiled by replaying the isolated worktree's existing `mmvq.cu` and `quantize.cu` commands from `build/compile_commands.json`, adding `-Xptxas=-v -DPTQ1_0_PT_MINB_1={4,6,7,8}`. The two changed objects were linked with the unchanged CUDA object list into each isolated arm library. Short screens used:

```bash
LD_LIBRARY_PATH="$PWD/results/exp045/minN:$PWD/build/bin" python3 benchmark/run.py \
  --binary results/exp045/minN/llama-bench \
  --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --modes decode --contexts 512 4096 --decode-tokens 128 --repetitions 3 \
  --cooldown-temp-c 60 --output results/exp045/minN_screen_<arm>.json
```

`ldd` confirmed each candidate executable loaded its matching isolated CUDA library and the unchanged `libggml-base.so.0` from `build/bin`. Candidate source header SHA-256: `d5f4c7c030c33e0999724be225a079fdaa19fd56aebcbfdd123b9ebe55d3a917`. Candidate CUDA library hashes were min6 `b5053ce6e0316bd292f4a25c4dfa263e21c98ce074cc122a6712520ffca319c0`, min7 `d9f8246b87b63edba5a8727f52eec0a86df2d9639085834410e6a61bf7795279`, and min8 `1e6c64f8d972b4df59617aeb9bd2ba4a147108674e6259d5da06ba84f6bf912e`. The production source hash remains `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`; the production CUDA library remains `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`. The manager checkout remains at `b6464b9cf5eafe7f4931b12ddb5708ea7c498489`.

No commit was made. Temporary runtime libraries, executables, and objects were removed after screening; only compact source, resource, SASS/PTX excerpts and benchmark records remain in `results/exp045/`.
