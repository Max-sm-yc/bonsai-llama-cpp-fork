# Experiment 001: PTQ1_0 sm_86 GEMV L2 prefetch

## HYPOTHESIS

The PTQ1_0 GEMV's explicit next-iteration `prefetch.global.L2` may add instruction/scoreboard cost without hiding enough latency on RTX 3080. Disabling it might improve batch-1 decode, especially at long contexts where kernels repeatedly stream the packed weights.

## IMPLEMENTATION

Temporarily disabled the existing PTQ1_0-only prefetch block in `ggml/src/ggml-cuda/mmvq.cu` (the `prefetch.global.L2` calls for the next K-block). No unpack math, quantization, launch geometry, dispatch, PQ2_0 behavior, or model data changed. Built with the existing Release/CUDA sm_86 build tree. The candidate source was restored to the experiment starting revision after evaluation; no source change is retained.

## RESULT

The three-repetition screen initially appeared much faster because decode was the first mode run, at a 54 C idle start, while the reference matrix runs prefill before decode and heats the GPU. That screen is not a valid performance comparison. The required full matrix, which preserves the reference mode order and <=60 C format start gate, showed modest higher PTQ1_0 medians but broad, highly variable sample distributions. The strongest median delta was at combined context 4096, yet candidate samples there ranged from 384.5 to 692.1 tok/s. There was no same-build prefetch-on paired control. The evidence does not support retaining the change.

## CORRECTNESS

`tests/run_correctness.sh` passed on the no-prefetch candidate build:

- 4/4 upstream quantization/layout tests passed.
- 96/96 CUDA-vs-CPU PTQ1_0/PQ2_0 `MUL_MAT` comparisons passed.
- Both PTQ1_0 and PQ2_0 model smoke runs produced non-empty 32-token completions.

The script also built `test-backend-ops`. Source was restored to the experiment baseline after correctness completed.

## MICROBENCHMARK

Quick screen command:

```bash
python3 benchmark/run.py --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --modes decode combined --contexts 512 4096 --repetitions 3 \
  --cooldown-temp-c 60 --output results/exp001_no_prefetch_screen.json
```

The screen ran decode first at 54 C, so its medians (78.1 tok/s at context 512 and 75.9 tok/s at 4096) are not comparable to the full reference matrix. Its raw samples and commands are in the JSON above and adjacent `results/raw/` logs.

## END-TO-END IMPACT

Exact full matrix candidate command:

```bash
python3 benchmark/run.py --cooldown-temp-c 60 --output results/exp001_no_prefetch_full.json
```

Seven samples per row, medians and sample ranges in tok/s:

| PTQ1_0 mode | Context | Reference median (range) | No-prefetch median (range) | Median delta |
|---|---:|---:|---:|---:|
| Decode | 128 | 46.88 (39.8–76.9) | 49.40 (49.3–77.0) | +5.4% |
| Decode | 512 | 46.09 (40.5–76.2) | 46.91 (42.2–76.5) | +1.8% |
| Decode | 2048 | 42.82 (42.8–74.7) | 45.40 (43.1–74.8) | +6.0% |
| Decode | 4096 | 39.21 (33.6–72.8) | 43.04 (40.0–73.3) | +9.8% |
| Combined | 128 | 85.54 (59.8–143.8) | 86.53 (71.9–144.0) | +1.2% |
| Combined | 512 | 236.08 (139.1–291.8) | 247.32 (162.6–306.2) | +4.8% |
| Combined | 2048 | 436.15 (280.1–558.6) | 442.49 (321.1–562.3) | +1.5% |
| Combined | 4096 | 426.16 (378.2–606.6) | 529.64 (384.5–692.1) | +24.3% |

Prefill PTQ1_0 medians were 1299.78/1378.29/1357.55/1334.12 tok/s for contexts 128/512/2048/4096 versus reference 1292.26/1378.14/1355.30/1331.25. PQ2_0 was included in the full matrix to check for unintended shared-path behavior; its medians varied in both directions and it was not changed intentionally. Raw per-repetition data, commands, and telemetry are in `results/exp001_no_prefetch_full.json` and `results/raw/`.

This is not a controlled paired comparison: the reference is the separately captured `results/baseline.json`, while the candidate was rebuilt as build number 2 (reference build number 1). GPU utilization and thermal state also varied during seven-sample runs. The overlapping ranges and high within-row spread prevent a robust attribution to the prefetch change.

## ANALYSIS

The explicit prefetch is a plausible source of instruction overhead, but the experiment did not isolate it well enough to establish a benefit. The highly elevated first samples in the reference and candidate decode rows show that clocks/thermal history materially affected these measurements. The full candidate run used the prescribed prefill/decode/combined order and a 60 C gate before each model format, so it is more informative than the quick screen; however, mode-by-mode paired runs and a same-build prefetch-on control remain necessary to identify small effects. No quantization or numerical issue was observed.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** Reverted the only source edit. No candidate has demonstrated a robust improvement against a controlled same-build comparison. The tree's tracked source is back at the experiment starting revision. The experiment remains inconclusive about the performance effect of this prefetch.

## FOLLOW-UPS

- If revisited, collect a same-build A/B with prefetch on and off, alternate run order, and pair decode plus combined at contexts 512 and 4096 under identical idle-temperature gates.
- Capture clocks and temperature per repetition, not only per process, to explain the high and unstable samples.
- Keep candidate full-matrix and screen JSON as raw evidence; do not use the screen medians for performance claims.

## IMPORTANT DISCOVERIES

- A decode-first quick screen at a cool start can report roughly 78 tok/s at context 512, while a decode run after prefill falls to the mid-40s. Mode order and thermal history materially affect this model's apparent throughput.
- The no-prefetch candidate passed all requested correctness checks, but passing correctness does not resolve its uncertain performance effect.
