# Correctness checks

The PrismML fork already provides quantization and ternary-layout tests. The project correctness run builds and executes those upstream tests, then checks actual CUDA model loading and generation for each shipped packing.

Run the complete correctness pass with:

```bash
tests/run_correctness.sh
```

The upstream quantization test checks reference encodings with the 0.01 total-error bound for ternary types; the PTQ1_0 element-map test covers all 128 packed positions; and the PTQ1_0 dot transcription test checks exact integer accumulators plus a float tolerance of `1e-6 * max(1, |reference|)`. The PQ2 row-shape test checks valid lengths. `test-backend-ops` then exercises CUDA `MUL_MAT` on random inputs and compares it to the CPU backend for both formats, using the upstream `5e-4` normalized mean-square error limit. Its selected dimensions include odd output-row tails, batch-1 decode and small multi-column batches, and model K sizes from 1024 through 17408.

Then run actual model smoke inference on the RTX 3080:

```bash
python3 tests/model_smoke.py --output results/baseline_smoke.json
```

This loads both files and generates 32 greedy tokens from a fixed prompt in a single turn, using the model's normal chat template with EOS ignored. Keep per-format baseline output and compare it after changes. Add kernel-specific GPU numerical tests for any newly implemented arithmetic path before accepting it.

When the CUDA backend is enabled, `tests/run_correctness.sh` also builds and runs `test-fwht-rms-q8`. This directly launches the coordinated RMSNorm/FWHT/Q8_1 CUDA entry point and compares its packed PT values, half scales, dequantized outputs, and stored block sums with a deterministic host reference at the supported 5120-element row width. It covers one-row decode and a three-row case, and checks the shape-support fallbacks.

The CUDA correctness suite also compares the Exp083 strided-view `CONT -> SIGMOID -> MUL` fusion against a deterministic host scalar reference for sequence lengths 1, 2, 128, 512, and 4096, plus a contiguous-view matcher fallback. The maximum absolute output tolerance is `2e-6`.
