# Experiment index

| ID | Hypothesis | Result | E2E delta | Decision | Report |
|---|---|---|---:|---|---|
| BASE | Establish matched PTQ1_0/PQ2_0 baseline on RTX 3080 | PTQ1_0 decode leads 32–54%; prefill is nearly tied | baseline | VERIFIED | [BASELINE.md](../BASELINE.md) |
| 001 | Disable PTQ1_0 sm_86 batch-1 GEMV L2 prefetch | Unpaired comparison; apparent trend inconclusive | inconclusive | REVERT | [Report](../experiments/001-ptq1-sm86-gemv/REPORT.md) |
| 002 | Paired PTQ1_0 sm_86 batch-1 GEMV prefetch on/off | 128-token workloads tied; 512-token long-context tail failed reversed-order r7 control | <=0.04% at 128 tokens; sustained effect inconclusive | INCONCLUSIVE / REVERT | [Report](../experiments/002-ptq1-prefetch-ab/REPORT.md) |
| 003 | PTQ1_0 batch-1 GEMV warp geometry (2/4/8 warps) | No repeatable decode change at contexts 512 or 4096 | -0.025% / +0.026% for 8 warps; 2 warps tied | REVERT | [Report](../experiments/003-ptq1-gemv-geometry/REPORT.md) |
| 004 | Replace PTQ1_0 base-3 `qs` expansion with constant-memory LUT | Exact in 65,536 dot outputs; LUT was 5.89x slower in focused CUDA timing | Not run | REVERT | [Report](../experiments/004-ptq1-trit-decoder/REPORT.md) |
