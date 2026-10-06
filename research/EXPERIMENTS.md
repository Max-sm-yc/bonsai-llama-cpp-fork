# Experiment index

| ID | Hypothesis | Result | E2E delta | Decision | Report |
|---|---|---|---:|---|---|
| BASE | Establish matched PTQ1_0/PQ2_0 baseline on RTX 3080 | PTQ1_0 decode leads 32–54%; prefill is nearly tied | baseline | VERIFIED | [BASELINE.md](../BASELINE.md) |
| 001 | Disable PTQ1_0 sm_86 batch-1 GEMV L2 prefetch | Unpaired comparison; apparent trend inconclusive | inconclusive | REVERT | [Report](../experiments/001-ptq1-sm86-gemv/REPORT.md) |
| 002 | Paired PTQ1_0 sm_86 batch-1 GEMV prefetch on/off | 128-token workloads tied; 512-token long-context tail failed reversed-order r7 control | <=0.04% at 128 tokens; sustained effect inconclusive | INCONCLUSIVE / REVERT | [Report](../experiments/002-ptq1-prefetch-ab/REPORT.md) |
