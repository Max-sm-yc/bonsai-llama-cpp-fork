# Experiment index

| ID | Hypothesis | Result | E2E delta | Decision | Report |
|---|---|---|---:|---|---|
| BASE | Establish matched PTQ1_0/PQ2_0 baseline on RTX 3080 | PTQ1_0 decode leads 32–54%; prefill is nearly tied | baseline | VERIFIED | [BASELINE.md](../BASELINE.md) |
| 001 | Disable PTQ1_0 sm_86 batch-1 GEMV L2 prefetch | No robust matched-control gain; elevated quick-screen result traced to mode order/start thermal conditions | inconclusive | REVERT | [Report](../experiments/001-ptq1-sm86-gemv/REPORT.md) |
