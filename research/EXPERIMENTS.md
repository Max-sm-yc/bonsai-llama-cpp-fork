# Experiment index

| ID | Hypothesis | Result | E2E delta | Decision | Report |
|---|---|---|---:|---|---|
| BASE | Establish matched PTQ1_0/PQ2_0 baseline on RTX 3080 | PTQ1_0 decode leads 32–54%; prefill is nearly tied | baseline | VERIFIED | [BASELINE.md](../BASELINE.md) |
| 001 | Disable PTQ1_0 sm_86 batch-1 GEMV L2 prefetch | Unpaired comparison; apparent trend inconclusive | inconclusive | REVERT | [Report](../experiments/001-ptq1-sm86-gemv/REPORT.md) |
| 002 | Paired PTQ1_0 sm_86 batch-1 GEMV prefetch on/off | 128-token workloads tied; 512-token long-context tail failed reversed-order r7 control | <=0.04% at 128 tokens; sustained effect inconclusive | INCONCLUSIVE / REVERT | [Report](../experiments/002-ptq1-prefetch-ab/REPORT.md) |
| 003 | PTQ1_0 batch-1 GEMV warp geometry (2/4/8 warps) | No repeatable decode change at contexts 512 or 4096 | -0.025% / +0.026% for 8 warps; 2 warps tied | REVERT | [Report](../experiments/003-ptq1-gemv-geometry/REPORT.md) |
| 004 | Replace PTQ1_0 base-3 `qs` expansion with constant-memory LUT | Exact in 65,536 dot outputs; LUT was 5.89x slower in focused CUDA timing | Not run | REVERT | [Report](../experiments/004-ptq1-trit-decoder/REPORT.md) |
| 005 | Exact 2-bit PTQ1_0 side representation to simplify batch-1 decode | Prototype wrote beyond its 32-byte packed-code field and used the wrong element map; 65,531/65,536 dot outputs mismatched, so timing is invalid. 28B to 34B per block (+21.43%) | Not run | INCONCLUSIVE | [Report](../experiments/005-ptq1-2bit-side/REPORT.md) |
| 006 | Exact 2-bit PTQ1_0 side representation screen | Code/dot exact for synthetic activation, but SOA address was not production mapping; reported 18.4–18.8% slowdown is invalid for decisions. 28B to 34B (+21.43%) | Not run | INCONCLUSIVE | [Report](../experiments/006-ptq1-2bit-corrected/REPORT.md) |
| 007 | Correct SOA and DP4A-matched PTQ1_0 2-bit side dot | Exact and sanitizer-clean; side path slower in every RTX 3080 screen: 3.44–3.50% at 65,536 blocks, 5.02% at 16,384. Payload +21.43% | Not run | REJECT | [Report](../experiments/007-ptq1-side-production-dot/REPORT.md) |
| 008 | Parallel floor-difference decoder for original PTQ1 bytes | Host identity exact for all bytes; CUDA block failed because `qh`'s two streams were not interleaved. No timing. | Not run | INCONCLUSIVE | [Report](../experiments/008-ptq1-parallel-trit-decode/REPORT.md) |
| 009 | Correct `qh` and device-verify PTQ1 parallel floor decoder | Exact device and full-block checks; slower than production by 7.04%, 4.11%, and 1.24% at 1,024/16,384/65,536 blocks; no E2E run | Not run | REJECT | [Report](../experiments/009-ptq1-floor-decoder-qh/REPORT.md) |
