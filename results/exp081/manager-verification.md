# Exp081 manager verification

- The experiment worktree is based on manager commit `7464277`. Its PTQ1_0 quantization, dispatch, and GEMV source hashes match the recorded production baseline; the experimenter made no production source changes. The active source also remains byte-identical to production best commit `ffb0ef3`.
- Verified the experimenter's artifact manifest and reran the standalone CUDA Graph screen on the RTX 3080 with 300 replays/sample. All six active-layout shapes again reported zero output mismatches, and all direct-code timing ranges remained disjoint and slower than base-3:

| K blocks | Rows | Manager base-3 median (us) | Manager direct median (us) | Direct delta |
|---:|---:|---:|---:|---:|
| 40 | 257 | 5.569493 | 6.167893 | +10.744% |
| 40 | 1,025 | 6.543360 | 7.116800 | +8.764% |
| 40 | 4,099 | 8.649386 | 10.601813 | +22.573% |
| 136 | 257 | 20.179413 | 26.248533 | +30.076% |
| 136 | 1,025 | 22.014933 | 27.863043 | +26.564% |
| 136 | 4,099 | 37.576962 | 73.460693 | +95.494% |

- The retained Compute Sanitizer memcheck transcript covers all six K/row shapes and reports zero errors. The focused screen compares device output to both the base-3 path and the independent CPU reference before timing.
- No runtime library, model A/B, or peak-VRAM test was performed. This is a decisive microbenchmark rejection, not an end-to-end result. The direct representation adds 1,199,923,200 bytes (1,144.34 MiB) to the active 5,599,641,600-byte PTQ payload, with no measured runtime benefit.
- **Decision: REVERT / NO CANDIDATE.** Do not integrate the 34-byte side representation. Current best commit, results, and production sources remain unchanged.
- `raw/manager_rerun.txt` preserves the independent timing and correctness rerun.
