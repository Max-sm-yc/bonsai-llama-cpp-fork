# Exp079 manager verification

- **Decision:** reject both L2 persisting-policy variants; no current-best change.
- The experimenter restored both modified CUDA files. The manager checkout matches the production baseline hashes: `mmvq.cu` `e889b154…`, `mmvq-ptq1_0.cuh` `f398417a…`.
- Recomputed medians from the paired JSON: maximum reservation, ctx512 84.170→83.418 tok/s (−0.894%), ctx4096 81.747→81.051 (−0.852%); exact-size reservation, ctx512 84.265→84.130 (−0.159%), ctx4096 81.762→81.762 (−0.0003%). Each used two reversed pairs, seven repetitions per arm, and 128 generated tokens.
- All candidate start-gate records were at or below 60 C and 5% GPU utilization. Telemetry peaks were 6,579 MiB at ctx512 and 6,803 MiB at ctx4096.
- Both fixed 32-token smoke comparisons report exact normalized response equality (119 characters). The direct production-kernel comparison against CPU dequantization had 0/1,024 output mismatches.
- Event CSVs contain 25 samples per warm/cold condition for each matrix-set/policy combination. The maximum-reservation one-matrix forced-cold median improved 7.168→5.440 µs, while warm median moved 5.120→5.472 µs; the exact-size condition was similarly cold-only. Four- and 32-matrix sets did not show a meaningful gain.
- The pre-refresh experiment manifest contained 100 checksummed artifacts and was independently verified. The portable manifest now excludes three ignored local `.so` build outputs; their hashes remain recorded in `raw/source_build_hashes.txt`. Build directory outputs are not part of the portable manifest.
- No candidate source is retained in the production checkout. The benchmark/report artifacts preserve the rejected policy captures and exact measurements.
