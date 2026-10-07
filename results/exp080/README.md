# Exp080 artifacts

`experiments/080-ptq1-gemv-dataflow/REPORT.md` contains the no-candidate result and exact build/configuration evidence. `raw/` retains the dispatch/dataflow audit, source/build hashes, active sm_86 SASS, resource records, commands, loader paths, and GPU snapshot. The 212 MiB isolated build directory is intentionally not copied; the exact build command and output hashes are recorded in `raw/`.

No kernel candidate, correctness delta, microbenchmark, or end-to-end result was produced. The production implementation remains at commit `ffb0ef37690b902829ea1158b02b14517ed93c2b`. `SHA256SUMS` covers the report, this README, manager verification, and every other file in `results/exp080/` except the manifest itself.
