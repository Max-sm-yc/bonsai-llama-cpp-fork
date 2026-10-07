# Exp079 artifacts

`experiments/079-ptq1-l2-persistence/REPORT.md` contains the complete experiment report and both policy decisions. `manager-verification.md` records the manager's independent checks. `raw/` holds model weight extracts, the fixed planar Q8_1 activation, exact production-kernel replay source, CUDA-event distributions, telemetry, paired decode arm captures, preserved source snapshots, and source/build hashes.

The maximum-reservation policy is rejected. The separate exact-size reservation is also rejected; it improved the forced-cold result for one matrix but did not improve paired end-to-end decode. Neither policy remains in the production CUDA sources. Three local `.so` snapshots are ignored build outputs; their hashes are recorded in `raw/source_build_hashes.txt`. The portable `SHA256SUMS` covers this README, the report, the manager verification, and every other file under `results/exp079/` except itself and the ignored `.so` snapshots.
