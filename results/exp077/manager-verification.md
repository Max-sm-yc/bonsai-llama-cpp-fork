# Exp077 manager verification

Independent checks run after the experimenter completed, from the eight raw JSON and GPU telemetry files.

- Each arm/context JSON contains 21 rows (3 prompt families x 7 repetitions); every row generated 128 tokens and retained 128 token IDs. All server start gates were at or below 60 C and 5% GPU utilization.
- Batch-invariant MTP vs target-only: 42/42 exact token-ID sequences. Default-mode MTP vs target-only: 21/42 exact; first divergences occur in the Qwen ctx512, reports ctx4096, and Qwen ctx4096 cells.
- Target-only batch-invariant vs default: 21/42 exact streams. The 21 changed streams are ctx512 speculative C++ (first token index 104) and ctx4096 reports/Qwen (indices 17/94), seven repetitions each.
- Recomputed pooled decode rate as total generated tokens / total server-reported generation milliseconds: ctx512 default target/MTP 69.84/88.29, invariant target/MTP 63.78/85.57 tok/s; ctx4096 default 49.67/63.51, invariant 45.78/58.61 tok/s. These are PQ2_0 bundle results, not a paired PTQ1_0 comparison.
- GPU CSV maxima were 7,593 MiB for target-only and 8,485 MiB for MTP, both below 10,240 MiB.
- SHA256 values for the isolated server and CUDA library match `raw/build-info.txt`. No production source was changed; the current best commit is unchanged.

Decision: keep the global mode and MTP bundle out of production. Exact parity is established only against the altered invariant target; target-only outputs change in 21/42 streams and were not quality-validated.
