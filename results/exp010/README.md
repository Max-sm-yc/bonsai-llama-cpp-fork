# Experiment 010 artifacts

The first `screen/` outputs and the first `followup/` outputs are invalid for
performance comparison. Those archived `llama-bench` executables had an
absolute RUNPATH into `build/bin`, so they all loaded the same CUDA library.
Their benchmark numbers are preserved as no-op controls only; do not use them
to rank row schedules.

Valid screens and follow-ups have `_isolated` in their JSON filenames and were
run with `LD_LIBRARY_PATH` set to the matching per-variant build directory.
`ldd` plus `LD_DEBUG=libs` path checks are saved under `raw/`. The direct
ROWS=1-vs-ROWS=2 comparison and the final source-default ROWS=1 check are in
`rows1_vs_rows2/` and `final_build_check/`. The final report is
`../../experiments/010-ptq1-planar-rows/REPORT.md`.

Candidate binaries and libraries under `builds/` are local-only build
artifacts and are excluded from Git. Compact logs, benchmark JSON, loader
checks, correctness outputs, and the report are retained for reproducibility.
