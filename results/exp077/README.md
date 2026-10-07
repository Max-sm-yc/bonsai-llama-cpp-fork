# Exp077 artifacts

- `raw/prompt-seeds-natural.json`: exact fixed Exp071 token IDs used by all cells.
- `raw/server_bench.py`: copied Exp071 harness, updated to read fixed IDs, record generated IDs, and use the isolated server build.
- `raw/natural_r{770,771}_{bundle_target,pq2_0_mtp}_ctx{512,4096}.{json,server.log,gpu.csv}`: all eight server runs. Round 770 has the environment variable unset; round 771 sets it to `1`. JSON rows contain 128 generated token IDs, text, server timings, and acceptance counts. Each GPU CSV samples timestamp, temperature, utilization, and memory every 200 ms.
- `raw/summary.json` and `raw/summary_env_off.json`: exact-ID and performance summaries derived from the raw files. `manager-verification.md` records independent recomputation from the raw samples.
- `raw/commands.txt`, `raw/build-info.txt`: invocation and build records.
- `raw/exp071_bundle_target_ctx{512,4096}.*` and `raw/prior_natural_r0_bundle_target_ctx512.json`: prior target-only outputs/logs/telemetry retained for comparison. Historical Exp071 generated text matches 3/6 family/context cells; no historical output IDs were saved.
- `build/` is omitted from the tracked artifacts because it occupies about 976 MiB. The experiment's exact server and CUDA library were retained in the isolated worktree at `/home/maxsun/autonomous_projects/.worktrees/exp077-mtp-batch-invariant/results/exp077/build/`; their SHA256 hashes are in `raw/build-info.txt`. Rebuild from the project root with the recorded CMake cache settings and `cmake --build results/exp077/build --target llama-server -j 8`.
