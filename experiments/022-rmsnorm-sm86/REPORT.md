# Experiment 022: RMSNorm geometry on sm_86

## HYPOTHESIS

The mixed PTQ1_0 Nsight Systems trace attributed visible time to RMSNorm work. The dominant weighted RMSNorm specialization uses a 1024-thread CTA; using 256 threads could improve occupancy and reduce CTA resource cost for batch-1 decode.

## IMPLEMENTATION

Inspected `norm.cu`, `norm.cuh`, the graph fusion dispatch, and the post-ROWS=1 Nsight Systems SQLite trace. The active weighted signatures were `rms_norm_f32<1024,true,false>` and `<256,true,false>`. The template number is the CTA block size, not the input width. Source dispatch sends fused-weight calls with `ncols >= 1024` to the 1024-thread path, and smaller widths to the 256-thread path. Nsight records confirm block dimensions of 1024 and 256 respectively. The trace did not capture kernel argument values, so exact `ncols` per graph node is not recoverable from these artifacts; the source dispatch gives the width class.

Changed only the `mul != nullptr && add == nullptr && ncols >= 1024` fused-weight branch to launch `rms_norm_f32<256,true>` with 256 threads. The ordinary RMSNorm path, the existing RMSNorm+weight+RoPE fusion, other fused variants, arithmetic, and sub-1024 branch were unchanged. Patch: `results/exp022/norm-candidate.patch`. Candidate source and libraries are preserved under `results/exp022/`.

The original build Ninja log was truncated, so `cmake --build` recovered by rebuilding broadly (191 targets for the candidate; the correctness script subsequently began a 393-target rebuild). Candidate CUDA compilation and linking succeeded. The candidate library was retained before restoring the active CUDA library. `loader_audit.txt` records the canonical executable RUNPATH, resolved library location, and hashes.

## RESULT

The candidate regressed by about 3.3% at both contexts. **REVERT.** A second reversed-order pair was not run because this is a clear regression at both decode lengths.

## CORRECTNESS

A fixed-seed, 32-token PTQ1_0 and PQ2_0 model smoke was run with the candidate CUDA library preloaded. After normalizing the build banner and prompt/generation timing line, both candidate completions exactly match the saved reference completions in `results/exp022/baseline_smoke-before.json`. Candidate smoke output is `results/exp022/candidate_smoke.json`; raw model stdout/stderr are in `results/raw/` and referenced there.

`bash tests/run_correctness.sh` was started on the candidate but interrupted during its automatically triggered broad rebuild after the full-model runs showed a clear regression. The selected CTests and CUDA-vs-CPU matmul suite therefore were not completed. Candidate compilation succeeded. This variant is rejected and not retained as production code.

## MICROBENCHMARK

No standalone kernel microbenchmark was run. The decision used controlled end-to-end decode, which directly exercises the candidate path. The profile audit quantified the actual RMSNorm family before selecting the branch.

## END-TO-END IMPACT

Both arms used the canonical runner and identical requested settings: PTQ1_0, contexts 512 and 4096, 128 generated tokens, seven repetitions, default warmups, 99 GPU layers, Flash Attention on, batch/ubatch 2048/512, F16 KV, 8 CPU threads, and the <=60°C start gate. Both began at 0% utilization and <=50°C; peak whole-GPU memory was 6805 MiB control and 6803 MiB candidate. Raw samples and telemetry are in `results/exp022/control.json` and `candidate.json`.

| Context | Control median; mean ± SD; range (tok/s) | Candidate median; mean ± SD; range (tok/s) | Median delta |
|---:|---|---|---:|
| 512 | 82.2854; 82.1467 ± 0.3637; 81.3283–82.3253 | 79.5519; 79.4138 ± 0.3065; 78.7317–79.5821 | -3.32% |
| 4096 | 79.7347; 79.6230 ± 0.2876; 78.9751–79.7623 | 77.1453; 77.0502 ± 0.2670; 76.4454–77.1726 | -3.25% |

Prefill was not measured because the candidate lost decode at both contexts and the changed dispatch targets decode-relevant fused weighted RMSNorm calls.

## ANALYSIS

The audit prevents treating the PROFILE.md 3.9% row as all normalization work. In the post-ROWS=1 mixed profile, `<1024,true,false>` accounts for 16,770 calls / 76.18 ms, while `<256,true,false>` accounts for 10,400 calls / 24.43 ms. Together they account for 100.61 ms, or 5.21% of the 1.933 s summed kernel time in that profile. The 1024 signature alone is 3.94%, matching the table's rounded 3.9% row; the separate 256 signature was omitted from that named family total. The same two-signature pattern appears in the baseline and PQ2_0 profile CSVs.

The 256-thread geometry slowed whole-model decode consistently. Fewer threads likely increased per-thread serial work for widths at or above 1024, without enough saved CTA cost to offset it. This experiment does not rule out all RMSNorm vectorization or shape-specific designs, but does reject this geometry change.

Nsight Compute remains blocked by `ERR_NVGPUCTRPERM`; no permission setting was changed. Nsight Systems provides call counts/timing and block/grid dimensions but not argument values here.

## DECISION

**REVERT.** Restored source SHA-256 is `8cc9c21485c03dbced0757bf1623529ddd1547a3542a4c6a6f44e1ed5a26db60`; active CUDA library SHA-256 is `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`, matching the pre-experiment control. No commit was made. Hash inventory is `results/exp022/HASHES.txt`.

## FOLLOW-UPS

Keep the current 1024-thread dispatch. A future RMSNorm experiment should recover exact call widths from graph/source-level metadata or targeted instrumentation before attempting shape gating, and should screen vectorized loads or multi-row CTA mappings without changing the active 1024+ branch globally.

## IMPORTANT DISCOVERIES

- RMSNorm's profile row is not a complete family total: the 1024-block signature's 3.9% excludes 10,400 smaller-signature calls and 24.43 ms.
- The dominant signature runs one 1024-thread block per CTA; the 256 signature uses 256-thread blocks. The ncols argument itself is not captured by the trace.
- Changing the fused-weight 1024+ branch to 256 threads reduced decode median by 3.25–3.32% at both tested contexts.
- Existing RMSNorm+weight and RMSNorm+weight+RoPE fusions remain active; experiment 019's Q/K/V fanout still rules out single-branch RMSNorm→FWHT elimination.
