# Exp072: MTP token correctness audit on RTX 3080 sm_86

## HYPOTHESIS

Exp071's PQ2_0+MTP candidate diverged from target-only greedy output in 3/6 family/context cases. The audit tested whether the first mismatch came from incorrect speculative verification, sampler state, or target logits that depend on verification batch shape.

## IMPLEMENTATION

- Starting main commit: `32aaee6ee8f9eafb788f9c9f815140b23f1bd600`. The fresh Luna experimenter inspected the verifier, sampler, draft generation, and Qwen3.5 MTP graph, then added opt-in token/logit tracing in `common/sampling.cpp` and `tools/server/server-context.cpp` in its isolated worktree.
- Its first fresh build stopped at the user's `/tmp` quota. The manager removed stale CUDA compiler scratch under `/tmp`, moved compiler temporaries to the project build directory, and applied the trace-only patch locally. After restoring both source files, a clean Release CUDA sm_86 `llama-server` rebuild completed successfully (397 Ninja steps) at project commit `950c7cc`. `ldd` resolves project-local libraries and `MTPTRACE` strings are absent from the rebuilt common/server libraries. The new CUDA-library SHA-256 is `7b0c851cd4c1ff7800dfe88aaf1e102922a5712274f7f70bc320532452b991a1`; relevant runtime source files are identical to best code commit `ffb0ef3`.
- The diagnostic used the cached `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` bundle, fixed Exp071 natural prompt token IDs, Qwen source prompt (index 1), 512 prompt tokens, 128 generated tokens, seed 42, temperature 0/top-k 1, `-ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 -c 4608 -np 1`. Target-only and MTP traces use the same bundle target weights; MTP adds `--spec-type draft-mtp --spec-draft-n-max 2`.
- Trace rounds 2–4 exposed logger omissions and are retained as development diagnostics. Round 5 captures both sampler branches, verification decisions, and final server emissions. `results/exp072/raw/server_bench.py` and `prompt-seeds-natural.json` reproduce the focused request; `audit_trace.py` checks sampler-to-emission equality and recomputes the shared-prefix score margins from the raw logs. Raw logs and aligned top-logit data are in the same directory.

## RESULT

The two outputs match through generated position 65. At position 66, target-only emits token **1167**, while MTP's target verifier emits **6195**. MTP's draft token at that row is **18912**; it is rejected, and the verifier emits 6195. Across the 128-token diagnostic request, target-only's 128 sampler selections match all 128 server emissions, and MTP's target-context selections also match all 128 emissions.

At this identical prefix, after the configured logit bias and sampler transforms, target-only scores token 1167 at 8.84235477 and 6195 at 8.55384827, a 0.28850650 margin for 1167. The MTP batched target scores 6195 at 8.84262085 and 1167 at 8.83678246, a 0.00583839 margin for 6195. The relative score gap shifts by about 0.294345. The complete top candidates and trace lines are recorded in `results/exp072/raw/aligned_logits_ctx512_qwen.csv` and the round-5 server logs.

After the clean rebuild, the fixed-seed 32-token CUDA smoke passed for PTQ1_0 and PQ2_0. Both completions exactly match `results/baseline_smoke.json` after removing only the build-banner line and timing line; the captured results are in `results/exp072/raw/clean_rebuild_smoke.json`.

**Clean-build decode refresh:** the standard harness then ran seven samples per format at contexts 512 and 4096, with 128 generated tokens and the usual `-ngl 99 -fa on -b 2048 -ub 512 -ctk/-ctv f16 -t 8` settings. PTQ1_0 medians were 84.7407 and 82.1701 tok/s (peaks 6,803 MiB); PQ2_0 medians were 70.9778 and 68.8237 tok/s (peak 7,949 MiB). PTQ1_0's 512/4096 medians are 0.40%/0.35% above the prior paired-current medians, which is consistent with a rebuild smoke but is not a paired performance comparison or an optimization claim. Per-sample ranges and full command/hardware telemetry are in `results/exp072/raw/clean_rebuild_decode.json`.

## CORRECTNESS

The verifier's reject-and-emit behavior is correct relative to its own batched target sample in this case. Exact greedy parity against target-only sequential decode **fails** at position 66. The evidence establishes batch-shape-sensitive target logits as the immediate divergence boundary; it does not isolate the underlying model graph, recurrent-state, or CUDA-kernel cause. Do not treat the score movement as harmless tie noise: the candidate ordering reverses and the relative gap moves by ~0.294.

## MICROBENCHMARK

Not applicable: this is a correctness diagnostic, not a kernel candidate. No isolated kernel timing was collected.

## END-TO-END IMPACT

The token-level diagnostic itself is not a performance screen. Exp071's MTP comparison remains +9.9% pooled at context 512 and -0.9% at 4096 versus PTQ1_0, with 8,485 MiB peak VRAM. The clean-source rebuild refresh produced the PTQ1_0/PQ2_0 medians above, with no paired baseline comparison and no production source change. Since exact MTP greedy equivalence fails and long-context MTP throughput regressed, the MTP model bundle is not promoted. Production PTQ1_0 remains unchanged.

## ANALYSIS

The verifier does not blindly accept the proposal: at the first divergence it samples the target batch row, rejects draft 18912, and emits target token 6195. The same target weights on the same prefix in the one-token path instead rank 1167 above 6195 by 0.2885. The first cause is therefore upstream of the verification decision and is sensitive to target decode shape. The sampler's final output plumbing is consistent in both runs. More instrumentation is required to localize the numerical change within batched target evaluation; no specific kernel mechanism is established.

## DECISION

**REVERT candidate; correctness parity fails.** Diagnostic question resolved at the verifier-versus-batched-logit boundary. The community MTP bundle remains rejected for production; the verified PTQ1_0 best is unchanged.

## FOLLOW-UPS

1. Close MTP performance promotion for this bundle: it lacks target-only greedy parity and has no context-4096 gain.
2. If a later model/runtime change creates a compelling performance case, replay the recorded prefix with row-by-row intermediate logits and recurrent/KV state checks to locate the batch-shape-sensitive target change before reconsidering correctness.
3. Resume inference optimization on the measured PTQ1_0 batch-1 GEMV bottleneck using a genuinely new design premise; prior decoder, layout, staging, and scheduling results are indexed in `research/EXPERIMENTS.md`.

## IMPORTANT DISCOVERIES

- A rejected draft emits the target verifier's sampled token exactly in the captured path; MTP output divergence is not caused by accepting draft 18912.
- At the first divergent shared prefix, one-token target decode chooses 1167 while three-position MTP target verification chooses 6195. The winner margin in the MTP batch is narrow, but the underlying relative logit gap changes by ~0.294, so this cannot be dismissed as a tie.
- The exact source of target batch-shape sensitivity remains unlocalized. No MTP correctness or performance change is retained in production.
