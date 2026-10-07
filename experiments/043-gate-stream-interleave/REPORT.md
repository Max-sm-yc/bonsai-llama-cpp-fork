# Experiment 043: PTQ1_0 fused gate/up stream interleaving

## HYPOTHESIS

The fused `<1,1,true,true>` PTQ1_0 GEMV calls `ptq1_0_pt_block_dot` serially for main and gate weights. Explicitly interleaving their independent ternary decode and DP4A work might expose instruction-level parallelism and hide dependency latency. Exp026 already established that both paths share the nine 128-bit Q8_1 activation loads, so duplicate-load removal was excluded from this hypothesis.

## IMPLEMENTATION

No candidate implementation was made. The active source helper and launcher were inspected along with the active `<1,1,true,true>` sm_86 SASS and `<1,1,true,false>` resource record. SASS was extracted from the existing source-default production library. The source-level calls are at lines 346 and 362 in `mmvq-ptq1_0.cuh`; main and gate partials remain separate.

The key audit artifact is [sass_schedule_excerpt.txt](../../results/exp043/sass_schedule_excerpt.txt). It records alternating IDP.4A instructions from the two independent accumulator chains in the compiled fused body. Initial packed loads from both weight bases are also issued before the recurrence/dot region. The activation load audit is preserved in [exp026_activation_audit.txt](../../results/exp043/exp026_activation_audit.txt), and the full gate load listing is [gate_global_loads.txt](../../results/exp043/gate_global_loads.txt).

The bounded SASS screen used the existing active library rather than building a candidate. All experiment documentation and evidence are in an isolated detached worktree at `/tmp/exp043-gate-stream-interleave`, based on manager HEAD `5c937b83827bad67d5f0e872e9ea5ecd9a012fa5`. Production code commit is `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`.

## RESULT

The source calls are sequential, but the compiled DP4A dependency chains are already interleaved. For example, independent accumulators issue at offsets `0x0870/0x0880`, then `0x0910/0x0b40`, `0x0d40/0x0e00`, `0x0fa0/0x0fb0`, and repeatedly through the unrolled body. Paired instructions reuse the same activation register while updating distinct accumulator dependencies. This removes the proposed scheduling premise: a manually joint helper has no demonstrated remaining serial DP4A schedule to fix.

## CORRECTNESS

No candidate was implemented, so no output or tolerance claim applies. No correctness test was run. Existing Exp026 found nine `LDG.E.128` activation loads in both gated and ungated specializations; this audit reconfirmed that load-list evidence without changing the code.

## MICROBENCHMARK

No CUDA-event work-plus-fold benchmark was run because no candidate survived the SASS precondition. The active gate specialization uses 98 registers/thread; the fused no-gate specialization uses 74. Both report zero stack and zero local memory. There is no measured candidate spill reduction or focused timing result.

## END-TO-END IMPACT

Not run. No candidate passed the required focused screen, so model integration and the decode A/B were not warranted.

## ANALYSIS

The scheduling opportunity is already realized by ptxas for the active sm_86 binary. Activation reuse is also already realized, as Exp026 showed. The gate body does use 24 more registers/thread than no-gate but has no stack/local spill use. Explicit source interleaving could alter register pressure and generated scheduling, but absent a measurable unscheduled dependency chain this would be speculative and could regress occupancy. The bounded experiment therefore stops before implementation.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**INCONCLUSIVE / no candidate.** No production source, build, or binary was changed. This result does not claim that a different CUDA compiler or a different helper shape could not schedule differently; it records that the active codegen already interleaves the streams and gives no evidence for implementing this source change.

## FOLLOW-UPS

Do not implement the joint helper against this compiler output. Reopen only if a future active SASS audit shows the two DP4A dependency chains becoming serial, or another measured bottleneck provides a concrete schedule change to test.

## IMPORTANT DISCOVERIES

- The nine 128-bit activation loads are shared between gated and ungated code (Exp026).
- Despite sequential helper calls in source, the active gate SASS alternates the independent DP4A accumulator chains across the unrolled recurrence.
- The active fused-gate specialization has 98 registers/thread and zero stack/local use; fused no-gate has 74 registers/thread and zero stack/local use.
- No focused timing, correctness candidate, or end-to-end result exists because no candidate was built.

## COMMANDS AND HASHES

Read-only commands used for the active build audit:

```bash
cuobjdump --dump-sass build/bin/libggml-cuda.so
cuobjdump --dump-resource-usage build/bin/libggml-cuda.so
sha256sum ggml/src/ggml-cuda/mmvq-ptq1_0.cuh build/bin/libggml-cuda.so.0
```

Observed source SHA-256: `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`.
Observed active production CUDA library SHA-256: `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`.
Manager HEAD at experiment start: `5c937b83827bad67d5f0e872e9ea5ecd9a012fa5`. Expected production code commit: `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`.

No commit was made. Production source and build remained unchanged. `research/BEST_RESULTS.json` was not modified because this bounded negative result does not replace the current-best entry.
