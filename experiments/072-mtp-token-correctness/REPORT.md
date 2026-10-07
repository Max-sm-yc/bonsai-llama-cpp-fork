# Exp072: MTP token correctness audit on RTX 3080 sm_86

## HYPOTHESIS

Exp071's MTP run diverged from target-only greedy output in three of six family/context cases. Candidate causes were an incorrect verification decision, different target logits from batched versus single-token decode, sampler state/position/seed behavior, or ordinary continuation after a prior divergence. The required discriminator is token IDs and logits at a shared prefix. This audit did not reach that discriminator because the instrumented runtime could not be built within the available user disk quota.

## IMPLEMENTATION/DIAGNOSTIC

- Verified starting `HEAD` was `32aaee6ee8f9eafb788f9c9f815140b23f1bd600`; the working branch is `exp072-mtp-token-correctness` in an isolated worktree.
- Inspected `tools/server/server-context.cpp` around speculative verification, `common/speculative.cpp` draft generation, `common/sampling.cpp` sampling/accept logic, and `src/models/qwen35.cpp` MTP graph construction.
- Added opt-in `LLAMA_MTP_TOKEN_TRACE` instrumentation in `common/sampling.cpp`. It records the chosen token, raw target-logit top five and top-1/top-2 margin per sampler call, and each MTP draft-versus-target accept/reject decision. No production-best code or main worktree files were changed.
- Configured a fresh CUDA sm_86 build. Compilation failed with `Disk quota exceeded` in compiler temporary output; `quota -s` showed the user's 12,795 MiB quota at its limit. Removed the incomplete build directory. No diagnostic server run or token/logit trace was produced.
- Cached model artifact was found at `models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` in the main checkout; its identity is recorded in Exp071.

## RESULT

The audit is incomplete. The reproducible Exp071 ctx512 Qwen graph C++ case remains the best target for a trace: 512 fixed prompt tokens, seed 42, 128 generated tokens, greedy temperature 0/top-k 1, and the first reported text difference at character 178. Exp071's focused MTP trace reports 36/62 proposals accepted and 2/2 at the approximate round suggested by character position, but that character alignment does not identify the first differing token. Existing raw data is under `results/exp071/raw/`.

## CORRECTNESS

No direct token-level correctness conclusion is supported. The source verifier calls `common_sampler_sample` for each target verification row, accepts draft token `i` only when the target sampler returns the same ID, and stops at the first mismatch. The sampler then emits its target-selected token. This establishes the intended verification semantics at source level, not the exact runtime behavior at the divergent prefix.

The unanswered comparison is whether single-token and MTP batched target logits at identical prefixes choose the same token, including the max/relative logit deltas and top-1/top-2 margins. No claim of harmless ties is made.

## END-TO-END RELEVANCE

Exp071 found MTP output divergence in 3/6 family/context cells, including ctx512 Qwen graph C++ and ctx4096 reports and Qwen graph C++. Speculative C++ matched both contexts. The MTP candidate reached 8,485 MiB peak device memory and was already rejected: pooled server decode was +9.9% at ctx512 and -0.9% at ctx4096 versus PTQ1_0, with family-dependent results. This audit makes no performance recommendation and does not alter the production path.

## ANALYSIS

Sampler source shows token-by-token exact-ID acceptance against the target sampler, including its evolving sampler state. Therefore, a rejected draft by itself should emit the target sample for that same verification row. Divergence could still arise if batched target logits differ from sequential logits, if target sampler behavior differs across the run paths, or if a later token diverges after an earlier matching token. Exp071's text alignment and aggregate acceptance counts cannot distinguish those explanations.

The attempted build failure is an environmental limit, not evidence about model correctness. The opt-in trace code is committed for a rerun after quota space is available; it logs top-five logits and decision IDs but does not capture a complete logits vector. A future run should compare target-only and MTP logs from the same seed/prefix and supplement the trace with a focused batched-versus-single-token replay if they differ.

## DECISION

INCONCLUSIVE; correctness gate remains failed. Keep Exp071 rejected and keep the PTQ1_0 production best unchanged. Do not describe the observed divergences as ties or harmless precision effects.

## FOLLOW-UPS

1. Free enough user quota for a CUDA build, then run the ctx512 Qwen family with `LLAMA_MTP_TOKEN_TRACE=1` for both target-only and MTP arms.
2. Align traces by generated token IDs and report the first mismatch, draft ID, target sample ID, verifier verdict, top logits and margins.
3. If the first mismatch involves different target choices at the same prefix, replay that prefix with target decode once per token and in a verification batch. Extend to ctx4096 reports/Qwen only if the first case does not locate the cause.

## IMPORTANT DISCOVERIES

- The inspected verifier checks draft tokens in order and breaks on the first unequal target sample; accepted drafts are not simply trusted without target sampling.
- Raw Exp071 acceptance totals and character offsets are insufficient to find the first differing token.
- The isolated audit branch started from the exact specified main commit; no shared/main worktree change was made.
