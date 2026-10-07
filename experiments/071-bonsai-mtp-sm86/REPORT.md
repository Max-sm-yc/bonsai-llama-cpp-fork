# Exp071: Ternary Bonsai 2 PQ2_0 + MTP-Q8_0 on RTX 3080 sm_86

**Decision: REVERT (model-only candidate).** The candidate loaded and ran on CUDA within the 10 GiB device, and speculative decoding gave a clear average speedup on this three-family natural-context set. It did not beat PTQ1_0 consistently across prompt families, and speculative completions differed materially from target-only greedy completions in three of six family/context cases, including a repetitive `corrupted` continuation. Target verification is present in the runtime, but the observed text differences were not traced to harmless floating-point ties. The evidence is not sufficient to promote it as the production best.

## Hypothesis and setup

A Q8 MTP head attached to the community PQ2_0 bundle could propose two tokens per round, amortize the target's batch-1 GEMV, and improve sustained decode on this actual RTX 3080 10 GB (sm_86). This was a feasibility and end-to-end experiment; candidate model-card claims were not treated as validation.

- Worktree branch `exp071-bonsai-mtp-sm86`, based on isolated experiment start `de1ad2cea62a5e30db48f36b85f5de7ff755ebd3`; pinned runtime/source revision `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17` and optimized inference commit `ffb0ef37690b902829ea1158b02b14517ed93c2b`.
- Runtime shared library SHA256: `860fcca9977ed5ba9f9d81ce7d310481db9e9cefbe14ac30987ceb2867b30f3e`.
- Candidate HF revision `efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51`, file `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf`, SHA256 `3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4` (matches repo `SHA256SUMS`). No candidate runtime patch was needed; pinned source has the MTP Qwen3.5 graph and inverse Hadamard handling for `nextn.embed_tokens` / trunk embedding rows (`src/models/qwen35.cpp`, source hash `056ae5e71776e1cf54d7d3eb48f34eeafa3a6d7eae2f9130044585b67c06625a`).
- Device: NVIDIA GeForce RTX 3080, compute capability 8.6, 10,240 MiB, driver 580.178.04.
- All arms used the same `llama-server` build, greedy sampler (temperature 0, top-k 1, seed 42), 128 generated tokens, 7 repetitions per each of three fixed natural prompt families, cache disabled, EOS ignored, `-ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 -c 4608 -np 1`. MTP arm added `--spec-type draft-mtp --spec-draft-n-max 2`. Each server start was gated at <=60 C / <=5% GPU utilization. See exact commands and raw JSON/log/telemetry in `results/exp071/raw/`.
- Natural context IDs were tokenized once from fixed local report prose, Qwen graph C++, and speculative implementation C++ sources, then reused unchanged in all arms at both context lengths. Source hashes and token IDs are in `natural-prompt-sources.json` and `prompt-seeds-natural.json`. A separate cycled short-prompt set is retained only as a synthetic steady-state diagnostic; it is not used for the primary decision.
- Metrics below use server-reported generation time/rate. Each family result is median of 7, with min-max and sample standard deviation. Aggregate rate is total generated tokens divided by sum of all 21 request generation times, not an unweighted mean of rates.

## Load, CUDA, and VRAM

The 7.2 GiB candidate loaded and ran with `-ngl 99`; server logs reported a CUDA MTP draft context and `draft-mtp` enabled. `nvidia-smi` telemetry recorded sustained GPU compute and peak whole-device memory of **8,485 MiB / 10,240 MiB** for MTP (1,755 MiB headroom). Target-only bundle peak was 7,593 MiB; PTQ1_0 6,445 MiB. No CPU layer-offload warning or OOM appeared. This establishes feasibility on sm_86 for the tested context and batch settings.

The runtime does not skip the target pass: `tools/server/server-context.cpp` calls `llama_decode(ctx_tgt, batch_view)` for the verification batch and then `common_sampler_sample_and_accept_n(...)` against `ctx_tgt`; accepted draft prefixes are retained only after target sampling. Draft tokens are generated from the MTP context (`common/speculative.cpp`). `llama-bench` has no speculative options, so the same request/server harness was used for every target-only and speculative arm.

## Decode results

Rates in tok/s. Each cell gives median [min–max], sample SD, over seven repetitions for that family/context. Contexts and families use identical token IDs across arms.

### Context 512

| Prompt family | PTQ1_0 | Original PQ2_0 | Bundle target only | Bundle + MTP |
|---|---:|---:|---:|---:|
| Reports | 81.20 [81.07–81.29], 0.08 | 68.87 [68.44–68.89], 0.17 | 68.83 [68.66–68.95], 0.11 | 119.74 [117.67–120.03], 1.01 |
| Qwen graph C++ | 80.83 [77.38–81.03], 1.45 | 64.91 [54.13–68.62], 5.16 | 67.00 [54.78–68.62], 4.97 | 84.76 [84.49–85.07], 0.19 |
| Speculative C++ | 76.38 [67.17–80.61], 5.64 | 65.89 [43.89–67.22], 8.99 | 65.80 [42.46–66.99], 10.12 | 69.28 [62.43–73.01], 3.97 |

Aggregate across 21 requests: PTQ1_0 **79.02**, PQ2_0 **63.80**, bundled target **63.62**, MTP **86.86** tok/s. MTP beat PTQ1_0 on reports (+47.5%) and Qwen C++ (+4.9%), but lost on speculative C++ (-9.3%). It is not a uniform gain.

### Context 4096

| Prompt family | PTQ1_0 | Original PQ2_0 | Bundle target only | Bundle + MTP |
|---|---:|---:|---:|---:|
| Reports | 78.73 [78.04–79.03], 0.33 | 67.14 [65.17–67.28], 0.76 | 67.15 [65.53–67.31], 0.76 | 62.65 [62.08–63.18], 0.39 |
| Qwen graph C++ | 76.37 [51.45–78.16], 9.64 | 61.71 [39.41–64.32], 9.15 | 62.31 [45.41–64.21], 7.02 | 65.21 [55.48–70.61], 4.91 |
| Speculative C++ | 69.00 [32.48–76.41], 19.30 | 24.90 [22.29–58.26], 14.58 | 31.75 [21.84–49.87], 11.49 | 67.33 [62.73–73.69], 4.48 |

Aggregate across 21 requests: PTQ1_0 **65.84**, PQ2_0 **45.27**, bundled target **46.61**, MTP **65.26** tok/s. MTP was slightly below PTQ1_0 pooled (-0.9%); by family it lost on reports (-20.4%), lost on Qwen C++ (-14.6%), and nearly matched speculative C++ (-2.4%). Several non-MTP 4096 timings were noisy, especially speculative C++ (32–76 tok/s); ranges are retained rather than presenting the medians as precise.

For reference only, the current `llama-bench` PTQ1_0 best is 84.407 tok/s at 512 and 81.885 at 4096. These are not directly comparable to server numbers because harness/workload differs. Under the controlled server harness here, MTP pooled 86.86 at 512 vs PTQ1_0 79.02 and 65.26 at 4096 vs 65.84; per-family evidence shows no repeatable win across contexts/prompts.

### Prefill and acceptance

Median prefill tok/s by family (reports / Qwen C++ / speculative C++):

- Context 512: PTQ1_0 1226.9 / 1218.1 / 803.1; bundled target 1191.5 / 1017.8 / 717.0; MTP 1054.4 / 1046.7 / 1003.0.
- Context 4096: PTQ1_0 1310.2 / 1091.4 / 898.3; bundled target 1302.2 / 1069.0 / 880.0; MTP 1213.2 / 1034.9 / 863.1.

MTP proposals accepted (runtime `draft_n_accepted / draft_n` summed over repetitions):

| Context | Reports | Qwen graph C++ | Speculative C++ | Overall |
|---|---:|---:|---:|---:|
| 512 | 581/609 (95.4%) | 448/861 (52.0%) | 372/1027 (36.2%) | 1401/2497 (56.1%) |
| 4096 | 302/1167 (25.9%) | 366/1025 (35.7%) | 455/868 (52.4%) | 1123/3060 (36.7%) |

The family spread is substantial. The very high short-context reports acceptance drives the positive 512 aggregate; the C++ family at context 512 has only 36.2% acceptance and MTP is slower than PTQ1_0 there.

## Output behavior and correctness evidence

Target-only outputs from the candidate bundle matched original PQ2_0 exactly for all three fixed prompts at both contexts. This supports target-weight compatibility with the original PQ2 model on this sample set.

MTP outputs were stable across the seven repetitions within each prompt, but differed from candidate target-only greedy text for three of six family/context combinations:

| Context | Reports | Qwen graph C++ | Speculative C++ |
|---|---|---|---|
| 512 | exact match | first text difference at character 178; speculative result later repeats `corrupted` | exact match |
| 4096 | first difference at character 87 | first difference at character 250 | exact match |

These are observed text divergences, not a bitwise identity or correctness pass. Target verification is implemented as described above, but this experiment did not capture logits/margins around the first divergence, so it cannot attribute the differences to batched floating-point ties. The 512 Qwen-family continuation is especially concerning as a quality signal. A one-request focused trace for this family generated 123 draft tokens, accepted 64 (36/62 proposals; acceptance ratio 52.0%), and emitted per-verification counts in `natural_r1_pq2_0_mtp_ctx512_trace.server.log`. The output first differs at character 178; that places it approximately around verification round 21 by cumulative emitted-token counts, where the trace records 2/2 accepted. Character positions cannot identify exact token positions, and the runtime trace does not print the draft token IDs or target-vs-draft logits at that point. Thus the trace cannot establish whether the first text divergence was a rejected proposal, later continuation after an accepted proposal, or a batched floating-point tie. The server target-verification path is active; the tie explanation remains unproven.

## Conclusion

The model loads, fits with margin, runs MTP on CUDA sm_86, and can improve decode substantially on a high-acceptance prompt family. It fails the promotion bar for the current PTQ1_0 best: the pooled gain is prompt-family dependent, context 4096 has no aggregate improvement, and observed greedy output divergences include a pathological continuation that was not explained by target-logit evidence. Revert the candidate runtime/model selection and retain no experimental source changes. Preserve this report and raw measurements for follow-up correctness investigation.

Community report versus official status: the model card's RTX 4080 result is a community measurement; official Bonsai-demo documentation describes community Bonsai 2 MTP routes as experimental. Neither claim substitutes for this sm_86 trial.
