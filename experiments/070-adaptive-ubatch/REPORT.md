# Experiment 070: Adaptive prompt-length ubatch

## HYPOTHESIS

Use token count before context creation to select a larger physical batch for long prompts and retain ub=512 for short prompts. Exp067 showed that ub=2048 helps long prefill but hurts short prefill. A simple threshold could retain that long-prompt gain without choosing settings by benchmark ID or prompt text.

## IMPLEMENTATION

The normal `llama-cli` frontend is server-backed: it starts a server, and the server creates its shared context before later chat requests arrive. Requests can have arbitrary token counts, so that path cannot choose one context's `n_ubatch` from a future request safely. The standalone `examples/simple/simple.cpp` path is different: it tokenizes its one command-line prompt before context creation (model load/tokenize at lines 89-107; context creation at 111-119).

A temporary opt-in for the standalone example selected requested ub=2048 when the tokenized prompt had at least 1536 tokens and ub=512 below that. It used the token count including tokenizer special tokens, not text or benchmark IDs. The candidate was compiled and exercised, then its source change was reverted after the batch-1 decode check. No candidate code remains. The default server and production runtime were not changed.

The context clamps effective ubatch to the configured logical batch and context size. At 1536 tokens the requested 2048 is therefore clamped to the available batch/context. `n_ubatch` is fixed during context creation; the server later reads that value during request processing.

## RESULT

The isolated screen supports a prompt-side crossover near 1024-1536 tokens. At 128 and 512, ub=2048 was slower than ub=512. At 1536 and above it was faster in both three-sample screens, but the observed gain varied between runs. The best plausible rule was `tokens >= 1536 -> requested ub=2048; otherwise ub=512`.

A seven-repetition sweep with both ubatches in one `llama-bench` process was **invalid for crossover comparisons**. The executable ran every ub=512 prompt before every ub=2048 prompt; later telemetry reached 86 C and 1065 MHz. Those raw samples are preserved but excluded from the speed claim.

## CORRECTNESS

The temporary standalone opt-in was exercised with seed 42, the same input, and 32 generated tokens for both fixed ub=512 and adaptive. Short (4-token) and long (1800-token) prompts both exited successfully and produced byte-identical generated text after stripping the echoed prompt. This compares generated text only; no logit or tensor equality is claimed. The long prompt's single-run prompt speed was 1283.44 tok/s fixed versus 1309.93 tok/s adaptive; decode was 69.71 versus 69.58 tok/s. These single-run figures are smoke data, not timing confirmation.

Whole-GPU sampled peaks for those one-shot runs were 6013 MiB (short, both arms), 6659 MiB (long fixed), and 7973 MiB (long adaptive). The standalone candidate stayed below the 10,240 MiB device limit.

## MICROBENCHMARK

PTQ1_0 prefill used `-b 2048`, warmups enabled, and the Exp067 harness on prompts 128, 512, 1024, 1536, 2048, 3072, and 4096. Initial settings were run at three repetitions each. Two additional three-repetition screens used separately gated arms in reverse order (ub=2048 then ub=512). Every run used the <=60 C / <=5% utilization start gate. The table shows medians from the initial screen; exact per-sample means, standard deviations, and ranges are in the listed JSON artifacts.

| Prompt tokens | ub=512 median | ub=1024 median | ub=2048 median | ub=1024 vs 512 | ub=2048 vs 512 |
|---:|---:|---:|---:|---:|---:|
| 128 | 1316.94 | 1303.67 | 1298.01 | -1.01% | -1.44% |
| 512 | 1394.17 | 1376.60 | 1372.88 | -1.26% | -1.53% |
| 1024 | 1388.18 | 1403.58 | 1393.49 | +1.11% | +0.38% |
| 1536 | 1385.90 | 1385.62 | 1395.44 | -0.02% | +0.69% |
| 2048 | 1380.85 | 1390.66 | 1398.87 | +0.71% | +1.30% |
| 3072 | 1370.68 | 1377.29 | 1380.12 | +0.48% | +0.69% |
| 4096 | 1358.59 | 1367.16 | 1373.56 | +0.63% | +1.10% |

In the separately gated reverse-order screen, ub=2048 medians were +2.16%, +2.40%, +2.94%, +2.55%, and +3.12% at 1024/1536/2048/3072/4096 versus the ub=512 medians; at 128/512 they were -0.27%/+0.05%. The initial screen's gains were smaller. Run-to-run drift in the ub=512 arm prevents treating the difference between these screens as a precise speedup. The seven-sample separate runs likewise showed larger deltas, but were not a paired, interleaved confirmation.

Peak whole-GPU memory in the three-setting prefill screen was 6793 MiB (ub=512), 7313 MiB (ub=1024), and 8363 MiB (ub=2048). The setting stayed within 10 GiB.

## END-TO-END IMPACT

`llama-bench` combined prompt-plus-128-token generation was run for the same long contexts in separately gated arms, each with three repetitions. At 2048 tokens, ub=2048 reached a 722.375 tok/s median versus 710.594 for ub=512 (+1.66%). At 4096, it reached 929.691 versus 909.639 (+2.20%). Whole-GPU peaks were 8371 and 6803 MiB. The policy leaves prompts below 1536 at ub=512, so their selected setting is identical to the control; the fixed ub=512 combined screen measured 157.025 tok/s at 128 and 335.169 at 512.

Batch-1 decode was measured with `-p 0 -n 128` at contexts 512 and 4096 in seven-repetition runs, then repeated in reverse arm order. At context 512 ub=2048 was flat: +0.05% in the first order and +0.01% in reverse. At context 4096 its median was lower by 1.85% and 1.77% in the two orders. Both arms had overlapping slow tails (roughly 58-82 tok/s), so the exact size is noisy, but the long-context result consistently points to a decode cost. Decode peaks were 6803 MiB at ub=512 and 8373 MiB at ub=2048.

## ANALYSIS

The length rule is feasible and the separate screens support long-prompt prefill and combined-request gains with memory below the device limit. It also keeps the measured short-prompt path at ub=512. However, the context retains ub=2048 after prefill, and the repeated 4096-context decode medians were about 1.8% lower. The objective requires preserving batch-1 decode as well as gaining prefill; the current runtime has no safe per-request or post-prefill ubatch change. Net combined throughput improved, but that does not satisfy the decode constraint.

The most controlled paired-prefill attempt used one process with `-ub 512,2048`, but llama-bench iterates ubatch before prompt lengths. Its arms were thermally separated in time and GPU clocks fell to 1065 MHz, invalidating that run for crossover claims. We retain it as a cautionary artifact, not evidence for or against the policy.

## DECISION

**REVERT / NO PRODUCTION CANDIDATE.** The one-shot example can select a generic token-count policy before context creation, and long combined requests gained 1.7-2.2%, but the policy did not meet the no-long-context-decode-regression requirement. The temporary example change was reverted. Production remains at `-ub 512`; llama-cli/server behavior is unchanged.

## FOLLOW-UPS

- Reopen only with a way to use a larger ubatch for prefill and return to the smaller setting for decode without rebuilding or losing context state, or with new matched evidence that clears the decode gate.
- Keep the one-shot-versus-server initialization distinction in mind for any future workload-aware context parameters.
- No PQ2_0 test was run because this PTQ1_0 policy did not pass the decode gate.

## IMPORTANT DISCOVERIES

- A direct one-shot caller can tokenize the entire prompt before context construction, while the default CLI delegates to a server whose shared context exists before user requests arrive.
- The measured prefill crossover is gradual: ub=2048 remained slower at 512, was nearly flat at 1024 in the first screen, and gained at 1536-4096.
- In the long one-shot smoke, adaptive ub used 7973 MiB and fixed ub used 6659 MiB; both stayed under 10 GiB and produced the same seeded completion.
- The standalone build target could not complete because nvcc/g++ hit the shared `/tmp` disk quota. The changed example was compiled and linked directly against the verified baseline libraries; its source was then reverted. No CTests or backend arithmetic tests were needed for the parameter-only experiment.

## EXACT COMMANDS AND ARTIFACTS

Exact benchmark argv, raw `llama-bench` stdout/stderr, per-repetition samples, telemetry, seed-42 completion files, and source audit are under [`results/exp070/`](../../results/exp070/). See [`raw/commands.txt`](../../results/exp070/raw/commands.txt) for the full command set and the invalid paired-run explanation. The CMake build attempt and one-file standalone compile are documented there.

Starting docs HEAD: `838a19f4fbe59532fc85f9c2dcfb257ce4146b66`. Production implementation source commit: `ffb0ef37690b902829ea1158b02b14517ed93c2b`. Baseline runtime library hashes are listed in `results/exp070/raw/source-audit.txt`; CUDA library SHA-256 was `860fcca9977ed5ba9f9d81ce7d310481db9e9cefbe14ac30987ceb2867b30f3e`. The temporary standalone candidate executable SHA-256 was `849ce6c066f05cfb32d713d81dbdad8205c7cdeafd88a49ba7bef1b0dc73d68e`.
