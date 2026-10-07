# Exp062: repeated SSM + SiLU + L2 norm fusion

## HYPOTHESIS

The Exp060 steady-state CUDA graph has 48 repeated `SSM_CONV → L2_NORM` adjacent kernel pairs per token. For 24 pairs, the L2 input is a zero-offset view of the SiLU output from the preceding SSM node. Fusing that exact sequence should remove 24 launches while preserving the full SiLU tensor and producing the same normalized QK values. This is a repeated recurrent path and may reduce replay time beyond the isolated gather/add site tested in Exp061.

## IMPLEMENTATION

Added one guarded sm_86 CUDA path for the one-token model shape. The scheduler matcher requires the expected SSM/SILU/L2 node sequence, F32 types, dimensions and strides, use counts, view alias at zero offset, unpinned outputs, and disjoint output/input ranges. Unmatched shapes and graphs use the generic implementation. `GGML_CUDA_DISABLE_SSM_L2_FUSION=1` selects that generic path for comparisons.

The fused kernel computes and stores the complete SSM+SiLU result, then computes L2 normalization for the 32 QK groups that view its leading values. A block barrier makes the SiLU stores visible before the normalization reads them. The reduction follows the generic kernel's 32-thread accumulation order.

Changed files:

- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/ssm-conv.cu`
- `ggml/src/ggml-cuda/ssm-conv.cuh`
- `tests/CMakeLists.txt`
- `tests/test-exp062-ssm-l2.cpp`

Build source commit: `4cb2072d8534c45d0aed0da7472dfe6587e9c3ee` plus the changes above. Candidate `libggml-cuda.so.0` SHA-256: `1e29adc0a6f77047e81d45694cd63c58ef71054b464de7ec506d5db6345c76ed`. `ldd` resolved all GGML libraries from this isolated worktree. Control library SHA-256: `5a82af131b322d7a15ff3a0951ad2953a2e67eb83a874d5a1f1f724916532a26` from the Exp060 worktree.

## RESULT

The graph replay screen improved in both contexts. Paired end-to-end decode showed a repeatable gain at context 4096 and effectively flat, noisy results at context 512.

## CORRECTNESS

- Focused exact-byte comparison passed for the model shape: 10,240-element SSM+SiLU output plus 4,096 normalized output values matched the generic path exactly.
- Focused exact-byte comparison passed for the fallback shape.
- `tests/run_correctness.sh` built all requested targets, passed its selected CTests, and passed 96/96 CUDA backend operation checks. Its final default smoke invocation could not locate the model because the isolated worktree has no `models/` directory. Reran the same fixed-seed PTQ1_0 smoke with the model's absolute path; 32-token generation succeeded.
- Focused result files and full logs are in `results/exp062/raw/`.

## MICROBENCHMARK / GRAPH REPLAY

The captured one-token graph has 1,384 nodes per replay before fusion and 1,360 after fusion. It contains 48 adjacent SSM/L2 pairs per replay; 24 satisfy the exact matcher and become 24 calls of `ssm_conv_silu_l2_norm_f32`, while the other 24 use the generic path. The candidate capture confirms one fused call per matched site.

Fresh paired captures measured summed GPU kernel time per replay:

| Context | Generic control | Candidate | Difference |
| --- | ---: | ---: | ---: |
| 512 | 11.728813 ms | 11.696529 ms | −0.032284 ms (−0.275%) |
| 4096 | 12.102607 ms | 12.072064 ms | −0.030543 ms (−0.252%) |

The candidate fused calls total about 42.4–42.7 µs per replay; generic SSM and L2 each account for about 37–38 µs over their remaining 24 calls. Full captures, SQLite exports, and summaries are under `results/exp062/raw/`.

## END-TO-END IMPACT

Each arm used PTQ1_0, batch 1, `-p 0 -n 128 -r 7`, 99 GPU layers, FlashAttention on, batch 2048, ubatch 512, F16 K/V, and 8 CPU threads. Each arm started at ≤60 C and ≤5% GPU utilization. Pair 1 ran base then candidate; pair 2 ran candidate then base. Raw llama-bench JSON and the per-arm gate snapshots are in `results/exp062/raw/`.

| Context | Pair order | Base avg tok/s | Candidate avg tok/s | Change |
| --- | --- | ---: | ---: | ---: |
| 512 | Base → candidate | 84.344 | 84.321 | −0.028% |
| 512 | Candidate → base | 84.074 | 84.241 | +0.199% |
| 4096 | Base → candidate | 81.553 | 81.779 | +0.278% |
| 4096 | Candidate → base | 81.562 | 81.790 | +0.280% |

The seven-sample medians give the same picture: context 512 is mixed (−0.063%, +0.312%); context 4096 is positive in both orders (+0.188%, +0.275%). Context 4096 is repeatably about +0.28%; context 512 is flat within run variation and shows no meaningful regression.

## ANALYSIS

The graph offers 24 safe fusion opportunities per token, enough to save roughly 30 µs in a roughly 12 ms replay. The observed replay-time reduction is consistent across contexts. End-to-end impact is limited by the dominant PTQ1_0 GEMV work, but the longer-context gain repeats across both orderings. At context 512, the paired results straddle zero and are small relative to normal run spread.

The exact-view and use-count checks are essential: the graph has 48 adjacent SSM/L2 pairs, but only half have the QK-view layout required by this kernel. The remaining sites correctly fall back. The focused exact comparison also caught a missing device barrier in an early kernel draft; the final barriered implementation matches the generic output bit-for-bit.

## DECISION

**KEEP** in this isolated worktree. The candidate passes focused exact-value checks and CUDA correctness coverage, removes 24 launches per replay, improves replay time in both contexts, and gives a repeatable +0.28% end-to-end gain at context 4096. Context 512 is flat within measurement variation.

## FOLLOW-UPS

- Keep the matcher narrow and preserve the generic fallback for other layouts, use counts, and pins.
- Consider other recurrent chains only after confirming their live views, use counts, and output aliases from graph evidence.
- Do not generalize the kernel to wider QK shapes without new exact-output and replay measurements.

## IMPORTANT DISCOVERIES

- The graph has 48 adjacent SSM/L2 sites per token, but only 24 use the exact zero-offset `[128,32]` QK view of the SSM+SiLU output.
- The SiLU tensor is also used elsewhere and must remain fully written; the fusion cannot truncate SSM+SiLU stores to the 4,096 normalized values.
- The candidate needs a block barrier between global SSM+SiLU stores and the normalization reads to reproduce the generic path exactly.
- The 24-site fusion saves approximately 0.25% of full graph replay time and produces a stable context-4096 decode gain despite GEMV dominance.

## MANAGER INTEGRATION VERIFICATION

The manager integrated the candidate as code commit `ffb0ef37690b902829ea1158b02b14517ed93c2b`. It changed the epsilon load from a type-punned pointer read to `memcpy` after the first integrated compile warned about strict aliasing; this does not change the fused arithmetic. The final integrated CUDA library SHA-256 is `860fcca9977ed5ba9f9d81ce7d310481db9e9cefbe14ac30987ceb2867b30f3e`.

The integrated `test-exp062-ssm-l2` CTest passed 1/1. Its wrapper runs both model and fallback shapes with fusion disabled and enabled in separate processes and compares their outputs byte-for-byte. A main-build, fixed-seed 32-token PTQ1_0 smoke produced the same completion with fusion enabled and disabled after stripping only the CLI timing diagnostic.

Two additional manager reversed-order A/B pairs used the same seven-repetition, 128-token configuration and per-arm idle gate. Combined with the experimenter pairs, four pair-median samples per arm give:

| Context | Exp060 control median | Exp062 candidate median | Change |
|---:|---:|---:|---:|
| 512 | 84.376 tok/s | 84.407 tok/s | +0.037% (flat within variation) |
| 4096 | 81.696 tok/s | 81.885 tok/s | +0.231% |

Both additional context-4096 pairs favored the candidate by about 0.25%; the context-512 pair directions disagreed. Peak VRAM remained 6,579/6,803 MiB. The final manager runs and loader paths are in `results/exp062/manager_ab.json` and `results/exp062/raw/manager_ldd_*.txt`.
