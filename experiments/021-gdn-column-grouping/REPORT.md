# Experiment 021: GDN columns per warp on sm_86

## HYPOTHESIS

The post-ROWS=1 profile assigns 4.6% of mixed GPU kernel time to the active `gated_delta_net_cuda<128, KDA=false, KEEP=false, RAW=true, G_PRECOMPUTED=false>` specialization. Changing columns handled by each warp may trade CTA count and q/k reuse against per-thread recurrent-state registers, improving sustained batch-1 decode.

## IMPLEMENTATION

Changed only the NVIDIA Ampere+ selector for `S_v == 128 && !KDA` in `ggml/src/ggml-cuda/gated_delta_net.cu`, compiling values 1, 2, 4 (control), and 8. The kernel arithmetic, layouts, warp count (4), and other specializations were unchanged. With 4 warps per CTA, the S_v=128 grid has 32/16/8/4 column tiles per head for values 1/2/4/8 respectively. All candidate libraries were compiled from the tested selector and linked against the same build objects; candidates are isolated under `results/exp021/libs/`.

The initial Ninja dependency log was truncated, which caused broad rebuilds. After one clean CUDA build, each candidate was compiled with the generated NVCC command for `gated_delta_net.cu` and relinked against the build’s CUDA objects. The three candidate `.so.0.21.0` files have distinct SHA-256 hashes. No production source change remains.

The model profile confirms actual dispatch: the active trace row is `gated_delta_net_cuda<(int)128, (bool)0, (bool)0, (bool)1, (bool)0>` with 6,240 launches. This is S_v=128, KDA=false, KEEP=false, RAW=true, G_PRECOMPUTED=false. The graph evaluator’s recurrent-state gather fusion is enabled by default. In a 16-token Nsight Systems comparison, default fusion emitted 864 GDN launches and no `k_get_rows_float_vec`; setting `GGML_CUDA_GDN_GATHER_FUSION=0` emitted exactly 864 such GET_ROWS launches, matching the GDN count. The graph source audit also found the existing GDN-to-cache-copy matcher (`ggml_cuda_try_gdn_cache_fusion`) and its matching cache-view/slot conditions; it was left untouched. The gather rewrite was runtime-verified by the GET_ROWS count comparison; the cache-copy matcher was source-audited, not independently toggled in this run.

## RESULT

**No repeatable end-to-end winner. REVERT.** One column led in the first candidate-then-control pair by 0.52% at context 512 and 0.41% at 4096, but in the reversed control-then-candidate pair its lead shrank to 0.10% and 0.11%. The 2- and 8-column screens ran between the two four-column controls, before control2; their medians were 0.05–0.32% below that later control, but these were not reversed-order pairs. The small differences do not establish a gain or regression. Context-4096 runs had large slow-tail outliers in both control and candidates.

## CORRECTNESS

- `bash tests/run_correctness.sh` passed on the restored source-default library: 4/4 selected CTests, 96/96 CUDA-vs-CPU PTQ1_0/PQ2_0 matmul cases, and fixed-seed PTQ1_0/PQ2_0 model smokes. Full output is in `results/exp021/correctness.log`.
- Each tested library generated a fixed-seed 32-token PTQ1_0 completion. Outputs match control after normalizing only the `[ Prompt: ... | Generation: ... ]` timing line; see `results/exp021/model_output_comparison.txt` and the `smoke_*.json` files.
- Because no candidate survived the reversed E2E comparison, the full correctness suite was run on the restored four-column source-default build, not repeated for all variants.
- The tracked `results/baseline_smoke.json` was restored byte-for-byte (SHA-256 `2a502e2d53f1ccb88c6944e4f84dc0f27f6b5ca8ea9e95e324084a0c20f7454d`).

## MICROBENCHMARK

No standalone kernel microbenchmark was used to select a winner. Full-model decode was the decision metric. Nsight Systems was used only to verify active dispatch and the gather-fusion state; compact summaries are in `results/exp021/raw/gather_*_stats.txt`.

## END-TO-END IMPACT

All benchmark runs used PTQ1_0, RTX 3080/sm_86, 99 GPU layers, Flash Attention, contexts 512 and 4096, 128 generated tokens, 7 repetitions, default warmups, batch/ubatch 2048/512, F16 KV, 8 CPU threads, and the <=60°C/<=5% utilization start gate. Every run started at or below the gate and peaked at 6,805 MiB. Exact samples and per-run GPU telemetry are in the benchmark JSON files and `measurements.csv`.

| Columns/warp | Run order | Context | Candidate median; mean ± sample SD; range (tok/s) | Control median; mean ± sample SD; range (tok/s) | Median delta |
|---:|---|---:|---|---|---:|
| 1 | candidate → control | 512 | 82.2421; 82.1146 ± 0.3363; 81.3774–82.3272 | 81.8205; 81.7364 ± 0.3149; 81.0368–81.9817 | +0.52% |
| 1 | candidate → control | 4096 | 79.7570; 79.6545 ± 0.2844; 79.0135–79.8112 | 79.4284; 79.2487 ± 0.3024; 78.7098–79.4526 | +0.41% |
| 1 | control → candidate | 512 | 81.7676; 81.6517 ± 0.3272; 80.9111–81.8155 | 81.6825; 81.5549 ± 0.3278; 80.8662–81.7815 | +0.10% |
| 1 | control → candidate | 4096 | 78.5492; 70.1108 ± 11.0161; 58.2580–79.2790 | 78.4597; 75.1786 ± 6.1162; 62.4116–79.1131 | +0.11% |
| 2 | between control1 and control2 (candidate before later control2) | 512 | 81.6299; 81.4945 ± 0.3603; 80.6785–81.6570 | 81.6825; 81.5549 ± 0.3278; 80.8662–81.7815 | -0.06% |
| 2 | between control1 and control2 (candidate before later control2) | 4096 | 78.4208; 75.8006 ± 6.6468; 60.8744–79.1755 | 78.4597; 75.1786 ± 6.1162; 62.4116–79.1131 | -0.05% |
| 8 | between control1 and control2 (candidate before later control2) | 512 | 81.4193; 81.2935 ± 0.3390; 80.5266–81.4539 | 81.6825; 81.5549 ± 0.3278; 80.8662–81.7815 | -0.32% |
| 8 | between control1 and control2 (candidate before later control2) | 4096 | 78.2267; 74.7274 ± 8.7472; 55.1341–78.9678 | 78.4597; 75.1786 ± 6.1162; 62.4116–79.1131 | -0.30% |

The later context-4096 results contain severe slow tails in both control and candidate samples. The medians remain close, so these tails do not support a small positive change.

## ANALYSIS

At one column per warp, each CTA carries less recurrent state per warp but the grid has four times as many column tiles as the control. The first run’s small apparent lead did not repeat after reversing order. The 2- and 8-column screens were within noise of the later control; their order was not reversed. The trace shows that this kernel is measurable but secondary; these mapping changes did not produce a stable whole-model benefit.

The gather-path comparison confirms the active graph skips one float GET_ROWS per GDN launch and lets the GDN kernel index the recurrent cache directly. It does not show a timing win for column regrouping. The GDN-to-cache-copy fusion implementation was preserved without modification.

## DECISION

**REVERT.** Keep the production four-column mapping. Restored source hash: `b86838997dcd6e1e4d703a4f668a92f810e5ecc4d5070019abbd4701b2c95bc6`. Restored active `build/bin/libggml-cuda.so.0.21.0` hash: `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`. The A/B four-column control artifact is separately preserved at hash `a7651512d17d57d0f4ae76440d6f8453ab9e756057d3faa65489ca49b95ba4c3`; candidate/control loader paths and hashes are in `results/exp021/loader_audit.txt` and `HASHES.txt`. No commit was made.

## FOLLOW-UPS

No follow-up is justified for this mapping sweep. Continue to prioritize measured decode changes that exceed normal run-to-run variation.

## IMPORTANT DISCOVERIES

- The model really dispatches the profiled S_v=128 scalar raw-gate specialization; the profile row records all five template booleans and 6,240 launches.
- State gather fusion is active: disabling its environment-controlled graph rewrite adds exactly one `k_get_rows_float_vec` launch per observed GDN call (864/864 in the compact trace).
- One column’s candidate-first advantage collapsed from +0.52%/+0.41% to +0.10%/+0.11% in reversed order. The 2- and 8-column screens showed no repeatable decode improvement.
- Source and active production-library hashes match the pre-experiment values; the tracked baseline smoke JSON was restored exactly.
