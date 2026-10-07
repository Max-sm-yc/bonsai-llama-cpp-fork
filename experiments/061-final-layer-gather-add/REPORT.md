# Experiment 061: final-layer gather + residual add

## Hypothesis and decision

The narrow final-layer path in Qwen3.5 can compute two indexed F32 reads and their residual sum in one CUDA launch. The fused result may remove two graph nodes, but the path occurs only once per token. **REVERT:** the matcher activated and was exact, and its focused savings repeated, but two reversed-order model A/B pairs were flat at both contexts.

## Runtime graph evidence

Captures used PTQ1_0, batch 1, `-p 0 -n 16 -r 2`, and Nsight Systems node tracing. The base graph had 1,384 CUDA nodes per replay at both contexts, with 31 complete replay groups. Baseline signature counts were 50 F32 `k_get_rows_float` calls and 49 F32 ADD calls per replay.

The one-shot scheduler metadata dump identified the target chain at nodes 4613–4615:

| Node | Operation | Output shape / strides (bytes) | Use count | Flags |
|---:|---|---|---:|---:|
| 4613 | GET_ROWS(attention output) | F32 `[5120,1,1,1]`; `[4,20480,20480,20480]` | 1 | COMPUTE only |
| 4614 | GET_ROWS(layer residual) | F32 `[5120,1,1,1]`; `[4,20480,20480,20480]` | 1 | COMPUTE only |
| 4615 | ADD | F32 `[5120,1,1,1]`; `[4,20480,20480,20480]` | 2 | COMPUTE only |

Both gathers consume the same I32 index tensor with shape `[1,1,1,1]` and strides `[4,4,4,4]`. The F32 source tensors have innermost stride 4 and row stride 20,480 bytes. Their active one-token decode shapes are `[5120,512,1,1]` at context 512 and `[5120,4096,1,1]` at context 4096, so the outer strides are `[4,20480,10485760,10485760]` and `[4,20480,83886080,83886080]`, respectively. The output ADD has two downstream uses. It is not an output pin. The allocator aliases the ADD output buffer with the first GET_ROWS temporary, which the fused kernel safely overwrites after reading the original attention source directly. The remaining 48 GET_ROWS and 48 ADD operations are unrelated sites and stay on the generic path.

The candidate matcher guards the exact adjacent GET_ROWS, GET_ROWS, ADD chain; F32 types; hidden width 5120; one-row contiguous outputs; scalar contiguous I32 indices; one-use gather results; non-pinned intermediates; valid data pointers; and fusion memory-range safety. Other widths, multirow outputs, and layouts fall back to the existing operations. The sm_86 kernel directly loads each source row selected by its index and writes their F32 sum.

## Correctness

`test-exp061-gather-add` passed on CUDA. It compared outputs bit-for-bit with the scalar reference for width 5120 at row indices 0 and the last valid row (2), including reversed source indices. A width-1024, two-row case exercised the generic fallback. The five selected Exp060-era CTests plus the Exp060 and Exp061 focused tests passed 7/7. A fixed-seed 32-token PTQ1_0 CLI smoke produced identical baseline and candidate completion text after removing build and timing lines.

The reported numerical tolerance is exact equality for all 5,120 output values in each focused case. Raw correctness log: `results/exp061/raw/ctest.log`.

## CUDA graph replay

Each context capture contains 31 replay groups. The candidate reduced the graph from 1,384 to 1,382 nodes and changed per-replay signature counts to 48 generic GET_ROWS, 48 generic ADD, and one fused gather+add kernel.

| Context | Base total kernel time (ms/replay) | Candidate total (ms/replay) | Base GET_ROWS + ADD (µs/replay) | Candidate GET_ROWS + fused + ADD (µs/replay) |
|---:|---:|---:|---:|---:|
| 512, capture 1 | 11.745254 | 11.743888 | 178.916 | 176.235 |
| 512, reversed capture 2 | 11.750531 | 11.744807 | 179.321 | 176.264 |
| 4096, capture 1 | 12.099010 | 12.143751 | 178.183 | 177.097 |
| 4096, reversed capture 2 | 12.107889 | 12.100202 | 178.570 | 175.351 |

The focused replacement saved 2.7–3.1 µs/replay at context 512 and 1.1–3.2 µs at 4096 across both capture orders. Whole-graph replay timings were noisier, especially at context 4096, and did not show a stable total-time reduction.

## End-to-end PTQ1_0 decode

Two reversed-order pairs used 128 generated tokens, seven repetitions per run, 99 GPU layers, FlashAttention, batch 2048, ubatch 512, F16 KV, and 8 CPU threads. Every run passed the `<=60 C` / `<=5%` utilization gate. Each binary loaded its own `build/bin/libggml-cuda.so.0` (saved `ldd` output and hashes below).

| Context | Pair 1 base → candidate (tok/s) | Pair 2 candidate → base (tok/s) | Median base → candidate | Change |
|---:|---:|---:|---:|---:|
| 512 | 84.3536 → 84.2121 | 84.2112 → 84.1393 | 84.24645 → 84.21165 | −0.041% |
| 4096 | 81.7262 → 81.7133 | 81.7037 → 81.6841 | 81.70515 → 81.70850 | +0.004% |

The pair directions disagree and the median changes are negligible. The candidate is therefore flat for model decode and does not qualify as a KEEP.

## Build and artifacts

The isolated baseline binary is Exp060 commit `195b416`; the candidate is built from manager HEAD `8dba09a` plus the Exp061 changes. The baseline and candidate resolve llama and ggml shared libraries from their respective `build/bin` directories. Comparing the two source commits in `ggml/src/ggml-cuda` and `src/models` shows only Exp060’s `concat.cu`, `concat.cuh`, and `ggml-cuda.cu` changes, so both arms include the current Exp060 code while only the candidate includes Exp061.

- Binary and CUDA library hashes: `results/exp061/raw/build_hashes.txt`
- Loader paths: `results/exp061/raw/ldd_base.txt`, `ldd_candidate.txt`
- Runtime graph metadata: `results/exp061/raw/metadata.stderr.txt`
- Node traces, SQLite exports, and profiles: `results/exp061/raw/{base,cand,base2,cand2}_ctx{512,4096}.*`
- Focused correctness and fixed-seed smoke: `results/exp061/raw/`
- Preserved candidate patch and test source: `results/exp061/raw/attempt.patch`, `test-exp061-gather-add.cpp`
- Eight model A/B run files: `results/exp061/pair{1,2}_{base,cand}_ctx{512,4096}.json`

## Analysis and follow-up

This is a valid graph-local fusion with exact values and a repeatable few-microsecond focused saving. Its single occurrence per token makes that saving too small to move end-to-end decode above measurement variation. **Decision: REVERT.** No additional tuning is warranted for this path unless a future model/layout creates more eligible sites or a larger gathered width.
