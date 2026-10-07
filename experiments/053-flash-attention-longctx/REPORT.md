# Experiment 053: long-context FlashAttention decode

## HYPOTHESIS

Exp052 measured the active PTQ1_0 context-4096 attention path as 16 calls per decode token to `flash_attn_ext_f16<256,256,1,8,...>` plus 16 Stream-K fixups. These kernels together cost about 0.587 ms/token, compared with 0.235 ms/token at context 512. The active sm_86 Ampere configuration uses 64 threads, occupancy target 4, `nbatch_fa=64`, 128-element K/V tiles, `nbatch_combine=128`, and a two-stage pipeline. Reducing shared tile storage and changing the stage/tile geometry might let Stream-K expose more CTAs at long context, while retaining the current `ncols=8` grouping and exact attention math.

The active dispatch is `Q->ne[0] == 256`, GQA ratio 8, and one query row. The GQA switch chooses `ncols2=8`, and the query-row switch chooses `ncols1=1`. The RTX 3080 dispatches through `ggml_cuda_fattn_mma_get_config_ampere`; this helper selection was verified before interpreting candidate timings.

## IMPLEMENTATION

The valid candidate changed only the Ampere `DKQ=DV=256,ncols=8` entry in `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: K/V tiles 128→64 and the stage target 2→1. The baseline uses 67,584 bytes for two-stage shared K/V storage; the candidate uses a single-stage 17,408-byte K/V tile. The same output grouping, context, batch, F16 KV format, FlashAttention enablement, and mathematical operation were retained.

Two additional checks bounded feasibility. A two-stage 64-wide K tile failed the kernel's `nbatch_K2 == DKQ/2` static assertion. A single-stage 96/96 tile failed the `bad loop size` static assertion at `fattn-mma-f16.cuh:1071`. The failed build logs are `results/exp053/raw/cand64_compile.log` and `cand96_compile.log`. The initial two-stage 64 attempt did not produce a binary and is excluded from timings.

The single-stage 64/64 candidate compiled successfully as CUDA library SHA-256 `a89bd6460f0579701da4077ebf18c60a61b9deefc1911c228e2ca6bc1d767da9`. It was profiled and correctness-checked. The source was then restored to the baseline and rebuilt in the isolated worktree; the rebuilt library SHA-256 is `bc8ce63fe830d5b3a431113ca33e6512ba936a27f3c50b11e3f557b0099b98d3`. Source comparison against production commit `c6cdaa5` is empty. No production source was changed and nothing was committed.

## RESULT

Reject the candidate. The 64/64 single-stage tile regressed attention by 17.2% at context 512 and 11.3% at context 4096 in focused graph-call timing. Its 4096 fixup grew from 0.0359 to 0.0920 ms/token, overwhelming any benefit from the smaller main tile. The 96/96 variant was infeasible at compile time. No candidate passed the focused screen, so no full-model decode A/B pairs were run.

## CORRECTNESS

The successfully compiled 64/64 candidate passed the selected correctness checks:

- CTest: 5/5 (`results/exp053/raw/ctest_cand64.log`).
- CUDA-vs-CPU backend operations: 96/96 (`results/exp053/raw/backend_ops_cand64_cuda.log`).
- Fixed-seed PTQ1_0 model smoke: 32 generated tokens; the generated completion body matched the production baseline exactly. See `results/exp053/raw/model_smoke_base.json` and `model_smoke_cand64.json`.

`ldd` and `LD_DEBUG=libs` confirmed the baseline loaded `/home/maxsun/autonomous_projects/bonsai2-rtx3080/build/bin/libggml-cuda.so.0` (the expected production library SHA-256 is `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`) and the candidate loaded `/home/maxsun/autonomous_projects/.worktrees/exp053-flash-attention/build/bin/libggml-cuda.so.0` (SHA-256 above). Both resolved CUDA runtime and cuBLAS libraries from `/usr/local/cuda/lib64`. Loader traces are `results/exp053/raw/20261007T085517Z_PTQ1_0_smoke.stderr.log` and `results/exp053/raw/20261007T090259Z_PTQ1_0_smoke.stderr.log`.

## MICROBENCHMARK

Each Nsight Systems capture had 31 complete one-token graph replays with 1,432 nodes per replay. The attention family comprises 16 main attention calls and 16 fixups per replay. Values below are mean summed kernel duration per replay (equivalent to ms/token); parentheses give the full minimum–maximum range across replays.

| Context | Variant | Main attention ms | Stream-K fixup ms | Attention total ms |
|---:|---|---:|---:|---:|
| 512 | Baseline | 0.200702 (0.199072–0.202208) | 0.034404 (0.034240–0.034560) | 0.235106 |
| 512 | 64/64, single-stage | 0.238232 (0.236355–0.239204) | 0.037213 (0.037025–0.037376) | 0.275445 |
| 4096 | Baseline | 0.551083 (0.549544–0.552607) | 0.035861 (0.035617–0.036096) | 0.586944 |
| 4096 | 64/64, single-stage | 0.561281 (0.558918–0.564164) | 0.091991 (0.091329–0.092834) | 0.653272 |

The candidate attention-family total increased 0.040339 ms/token at context 512 (+17.2%) and 0.066328 ms/token at context 4096 (+11.3%). The fixup's larger mean and narrow per-replay range show a stable cost increase, not measurement noise. Full per-family and per-signature timing data are in `results/exp053/raw/{base,cand64}_ctx{512,4096}.profile.json`; raw `.nsys-rep` and exported SQLite files use the same prefixes.

## END-TO-END IMPACT

Not measured. Neither candidate survived focused timing, so the conditional two reversed-order PTQ1_0 A/B pairs were not warranted. Therefore this experiment has no end-to-end throughput range or peak-VRAM measurement. The previously verified best remains about 83.35 tok/s at context 512 and 80.35 tok/s at context 4096; Exp053 establishes no new best.

## ANALYSIS

The proposed smaller tile did reduce per-stage shared K/V storage, but in the valid single-stage configuration the decode graph did more total attention work. At context 512, both the main kernel and fixup slowed. At context 4096, the main kernel slowed slightly and the fixup cost rose by about 0.056 ms/token. The profile observes kernel durations, not CTA occupancy or hardware counters; no claim is made about why the scheduler selected that partial-work geometry. Nsight Compute counters were not collected, and no driver permissions were changed.

An initial source edit targeted the Turing helper rather than the Ampere helper. Those captures exercised the baseline path and were discarded. The reported candidate captures were taken only after changing the active Ampere helper and verifying the candidate library path. This dispatch correction is why the report includes the failed-build and final candidate build hashes.

## DECISION

Restore the baseline source. Keep the existing Ampere configuration and current best. Do not add the 64/64 tile or the infeasible 96/96 tile. The experimental worktree was rebuilt from restored baseline source; its `fattn-mma-f16.cuh` diff is empty. Manager checkout remains untouched. No commit was created.

## FOLLOW-UPS

- Close the current long-context tile-size hypothesis unless a new premise reduces the Stream-K partial-result/fixup cost.
- Preserve the active dispatch and this profile as the baseline for future context-4096 attention work.

## IMPORTANT DISCOVERIES

- On sm_86 the 256-wide, GQA-8 decode path selects the Ampere configuration and the `(ncols1,ncols2)=(1,8)` kernel.
- The current two-stage pipeline statically requires `nbatch_K2=DKQ/2`; reducing the K tile requires changing the stage target.
- The 64/64 single-stage option compiles and is correct but increases attention time at both tested contexts. The 96/96 option is rejected at compile time by the kernel's loop-size invariant.
- The earlier 64/96 profiles produced before correcting the helper selection were no-op controls and are not used in the timing table.

## EXACT COMMANDS AND RAW ARTIFACTS

Worktree creation and build configuration:

```bash
git worktree add /home/maxsun/autonomous_projects/.worktrees/exp053-flash-attention \
  -b exp053-flash-attention c817389965760726028f3106e8d88498d8f29aab
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_COMPRESSION_MODE=size \
  -DGGML_CUDA_NCCL=ON -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_CUDA_FORCE_MMQ=OFF
TMPDIR="$PWD/.cuda-tmp" cmake --build build --target llama-bench -j 8
```

The first build attempt without a worktree `TMPDIR` failed when `nvcc` wrote an intermediate under `/tmp`; subsequent CUDA builds used `$PWD/.cuda-tmp`. The valid candidate profile command, with `{context}` set to 512 and 4096 and `{prefix}` set to `cand64_ctx{context}`, was:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --cuda-memory-usage=true --force-overwrite=true --output results/exp053/raw/{prefix} \
  build/bin/llama-bench \
  -m /home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 \
  -r 2 -o json -p 0 -n 16 -d {context}
nsys export --type sqlite --force-overwrite=true \
  --output results/exp053/raw/{prefix}.sqlite results/exp053/raw/{prefix}.nsys-rep
python3 results/exp052/analyze.py results/exp053/raw/{prefix}.sqlite
```

Baseline captures used the same command and settings with `/home/maxsun/autonomous_projects/bonsai2-rtx3080/build/bin/llama-bench` and the `base_ctx{context}` prefix. Start checks before profile runs were within the experiment's <=60 C and <=5% utilization bounds; full end-to-end A/B start gates were not used because no candidate survived.

Correctness commands:

```bash
ctest --test-dir build --output-on-failure -R \
  'test-quantize-fns|test-ptq1_0-element-map|test-ptq1_0-cuda-dot|test-pq2-row-shapes|test-fwht-rms-q8'
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT \
  -p '^type_a=(ptq1_0|pq2_0),type_b=f32,m=(67|70),n=(1|2|4|8),k=(1024|5120|6144|17408)'
LD_DEBUG=libs python3 tests/model_smoke.py --binary build/bin/llama-cli \
  --model PTQ1_0=/home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --output results/exp053/raw/model_smoke_cand64.json
```

Raw timing artifacts, correctness logs, model smoke outputs, and loader traces are in `results/exp053/raw/`.
