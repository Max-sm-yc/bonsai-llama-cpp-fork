# Experiment 054: grouped standard-attention projections

## HYPOTHESIS

Grouping same-input, same-shape PTQ1_0 projections could reduce batch-1 CUDA launches. In Qwen3.5 standard attention, Q plus gate is already represented by one projection output; K and V are the remaining shape-compatible pair. A useful candidate would have to group K and V while preserving their separate output tensors and the K-only normalization/RoPE work.

## IMPLEMENTATION

No implementation was made. The experiment stopped after graph/dispatch feasibility inspection because the current dedicated `mul_mat_vec_ptq1_0_pt` interface accepts one weight matrix plus an optional gate matrix. Its gate path computes the gated output, which is not the K/V operation. Grouping K/V would require new graph fusion and dispatch/allocation semantics plus a new kernel output path; it is not a guarded change to the existing Q+gate route.

`src/models/qwen35.cpp::build_layer_attn` constructs Q+gate as one `wq` projection with two views into `Qcur_full`; K and V are separate `wk`/`wv` products of the same `cur`. The code asserts equal K/V head width. The profile's active FlashAttention signature confirms 256-wide heads and GQA grouping 8; standard-attention K/V output width is therefore 8 × 256 = 2048 elements each. Q+gate width is 2 × `n_head` × 256. The PTQ1_0 profile confirms that all three PTQ kernel specializations are active, with 361 PTQ GEMV launches per token (242 plain, 79 fused, 40 fused-gate), but kernel signatures do not retain tensor names/shapes, so they cannot safely attribute the 242/79 counts to individual Q/K/V operators. No per-operator graph shape trace was available in the retained profile.

Active batch-1 dispatch in `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh` requires one destination column and launches the specialized PTQ kernel; `mmvq.cu` routes the one-column case to this function. The existing `fusion.gate` handling in the kernel performs a second dot product and gate epilogue. Reusing that path for V would be semantically wrong.

## RESULT

No candidate was built or timed. K/V have compatible source activation, type family, and output width, so a future dedicated K/V output-pair kernel is technically conceivable. The present code does not provide a safe launch-grouping hook, and the evidence does not justify a broad graph/dispatch rewrite within this bounded experiment. Q+gate is already grouped. Stop before speculative implementation.

## CORRECTNESS

No candidate exists, so no candidate correctness run was performed. Baseline graph/profile artifacts are reused from Exp053; this experiment changed no production source. No tolerances apply.

## MICROBENCHMARK

No candidate microbenchmark. The reused steady decode profiles contain 31 complete one-token graph replays per context, 1,432 nodes/replay, and 361 PTQ1_0 GEMV launches/replay. PTQ GEMV duration was 9.006 ms/token at context 512 and 9.03 ms/token at 4096 (Exp047's reported 9.014/9.022 ms/token). The context-512 profile shows PTQ specializations at 242/79/40 calls for plain/fused/fused-gate. These counts do not isolate K/V call counts.

## END-TO-END IMPACT

Not measured; there is no candidate. The verified baseline remains 83.35 tok/s at context 512 and 80.35 tok/s at 4096. No throughput delta or peak-memory delta is claimed.

## ANALYSIS

Q+gate cannot be grouped further because Qwen3.5 already emits one Q+gate projection and the active GEMV gate fusion computes both outputs in one launch. K/V are shape-compatible by model construction, but grouping them requires a new output-pair kernel and graph-level fusion that preserves the distinct K normalization/RoPE and V cache paths. The existing gate fusion is not a generic second-output interface. With 361 PTQ GEMV launches per token and no per-operator count attribution in retained profiles, the launch savings cannot be quantified from available trace data. A broader change would therefore be speculative under this experiment's stop condition.

## DECISION

**INCONCLUSIVE / NO CANDIDATE.** Source remains at production baseline commit `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`; `git diff c6cdaa5 -- src/models/qwen35.cpp ggml/src/ggml-cuda/mmvq-ptq1_0.cuh ggml/src/ggml-cuda/mmvq.cu` is empty. No build, correctness run, timing comparison, or commit was made.

## FOLLOW-UPS

- Only revisit K/V grouping with a concrete graph-level fusion design that emits separate K and V outputs without staging an intermediate concatenated weight tensor.
- Capture named graph tensor dimensions and per-operator call counts before implementation; the retained Nsight Systems kernel signatures do not identify Q/K/V weights.

## IMPORTANT DISCOVERIES

- Qwen3.5's Q and gate are already one `wq` output, sliced into Q and gate views.
- K and V share input and output width (2048 for this model's 8 KV heads × 256 head width), but have different subsequent consumers.
- Active PTQ1_0 batch-1 dispatch has one output matrix plus an optional nonlinear gate matrix, not a general multiple-output mode.
- The retained graph profile counts 361 PTQ1_0 GEMV launches per token but cannot map those signatures to individual projection names.

## EXACT COMMANDS AND RAW ARTIFACTS

Required source/profile audit commands:

```bash
sed -n '327,380p' src/models/qwen35.cpp
sed -n '301,380p' ggml/src/ggml-cuda/mmvq-ptq1_0.cuh
sed -n '530,560p' ggml/src/ggml-cuda/mmvq-ptq1_0.cuh
python3 - <<'PY'
import json
for p in ('results/exp054/raw/base_ctx512.profile.json', 'results/exp054/raw/base_ctx4096.profile.json'):
    x = json.load(open(p))
    print(p, x['graph_replays'], x['nodes_per_replay'], x['families']['GEMV']['count_per_replay'])
    for name, data in x['signatures'].items():
        if 'mul_mat_vec_ptq1_0_pt' in name:
            print(name, data['count_per_replay'])
PY
```

Copied, unmodified profile JSON artifacts:

- `results/exp054/raw/base_ctx512.profile.json` (SHA-256 `fc64684ec3be4a06b15f5634523f1bcedfda4e6c385e510ea4efba3d170c8a3a`), source `results/exp053/raw/base_ctx512.profile.json`.
- `results/exp054/raw/base_ctx4096.profile.json` (SHA-256 `1853f0b845b30991dd2cbbc4e4cc1a87dc54d6158865a38ad182c06ad640789f`), source `results/exp053/raw/base_ctx4096.profile.json`.

The corresponding source captures and SQLite traces remain at `/home/maxsun/autonomous_projects/bonsai2-rtx3080/results/exp053/raw/`. No candidate binary exists, so there are no candidate library hashes or candidate loader traces to compare.
