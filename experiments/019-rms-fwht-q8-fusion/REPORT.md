# Experiment 019: RMSNorm and FWHT/Q8_1 fusion

## HYPOTHESIS

Fuse RMSNorm plus its learned weight multiply into the existing FWHT/Q8_1 preparation kernel, eliminating the normalized activation intermediate and a launch on PTQ1_0 batch-1 decode.

## IMPLEMENTATION

No implementation was made. The active source and CUDA library were left untouched. The requested consumer audit found that this fusion cannot remove the normalized activation intermediate on the Bonsai Qwen3.5 attention path: `build_layer` computes `cur = build_norm(inpL, attn_norm, ...)` and passes `cur` into `build_layer_attn`; that builder uses it independently for Q, K, and V projections (`build_lora_mm` calls in `src/models/qwen35.cpp`). For folded weights, `build_lora_mm` adds signs then calls `llama_mul_mat_hadamard`; that graph builder memoizes the transformed output by `(cur, rotation)`, but does not make the RMS result private to a single transform.

The existing CUDA FWHT matcher begins at the optional sign multiply, then reshape and hinted Hadamard matmul. It checks every transform-result/view consumer and only virtualizes the result for PTQ1_0 matvec consumers. Its explicit overlap handling is necessary because the q8 output can alias the transform input allocation. The present fusion kernel processes independent transform-width blocks, whereas RMSNorm's scale is computed over the full row; fusing RMS work into that kernel would also require a cross-block reduction strategy (or redundant row reductions).

## RESULT

The key optimization condition fails: the normalized weighted activation is shared by separate projection branches, not solely consumed by one sign/FWHT/Q8 path. The fused consumer would still need to preserve the normalized tensor for sibling Q/K/V paths, so it cannot remove the intermediate write/read or RMSNorm launch. Proceeding would add complexity without delivering the stated benefit.

## CORRECTNESS

No candidate was built or tested because a single-branch intermediate elimination would leave sibling consumers without the normalized activation. Manager independently checked `src/models/qwen35.cpp`: `build_layer` creates `attn_norm` and passes it into `build_layer_attn` (lines 198–210); `build_layer_attn` passes that same `cur` to Q, K, and V `build_lora_mm` calls (lines 327–355). `src/llama-graph.cpp` applies optional signs and Hadamard transforms in each projection builder (lines 1569–1573); its transform memo is keyed by `(cur, rotation)` and does not make the original normalized activation single-consumer. No code or binary changed. SHA-256 of `ggml/src/ggml-cuda/ggml-cuda.cu`: `2ae527aba2b42f6658a2b856c605a8d40e2f3e1ee8c37a3448cb76c409b28e5e`; active `build/bin/libggml-cuda.so`: `72d89c0c69200865b2200ef35b94e14b9a6a52a840c17cb031e987d809207e72`.

## MICROBENCHMARK

Not run; there is no candidate kernel.

## END-TO-END IMPACT

Not run; there is no candidate implementation.

## ANALYSIS

The model graph establishes that the norm output feeds three projection builders. The Hadamard memo only shares a transform when multiple consumers use the same rotation; it does not eliminate the original norm result needed to construct distinct transforms or feed other projections. A fusion could only remove the RMS result when all of its consumers are transformed together in one coordinated multi-output operation, which is outside this bounded FWHT/Q8 preparation fusion. In addition, the current per-transform-block CTA mapping does not directly provide the full-row RMS scale.

## DECISION

**DO NOT IMPLEMENT in this single-branch form.** No tracked CUDA source or binary modifications were made; hashes above identify the existing state. This is a structural finding, not evidence against coordinated multi-consumer normalization reuse.

## FOLLOW-UPS

- Consider a coordinated multi-projection fusion only if graph inspection shows a profitable way to consume all Q/K/V branches together and preserve required outputs.
- Otherwise target the separately measured RMSNorm and FWHT/Q8 costs independently.

## IMPORTANT DISCOVERIES

- On the model attention path, `attn_norm` output is shared into Q, K, and V projection construction.
- The normalized weighted activation precedes the signs multiply for eligible folded projections, but it is not single-consumer.
- FWHT output virtualization already protects its own consumers and allocation aliasing; that does not imply the upstream normalized activation can be discarded.
