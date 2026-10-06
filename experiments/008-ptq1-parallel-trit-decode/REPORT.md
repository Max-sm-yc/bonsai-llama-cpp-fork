# Experiment 008: PTQ1_0 parallel floor-difference decoder

## HYPOTHESIS

For a source byte `x`, the recurrence-emitted trit at step `k` can be computed independently as `H_k - 3 H_(k-1)`, where `H_k=floor(x*3^k/256)`. Since the products through `k=5` fit 16 bits for every byte, independently calculating scaled floors could shorten the dependency chain in the original base-3 decoder. This experiment screened that distinct arithmetic formulation; it did not use a table or a side representation.

## IMPLEMENTATION

`results/exp008/parallel_bench.cu` is an isolated CUDA harness based on a reviewed copy of the experiment-007 full-block harness. It retains the PTQ1_0 28-byte block, full `qs` and `qh`, the four DP4A/isum/scale groups, signed Q8 activations, and the production SOA_ISUM mapping (`group=b>>5`, `lane=b&31`, `word=e>>2`, byte `e&3`, 32*36-word group stride). It adds an independently multiplied floor-difference decoder alongside the production recurrence decoder. Runtime sources and build were not changed.

Build/run attempted with `nvcc -O3 -arch=sm_86`. The candidate was not timed because it did not pass exact full-block output validation.

## RESULT

The candidate is rejected at the correctness gate. The baseline and host reference agreed, but the floor-difference device implementation disagreed on all 128 tested full-block outputs in the smallest diagnostic run. No candidate timing, sanitizer run, runtime integration, or E2E run was performed.

## CORRECTNESS

A host exhaustive check confirmed the mathematical floor-difference identity for all 256 source byte values and all five recurrence digits. The harness also retains base-3 source-code and host full-block checks. However, the device candidate packing/extraction implementation did not produce matching full-block results, so it is not a valid candidate. The output mismatch was observed before any timing. Per the gate, no performance measurement is reported.

## MICROBENCHMARK

Not run. Exact device output equivalence failed. GPU was idle at 45–46 C and 210 MHz at the time of the attempted check; no timing samples were collected.

## END-TO-END IMPACT

Not run because the candidate failed exactness. No model, GGUF, runtime, or build changes were made.

## ANALYSIS

The byte-level identity is correct. Source review localized the full-block mismatch to `qh`: `dot_parallel` passes `qh[0] | (qh[1] << 16)` to a helper that treats all four bytes as independent code streams. The production layout instead interleaves two streams across four consecutive weights: `[qh0 digit t, qh1 digit t, qh0 digit t+1, qh1 digit t+1]`. The helper therefore emits zeroes for two of those four positions. This experiment gives no evidence that the arithmetic formulation is faster or slower. A future attempt should first verify the direct decoder on device for every byte and digit, then add the `qh` stream interleave and rerun the production-layout block check.

## DECISION

**REJECT THIS IMPLEMENTATION; NO RUNTIME INTEGRATION.** Production source remains at the requested baseline.

## FOLLOW-UPS

If this direction is revisited, verify each direct digit byte against the recurrence on device for all 256 byte inputs, separately for each of five positions, before full-block timing. Keep the resulting device comparison exact and retain the original base-3 layout.

## IMPORTANT DISCOVERIES

- The floor-difference formula matches the recurrence mathematically for all 256 byte values and five emitted digits.
- The initial packed-lane CUDA implementation failed full-block exactness because `qh` requires paired-stream interleaving; no performance conclusions can be drawn.

## MANAGER AUDIT (post-exp009 dispatch review)

The harness transcribes the SOA_ISUM activation layout. On the target RTX 3080 / sm_86, batch-1 PTQ1_0 uses planar-transposed `GGML_CUDA_Q8_1_PT` activations and the dedicated `mul_mat_vec_ptq1_0_pt` kernel. This report's correctness failure remains valid for its harness, but it does not test the target runtime's active block-dot path.
