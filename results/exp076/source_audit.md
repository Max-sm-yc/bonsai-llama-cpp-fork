# Exp076 source audit

Checkout: `5d1b4f74d1d446c2939209562f1d8498ddf11f63` (production CUDA implementation `ffb0ef37690b902829ea1158b02b14517ed93c2b`)

## Finding

On NVIDIA CUDA builds, the premise of adding an int8 Tensor Core path is already true. `ggml_cuda_mmq_config::use_mma_data_layout` returns true when `TURING_MMA_AVAILABLE` is defined. The PTQ1_0 dispatch in `ggml_cuda_mmq_get_util_funcs` then selects `ggml_cuda_mmq_load_tiles_ptq1_0` and `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma`. On sm_86, `mma.cuh::mma(tile<16,8,int>&, tile<16,4,int> const&, tile<8,4,int> const&)` emits `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`; the sibling overload emits m16n8k32 signed-int8 MMA. Thus prompt positions already occupy the N dimension of signed-int8 Tensor Core operations. Exp073's batch-one N waste does not apply to current prefill MMQ, and a second path implementing the same math is unsupported by this audit.

The loader expands packed base-3 PTQ weights into signed bytes in the MMA shared-memory layout (`x_qs`), and stages PTQ block scales (`x_df`). The active sm_86 PTQ1_0 instantiation was compiled from `template-instances/mmq-instance-ptq1_0.cu`; its extracted cubin disassembly in `ptq1-sm86.sass.txt` contains 1,792 `IMMA.16832.S8.S8` instructions across its emitted template variants. The active MMA consumer iterates K in `QI8_0` chunks (32 values), matching the `m16n8k32` signed-int8 fragment. The adjacent `m16n8k16` overload exists in the generic primitive but is not the active PTQ1_0 MMQ consumer. The MMA output loop multiplies int32 accumulators by PTQ and Q8 subgroup scales before adding to float sums. `Q8_1` block scale handling therefore remains in the accumulation path; it is not represented by an unscaled final integer dot. Existing Q8_1 sums are not needed as a zero-point correction for this signed PTQ representation: PTQ trits are centered to signed values in expansion.

## Source paths inspected

- `ggml/src/ggml-cuda/mmq.cuh`: `ggml_cuda_mmq_config::use_mma_data_layout`, PTQ1_0 utility-function dispatch.
- `ggml/src/ggml-cuda/mmq-load-tiles.cuh`: PTQ1_0 trit expansion and per-subgroup scale staging.
- `ggml/src/ggml-cuda/mmq-vec-dot.cuh`: signed int8 MMA consumer and scale application.
- `ggml/src/ggml-cuda/mma.cuh`: sm_86 signed-int8 MMA inline assembly.
- `ggml/src/ggml-cuda/mmq-config-ampere.cuh`: PTQ1_0 schedule (256 threads, occupancy target 1, I=128; J selected by matrix/prompt shape).

The five inspected source hashes match the preexisting production implementation. The CUDA `mmq-instance-ptq1_0.cu` object compiled successfully with CUDA 13.2 for sm_86; extracted SASS evidence is preserved at `results/exp076/ptq1-sm86.sass.txt` (the extracted cubin is in `results/exp076/extracted/`). The full llama-bench build was intentionally stopped after this focused compile when the audit found no candidate.

## Candidate feasibility

A distinct optimization would have to change an already-active MMA tile schedule, its PTQ decoder, or the MMA fragment/reduction geometry. Exp065 already tested the obvious schedule lever (128 threads/I=64 to enable a two-CTA resource budget); it slowed MMQ totals +15.6%, +3.8%, and +3.3% at prompt 128/512/4096. Exp066 audited the current decoder and found no locally grounded replacement. Reimplementing signed-int8 MMA would therefore duplicate active machinery, while a tile sweep would repeat the failed schedule premise. No distinct candidate with an evidence-based expected win was identified; no speculative source modification was made.
