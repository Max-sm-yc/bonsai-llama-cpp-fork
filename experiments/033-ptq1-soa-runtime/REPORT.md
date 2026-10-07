# Experiment 033: PTQ1_0 SoA runtime feasibility

## HYPOTHESIS

Exp032 found exact no-padding SoA storage (seven 32-bit word planes per row) tied for 40 K blocks/row and improved the 136-block/row screen by about 7.8%. A CUDA runtime could potentially hold only the SoA representation on device, retaining the original 28 bytes/block footprint and improving long-row GEMV.

## IMPLEMENTATION

No source or build changes were made. This was a bounded source-path and upload-path feasibility audit. The active checkout is `research/rtx3080` at manager HEAD `1f4e6a84ae36e924accc9556d069e350ad7551a`; this is not the best-code commit identified in the research state, so the experiment deliberately did not switch branches or modify source. No isolated worktree was needed because no prototype was attempted.

The inspected paths include `ggml-cuda/ggml-cuda.cu` tensor set/get callbacks (including byte-range and 2D copies), `mmvq.cu` and `mmvq-ptq1_0.cuh`, `mmq.cu`/`mmq.cuh`/`mmq-load-tiles.cuh`, `vecdotq.cuh`, `dequantize.cuh`, `convert.cu`, and `getrows.cu`. The CUDA MMVQ launch path uses ordinary `block_ptq1_0` row/block addressing in plain and fused gate kernels and supports multi-column execution. CUDA MMQ, conversion/dequantization, and row gather also interpret the tensor as AoS blocks. CPU vec-dot/dequantization and other backends likewise expect the canonical GGUF/AoS encoding.

The generic CUDA upload callback is a raw `cudaMemcpyAsync` for arbitrary byte ranges; its 2D sibling copies rows using caller-provided strides. It does not carry a special PTQ1_0 representation contract. Moving the active batch-1 GEMV to SoA alone would therefore break the tensor for the other CUDA consumers. Keeping an AoS device copy alongside SoA would violate the single-copy/VRAM requirement.

## RESULT

No runtime candidate was feasible within this focused experiment. A single SoA device copy is possible in principle only as a coordinated CUDA physical-layout change: transform each complete PTQ1_0 row before/while uploading, maintain the existing logical tensor size/row stride contract, and update every CUDA path that reads weights (including multi-column and fused-gate paths, MMQ, dequantization/conversion, and get-rows). The public/generic tensor upload operation also permits partial offsets and 2D row copies, so conversion cannot safely be inserted as an unconditional byte copy without defining its behavior for each transfer form and view.

This is an integration and correctness scope blocker, not evidence that the layout cannot be made to work. It is not safe to test only the batch-1 kernel because ordinary prefill/multi-column or utility operations would read scrambled AoS bytes. Repacking only inside one GEMV call would add a full-size temporary allocation or repeated work and is not an acceptable single-copy implementation.

## CORRECTNESS

No runtime correctness gates were run because no runtime candidate was built. Exp032's standalone screen passed exact code/output checks and Compute Sanitizer, as documented in its report. This experiment makes no new runtime correctness claim.

## MICROBENCHMARK

No new microbenchmark was run. Exp032 remains standalone screening evidence only: 40 blocks/row tied; 136 blocks/row improved 7.82% in the manager rerun. It does not include production loader, CUDA MMQ, conversion, row-gather, or multi-column behavior.

## END-TO-END

Not run. There is no candidate runtime to compare, so no production speedup is claimed.

## VRAM / LOAD COST

No load-time or peak-VRAM candidate measurement was made. SoA has exactly the same nominal payload size (28 bytes per block), so the final single GPU copy would add zero weight payload if all consumers used it. A retained AoS plus SoA copy would add another full PTQ1_0 weight set; the project's 6,805 MiB whole-GPU baseline plus that copy risks exceeding the 10 GiB RTX 3080 limit. A host-side per-row repack could avoid a second device copy, but its CPU time, staging memory, loader integration, tensor views, and partial upload handling are unmeasured.

## ANALYSIS

The intended layout is per tensor row: for row `r`, word `w` and K block `b`, the word is at `row + w*nblk + b`. The CUDA kernels currently derive `block_ptq1_0 *` pointers by row stride and K-block index. The mismatch affects more than the active ROWS=1 specialization. Multi-column MMVQ uses the same physical block access, while MMQ tile loaders, gate operands, conversion/dequantization, and get-rows all use canonical block fields. Preserving non-CUDA formats/backends also requires keeping the serialized GGUF encoding canonical and restricting any alternate layout to CUDA device storage.

The safe next direction is a dedicated CUDA buffer/tensor physical-layout design with explicit metadata or a CUDA-private wrapper, plus row-wise loader conversion into the one final allocation. Before performance work, inventory and adapt every CUDA reader and define get/set semantics for full tensors, subranges, 2D copies, views, and host/device copies. Then verify multi-column/prefill and batch-1 decode paths before any E2E A/B. An in-kernel temporary conversion is not attractive unless it can be proven to avoid extra resident storage and repeated conversion overhead.

## DECISION

**STOP / INFEASIBLE FOR THIS BOUNDED PROTOTYPE.** Do not alter source/default library. The experiment found no safe narrow integration point because the tensor's physical representation is assumed by multiple consumers and by generic upload semantics. Continue only as a separate coordinated layout project with an explicit CUDA-private representation contract.

## FOLLOW-UPS

- Preserve the Exp032 screen as the performance motivation; do not present its microbenchmark as runtime evidence.
- If resumed, first establish a CUDA-private storage descriptor/representation marker and enumerate every CUDA reader, including every multi-column path.
- Define loader conversion with row alignment and upload offset/view semantics before editing kernels.
- Keep a single device allocation; measure CPU repack/load time and total device peak during the first prototype.
- Run multi-column CUDA-vs-CPU PTQ1_0/PQ2_0 cases and utility operation tests before model inference or benchmarks.

## IMPORTANT DISCOVERIES

- The CUDA buffer upload callbacks perform generic raw byte copies; they do not transform quantized rows.
- PTQ1_0 physical AoS assumptions appear in batch-1 and multi-column MMVQ, fused gate, MMQ tile loading, conversion/dequantization, and get-rows.
- A successful SoA loader path must preserve canonical GGUF/AoS bytes on host and confine the alternate layout to a marked, single CUDA allocation. Without updating all readers, changing the upload bytes silently corrupts non-decode operations.
- Source and active source-default library SHA-256 values remain the documented values: `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496` and `c828135b126ec507ffbecb4dc11b6a7a9ac5cd0fe050553323d7f35c38fae6c7`.

## Manager follow-up: selective long-K sidecar sizing

The audit ruled out a full single-layout conversion as a bounded GEMV-only change, but it does not rule out a selective sidecar. The manager parsed the local PTQ1_0 GGUF with `gguf-py`: exactly 64 tensors have shape `[17408, 5120]` (`blk.*.ffn_down.weight`), each 19,496,960 bytes, totaling 1,247,805,440 bytes (1,190 MiB). Keeping SoA sidecars only for these K=17,408 matrices while retaining their original AoS tensors projects 6,805 MiB baseline peak to about 7,995 MiB, below the 10,240 MiB card limit. This is a projection, not measured runtime allocation. The reproducible inventory script/output are `results/exp033/count_selective_soa_bytes.py` and `results/exp033/selective_soa_bytes.txt`.

This is a distinct follow-up design: a sidecar registry used only by the batch-1 ROWS=1 GEMV can preserve all existing consumers' AoS storage, at an estimated 1,190 MiB cost. Loader timing, allocation peak, and E2E impact remain unmeasured.
