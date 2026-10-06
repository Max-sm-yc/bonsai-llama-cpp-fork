# Experiment 005: PTQ1_0 exact 2-bit side representation

## HYPOTHESIS

Storing one 2-bit code per PTQ1_0 ternary weight could simplify sm_86 batch-1 GEMV decode enough to compensate for the larger weight payload.

## IMPLEMENTATION

No production files were edited. A standalone CUDA full-block screen was written to `results/exp005/side_bench.cu`. It declares the existing 28-byte block layout and a proposed 34-byte side layout (32 code bytes plus the unchanged 2-byte scale). Review found that the prototype did not actually pack four 2-bit codes per byte: its converter writes 128 bytes through the 32-byte `q` field, overrunning each side block and overlapping following records. Its base-3 element accessor also uses the wrong PTQ1_0 stage order. The input activations are generated as contiguous signed Q8 samples; the indexing expression algebraically reduces to `act[b*128+i]` and does not reproduce the production warp-transposed Q8_1 layout. This is not a valid packed-side or production-kernel screen.

Build and run command:

```sh
nvcc -O3 -arch=sm_86 results/exp005/side_bench.cu -o results/exp005/side_bench
results/exp005/side_bench 65536 100
```

The tested source had an out-of-bounds side-code write/read and a field-order translation error in the base-3 element accessor. Its comparison failed, so its timings are retained only as a failed screen and are not valid evidence of a decoder speedup. Raw output and GPU status are in `results/exp005/microbenchmark.txt`.

## RESULT

The standalone comparison failed exact output agreement (65,531 of 65,536 outputs differed in the repeated run; the verifier also reported 4,244,304 decoded-code mismatches). The converter stores one byte per trit into a 32-byte packed-code field, so it writes beyond the field and can corrupt scales and adjacent records; the decoder also reads beyond the declared field. Independently, its element accessor uses a different order from the CPU reference. The test is invalid for deciding decoder performance. No runtime route, valid 2-bit conversion/loading path, or model benchmark was attempted.

## CORRECTNESS

The required exact-output gate failed in the prototype microbenchmark. There is no valid 2-bit pack/unpack implementation to verify: the prototype writes four times beyond the declared code array, and its base-3 accessor does not match the canonical `qs` stage order. The correct element order is `qs[e & 15]`, digit `e >> 4`, for `e < 80`; `qs[16 + ((e-80) & 7)]`, digit `(e-80) >> 3`, for `80 <= e < 120`; and `qh[(e-120) & 1]`, digit `(e-120) >> 1`, for the final eight values. The existing `tests/run_correctness.sh` and model smoke tests were not run because no production change was made.

## MICROBENCHMARK

RTX 3080, sm_86, CUDA 13.2, `n=65536`, 100 CUDA-event-timed launches per variant after warmup; three runs. The measurements are invalid for performance conclusions because outputs differ.

| Variant | Run 1 ms/launch | Run 2 | Run 3 |
|---|---:|---:|---:|
| Base-3 prototype | 0.075756 | 0.075812 | 0.075804 |
| 2-bit side prototype | 0.081848 | 0.081930 | 0.081960 |

All three runs reported 65,531 output mismatches. Do not interpret the apparent slower side path as a representative performance result.

## END-TO-END IMPACT

Not run. The prototype failed exact output agreement and did not demonstrate a credible kernel-level advantage. There is no measured end-to-end inference impact.

## ANALYSIS

The exact storage expansion is 28 to 34 bytes per 128-weight block, or 21.43% for PTQ1_0's encoded weight blocks. Scale storage is unchanged. The experiment did not measure model-wide converted storage, peak load-time memory, or runtime VRAM use. No claim can be made that the side representation fits under the 10 GiB limit during conversion/loading.

The timing harness is not a faithful production-kernel microbenchmark and its correctness failure means its timing cannot separate decode cost from erroneous data mapping. Therefore the experiment does not answer whether simpler decoding pays for the larger payload.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**INCONCLUSIVE.** No production changes were made, so there was nothing to revert. Retain the prototype and failure output for debugging; do not integrate it or change the baseline. The timing is invalid because memory bounds and exact representation were both wrong.

## FOLLOW-UPS

- Implement actual 2-bit packing/extraction without out-of-bounds accesses, derive the full element mapping from `dequantize_row_ptq1_0`, and verify all 128 weights against that reference before timing.
- Only after exact packed-block agreement, compare against the production `vec_dot_ptq1_0_q8_1_multi` decoder and real warp-transposed Q8_1 activation path.
- Measure converted model payload and peak VRAM before any runtime loader integration; keep the <10 GiB cap.

## IMPORTANT DISCOVERIES

- The proposed fixed-width payload is 6 bytes larger per block, exactly 21.43% over the 28-byte PTQ1_0 block.
- The first full-block prototype wrote per-weight bytes into a 32-byte array intended for packed codes and used an incorrect source element order. Its dot and code mismatch counts are diagnostic only; its timing does not measure an exact 2-bit representation.
- Production source and the existing build/binaries remained unchanged; no PTQ1 model files were modified.
