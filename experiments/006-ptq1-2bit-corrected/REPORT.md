# Experiment 006: corrected PTQ1_0 exact 2-bit side screen

## HYPOTHESIS

Packing each PTQ1_0 trit into two bits and retaining its half scale (34 bytes per 128-weight block rather than 28) might simplify the RTX 3080 batch-1 decode path enough to improve full-block dot throughput.

## IMPLEMENTATION

No production source, GGUF, or runtime loader was changed. `results/exp006/side_bench.cu` is a standalone full-block CUDA screen. It packs four canonical trit digits into each of 32 bytes, bounds each code write to its corresponding packed byte, and keeps scale after the 32-byte array. Its base-3 mapping follows the CPU `dequantize_row_ptq1_0` traversal and the CUDA element map: `qs[0..15]` for elements 0–79, `qs[16..23]` for 80–119, then `qh[0..1]` for 120–127.

Before GPU timing, an independent host reference applies the canonical low-byte base-3 arithmetic to every element. The GPU then checks all decoded 2-bit codes and computes exact full-block dot outputs for both representations. The harness intended to use the production SOA_ISUM word-transposed activation mapping (`ggml_cuda_ptq1_q8_word`), with signed Q8 values, four per-32-element exact integer sums, four half activation scales, and the same unbiased-trit correction `sumi - isum` as `vec_dot_ptq1_0_q8_1_multi`. Manager review found its address calculation is wrong: it uses `kb=b>>2`, `sub=b&3`, and `word=sub*8+(e>>2)`. Production uses PTQ block index `b` for its group/lane (`group=b>>5`, `lane=b&31`) and derives the Q8 word from each weight element (`word=e>>2`). The dot visits all 128 weights including `qh`, but consumes a synthetic, incorrectly laid out activation buffer.

Build: `nvcc -O3 -arch=sm_86 results/exp006/side_bench.cu -o results/exp006/side_bench`. Sanitizer command: `compute-sanitizer --tool memcheck results/exp006/side_bench 16384 50`. Raw sanitizer output is in `results/exp006/sanitizer.txt`; unsanitized CUDA-event output is in `results/exp006/microbenchmark.txt`.

## RESULT

**INCONCLUSIVE.** The packed representation and its synthetic full-block comparison passed exact checks, but the benchmark used the wrong activation address mapping and is not performance evidence for the production workload. The observed 18.4–18.8% slowdown must not be interpreted as a decoder or end-to-end result. No runtime integration or model benchmark was done.

## CORRECTNESS

For each 65,536-block run, the independent host reference matched all 8,388,608 codes; GPU unpacking matched every canonical source code, and full-block outputs matched bit-for-bit for the synthetic activation buffer (maximum absolute error 0). All generated 2-bit codes were valid trits 0, 1, or 2. A separate Compute Sanitizer memcheck run checked 16,384 blocks (2,097,152 codes), reported exact synthetic outputs and zero device memory errors. These checks do not correct the activation layout mistake or make the timing representative.

## MICROBENCHMARK

RTX 3080, sm_86, 65,536 blocks, 300 CUDA-event-timed launches per variant, 30 warmups, three runs, one CTA thread per block. Start GPU was at 47 C and idle; it ran at 1845 MHz and 49–50 C during measurements. Timing variants were run serially in the same process. Whole-device use was 173 MiB including desktop use. Three paired samples:

| Run | PTQ1 base-3 ms/launch | Packed 2-bit ms/launch | Side overhead |
|---|---:|---:|---:|
| 1 | 0.0133190 | 0.0157787 | +18.5% |
| 2 | 0.0134588 | 0.0159266 | +18.4% |
| 3 | 0.0134002 | 0.0159161 | +18.8% |

The base-3 accessor is a scalar canonical device decoder rather than the production dp4a-unrolled `vec_dot_ptq1_0_q8_1_multi`; absolute timings are not production-kernel timings. More seriously, the Q8 activation addresses differ from production: the correct standalone mapping is `group=b>>5`, `lane=b&31`, `word=e>>2`, byte `e&3`, with 32-lane groups at a `32*36`-word stride. This harness used a block-index-derived `sub` and `kb=b>>2`. Its timing only compares scalar decoders on that synthetic access pattern, so the slowdown cannot reject the production representation. No end-to-end claim is made.

## END-TO-END IMPACT

Not run. No runtime integration was attempted because the validated packed path lost in the block-dot screen. There was no model conversion or duplicate model payload, so no runtime load-time or resident VRAM measurement was needed.

## ANALYSIS

The experiment resolves the prior side-code packing and element-order errors: it uses four packed codes per byte with bounded writes and canonical PTQ1_0 element order. It does not resolve activation indexing: the standalone address is not the production SOA layout. Code correctness is established, but decoder performance is not. No conclusion about a runtime side representation can be drawn from the timing.

## DECISION

**INCONCLUSIVE.** Keep no production changes. Correct the SOA activation address and group sizing, then repeat the focused comparison only if this remains a priority; do not integrate based on this timing. Baseline source and build remain untouched.

## FOLLOW-UPS

- Correct the harness to `group=b>>5`, `lane=b&31`, `word=e>>2`, byte `e&3` and allocate one `32*36`-word group per 32 PTQ1 blocks. Recheck address bounds and run Compute Sanitizer before timing.
- After a faithful activation mapping, compare packed extraction with the canonical base-3 decoder; consider a production dp4a-unrolled version only if the screen shows a credible gain.

## IMPORTANT DISCOVERIES

- Correctly packed and unpacked all 8,388,608 trit positions across 65,536 random PTQ1_0 blocks, with exact block dot outputs.
- Compute Sanitizer reported zero device memory errors on a separate 16,384-block screen.
- The corrected packed side kernel was 18.4–18.8% slower than the scalar base-3 comparator across three paired measurements, while its per-block payload is 21.43% larger.
- The production SOA_ISUM Q8 activation layout groups 32 PTQ1 K-blocks and transposes by word. This harness's activation indexing is wrong despite being described as production-like; derive future screens directly from `PTQ1_U` in `vecdotq.cuh`.
