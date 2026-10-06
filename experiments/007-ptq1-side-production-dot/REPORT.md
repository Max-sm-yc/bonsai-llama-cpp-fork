# Experiment 007: PTQ1_0 side codes with production SOA_ISUM dots

## HYPOTHESIS

The 34-byte 2-bit PTQ1_0 side block (32 packed code bytes plus the original half scale) might reduce weight decode cost enough to offset its 21.43% payload increase over the 28-byte base-3 block. The screen compares full 128-weight blocks using production SOA_ISUM activation addresses and the production DP4A arithmetic structure.

## IMPLEMENTATION

No runtime, model, or GGUF files were changed. The standalone harness is `results/exp007/side_bench.cu`; it transcribes the `vec_dot_ptq1_0_q8_1_multi` base-3 byte expansion/DP4A reductions and compares it with packed 2-bit extraction feeding the same Q8 words and four DP4A accumulators. Both paths cover `qs` and `qh`, then use the same four exact 32-element `isum` corrections, Q8 half scales, and PTQ1 half scale.

For PTQ block `b` and element `e`, the harness indexes `group=b>>5`, `lane=b&31`, `word=e>>2`, byte `e&3`; byte address is `((group*1152 + word*32 + lane)*4 + (e&3))`. This is the `PTQ1_U` address with `(e>>5)*8 + ((e&31)>>2) == e>>2`. It allocates `ceil(nblocks/32)*1152` 32-bit words, including the four `ds/isum` words per lane. It explicitly prints and checks first/last block-element and `ds` addresses. This corrects experiment 006's invalid `b>>2` / `sub*8` mapping.

Build and run:

```sh
nvcc -O3 -arch=sm_86 results/exp007/side_bench.cu -o results/exp007/side_bench
compute-sanitizer --tool memcheck --error-exitcode 99 results/exp007/side_bench 16384 20
results/exp007/side_bench 65536 300
```

The side conversion and full-block outputs are checked before timing. CUDA-event measurements use 40 warmup launches followed by 300 serialized launches per sample; seven samples per variant alternate order. Raw runs are in `results/exp007/microbenchmark.txt`, `repeated.txt`, and `microbenchmark_16384.txt`; sanitizer output and GPU telemetry are in the adjacent `results/exp007` files.

## RESULT

The candidate loses in every unsanitized workload: at 65,536 blocks, the median packed-side dot is 3.4–3.5% slower across three separate invocations; at 16,384 blocks, it is 5.0% slower. Each invocation's side timing range remains above its baseline range. The exact production-layout screen therefore shows no advantage to justify a runtime path. Decision: reject the side representation for runtime integration and leave the current baseline unchanged.

## CORRECTNESS

The host generated canonical base-3 codes, required each to be in 0..2, and device conversion/unpacking matched all 8,388,608 codes in each 65,536-block timing run. Base-3 and side DP4A paths produced bit-identical outputs and matched an independent host full-block dot reference (maximum absolute error 0). The 16,384-block sanitizer run checked 2,097,152 codes and outputs with zero Compute Sanitizer memcheck errors.

The Q8 bytes are signed values in words laid out by production group/lane/element-word address. For each block, the four stored half activation scales are paired with the exact signed Q8 sums of their respective 32 values. The `qh` values contribute to elements 120–127 and the final DP4A sum, matching the production dot.

## MICROBENCHMARK

RTX 3080, sm_86, CUDA 13.2. GPU was idle at 46°C and 210 MHz before timing. Telemetry samples during repeated runs were 51°C and 1845–2010 MHz; memory use was 173 MiB including desktop use. Each sample is milliseconds per launch. Timings below are medians with observed ranges across the seven paired-order samples.

| Blocks | Base-3 median (range) | 2-bit median (range) | Side delta |
|---:|---:|---:|---:|
| 65,536, invocation 1 | 0.0186259 (0.0186163–0.0186505) | 0.0192780 (0.0192614–0.0194859) | +3.50% |
| 65,536, invocation 2 | 0.0184661 (0.0184546–0.0185303) | 0.0191009 (0.0190874–0.0191482) | +3.44% |
| 65,536, invocation 3 | 0.0184556 (0.0184491–0.0185344) | 0.0190900 (0.0190805–0.0190976) | +3.44% |
| 16,384 | 0.0029218 (0.0029212–0.0029286) | 0.0030686 (0.0030672–0.0030720) | +5.02% |

The 65,536-block candidate range is disjoint from the base-3 range in every invocation. The smaller workload is noisier on the baseline, but its candidate range is also wholly slower. Runs were serialized on one GPU, with event-timed launches and alternating variant order within each invocation.

## END-TO-END IMPACT

Not run. The focused production-style dot screen found a repeatable slowdown, so no runtime integration, model conversion, model smoke test, or decode benchmark was warranted. Existing production and model files remain unchanged.

## ANALYSIS

This corrects both material comparability errors in experiment 006: SOA activation addressing now follows production block groups and element words, and base-3 uses the same four-sum DP4A block arithmetic rather than a scalar dot. The results reject this side-code implementation for the tested RTX 3080 path. The packed representation still costs 34 versus 28 bytes per block (+21.43%), with no measured compute benefit to offset that traffic increase. The screen does not establish end-to-end impact for other devices or code-generation variants.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REJECT / DO NOT INTEGRATE.** The corrected, exact, production-layout screen repeatedly measured the 2-bit path slower. No runtime or model changes were made; baseline source and build are untouched.

## FOLLOW-UPS

- Keep the baseline PTQ1_0 path. Do not proceed to the conditional loader/VRAM or end-to-end integration work for this candidate.
- If revisited, first test a meaningfully different packed decoder or layout in this same production-style harness; the current side extraction loses before runtime integration.

## IMPORTANT DISCOVERIES

- Production SOA_ISUM maps each PTQ K-block to one lane in a group of 32; each of the 32 Q8 words is indexed by `e>>2`. The group stride is 1,152 words, followed by four per-lane scale/isum words.
- The corrected harness checked all 8,388,608 codes and exact full-block outputs and passed memcheck with zero errors.
- On repeated 65,536-block runs, packed 2-bit dots were 3.44–3.50% slower than the production-style base-3 DP4A path. The side block remains 21.43% larger.

## MANAGER AUDIT (post-exp009 dispatch review)

This harness reproduces the SOA_ISUM activation layout. On the target RTX 3080 / sm_86, `ggml_cuda_q8_1_layout_host` selects planar-transposed `GGML_CUDA_Q8_1_PT`, and batch-1 inference uses `mul_mat_vec_ptq1_0_pt`. Therefore the measured slowdown rejects this exact 2-bit SOA dot variant, but is not evidence that a side representation would lose in the active sm_86 kernel. Re-evaluate only with the planar layout and active CTA work mapping.
