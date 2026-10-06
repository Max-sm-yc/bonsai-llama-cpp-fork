# Experiment 011: exact 2-bit side codes on planar PTQ1_0 GEMV

## HYPOTHESIS

An exact 2-bit-per-weight code can replace the active sm_86 planar PTQ1_0 base-3 decoder, and its simpler decode may offset the 21.43% increase in PTQ1_0 block bytes. The decision metric is end-to-end decode with conversion/repacking, runtime storage, and VRAM included.

## IMPLEMENTATION

The working tree started at docs commit `3d045e695e178c4f9c8b7bc1da8e4d3454288ab4`; active production source remained the `9fa97200e68fd798ef027470c8e420172a0ac719` ROWS=1 kernel. No production source was modified.

`results/exp011/planar_side_bench.cu` is a standalone CUDA screen of the active planar activation map. It generates canonical PTQ1_0 blocks, exactly repacks each block into 32 code bytes plus the existing half scale, and runs the base-3 dot plus two side decoders: scalar 2-bit field extraction and packed-byte expansion directly into DP4A's four byte lanes. One CUDA thread handles one K block, as in the active block-dot portion of ROWS=1. The activation address is derived from the active kernel's `int4` planes: 8 planes of four Q8 words per block and a ninth plane with four (scale, isum) half2 records. The measured kernel excludes conversion; this favors the side representation relative to any runtime conversion path.

The matched current-best decode control was run with the prescribed 7 repetitions, 128 generated tokens, contexts 512/4096, F16 KV, FA on, 99 GPU layers, and 8 CPU threads:

```sh
LD_LIBRARY_PATH="$PWD/build/bin" python3 benchmark/run.py \
  --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --modes decode --contexts 512 4096 --decode-tokens 128 --repetitions 7 \
  --cooldown-temp-c 60 --binary "$PWD/build/bin/llama-bench" \
  --output results/exp011/rows1_control.json
```

`ldd` with the same `LD_LIBRARY_PATH` resolves `llama-bench` and all `libggml*` / `libllama*` shared libraries from `build/bin`; see `results/exp011/ldd_rows1_control.txt`. The binary path is recorded in the JSON. No copied benchmark binary or alternate library was involved.

## RESULT

**REVERT.** Both exact 2-bit side decoders lost at both tested planar K-block counts, with disjoint sample ranges from the base-3 dot. Since they lose before conversion and model/runtime integration, no side candidate survived to end-to-end benchmarking. The only end-to-end run is the current ROWS=1 control. Production source remains at ROWS=1.

## CORRECTNESS

At 65,536 blocks, all 8,388,608 trits matched the independent host canonical decode and the packed 2-bit codes; both side-dot decoders matched the base-3 CUDA outputs and independent CPU full-block reference bit-for-bit (maximum absolute error 0). At 16,384 blocks, all 2,097,152 trits and outputs matched bit-for-bit (maximum absolute error 0). These tests include `qs` stage boundaries and the final `qh` trits because every element 0–127 in each block is checked.

Compute Sanitizer memcheck on the 16,384-block run reported **0 errors**; exactness output is saved in `results/exp011/sanitizer_16384.txt`. Its event timings are sanitizer-instrumented and excluded from performance results. No production-path correctness suite or model smoke was run for the candidate because no production code/runtime route was added. The current ROWS=1 model itself completed both control decode workloads successfully.

The manager independently rebuilt this harness with `nvcc -O3 -arch=sm_86` and reran the 16,384-block exactness case: all 2,097,152 codes and outputs matched, with maximum absolute error 0. The run is retained in `results/exp011/manager_verify_16384.txt`; its one-launch timings are deliberately not used for performance conclusions.

## MICROBENCHMARK

RTX 3080, sm_86, CUDA 13.2, unsanitized CUDA-event timings, 40 warmups followed by 300 launches per sample, nine alternating-order samples. `results/exp011/planar_65536.txt` and `planar_16384.txt` contain the full samples. Median, mean, sample standard deviation, and range are milliseconds per launch:

| K blocks | Base-3 median (mean ± SD; range) | 2-bit scalar median (mean ± SD; range) | Scalar delta | 2-bit expand median (mean ± SD; range) | Expand delta |
|---:|---:|---:|---:|---:|---:|
| 65,536 | 0.01879040 (0.01873056 ± 0.00008486; 0.01863680–0.01882112) | 0.01935701 (0.01929134 ± 0.00009011; 0.01918133–0.01937749) | +3.02% | 0.01966955 (0.01961091 ± 0.00008548; 0.01948352–0.01968107) | +4.68% |
| 16,384 | 0.00305152 (0.00304973 ± 0.00000434; 0.00304128–0.00305472) | 0.00326539 (0.00326573 ± 0.00000298; 0.00326315–0.00327275) | +7.01% | 0.00314613 (0.00314630 ± 0.00000393; 0.00314027–0.00315339) | +3.10% |

Both side-code ranges are wholly slower in both screens. The 65,536-block case is a more stable, bandwidth-relevant screen: scalar extraction loses 3.02% and packed-byte expansion loses 4.68%. At 16,384 blocks, packed-byte expansion narrows the loss to 3.10% but still trails. `nvidia-smi` recorded 57 C / 210 MHz graphics before the timed runs and 56 C / 2,010 MHz after; whole-GPU use was 173 MiB before/after. These are start/end readings, not a per-sample clock trace.

## END-TO-END IMPACT

The ROWS=1 control completed with a 6,805 MiB whole-GPU peak and a 173 MiB idle baseline. Its sampler observed 50–67 C and 210–2,010 MHz SM clocks over the full load/run/cooldown window; samples at 50%+ utilization reached 1,815–2,010 MHz. At context 512, the seven tok/s samples were 81.4375, 82.3129, 82.3217, 82.2830, 82.3062, 82.2481, 82.1477 (mean 82.1510, median 82.2830, sample SD 0.3203, range 81.4375–82.3217). At context 4096 they were 79.0586, 79.7735, 79.7709, 79.7657, 79.7604, 79.7509, 79.7233 (mean 79.6576, median 79.7604, sample SD 0.2647, range 79.0586–79.7735). The fresh medians agree with the prior 82.22/79.70 measurements within run variation.

No candidate end-to-end run, model smoke, conversion, or runtime-repack peak was measured because the planar inner-dot candidate lost before integration. The GGUF reader reports 5,877,760,000 tensor-data bytes of type 143 (PTQ1_0). Converting those blocks from 28 to 34 bytes would add 1,259,520,000 bytes (1,201.17 MiB) of resident weights and make that payload 7,137,280,000 bytes. If the side blocks replaced the GPU-resident PTQ1_0 blocks and other allocations stayed constant, the measured 6,805 MiB peak projects to about 8,006 MiB, under the 10,240 MiB device limit. That is an estimate, not a measured runtime peak. Simultaneously holding the original 5.88 GB PTQ1_0 payload and its expanded GPU copy would exceed the device limit; streaming or pre-repacking would be needed to avoid that transient.

## ANALYSIS

The active planar address mapping alone does not rescue 2-bit codes: both side decoders are slower at 16K and 65K K blocks despite excluding conversion. The direct decode wins no time to pay for its 6 extra bytes per block. A model loader, repack path, or end-to-end test would add work and memory pressure without a supporting kernel-level gain, so none was implemented.

The microbenchmark captures the active block-dot addressing and base-3 versus packed extraction, but it is not a full launch/CTA reduction benchmark; the active kernel's partial-write and shared-memory reduction overhead is outside the comparison. This limits its absolute timing as a predictor of token throughput. It does not reverse the decision: the candidate loses in the isolated decode portion that was intended to improve.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** Keep the current ROWS=1 production source. No implementation changes need reverting. The side representation is rejected for this active RTX 3080 planar decode path.

## FOLLOW-UPS

- Do not integrate this 2-bit side block or add loader conversion for it.
- Continue with a distinct planar trit-unpack/reduction mapping if PTQ1_0 GEMV remains the optimization target.

## IMPORTANT DISCOVERIES

- The active planar-transposed activation mapping can be reproduced as eight 16-byte Q8 planes plus a ninth 16-byte scale/isum plane per K block; all exact block dots passed under that mapping.
- At 65,536 blocks, scalar extraction loses 3.02% and packed-byte expansion loses 4.68%; at 16,384 blocks the losses are 7.01% and 3.10%, before accounting for conversion.
- The measured PTQ1_0 tensor payload is 5.878 GB; 34-byte blocks would add 1,201 MiB to the resident model. Replacing weights likely fits the recorded peak estimate, but retaining both source and expanded GPU copies at once does not.
