# Experiment 012: pairwise radix-3 trit decode on planar PTQ1_0

## HYPOTHESIS

Replace two serial byte-remainder steps (`x *= 3`) with one packed-lane `x *= 9`; split `floor(9*x/256)` into the ordered trit pair `q/3, q%3`. The intended benefit was a shorter per-byte dependency chain in the active planar Q8_1 dot.

## IMPLEMENTATION

`results/exp012/pairwise_bench.cu` is an isolated full-block harness based on experiment 011. It keeps 28-byte PTQ1_0 blocks, all `qs` and `qh` bytes, four DP4A/isum/scale groups, and the 9-plane planar activation address `((plane * nblocks + block) * 4 + slot)`. The candidate pair-decodes both qs streams and leaves qh on the production recurrence/interleave.

The first candidate draft incorrectly right-shifted already-computed activation word indices for the first qs stream. `results/exp012/pairwise_qs_isolation.cu` checks every group and pair position, emitted pair vectors, and post-pair remainder against two sequential recurrence steps. Across 257 random blocks it verifies 18,504 pair transitions / 37,008 emitted vectors with zero mismatches. It also verifies the stream A word/bucket formulas `word=4*digit+g`, `bucket=word>>3`, and stream B formulas `word=20+2*digit+g`, `bucket=(80+8*digit+4*g)>>5`. The diagnostics exposed the redundant `>>2` in both pair and final-step word lookups; removing it fixed the full block dot.

`results/exp012/pairwise_gate.cu` also tests all 256 byte values across four distinct byte lanes for a pair transition. It reports zero mismatches. The corrected harness was built using `nvcc -O3 -arch=sm_86`; ptxas reports 40 registers, zero stack, and zero spill loads/stores for both base-3 and pairwise dot kernels (`results/exp012/compile_resources.txt`).

No production source or build library was modified.

## RESULT

**REVERT.** The pairwise candidate is now exact, including all qs/qh positions, but is slower than the production recurrence in every controlled full-block screen. No production integration, correctness-suite run, model smoke, or candidate model benchmark was warranted.

## CORRECTNESS

For 128, 16,384, and 65,536 blocks, the harness checked every trit and full-block output against both the current base-3 device path and independent CPU dot reference. All code and output mismatch counts are zero and maximum absolute error is 0. Boundary counts 1, 127, and 129 also passed. The 128-block Compute Sanitizer memcheck reports `ERROR SUMMARY: 0 errors` (`results/exp012/sanitizer_128.txt`).

The manager independently rebuilt the corrected harness with `nvcc -O3 -arch=sm_86` and reran the 127-block boundary; all 16,256 trits and full-block outputs matched, with zero independent-reference mismatches (`results/exp012/manager_verify_127.txt`). Its one-launch times are not used for the performance decision.

The exactness gates also cover all byte values, every qs digit pair position in both streams, each pair's updated remainder, and activation word/sum bucket placement. `results/exp012/qh_gate.cu` exhausts all 65,536 qh byte pairs and confirms all 131,072 interleaved output vectors against independent host recurrence (zero mismatches). No scale/isum or qh changes were introduced in the candidate.

## MICROBENCHMARK

RTX 3080, sm_86, unsanitized CUDA events, 300 launches per sample, 40 warmups, nine alternating-order samples per variant. Each run performed the full exactness checks before timing. Raw sample distributions are in `results/exp012/pairwise_128.txt`, `pairwise_16384.txt`, and `pairwise_65536.txt`.

| Blocks | Base-3 median (mean; range) ms | Pairwise median (mean; range) ms | Pairwise delta |
|---:|---:|---:|---:|
| 128 | 0.00243285 (0.00243266; 0.00242005–0.00246784) | 0.00303104 (0.00303092; 0.00302763–0.00303445) | +24.59% |
| 16,384 | 0.00336896 (0.00336948; 0.00336555–0.00337237) | 0.00369963 (0.00369988; 0.00369323–0.00371029) | +9.82% |
| 65,536 | 0.01884139 (0.01886177; 0.01882603–0.01894400) | 0.01921269 (0.01922293; 0.01918923–0.01930581) | +1.97% |

The pairwise ranges do not overlap the base-3 ranges at any measured size. Ptxas reports equal 40-register usage and no spills; the expected shorter recurrence did not overcome the quotient split and added instruction work. `nvidia-smi` snapshot after measurements: 52 C, 270 MHz SM / 5001 MHz memory (idle-transition snapshot), 89 W.

## END-TO-END IMPACT

Not run. The candidate loses the isolated full-block decode at all tested sizes, including the largest, so it did not qualify for integration. The active ROWS=1 path remains the current best.

## ANALYSIS

The pairwise radix-3 identity and packed-lane remainder update are correct. The initial full-dot mismatch was an implementation indexing error: the first stream had already converted element positions to activation word indices, then shifted those indices again before calling the planar accessor. Independent per-pair and per-stream gates made this error explicit; correcting the redundant shift produced exact full-block outputs.

Even exact pairwise decoding is slower. The 65,536-block penalty is smaller than at medium sizes but remains consistent and larger than sample noise. The packed `x*9` step saves one recurrence stage per pair, but splitting the quotient into two trits costs extra integer operations. No benefit supports production complexity or end-to-end testing.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** Do not integrate. Preserve the production ROWS=1 decoder unchanged.

## FOLLOW-UPS

No further pairwise decoder work is recommended for the active planar path. Any different decoder should first pass the per-position stream mapping gate, then beat the 65,536-block full-dot control before model integration.

## IMPORTANT DISCOVERIES

- The pairwise identity is exact for all 256 byte values; repeated packed-lane advancement and both qs stream layouts pass direct device/CPU gates.
- The initial mismatch came from shifting an already-converted first-stream activation word index twice. Independent sum diagnostics had used this flawed candidate and were superseded by the corrected exact gates.
- Correct pairwise decode still loses 1.97% at 65,536 blocks and 9.82% at 16,384 blocks, with no spills and the same 40-register footprint as the production recurrence.
