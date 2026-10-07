# Experiment 039: fixed-point trit decoder in active planar PTQ1_0 GEMV

## HYPOTHESIS

For byte `x`, `t_i=floor(3^i*x/256)` permits digit `i` to be formed as `t_i - 3*t_(i-1)`, exposing fixed-point products independent of the production multiply-by-three remainder chain. The unresolved question was whether that arithmetic wins inside the RTX 3080 sm_86 planar-transposed batch-1 GEMV, where decoded vectors immediately feed active DP4A work and the row fold.

## IMPLEMENTATION

Built the standalone screen in `/tmp/exp039-planar-trit` from manager HEAD `e7335070945e80be2ff564b80ee7df353f7c3b84`. The harness is [parallel_screen.cu](../../results/exp039/parallel_screen.cu); exactness gate is [exhaustive_gate.cu](../../results/exp039/exhaustive_gate.cu). No manager production source or build was changed.

The fixed-point path widens the four packed `qs` bytes into two 16-bit-lane words, multiplies original lane values by 3, 9, 27, 81, and 243, computes adjacent-floor differences, then packs the trits into the same DP4A order. `qh` retains the existing production recurrence/interleave. The initial draft put each floor in the lower byte of each lane, incompatible with the recurrence's `0x7531` byte permutation. Shifting the difference words left by eight before that permutation fixed the error.

This compares fixed-point decode and recurrence inside the same one-thread-per-K-block PT-layout work kernel, with the same 9-plane activation addressing, four DP4A/sub-block accumulators, Q8 scale/sum correction, and separate output fold. It is an active-kernel work-plus-fold event screen, not a full runtime kernel replacement. Unlike Exp009, it does not use SOA_ISUM addressing/harness; it exercises planar PT indexing. It is related arithmetic to Exp009's floor-difference identity, with the new premise being active packed-vector scheduling and planar dot costs.

Device: NVIDIA GeForce RTX 3080, Ampere sm_86; driver 580.178.04; CUDA toolkit 13.2.86; `nvcc -O3 -arch=sm_86`. Host: Fedora Linux x86_64, kernel 7.1.8. Screen source SHA-256 `e4dcb0df2086abdd062694a0cf3e8324641a96ab1d0293981bd08b6f5f1c6140`; binary SHA-256 `31d33a65918d5977c9a612c3f13184ec6490867e0d628f608184c1432d228f31`.

## RESULT

**REJECT; do not integrate.** The fixed-point candidate is exact after correcting byte alignment, but it loses at both active PT workload shapes. The sample ranges do not overlap. It did not qualify for model integration.

## CORRECTNESS

The CUDA exhaustive gate reports zero mismatches for all 256 byte inputs across four packed lanes and five digits, plus all 65,536 `qh` byte pairs across eight interleaved outputs. The whole-block/row screen compares every emitted device trit and row result against the recurrence and independent host reference:

- 40 K blocks/row, 4,096 rows: 20,971,520 device code positions, zero code mismatches; zero recurrence-vs-candidate row mismatches; zero independent-host-output mismatches; maximum error 0.
- 136 K blocks/row, 2,048 rows: 35,651,584 device code positions, zero code mismatches; zero recurrence-vs-candidate row mismatches; zero independent-host-output mismatches; maximum error 0.

Artifacts: `results/exp039/exhaustive_gate.txt`, `screen_40.txt`, and `screen_136.txt`.

## MICROBENCHMARK

CUDA events measure work kernel plus output fold, nine rotated-order samples per variant. 40-block tests use 4,096 rows and 100 repetitions/sample; 136-block tests use 2,048 rows and 60 repetitions/sample. Times are milliseconds per work-plus-fold pair.

| Blocks/row | Recurrence median (range) | Fixed-point median (range) | Candidate delta |
|---:|---:|---:|---:|
| 40 | 0.01268576 (0.01267584–0.01281024) | 0.01301856 (0.01300480–0.01319936) | +2.62% slower |
| 136 | 0.02512213 (0.02508800–0.02522933) | 0.02619733 (0.02613973–0.02626453) | +4.28% slower |

Raw samples are in `results/exp039/screen_40.txt` and `screen_136.txt`. Candidate ranges are wholly above controls at both sizes.

Ptxas reports 40 registers/thread, zero stack/local storage, zero spills, zero shared memory, and zero barriers for both work kernels (`resources.txt`). SASS inspection shows 490 instructions in `work_parallel` versus 338 in `work_aos`; candidate counts include 173 vs 96 IMAD-family instructions and 68 vs 8 SHF instructions, with 45 PRMT in each. The candidate adds product, floor extraction, and difference work while preserving the same register count. The recurrence's multiply-by-three chain is serial, but floor-difference emits more total work; the event regression is consistent with that code structure. Full SASS is retained in `parallel_screen.sass`.

## END-TO-END IMPACT

Not run. The exact candidate loses at both focused shapes, so it was not integrated into the isolated runtime build or benchmarked against the model. No production result is claimed.

## ANALYSIS

This closes the specific active planar PT question: changing from recurrence to independent fixed-point products did not improve the planar-layout block-dot screen. Exp009's SOA_ISUM slowdown did not decide active-path performance; this experiment tested that missing path and found a focused slowdown. The byte-lane fix was essential: floor values occupy low bytes after shifting, while the reused recurrence permutation expects upper-byte positions. Exhaustive byte and qh gates cover this alignment and stream ordering.

The harness preserves active PT activation-plane mapping and DP4A/fold arithmetic, but separates per-block work and row fold into launches for event measurement. It does not measure runtime dispatch, the complete production CTA scheduling, or E2E throughput; none is needed to reject a candidate whose ranges are consistently slower here.

## DECISION

Reject the fixed-point decoder. Keep production recurrence unchanged. Do not run model integration, broad correctness, or end-to-end A/B for this candidate.

## FOLLOW-UPS

No follow-up for this fixed-point variant. Revisit trit arithmetic only with a materially different codegen premise that reduces both dependency depth and instruction count in the active planar work kernel.

## IMPORTANT DISCOVERIES

- The fixed-point identity is device-exact across all byte inputs/packed lanes; qh stream interleave is also exhaustively exact.
- Correctly aligning floor-difference byte lanes before the production permutation resolves the initial full-block mismatch.
- On sm_86, the candidate has the same 40-register/no-spill footprint but emits substantially more integer and shift instructions, and loses 2.62% at 40 blocks and 4.28% at 136 blocks in the active planar work-plus-fold screen.
- Exp009 tested related arithmetic but did not test this planar PT work mapping; Exp039 supplies that missing active-layout evidence.
