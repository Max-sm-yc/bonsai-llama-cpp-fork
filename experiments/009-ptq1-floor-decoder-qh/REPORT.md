# Experiment 009: PTQ1_0 parallel floor-difference decoder with qh interleave

## HYPOTHESIS

The floor-difference digit identity from experiment 008 may reduce PTQ1_0 decode dependency depth if each byte's digits are calculated independently. The prior block mismatch was caused by interpreting the two `qh` streams as four streams. Correctly assembling the `qh` byte lanes could make the candidate both exact and faster than the production base-3 recurrence.

## IMPLEMENTATION

The isolated harness is [parallel_bench.cu](../../results/exp009/parallel_bench.cu). It retains the 28-byte PTQ1_0 block, full `qs` and `qh`, four DP4A/isum/scale groups, and exact Q8 activation correction. The corrected `qh` helper duplicates the two source streams into four packed lanes, computes adjacent digit pairs, then interleaves them as `[qh0[t], qh1[t], qh0[t+1], qh1[t+1]]`.

Before the full block path or timing, the harness runs device exhaustive gates: 256 input bytes × five direct digits, and all 65,536 `qh0/qh1` pairs × all four digits (eight interleaved outputs per pair). Full blocks are checked against the recurrence decoder and an independent host dot reference. Activation addresses use production SOA_ISUM mapping: `group=b>>5`, `lane=b&31`, `word=e>>2`, byte `e&3`, with a 32*36-word group stride.

Built with `nvcc -O3 -arch=sm_86`. Production source and the build tree were not modified. The original experiment-008 source was restored after copying the corrected harness into results/exp009. The final binary was rebuilt from `results/exp009/parallel_bench.cu`.

## RESULT

The correction passes all exhaustive and full-block correctness checks. The floor-difference implementation is slower in all three measured block-count distributions: 7.04% at 1,024 blocks, 4.11% at 16,384, and 1.24% at 65,536 blocks. No runtime integration or model benchmark was performed.

## CORRECTNESS

- Device direct digit gate: 1,280 outputs (all 256 byte inputs × five positions), zero mismatches versus the production recurrence.
- Device qh gate: 524,288 outputs (65,536 byte pairs × four digits × two streams), zero mismatches. This validates the required two-stream interleave.
- Full blocks at 128, 1,024, 16,384, and 65,536 blocks: zero code mismatches, zero candidate-vs-production output mismatches, and zero production-vs-independent-host-reference mismatches. Outputs matched exactly (`max_abs_error=0`).
- Compute Sanitizer memcheck at both 128 and 65,536 blocks: `ERROR SUMMARY: 0 errors`; raw logs are `results/exp009/compute_sanitizer_128.txt` and `compute_sanitizer_65536.txt`.

Raw device/timing output is in `results/exp009/bench_*.txt`; sanitizer output is in `results/exp009/compute_sanitizer_*.txt`.

## MICROBENCHMARK

RTX 3080, CUDA event time per launch, seven paired samples per variant, 300 launches per sample, warm-up before each timed sample. The order alternated by pair. The gate and full-block validation ran before measurements on every process invocation.

| Blocks | Production base-3 median (ms) | Floor-difference median (ms) | Candidate change | Candidate sample range (ms) |
|---:|---:|---:|---:|---:|
| 1,024 | 0.0022959 | 0.0024576 | +7.04% slower | 0.0024542–0.0024610 |
| 16,384 | 0.0031606 | 0.0032905 | +4.11% slower | 0.0032894–0.0032931 |
| 65,536 | 0.0186231 | 0.0188539 | +1.24% slower | 0.0188387–0.0189030 |

Production and candidate sample values are retained verbatim in `results/exp009/bench_1024.txt`, `bench_16384.txt`, and `bench_65536.txt`. GPU telemetry snapshots around the measurement series are in `results/exp009/telemetry.txt`: temperature was 46–49 C; SM clocks were 1,845–1,995 MHz under active runs (idle P8 snapshots showed 210 MHz); memory clocks were 9,501 MHz under load (405 MHz idle). Measurements were serialized; no other compute process was present.

## END-TO-END IMPACT

Not run. The candidate loses the focused full-block comparison at every representative size, so integration and model runs are not warranted. No GGUF, runtime source, quality, context, decode length, VRAM limit, or build configuration was changed.

## ANALYSIS

The original `qh` bug is resolved: it was a stream-layout error, not a failure of the floor-difference identity. Both isolated exhaustive device gates and the production-layout DP4A block harness now establish exactness. However, independent scaled-floor products and extraction do not beat the production multiply-by-three recurrence in this harness. The penalty shrinks at larger block counts but remains consistent and positive at 65,536 blocks; there is no repeatable gain to justify runtime complexity.

## DECISION

**REJECT; do not integrate.** Keep the production base-3 PTQ1_0 decoder and project baseline unchanged.

## FOLLOW-UPS

No follow-up is needed for this decoder variant. Any future decoder experiment should preserve the verified `qh` pair interleave and SOA_ISUM mapping and pass exhaustive device checks before timing.

## IMPORTANT DISCOVERIES

- The two `qh` bytes are two independent streams interleaved at each digit position; packing them as two separated bytes leaves two lanes zero in a four-byte decoder helper.
- Correct qh packing is `[qh0[t], qh1[t], qh0[t+1], qh1[t+1]]`; exhaustive device coverage of every qh byte pair and digit is practical and passed.
- The floor-difference identity is device-exact across all source bytes and positions, but is slower than production base-3 arithmetic by 1.24–7.04% in the tested full-block workload.

## MANAGER AUDIT (post-exp009 dispatch review)

This harness uses SOA_ISUM activation addressing. The target RTX 3080 / sm_86 uses planar-transposed `GGML_CUDA_Q8_1_PT` activations and the dedicated `mul_mat_vec_ptq1_0_pt` path for batch-1 PTQ1_0. The reported exactness and slowdown are valid for the isolated SOA harness, but do not establish the candidate's performance in the active target kernel.
