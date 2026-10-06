# Experiment 004: PTQ1_0 base-3 `qs` decoder

## HYPOTHESIS

PTQ1_0 batch-1 GEMV's repeated multiply-by-three sequence might be replaced with a small exact byte-to-five-trits LUT and reduce decode instruction cost on sm_86.

## IMPLEMENTATION

No production source or build changes were made. A standalone CUDA microbenchmark in `results/exp004/decoder_bench.cu` compares the existing packed-lane multiply/`__byte_perm` expansion against a constant-memory 256-by-5 LUT for the six groups of four `qs` bytes (120 positions) in a PTQ1_0 block. Both feed the same DP4A and signed-trit bias correction and use 32-byte packed-block stride and Q8 activation rows. `qh` handling is excluded so the timing isolates the `qs` decoder. The generated weights use valid `qs` byte values 0–242; activations are deterministic signed Q8 values.

Build and run:

```bash
nvcc -O3 -arch=sm_86 results/exp004/decoder_bench.cu -o results/exp004/decoder_bench
./results/exp004/decoder_bench 65536 100
```

The first draft incorrectly treated the encoded byte as an ordinary base-3 integer; it mismatched and was discarded. The correct digit recurrence is `w = x * 3`, digit `w >> 8`, next state `w & 255`, matching the production decoder's repeated multiply and byte extraction.

## RESULT

The LUT candidate is substantially slower in the focused CUDA timing. It was not integrated; no full-model evaluation was warranted.

## CORRECTNESS

For 65,536 deterministically generated blocks, the baseline and LUT variants produced identical `qs` dot outputs (0 mismatches). Each of the 120 `qs` positions is consumed in order through the same Q8 activation positions and DP4A accumulation; the same exact per-word activation sum is subtracted to convert raw digits 0/1/2 to signed trits -1/0/1. The harness does not validate `qh` positions, model outputs, or exhaustive coverage of all 243 valid byte codes. No PTQ1_0 packed-position or production-kernel correctness suite was run because the candidate did not pass the performance screen.

## MICROBENCHMARK

RTX 3080, sm_86, CUDA 13.2, `n=65536`, 100 CUDA-event timed launches after warmup:

| Decoder | ms per launch |
|---|---:|
| Existing multiply/byte-permute | 0.022250 |
| Constant-memory LUT | 0.130949 |

The LUT path took 5.89x as long in this harness. Both variants use the same data, Q8 layout, dot/bias correction, and output store. This is a focused `qs` microbenchmark, not a complete reproduction of the production `vec_dot_ptq1_0_q8_1_multi` function or its CTA-level workload.

Raw terminal output is retained in `results/exp004/microbenchmark.txt`.

## END-TO-END IMPACT

Not run. The candidate had no kernel-level advantage, so it was not integrated or evaluated with llama-bench.

## ANALYSIS

The current decoder extracts four trits in parallel from widened byte lanes using multiply-by-three and byte permutations. The LUT adds five dependent constant-memory lookups per source byte, which outweighed the multiply sequence in the measured implementation. The result rejects this direct LUT variant; it does not show whether a bit-sliced decoder or a more specialized table arrangement could help.

## DECISION

**REVERT / do not integrate.** There were no production edits. The verified PTQ1_0 baseline remains intact.

## FOLLOW-UPS

- Only pursue a different exact decoder if its packed-position and full-block correctness can be established independently and it first shows a clear focused-kernel advantage.
- If revisiting, benchmark the full `vec_dot_ptq1_0_q8_1_multi` function, including `qh` and the real warp-transposed activation layout.

## IMPORTANT DISCOVERIES

- PTQ1_0's encoded `qs` byte is a fixed-point base-3 stream: trits are emitted from the high byte after repeated multiplication by three. Ordinary `% 3`/`/ 3` decomposition is incorrect.
- The tested constant-memory LUT is nearly six times slower than the multiply/byte-permute path in this focused screen.

## FINAL SOURCE/BUILD STATE

`ggml/src/ggml-cuda/vecdotq.cuh` and all tracked source remain unchanged; `git status --short` shows only the new experiment report and standalone benchmark source/evidence. The existing `build/` tree was not rebuilt or modified. No commit was made.
