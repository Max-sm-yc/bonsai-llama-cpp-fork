# Experiment 031: padded PTQ1_0 blocks and vector loads

## HYPOTHESIS

The active RTX 3080 plain PTQ1_0 GEMV reads six 32-bit words from each 28-byte block. Padding each block to 32 bytes and issuing two aligned 16-byte loads could reduce global load instruction count while preserving the exact base-3 payload and default caching. Expected resident cost is +4/28 (14.29%) of PTQ1_0 data, about 801 MiB against the recorded 6,805 MiB model peak.

## IMPLEMENTATION

Read the current production source and verified it is `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`, SHA-256 `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`. The active block is 24 `qs` bytes, two `qh` bytes, and a half scale (28 bytes). The active GEMV's planar activation layout is the existing nine-plane layout documented in that file. The standalone CUDA harness is `results/exp031/padded_loads.cu`; it compares the existing 28-byte block-dot with the same block-dot fed from a 32-byte padded representation via two `int4` loads. The base-3 decoder, activation plane addressing, DP4A grouping, scale/isum correction, and output accumulation order are held fixed. Every generated block is independently checked against host-decoded trits and host full-block dots.

The harness uses one CUDA thread per PTQ1_0 block, 128-thread CTAs, and the active planar Q8 layout. It times the block-dot work, not the production kernel's shared-memory partial array and final row reduction. This is an early standalone screen; because it lost at both sizes, no runtime integration was attempted. Production source and library were not modified, and no broad build was run.

The raw harness also compiled the previously used 34-byte 2-bit side-format arms. Its generic `EXACT` summary line prints `payload=28->34` for that side format; this field does not describe the padded arm. The padded structure is independently declared and compile-time checked as 32 bytes (`static_assert(sizeof(BaseBlock)==28 && sizeof(PaddedBlock)==32)`), and its upload construction copies exactly 28 payload bytes plus four padding bytes.

## RESULT

**REVERT.** The padded vector-load block-dot was slower at both tested sizes: +3.37% at 16,384 K blocks and +21.39% at 65,536 K blocks. This clearly fails the requested early screen, so upload repacking, full-model correctness, and end-to-end runs were correctly skipped. The experiment therefore has no measured candidate VRAM peak or setup/repacking cost.

## CORRECTNESS

The harness verified every trit, scale/isum correction, and per-block output against an independent host reference. At 16,384 blocks, 2,097,152 trits and 16,384 outputs matched, with zero mismatches and zero maximum error. At 65,536 blocks, 8,388,608 trits and 65,536 outputs matched, also with zero mismatches and zero maximum error. Blocks are individually randomized and checked across the full 128-element decode, including qs/qh stream boundaries and adjacent-block transitions. Compute Sanitizer memcheck at 16,384 blocks reported zero errors. See `results/exp031/screen_16384.txt`, `screen_65536.txt`, and `memcheck_16384.txt`.

These are standalone block-dot correctness results, not full model output equivalence. No candidate runtime was integrated.

## MICROBENCHMARK

RTX 3080, sm_86, CUDA event timings; nine alternating-order samples per arm. 16K used 200 launches/sample and 65K used 100 launches/sample. Medians in milliseconds per block-dot launch:

| K blocks | 28-byte base-3 | 32-byte padded/vector | Padded delta |
|---:|---:|---:|---:|
| 16,384 | 0.00338224 | 0.00349632 | +3.37% |
| 65,536 | 0.01881088 | 0.02283488 | +21.39% |

All nine raw samples are in the screen text files. The full harness was compiled with `nvcc -O3 -arch=sm_86`. ptxas reports 40 registers/thread, zero stack frame, and zero spills for each dot kernel (`results/exp031/resources.txt`). SASS for `dot_padded` confirms two `LDG.E.128` global loads at offsets 0 and 16 (`results/exp031/padded_loads.sass`). Both are default-cache loads. This confirms the intended instructions emitted, but the second load fetches the four padding bytes along with payload and the candidate moves 14.29% more bytes per PTQ1 block.

## END-TO-END IMPACT

Not measured because the standalone screen lost. The preserved source-default library is SHA-256 `c828135b126ec507ffbecb4dc11b6a7a9ac5cd0fe050553323d7f35c38fae6c7`; `/tmp/bonsai2-exp030-baseline-libggml-cuda.so.0` has the same hash. No candidate upload, full-model output comparison, or decode A/B occurred.

The +4/28 payload increase applied to the recorded 5,877,760,000 PTQ1_0 tensor bytes is 839,680,000 bytes (800.78 MiB); holding other allocations fixed would project 6,805 MiB to about 7,606 MiB. This is a projection, not a measured runtime peak. Current post-test GPU use was 173 MiB idle; no candidate model memory use was measured.

## ANALYSIS

The vector loads did reach SASS and did not increase register or spill resources in the isolated kernel, but reducing instruction count did not offset the extra traffic and padding. The loss widened substantially at 65K blocks, the more bandwidth-relevant screen. Since the early screen excludes production reduction overhead and already loses clearly, there is no evidence to justify the extra 801 MiB resident footprint or upload repacking complexity.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** No production changes were made. Production source remains SHA-256 `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`; active source-default library and protected backup both remain SHA-256 `c828135b126ec507ffbecb4dc11b6a7a9ac5cd0fe050553323d7f35c38fae6c7`. Repository HEAD remains `2287b5006899da645ed46fc77d93c9263cd64df6`; no commit was made.

## FOLLOW-UPS

Do not integrate 32-byte PTQ1_0 blocks or loader repacking on this path based on this design. Revisit only with new evidence that changes the memory-traffic tradeoff or a different measured staging design.

## IMPORTANT DISCOVERIES

- Two aligned 16-byte loads emitted as intended and retained default cache semantics.
- Exact block-dot work lost 3.37% at 16K K blocks and 21.39% at 65K K blocks.
- The representation costs 14.29% additional PTQ1_0 payload; estimated peak remains under 10 GiB, but this was not measured because the candidate was rejected before integration.
