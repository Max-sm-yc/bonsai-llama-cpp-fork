# Experiment 044: CTA-local activation tile reuse

## HYPOTHESIS

The active sm_86 batch-1 PTQ1_0 kernel assigns a 16-row output tile to each 128-thread CTA. Every row/K-block item reads the same nine planar Q8_1 activation vectors. Staging those vectors once per CTA could reduce repeated global-load issue while retaining the 128-thread, ROWS=1 packed weight/dot schedule.

## IMPLEMENTATION

The source mapping was inspected in `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`: planes 0–7 and the scale plane each have `nblk * 16` bytes, with one `int4` per plane and K block. The active dot helper issues nine `int4` loads per activation column/K-block item (eight quant planes plus the scale/sum plane). Exp026 independently confirmed nine `LDG.E.128` activation loads in active specializations.

An isolated CUDA-event screen was built from manager HEAD `99f968c00eb6706d1e9ca6e1ddcf45ffbfdbf19c` in `/tmp/exp044-activation-tile-reuse`. The candidate cooperatively copies the complete planar activation tile into CTA shared memory, synchronizes once, then runs the same ROWS=1 packed decoder, DP4A/FMA accumulation, per-block partial writes, and four-accumulator row fold as the direct-global control. CTA width is 128 and row tile is 16 for both shapes. The direct-global control and staged candidate were alternated in nine sample pairs; each sample averaged 100 work-plus-fold launches at K=40 and 80 at K=136, over 2,048 rows.

The full activation tile is 5,760 B at K=40 and 19,584 B at K=136. Combined with production partial scratch (2,624 B / 8,832 B), estimated plain-path shared use is 8,384 B / 28,416 B per CTA. The K=136 total permits at most three CTAs per SM by shared-memory capacity on this GPU. The screen itself stages only the activation tile and writes partials globally, so its dynamic shared allocations are 5,760 B / 19,584 B. It is a focused work-plus-fold screen, not a runtime integration.

Artifacts: [`activation_tile_screen.cu`](../../results/exp044/activation_tile_screen.cu), raw samples [`screen_40.txt`](../../results/exp044/screen_40.txt) and [`screen_136.txt`](../../results/exp044/screen_136.txt), ptxas output [`resources.txt`](../../results/exp044/resources.txt), resource dump [`resource_usage.txt`](../../results/exp044/resource_usage.txt), and SASS [`activation_tile_screen.sass`](../../results/exp044/activation_tile_screen.sass). Build command:

```bash
nvcc -O3 -arch=sm_86 -Xptxas=-v results/exp044/activation_tile_screen.cu \
  -o results/exp044/activation_tile_screen 2> results/exp044/resources.txt
```

## RESULT

**REVERT.** The shared-activation candidate was slower at both tested shapes. It did not qualify for runtime integration or model A/B.

| K blocks | Direct-global median (range), µs | Shared activation median (range), µs | Candidate delta |
|---:|---:|---:|---:|
| 40 | 9.97088 (9.93696–9.98144) | 10.17856 (10.10688–10.19456) | +2.08% |
| 136 | 27.86560 (27.81440–27.99360) | 28.21920 (28.14720–28.40280) | +1.27% |

## CORRECTNESS

An independent host PTQ1_0 decoder and full-row output reference were used. At both K shapes, all 2,048 candidate row outputs matched the host reference bit-for-bit (zero mismatches; max absolute error 0). Direct-global control outputs also matched exactly. No tolerance-based discrepancy occurred.

## RESOURCES AND SASS

Ptxas reported 40 registers/thread, zero stack, zero spills, one barrier for `work_shared`, and zero barriers for direct `work`. The dynamic activation allocation is 5.76/19.58 KiB for K=40/136. SASS confirms initial global tile loads, shared stores, one CTA barrier, and shared reads for candidate activation access. The active direct source has nine 128-bit activation-vector reads per work item; repeating those reads remains generated in the screen's direct-global version. The tile copy removes row-repeated global issue, but its initial copy, shared traffic, barrier, and added indexing cost exceed that saving in both measured shapes. Production's existing partial array increases shared demand to an estimated 8.38/28.42 KiB, limiting K=136 to three resident CTAs per SM by shared capacity.

## END-TO-END IMPACT

Not measured. The candidate lost the required focused screen at both shapes, so no runtime build, fixed-seed model smoke, correctness suite, or decode A/B was run.

## ANALYSIS

This result tests activation reuse directly, distinct from Exp037's shared weight staging. Reusing the activation tile does eliminate repeated global accesses across rows in the screen, but it adds a CTA barrier and shared stores/loads on every tile. The loss is repeatable and the long-K shape also increases shared occupancy pressure. Do not integrate this full-tile design. Any follow-up would need to reduce copy/communication overhead materially while preserving exact planar indexing and the established dot schedule.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** Keep the existing 128-thread, ROWS=1 production path. No production source, build, or binary was changed.

## FOLLOW-UPS

Do not rerun this full-tile staging design. Revisit activation reuse only with a materially different mechanism that removes the extra CTA barrier/copy cost and demonstrates a repeatable gain at both K shapes before runtime integration.

## IMPORTANT DISCOVERIES

- Full planar activation staging is resource-feasible in the plain path, but estimated production shared use reaches 28.4 KiB at K=136 and caps residency at three CTAs/SM.
- Candidate outputs were exactly equal to independent host-decoder outputs for all tested rows.
- Despite removing repeated global activation loads across the 16-row tile, the candidate lost 2.08% at K=40 and 1.27% at K=136 in work-plus-fold timing.

## HASHES AND ISOLATION

- Manager HEAD at experiment start: `99f968c00eb6706d1e9ca6e1ddcf45ffbfdbf19c`.
- Production source SHA-256: `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`.
- Active production CUDA library SHA-256 recorded at assignment: `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`.
- Isolated screen source SHA-256: `f9f807ea8f95f01076c266cb5e2614318dd6f5db14c76d9794ebc4c96382cbbc`.
- Screen binary SHA-256: `3814910f8aa0fa0a6b55512193da7436cd8a1524b1f2fc822450d689592996c2`.
- Production checkout/build remained untouched; all candidate source and binary artifacts are in the detached `/tmp/exp044-activation-tile-reuse` worktree. No commit was made.
