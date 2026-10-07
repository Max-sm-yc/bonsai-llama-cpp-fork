# Experiment 081: direct 2-bit PTQ1_0 in active planar GEMV

## HYPOTHESIS

The 34-byte direct 2-bit representation (32 code bytes plus the existing 2-byte scale) may remove enough base-3 recurrence work to offset its 21.43% block expansion in the RTX 3080's active planar-transposed, batch-1 GEMV kernel. Exp007's 3.4–5.0% slowdown used SOA_ISUM and did not answer this path. The question here was whether direct codes win under the active `<ncols=1, ROWS=1>` mapping.

## IMPLEMENTATION

The experiment ran in the isolated worktree `/home/maxsun/autonomous_projects/.worktrees/exp081-ptq1-direct-2bit-planar`, based at `74642770549fe3bb53ef876be77082a5214bc242` (`7464277`). Before adding artifacts, the worktree was detached and clean. Initial source hashes are in [`initial_source_hashes.txt`](../../results/exp081/raw/initial_source_hashes.txt). The manager checkout was not modified. Production source files remained unchanged; only `results/exp081/` and this report were added.

The standalone CUDA Graph screen is [`planar_screen.cu`](../../results/exp081/planar_screen.cu). It implements canonical base-3 block encoding and calls the repository's `dequantize_row_ptq1_0` to verify all 128 canonical elements, including the 24-byte `qs` region and 2-byte `qh` tail. The device packer stores four codes per byte; its device unpack checker compared every code to the source and rejects code 3. Random deterministic trit blocks and planar Q8_1 activation data were used in the timed cases.

The screen transcribes the active work/dataflow: nine planar activation planes (`nblk * 16` bytes each), activation scale/sum records in plane 8, `<1,1,false,false>` work, 128-thread CTAs, row/K-block flattened work with `idx += 128`, host-selected `rows_per_cta`, tail-row clamping, one shared FP32 partial per row/K block, and the four modulo-4 FP32 accumulation streams with the production final fold. For K=40 and K=136, the active padding formula yields exactly 40 and 136 blocks, so the screen's `nblk=bpr` has no hidden padding. The Q8 activation address for element `e` in K-block `kb` is plane `e/16`, byte `e%16`, at `((e/16)*nblk+kb)*16+(e%16)`; the scale/sum address is `8*nblk*16+kb*16`. Each stored signed isum is the exact sum of its 32 activation bytes, bitcast into the high half of the corresponding half2, as in production.

Host translation unit and focused executable commands:

```sh
cc -O2 -ffunction-sections -fdata-sections -Iggml/include -Iggml/src \
  -c ggml/src/ggml-quants.c -o results/exp081/raw/ggml-quants.o
nvcc -std=c++17 -O3 -arch=sm_86 --ptxas-options=-v -Xlinker --gc-sections \
  results/exp081/planar_screen.cu results/exp081/raw/ggml-quants.o \
  -o results/exp081/raw/planar_screen 2> results/exp081/raw/ptxas.txt
```

Source and executable hashes, ptxas output, SASS, CUDA compiler version, and loader inspection are recorded under [`results/exp081/raw/`](../../results/exp081/raw/). The focused executable links only standard host runtime libraries; `ldd` and `readelf` show no project CUDA/GGML library dependency and no RPATH/RUNPATH. No candidate runtime library or model executable was built, so the manager's best runtime build/library path was not loaded by this screen.

## RESULT

**The direct 2-bit path loses the active planar screen.** Across all six K/row cases it was 9.68–95.35% slower by median, with zero output mismatches. The observed base and candidate timing ranges were disjoint in every case. No production candidate qualified for model integration.

The active PTQ payload is 5,599,641,600 bytes. It contains 199,987,200 28-byte blocks; changing each to 34 bytes adds 1,199,923,200 bytes (1.1999 GB / 1,144.34 MiB), a 21.43% increase, for 6,799,564,800 bytes of encoded weights.

## CORRECTNESS

- Host canonical packing was checked against `dequantize_row_ptq1_0` for all 128 positions. The check includes `qs` elements 0–119 and interleaved `qh` elements 120–127.
- The CUDA side pack/unpack check compared every packed code for all generated blocks with the input 0/1/2 codes; no invalid code 3 occurred.
- Baseline and direct CUDA outputs matched bit-for-bit for every output at K=40/136 and rows=257/1025/4099. Both also matched a CPU reference built from `dequantize_row_ptq1_0`, the generated Q8 bytes, and the production four-stream FP32 fold.
- `compute-sanitizer --tool memcheck` ran the six shape/tail checks with **0 errors**. Full `tests/run_correctness.sh` and model smoke were not run because no production source or runtime candidate was integrated.

The exact sanitizer command was:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 \
  results/exp081/raw/planar_screen 1 --check-only
```

## MICROBENCHMARK

Hardware was an NVIDIA GeForce RTX 3080, sm_86; driver 580.178.04; `nvcc` 13.2.86. Each variant had its own one-kernel CUDA Graph. Twenty event samples per variant were taken in alternating order; each event covered 300 graph replays. Repeated replay of the same allocations provided warm-cache measurements. The kernel used deterministic Q8_1 planar activations and randomized valid PTQ1_0 blocks. The largest K=40 case places baseline weights at 4.59 MB and side weights at 5.57 MB, straddling the reported 5 MiB L2 capacity; K=136/4099 rows exceeds L2 for both formats.

The exact command was:

```sh
results/exp081/raw/planar_screen 300
```

All 20 samples for each arm and case are retained in [`planar_screen_samples.txt`](../../results/exp081/raw/planar_screen_samples.txt). Medians and full observed ranges, in microseconds per kernel replay:

| K blocks | Output rows | Base-3 median [range] | Direct 2-bit median [range] | Direct delta |
|---:|---:|---:|---:|---:|
| 40 | 257 | 5.560 [5.553, 5.573] | 6.146 [6.141, 6.192] | +10.54% |
| 40 | 1,025 | 6.532 [6.018, 6.537] | 7.165 [6.605, 7.188] | +9.68% |
| 40 | 4,099 | 8.584 [8.424, 8.669] | 10.571 [10.338, 10.710] | +23.15% |
| 136 | 257 | 20.241 [20.234, 20.306] | 26.310 [26.296, 26.469] | +29.98% |
| 136 | 1,025 | 22.042 [22.019, 22.064] | 27.957 [27.938, 28.002] | +26.84% |
| 136 | 4,099 | 37.615 [37.082, 37.874] | 73.482 [71.621, 74.113] | +95.35% |

`ptxas` reports 40 registers/thread, zero stack and spills, and one barrier for both active-like kernel specializations. The screen uses dynamic shared memory of `rows_per_cta * (K_blocks+1) * 4` bytes (2,624 bytes at K=40; 8,768 bytes at K=136). Extracted sm_86 SASS sites per kernel: both have 32 `IDP.4A`, 4 `FFMA`, 186 `LDS`, 1 `STS`, and 1 `BAR.SYNC`; the base-3 kernel has 88 `IMAD` and 35 `LDG.E` sites, while the direct decoder has 109 `IMAD` and 42 `LDG.E` sites. These are static instruction sites for the focused harness specialization, not runtime hardware counters. Nsight Compute was not used; its counters remain unavailable under the known `ERR_NVGPUCTRPERM` condition.

## END-TO-END IMPACT

No model A/B was run. The corrected active-planar screen showed a repeatable slowdown in every case, so the candidate did not meet the integration gate. There are no candidate model rates, output smoke comparison, or candidate VRAM measurements. The 1,144.34 MiB figure above is payload arithmetic only; sidecar conversion/loading lifecycle and actual whole-GPU peak were not measured.

## ANALYSIS

This is not the Exp007 SOA_ISUM experiment: the screen follows the production planar-transposed address calculation and the active one-column work/reduction mapping. The exact 2-bit codes reduce the packed-code recurrence, but this per-code extraction did not translate into fewer instructions in generated code: the candidate had more `IMAD` and global-load instruction sites, while DP4A, partial storage, and reduction sites were unchanged. The 21.43% weight expansion adds traffic at the same time. The measured result is therefore a clear rejection of this direct-access implementation on the active sm_86 path. It does not establish that every possible vectorized 2-bit decoder must lose, but no such decoder was proposed or measured here.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT / NO CANDIDATE.** Production sources and the current-best build were not changed. Retain only the experiment report and evidence in this isolated worktree; do not integrate the representation or claim E2E impact.

## FOLLOW-UPS

No integration follow-up is warranted for this candidate. Revisit only with a demonstrably different packed-word load/decode that reduces the observed global-load and integer instruction work, and screen it again in this active planar mapping before any runtime conversion or model A/B.

## IMPORTANT DISCOVERIES

- Canonical ordering is stage-wise: `qs` covers elements 0–79 then 80–119, while `qh` interleaves the final eight elements by half-byte source.
- A direct code layout can be packed and unpacked exactly, but exactness alone did not offset the additional active-path cost.
- The final corrected CUDA Graph comparison measured distinct graphs per variant. Earlier exploratory samples that captured both kernels in the same graph were discarded and are not part of the reported evidence.
