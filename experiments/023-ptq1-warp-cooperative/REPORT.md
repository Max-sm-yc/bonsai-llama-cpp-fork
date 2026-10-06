# Experiment 023: warp-cooperative PTQ1_0 block mapping

## HYPOTHESIS

The active ROWS=1 GEMV gives each thread one full 128-weight block and serially performs the trit decode and 32 DP4A operations. A warp-cooperative block mapping might shorten this per-thread chain by assigning lanes to the four 32-weight Q8_1 sub-blocks, while keeping each activation scale and exact integer `isum` correction local.

## IMPLEMENTATION

The production implementation was not changed. Current inspection confirms `ptq1_0_pt_block_dot` remains thread-serial and the batch-1 dispatch in `mul_mat_vec_ptq1_0_pt` instantiates ROWS=1. PTQ1_0 planar activation storage remains unchanged.

I screened a four-lane-per-block mapping in [coop_screen.cu](../../results/exp023/coop_screen.cu). Four lanes own the four contiguous 32-weight sub-blocks; each lane issues eight DP4A operations, applies that sub-block's exact stored activation-sum correction and scale, then a width-four shuffle reduction combines the results. Eight block groups occupy each warp, and a 128-thread CTA processes 32 blocks. The screen uses independent scalar per-element ternary extraction so that the lane mapping can be checked without editing the production decoder. This is intentionally a mapping prototype, not a performance-faithful port of the production base-3 recurrence.

The existing ROWS=1 control was run first with the canonical decode command:

```sh
python3 benchmark/run.py --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --modes decode --contexts 512 4096 --decode-tokens 128 --repetitions 7 --cooldown-temp-c 60 --output results/exp023/control.json
```

The screen was built and run as:

```sh
nvcc -O3 -arch=sm_86 results/exp023/coop_screen.cu -o results/exp023/coop_screen
results/exp023/coop_screen 16384 200
```

## RESULT

**INCONCLUSIVE / DO NOT PROMOTE.** The mapping prototype is exact against its scalar-decoder reference, but it is 3.82x slower in the focused screen. Since both screen kernels use the deliberately simple scalar code extractor rather than the active production recurrence, this result rejects the current prototype only; it does not isolate the benefit of cooperative execution with a production-equivalent decoder. No production integration or candidate end-to-end run was warranted from this screen.

## CORRECTNESS

The 16,384-block focused run reported zero bitwise output mismatches between the one-thread and cooperative kernels, with maximum absolute error 0. Both use the same independent scalar extraction for ternary values and preserve per-sub-block `isum` correction/scaling in the same accumulation order. This verifies the prototype's lane/block indexing and reduction, not the active kernel's decoder or full fused epilogues.

No production candidate existed, so CUDA-vs-CPU backend tests, fixed-seed PTQ1_0/PQ2_0 model smokes, and `tests/run_correctness.sh` were not run for a candidate.

## MICROBENCHMARK

RTX 3080, sm_86, CUDA compiler `nvcc -O3 -arch=sm_86`, 16,384 blocks, 200 launches/sample, nine alternating-order samples measured with CUDA events. Median milliseconds per launch:

| Screen | Median | Samples (ms) |
|---|---:|---|
| One thread per block, scalar-code reference | 0.00348992 | 0.00348448, 0.00348656, 0.00348848, 0.00348960, 0.00348992, 0.00349136, 0.00349168, 0.00349184, 0.00349184 |
| Four lanes per block, scalar-code reference | 0.01331200 | 0.01330688, 0.01331056, 0.01331120, 0.01331200, 0.01331200, 0.01331200, 0.01331248, 0.01331568, 0.01331648 |

The cooperative kernel takes 3.82x as long. Both kernels' decoder is a test-only scalar per-element recurrence, so these timings do not quantify the production decode dependency chain. Nsight Compute counters were not used.

The manager independently rebuilt and reran the same screen on the RTX 3080 at 53°C/0% idle: outputs again matched bitwise, with medians 0.00351088 ms for the scalar screen and 0.01331200 ms for `coop4` (3.79x slower). The rerun is recorded in `results/exp023/manager_recheck.txt`.

## END-TO-END IMPACT

No candidate was integrated or timed end-to-end. The matched same-session ROWS=1 control completed the required 7 repetitions at both contexts. The samples, mean, sample standard deviation, range, and median (tok/s) were:

| Context | Median | Mean ± SD | Range |
|---:|---:|---:|---:|
| 512 | 82.1718 | 82.0668 ± 0.3018 | 81.4023–82.2737 |
| 4096 | 79.7003 | 79.5909 ± 0.2887 | 78.9372–79.7224 |

Peak whole-GPU memory was 6,805 MiB. This is a control only; no before/after candidate delta exists. See [control.json](../../results/exp023/control.json).

## ANALYSIS

The four-lane mapping offers a clean data partition aligned with the four scale/sum records in each activation block, and its tested reduction/indexing is exact. But a naïve lane-local implementation must independently recover the relevant trits and introduces shuffle reduction work. The current screen's very large loss means this version is not a viable direction. It cannot answer whether a cooperative mapping would help when trit unpack is partitioned from the active packed recurrence rather than replaced by scalar extraction.

There is no verified kernel-only win and no canonical end-to-end candidate measurement. Production correctness, fusion behavior, active-library candidate hash, and candidate VRAM were therefore not applicable. The active baseline remains unchanged.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**INCONCLUSIVE.** No production source change was made. Current source SHA-256: `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`. Active CUDA library SHA-256: `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`. `ldd build/bin/llama-bench` resolves `libggml-cuda.so.0` from this workspace's `build/bin`.

## FOLLOW-UPS

Only revisit with a design that partitions the production packed recurrence itself across lanes and has a focused screen that includes block reduction and the same activation layout. Preserve ROWS=1 and the planar-transposed Q8_1 layout until such a candidate demonstrates a material kernel improvement and then improves canonical decode.

## IMPORTANT DISCOVERIES

- Four lanes per block, each owning one exact 32-weight activation sub-block, gives a simple and correct warp reduction mapping.
- The scalar-code prototype loses 3.82x versus its same-decoder serial reference; this is a rejected prototype, not evidence that a production-recurrence cooperative mapping must lose.
- Production source/library hashes and ROWS=1 state are unchanged. No full candidate build, candidate smoke, or candidate end-to-end run occurred.
