# Experiment 032: no-padding per-row PTQ1_0 K-block SoA

## Hypothesis

Keep each 128-weight PTQ1_0 block at 28 bytes, but arrange each row as seven 32-bit word planes across K blocks. Word `w` of block `b` is at `row + w*nblk + b`. With the ROWS=1 flattened `(row, K block)` work mapping, lanes that process adjacent K blocks then load adjacent addresses for a given packed word. This tests coalescing without Exp031's 32-byte padding or 14.29% payload increase.

## Setup and implementation

The standalone CUDA harness is [soa_screen.cu](../../results/exp032/soa_screen.cu). It compares the exact 28-byte AoS block against a 28-byte-per-block row SoA with seven word planes. Both use the active base-3 recurrence, the production planar Q8_1 activation mapping, per-32-element scale/isum FMA fold, and the ROWS=1 flat item ownership (`item = row*nblk + kbx`). A second kernel applies the production four-interleaved-accumulator output fold. Each run uses 2,048 rows, enough to exercise many row/warp boundaries, and nine alternating-order samples.

An independent host decoder checked all generated weight codes against a diagnostic CUDA kernel that emits codes using the packed recurrence: zero code mismatches over 10,485,760 codes at 40 blocks/row and 35,651,584 at 136 blocks/row. Every final row output from both layouts matched an independent host block-dot and output-fold reference exactly (zero mismatches, max absolute error 0). The reference includes the block half scale, and qh reconstruction uses only qh bytes. Compute Sanitizer memcheck at 40 blocks/row and 257 rows reported zero errors.

The measured harness includes two launches per sample (work and fold). Its output fold matches the production four-accumulator association, but it does not reproduce the production shared-memory partial array, CTA row grouping, or full model loader/runtime. The diagnostic code-emission kernel runs only before timing.

## Exact commands

From the repository root:

```sh
nvcc -O3 -arch=sm_86 --ptxas-options=-v results/exp032/soa_screen.cu -o results/exp032/soa_screen 2>results/exp032/resources.txt
./results/exp032/soa_screen 40 2048 150 > results/exp032/screen_40.txt
./results/exp032/soa_screen 136 2048 80 > results/exp032/screen_136.txt
compute-sanitizer --tool memcheck --error-exitcode=5 ./results/exp032/soa_screen 40 257 5 > results/exp032/memcheck_40.txt 2>&1
cuobjdump --dump-sass results/exp032/soa_screen > results/exp032/soa_screen.sass
```

GPU was NVIDIA GeForce RTX 3080, compute capability 8.6, driver 580.178.04, CUDA Toolkit 13.2.

## Results

CUDA event milliseconds for the work-plus-fold pair; nine samples per arm, alternating AoS/SoA order:

| K blocks/row | AoS median (range) | SoA median (range) | SoA delta |
|---:|---:|---:|---:|
| 40 | 0.00920768 (0.00919552–0.00924864) | 0.00916011 (0.00915413–0.00916821) | -0.52% |
| 136 | 0.02507520 (0.02503440–0.02520520) | 0.02311120 (0.02307840–0.02314120) | -7.83% |

Raw samples and correctness summaries are in `results/exp032/screen_40.txt` and `screen_136.txt`; sanitizer output is in `results/exp032/memcheck_40.txt`.

## SASS and resources

`nvcc -O3 -arch=sm_86 --ptxas-options=-v` reported 40 registers/thread, zero stack, and zero spills for each timed work kernel (`work_aos`, `work_soa`). The fold used 39 registers, also without stack or spills. `results/exp032/soa_screen.sass` shows 43 scalar `LDG.E` instructions in each timed work function and no `LDG.E.128` instructions. The static load instruction count is unchanged; the candidate's premise is improved lane-to-address correspondence and fewer memory sectors, not fewer load instructions.

## Integration implications and limits

The representation keeps payload at 28 bytes/block, so it has no projected weight-memory increase. A runtime path would have to repack each PTQ1_0 row into seven planes, or teach the model loader to produce this form, and route the active ROWS=1 kernel through those planes. Keeping a second copy while retaining AoS for other consumers would erase the memory-footprint advantage. The 40-block case, representative of common K=5120 rows, is effectively a tie at -0.52%; the clearer -7.83% result occurs at 136 blocks. The measured harness also omits production CTA/shared-memory overhead. No production or end-to-end result is established.

The active production source hash remains `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`; the active CUDA library remains `c828135b126ec507ffbecb4dc11b6a7a9ac5cd0fe050553323d7f35c38fae6c7`. Both were unchanged. Repository HEAD remains `c0dbe28678598d749afa1aa3967e8e9da19a46fb`; no commit was made.

## Decision (KEEP/REVERT/INCONCLUSIVE)

**KEEP for a separately scoped runtime feasibility study.** The long-row screen is an exact, repeatable 7.83% win with unchanged footprint; the 40-block result is a small 0.52% win and should be treated as a tie. The evidence justifies checking whether model-loader repacking and production CTA mapping preserve the 136-block gain. It does not justify claiming a production win or integrating this layout without a runtime A/B.

## Manager verification

The manager independently reran the final screen binary at both shapes with nine alternating samples and 2,048 rows. All device codes, AoS/SoA outputs, and independent host row outputs matched exactly; a second Compute Sanitizer run reported zero errors. The 40-block case remained a tie (-0.22%); the 136-block case remained faster (-7.82%). Full samples and hashes are in `results/exp032/manager_rerun_40.txt`, `manager_rerun_136.txt`, `manager_memcheck_40.txt`, and `manager_verification.txt`. Production source/library hashes were unchanged.
