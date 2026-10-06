# Experiment 028: PTQ1_0 work-list lookahead prefetch

## HYPOTHESIS

In the active sm_86 `mul_mat_vec_ptq1_0_pt<1,1,false,false>` batch-1 GEMV, each lane visits its work-list indices `tid`, `tid + 128`, `tid + 256`, and so on. A prefetch of a later item while the current dot is running could overlap the next weight-block fetch with the serial packed-trit recurrence. This differs from experiments 001/002, whose generic-path change was bypassed by this dedicated kernel, and from experiment 027's current-item prefetch, which had no lookahead.

## IMPLEMENTATION

Temporarily added a compile-time lookahead in the active dedicated kernel. It computes `prefetch_idx = idx + distance*128`, checks `prefetch_idx < n_items`, decodes its `(row group, K block)` with the same `bpr_fd`, applies the same final-row clamp as the current work item, and issues `prefetch.global.L1` or `.L2` on that future block's address. It does not prefetch the current block. The dot recurrence, partial slot, output write, reduction/fold order, work ownership, activation layout, launch geometry, and dispatch are unchanged. The macro default is off; each tested candidate was compiled with an explicit distance/policy. The candidate source and exact patch are [candidate-source.cuh](../../results/exp028/candidate-source.cuh) and [candidate.patch](../../results/exp028/candidate.patch).

The tested variants were L1 distance 1 (the lane's next item), L1 distance 2, and L2 distance 1. SASS excerpts are [target_codegen_excerpt.txt](../../results/exp028/target_codegen_excerpt.txt) and [l2d1_prefetch_excerpt.txt](../../results/exp028/l2d1_prefetch_excerpt.txt); resource records are in the `*_target_resources.txt` files. [library-hashes.txt](../../results/exp028/library-hashes.txt) records the tested binary hashes and [trace-configuration.txt](../../results/exp028/trace-configuration.txt) records each run's selected library path and workload. To keep the research commit compact, the locally used shared libraries were removed after their hashes, codegen, resource data, and raw traces were saved; the candidate source and patch are sufficient to rebuild them.

## RESULT

**REVERT.** The address calculation is safe, and the prefetch instruction survives compilation, but none of the three options improved the focused active-kernel screen. L1 distance 2 was clearly slower in the single trace. L1 distance 1 and L2 distance 1 were also slower than the fresh control in that trace and showed no promising signal to justify correctness plus full-model A/B work.

## CORRECTNESS

The future-address calculation was exhaustively checked over boundary shapes and randomized shapes. It covered 4,575 shape/pitch combinations and 21,626,622 in-range future indices, with distances 1/2/3, all 128 thread IDs, short final CTA row groups, and padded row pitches; every computed K block and clamped row stayed inside its logical row and allocation. The check and log are [address_check.py](../../results/exp028/address_check.py) and [address_check.txt](../../results/exp028/address_check.txt).

No candidate passed the performance screen, so the full correctness suite and fixed-seed model smokes were not run against a retained implementation. Each profiling invocation did load and decode the PTQ1_0 model for 16 tokens, but this was not a correctness comparison. Production source and active library were restored to their requested SHA-256 values; see [restored_hashes.txt](../../results/exp028/restored_hashes.txt).

## MICROBENCHMARK

Nsight Compute request counters are unavailable on this host (`ERR_NVGPUCTRPERM`), so there is no request-traffic measurement. As a focused alternative to a synthetic CUDA-event kernel, Nsight Systems traced the actual model dispatch and aggregated durations for the exact active plain specialization. Each run used context 512, 16 generated tokens, and counted 485 launches of `mul_mat_vec_ptq1_0_pt<(int)1,(int)1,(bool)0,(bool)0>`. The raw traces, stdout, and per-kernel CSVs are in `results/exp028/raw/`.

| Variant | Plain-kernel total | Per-launch median | Delta in total vs control |
|---|---:|---:|---:|
| Fresh control | 9.577 ms | 14.560 µs | — |
| L1 distance 1 | 9.799 ms | 15.904 µs | +2.3% |
| L1 distance 2 | 10.281 ms | 17.313 µs | +7.4% |
| L2 distance 1 | 9.793 ms | 15.936 µs | +2.3% |

These are single traces, with wide within-run duration ranges and sequential candidate/control order. Treat the small differences as a screen, not a precise performance estimate. The L1 distance-2 trace is especially unfavorable. Resource output reported 74 registers/thread and no spills for one candidate code entry, with the alternate entry at 76 registers and no spills; the control entries were 76 registers and no spills.

## END-TO-END IMPACT

No controlled seven-repetition decode A/B was run because the kernel-level screen did not identify a promising candidate. The one-repetition, 16-token profiling run is not comparable to the requested 128-token benchmark or experiment 026's fresh control (81.5795 / 79.0749 tok/s at contexts 512 / 4096). No end-to-end gain is established.

## ANALYSIS

This is a genuine lookahead in the active dedicated specialization: SASS places the prefetch before the current item's weight/activation loads, and the address points to the same lane's next work-list item. The implementation avoids speculative out-of-range addresses by checking the logical future index first. For short final row groups, its row clamp matches the existing compute path.

The lookahead requires extra future-index division/address instructions and a conditional before issuing the hint. The trace did not show a time reduction that repays that work. This is consistent with earlier evidence: experiment 025's source-level strip mining did not help, and experiment 024's isolated packed-recurrence speedup did not predict the production schedule. Here the screen observes the real active kernel rather than a surrogate, so the negative result is sufficient to stop before an end-to-end candidate run. Nsight Systems does not report cache request counts, and Nsight Compute remains unavailable.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** The tested lookahead is address-safe and emits the intended SASS operation, but none of the measured distance/policy choices beat control in the focused screen. Restored source SHA-256 `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496` and active library SHA-256 `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`. No commit was made.

## FOLLOW-UPS

Only revisit this lookahead if new active-kernel evidence shows the weight fetch is latency-bound enough to overcome the extra index/address work, or if request counters become available. Preserve the ROWS=1 production implementation meanwhile.

## IMPORTANT DISCOVERIES

- The per-thread future work-list address can be decoded safely with the existing fast-divisor and final-row clamp; a large boundary/random screen found no invalid future address.
- The `.L1` and `.L2` PTX hints reached sm_86 SASS as `CCTL.E.PF1` and `CCTL.E.PF2` in the active target function.
- In a single actual-kernel trace, L1 distance 1 and L2 distance 1 were each 2.3% above control aggregate time; L1 distance 2 was 7.4% above. The screen gave no reason to pay the correctness and full-model A/B cost.
- Production source and active-library hashes were restored exactly; the candidate patch/source, binary hashes, codegen/resource excerpts, and raw traces remain under `results/exp028/`. Candidate shared objects were omitted from the research commit.

## MANAGER VERIFICATION

The manager reran `address_check.py` and reproduced its 4,575 shape/pitch and 21,626,622 address result. It independently recomputed all four plain-kernel totals/medians from the saved Nsight Systems CSVs and confirmed each `.nsys-rep` used the intended `LD_LIBRARY_PATH` and captured 485 launches of the target specialization. Direct SHA-256 checks matched the restored production source and library. The one-trace screen is sufficient to reject these variants from further work, but it does not establish a statistically robust end-to-end regression or improvement.
