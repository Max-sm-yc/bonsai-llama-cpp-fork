# Experiment 025: PTQ1_0 K-block work-list strip mining

## HYPOTHESIS

The active ROWS=1 planar GEMV assigns each 128-thread lane independent `(row-group, K-block)` work-list items, iterating by 128. Grouping two or four such items per lane could expose instruction-level parallelism and hide K-block load latency without changing the work-item owner, serial PTQ1_0 block-dot recurrence, partial-buffer slot, or final reduction/fold order.

## IMPLEMENTATION

Added compile-time `PTQ1_0_PT_ITEMS_PER_THREAD` (default 1). The grouped loop visits `tid + item*128` for each unrolled compile-time item and advances its base by `128*group`; each valid work item uses the original address calculation, serial dot function, partial slot, and gate slot. The candidates were compiled with values 2 and 4 in the existing configured sm_86 CUDA translation unit. Each candidate library was isolated under `results/exp025/libs/{items2,items4}` and verified with `LD_LIBRARY_PATH`/`ldd`; the source-default control library was separately archived before the builds.

The captured candidate patch is [candidate.patch](candidate.patch); control source is [control-mmvq-ptq1_0.cuh](control-mmvq-ptq1_0.cuh). CUDA build logs, library hashes, and full `cuobjdump --dump-resource-usage` outputs are under `results/exp025/`.

## RESULT

**REVERT.** Neither strip-mined form improved decode. Both lost to the same-session control at both contexts; the four-item candidate also had severe context-4096 outliers. ROWS=1 production source and its archived baseline CUDA library were restored and hash-verified.

## CORRECTNESS

The normal `bash tests/run_correctness.sh` build step was not compatible with the current build tree: Ninja reported a premature end/recovery and a dry run showed CMake regeneration with a broad rebuild. To avoid changing the active build while checking isolated libraries, the script's test commands were run directly for each candidate with its library selected by `LD_LIBRARY_PATH`:

- Four selected CTests passed for items2 and items4; logs: `results/exp025/items2_ctest.log`, `items4_ctest.log`.
- CUDA-vs-CPU `MUL_MAT` coverage passed 96/96 PTQ1_0/PQ2_0 cases for both variants; logs: `items2_backend_ops.log`, `items4_backend_ops.log`.
- Fixed-seed 32-token smokes (context 512, seed 42, temperature 0, 8 CPU threads, 99 GPU layers) passed for PTQ1_0 and PQ2_0 for both variants. Normalized completions matched the archived ROWS=1 baseline exactly (872 and 862 normalized characters, respectively). JSON: `items2_smoke.json`, `items4_smoke.json`.

## MICROBENCHMARK

No separate kernel timer was run; the required full-model decode benchmark was the performance screen. The active plain `ncols=1, ROWS=1` specialization reports the same compiler resources for control, items2, and items4: **76 registers/thread, stack 0, shared 0, local 0**. This does not indicate register pressure or spills from strip mining. The complete `cuobjdump` inventories are `control_resources.txt`, `items2_resources.txt`, and `items4_resources.txt`; two matching cubin entries are present for the specialization in each library.

## END-TO-END IMPACT

All runs used the canonical seven-repetition decode command, contexts 512/4096, 128 generated tokens, and the <=60°C / <=5% utilization start gate. Full samples and telemetry are in `results/exp025/{control,items2,items4}.json`. Throughput is tok/s; SD is sample SD.

| Variant | Context | Median | Mean ± SD | Range | Median delta vs control |
|---|---:|---:|---:|---:|---:|
| Control | 512 | 81.9018 | 81.8663 ± 0.3069 | 81.1974–82.0871 | — |
| Items2 | 512 | 81.5518 | 81.4192 ± 0.3298 | 80.6809–81.6056 | -0.43% |
| Items4 | 512 | 81.4783 | 81.3369 ± 0.3411 | 80.5738–81.5063 | -0.52% |
| Control | 4096 | 79.3736 | 79.3066 ± 0.2646 | 78.7344–79.4928 | — |
| Items2 | 4096 | 79.0432 | 78.6758 ± 0.5916 | 77.4893–79.0805 | -0.42% |
| Items4 | 4096 | 78.2916 | 71.8525 ± 11.2666 | 49.8427–79.0052 | -1.36% |

Every process began idle at 56–60°C and 0% GPU utilization; sampled temperature ranges were 56–74°C (control), 58–75°C (items2), and 60–77°C (items4). All peaked at 6,805 MiB whole-GPU memory. `ldd` confirmed each candidate process resolved its intended candidate library.

## ANALYSIS

Grouping preserved per-lane work ownership and exact slot/reduction ordering, and candidate correctness matched. However, the active specialization remained at 76 registers with no stack/local usage, and neither group size improved model throughput. The small consistent items2 losses and items4 losses do not justify promotion; context-4096 variability remains visible, especially in the four-item run. The experiment provides no evidence that this simple unroll exposes useful overlap in the active generated code.

## DECISION

**REVERT.** Restored production source SHA-256 `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496` and active CUDA library SHA-256 `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`, matching the requested ROWS=1 baseline. Runtime revision was not changed. No commit was made.

## MANAGER VERIFICATION

The manager independently parsed the control/items2/items4 JSON and confirmed the reported medians, ranges, and deltas. Re-hashed the active source and CUDA library; both match the ROWS=1 baseline. Recompared the generated PTQ1_0 and PQ2_0 32-token text with the archived baseline completions; all four candidate completions match exactly after removing runtime boilerplate. The selected test log shows 4/4 passing, and the backend-op logs show 96/96 passing for both candidates.

## FOLLOW-UPS

Retain the current one-item loop. Further ILP experiments should first inspect generated instruction scheduling or use an active-kernel timer/counters; static source unrolling did not produce a resource or end-to-end benefit here.

## IMPORTANT DISCOVERIES

- Items2/items4 preserve all 96 CUDA-vs-CPU matmul cases and both fixed-seed model completions.
- The active specialization reports identical 76-register, zero stack/local use across all builds.
- Decode medians did not improve: items2 lost 0.42–0.43%; items4 lost 0.52–1.36%, with a 49.84 tok/s long-context outlier.
- The exact source-default source and archived active library were restored; candidate binaries remain only as experiment artifacts under `results/exp025/libs/`.
