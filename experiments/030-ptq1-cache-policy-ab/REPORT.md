# Experiment 030: PTQ1_0 cache policy A/B

## HYPOTHESIS

The active plain `<ncols=1, ROWS=1, fusion=false, gate=false>` sm_86 planar GEMV streams packed PTQ1_0 weights while reusing activation planes. Bypassing L1 with `.cg`, or marking packed words streaming/evict-first with `.cs`, might preserve useful activation cache residency.

## IMPLEMENTATION

Compared ordinary global loads, inline `ld.global.cg.u32`, and inline `ld.global.cs.u32` on only the six aligned packed `qs` word reads. Arithmetic, work ownership, activation loads, dispatch, and the 28-byte block layout stayed fixed. The candidate helper and exact patch are in [`candidate.patch`](../../results/exp030/candidate.patch) and [`production-source.cuh`](../../results/exp030/production-source.cuh).

The Exp029 `.cg` shared object was not available for reuse. Both candidate variants were freshly compiled by invoking only the generated `mmvq.cu.o` compile command with a policy define, then linked separately against the existing object set. The default arm used the preserved current source-default library copy. `ldd` with each arm's `LD_LIBRARY_PATH` resolved `llama-bench` to the intended isolated `.so`; see [`runtime-resolution.txt`](../../results/exp030/runtime-resolution.txt). No broad Ninja build was run. The source was restored to the production hash, and the generated `mmvq.cu.o` was recompiled with the source-default command after the experiment.

Candidate library hashes: default `c828135b...fae6c7`, `.cg` `8606d293...a2e19`, `.cs` `96e376ee...561d7`; full hashes and paths are in [`library-hashes.txt`](../../results/exp030/library-hashes.txt).

## RESULT

**REVERT.** `.cg` is decisively slower in the matched active-kernel screen. `.cs` was about 0.30% faster there, but repeated end-to-end comparisons were slower at both contexts. No cache-policy candidate is retained.

## CORRECTNESS

All actual-model kernel traces and benchmark runs completed without runtime errors. A fixed-seed 32-token completion comparison was attempted but did not complete: `llama-cli` entered an unbounded terminal/UI output path despite the requested token limit, so it was interrupted and its oversized output removed. No exact numerical equivalence claim is made. Since both policy candidates regressed end-to-end, neither is retained; a full correctness suite was not run against them.

## MICROBENCHMARK

Captured three Nsight Systems traces per arm in rotated order (`default`, `.cg`, `.cs`; `.cs`, `default`, `.cg`; `.cg`, `.cs`, `default`). Each trace used the PTQ1_0 model, context 512, 16 generated tokens, `-ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8`, and captured 485 calls of the exact active plain specialization. Aggregated target-kernel times:

| Policy | Trace totals (ms) | Mean total (ms) | Mean per launch (µs) | Median per launch (µs) | Delta vs default |
|---|---:|---:|---:|---:|---:|
| Default | 9.5795, 9.5812, 9.5796 | 9.5801 | 19.753 | 14.592 | — |
| `.cg` | 21.8659, 21.8774, 21.9082 | 21.8838 | 45.121 | 28.929 | +128.4% |
| `.cs` | 9.5504, 9.5480, 9.5547 | 9.5510 | 19.693 | 14.016 | -0.30% |

Per-trace counts, averages, medians, min/max, Nsight Systems reports, SQLite exports, and benchmark stdout are under [`raw/`](../../results/exp030/raw/); the concise table is [`microbenchmark.csv`](../../results/exp030/microbenchmark.csv). The small `.cs` edge is a screening signal only.

SASS confirms six packed-word loads in the active function: `.cg` emits `LDG.E.STRONG.GPU`, `.cs` emits `LDG.E.EF`. The activation vector loads remain `LDG.E.128.CONSTANT`. Concise function-filtered disassembly is in [`cg-plain.sass`](../../results/exp030/cg-plain.sass) and [`cs-plain.sass`](../../results/exp030/cs-plain.sass). Resource records show 74 registers/thread for the candidate object and 76 for source-default, with no stack, shared, or local memory in either active entry; see [`default-resources.txt`](../../results/exp030/default-resources.txt), [`cg-resources.txt`](../../results/exp030/cg-resources.txt), and [`cs-resources.txt`](../../results/exp030/cs-resources.txt). Resource output contains PTX and ELF entries, so both records are preserved.

## END-TO-END IMPACT

Ran `benchmark/run.py` with identical PTQ1_0 decode settings, seven repetitions, 128 generated tokens, contexts 512 and 4096, and the 60 C cooldown gate. Two passes were used: default then `.cs`, followed by `.cs` then default. Start temperature was 51 C / 60 C in the first pass and 60 C / 60 C in the reversed pass; all arms began below the configured gate. Median throughput:

| Pass/order | Policy | 512 (tok/s) | 4096 (tok/s) |
|---|---|---:|---:|
| 1: default → `.cs` | Default | 82.1863 | 79.6259 |
| 1: default → `.cs` | `.cs` | 80.9017 (-1.56%) | 78.4684 (-1.45%) |
| 2: `.cs` → default | `.cs` | 80.8342 (-1.10%) | 77.7345 (-0.91%) |
| 2: `.cs` → default | Default | 81.7329 | 78.4486 |

At context 4096, both arms in the second pass had large slow-tail repetitions; raw samples are retained in the JSON outputs. The direction of the median result was the same in both orders and at both contexts. The four complete run.py outputs are [`default-e2e.json`](../../results/exp030/default-e2e.json), [`cs-e2e.json`](../../results/exp030/cs-e2e.json), [`cs-e2e-r2.json`](../../results/exp030/cs-e2e-r2.json), and [`default-e2e-r2.json`](../../results/exp030/default-e2e-r2.json). The four raw llama-bench JSON outputs were also moved into [`results/exp030/`](../../results/exp030/) from the harness default `results/raw/` location. Peak whole-GPU memory was 6805 MiB.

## ANALYSIS

The load modifiers reached the intended six packed reads without changing the activation path. `.cg` substantially worsened the actual kernel, consistent across all three traces. `.cs` produced a repeatable but very small kernel-level reduction; that did not translate into model throughput. The reversed-order end-to-end pass still showed `.cs` behind default by 0.9–1.1% in median throughput, so the result is not explained by the first pass's different initial temperatures. Nsight Systems does not provide cache-request counters here, and Nsight Compute remains blocked by `ERR_NVGPUCTRPERM`.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** Production source hash is restored to `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`. Active source-default CUDA library and `/tmp` backup both remain at `c828135b126ec507ffbecb4dc11b6a7a9ac5cd0fe050553323d7f35c38fae6c7`. Candidate library hashes remain in `results/exp030/library-hashes.txt`; the temporary candidate `.so` and object copies were removed after the comparison. No commit was made.

## FOLLOW-UPS

Do not advance `.cg` or `.cs` for this specialization. Revisit cache policy only with new cache-traffic evidence or a materially different data-reuse premise; the current `.cs` microbenchmark edge is outweighed by repeated end-to-end regression.

## IMPORTANT DISCOVERIES

- The cache modifiers apply to exactly six aligned `qs` u32 loads; the activation vectors remain cached vector loads.
- `.cg` generated `LDG.E.STRONG.GPU` and was about 2.28x slower in the matched actual-kernel trace.
- `.cs` generated `LDG.E.EF`, improving active-kernel aggregate time by about 0.30%, but lost 0.9–1.6% in matched end-to-end median throughput at both contexts and in both run orders.
- Source-default source and active library hashes were preserved; the baseline object was restored after the candidate builds.
