# Experiment 029: PTQ1_0 weight cache policy

## HYPOTHESIS

The active plain `<ncols=1, ROWS=1, has_fusion=false, has_gate=false>` planar GEMV streams packed PTQ1_0 weights while reusing smaller activation planes. Bypassing L1 for aligned packed `qs` words with `.cg` might preserve activation residency and improve decode.

## IMPLEMENTATION

Added a temporary narrow helper for aligned `qs` u32 reads in `mmvq-ptq1_0.cuh`. Compile-time policy 0 used ordinary loads; policy 1 used inline PTX `ld.global.cg.u32`; policy 2 was prepared for `.cs` but was not built or measured. Only the two packed-word loops used the helper; activation loads and the 28-byte block representation were unchanged. The exact candidate patch is `results/exp029/candidates.patch`; the production source snapshot is `results/exp029/production-source.cuh`.

## RESULT

The candidate compiled for sm_86 and was captured in a short actual-model profile. No matched control trace was captured, so the one run is only a workload/code-path confirmation and does not decide performance. No candidate passed a comparison screen; the experiment is inconclusive.

## CORRECTNESS

No candidate correctness suite or fixed-seed comparison was run. After restoring the source-default build, a fixed-seed PTQ1_0 smoke generated a non-empty 32-token completion (`results/exp029/production-smoke.json`). `ldd` confirmed that `llama-cli` and `llama-bench` resolve `build/bin/libggml-cuda.so.0`. The smoke checks production runtime usability, not candidate numerical equality. Source was restored to SHA-256 `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`.

## MICROBENCHMARK

A short Nsight Systems capture used RTX 3080 / sm_86 and the PTQ1_0 model at context 512 for 16 decode tokens. The target plain specialization was invoked 4,115 times, with 177.969 ms aggregate time, 43.249 µs average, and 28.896 µs median per invocation in the Nsight Systems summary. This is one candidate trace without a contemporaneous matched control; compare neither its count nor total to the differently configured experiment 028 trace. Raw profile, exported CSV, valid llama-bench JSON (`cg-benchmark.json`), combined command stdout log, and stderr are in `results/exp029/`.

The candidate object’s exact `<1,1,false,false>` specialization emitted six `LDG.E.STRONG.GPU` packed-word loads corresponding to the six u32 reads; the nine activation vector loads remained `LDG.E.128.CONSTANT`. Static resources were 74 registers/thread, zero stack, zero shared memory, and zero local memory. The concise SASS excerpt is `results/exp029/cg-plain-specialization.sass`; resource data are in `results/exp029/cg-resource.txt`.

## END-TO-END IMPACT

Not measured. There was no matched source-default control, no 128-token seven-repetition decode A/B, and no evidence sufficient to advance.

## ANALYSIS

The `.cg` modifier survived into the exact active sm_86 plain specialization and did not raise register or spill resources relative to the 74-register object screen. However, without a matched control timing this only validates code generation. `.cs` was not tested. A Ninja dependency recovery caused a full CUDA rebuild; an interrupted subsequent rebuild removed some link inputs, and a direct relink failed. The source is restored, but the active build library currently needs a successful source-default rebuild before it can be considered restored.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**INCONCLUSIVE; do not keep the candidate.** Source is restored to the production SHA. The source hash matches production exactly. A full source-default rebuild restored a usable CUDA library with SHA-256 `c828135b126ec507ffbecb4dc11b6a7a9ac5cd0fe050553323d7f35c38fae6c7`, compared with archived production hash `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`. `ldd` confirms both binaries resolve the rebuilt library; a fixed-seed PTQ1_0 smoke passed. The active target resource record exactly matches exp028 control (76 registers, no stack/shared/local). The byte-hash difference remains unexplained: a full rebuild or code-generation metadata could account for it, but no binary-level comparison was possible because the archived file was not preserved. Do not treat the rebuilt binary as a verified byte-for-byte reproduction. Candidate library SHA-256 was `4f9417e30f85f3d929040a32db7767d77a09bf789c6dc1b413eff200b714e2b5`. No source commit was made.

## FOLLOW-UPS

If revisiting cache policy, first establish a contemporaneous source-default actual-kernel trace and compare against the exact current build. The restored source-default build passed a fixed-seed PTQ1_0 smoke, though its library SHA differs from the archived binary. Do not infer an end-to-end benefit from this single screen.

## IMPORTANT DISCOVERIES

- `.cg` reached sm_86 in the active plain specialization as `LDG.E.STRONG.GPU` on the packed u32 reads; activation vectors remained cached vector loads.
- The object retained 74 registers/thread and no stack/local storage.
- The candidate profile completed an actual PTQ1_0 model decode and observed 4,115 active plain-kernel invocations.
- The build tree’s Ninja dependency database repeatedly requested broad rebuilds after recovery. An interrupted pass removed several CUDA object files; a subsequent full source-default rebuild restored the library. Its target specialization has the archived resource record, but the rebuilt binary hash differs and was not verified byte-for-byte.

## BUILD RESTORATION CHECK

The full source-default rebuild completed after Ninja recovered its dependency database. A PTQ1_0 fixed-seed 32-token smoke passed, and `ldd` resolves `build/bin/llama-cli` / `llama-bench` to `build/bin/libggml-cuda.so.0`. Source SHA is the recorded production value. The rebuilt library SHA is `c828135b126ec507ffbecb4dc11b6a7a9ac5cd0fe050553323d7f35c38fae6c7`, which differs from archived `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`. The active `<1,1,false,false>` resource record matches archived control (76 registers, zero stack/shared/local), but no byte-level binary comparison was available. A full rebuild or code-generation metadata could explain the difference; that is not demonstrated, so this is not a verified binary reproduction.
