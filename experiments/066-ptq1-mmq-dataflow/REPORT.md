# Experiment 066: PTQ1_0 MMQ dataflow challenge

## HYPOTHESIS

Exp065 established that PTQ1_0 type-143 MMQ contributes 65.5% of captured kernel time for a 4,096-token prefill. Its PTQ1_0 loader performs a serial five-step base-3 expansion for each packed word and stages expanded signed bytes in shared memory before the DP4A dot path. A materially different expansion or producer/consumer mapping could reduce the measured MMQ cost.

## IMPLEMENTATION

Created isolated worktree `/home/maxsun/autonomous_projects/.worktrees/exp066-ptq1-mmq-dataflow` at manager documentation HEAD `8e53e3a5f4f4a7974127b03b479038051b118e73`. Read `research/STATE.md`, the Exp065 experiment row and relevant idea, and `experiments/065-ptq1-prefill-mmq/REPORT.md` before source review.

Inspected the active loader (`mmq-load-tiles.cuh`, `ggml_cuda_mmq_load_tiles_ptq1_0`) and current config. The source uses eight lanes per packed word: six lanes emit values and the high-bit lane repairs the final two values; lane 7 duplicates the word load. Each emitted pair of byte digits advances through five dependent multiply-by-three steps and signed-byte packing, followed by shared-memory stores. The type-143 dot consumer uses the established staged tile and DP4A path. The active PTX for `mul_mat_q<GGML_TYPE_PTQ1_0,128,false>` confirms the repeated `mul.lo.s32 ..., 3` and `st.shared` sequence. Its excerpt is saved at `results/exp066/type143-j128-baseline.ptx` (SHA-256 `2722a4b442817c90ba63377441f4299027bae96c8f4066ea2f77cd9881688206`). The freshly built type-specific CUDA object also provides the active SASS entry, saved at `results/exp066/type143-j128-baseline.sass` (SHA-256 `57cb203d6f86c3477602f2b065254d82e0961bf8898a5f39e3274dc9c5a85ae3`). Broad all-kernel dumps and compiler scratch were removed after extracting these relevant artifacts.

No suitable candidate emerged from the source/compiler evidence. A parallel exact digit expansion needs a new decoder; eliminating shared staging needs a new direct packed-weight dot implementation while preserving the MMQ tile and output semantics. Neither is a local, evidence-supported edit, and attempting either speculatively would risk arithmetic/layout correctness without a grounded expected benefit. Exp065's I=64/128-thread schedule is explicitly excluded from reconsideration. No production source was changed.

A clean Release CUDA sm_86 baseline build completed all 441 Ninja steps in the isolated worktree with the prescribed CMake/Ninja configuration and worktree-local `TMPDIR=tmp-exp066`. The resulting `build/bin/llama-bench` SHA-256 is `48f7dae8aa4d8553659d83f56afed7d4899d2c359e95026cb362002acdc38e71`; `build/bin/libggml-cuda.so.0.21.0` SHA-256 is `21bff8d31b9cf49358434f670f82551a8a58612237d95d7e185bd9180f4ed692`. `ldd` resolves `libllama.so.0`, `libggml-cuda.so.0`, and `libggml-base.so.0` from this worktree's `build/bin`. The three relevant source files match the code baseline commit `ffb0ef37690b902829ea1158b02b14517ed93c2b`; their SHA-256 values are: `mmq-load-tiles.cuh` `4312755ddfc498f23c9be34f1771bb929c9c474fd8089451c41234968e6400ba`, `mmq-vec-dot.cuh` `071639b9d69f97d1bd28819b5cb2b21a814e17514dd74da1f3d4196151497720`, and `mmq-config-ampere.cuh` `6631e4c06c0538774822e850414dc5846c1a0486ea5a5c9dab541e9b0b8b07e0`. No candidate or candidate library was produced.

## RESULT

No candidate was implemented, so there is no before/after candidate measurement. This is a reasoned no-candidate outcome, not evidence that the dataflow hypothesis is false. Current production source remains the manager baseline at commit `ffb0ef37690b902829ea1158b02b14517ed93c2b`; the worktree is based on manager HEAD, which adds documentation. Model SHA-256 is `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.

## CORRECTNESS

No arithmetic or layout candidate was created, so no candidate correctness claim or comparison applies. The active baseline is unchanged. Existing correctness evidence remains the project record in `research/STATE.md` and Exp065.

## MICROBENCHMARK

No new profile or candidate timing was run because no candidate passed the feasibility screen. The applicable baseline diagnostic remains Exp065: at prompt 4,096, 9,528 type-143 launches took 5.882 s total (617.3 us mean, 667.5 us median; 55.4–916.8 us range), 65.5% of captured GPU-kernel time. Exp065's 128-thread/I=64 candidate was slower than baseline by 15.6%, 3.8%, and 3.3% at prompts 128, 512, and 4,096 respectively; it is not repeated here. Those are profiled diagnostics, not unprofiled end-to-end results.

## END-TO-END IMPACT

No candidate A/B, decode benchmark, or unprofiled E2E candidate comparison was run. There is no new performance claim and no production dispatch change, so decode behavior is unchanged by this experiment.

## ANALYSIS

The compiled baseline confirms that the loader contains the hypothesized serial trit extraction and shared staging. However, the existing implementation's digit order and byte layout are coupled to its DP4A tile consumer. A meaningful replacement needs a concrete exact parallel decoder or a new direct compute path, plus a kernel-specific reference test before timing. The evidence reviewed did not identify a small implementation with a defensible expected gain. Repeating the prior tile-size change would not test the requested distinct dataflow premise.

## DECISION

**INCONCLUSIVE — NO CANDIDATE.** Source remains untouched. This closes the current screen without claiming an MMQ speedup or disproving the hypothesis. Reopen only with a concrete decoder or direct packed-compute derivation and exact test plan.

## FOLLOW-UPS

- In a separate follow-up, derive an exact SIMD/warp-parallel base-3 digit expansion and compare generated instruction count against the five dependent `mul.lo` chain before integrating it.
- Alternatively, prototype a standalone direct packed-weight × Q8 dot kernel with exact block-reference coverage, then compare it against staged MMQ on the same prompt-side shapes.

## IMPORTANT DISCOVERIES

- The active PTX for the J=128 type-143 specialization preserves the dependent multiply-by-three expansion and shared stores seen in source; the codegen has not eliminated that work.
- Exp065's unsuccessful tile geometry result remains relevant but does not test a new unpack/compute dataflow.
- No achieved occupancy claim is made; this experiment used no occupancy counters.
