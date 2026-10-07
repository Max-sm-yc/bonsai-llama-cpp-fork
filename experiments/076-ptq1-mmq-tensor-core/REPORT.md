# Experiment 076: PTQ1_0 prefill int8 Tensor Core feasibility

## HYPOTHESIS

An sm_86 signed-int8 Tensor Core MMQ path might improve PTQ1_0 prompt throughput when multiple prompt positions share each decoded weight tile. Exp073's batch-one `m16n8k16` waste does not apply to this regime. The feasibility question was whether prefill MMQ lacks such a path, or whether a concrete tile/fragment variant had a defensible advantage over the active implementation.

## IMPLEMENTATION

Created the required isolated worktree `/home/maxsun/autonomous_projects/.worktrees/exp076-sm86-ptq1-mmq-tc` from main HEAD `5d1b4f74d1d446c2939209562f1d8498ddf11f63`. Required experiment notes and reports 065/066/073 were read. Source inspection covered `mmq.cu`, PTQ1_0 MMQ configuration/dispatch, PTQ1_0 tile decoding, MMQ dot consumers, and the CUDA MMA primitives. Detailed audit is in [`source_audit.md`](../../results/exp076/source_audit.md).

The premise is already implemented: sm_86 PTQ1_0 prefill MMQ expands PTQ weights to signed bytes in the MMA tile layout and calls `mma.sync` signed-int8 Tensor Core instructions. The prompt positions are the MMQ N dimension. Q8_1 activation subgroup scales and PTQ weight scales are multiplied into the int32 MMA results before float accumulation. As a result, no separate int8 Tensor Core path was added.

No candidate source changes were made. I did not tune tile/warp/fragment variants: the obvious schedule lever was already tested in Exp065 (and lost at every measured prompt length), while Exp066 found no concrete decoder alternative. Repeating either would not test a distinct premise.

A Release CUDA sm_86 build was configured with:

```bash
cmake -S . -B results/exp076/build-exp076 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
ninja -C results/exp076/build-exp076 ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-ptq1_0.cu.o
```

The focused PTQ1_0 MMQ CUDA translation unit compiled successfully. Its cubin was extracted and disassembled; `results/exp076/ptq1-sm86.sass.txt` contains 1,792 `IMMA.16832.S8.S8` instructions across emitted template variants. The active PTQ1_0 consumer iterates K in 32-element chunks, so the emitted operation is `m16n8k32`; the generic `m16n8k16` overload is not the active PTQ1_0 MMQ fragment. The full build was stopped at 79/418 objects after the focused audit established there was no candidate to benchmark. Partial build and compiler temporary files remain locally under `results/exp076/build-exp076/` and `results/exp076/compiler-tmp/` and are excluded from Git; the exact cubin, SASS, source audit, and hashes are tracked.

## RESULT

No candidate or candidate benchmark samples were produced. The active CUDA prefill implementation already uses the requested signed-int8 Tensor Core operation, so this investigation found no new path to compare. The repository's current-best PTQ1_0 prefill medians remain 1,292.26 / 1,378.14 / 1,355.30 / 1,331.25 tok/s at prompts 128 / 512 / 2,048 / 4,096, respectively. These are existing results, not Exp076 measurements. There are no before/after candidate samples.

## CORRECTNESS

There was no candidate arithmetic change to compare against CUDA, CPU, or reference behavior. The audited active path preserves signed ternary expansion, Q8_1 subgroup scales, PTQ scales, and int32 MMA accumulation followed by float accumulation. No new correctness claim is made. Candidate CTests and fixed-seed model smoke were not applicable because no candidate was implemented.

## MICROBENCHMARK

No candidate microbenchmark or profiler capture was run. Existing evidence relevant to an altered tile schedule is Exp065: the 128-thread/I=64 schedule regressed type-143 MMQ totals by 15.6% / 3.8% / 3.3% at prompts 128 / 512 / 4,096. Exp066 found no grounded alternate trit decoder. These are prior experiment results, not Exp076 measurements.

## END-TO-END IMPACT

No candidate A/B or decode run was appropriate without a source-level candidate. Existing prefill and decode bests are unchanged. Peak VRAM was not measured for this experiment; no candidate allocations were introduced.

## ANALYSIS

The key distinction from Exp073 is that batch-one output-column waste is absent from prefill: current MMQ already shares weight tiles across prompt positions. Source dispatch chooses the MMA layout on NVIDIA CUDA and routes PTQ1_0 through `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma`. The emitted PTQ1_0 sm_86 cubin contains signed-int8 `IMMA.16832.S8.S8`; a new implementation of this operation would duplicate existing work.

Potential optimization is therefore limited to schedule, fragment, decoder, or scale-fusion changes within the current MMA pipeline. Exp065 already changed every PTQ1_0 Ampere config entry across its supported J values from 256 threads / occupancy target 1 / I=128 to 128 threads / occupancy target 2 / I=64; it lost at prompts 128, 512, and 4,096. Exp066 audited the active decoder and shared staging but did not implement a replacement. Unscreened possibilities include intermediate/larger I values, different thread/occupancy combinations, altered K iteration or MMA fragment sizes, and a new parallel trit decoder or fused scale accumulation. None had a new resource or codegen observation supporting a likely win in this pass; the new decoder/fragment options also need a full exact-layout design before implementation. There is therefore no distinct low-risk candidate to screen, and this experiment makes no claim that the remaining schedule space is exhausted.

## DECISION

**NO CANDIDATE; retain current implementation.** Candidate code was not retained because none was created. Production source remains clean in this worktree; no production merge or manager checkout changes were made.

## FOLLOW-UPS

A future screen should start from a measured bottleneck within the existing signed-int8 MMA pipeline, such as a verified fragment-layout or scale-accumulation cost, and compare an exact specialized path against current MMQ. It should not treat Tensor Core usage itself as a new optimization opportunity.

## IMPORTANT DISCOVERIES

- PTQ1_0 prefill MMQ already uses sm_86 signed-int8 Tensor Core MMA; the requested path is active today.
- Prompt positions occupy the output N dimension, so Exp073's batch-one output waste does not characterize prefill.
- PTQ1_0 weight scales and Q8_1 activation subgroup scales are applied to integer MMA accumulators before float sums.
- Exp065's occupancy-motivated tile change regressed across all measured prompt lengths; Exp066 did not find a validated alternate decoder.
