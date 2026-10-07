# Exp056: PTQ1_0 paired-output kernel screen

## HYPOTHESIS

A direct K/V paired-output PTQ1_0 kernel can share one already-prepared planar Q8_1 activation and save enough launch work to beat two current single-output kernels at K=5120 and 2048 rows per matrix.

## IMPLEMENTATION

In the isolated worktree at `/home/maxsun/autonomous_projects/.worktrees/exp056-kv-pair-screen` (manager HEAD `e249f20f1206b5f4571c6a36c5105349bae84a1a`), a temporary paired kernel used `ptq1_0_pt_block_dot` twice per K block, once for the corresponding K row and once for V, with the same planar Q8 activation pointer. It reduced both dot streams in one CTA and wrote two separate outputs. The paired grid had one CTA per row; the single-output controls used the production launcher geometry and kernel specialization.

The source patch is preserved at `results/exp056/paired_kernel_patch.diff`; the standalone correctness and benchmark harness is `results/exp056/bench_pair.cu`. The experimental CUDA object/shared library and executable are also retained for review. Both modified production source files were restored after screening. No graph scheduler integration was attempted because the corrected graph screen lost.

## RESULT

The paired kernel was numerically correct against two separate current-kernel invocations. Sequential stream-launch timing showed a small apparent gain, but the corrected CUDA Graph comparison showed the paired graph slower by about 3.7% at the median. This fails the integration gate. No candidate source remains checked out, no model A/B was run, and no commit was made.

## CORRECTNESS

At K=5120 and 2048 rows, all 2048 K and V outputs were compared against independent current-kernel results using the same packed weights and prepared planar activation. The K maximum absolute/relative errors were `1.2207031e-4` / `2.1492259e-7`; V errors were `1.2207031e-4` / `2.0508114e-7`. This covers all 40 K blocks per row and includes boundary rows 0 and 2047 and blocks 0 and 39. The harness rejects either output if maximum relative error exceeds `2e-6`.

No project CTests, CUDA-vs-CPU backend-op cases, or model smoke were run: the candidate never passed the graph-level performance gate and was not integrated.

## MICROBENCHMARK

The harness used CUDA 13.2, sm_86, K=5120, 2048 rows, one shared planar Q8_1 activation, and two independent PTQ1_0 matrices. Each arm was warmed, then timed for 1,000 invocations per repetition across 15 repetitions with alternating arm order. Timing used CUDA events on the launch stream.

For sequential stream launches, median paired time was 10.869 us versus 11.037 us for two single-output launches (median paired/two ratio 0.9750; ratio range 0.9674–1.0057). This apparent 2.5% gain did not survive graph replay.

For CUDA Graph replay, a graph with one paired kernel node was compared with a graph containing the same two single-output kernel nodes in sequence. With corrected stream-associated event timing, median paired time was 10.623 us versus 10.256 us for two nodes (ratio 1.0369; 15 ratios ranged 1.0290–1.0744). The single-node graph was consistently faster. Paired/two timing ranges were 9.765–10.668 / 9.418–10.282 us as GPU clocks ramped during the run.

The first two graph attempts used `cudaEventRecord(event)` without its stream argument after switching graph capture to a nonblocking stream. Those events were ordered on the legacy default stream and did not measure the graph launches. Their artifacts (`graph-run.txt`, `graph-run2.txt`) are invalid and retained only to document the diagnosed harness bug. `graph-run-corrected.txt` is the valid result; its first paired sample is ordinary (10.577 us versus 10.279 us), with no startup outlier after fixing event ordering.

Resource usage from `cuobjdump --dump-resource-usage results/exp056/libexp056.so`: paired kernel 50 registers/thread, 1,024 bytes shared memory, no stack/local memory; current single kernel 98 registers/thread, no shared memory. Reduced register use and one fewer graph node were insufficient to offset paired-kernel execution cost.

The harness loaded `libexp056.so` from this experiment's result directory and resolved `libggml-cuda.so.0` from the Exp053 sm_86 build. Its SHA-256 was `bc8ce63fe830d5b3a431113ca33e6512ba936a27f3c50b11e3f557b0099b98d3`; the exact `ldd` resolution and binary/library hashes are in `results/exp056/ldd_bench_pair.txt` and `sha256sums.txt`. The CUDA backend source for the referenced runtime matches the production source; no model runtime A/B used that library.

## END-TO-END IMPACT

Not measured. The faithful graph replay screen lost, so scheduler integration and full-model A/B were not justified.

## ANALYSIS

The sequential event screen's apparent gain came from ordinary stream-launch submission behavior and did not represent the graph-replay steady state. Under graph replay, launch scheduling was already captured, and the paired CTA mapping incurred enough work to lose despite one fewer node and fewer registers. The correct decision is to stop this design before graph matching, dispatch guards, and output-lifetime integration.

## DECISION

**REJECT THIS PAIRED CTA MAPPING; DO NOT INTEGRATE.** Output correctness passed, but CUDA Graph replay was 3.7% slower by the median. Production source has been restored to baseline in the isolated worktree. Main checkout remains untouched.

## FOLLOW-UPS

None for this paired-output design. Revisit only with a materially different paired CTA mapping or dataflow premise.

## IMPORTANT DISCOVERIES

- A direct paired-output PTQ1_0 kernel using the existing dot helper is implementable and numerically tracks two current kernels closely.
- Its 50-register, 1-KiB-shared kernel and one graph node did not beat two 98-register single-kernel nodes; it lost by 3.7% in corrected graph replay.
- Event records for graph timing must use the captured nonblocking stream. Omitting the stream argument produced misleading timing because the events landed on the legacy default stream.

## Commands and artifacts

The CUDA translation unit was compiled with the sm_86 command template in `/home/maxsun/autonomous_projects/.worktrees/exp053-flash-attention/build/compile_commands.json`, substituting this worktree's `ggml/src/ggml-cuda/mmvq.cu` and output `results/exp056/mmvq-exp056.o`; it was linked as `results/exp056/libexp056.so`. The test harness command was:

```bash
nvcc -O3 -std=c++17 -arch=sm_86 results/exp056/bench_pair.cu \
  -Lresults/exp056 -lexp056 \
  -L/home/maxsun/autonomous_projects/.worktrees/exp053-flash-attention/build/bin \
  -lggml-cuda -lggml-base \
  -Xlinker -rpath -Xlinker /home/maxsun/autonomous_projects/.worktrees/exp056-kv-pair-screen/results/exp056 \
  -Xlinker -rpath -Xlinker /home/maxsun/autonomous_projects/.worktrees/exp053-flash-attention/build/bin \
  -o results/exp056/bench_pair
./results/exp056/bench_pair
```

Raw sequential timings: `results/exp056/run.txt`. Correct CUDA Graph timings: `results/exp056/graph-run-corrected.txt`. Invalid early graph captures, explicitly superseded above: `graph-run.txt` and `graph-run2.txt`. Candidate source patch: `paired_kernel_patch.diff`; the extracted SASS resource summary is `resource_usage.txt`. No commit was created. The only remaining untracked paths in the isolated worktree are this report and `results/exp056/`; `ggml/src/ggml-cuda/mmvq.cu` and `mmvq-ptq1_0.cuh` match HEAD.
