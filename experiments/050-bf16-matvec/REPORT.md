# Experiment 050: BF16 batch-1 matvec audit

## HYPOTHESIS

The recurring BF16 matvec family might have a distinct low-cost sm_86 geometry or dataflow improvement. It is a secondary target: Exp047 measured 0.311 ms/token at context 512 and 0.306 ms/token at 4096, about 2.6% and 1.3% of summed graph kernel time.

## IMPLEMENTATION

No source change. Work was done in detached worktree `/tmp/exp050-bf16-matvec` at manager HEAD `b4d14820a66669db485bc65eeaf5e34f627611e2`.

## RESULT

**NO CANDIDATE.** The model-specific BF16 `ssm_alpha` and `ssm_beta` projections have 48 output rows and K=5,120, as documented in the backend cases (`tests/test-backend-ops.cpp`). For each such projection, source launch geometry is `(nrows,nchannels_dst,nsamples)`, which is `(48,1,1)` in batch-1 decode: 48 CTAs in one kernel launch. There is no gate/bias fusion in the active signature (`false,false`).


## INVOCATION COUNTS AND SHAPES

Exp047's `cuda_gpu_kern_sum` CSV reports **24,672 kernel instances/launches**, not CTAs. Its capture has exactly 255 `cudaGraphLaunch` calls; the retained per-replay CSV/JSON confirm 255 graph replay groups with 1,432 graph nodes apiece, but do not provide per-replay counts for this individual signature. Dividing the aggregate stats count by 255 gives 96.75 instances/replay as a normalization only; it does not establish the number of this kernel's graph nodes per replay because the summary count may also include non-graph capture/setup launches. If the `(48,1,1)` shape applies to all 24,672 instances, that is 1,184,256 CTA executions over the full stats capture. The two named projection types and their 48x5120 shapes are the supported model mapping; exact layer-by-layer invocation contributions are not recoverable from the retained compact replay artifacts.

Each CTA owns one output row, so a 48-row projection launch places 48 CTAs on a 68-SM RTX 3080. Reducing block width does not create more independent CTAs and raises per-thread K work; splitting K would need cross-CTA partial reduction/atomics and extra coordination for a small 0.31-ms family. The active code already loads paired BF16 and F32 values, accumulates in float, and reduces within warps followed by a shared-memory inter-warp fold. No low-cost geometry/dataflow premise survived this source/codegen audit, so no variant was built or timed.

## CORRECTNESS

No candidate was implemented; no correctness behavior changed. Existing backend tests include BF16/F32 matmul cases for the same 48x5120 `ssm_alpha`/`ssm_beta` projections. The full suite was not run.

## MICROBENCHMARK

Matched focused baseline is the Exp047 direct CUDA Graph replay profile: 79,212,644 ns / 24,672 kernel instances at context 512 (3,210.6 ns mean per kernel instance) and 77,952,315 ns / 24,672 at context 4096 (3,159.5 ns mean). These are profile measurements, not new unprofiled timing samples. No candidate timing was warranted after the shape/codegen audit.

## END-TO-END IMPACT

Not measured. This family is 2.6% of context-512 graph kernel time and 1.3% at context 4096; there is no candidate to justify model A/B.

## ANALYSIS

Source inspection (`mmvf.cu`) confirms the kernel distributes `col2` by `threadIdx.x` with stride `block_size`, loads `nv_bfloat162` and `float2`, accumulates two scalar FMAs per pair, performs warp reduction, then writes warp partials to shared memory and folds them. The launch chooser selects 256 threads for K=5,120 to minimize loop iterations; the active specialization is `ncols_dst=1`, `block_size=256`, no fusion. For a 48-row, one-channel, one-sample projection, its grid is `(48,1,1)` per kernel launch. Exp047 artifacts prove 24,672 aggregate kernel instances for the signature and 255 graph launch calls, not a per-replay family invocation count.

The active cubin is from the verified current manager library, SHA-256 `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`. `cuobjdump` identifies the sm_86 target function with 39 registers/thread and no stack/local spills. SASS inspection confirms the specialization has ordinary global loads, FP32 fused multiply-add accumulation, warp shuffle reduction, shared partial storage and a CTA barrier; it contains no tensor-core operation. Artifact `results/exp050/codegen_audit.txt` records the codegen query and concise findings. Nsight Compute counters remain unavailable per Exp047 (`ERR_NVGPUCTRPERM`).

## DECISION

No implementation, model A/B, or build. Preserve production behavior and leave the active CUDA library unchanged. Keep end-to-end decode throughput as the optimization decision metric.

## FOLLOW-UPS

Reopen only if a future compiler/library change exposes a cheaper way to create more parallel row work or removes the inter-CTA K-split cost. A single 48-row kernel launch has fewer CTAs than the device has SMs; the available compact profile does not establish the layer-by-layer count of such launches per replay. No geometry variant was justified for this audit.

## IMPORTANT DISCOVERIES

- The profile's `Instances` field counts kernel launches. It is 24,672 over the capture; dividing by 255 graph launches gives an aggregate normalization of 96.75 instances/replay, not a verified per-replay count for this family. A `(48,1,1)` kernel grid means 48 CTAs for each instance.
- The shape audit points to BF16 `ssm_alpha` and `ssm_beta`, not the large PTQ1_0 attention/FFN projections.
- Both profile contexts exercise the same active template count; total family time is 79.2 ms versus 78.0 ms across 255 replay groups.
