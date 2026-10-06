# Candidate hypotheses

Ranked after experiment 019 and the ROWS=1 profile. Judge every candidate by controlled end-to-end decode and combined throughput.

1. Optimize the active PTQ1_0 fused FWHT/Q8_1 kernel (4.4% of the post-ROWS=1 mixed trace), especially the PT Q8 layout and transform-block scheduling; require full model decode impact.
2. Investigate gated-delta network work/data movement (4.6%), checking whether the existing gather fusion is active before proposing changes.
3. Profile the standalone RMSNorm path (3.9%) and optimize only where launch or data movement remains exposed.
4. Treat active PTQ1_0 multiwarp row reduction as exhausted for now: experiments 015, 017, and 018 all failed to show E2E gain.

Experiment 019 found that the model's attention norm output fans out into Q/K/V projection construction, while the FWHT/Q8_1 path is already fused. Do not repeat a single-branch RMS→FWHT fusion proposal; a multi-branch design must account for shared normalization and its reduction cost.

Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, and controlled size/workload sweeps.
