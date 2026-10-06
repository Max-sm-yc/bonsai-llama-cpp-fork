# Candidate hypotheses

Ranked after experiment 018 and the ROWS=1 profile. Judge every candidate by controlled end-to-end decode and combined throughput.

1. Investigate fusion across PTQ1_0 RMSNorm and the following FWHT/Q8_1 activation-preparation path, which account for 3.9% and 4.4% of the mixed trace. Confirm graph/operator ordering and exact math before coding; benchmark full decode.
2. Revisit PQ2_0 activation fusion or gated-delta/RMSNorm work if fusion is blocked or if a new profile raises their measured share.
3. Treat active PTQ1_0 multiwarp row reduction as exhausted for now: experiments 015, 017, and 018 all failed to show E2E gain.

Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, and controlled size/workload sweeps.
