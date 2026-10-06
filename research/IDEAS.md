# Candidate hypotheses

Ranked after experiment 020 and the ROWS=1 profile. Judge each candidate by controlled end-to-end decode and combined throughput.

1. Tune the active GDN S_v=128 scalar raw-gate (`KDA=false`, `RAW=true`) column mapping on sm_86. The mixed trace attributes 4.6% to GDN, and the current kernel maps four columns per warp with four warps per CTA. Sweep practical column groupings, preserve numerical behavior, and verify the actual full-model decode path. The runtime already has fused recurrent-state gather/cache paths; first confirm which are active.
2. Profile standalone RMSNorm (3.9%) and optimize only measured launch or data-movement overhead.
3. Periodically challenge the active PTQ1_0 GEMV design with a substantially different approach. Experiments 015, 017, and 018 rejected fixed and shape-gated multiwarp row reductions; do not repeat them without a changed architectural premise.
4. Revisit PQ2_0 activation fusion after PTQ1_0's active GDN/RMS paths.

Experiments 019–020 exhausted the current per-branch RMS→FWHT fusion and FWHT CTA-width sweeps: normalization output is shared across Q/K/V, and NT=128 regressed while NT=512 tied with NT=256. Do not repeat either idea without a concrete design change.

Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, and controlled size/workload sweeps.
