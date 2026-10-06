# Candidate hypotheses

Ranked after experiment 010 and the ROWS=1 re-profile. Judge every candidate by controlled end-to-end decode and combined throughput.

1. Test a compact 2-bit side-code decoder against the actual planar-transposed activation layout. Previous side-code tests used SOA and do not settle the sm_86 path. Include model-load repacking, extra 21.43% weight storage, peak VRAM, and end-to-end decode in the comparison.
2. Try a distinct trit unpack/reduction mapping inside the active planar `mul_mat_vec_ptq1_0_pt` kernel; the earlier LUT and floor paths were tested in SOA harnesses only.
3. Revisit Hadamard-to-Q8_1 fusion for PQ2_0; then inspect recurrent attention, RMSNorm, and launch gaps if their share rises after PTQ1 improvements.

Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); do not change system-wide driver permissions. Use Nsight Systems, static cubin resources, and controlled size/workload sweeps.
