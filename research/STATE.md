# Research state

## Current best

- Project baseline commit `2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c` (unchanged PrismML source `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`); PTQ1_0. Median decode: 46.88/46.09/42.82/39.21 tok/s at contexts 128/512/2048/4096. Prefill: 1292/1378/1355/1331 tok/s. Peak whole-GPU use 6805 MiB. See `results/baseline.json`; 7 repetitions, 60°C format start gate, then prefill/decode/combined in sequence.
- Correctness on the restored baseline, rerun after experiment 001: 4/4 upstream tests, 96/96 CUDA-vs-CPU ternary matmul cases, and both CUDA model smoke runs pass.

## Bottlenecks

1. Ternary batch-1 GEMV kernels: 61.8% of PTQ1_0 and 62.6% of PQ2_0 GPU kernel time in the context-512 Nsight Systems trace.
2. Ternary GEMM: 12.1% PTQ1_0, 11.1% PQ2_0.
3. Recurrent gated-delta attention, Hadamard/Q8_1 preparation, and RMSNorm: about 4% each.

## Successful optimizations

- None yet. Upstream reference paths are the baseline.

## Failed or exhausted approaches

- The first format comparison started PTQ1_0 cool and PQ2_0 hot; it is retained as `results/baseline_initial_uncontrolled.json` but excluded from decisions.
- Experiments 001/002 disabled PTQ1_0's L2 prefetch; the paired 128-token workloads tied within 0.04%, and long-run tail effects reversed with process order and broad clock variation.
- Experiment 003 changed the generic one-column MMVQ warp count, but the target sm_86 runtime routes batch-1 PTQ1_0 through the dedicated planar-transposed kernel and bypasses that code. Its decode measurements compared the same active kernel; they are a no-op check, not evidence about active-kernel geometry. See report 003's manager audit.
- Experiment 004 tested a constant-memory lookup table for PTQ1_0 `qs` trit expansion. It matched the multiply decoder but took 5.89x longer in a focused 120-trit dot kernel; no production change or E2E run. The test excluded `qh` and production activation layout, so it rejects this direct LUT design only.
- Experiment 005's 2-bit side-format screen is inconclusive: it wrote 128 bytes through a 32-byte packed-code field (overrunning adjacent records) and used the wrong PTQ1_0 element order. No valid packed conversion was tested, so its timing is invalid; no runtime integration or model benchmark occurred. The proposed block grows from 28 to 34 bytes (+21.43%).
- Experiment 006 corrected side-code packing and element mapping and passed exact code/dot checks, but its purported SOA_ISUM address is wrong (`kb=b>>2`, `sub=b&3` instead of `group=b>>5`, `lane=b&31`, `word=e>>2`). Its 18.4–18.8% slower timing is inconclusive; no runtime integration or E2E test occurred. Correct and remeasure only if this remains a priority. Side payload is +21.43%.
- Experiments 005–007 screened 2-bit side codes in the warp-transposed SOA_ISUM harness. On sm_86, batch-1 PTQ1_0 instead uses the planar-transposed layout; the slowdown rejects the tested SOA dot only and does not settle side-code performance in the active RTX 3080 kernel. See report 007's manager audit.
- Experiments 008/009 tested floor-difference decoding in the SOA_ISUM harness. Experiment 008's `qh` packing failed; experiment 009 fixed the interleave, passed exhaustive device/full-block/sanitizer checks, and remained 1.24–7.04% slower in that harness. This does not establish end-to-end impact in the active sm_86 planar kernel. See report 009's manager audit.
- Nsight Compute counters are blocked by `ERR_NVGPUCTRPERM`; do not change system-wide driver permissions. Nsight Systems and static cubin resource reports are available.

## Important discoveries

- Under matched conditions, PTQ1_0 decode is faster than PQ2_0 on this RTX 3080 by 32–54%, while prefill is nearly tied. This differs from the model card's broad Ampere result, which has no RTX 3080 row.
- PTQ1_0's three dominant specialized GEMV variants take about 140 ms less total in the Nsight trace than PQ2_0's corresponding variants. Two variants are 13–16% faster per launch; one is 2.5% slower. The 17.6% lower PTQ1_0 payload is consistent with a weight-traffic advantage, but NCU bandwidth/instruction counters are unavailable.
- The hot PTQ1_0 GEMV variants compile to 106–126 registers/thread with no local spills. PTQ1_0 also fuses Hadamard and Q8_1 quantization; PQ2_0 uses separate kernels.
- The baseline matrix temperature gate runs once per format, not once per workload. Isolated decode processes at <=62 C start measured about 76–78 tok/s, unlike the hotter baseline matrix. In 512-token, 4096-context runs, process tails varied strongly with pair order and SM clocks; compare matched isolated runs and capture per-process telemetry.
- On RTX 3080 / sm_86, `ggml_cuda_q8_1_layout_host` selects `GGML_CUDA_Q8_1_PT`; plain one-column PTQ1_0 `MUL_MAT` dispatches to `mul_mat_vec_ptq1_0_pt`, a dedicated 128-thread CTA kernel. Its `ptq1_0_pt_block_dot` reads the planar-transposed activation planes. The generic `calc_nwarps` change in experiment 003 is bypassed on this path.
- SOA_ISUM activation addressing for PTQ1 K-block `b` and element `e` is `group=b>>5`, `lane=b&31`, `word=e>>2`, byte `e&3`; groups stride 32*36 words. This applies to the SOA path, used for one-column on Ada and newer; do not treat it as the target sm_86 layout.
- Both files fit at 4096 context with F16 KV. Peak whole-GPU memory is 6805 MiB PTQ1_0 and 7949 MiB PQ2_0.

## Next candidates

1. Tune the active sm_86 `mul_mat_vec_ptq1_0_pt` kernel's rows-per-item/CTA work mapping on RTX 3080, with exact planar Q8 layout, CUDA correctness, and matched end-to-end decode measurements.
2. Measure whether Hadamard/Q8_1 fusion benefits PQ2_0; keep secondary to the faster PTQ1_0 decode path.
