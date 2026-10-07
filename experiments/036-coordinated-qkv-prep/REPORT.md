# Exp036: coordinated Q/K/V activation preparation

## HYPOTHESIS

The Qwen35 graph computes weighted RMSNorm, applies a sign vector and shared 1024-wide FWHT, then quantizes the transformed activation to Q8_1 for PTQ1_0 matvecs. In full-attention layers Q, K, and V consume the same transformed activation; the graph memoizes the transform, and CUDA caches its Q8_1 result. A specialized kernel could therefore fuse the one shared RMS scale with sign/FWHT/Q8_1 preparation and remove intermediate norm, sign, and transform work. It must preserve other raw RMS consumers and existing branch outputs.

## IMPLEMENTATION

Implemented an opt-in candidate only in detached worktree `/tmp/bonsai-exp036` (based on main HEAD `8eb4582`), built for sm_86 with its own CUDA library. `GGML_CUDA_RMS_FWHT_Q8=1` enables graph recognition for the exact supported RMS → weighted multiply → sign multiply → reshape → Hadamard path with width 5120, block 1024, contiguous F32 inputs, and PT Q8_1 output. The graph matcher requires exclusive use of the candidate intermediates; otherwise it falls back to the normal operations, preserving raw norm consumers and unsupported shapes. The kernel uses five 1024-wide transform CTAs per row; each CTA independently computes the shared 5120-wide RMS reduction, then emits its transform tile in the existing PT layout. Existing Q8 cache aliases remain responsible for fan-out to all projection consumers.

The candidate changes are uncommitted and isolated. Candidate CUDA library SHA-256: `94efaa23a18986647ce59ed811059fead1fa0ff6d84f0a762f7725f9e5ed2392`; candidate `llama-bench`: `bc8d64ee3bf8e35a611e0cb566147e8e9ff5506323300a449c3597d106766b6e`.

## RESULT

The candidate dispatched in a real PTQ1_0 model run and passed correctness. The experimenter captured two matched-order pairs, then the manager independently repeated two more pairs using the current production binary as control and the isolated candidate binary. Across all four pairs, the median-of-pair-medians improved 1.68% at context 512 and 1.06% at 4096, with no material VRAM change. All four 512 pairs and three of four 4096 pairs favored the candidate; the remaining manager-forward 4096 pair was -0.72% with multiple candidate slow-tail samples. The production checkout remained unchanged during those measurements.

## CORRECTNESS

- Candidate `tests/run_correctness.sh`: CTest 4/4 passed; CUDA backend-op suite 96/96 passed; PTQ1_0 and PQ2_0 fixed-seed model smokes completed.
- Explicit candidate `python3 tests/model_smoke.py --output results/exp036/model_smoke.json` completed for both model formats. Compared with a source-default binary run under the same seed/configuration, generated PTQ1_0 text matched exactly; only run metadata, paths, and timing differ.
- Nsight Systems of the candidate 32-token PTQ1_0 smoke recorded 240 `fwht_rms_quantize_q8_1<(1024,1024,PT)>` kernel instances (945,224 ns total, 3,938 ns average). This verifies the specialized dispatch is reached. The trace also records 531 existing `fwht_quantize_q8_1<(1024,256,PT)>` instances for other paths/shapes; it is not a one-for-one count comparison.
- Evidence: `results/exp036/ctest.log`, `backend_ops.log`, `model_smoke.json`, `control_model_smoke.json`, `dispatch_smoke.nsys-rep`, and `dispatch_kernel_summary.txt`.

## MICROBENCHMARK

No standalone kernel-only result was used to make the decision. The targeted kernel averaged 3.94 μs in the model trace; its total cost must be judged in the full graph and decode workload.

## END-TO-END IMPACT

All runs used the same PTQ1_0 model, RTX 3080 sm_86, 99 GPU layers, FA on, F16 KV, batch/ubatch 2048/512, 8 CPU threads, 128 decode tokens, seven repetitions, and contexts 512/4096. First pair ran control then candidate; second pair reversed that order. The JSON files contain GPU samples and raw repetition samples.

| Order | Arm | ctx 512 median tok/s (samples) | ctx 4096 median tok/s (samples) | Peak GPU memory |
|---|---|---|---|---:|
| Forward | control | 82.1222 (`81.3229, 82.3079, 82.2825, 82.1712, 82.0363, 82.0934, 82.1222`) | 79.7118 (`78.9585, 79.7244, 79.7575, 79.7253, 79.7118, 79.5931, 79.5613`) | 6805 MiB |
| Forward | candidate | 83.4146 (`82.5048, 83.4958, 83.4778, 83.3692, 83.4377, 83.4146, 83.3926`) | 80.8443 (`80.1827, 80.9170, 80.9335, 80.8443, 80.8615, 80.8182, 80.6202`) | 6803 MiB |
| Reversed | candidate | 83.3497 (`82.4431, 83.3810, 83.3705, 83.3574, 83.3497, 83.3262, 83.2144`) | 80.0209 (`80.0209, 80.7444, 80.6624, 80.3082, 77.9687, 61.7936, 78.4595`) | 6803 MiB |
| Reversed | control | 81.6803 (`80.9135, 81.8414, 81.8130, 81.7920, 81.6803, 81.6676, 81.6392`) | 78.5532 (`78.5532, 79.2233, 79.2497, 78.6448, 75.1186, 71.2983, 69.8019`) | 6805 MiB |
| Manager forward | control | 82.0750 (`81.1775, 82.0856, 82.0804, 82.0750, 82.0815, 82.0478, 81.8852`) | 79.4663 (`78.7506, 79.5389, 79.4933, 79.4231, 79.4918, 79.4663, 79.4516`) | 6805 MiB |
| Manager forward | candidate | 83.2365 (`82.4073, 83.3095, 83.3365, 83.2365, 83.2771, 83.1375, 83.1110`) | 78.8913 (`79.9362, 80.6090, 80.2570, 78.8913, 75.5500, 66.9787, 64.3891`) | 6803 MiB |
| Manager reversed | candidate | 83.2693 (`82.3084, 83.3209, 83.3059, 83.2693, 83.2955, 83.1527, 83.0927`) | 79.6765 (`79.9112, 80.6690, 80.5843, 79.6765, 76.0330, 67.0854, 64.1950`) | 6803 MiB |
| Manager reversed | control | 81.7922 (`80.8767, 81.8221, 81.7987, 81.8003, 81.7922, 81.7806, 81.6902`) | 78.5112 (`78.5112, 79.1971, 79.0543, 78.5758, 61.3194, 61.7754, 61.5556`) | 6805 MiB |

Using the median of the four pair-specific medians per arm: at context 512, control 81.9336 vs candidate 83.3095 tok/s (+1.68%); at context 4096, control 79.0098 vs candidate 79.8487 tok/s (+1.06%). The original two pairs alone were +1.81%/+1.64%. The manager-forward deltas were +1.42% at 512 and -0.72% at 4096; its candidate 4096 samples included 75.55, 66.98, and 64.39 tok/s, while the reversed manager control had three slow tails. All tails are retained in JSON and the table. Peak GPU memory was 6803 MiB candidate versus 6805 MiB control.

Raw benchmark artifacts are `control_forward.json`, `candidate_forward.json`, `candidate_reverse.json`, `control_reverse.json`, `manager_control.json`, `manager_candidate.json`, `manager_candidate_reverse.json`, and `manager_control_reverse.json` in `results/exp036/`. The manager pairs were run independently after checking the candidate binary's RUNPATH and comparing normalized fixed-seed PTQ1_0 output to the production control.

## ANALYSIS

The graph audit confirms why a single-projection norm fusion did not establish this opportunity: Q/K/V share a memoized transformed activation, and its Q8_1 representation is already reused. The coordinated candidate targets the actual shared path and avoids tripling transform outputs. Five CTAs preserve parallelism across the five 1024-element FWHT blocks, at the cost of each CTA rereading the 5120-element input for its RMS reduction (five CTA reads, about 100 KiB logical input traffic per row). Despite that redundancy, end-to-end medians improved in both process orders and contexts. The 2 MiB peak-memory difference is within run-level sampling and does not indicate a material allocation change.

## DECISION

**KEEP.** The manager independently verified the correctness evidence, exact normalized completion, candidate RUNPATH, and four isolated candidate/control pairs. The aggregate gain is modest but positive at both tested contexts; 4096 remains noisier and one of four isolated pairs regressed slightly. The implementation has since been promoted, rebuilt, and verified in the main checkout. It is enabled by default; `GGML_CUDA_RMS_FWHT_Q8=0` disables the new path for controls.

## FOLLOW-UPS

Keep the exact-shape/use-count guards and the disable override. Re-profile after the end-to-end gain, then challenge a different high-cost part of decode. The new profile confirms PTQ1_0 GEMV remains dominant; the fused preparation kernel accounts for about 39 ms in the mixed context-512 trace, while the RMSNorm family falls from 100.6 ms to 56.8 ms relative to the ROWS=1 trace. Five CTAs still repeat RMS reads, so further changes to this fusion need new end-to-end evidence.

## IMPORTANT DISCOVERIES

- Full-attention Q, K, and V use the same `(activation, rotation)` memoized FWHT node. CUDA’s transform-to-Q8 path already caches one Q8_1 activation for PTQ consumers.
- The model input width is 5120, with a 1024-element Hadamard block and a 5120-wide sign vector. The current candidate is deliberately specialized to this exact geometry.
- Shared transformed activation is not evidence that RMS output has no raw consumers in every graph. Candidate dispatch checks use counts and falls back if those intermediates fan out elsewhere.
- Main production source hashes remained: `ggml-cuda.cu` `2ae527aba2b42f6658a2b856c605a8d40e2f3e1ee8c37a3448cb76c409b28e5e`; `qwen35.cpp` `056ae5e71776e1cf54d7d3eb48f34eeafa3a6d7eae2f9130044585b67c06625a`; `llama-graph.cpp` `03b2f05258be1025dd99704f88e2da22f7595f797111996e572ac1b53c16c51a`. Main `build/bin/libggml-cuda.so` remained SHA-256 `c828135b126ec507ffbecb4dc11b6a7a9ac5cd0fe050553323d7f35c38fae6c7`.

## MANAGER PROMOTION AND SAME-BINARY VERIFICATION

The change was promoted in code commit `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`. The main-tree path is enabled when `GGML_CUDA_RMS_FWHT_Q8` is unset and can be disabled with `GGML_CUDA_RMS_FWHT_Q8=0`. A fresh main-tree build completed; `tests/run_correctness.sh` then passed CTest 4/4, CUDA-vs-CPU backend ops 96/96, and fixed-seed PTQ1_0/PQ2_0 model smokes. The main-tree PTQ1_0 completion exactly matched the previous normalized completion. See `main_tree_smoke.json` and `main_tree_verification.md`.

The manager then used the same main `llama-bench` binary for two 7-repetition A/B pairs, changing only the environment override and reversing arm order. Context-specific results are medians of the two run medians:

| Order | Disabled context 512/4096 | Enabled context 512/4096 | Delta 512/4096 | Peak memory |
|---|---:|---:|---:|---:|
| Disabled then enabled | 82.1918 / 79.7605 | 83.3770 / 80.7197 | +1.44% / +1.20% | 6805 / 6803 MiB |
| Enabled then disabled | 81.7960 / 78.4931 | 83.3145 / 79.9846 | +1.86% / +1.90% | 6803 / 6805 MiB |
| Median of pair medians | 81.9939 / 79.1268 | 83.3458 / 80.3522 | +1.65% / +1.55% | 6803 / 6805 MiB |

All four same-binary context comparisons favored the enabled path. The reverse 4096 runs had slow tails in both arms (enabled minimum 58.72 tok/s, disabled minimum 65.76); those raw samples remain in the JSON and are not hidden by the medians. The same-binary pairs are the primary promotion evidence; the earlier four isolated binary pairs remain supporting evidence. Main-tree A/B records: `main_default_off.json`, `main_default_on.json`, `main_default_on_reverse.json`, and `main_default_off_reverse.json`.

Main-tree source hashes after promotion: `ggml-cuda.cu` `aee803e29b853a70d3e5274606cad32cdefc93d72578b1824df259dd2ba86351`, `quantize.cu` `dd55176be1e6639d102d1d18bf3644795fa0ae73a85bd5ef26188ba8ec59e03e`, and `quantize.cuh` `cef94b4f946ec874fa55544f7eb00e20096e36a91cb3bdd00eed7e1a5e24b719`. The rebuilt main CUDA library SHA-256 is `4b4adb58e3d26cb8694aebf0843112b981de66ac54441760ec8290bb2b021dcf`.
