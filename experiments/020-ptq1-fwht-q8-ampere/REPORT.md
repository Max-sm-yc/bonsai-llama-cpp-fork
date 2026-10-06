# Experiment 020: PTQ1_0 FWHT/Q8_1 CTA mapping on sm_86

## HYPOTHESIS

The fused Hadamard-to-Q8_1 kernel still accounts for about 4.4% of the post-ROWS=1 mixed trace. On RTX 3080/sm_86, changing its CTA width while holding transform and arithmetic fixed could improve activation preparation and real decode throughput.

## IMPLEMENTATION

Tested the existing fused `fwht_quantize_q8_1` kernel with transform width fixed at N=1024 and the PTQ1_0 planar-transposed stores unchanged. The existing NT=256 mapping was screened against NT=128 (eight register elements per thread at N=1024) and NT=512 (two elements per thread). These variants changed only the thread count and the existing `NE=N/NT` work mapping; the butterfly, scale, quantization, reductions, and stores were unchanged.

The current model’s output-projection activation has shape `(17408, 1, 1, 1)` and F32 type at the fused call; the explicit signs vector has width 17,408. GGUF metadata gives `prism.hadamard.block_size=1024` and sign width 17,408. The fused dispatch therefore uses N=1024, NT=256 in control, ncols=1, K=ne00=17,408, and padded ne0=17,408. The sm_86 host selector explicitly selects `GGML_CUDA_Q8_1_PT` for the one-column PTQ1_0 case. This keeps the same row stride and planar-transposed layout read by the PTQ1_0 matvec consumers.

The temporary NT cap parameter was removed. The final tracked source is restored byte-for-byte. Candidate libraries were copied to `results/exp020/lib_nt128` and `lib_nt512`; the matched NT=256 control used for all A/B runs is in `lib_control`. Candidate and control processes were isolated with `LD_LIBRARY_PATH`, `.so.0` symlinks, and `LD_DEBUG=libs`; the six child stderr logs under `results/exp020/raw/bench_stdout/` show each process loading the intended `libggml-cuda.so.0`. `results/exp020/loader_audit.txt` extracts that mapping, and `loader_*.txt` records `ldd` resolution for each library directory.

The A/B control library SHA-256 is `23bd3d1201b2378ef0e343a313e1a01651d5749a7a155af02180c30f8d565811`; it was built from the temporary NT-cap source with the cap set to 256, so its generated host specializations use the same NT=256 code. The final active library rebuilt after restoring the exact tracked source has a different SHA-256 (`708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`) and a different ELF build ID. `nm -C` output for the N=1024, NT=256 host specializations matches exactly between the A/B control and final library (`raw/control_fwht_host_symbols.txt` and `raw/final_fwht_host_symbols.txt`). I do not claim byte-for-byte binary identity. The A/B results apply to the explicitly logged control artifact; the active library is the fresh build from the restored source.

The child `LD_DEBUG=libs` logs show the intended candidate/control `libggml-cuda.so.0` dependency initialized. They also show a separate probe of `build/bin/libggml-cuda.so` that cannot find `ggml_backend_score` / `ggml_backend_init`. This build has `GGML_BACKEND_DL=OFF`; source inspection shows CUDA is registered directly via `ggml_backend_cuda_reg()` under `GGML_USE_CUDA` in `ggml-backend-reg.cpp`, and each benchmark JSON reports `backends: CUDA`. Both A/B arms share this plugin-probe behavior; the linked dependency used for each arm is the explicitly selected library. The compact per-process evidence is committed in `results/exp020/loader_audit.txt`; the full child stderr logs remain locally available but are ignored as raw logs.

## RESULT

**No repeatable end-to-end gain. REVERT.** NT=128 clearly regressed. NT=512’s two reversed-order pairs tied within normal decode noise; its apparent first-pair long-context gain was due to slow-tail samples in the control process.

All runs used RTX 3080/sm_86, PTQ1_0, F16 KV, Flash Attention, 99 GPU layers, batch/microbatch 2048/512, eight CPU threads, 128 decode tokens, seven repetitions, and a <=60°C/<=5% utilization start gate. Candidate/control library hashes and every JSON sample are preserved under `results/exp020/`.

Median throughput, with mean ± sample SD and full sample range:

| Variant / pair order | Context | Candidate median; mean ± SD; range (tok/s) | NT=256 control median; mean ± SD; range (tok/s) | Median delta |
|---|---:|---|---|---:|
| NT=128; candidate then control | 512 | 81.1476; 81.0336 ± 0.3109; 80.3464–81.2254 | 81.9661; 81.8066 ± 0.3698; 80.9816–81.9979 | -1.00% |
| NT=128; candidate then control | 4096 | 78.6971; 78.5891 ± 0.2597; 78.0086–78.7366 | 79.4136; 79.2233 ± 0.3479; 78.7085–79.4540 | -0.90% |
| NT=512; candidate then control | 512 | 82.0462; 81.9660 ± 0.3018; 81.2991–82.2053 | 81.7084; 81.6031 ± 0.3109; 80.9123–81.8475 | +0.41% |
| NT=512; candidate then control | 4096 | 79.5969; 79.5177 ± 0.2576; 78.9443–79.7042 | 78.6266; 74.4586 ± 7.2898; 61.0149–79.3658 | +1.23% |
| NT=512; control then candidate | 512 | 81.8397; 81.7315 ± 0.3456; 80.9652–81.9425 | 81.7174; 81.6198 ± 0.3601; 80.8171–81.8630 | +0.15% |
| NT=512; control then candidate | 4096 | 78.6604; 75.6760 ± 6.3140; 62.3841–79.4227 | 78.6099; 73.6660 ± 8.6892; 56.5916–79.3543 | +0.06% |

Exact samples, in each row’s candidate/control order:

- NT=128, context 512: candidate `[80.3464, 81.2124, 81.1895, 81.2254, 81.1476, 81.0725, 81.0415]`; control `[80.9816, 81.9801, 81.9979, 81.9804, 81.9661, 81.9390, 81.8012]`.
- NT=128, context 4096: candidate `[78.0086, 78.7366, 78.7095, 78.7146, 78.6971, 78.6498, 78.6076]`; control `[78.7085, 79.4540, 79.4375, 79.4197, 79.4136, 79.4092, 78.7205]`.
- NT=512 pair 1, context 512: candidate `[81.2991, 82.2053, 81.9856, 82.0958, 82.0887, 82.0411, 82.0462]`; control `[80.9123, 81.8475, 81.7148, 81.6823, 81.7099, 81.6465, 81.7084]`.
- NT=512 pair 1, context 4096: candidate `[78.9443, 79.7042, 79.6472, 79.5750, 79.5982, 79.5969, 79.5579]`; control `[78.6266, 79.3658, 79.1899, 78.9037, 76.6890, 67.4202, 61.0149]`.
- NT=512 pair 2, context 512: candidate `[80.9652, 81.9375, 81.9425, 81.8980, 81.8397, 81.7771, 81.7608]`; control `[80.8171, 81.8630, 81.8081, 81.7648, 81.6944, 81.7174, 81.6739]`.
- NT=512 pair 2, context 4096: candidate `[78.6604, 79.4227, 79.4123, 78.9882, 62.3841, 78.1279, 72.7366]`; control `[78.6099, 79.3543, 79.3228, 78.7024, 75.9155, 67.1658, 56.5916]`.

Raw paired benchmark files are `nt128_pair1_{candidate,control}.json` and `nt512_pair{1,2}_{candidate,control}.json`; process logs, telemetry, and captured llama-bench stdout JSON are in `results/exp020/raw/`.

## CORRECTNESS

- Source-default NT=256 `bash tests/run_correctness.sh` passed all four selected CTests, 96/96 CUDA-vs-CPU PTQ1_0/PQ2_0 matmul cases, and fixed-seed PTQ1_0 and PQ2_0 model smokes.
- NT=128 and NT=512 each passed the same four selected CTests, 96/96 CUDA-vs-CPU matmul cases (including K=17,408 and n=1/2/4/8), and both model smokes.
- The fixed-seed 32-token PTQ1_0 and PQ2_0 outputs for both candidates match the NT=256 outputs exactly after trimming the final newline and normalizing only the timing line. Captured outputs and generated JSON are under `results/exp020/raw/model_outputs/` and `model_smoke_nt*.json`.
- The mapping variants preserve the exact FWHT butterfly and quantization/store code. Model output equality exercises the actual fused N=1024 decode path and its PT stores; backend cases exercise the PTQ1_0 consumers at K=17,408.

## MICROBENCHMARK

No separate kernel microbenchmark was run. The repository has no focused fused-FWHT/PT-output reference harness; screening used the real decode path directly. Per-process telemetry and exact samples are retained in the benchmark JSON files. No profiler-only timing was used to claim a gain.

## END-TO-END IMPACT

NT=128 lost about 0.9–1.0% in its candidate-first pair. NT=512’s context-512 gain shrank from +0.41% to +0.15% in reversed order. At context 4096 the reversed medians differed by only +0.06%; both variants and controls had long slow tails. Peak whole-GPU memory remained 6,805 MiB. This does not establish a decode improvement.

## ANALYSIS

The N=1024 transform has 4 KiB of shared memory and currently distributes four register elements per thread across 256 threads. Halving NT increases per-thread register work and did not improve throughput. Doubling NT to 512 showed a tiny positive median delta, but it was not repeatable at a magnitude distinguishable from noise. Context-4096 variability obscured small differences and produced severe tails for candidate and control alike. The fused transform/quantizer is already one launch, so the tested CTA width change did not remove synchronization or memory traffic.

## DECISION

**REVERT.** Keep the source-default NT=256 implementation and active source-default library. No changes were committed.

## FOLLOW-UPS

- Keep the PT layout and transform/quantization code unchanged unless a future experiment measures a repeatable full-model decode gain.
- A direct bytewise fused-output reference test could improve isolated PT-store coverage if a future kernel change alters the transform, reduction, quantization, or stores; this experiment only changed CTA mapping and produced identical fixed-seed model completions.

## IMPORTANT DISCOVERIES

- The GGUF rotation block is N=1024, while sign/K width is 17,408. This makes the model path 17 transform CTAs per row, one decode row, and no transform-padding beyond K.
- Ampere’s host layout selector overrides the one-column default and selects `GGML_CUDA_Q8_1_PT` on sm_86.
- NT=128 regresses; NT=512 ties within noise after reversed-order comparison. The initial long-context apparent win was caused by control slow-tail samples.
- Source hash after restoration: `6027c6ab846050ec4c63009f9b0eea70edc29a2447a3801a8c31fb937e866f33` (`quantize.cu`); `quantize.cuh` hash: `42a19a0df6479bffb65ac43af8f2f57cc3c1f0e7c175bc4683333deded8f5f77`. Final active source-default library rebuilt from restored source: `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`. The initially active library was `72d89c0c69200865b2200ef35b94e14b9a6a52a840c17cb031e987d809207e72`; rebuilding the restored source produced the recorded final active hash. With no `LD_LIBRARY_PATH`, `ldd build/bin/llama-bench` resolves `build/bin/libggml-cuda.so.0` (saved in `loader_default.txt`); no candidate is active.
