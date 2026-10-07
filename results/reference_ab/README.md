# Matched reference/current results

This directory contains a direct comparison between the unmodified project baseline and the current verified production implementation on the RTX 3080.

## Builds

| Arm | Source commit | Runtime source | Binary | CUDA library |
|---|---|---|---|---|
| Reference | `2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c` | PrismML `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17` | `/tmp/bonsai2-reference/build/bin/llama-bench`, SHA-256 `0ac0b7c1a08829d3fc4fca2d328daaf29c57acd1c63f1cf727b18bcd7dd74042` | `/tmp/bonsai2-reference/build/bin/libggml-cuda.so.0`, SHA-256 `d3286529a3df9db8d53fe89145bf4c4f69062dcc9f49ba1f85216d571f55a51b` |
| Current | `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5` | PrismML `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17` | `build/bin/llama-bench`, SHA-256 `81187ab3fc4aeda74f92b08ca21ad774d74d1418fb2467d278b41dfe8dcdab13` | `build/bin/libggml-cuda.so.0`, SHA-256 `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642` |

Both builds used Release, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, CUDA graphs enabled, Flash Attention enabled, and the same CMake settings. Their CMake caches were checked. `ldd` plus `readelf -d` verified each binary's RUNPATH resolved ggml libraries from its own build directory; no `LD_LIBRARY_PATH` override was used.

## Method

The same `benchmark/run.py` harness, model files, and options were used for both arms. Each mode had two seven-repetition runs per arm in reversed order: pair 1 reference→current, pair 2 current→reference. Every run used the ≤60°C / ≤5% utilization gate. Model: `models/Ternary-Bonsai-2-27B-PTQ1_0.gguf`, SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.

Configuration: decode contexts 512/4096 with 128 generated tokens; prefill contexts 128/512/2048/4096; combined prompts 512/4096 with 128 generated tokens; seven repetitions; llama-bench default warmups; 99 GPU layers; Flash Attention on; batch/microbatch 2048/512; F16 K/V cache; 8 CPU threads. Start temperature was at most 60°C; utilization at most 5%. Current-versus-reference statistics use the median of the two run medians. All raw seven-sample arrays and telemetry are in the JSON files below; `summary.json` gives per-arm aggregates, ranges, latency, peak memory, and binary hashes.

Reproduction pattern (select the intended binary and output path):

```bash
python3 benchmark/run.py \
  --binary /tmp/bonsai2-reference/build/bin/llama-bench \
  --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --modes decode --contexts 512 4096 --decode-tokens 128 --repetitions 7 \
  --cooldown-temp-c 60 --output results/reference_ab/decode_reference_pair1.json
```

Use `--modes prefill --contexts 128 512 2048 4096` and `--modes combined --contexts 512 4096` for the other workloads. Set `--binary` to the current build for that arm; reverse the arm order for pair 2.

## Files

- `decode_reference_pair1.json`, `decode_current_pair1.json`, `decode_current_pair2.json`, `decode_reference_pair2.json`
- Equivalent `prefill_*` and `combined_*` files
- `summary.json`: parsed medians, all samples, standard deviations, ranges, latencies, memory, and executable/library hashes
- The matching `results/raw/20261007T*PTQ1_0_{decode,prefill,combined}.*` files preserve the original llama-bench stdout JSON and stderr startup output for all twelve invocations.

The cooldown gate is applied before each arm. The reference and current executables/build libraries are isolated; no benchmark was run concurrently.
