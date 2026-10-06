# Setup and reference revisions

## Reference source

The model card and PrismML's current demo identify the PrismML fork of llama.cpp as the reference runtime. Stock llama.cpp does not implement the model's `PTQ1_0` and `PQ2_0` types or the required runtime Hadamard rotation.

- Runtime: <https://github.com/PrismML-Eng/llama.cpp>
- Branch: `prism`
- Upstream commit at project start: `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17` (`arm: NEON vec_dot for PQ2_0 (#265)`)
- Demo/reference scripts: <https://github.com/PrismML-Eng/Bonsai-demo>
- Demo commit at project start: `74fab33d81d81a535525bd98c7a676b22d4dca46`
- Local demo checkout: `../bonsai-demo-reference`
- Model repository: <https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf>

The local research branch starts at the runtime commit above. Upstream source is preserved in Git history; do not rebase experiments onto an unrecorded update. The model repository revision is pinned below.

### Implementation map at the reference commit

- Quantized group layouts and scales: `ggml/src/ggml-common.h`
- CPU quantization/dequantization references: `ggml/src/ggml-quants.c`
- CUDA PTQ1_0/PQ2_0 dot products: `ggml/src/ggml-cuda/vecdotq.cuh`
- CUDA batch-1 and multi-column GEMV dispatch: `ggml/src/ggml-cuda/mmvq.cu`
- CUDA quantized GEMM: `ggml/src/ggml-cuda/mmq.cu`
- Hadamard-to-Q8_1 fusion and CUDA graph scheduling: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Hybrid attention and model graph: `src/models/qwen35.cpp`

The fork README identifies this branch as the current PrismML runtime and warns against the stale `prism-v6` line. The model card reports PTQ1_0 decode ahead on Ada and L4, while PQ2_0 is ahead on the published Ampere/Hopper/Blackwell rows and on prompt processing; its published table has no RTX 3080 row. This is prior evidence to test here, not an RTX 3080 result.

## Model representations

Download these text-only GGUF files from the public model repository:

- Repository revision: `b072e1d3b35a0a630cece372c2127528e0994386`
- `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5,946,648,928 bytes; 1.75 bits/weight)
  - SHA-256: `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`
- `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,206,168,928 bytes; 2.125 bits/weight)
  - SHA-256: `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`

Both use ternary weights in groups of 128 with FP16 scales. PTQ1_0 uses dense base-3 trit packing; PQ2_0 uses two-bit slots. Model metadata describes a 1024-wide block Hadamard rotation folded into weights, with matching activation transforms in the runtime. The optional vision projector is not used in text benchmarks.

Download the two files without installing anything system-wide:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 huggingface-cli download \
  prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PTQ1_0.gguf Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --revision b072e1d3b35a0a630cece372c2127528e0994386 --local-dir models --max-workers 3
```

Both SHA-256 values match the ETags recorded by the Hugging Face download client. `models/` is ignored by Git.

## Build

Use the installed Fedora user-space toolchain and the current CUDA toolkit:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel 4 --target llama-cli llama-bench llama-perplexity
```

The command specializes generated device code for the RTX 3080. Keep the default CUDA graph, Flash Attention, and quantized matmul settings fixed for baseline comparisons. Record any later build-flag changes with their benchmark results.

## Baseline commands

Run the full paired baseline with the same GPU start gate used in the recorded results:

```bash
python3 benchmark/run.py --cooldown-temp-c 60 --output results/baseline.json
```

For a single decode workload, the equivalent `llama-bench` command uses:

```bash
build/bin/llama-bench -m models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa on -b 2048 -ub 512 -ctk f16 -ctv f16 -t 8 \
  -p 0 -n 128 -d 2048 -r 7 -o json
```

Use the same command, context depth, warmup behavior, repetitions, batch sizes, and device start condition for PQ2_0. See `benchmark/README.md` for the full matrix and output schema.
