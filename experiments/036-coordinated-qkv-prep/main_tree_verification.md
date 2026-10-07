# Exp036 main-tree verification

Date: 2026-10-07 UTC. Code commit: `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`.

The main checkout was rebuilt with:

```bash
cmake --build build --parallel 4 --target llama-cli
```

Then `bash tests/run_correctness.sh` passed:

- selected CTest suite: 4/4
- CUDA-vs-CPU PTQ1_0/PQ2_0 backend operations: 96/96
- fixed-seed 32-token PTQ1_0 and PQ2_0 model smokes
- normalized PTQ1_0 completion: exact match with the pre-promotion completion

The main smoke JSON is `main_tree_smoke.json`. Its original harness destination, `results/baseline_smoke.json`, was restored so the reference baseline record remains unchanged.

Main source SHA-256:

| File | SHA-256 |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `aee803e29b853a70d3e5274606cad32cdefc93d72578b1824df259dd2ba86351` |
| `ggml/src/ggml-cuda/quantize.cu` | `dd55176be1e6639d102d1d18bf3644795fa0ae73a85bd5ef26188ba8ec59e03e` |
| `ggml/src/ggml-cuda/quantize.cuh` | `cef94b4f946ec874fa55544f7eb00e20096e36a91cb3bdd00eed7e1a5e24b719` |
| rebuilt `build/bin/libggml-cuda.so` | `4b4adb58e3d26cb8694aebf0843112b981de66ac54441760ec8290bb2b021dcf` |
