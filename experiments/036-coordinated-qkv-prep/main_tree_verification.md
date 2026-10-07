# Exp036 main-tree verification

Date: 2026-10-07 UTC. Code commit: `c6cdaa5fa62787c97db58d1d2e1db666a4aeddb5`.

The main checkout was rebuilt with:

```bash
cmake --build build --parallel 4 --target llama-cli
```

Then `bash tests/run_correctness.sh` passed:

- selected CTest suite: 5/5, including the new direct fused-kernel GPU numerical test
- CUDA-vs-CPU PTQ1_0/PQ2_0 backend operations: 96/96
- fixed-seed 32-token PTQ1_0 and PQ2_0 model smokes
- normalized PTQ1_0 completion: exact match with the pre-promotion completion

The direct test compares the production PT output against an independent host RMSNorm/sign/FWHT reference for one and three 5120-wide rows. Maximum error was 0.539778 and 0.542589 of the stored Q8 scale, respectively; packed block sums were exact. The CTest target passed 1/1.

The main smoke JSON is `main_tree_smoke.json`. Its original harness destination, `results/baseline_smoke.json`, was restored so the reference baseline record remains unchanged. The full correctness log and direct numerical output are `results/exp036/full_correctness.log` and `results/exp036/fused_quantizer_reference.log`.

Main source SHA-256:

| File | SHA-256 |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `aee803e29b853a70d3e5274606cad32cdefc93d72578b1824df259dd2ba86351` |
| `ggml/src/ggml-cuda/quantize.cu` | `dd55176be1e6639d102d1d18bf3644795fa0ae73a85bd5ef26188ba8ec59e03e` |
| `ggml/src/ggml-cuda/quantize.cuh` | `cef94b4f946ec874fa55544f7eb00e20096e36a91cb3bdd00eed7e1a5e24b719` |
| rebuilt `build/bin/libggml-cuda.so` | `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642` |
