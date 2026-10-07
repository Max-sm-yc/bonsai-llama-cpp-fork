# Exp042 artifacts and reproduction notes

Source base: main research HEAD `d3e7daf`. Production checkout remained at c6cdaa5 and its build/library were untouched. Isolated source worktree: `/tmp/exp042-cta-width`.

## Focused width screen

The harness binary and source are `cta_width_screen` and `cta_width_screen.cu`; `harness_resources.txt` is the ptxas report. `run_micro.sh` invokes all K/width combinations serially with 9 rotated sample pairs and 100 work-plus-fold iterations per sample. Raw correctness/timing output is in `screen_k{40,136}_t{64,128,256,512}.txt`.

Commands:

```bash
nvcc -O3 -arch=sm_86 -Xptxas=-v results/exp042/cta_width_screen.cu \
  -o results/exp042/cta_width_screen 2> results/exp042/harness_resources.txt
bash results/exp042/run_micro.sh
```

## Isolated runtime libraries

`control/` contains a copy of the current production runtime. `width64/`, `width128/`, `width256/`, and `width512/` contain candidate CUDA libraries linked from the same unchanged CUDA objects except for `mmvq.cu` and `quantize.cu`, which were compiled from the isolated source worktree with `-DPTQ1_0_PT_THREADS={64,128,256,512}`. `shape256_k40.patch` shows the additional candidate dispatch: width 256 for exactly 40 K blocks, otherwise 128. This library is in `shape256_k40/`.

Before benchmark runs, each copied executable retained RUNPATH `/home/maxsun/autonomous_projects/bonsai2-rtx3080/build/bin:`. Each run set `LD_LIBRARY_PATH` to its own arm directory; `ldd` confirmed CUDA and base libraries resolved inside that directory. Control CUDA library SHA-256: `bad70d76b19fdd1b21f61e5c9eb4900b5334c638ff75cdec11f3a4d3b2b28642`. Shape candidate CUDA library SHA-256: `85989591ba4abce7c77968ee7e2958ad2d2e5a5b28a94ed9e50f5611d755c7b6`.

The two reversed-order pairs used these exact commands; the benchmark runner applied its ≤60°C/≤5%-utilization start gate before each arm:

```bash
LD_LIBRARY_PATH=$PWD/results/exp042/control python3 benchmark/run.py \
  --binary results/exp042/control/llama-bench \
  --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --modes decode --contexts 512 4096 --decode-tokens 128 --repetitions 7 \
  --cooldown-temp-c 60 --output results/exp042/control_forward.json
LD_LIBRARY_PATH=$PWD/results/exp042/shape256_k40 python3 benchmark/run.py \
  --binary results/exp042/shape256_k40/llama-bench \
  --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --modes decode --contexts 512 4096 --decode-tokens 128 --repetitions 7 \
  --cooldown-temp-c 60 --output results/exp042/candidate_forward.json
LD_LIBRARY_PATH=$PWD/results/exp042/shape256_k40 python3 benchmark/run.py \
  --binary results/exp042/shape256_k40/llama-bench \
  --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --modes decode --contexts 512 4096 --decode-tokens 128 --repetitions 7 \
  --cooldown-temp-c 60 --output results/exp042/candidate_reverse.json
LD_LIBRARY_PATH=$PWD/results/exp042/control python3 benchmark/run.py \
  --binary results/exp042/control/llama-bench \
  --model PTQ1_0=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --modes decode --contexts 512 4096 --decode-tokens 128 --repetitions 7 \
  --cooldown-temp-c 60 --output results/exp042/control_reverse.json
```

All four raw JSON files retain all seven samples per context/run. The fixed-seed model smoke was:

```bash
LD_LIBRARY_PATH=$PWD/results/exp042/shape256_k40 python3 tests/model_smoke.py \
  --binary results/exp042/shape256_k40/llama-cli \
  --output results/exp042/shape256_k40_smoke.json --tokens 32
```

## Codegen and hashes

- `resources/width{64,128,256,512}.txt`: full `cuobjdump --dump-resource-usage` output.
- `resources/launch_bounds_width{64,128,256,512}.txt`: active plain ROWS=1 `.maxntid` extracted from PTX.
- `resources/shape256_k40.txt`, `resources/shape256_launch_bounds.txt`: candidate resources and emitted 128/256 launch bounds.
- `sass/width{64,128,256,512}.sass`: full-library SASS dumps for each fixed-width candidate.
- `sass/shape256_k40_active.sass`: active ROWS=1 candidate SASS for its 128- and 256-thread entries.
- `shape256_k40_smoke.json`: PTQ1_0 and PQ2_0 fixed-seed model smoke results.
- Source header SHA-256 (shape dispatch worktree): `41e9c219558ece8247f85c67ed752f1e81fba03a254f29070ee02c10c40e587c`.
- Focus harness SHA-256: source `12c3c302fff41c6b6fc8ec0f3969bd625093e02ed61eb81ba10de163adf29b1b`, executable `1240fd8ce8649357cbfe6333239446d1c4ec6303ca5dc8065aaf11266bad2cb3`.
