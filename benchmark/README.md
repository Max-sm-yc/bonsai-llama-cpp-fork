# Reproducible benchmark

`run.py` executes the upstream `llama-bench` workload on each selected GGUF and saves the exact commands, per-repetition samples, means, standard deviations, latencies, and sampled GPU memory to JSON. Raw stdout/stderr for each invocation goes under `results/raw/`.

Run after building the reference fork and downloading both files:

```bash
python3 benchmark/run.py --cooldown-temp-c 60 --output results/baseline.json
```

For paired-format comparisons, `--cooldown-temp-c 60` waits before each format until the GPU is at or below 60 C and utilization is at or below 5%. Use the same setting for every compared implementation. The gate is optional for quick investigations.

The default matrix is both packings, prompt processing at 128/512/2048/4096 tokens, batch-1 decode after each of those context depths for 128 generated tokens, and combined prompt-plus-generation at each context. Each row uses seven repetitions; llama-bench's warmups stay enabled. The harness pins GPU offload, Flash Attention, KV type, batch sizes, and CPU thread count. It stores each repetition and the median in addition to the runtime's mean and standard deviation. The model and machine must remain otherwise unchanged between format runs.

Run with the RTX 3080 visible through `/dev/nvidia*`. GPU access and both model runs were verified in this environment. `nvidia-smi` is sampled during each process. Its total used-memory figure includes the desktop, so `gpu_memory_increase_mib` is an estimate above the measured idle baseline.

The built-in `llama-bench` timer omits tokenization and sampling. Its combined mode measures the model's actual prompt evaluation and autoregressive model generation. A separate fixed-prompt `llama-cli` smoke run checks normal model loading and completion behavior; it is not substituted for the kernel-throughput benchmark.
