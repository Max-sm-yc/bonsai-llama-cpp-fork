# Experiment 084: expanded frozen-original vs research-fork comparison

## QUESTION

Do the retained CUDA/runtime changes improve more than PTQ1_0 token generation, and do the results hold for PQ2_0, prefill, and mixed prompt-plus-generation workloads?

This is a measurement of the retained candidate as a bundle. It does not isolate the contribution of any one optimization; the individual changes and their stage measurements are documented in the [repository README](../../README.md#what-changed).

## BASELINE AND CANDIDATE

The project is built on PrismML's Bonsai-capable `llama.cpp` fork. Its pre-existing runtime support provides the PTQ1_0 and PQ2_0 GGUF formats used here. “Original” is a frozen build of this project at commit `2a6ac568b69a61db0ee151b24c9b2cdb7a4f8a7c`. “Fork” is the retained research candidate, code commit `62b4b4ce0c2809272b9d69d09f3359abd7111848`. The frozen comparison keeps model-format and runtime support constant while comparing the performance changes. It is not stock upstream llama.cpp: the selected model files use project-specific formats and paths.

Each binary resolves its own `libggml-cuda.so.0.21.0`; this prevents one build from accidentally loading the other build's CUDA implementation.

| Build | `llama-bench` SHA-256 | CUDA library SHA-256 |
| --- | --- | --- |
| Original | `0ac0b7c1a08829d3fc4fca2d328daaf29c57acd1c63f1cf727b18bcd7dd74042` | `d3286529a3df9db8d53fe89145bf4c4f69062dcc9f49ba1f85216d571f55a51b` |
| Fork | `81187ab3fc4aeda74f92b08ca21ad774d74d1418fb2467d278b41dfe8dcdab13` | `14471383ade09fbfb6153970f340119bb91e7bbe27a85bb9c4a2ebbfa2c20f0d` |

## METHOD

Hardware was one NVIDIA GeForce RTX 3080 (sm_86, 10 GiB VRAM). Both builds used the same Ternary Bonsai 2 27B PTQ1_0 and PQ2_0 model files. All runs used 99 GPU layers, Flash Attention, F16 KV cache, batch/microbatch 2048/512, eight CPU threads, and `llama-bench`'s default warmups.

The workload matrix has seven shapes for each format:

| Mode | Workloads |
| --- | --- |
| Prefill | 512- and 4096-token prompts; no generation |
| Decode | Existing context lengths 512, 2048, and 4096; 128 generated tokens |
| Combined | 512- and 4096-token prompts plus 128 generated tokens |

Each format/workload combination has two seven-repetition runs per build. Run order was reversed between the pairs: original then fork, followed by fork then original. This gives 14 workload/format comparisons, 28 binary invocations, and 56 per-format run records. Table values are medians of the two run medians. The chart whiskers show each pair's percentage delta so order-to-order spread remains visible.

Before each format run, the GPU had to be idle at no more than 5% utilization and below its temperature gate. The gate was 65°C throughout except for the second combined-at-4096 pair, which used a matched 66°C gate for both builds. At the time of that final run, 66°C was the card's stable idle floor (zero utilization, roughly 88% fan); it did not reach 65°C. The raw record shows both arms started at 66°C. No pair compares one build at 65°C against the other at 66°C.

## RESULTS

The complete 14-row throughput table and chart are in the [README](../../README.md#extended-original-vs-fork-workloads). Machine-readable rows, including paired percentage deltas and peak memory, are in [`summary.csv`](../../results/exp084/summary.csv). Full samples and GPU telemetry are preserved in [`summary.json`](../../results/exp084/summary.json) and the [`raw/` directory](../../results/exp084/raw/).

The fork's PTQ1_0 decode improvement repeated at every tested context: +8.53% at 512, +8.35% at 2048, and +7.73% at 4096. The two reversed-order pairs differed by at most 0.18 percentage points at any one context. PQ2_0 decode improved by 1.02–1.26%. Prefill remained close to the frozen reference, from +0.09% to +0.30% across formats and prompt lengths.

For combined workloads, PTQ1_0 improved +6.78% at 512 and +1.02% at 4096. PQ2_0 improved +1.06% at 512 and was tied at 4096 (-0.03% overall; its two paired changes were +0.50% and -0.57%). These mixed results show why decode-only results should not stand in for prompt processing or whole-request throughput.

Peak whole-GPU memory was unchanged or differed by 2 MiB between the two builds for each workload/format. PTQ1_0 used about 6.6–6.8 GiB; PQ2_0 used about 7.7–7.9 GiB on this card.

An initial 70°C-gate pilot is preserved in [`pilot_70c_summary.json`](../../results/exp084/pilot_70c_summary.json) and [`pilot_70c_raw/`](../../results/exp084/pilot_70c_raw/), but excluded from all headline values. It reached 82–88°C and exhibited large within-run throughput drift; for example, the PQ2_0 combined-at-4096 fork samples fell from 850.8 to 394.7 tok/s across one seven-sample run. The final matrix uses the lower, matched start gates above.

## LIMITS

- Results describe one RTX 3080, one 27B model family, two quantizations, and the listed batch-one decode/prefill configurations. They do not establish performance on other GPUs, batch sizes, or models.
- `llama-bench` throughput is not server latency or user-perceived response time. This experiment does not measure token quality or output equivalence.
- The fork/original delta is cumulative across retained changes; it cannot be attributed to one kernel or graph fusion from this experiment alone.
- The final combined-at-4096 pair starts at 66°C while the other pairs start at 65°C. Both arms in that pair are matched, and its separate pair deltas are available in the CSV.

## FOLLOW-UP DIRECTIONS

1. Repeat the matrix on a newer GPU and another Ampere card, with the same start-temperature gate, to see whether the PTQ1_0 decode gain depends on sm_86.
2. Repeat on additional Bonsai model sizes and quantizations, then extend to larger batch sizes. This will show whether the gains persist when batch-one decode is no longer dominant.
3. Profile the retained candidate again before choosing another optimization target. The current research identifies PTQ1_0 batch-one GEMV as the largest decode cost, while the mixed-workload results show that prefill and generation gains differ.
4. Preserve temperature, utilization, power, and clock telemetry for future A/B runs. Use reversed order and multiple pairs when comparing close results.

## REPRODUCTION

Provision both model files and the two isolated binaries, then run from the repository root:

```sh
python3 experiments/084-fork-vs-original/run_comparison.py --cooldown-temp-c 65
python3 experiments/084-fork-vs-original/plot_comparison.py
```

The runner defaults to the reference build at `/tmp/bonsai2-reference/build/bin/llama-bench` and the candidate build at `build/bin/llama-bench`. `--resume` retains complete workload pairs if a run stops partway through. The raw per-run JSON includes the exact command, samples, start temperature, peak GPU memory, and telemetry.
