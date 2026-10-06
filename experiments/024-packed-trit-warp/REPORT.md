# Experiment 024: warp cooperation for the packed PTQ1_0 recurrence

## HYPOTHESIS

Distributing the production packed base-3 recurrence across lanes could shorten its dependent decode chain and improve the batch-1 PTQ1_0 GEMV on sm_86. This was distinct from experiment 023's scalar-per-element decoder.

## IMPLEMENTATION

The standalone screen in `results/exp024/packed_recurrence.cu` compared a serial recurrence with an eight-lane cooperative recurrence. Its first build had a misaligned packed-word load; that was corrected to use the aligned four-byte group pointer before any valid results were accepted. The exact cooperative recurrence was then integrated into the active production PTQ1_0 GEMV for a candidate build.

The temporary production source edit was reverted after the large throughput loss; the candidate diff was not retained. The standalone kernel, test log, candidate/control benchmark data, and final baseline hashes are retained.

## RESULT

**REVERT.** The isolated recurrence screen improved its median from 0.00961536 ms to 0.00645936 ms per launch (1.49x), but the production candidate reduced end-to-end decode throughput by about 81.5% at both measured contexts. The focused screen did not predict the production schedule's cost.

## CORRECTNESS

- Standalone screen: 16,384 packed blocks matched bitwise; Compute Sanitizer reported zero errors after the alignment fix.
- Production candidate: selected CTests passed 4/4; CUDA-vs-CPU PTQ1_0/PQ2_0 matmul cases passed 96/96; fixed-seed 32-token smokes for both formats generated non-empty outputs. Details are in `results/exp024/correctness.log`.
- After reverting, PTQ1_0 and PQ2_0 fixed 32-token baseline smokes passed with the archived baseline library.

## MICROBENCHMARK

On the RTX 3080 / sm_86, the standalone recurrence screen used 16,384 blocks and 200 launches per sample. Serial median was 0.00961536 ms; cooperative-eight median was 0.00645936 ms. It reported 40 registers for serial and 32 for cooperative, with no spills. The sanitizer run had zero errors, though sanitizer-instrumented timings contained large outliers and are not used for performance conclusions. Artifacts: `microbench.txt`, `sanitizer_microbench.txt`, and `resources.txt`.

## END-TO-END IMPACT

Both runs used the same PTQ1_0 model, RTX 3080 / sm_86, batch-1 decode, 128 tokens, seven repetitions, contexts 512 and 4096, F16 KV, Flash Attention on, 99 GPU layers, batch/ubatch 2048/512, and eight CPU threads.

| Context | Control median | Candidate median | Change | Control mean ± SD | Candidate mean ± SD |
|---:|---:|---:|---:|---:|---:|
| 512 | 82.1551 tok/s | 15.2228 tok/s | -81.47% | 82.0385 ± 0.3042 | 15.2331 ± 0.1097 |
| 4096 | 79.6645 tok/s | 14.7398 tok/s | -81.50% | 79.5543 ± 0.2664 | 14.5648 ± 0.3613 |

Peak whole-GPU use was 6,805 MiB for both. Raw runs are `results/exp024/control.json` and `results/exp024/candidate_first.json`. The candidate's temperatures rose to 79°C; this may have contributed some noise, but cannot account for a repeatable roughly 5.4x throughput loss at both contexts. A reversed-order pair was not justified after the large, stable regression in the seven samples.

## ANALYSIS

The standalone screen measures recurrence work in isolation. Production dispatch maps many independent `(row, K-block)` work items across 128 threads and relies on its existing partial-sum schedule. Cooperative recurrence adds lane communication and changes how many independent K blocks execute concurrently; those costs dominate the isolated decode-chain reduction in the integrated candidate. The experiment did not separately attribute the loss to communication, occupancy, or scheduling, so those remain explanations to test only with a materially different design.

## DECISION

**REVERT.** Keep the ROWS=1 production kernel. The active source SHA is `f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496`; the restored active CUDA library SHA is `708eceba48460ad3d963b88c7f84a0f60a2bbed061d2cf7fdec70e39b15e29a9`. The source and library match the recorded best production baseline.

## FOLLOW-UPS

Do not integrate this cooperative mapping again without preserving the production kernel's K-block parallelism and measuring that schedule directly. Prioritize a fundamentally different, profile-driven GEMV design; keep the recurrence microbenchmark as a warning that local decode-chain wins do not guarantee end-to-end gains.

## IMPORTANT DISCOVERIES

- The packed recurrence's cooperative eight-lane form is faster in isolation and uses fewer registers, yet its production integration is dramatically slower.
- The end-to-end result is clear enough to reject this integration without spending another full seven-repetition candidate run.
- Restoring a fresh link from source produced a different library hash because the full CUDA relink changed binary contents; the exact previously verified library was available in `results/exp021/libs/source_default/` and was restored. Final active hashes are recorded in `results/exp024/HASHES.txt`.
