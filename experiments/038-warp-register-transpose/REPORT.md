# Experiment 038: warp-register transpose for PTQ1_0 AoS

## HYPOTHESIS

The active sm_86 batch-1 PTQ1_0 GEMV loads 28-byte AoS blocks with a 28-byte lane stride. A warp could instead load 28 contiguous 32-bit words for four blocks and redistribute the seven words per block through register shuffles, avoiding Exp037's shared-memory store/load and barrier.

## IMPLEMENTATION

Built an isolated CUDA-event screen at `/tmp/exp038-warp-register` from manager HEAD `257bc1978956273df7b46d9902fc6615ae701bed`. `work_warp_register` assigns each warp four blocks: lanes 0–27 load the 28-word contiguous span, and lanes 0–3 gather seven words each using `__shfl_sync` before running the same packed recurrence, DP4A, scale correction, and row fold as direct AoS. The screen compares direct AoS and warp-register on 2,048 synthetic rows at 40 and 136 blocks per row, using nine rotated-order samples (100 launches/sample at 40 blocks; 80 at 136). The control and candidate time work plus row fold. No production source, build, or binary was changed.

Artifacts are in [`results/exp038/`](../../results/exp038/): harness, raw samples, ptxas output, and SASS. Harness SHA-256: `40ea02b9404e2d024ec9914bc27afa63a9dfe0c110dbe6408d701777822f4001`; screen binary SHA-256: `c97f93e2c4b5e5cc8f667fc8239c1b7276e816b3a9f7ccbd763e6afa93987a48`.

## RESULT

**REVERT.** The warp-register candidate lost by 142.13% (2.42× total time) at 40 blocks and by 175.23% (2.75×) at 136 blocks. Every candidate sample was slower than every direct-AoS sample at both shapes.

## CORRECTNESS

At both shapes, all 10,485,760 / 35,651,584 device-decoded trit codes matched the independent host decoder. Candidate row outputs matched direct AoS bitwise (zero mismatches), as did independent host outputs (zero mismatches, max absolute error 0). No production integration or model correctness run was performed.

## MICROBENCHMARK

CUDA-event work-plus-fold medians and ranges, nine samples per arm:

| Blocks/row | Direct AoS, µs | Warp-register, µs | Candidate delta |
|---:|---:|---:|---:|
| 40 | 9.30816 (9.27552–9.35552) | 22.53824 (22.51776–22.54848) | +142.13% |
| 136 | 25.12640 (25.04440–25.20880) | 69.15440 (69.07080–69.42400) | +175.23% |

Ptxas reports 40 registers/thread, zero stack/spills, zero shared memory, and no barriers for both `work_aos` and `work_warp_register`. SASS contains seven `SHFL` instructions in the candidate function, plus the contiguous global load stream. This confirms the intended register redistribution compiled without shared staging.

## END-TO-END IMPACT

Not measured. The focused screen is a decisive regression, so the candidate did not advance to integration or model A/B. The current reference remains 83.3458 tok/s at context 512 and 80.3522 tok/s at 4096.

## ANALYSIS

The global loads become contiguous, but four of 32 lanes execute the packed dot and each output lane needs seven shuffled words. The shuffle and underused arithmetic lanes cost much more than the saved transactions. Avoiding shared memory and its barrier does not recover that loss. This lane mapping is rejected; the result does not prove every possible register transpose loses, but a viable mapping must retain substantially more dot parallelism while redistributing the same seven words.

## DECISION (KEEP/REVERT/INCONCLUSIVE)

**REVERT.** Keep the existing direct-AoS ROWS=1 implementation. No model benchmark or integration was performed.

## FOLLOW-UPS

Only revisit register redistribution with a mapping that keeps multiple lanes computing each block's packed recurrence or amortizes each loaded word across useful DP4A work. Do not repeat this four-block/four-compute-lane layout.

## IMPORTANT DISCOVERIES

- Four blocks produce a naturally contiguous 28-word global span, and the seven words per block can be reconstructed exactly using register shuffles.
- This implementation uses no shared memory, barriers, stack, or spills, yet loses 2.4–2.75× because only four lanes per warp perform packed-dot work.
- Device code and full row output checks were exact at both the common and long-K shapes.
