# Exp080 active PTQ1_0 dispatch and dataflow audit

Source commit: `f4c8740d3307648626ad708671221c93ecab2083`.

## Reached specialization

- `ggml/src/ggml-cuda/mmvq.cu` includes `mmvq-ptq1_0.cuh` and calls `mul_mat_vec_ptq1_0_pt_switch` from `mul_mat_vec_q_switch_ncols_dst` for `GGML_TYPE_PTQ1_0` when not HIP.
- The call is guarded by `!ids && !y_soa`. The switch further requires one destination channel, one sample, `ncols_x % QK_PTQ1_0 == 0`, 1–8 destination columns, and enough device shared memory.
- Batch-1 decode selects `ncols_dst == 1`, which instantiates `mul_mat_vec_ptq1_0_pt<1,1,FUS,GATE>`. The production model's plain, fused-gate, and fused-bias calls are served by this dedicated kernel when those shape/layout guards hold. Multi-column cases and unsupported shapes use other paths.
- `ptq1_0_pt_rows_per_item(1) == 1`, so a work item covers one output row and one 128-weight block. The thread dot reads the 28-byte packed PTQ weight block, nine 16-byte planar activation vectors (eight quant planes plus one scale/isum plane), decodes raw base-3 digits in registers, and feeds DP4A. The signed weight bias is corrected by subtracting exact activation `isum` at the four 32-element folds.

## Output accumulation ordering constraint

Each item writes one FP32 block result to shared memory. After one CTA barrier, each output row's current epilogue reads those block results in four sequential modulo-4 streams: `s0 += partial[k+0]`, `s1 += partial[k+1]`, `s2 += partial[k+2]`, `s3 += partial[k+3]`, then computes `(s0+s1)+(s2+s3)`. The optional invariant path is a separate warp reduction with its own tested arithmetic contract. Changing which FP32 partials are added together or their order can change bits.

With the default 128-thread CTA and dynamic shared-memory cap, the host selector chooses three rows/CTA for K=40 (120 work items) and sixteen rows/CTA for K=136 (2,176 work items, exactly 17 full 128-thread waves). The exact tile changes with K shape; a per-row CTA or warp/CTA-width variant repeats previously measured row/CTA/reduction families.

To reduce partial count and preserve the exact non-invariant sum order, the only straightforward ownership is one sequential owner for each of the four modulo-4 streams per output row. That assigns 4 lane streams per row and requires each stream to execute 10 K-block dots at K=40 or 34 at K=136. It replaces high-parallelism block dots with long dependent work. Distributing one stream across more lanes changes its FP32 addition association. The existing implementation already exposes independent K-block dots across the CTA and then folds in source-defined order.

## Compiled active kernel evidence

The isolated Release sm_86 build compiled `mmvq.cu.o` from the unmodified source. The `<ncols=1, ROWS=1, FUS=false, GATE=false>` specialization has 76 registers/thread, 0 stack bytes, 0 spills, and 0 statically allocated shared bytes (the launcher supplies dynamic shared memory). Its extracted SASS has 32 `IDP.4A` instructions, nine `LDG.E.128.CONSTANT` activation loads, 214 `LDS` instructions, one `STS` instruction site, one `BAR.SYNC`, and four `FFMA` sites. This confirms generated work already contains the direct packed DP4A path, the shared partial consumer, and its barrier; ptxas reports no spill-based alternative opportunity.

The active SASS function is retained as `active-kernel.sass.txt`; full PTX/SASS dumps were removed after extraction to keep evidence compact. Full resource records are in `baseline-resource-usage.txt`.

## Rejected repeated directions

Reports inspected: Exp010, Exp024–025, Exp031–046, Exp047, Exp052, Exp055–056, and Exp068, Exp073–079. The measured negative evidence relevant to this challenge includes cooperative recurrence integration (Exp024, large E2E loss), multi-item strip mining (Exp025, losses), padded/vector loads (Exp031), row SoA/sidecars (Exp032–035), CTA staging (Exp037, +9.9–12.9%), warp-register transpose (Exp038, +2.42–2.75x), async pipeline (Exp040, +10.4–23.2%), CTA widths (Exp042), activation staging (Exp044), lower-register/higher-CTA launch bounds (Exp045, all regressions), paired K/V output (Exp056, +3.7% graph replay), and cache persistence (Exp079, no E2E gain). Decoder/format/Tensor Core alternatives and repeated dataflow challenge are covered in Exp046/068/073/078. Generic decoder, warp/multiwarp reduction, and row scheduling families are also already exhausted in the earlier reports. Nsight Compute is unavailable; no permissions were changed or retried.
