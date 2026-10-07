# Experiment 049: GDN batch-1 decode design challenge

## HYPOTHESIS

A distinct sm_86 dataflow, communication, or fusion design might reduce the measured active GDN family cost (~0.500 ms/token) without changing results. Exp021 exhausted columns-per-warp tuning; state-gather and GDN-to-cache-copy fusions must remain active.

## IMPLEMENTATION

Audited active source, graph fusion matchers, correctness cases, and the generated sm_86 cubin for `gated_delta_net_cuda<128,false,false,true,false>`. The active four-column-per-warp mapping reuses q/k across columns; each warp updates four contiguous state columns. The recurrence requires a first reduction (`S @ k`) before delta, then a second reduction (`S' @ q`) for output. Gates are scalar per (head, token), computed in-kernel from raw inputs.

The active cubin contains 40 `SHFL.BFLY` instructions: eight five-step warp reductions, exactly two per each of four columns. It has no CTA barrier in the recurrence. The three `MUFU.EX2` and one `MUFU.LG2` correspond to sigmoid/softplus/exponential gate math. No redundant state transfer or reduction was evident. Sharing gate results across CTAs would require inter-CTA communication or an extra launch, which has no credible advantage for the short decode sequence. The codegen summary is in `results/exp049/codegen_audit.txt`.

No candidate was implemented. No source files or build outputs were changed. State-gather fusion and cache-copy fusion remain untouched.

## RESULT

**NO CANDIDATE / REVERT.** The audit found no safe, distinct design with a plausible path to a focused win. The prior GDN family measurement remains 0.4998 ms/token at context 512 and 0.5003 ms/token at 4096 (Exp047).

## CORRECTNESS

No candidate was built, so no new correctness run was warranted. Existing backend GATED_DELTA_NET cases include recurrent state and output checks across decode, chunked, KDA, and cache-row paths; no source changes affect them.

## MICROBENCHMARK

Not run: there was no candidate/control kernel pair to measure. The active cubin/source inspection supplied no new timing claim.

## END-TO-END IMPACT

Not measured. No model A/B was warranted without a focused kernel candidate. The current best remains 83.34575 tok/s at context 512 and 80.35215 tok/s at context 4096.

## ANALYSIS

The recurrence is dependency ordered: each column needs the `S @ k` reduction before its state update and then needs `S' @ q` for the output. The current active mapping already keeps the common q/k vectors live and uses warp shuffle reductions without an inner CTA synchronization. The active state layout provides contiguous accesses per column. Gate arithmetic is replicated across CTAs, but reducing that duplication introduces a communication or launch cost that is unlikely to pay for the small scalar computation on this batch-1 path. Remaining mapping variations would repeat Exp021 or earlier rejected communication/launch approaches.

## DECISION

**NO CANDIDATE / REVERT.** Keep the production four-column mapping and graph fusions. No experimental source change remains; no commit was made.

## FOLLOW-UPS

Revisit GDN only if a future profile/codegen change exposes redundant state traffic, a removable launch, or a way to share per-head gate values without synchronization or another kernel launch.

## IMPORTANT DISCOVERIES

- Active sm_86 SASS performs the expected two five-step warp reductions per column and has no barrier in the recurrence.
- State gather and state-cache copy fusions already remove the evident adjacent operations.
- The active state access pattern is contiguous per column; four-column q/k reuse is already implemented.
