# Exp078 manager verification

- Independently checked all 5 source/artifact hashes from `source_hashes.txt`: 5/5 match the checked-out Exp078 base and committed prior evidence.
- Verified the bit-plane storage calculation: two 128-bit masks use 32 bytes per 128 weights versus PTQ1_0's 28 bytes, a 14.2857% increase. Applied to the measured 5,599,641,600-byte active GEMV payload, the increase is 799,948,800 bytes (~0.800 GB).
- Reviewed the active `ptq1_0_pt_block_dot` source: packed base-3 trits are decoded in registers and sent to DP4A with exact subgroup activation-sum correction; arbitrary signed Q8 values still require weighted selection/reduction in the bit-plane alternative.
- No candidate code or binary exists, and no new benchmark/correctness result is claimed. Production code and current best commit `ffb0ef37690b902829ea1158b02b14517ed93c2b` remain unchanged.
