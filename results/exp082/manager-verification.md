# Exp082 manager verification

- The isolated worktree is based on manager commit `39f6b1e`. The experiment changed only the active Ampere FlashAttention stage target from two stages to one; after the measurements, `fattn-mma-f16.cuh` matches the base source byte-for-byte. Production source in the manager checkout is unchanged.
- Verified the control CUDA library hash `7b0c851c…` and candidate library hash `88355f4d…`; the recorded `ldd` and loader traces resolve each executable to the intended build.
- Independently recomputed attention sums from all four retained 31-replay profile JSON files:

| Context | Control main + fixup (ms) | Candidate main + fixup (ms) | Candidate delta |
|---:|---:|---:|---:|
| 512 | 0.196410 + 0.034482 = 0.230892 | 0.225433 + 0.037158 = 0.262591 | +13.7% |
| 4096 | 0.546593 + 0.036184 = 0.582777 | 0.600891 + 0.056649 = 0.657541 | +12.8% |

- Queried the retained CUDA kernel activity tables directly. The context-4096 active main grid rises from 68 to 136 CTAs while dynamic shared storage drops from 67,728 to 33,936 bytes; at context 512 the active grid remains 48 CTAs. The occupancy premise was real, but total latency regressed.
- Verified `model_smoke_comparison.json` reports exact body equality for the fixed-seed 32-token completion. `backend_flash_candidate.log` reports 2,994/2,994 CUDA cases passed. Full project correctness was not run because the focused performance gate failed.
- **Decision: REVERT / NO CANDIDATE.** No model decode A/B or peak-VRAM comparison was run. Keep the current two-stage FlashAttention configuration and current-best commit.
