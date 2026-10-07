# Exp075: PTQ1_0 GEMV L2 residency and reuse audit

## HYPOTHESIS

A subset of the active batch-1 PTQ1_0 weight matrices might fit in the RTX 3080's 5 MiB L2 and remain resident between token replays, potentially lowering active GEMV time. Tensor size alone does not establish reuse: a matrix helps only if its lines are accessed again before intervening traffic displaces them.

## IMPLEMENTATION

This read-only profile/manifest study used detached worktree /home/maxsun/autonomous_projects/.worktrees/exp075-l2-cache-reuse, based on commit 7400002cae80f291b5bf0350e933b82e2ec78dfc. Source hashes match Exp074: mmvq-ptq1_0.cuh SHA-256 f398417accddf1948fbf12155b4180fd362a3dfea88366ef3980161700d56496, mmvq.cu SHA-256 e889b1543656cd2e5c0c6151f11bfdd7f641e88af441449d92064ac902b9483d, and vecdotq.cuh SHA-256 8a47f1b7ef1b1819e56e16611af4632fcd6152da449df94edeb7080b77d79e9e. Full hashes are in results/exp075/raw/source_hashes.txt. Production code commit remains ffb0ef37690b902829ea1158b02b14517ed93c2b.

Inputs were Exp074's GGUF tensor manifest and the existing Exp062 CUDA Graph profile/adjacency data at contexts 512 and 4096. No build, model run, or production source edit was made. I audited exact-kernel harness feasibility before timing anything; details are in results/exp075/raw/feasibility_audit.txt.

Analysis commands from this worktree:

    git worktree add --detach /home/maxsun/autonomous_projects/.worktrees/exp075-l2-cache-reuse 7400002cae80f291b5bf0350e933b82e2ec78dfc
    sha256sum ggml/src/ggml-cuda/mmvq-ptq1_0.cuh ggml/src/ggml-cuda/mmvq.cu ggml/src/ggml-cuda/vecdotq.cuh
    python3 -c 'import json; p=json.load(open("results/exp074/raw/ptq1_tensor_manifest.json")); x=[t for t in p["tensors"] if t["category"]=="ptq_gemv_candidate" and t["type"]==143]; print(len(x),sum(t["payload_bytes"] for t in x))'

The model manifest identifies /home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf, 5,946,648,928 bytes, SHA-256 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3. The full size-class list with tensor names is results/exp075/raw/tensor_size_classes.json.

## RESULT

The active PTQ1_0 set has 401 logical weight tensors totaling 5,599,641,600 bytes. Only 32 fit individually in 5 MiB; all are [5120,1024] K/V tensors of 1,146,880 bytes each (1.094 MiB), totaling 36,700,160 bytes. The remaining 369 tensors individually exceed L2.

| Tensor shape | Count | Payload each | L2 relation |
|---|---:|---:|---|
| [5120,1024] | 32 | 1,146,880 B (1.094 MiB) | fits |
| [5120,6144], [6144,5120] | 112 | 6,881,280 B (6.562 MiB) | 1.31× L2 |
| [5120,10240] | 48 | 11,468,800 B (10.938 MiB) | 2.19× L2 |
| [5120,12288] | 16 | 13,762,560 B (13.125 MiB) | 2.63× L2 |
| [5120,17408], [17408,5120] | 192 | 19,496,960 B (18.594 MiB) | 3.72× L2 |
| [5120,248320] (output.weight) | 1 | 278,118,400 B (265.234 MiB) | 53.1× L2 |

The existing Exp062 profile contains 361 mul_mat_vec_ptq1_0_pt nodes per one-token graph replay at each context. Logical model execution traverses projection weights once per token; some fused GEMV nodes read two different matrices, so node count is not tensor count. Under that traversal, the reuse distance for any matrix at its corresponding next-token use includes the other PTQ weight payload. Even the smallest K/V matrix therefore has approximately 5,598,494,720 bytes of intervening PTQ weight accesses, over 1,067 L2 capacities. A tensor fitting in L2 does not imply that it survives the next-token reuse distance.

The replay profile shows immediate activation-preparation-to-GEMV graph adjacency for many calls. A planar Q8 activation for the largest K dimension (17,408) is 9 × 136 × 16 = 19,584 bytes; other listed K sizes produce 1,152–13,824 bytes. These small inputs can be reused among CTAs during a GEMV and can be produced close to their consumer. That is intra-kernel/producer-consumer locality, not evidence that a weight tensor remains resident across token replays. Profiler node names do not identify activation or weight pointers, so exact cross-node buffer reuse is not attributed here.

## CORRECTNESS

No kernel or model arithmetic was changed. No candidate output check or sanitizer run applies. The source hashes above match the active Exp074 path. This report makes no new numerical correctness claim.

## MICROBENCHMARK

Not run. The exact production function is a static template in mmvq-ptq1_0.cuh and relies on ggml CUDA launch helpers, internal types, PDL synchronization, and launcher-computed scheduling parameters. Existing profiler captures expose graph node IDs, template signatures, and timing, but not tensor names or pointers. A valid cold/warm test needs an instrumented isolated engine build that identifies tensors and inserts eviction work between graph replays, or a standalone harness that reproduces the production launch and real model buffers/activations. No such exact harness exists here. Building and validating one would require engine instrumentation/build. I stopped before timing an altered or synthetic kernel as a production proxy.

The last recorded GPU gate sample before this audit was 47 C, 0% utilization, 173 MiB used, with idle clocks at 210 MHz SM / 405 MHz memory. Since no kernel timing was attempted, there are no Exp075 event samples, variance, or measured clock claims. Nsight Compute remains unavailable with ERR_NVGPUCTRPERM; no permission change or hardware memory-counter claim was made.

## END-TO-END IMPACT

Not measured. No model benchmark or production change was run. The established paired best remains 84.407 tok/s at context 512 and 81.885 tok/s at context 4096. Exp062's existing node profile measured 9.006 ms/token GEMV-family duration at context 512; that is prior context, not a new Exp075 result.

## ANALYSIS

The size classes rule out broad cross-token retention of complete active weight matrices. For all 32 matrices small enough to fit, the next-token working distance still spans nearly the full 5.6 GB PTQ payload. Larger tensors cannot reside in full, and their next use is also separated by the model traversal. Thus 32 matrices fit in L2 is not a basis for expecting meaningful inter-token weight hits. This does not rule out partial-line behavior, replacement-policy effects, or L2 reuse among CTAs while one GEMV is executing.

A cold-versus-warm replay comparison could still test those effects, but the defensible target is the actual production PTQ kernel with actual packed weights and planar activations, an explicit eviction condition outside the timed interval, and a no-eviction steady-replay case. Existing profiles cannot establish pointer-level ordering or cache residency, and the standalone Exp074 streaming test is not a substitute. The evidence supports narrowing a future cache experiment to a properly instrumented exact-kernel path; it does not support attributing the 9 ms GEMV-family time to cache misses or DRAM traffic.

## DECISION

**No kernel candidate; no cache-performance claim.** The manifest and replay order make meaningful next-token L2 reuse of complete PTQ weight matrices unlikely, despite 32 tensors fitting individually. Preserve the production kernel and current best. Do not use size fit alone to motivate cache hints or persistence policy.

## FOLLOW-UPS

If cache behavior remains a decision point, add a separate isolated engine harness that records each GEMV's actual weight/activation identity and replays the exact production launch. Compare cold and warm cases with eviction/flush work outside timed kernel intervals, include no-eviction steady replay, and report enough CUDA-event samples, start/end clocks, variance, and output checks. Keep the existing ≤60 C / ≤5% utilization start gate. Do not claim DRAM bytes without authorized hardware counters.

## IMPORTANT DISCOVERIES

- 32/401 PTQ GEMV tensors fit individually in 5 MiB L2, but they are only 1.094 MiB K/V tensors and total 36.7 MB.
- Every next-token reuse of one of those weights is separated by about 5.598 GB of other PTQ payload on the once-per-token traversal, far beyond L2 capacity.
- The other 369 tensors exceed 5 MiB; output.weight is 278.1 MB.
- Planar Q8 activations are only 1.1–19.6 KB for these K sizes and may have intra-GEMV producer/consumer locality. That is distinct from cross-token weight reuse.
- Existing graph traces do not map GEMV node IDs to tensor pointers, preventing exact per-tensor cold/warm attribution without engine instrumentation.
