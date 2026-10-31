# Versions, branches and component names

Only `main` is the submission: the v11 code (`f218f86`, branch `v11-candidate`) with documentation added and its comments and docstrings brought up to date. Every vX row of the README's ablation is a pushed branch (below), so it can be rerun with the same notebook by putting a commit of its branch in `COMMIT`. The first two rows (example only, K1 + K4) predate the candidate branches.

## Branches

| Version | Branch | Adds |
|---|---|---|
| v1 | `v1-candidate` | K3, K2 |
| v2 | `v2-candidate` | faster K1, U |
| v4 | `v4-candidate` | final U, tests |
| v5f | `v5f-candidate` | S4 |
| v6 | `v6-candidate` | B, A1, A2, C, D (ran out of memory at step 22) |
| v6.1 | `v61-candidate` | the v6 leak fixed |
| v7 | `v7-candidate` | I1 to I4 (I4 = S4 off on 7 of 28 layers) |
| v8b | `v8b-candidate` | S4 on all layers, C1, C2 |
| v9 | `v9-candidate` | M2, M3, M5 |
| v9b | `v9b-candidate`, code `91d70d3` | M5 off |
| v10b | `v10b-candidate`, code `69b0c95` | P2 |
| v10c | `v10c-candidate`, code `0b9e562` | fp16 head |
| v10d | `v10d-candidate`, code `6b56d31` | compile overlap |
| v11 | `v11-candidate`, code `f218f86` | Q, merged q/k/v GEMM (submitted) |

Other branches:

- `v3-candidate`, `v5-candidate`, `v8-candidate`, `v10-candidate`: intermediate candidates, not rows of the ablation. `v10-candidate` holds P1 and P3, both dropped.
- `bench-*`: scouting variants, each one flag away from a candidate (for example `bench-v9-h16`, commit `c6d756a`: v9 with the fp16 head; `bench-v10-noP1`: v10 without P1), and `bench-k1`, a K1 micro-benchmark.
- Any other branch is an unmerged experiment.

Full runs the README gives in short (`--repeats 2`, all 10 checks passed):

- v9b, second run (notebook version 17, a slower session): base 90.3 s, ours 72.6 s, s = 1.244, peak 2,515 MiB, m = 5.704.
- v10c (notebook version 20, code `0b9e562`): ours 65.56 and 67.02 s (66.3) against base 89.11 and 86.11 s (87.6), s = 1.322; peak 2,335 against 14,345 MiB, m = 6.144; 578 tests passed.
- v10d (notebook version 22, code `6b56d31`): ours 65.43 and 64.13 s (64.8) against base 86.24 and 83.54 s (84.9), s = 1.310; peak 2,335 against 14,345 MiB, m = 6.144; eval loss 2.04963 against 2.04964; 600 tests passed.

## Flags on `main`

Every component can be switched in `solution/__init__.py`; the comment next to each flag says which earlier path its other value runs. The branches hold the exact code each version was measured with.

| Flag | On `main` | Other value |
|---|---|---|
| `USE_MERGED_QKV` | `True` | `False`: v10d |
| `USE_PRECOMPILE_OVERLAP` | `True` | `False`, with Q off too: v10c |
| `STORE_HEAD_FP16` | `True` | `False`, with Q and the overlap off too: v10b |
| `USE_ELEMENTWISE_FUSION` | `True` | `False`, with the three flags above off too: v9b |
| `MERGED_QKV_CACHE` | `False` | `True`: Q keeps its merged weight for the run (126 MiB more, not measured) |
| `USE_BWD_CHUNKING` | `False` | `True`: M5 on, as in v9 |
| `USE_LORA_INPUT_RECOMPUTE`, `USE_SDPA_ROW_COMPACT` | `True` | `False`: v8b's path (no M2, no M3); Q needs M2 |
| `USE_UNIFIED_SIG`, `USE_SOUP_SWIGLU` | `True` | `False`: v7's launches and the starter kit's SwiGLU kernels |
| `S4_OFF_FRACTION` | `0.0` | v7 used `0.25` |
| `USE_SDPA_CAUSAL_PADDED` | `True` | `False`: v6.1's dense-mask SDPA call |
| `S4_DEQUANT_REUSE` | `"both"` | `"off"`: v6.1 |
| `USE_K1_FAST_HOST`, `USE_DENSE_CACHE` | `True` | `False`: v6.1's K1 host path |
| `USE_SDPA_KV_COMPACT` | `False` | `True`: A, taken only when A1 is off |

## Component names

The names used in the README, `SUBMISSION.md` and the code. Code comments tag a component with the version that added it (v5 = the v5f row, v8 = v8b; v10's P2 shipped in v10b). Paths are under `solution/`.

| Name | What | Where | On `main` |
|---|---|---|---|
| K1 | NF4 dequantization kernel | `kernels/nf4.py` | on |
| K2 | LM head + cross entropy without logits | `kernels/head_ce.py`, `head_loss.py` | on |
| K3 | fused residual add + RMSNorm | `kernels/addnorm.py`, `layers.py` | on |
| K4 | fp16 MLP | `mlp.py` | on |
| U (v2) | unpadding of the token-wise ops | `unpad.py` | on |
| S4 (v5f) | MLP recompute | `mlp_recompute.py` | on |
| B (v6) | fused block | `fused_block.py` | on |
| A1 (v6) | attention recompute | `sdpa_recompute.py` | on |
| A2 (v6) | LoRA dropout replay | `lora_replay.py` | on |
| C (v6) | in-place SwiGLU backward | `mlp_recompute.py` (`INPLACE_SWIGLU_BACKWARD`) | on |
| D (v6) | LoRA-A inputs kept on real rows only | `saved_compact.py` | on |
| A (v6) | one key/value head per GQA group | `saved_compact.py` | off (A1's fallback) |
| I1 (v7) | causal-only SDPA on the packed path | `sdpa_recompute.py`, `unpad.py` | on |
| I2 (v7) | one dequantization per weight in S4's backward | `mlp_recompute.py` | on |
| I3 (v7) | K1 host fast path and k/v dequant cache | `kernels/nf4.py` | on (q/o cache off) |
| I4 (v7) | S4 off on the first layers | `unpad.py` | off (`S4_OFF_FRACTION = 0.0`) |
| C1 (v8) | one launcher signature for all kernels | `kernels/__init__.py` | on |
| C2 (v8) | our SwiGLU kernel | `kernels/swiglu.py` | on |
| M2 (v9) | LoRA-A input recompute | `lora_recompute.py` | on |
| M3 (v9) | A1 saves real rows only | `sdpa_recompute.py`, `unpad.py` | on |
| M5 (v9) | row-chunked backward, smaller K2 tile | `mlp_recompute.py`, `kernels/head_ce.py` | off since v9b |
| P1 (v10) | compile pool joined inside `apply()` | branch `v10-candidate` | dropped |
| P2 (v10) | elementwise fusion into K3's launches | `kernels/addnorm.py`, `fused_block.py`, `lora_recompute.py` | on |
| P3 (v10) | merged LoRA (`USE_LORA_MERGED`) | branch `v10-candidate` | dropped |
| fp16 head (v10c) | K2 part 3: the tied embedding and LM head stored in fp16 after a lossless check | `head_loss.py` | on |
| compile overlap (v10d) | background compile while Soup loads the model | `precompile.py` | on |
| Q (v11) | q, k, v base projections as one fp16 GEMM, one dX GEMM in backward | `merged_qkv.py`, `unpad.py`, `lora_recompute.py` | on (weight built per call; `MERGED_QKV_CACHE` off) |
