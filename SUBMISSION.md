# Submission: Team Delta

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-8cddb4ae-delta/commit/[TODO final commit]
  (code v9b = commit `91d70d3`, branch `v9b-candidate`; the final commit adds only README.md and this file)
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in our README):
  [TODO final link]
- Demo video (up to 3 minutes): [?: link]
- Presentation: [?: link]

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Jamil Tlegenov | Captain | [?] | jamildeltacaptain | [?] |
| [?] | [?] | [?] | [?] | |

## Our kernels
K1, K2 and K3 are launched through `soup_*_usig` entries (change C1: one runtime signature, so Triton builds one launcher). Each entry is a `@triton.jit` function that inlines the original kernel body unchanged, so check 3 reports the bodies as "not launched" and passes.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| K1: `soup_nf4_dequant_usig`, body `soup_nf4_dequant_kernel` | `solution/kernels/nf4.py` | bitsandbytes NF4 dequantization on the dequantize + `F.linear` path, forward and backward re-dequantization; bit-exact | part of the 11.6% total |
| K2: `soup_head_ce_usig`, body `soup_head_ce_kernel` | `solution/kernels/head_ce.py` (wiring `solution/head_loss.py`) | `lm_head` + cross entropy and TRL's logits-based entropy and accuracy; the [B, T, V] logits are never built | part of the 11.6% total |
| K3: `soup_addnorm_fwd_usig`, `soup_addnorm_bwd_usig`, bodies `soup_addnorm_fwd_kernel`, `soup_addnorm_bwd_kernel` | `solution/kernels/addnorm.py` (wiring `solution/layers.py`, `solution/fused_block.py`) | every `Qwen2RMSNorm` forward and backward; the first residual add fused with `post_attention_layernorm` | part of the 11.6% total |
| C2: `soup_swiglu_kernel` | `solution/kernels/swiglu.py` | the starter kit's two example SwiGLU kernels (forward, backward, in-place backward) | counted by the harness; same math as the example, so we do not rely on it |
| All together | | | **11.6%** (final run); 9.8% at v7 without a SwiGLU kernel of ours; threshold 5% |

The example kernels stay in `solution/swiglu.py` as the test reference and the flag-off path; with C2 on they are never launched.

Other changes, no new kernel: K4 fp16 MLP (`solution/mlp.py`), U unpadding (`solution/unpad.py`), S4 MLP recompute with C in-place SwiGLU backward (`solution/mlp_recompute.py`), B fused block (`solution/fused_block.py`), A1 attention recompute (`solution/sdpa_recompute.py`), A2 dropout replay (`solution/lora_replay.py`), D real-row LoRA inputs (`solution/saved_compact.py`), and new in v9b: M2 LoRA input recompute (`solution/lora_recompute.py`) and M3 real-row q, k, v in A1 (`solution/sdpa_recompute.py`). Details are in README.md, part 1.

Ideas taken from other kernels, with links (we wrote the code ourselves; no banned library is imported):
- bitsandbytes: NF4 table, nested absmax layout, rounding order, tile idea of `kDequantizeBlockwise` (K1). https://github.com/bitsandbytes-foundation/bitsandbytes
- Soup issue #1693 and the `ChunkedStep` protocol in `soup_cli/utils/stream_loss_chunk.py`, soup-cli 0.77.0 (K2). https://github.com/MakazhanAlpamys/Soup
- Liger-Kernel: fused linear + cross entropy, fused add + RMSNorm, fused SwiGLU (K2, K3, B, C2), idea only. https://github.com/linkedin/Liger-Kernel
- Unsloth: fused add + RMSNorm, fused SwiGLU (K3, B, C2), idea only. https://github.com/unslothai/unsloth
- Apple Cut Cross-Entropy: cross entropy without full logits (K2), idea only. https://github.com/apple/ml-cross-entropy
- Online softmax, Milakov and Gimelshein 2018 (K2). https://arxiv.org/abs/1805.02867
- ByteDance Effective Transformer and FlashAttention's varlen unpad path: removing padding (U). https://github.com/bytedance/effective_transformer
- Activation recomputation: Chen et al. 2016 (S4, B, A1, M2), https://arxiv.org/abs/1604.06174; Korthikanti et al. 2022 (A1), https://arxiv.org/abs/2205.05198; FlashAttention, Dao et al. 2022 (A1), https://arxiv.org/abs/2205.14135
- `torch.utils.checkpoint` (`preserve_rng_state`) and the Megatron-LM RNG tracker: replaying a dropout mask from the generator state (A2, M2).
- The starter kit's example SwiGLU: the operation and its derivative, reproduced in C2.

ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
Notebook version 16, `--repeats 2` (base, ours, ours, base), code `91d70d3`.

| | Base Soup | Ours (v9b) | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 86.8 (88.68, 84.87) | 67.9 (68.53, 67.23) | speedup 1.278x (21.8% less) |
| Peak GPU memory, MiB | 14,345 | 2,515 | base/ours 5.704x (82.5% less) |
| Eval loss (public rows) | 2.04965 | 2.04964 | -0.000% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed: check 4 11.6%, check 6 0.00 to 0.01%, check 8 time x0.782 and memory x0.175, check 9 clean, check 10 79 passed
- [x] `pytest solution -q`: 485 passed, 0 failed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

Path to v9b, each version against the base of its own session (s / m): example only 0.937 / 1.158; K1 + K4 1.029 / 1.382; v1 1.078 / 2.454; v2 1.259 / 2.855; v4 1.298 / 2.855; v5f 1.192 / 3.731; v6 out of memory at step 22; v6.1 1.211 / 5.074; v7 1.292 / 4.437; v8b 1.275 / 4.958; v9 1.212 / 5.796; **v9b 1.278 / 5.704**. Full table in README.md, part 3.

## Presentation covers
Which operations we sped up and why (NF4 dequantization, fp32 elementwise ops and norms, padding, the full-vocabulary logits, the fp32 MLP activations, and the per-layer tensors kept for backward), how K1, K2, K3 and C2 work, the PyTorch-level changes around them, the v6 memory leak and its fix, what we tried and dropped, results against the base (speed, memory, quality), the limits and next steps.

## Limitations
- **fp16 LM head off.** `STORE_HEAD_FP16` is in the code: 440 MiB less peak at no time cost (same session, order v8b, head, head, v8b: 2,453 against 2,893 MiB, 65.08 and 65.64 s against 65.43 and 66.69 s). Off by team decision, because the task forbids changing the model (section 8) and suggests repacking weights (section 9).
- **M5 off.** Chunked S4 backward and a smaller K2 tile (`USE_BWD_CHUNKING`) gave 40 MiB less peak for an estimated 0.5 s. It is the only one of the v9 changes (M2, M3, M5) that is not bit-exact.
- **C2 is the example's math.** If the jury does not count `soup_swiglu_kernel`, our share was 9.8% without it at v7, still above 5%.
- **Noise.** Base time ran 78 to 90 s across sessions; compare s, not seconds. v9b and v8b never ran in one session. Only sums are measured, not the split between components.
- **Memory bought with time.** S4, A1 and M2 recompute in backward; in one session v9 (M2, M3, M5) ran 2.2 s slower than v8b.
- **Where the speedup stops.** The GPU idles about 100 to 180 ms per step in Soup and TRL host code. Step 1 still compiles our kernels: 3.5 s against 1.8 s for the base at v8b.
- **Scope.** Soup's T4 fp16 path and Qwen2 only; other GPUs and models are untested.
- **Next.** Measure M2 and M3 one by one in a single session, split check 4 per kernel, and switch on the fp16 head if the rules allow it. M5 stays off unless a chunking with equal bits is found.
