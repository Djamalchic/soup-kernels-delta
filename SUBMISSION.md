# Submission: Team Delta

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-8cddb4ae-delta/commit/[TODO final commit]
  (code v10d = commit `6b56d31`, branch `v10d-candidate`, which is v10c `0b9e562` plus the compile overlap; v10c is v10b `69b0c95` with `STORE_HEAD_FP16 = True`; v10b is v9b `91d70d3` plus P2; the final commit adds only README.md and this file)
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
K1 and K2 are launched through `soup_*_usig` entries, K3 through `soup_addnorm_fwd_v10` and `soup_addnorm_bwd_v10` (P2). All share one twelve-slot runtime signature (C1), so Triton builds one launcher. The `_usig` entries inline the original kernel bodies unchanged, so check 3 reports the bodies as "not launched" and passes. The v9b K3 entries stay in the file for the flag-off path and are not launched while P2 is on.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| K1: `soup_nf4_dequant_usig`, body `soup_nf4_dequant_kernel` | `solution/kernels/nf4.py` | bitsandbytes NF4 dequantization in forward and backward; bit-exact | part of the total |
| K2: `soup_head_ce_usig`, body `soup_head_ce_kernel` | `solution/kernels/head_ce.py` | `lm_head` + cross entropy and TRL's logits-based entropy and accuracy, without building the [B, T, V] logits; reads the fp16 head in v10c | part of the total |
| K3: `soup_addnorm_fwd_v10`, `soup_addnorm_bwd_v10` (v9b: `soup_addnorm_*_usig`, bodies `soup_addnorm_*_kernel`) | `solution/kernels/addnorm.py` | every `Qwen2RMSNorm` forward and backward, the first residual add fused with `post_attention_layernorm`; with P2 also B's fp16 casts and the q/v LoRA epilogue | part of the total |
| C2: `soup_swiglu_kernel` | `solution/kernels/swiglu.py` | the starter kit's two example SwiGLU kernels (forward, backward, in-place backward) | counted by the harness; same math as the example, so we do not rely on it |
| All together | | | **11.3%** (final run); v10c 11.2%; v10b 12.8%; v9b 11.6%; 9.8% at v7 without a SwiGLU kernel of ours; threshold 5% |

The example kernels stay in `solution/swiglu.py` as the test reference and the flag-off path; with C2 on they are never launched.

Other changes, no new kernel (details in README.md, part 1): K4 fp16 MLP, U unpadding, S4 MLP recompute with C in-place SwiGLU backward, B fused block, A1 attention recompute, A2 dropout replay and D real-row LoRA inputs. From v9b: M2 LoRA input recompute and M3 real-row q, k, v. From v10b: P2 elementwise fusion (`USE_ELEMENTWISE_FUSION`), which moves B's casts and the q/v LoRA epilogue into launches we already make. From v10c: the fp16 head (`STORE_HEAD_FP16 = True`), which stores the frozen tied embedding and LM head in fp16 after an on-device check that fp16 holds every value exactly. New in v10d: the compile overlap (`USE_PRECOMPILE_OVERLAP`, `solution/precompile.py`), one background thread that compiles the five kernels above while Soup loads the model, and the first `Qwen2ForCausalLM.forward` joins it (60 s timeout, then normal compilation).

Rule note for the fp16 head: the organizers confirmed to us (relayed by our team captain on 31 October) that storing fp32 weights in fp16 is allowed as storage repacking, when a round trip from fp32 to fp16 and back to fp32 preserves the values exactly and the weight tie and the computation stay the same.

Rule note for the compile overlap: the organizers confirmed to us (relayed by our team captain on 31 October) that a background thread in the training process that compiles our kernels with `kernel.warmup` while Soup loads the model is allowed, if everything stays inside the measured run with an empty cache, nothing is carried between runs and no separate process starts. Nothing is launched on the GPU and no cache is shipped. The harness counts each warmup as a launch and never times it.

Ideas taken from other kernels, with links (we wrote the code ourselves; no banned library is imported):
- bitsandbytes: NF4 table, nested absmax layout, rounding order, tile idea of `kDequantizeBlockwise` (K1). https://github.com/bitsandbytes-foundation/bitsandbytes
- Soup issue #1693 and the `ChunkedStep` protocol in `soup_cli/utils/stream_loss_chunk.py`, soup-cli 0.77.0 (K2). https://github.com/MakazhanAlpamys/Soup
- Liger-Kernel: fused linear + cross entropy, fused add + RMSNorm, fused SwiGLU (K2, K3, B, C2), idea only. https://github.com/linkedin/Liger-Kernel
- Unsloth: fused add + RMSNorm, fused SwiGLU, LoRA scale and add in one pass (K3, B, C2, P2), idea only. https://github.com/unslothai/unsloth
- Apple Cut Cross-Entropy: cross entropy without full logits (K2), idea only. https://github.com/apple/ml-cross-entropy
- Online softmax, Milakov and Gimelshein 2018 (K2). https://arxiv.org/abs/1805.02867
- ByteDance Effective Transformer and FlashAttention's varlen unpad path: removing padding (U). https://github.com/bytedance/effective_transformer
- Activation recomputation: Chen et al. 2016 (S4, B, A1, M2), https://arxiv.org/abs/1604.06174; Korthikanti et al. 2022 (A1), https://arxiv.org/abs/2205.05198; FlashAttention, Dao et al. 2022 (A1), https://arxiv.org/abs/2205.14135
- `torch.utils.checkpoint` (`preserve_rng_state`) and the Megatron-LM RNG tracker: replaying a dropout mask from the generator state (A2, M2).
- The starter kit's example SwiGLU: the operation and its derivative, reproduced in C2.

ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
Notebook version 22, `--repeats 2` (base, ours, ours, base), code `6b56d31`.

| | Base Soup | Ours (v10d) | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 84.9 (86.24 and 83.54) | 64.8 (65.43 and 64.13) | speedup 1.310 (wall 23.7% shorter) |
| Peak GPU memory, MiB | 14,345 | 2,335 | base/ours 6.144 (83.7% less) |
| Eval loss (public rows) | 2.04964 | 2.04963 | −0.000% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed: check 3 lists `soup_addnorm_bwd_v10`, `soup_addnorm_fwd_v10`, `soup_head_ce_usig`, `soup_nf4_dequant_usig` and `soup_swiglu_kernel`; check 4 11.3%; check 6 0.00 to 0.01%; check 8 time x0.763, memory x0.163; check 9 clean; check 10 79 passed. v10c passed all 10 too.
- [x] `pytest solution -q`: 600 passed, 0 failed (600 in notebook version 21; v10c: 578 passed, 0 failed; v10b: 578; v9b: 485). The run's `triton_cache` holds one `__triton_launcher*.so` and 5 cubins (v10b: the same). We did not read the head's store counts, so we do not claim them.
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

Previous version, v10c (notebook version 20, `--repeats 2`, code `0b9e562`): base 89.11 and 86.11 s (87.6), ours 65.56 and 67.02 s (66.3), speedup 1.322 (wall 24.3% shorter); peak 14,345 against 2,335 MiB, base/ours 6.144 (83.7% less); eval loss 2.04962 against 2.04964 (+0.001% vs base).

Path to v10d, each version against the base of its own session (s / m): example only 0.937 / 1.158; K1 + K4 1.029 / 1.382; v1 1.078 / 2.454; v2 1.259 / 2.855; v4 1.298 / 2.855; v5f 1.192 / 3.731; v6 out of memory at step 22; v6.1 1.211 / 5.074; v7 1.292 / 4.437; v8b 1.275 / 4.958; v9 1.212 / 5.796; v9b 1.278 / 5.704 (second run, slower session: 1.244 / 5.704); v10b 1.254 / 5.750; v10c 1.322 / 6.144; **v10d 1.310 / 6.144**. The full table and the scouting runs for the fp16 head, P2 and the compile overlap are in README.md, part 3.

## Presentation covers
Which operations we sped up and why (NF4 dequantization, fp32 elementwise ops and norms, padding, the full-vocabulary logits, the fp32 MLP activations, and the per-layer tensors kept for backward), how K1, K2, K3 and C2 work, the PyTorch-level changes around them, P2 (casts and the LoRA epilogue inside K3's launches), the fp16 head and the compile overlap with the organizers' answers on them, the v6 memory leak and its fix, what we tried and dropped (including P1's pool joined inside `apply()` and P3 merged LoRA), results against the base (speed, memory, quality), the limits and next steps.

## Limitations
- **fp16 head.** Stored in fp16 only after an on-device lossless check that matches the organizers' condition; `STORE_HEAD_FP16 = False` (with the overlap off too) gives v10b. Peak savings: 440 MiB on v8b code (no time cost in an ABBA session), 140 MiB on v9 code, and 160 MiB against v10b in the final run, which is our evidence that the store ran (we did not read the store counts). Each comes from one session and is below the 445 MiB freed on paper.
- **Compile overlap.** One ABBA session (notebook version 21), two runs per version: 2.1 s less on the means (v10c 66.23 and 67.08 s, v10d 63.97 and 65.11 s) against repeat spreads of 0.85 and 1.14 s; the peak stayed at 2,335 MiB. `USE_PRECOMPILE_OVERLAP = False` gives v10c.
- **M5 off.** Chunked S4 backward and a smaller K2 tile (`USE_BWD_CHUNKING`) gave 40 MiB less peak for an estimated 0.5 s and are not bit-exact.
- **C2 is the example's math.** If the jury does not count `soup_swiglu_kernel`, our share was 9.8% without it at v7, still above 5%.
- **Noise.** Base time ran 77 to 90 s across full runs; compare s, not seconds. The two full v9b runs gave s = 1.278 and 1.244, a spread larger than P2's gain (one pair of runs) and the head's time effect (one ABBA session). So we do not credit the head with the rise from v10b's 1.254 to 1.322. In the v10c run our repeats differed by 1.46 s and the base repeats by 3.00 s; in the v10d run by 1.30 s and 2.70 s. Only sums are measured, not per-component splits.
- **P2 and v9b's bits.** P2 is bit-exact against the torch ops it replaces and is built to give v9b's bits; the suite, including the v9b-bits tests, passed for v10b and v10c (578 passed, 0 failed).
- **Memory bought with time.** S4, A1 and M2 recompute in backward; v9 (M2, M3, M5) ran 2.2 s slower than v8b in one session.
- **Where the speedup stops.** The GPU idles about 100 to 180 ms per step in Soup and TRL host code. Step 1 compiled our kernels (3.5 s against 1.8 s for the base at v8b). P1 as a pool joined inside `apply()` moved that cost there and gained nothing; v10d overlaps it with model loading.
- **Scope.** Soup's T4 fp16 path and Qwen2 only; the head stays fp32 unless every element is fp16-exact under fp16 mixed precision. Other GPUs and models are untested.
- **Next.** Measure M2, M3 and P2's three flags one by one in a single session, and split check 4 per kernel. M5 stays off unless a chunking with equal bits is found.
