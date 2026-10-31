# Submission: Team Delta

## Links
- Repository, final commit: tag `submission`, https://github.com/BAITC-Hacks/hack-8cddb4ae-delta/tree/submission; public mirror with the same commits and tag: https://github.com/Djamalchic/soup-kernels-delta/tree/submission (the commit hash is shown on that page and in the notebook's first cell)
  - Code v11 = commit `f218f86`, branch `v11-candidate`. The final commit changes only documentation: README.md, this file, VERSIONS.md (every version's branch), and comments and docstrings in solution/.
  - Lineage: v9b `91d70d3` + P2 = v10b `69b0c95`; + `STORE_HEAD_FP16 = True` = v10c `0b9e562`; + compile overlap = v10d `6b56d31`; + Q, the merged q, k, v GEMM = v11 `f218f86`.
- Public Kaggle notebook, saved version with `COMMIT` = the final commit, also linked in README.md: https://www.kaggle.com/code/jamildeltacaptain/notebook0289d149f3
- Demo video, up to 3 minutes: [?: link]
- Presentation: [?: link]

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Jamil Tlegenov | Captain | [?] | jamildeltacaptain | [?] |
| [?] | [?] | [?] | [?] | |

## Our kernels
All five launched kernels share one twelve-slot runtime signature (C1), so Triton builds one launcher. K1 and K2 launch `soup_*_usig` entries, which inline the original kernel bodies unchanged; check 3 therefore reports the bodies as "not launched" and passes. K3 launches `soup_addnorm_fwd_v10` and `soup_addnorm_bwd_v10` (P2); the v9b K3 entries stay in the file for the flag-off path and do not launch while P2 is on. v11 adds no kernel: Q is a cuBLAS GEMM on a merged weight.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of training time (check 4) |
|---|---|---|---|
| K1: `soup_nf4_dequant_usig`, body `soup_nf4_dequant_kernel` | `solution/kernels/nf4.py` | bitsandbytes NF4 dequantization in forward and backward; bit-exact | part of the total |
| K2: `soup_head_ce_usig`, body `soup_head_ce_kernel` | `solution/kernels/head_ce.py` | `lm_head` + cross entropy and TRL's logits-based entropy and accuracy, without the [B, T, V] logits; reads the fp16 head since v10c | part of the total |
| K3: `soup_addnorm_fwd_v10`, `soup_addnorm_bwd_v10`; in v9b `soup_addnorm_*_usig` with bodies `soup_addnorm_*_kernel` | `solution/kernels/addnorm.py` | every `Qwen2RMSNorm` forward and backward and the first residual add fused with `post_attention_layernorm`; with P2 also B's fp16 casts and the q and v LoRA epilogue | part of the total |
| C2: `soup_swiglu_kernel` | `solution/kernels/swiglu.py` | the starter kit's two example SwiGLU kernels: forward, backward, in-place backward | counted by the harness; same math as the example, so we do not rely on it |
| All together | | | **11.5%** in the final run; v10d 11.3%, v10c 11.2%, v10b 12.8%, v9b 11.6%; 9.8% at v7 without a SwiGLU kernel of ours; threshold 5% |

The example kernels stay in `solution/swiglu.py` as the test reference and the flag-off path; with C2 on they never launch.

Other changes, no new kernel (README.md, part 1):
- K4 fp16 MLP, U unpadding, S4 MLP recompute with C (in-place SwiGLU backward), B fused block, A1 attention recompute, A2 dropout replay, D real-row LoRA inputs.
- From v9b: M2 LoRA input recompute and M3 real-row q, k, v.
- From v10b: P2 elementwise fusion (`USE_ELEMENTWISE_FUSION`), which moves B's casts and the q and v LoRA epilogue into launches we already make.
- From v10c: the fp16 head (`STORE_HEAD_FP16 = True`), which stores the frozen tied embedding and LM head in fp16 after an on-device check that fp16 holds every value exactly.
- From v10d: the compile overlap (`USE_PRECOMPILE_OVERLAP`, `solution/precompile.py`). One background thread compiles the five kernels while Soup loads the model; the first `Qwen2ForCausalLM.forward` joins it, with a 60 s timeout.
- From v11: Q, the merged q, k, v GEMM (`USE_MERGED_QKV = True`, `MERGED_QKV_CACHE = False`; `solution/merged_qkv.py`, wired in `solution/unpad.py` and `solution/lora_recompute.py`). On the packed attention path the three base projections run as one fp16 GEMM against [Wq; Wk; Wv] (2048 by 1536) with the three biases concatenated; the weight is assembled per call from K1's dequant (q) and I3's cached rows (k, v). The output is split into q, k, v, and backward computes dX as one GEMM. The q and v LoRA branches run exactly as before, with the same dropout draws in the same order, so the RNG stream is unchanged. Guards: exact classes, biases present and frozen, fp16 autocast, no hooks, shapes 1536, 256, 256; a miss runs the v10d path.

**Rule notes.** On 31 October the organizers confirmed to us, through our team captain, two points:
- Storing fp32 weights in fp16 is allowed as storage repacking when a round trip from fp32 to fp16 and back preserves the values exactly and the weight tie and the computation stay the same.
- A background thread in the training process that compiles our kernels with `kernel.warmup` while Soup loads the model is allowed if everything stays inside the measured run with an empty cache, nothing is carried between runs and no separate process starts. Our thread launches nothing on the GPU and ships no cache; the harness counts each warmup as a launch and never times it.

Ideas from other work, with links; we wrote the code ourselves and import no banned library.
- bitsandbytes: NF4 table, nested absmax layout, rounding order, tile idea of `kDequantizeBlockwise` (K1). https://github.com/bitsandbytes-foundation/bitsandbytes
- Soup issue #1693 and the `ChunkedStep` protocol in `soup_cli/utils/stream_loss_chunk.py`, soup-cli 0.77.0 (K2). https://github.com/MakazhanAlpamys/Soup
- Liger-Kernel: fused linear + cross entropy, fused add + RMSNorm, fused SwiGLU (K2, K3, B, C2). https://github.com/linkedin/Liger-Kernel
- Unsloth: fused add + RMSNorm, fused SwiGLU, LoRA scale and add in one pass (K3, B, C2, P2); its LoRA QKV path, idea only (Q). https://github.com/unslothai/unsloth
- Apple Cut Cross-Entropy: cross entropy without full logits (K2). https://github.com/apple/ml-cross-entropy
- Online softmax, Milakov and Gimelshein 2018 (K2). https://arxiv.org/abs/1805.02867
- ByteDance Effective Transformer and FlashAttention's varlen unpad path: removing padding (U). https://github.com/bytedance/effective_transformer
- Activation recomputation: Chen et al. 2016 (S4, B, A1, M2), https://arxiv.org/abs/1604.06174; Korthikanti et al. 2022 (A1), https://arxiv.org/abs/2205.05198; FlashAttention, Dao et al. 2022 (A1), https://arxiv.org/abs/2205.14135
- `torch.utils.checkpoint` (`preserve_rng_state`) and the Megatron-LM RNG tracker: replaying a dropout mask from the generator state (A2, M2).
- Megatron-LM's fused `query_key_value` linear: one GEMM for q, k, v (Q), idea only.
- The starter kit's example SwiGLU: the operation and its derivative, reproduced in C2.

ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow.

## Results on Kaggle T4
From `runs/result.json` and `runs/checks.json` of the notebook above, saved version 26 (`COMMIT` = `6376c0a`, code `f218f86`), `--repeats 2` in the order base, ours, ours, base. Version 24 ran the same code one session earlier: 64.1 (64.27 and 63.97) against 82.9 (82.02 and 83.83) s, speedup 1.293, the same peak and checks.

| | Base Soup | Ours (v11) | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 87.6 (90.57 and 84.65) | 65.1 (64.46 and 65.76) | speedup 1.346, wall 25.7% shorter |
| Peak GPU memory, MiB | 14,345 | 2,335 | base/ours 6.144, 83.7% less |
| Eval loss, public rows | 2.04965 | 2.04965 | +0.000% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed: check 4 11.4% (11.5% in version 24); check 8 time x0.743, memory x0.163. Q adds no Triton kernel. In the v11 runs (versions 24 and 26) check 3 listed `soup_addnorm_bwd_v10`, `soup_addnorm_fwd_v10`, `soup_head_ce_usig`, `soup_nf4_dequant_usig` and `soup_swiglu_kernel`; check 6 read 0.00 to 0.01%; check 9 was clean; check 10 had 79 passed. v10c passed all 10 too.
- [x] `pytest solution -q`: 623 passed, 0 failed; v10d 600 (also in notebook version 21), v10c and v10b 578, v9b 485, each with 0 failed. In v10d's run the `triton_cache` held one `__triton_launcher*.so` and 5 cubins, as in v10b. We did not read the head's store counts, so we do not claim them.
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits.

Previous version, v10d: notebook version 22, `--repeats 2`, code `6b56d31`. Ours 65.43 and 64.13 s (64.8) against base 86.24 and 83.54 s (84.9), speedup 1.310, wall 23.7% shorter; peak 2,335 against 14,345 MiB, base/ours 6.144; eval loss 2.04963 against 2.04964, −0.000% vs base.

v11's speedup moves with the session's base: 1.346 in version 26 (base 87.6 s), 1.293 in version 24 (base 82.9 s), against v10d's 1.310 (base 84.9 s). The same-session comparison is the reliable one: notebook version 23, ABBA, v10d 65.84 and 65.69 s, v11 65.24 and 65.19 s, 0.55 s less; peak 2,335 MiB in both.

README.md, part 3, has the path from the example only (s = 0.937, m = 1.158) to v11, one row per version, and the same-session runs for the fp16 head, P2, the compile overlap and Q.

## Presentation covers
Which operations we sped up and why, how K1, K2, K3 and C2 work, the PyTorch-level changes around them, P2, the fp16 head, the compile overlap with the organizers' answers, Q (the merged q, k, v GEMM), the v6 leak and its fix, what we dropped (P1, P3, Q's kept weight), results against the base, limits and next steps.

## Limitations
- **Q, merged q, k, v GEMM.** One ABBA session, notebook version 23, two runs per version: 0.55 s less on the means (v10d 65.84 and 65.69 s, v11 65.24 and 65.19 s), about 0.8%; the peak stayed at 2,335 MiB. Not bit-exact with v10d: cuBLAS may pick a different kernel for N = 2048 and dX is rounded once instead of three times. Tests hold hidden states within 5e-3, loss within 1e-4 and LoRA gradients within 2e-2 (relative norms), and compare the RNG state bit for bit. Its fp16 dX is infinite where the sum of the three terms exceeds 65,504, while in v10d a term is infinite only where it alone does; by the code's analysis the q term dominates, so the window is narrow, and a gradient-scaler skip could come one scale step earlier. We did not measure this. `USE_MERGED_QKV = False` gives v10d. `MERGED_QKV_CACHE = True` (kept weight, 126 MiB more, an estimated 1.7 ms per step) is off and not measured.
- **fp16 head.** Stored in fp16 only after an on-device lossless check that matches the organizers' condition; `STORE_HEAD_FP16 = False` with Q and the overlap off gives v10b. Peak savings: 440 MiB on v8b code with no time cost in an ABBA session, 140 MiB on v9 code, 160 MiB against v10b in the final run, our evidence that the store ran. Each comes from one session and is below the 445 MiB freed on paper.
- **Compile overlap.** One ABBA session, notebook version 21, two runs per version: 2.1 s less on the means (v10c 66.23 and 67.08 s, v10d 63.97 and 65.11 s) against repeat spreads of 0.85 and 1.14 s; the peak stayed at 2,335 MiB. `USE_PRECOMPILE_OVERLAP = False` with Q off gives v10c.
- **M5 off.** Chunked S4 backward and a smaller K2 tile (`USE_BWD_CHUNKING`) gave 40 MiB less peak for an estimated 0.5 s and are not bit-exact.
- **C2 is the example's math.** If the jury does not count `soup_swiglu_kernel`, our share was 9.8% without it at v7, still above 5%.
- **Noise.** Base time ran 77 to 90 s across full runs, so compare within a session, not seconds across sessions; even s moves with the base (v11 reads 1.346 in version 26 and 1.293 in version 24 because the base ran 87.6 and 82.9 s). The two full v9b runs gave s = 1.278 and 1.244, more than P2's gain or the head's time effect, so we do not credit the head with the rise from v10b's 1.254 to 1.322. Repeats differed by 1.46 s for us and 3.00 s for the base in the v10c run, by 1.30 s and 2.70 s in the v10d run, by 0.30 s and 1.81 s in the v11 run of version 24, and by 1.30 s and 5.92 s in version 26. Only sums are measured, not per-component splits.
- **P2 and v9b's bits.** P2 is bit-exact against the torch ops it replaces and built to give v9b's bits; the suite, including the v9b-bits tests, passed for v10b and v10c.
- **Memory bought with time.** S4, A1 and M2 recompute in backward; v9 with M2, M3 and M5 ran 2.2 s slower than v8b in one session.
- **Where the speedup stops.** The GPU idles about 100 to 180 ms per step in Soup and TRL host code. Step 1 compiled our kernels in 3.5 s against 1.8 s for the base at v8b. P1, a pool joined inside `apply()`, moved that cost there and gained nothing; v10d overlaps it with model loading.
- **Scope.** Soup's T4 fp16 path and Qwen2 only; Q only for q, k, v of sizes 1536, 256, 256, other shapes run the v10d path; the head stays fp32 unless every element is fp16-exact under fp16 mixed precision. Other GPUs and models are untested.
- **Next.** Measure M2, M3, P2's three flags and Q's kept-weight variant one by one in a single session, and split check 4 per kernel. M5 stays off unless a chunking with equal bits is found.
