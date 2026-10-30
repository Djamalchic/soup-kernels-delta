# Submission: Team Delta

## Links
- Repository, final commit on main: https://github.com/BAITC-Hacks/hack-8cddb4ae-delta/commit/[TODO final commit]
  (the submitted code is v8b, commit `599c58a`, branch `v8b-candidate`, which becomes `main`; the final commit on main adds our README and this file)
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in our README):
  [TODO final link]
- Demo video (up to 3 minutes): [TBD: link]
- Presentation: [TBD: link]

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Jamil Tlegenov | Captain | [TBD] | jamildeltacaptain | [TBD] |
| [TBD] | [TBD] | [TBD] | [TBD] | |
| [TBD] | [TBD] | [TBD] | [TBD] | |
| [TBD] | [TBD] | [TBD] | [TBD] | |

## Our kernels
K1, K2 and K3 are launched through `*_usig` entries (change C1), which are also `@triton.jit` functions in `solution/kernels/` and inline the v7 kernel body unchanged.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of training time (check 4) |
|---|---|---|---|
| `soup_nf4_dequant_usig`, body `soup_nf4_dequant_kernel` (K1) | `solution/kernels/nf4.py` | bitsandbytes NF4 dequantization on its dequantize + `F.linear` path (forward and backward re-dequantization), bit-exact | Counted in the K1 to K3 total below; 3.27 s GPU per run at v6.1, before I3 |
| `soup_head_ce_usig`, body `soup_head_ce_kernel` (K2) | `solution/kernels/head_ce.py` (wiring: `solution/head_loss.py`) | `lm_head` + `ForCausalLMLoss` in `Qwen2ForCausalLM.forward`, and TRL's logits-based entropy and accuracy metrics; the [B, T, V] logits are never materialized | Counted in the K1 to K3 total below; 0.98 s GPU per run at v6.1 |
| `soup_addnorm_fwd_usig`, `soup_addnorm_bwd_usig`, bodies `soup_addnorm_fwd_kernel`, `soup_addnorm_bwd_kernel` (K3) | `solution/kernels/addnorm.py` (wiring: `solution/layers.py`, `solution/fused_block.py`) | Every `Qwen2RMSNorm` forward and backward, and the first residual add of each decoder layer fused with `post_attention_layernorm` | Counted in the K1 to K3 total below; 0.66 s GPU per run at v6.1 |
| `soup_swiglu_kernel` (C2) | `solution/kernels/swiglu.py` | The starter kit's two example SwiGLU kernels in K4 and S4: forward, backward and in-place backward in one jit | **Not counted by us.** Same SwiGLU math, operation for operation, as the example, so the jury may treat it as the example. |
| K1 to K3 together | | | 11.5 % at `599c58a` (final run; the harness counts `soup_swiglu_kernel` in it); 9.8 % at `80c3606` (v7 with S4 on all layers, measured; no C2 kernel in it); 10.6 % at v6.1; threshold 5 % |

Our share is above 5 % without C2. The 9.8 % run already had I1 to I3 and the same kernel bodies; v8b adds only C1 and C2. The final run reports 11.5 % because the harness counts `soup_swiglu_kernel` too; we measured the share without it only at `80c3606`. The threshold holds either way. The per-kernel GPU times above are from v6.1, before I3 (which caches the k and v weights and may lower K1's time).

Check 3 in the final run lists `soup_addnorm_bwd_usig`, `soup_addnorm_fwd_usig`, `soup_head_ce_usig`, `soup_nf4_dequant_usig` and `soup_swiglu_kernel`.

Not counted as ours: the starter kit's example SwiGLU kernels. With C2 on they are never launched; they stay in `solution/swiglu.py` as the test reference and the flag-off path (`USE_SOUP_SWIGLU = False`).

Non-kernel changes (no Triton kernel in any of them):
- U, `solution/unpad.py`: RMSNorms, residual adds and the MLP run on the real tokens only; attention keeps [B, T, C], so the LoRA dropout masks and RNG state match the base.
- K4, `solution/mlp.py`: fp16 MLP path; autograd keeps gate and up in fp16 instead of three fp32 tensors per layer.
- S4, `solution/mlp_recompute.py`, flag `USE_MLP_RECOMPUTE`: the MLP saves only its fp16 input and recomputes gate and up in backward, bit-identical to K4. Our own autograd recompute, not Soup's `gradient_checkpointing`, which stays off. Saves 1,180 MiB, costs about 5.4 s (v4 against v5f, different sessions).
- B, `solution/fused_block.py`, flag `USE_FUSED_BLOCK`: add + norm + S4 MLP + add as one autograd node saving (r, fp16 a, rstd). Estimated saving 213 MiB [estimate].
- A1, `solution/sdpa_recompute.py`, flag `USE_SDPA_RECOMPUTE`: attention saves fp16 q and un-repeated k and v and recomputes torch's SDPA in backward. Estimated saving 507 MiB [estimate].
- A2, `solution/lora_replay.py`, flag `USE_LORA_DROPOUT_REPLAY`: LoRA dropout saves the generator state, not the mask. Estimated saving 162 MiB [estimate].
- D, `solution/saved_compact.py`, flag `USE_LORA_ROW_COMPACT`: LoRA-A inputs keep the real rows only (public `saved_tensors_hooks`, nothing offloaded). Estimated saving 115 MiB [estimate].
- C, `solution/mlp_recompute.py`, flag `INPLACE_SWIGLU_BACKWARD`: the SwiGLU backward writes over its own inputs.
- Changed since v6.1: I1 `USE_SDPA_CAUSAL_PADDED` (A1 uses `is_causal=True`, no bool mask, on the packed path); I2 `S4_DEQUANT_REUSE` (gate and up dequantized once in S4 backward, about 26 MiB transient [estimate]); I3 `USE_K1_FAST_HOST` and `USE_DENSE_CACHE` (K1 host fast path and a permanent fp16 cache of the `k_proj` and `v_proj` weights, about 42 MiB); S4 on all 28 layers (`S4_OFF_FRACTION = 0.0`; v7's 0.25 gave no speed gain and 340 MiB more peak); C1 `USE_UNIFIED_SIG` (one runtime signature, so Triton builds one launcher instead of six; step 1 fell from 5.1 s to 3.6 s in a same-session comparison); C2 `USE_SOUP_SWIGLU` (see the table); test fixes (LoRA gradient tolerance 2e-2 with the forward still bit for bit, held-memory check on requested bytes, k/v cache checked against a fresh dequantization).
- Switched off, not used in any measured run: `STORE_HEAD_FP16` and `USE_SDPA_KV_COMPACT`.

Ideas taken from other work (we wrote the code ourselves; none of the banned libraries is imported):
- bitsandbytes: NF4 format, double quantization layout, rounding order, tile idea of `kDequantizeBlockwise` (K1). https://github.com/bitsandbytes-foundation/bitsandbytes
- Soup issue #1693 and the `ChunkedStep` protocol in `soup_cli/utils/stream_loss_chunk.py` of soup-cli 0.77.0 (K2). Soup repository: https://github.com/MakazhanAlpamys/Soup. Issue: #1693 in that repository.
- Liger-Kernel: fused linear + cross entropy, fused add + RMSNorm, fused SwiGLU (K2, K3, B, C2), idea only. https://github.com/linkedin/Liger-Kernel
- Unsloth: fused add + RMSNorm, fused SwiGLU (K3, B, C2), idea only. https://github.com/unslothai/unsloth
- Apple Cut Cross-Entropy: cross entropy without full logits (K2), idea only. https://github.com/apple/ml-cross-entropy
- Online softmax, Milakov and Gimelshein 2018 (K2). https://arxiv.org/abs/1805.02867
- ByteDance Effective Transformer and the unpad path of FlashAttention's varlen BERT: removing padding (U). https://github.com/bytedance/effective_transformer
- Chen et al. 2016, "Training Deep Nets with Sublinear Memory Cost": activation recomputation (S4, B, A1). https://arxiv.org/abs/1604.06174
- FlashAttention, Dao et al. 2022: recompute attention intermediates in backward (A1), idea only. https://arxiv.org/abs/2205.14135
- Korthikanti et al. 2022, "Reducing Activation Recomputation in Large Transformer Models": selective recomputation (A1). https://arxiv.org/abs/2205.05198
- `torch.utils.checkpoint` (`preserve_rng_state`) and the Megatron-LM CUDA RNG tracker: restoring the generator state (A2), idea only.
- Starter kit example SwiGLU: the operation and its derivative, reproduced in C2 with our own code; the example file stays as the test reference.
- Triton's launcher generation (C1): described from memory of the 3.x source; `solution/tests/test_compile_probe.py` checks the effect on Triton 3.6.

AI assistants: ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow. Every Triton kernel was run with `pytest solution -q` and measured on a Kaggle T4. K1 v3s went through a Kaggle microbenchmark before it entered `solution/`. Drafts of several components were written by both assistants in parallel and compared; the chosen version was reviewed against the library sources and measured on Kaggle.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
Jury-style run of v8b (`--repeats 2`: base, solution, solution, base; notebook version 11, `COMMIT = 599c58a`, branch `v8b-candidate`):

| | Base Soup | Ours (v8b) | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 88.4 (89.39 and 87.44) | 69.3 (67.49 and 71.22) | s = 1.275 |
| Peak GPU memory, MiB | 14,345 | 2,893 | m = 4.958 |
| Eval loss (public rows) | 2.04963 | 2.04964 | +0.000 % |
| Share of own kernels (check 4) | n/a | 11.5 % | threshold 5 % |

The wall time is 21.6 % shorter and the peak 79.8 % lower. All 10 checks: PASS (check 4: 11.5 %; check 6: 0.00 to 0.01 %; check 8: time x0.784, memory x0.202; check 9: clean; check 10: 79 passed). `pytest solution -q` at `599c58a`: 397 passed, 0 failed.

Same session, v6.1 against v8b (notebook version 10: one base run, two runs of each version; v8b at `8eebfbf`, the same training path as `599c58a`: `git diff 8eebfbf 599c58a -- solution/` shows only the removed, switched-off per-row SDPA code, where `USE_SDPA_PER_ROW` was `False` in `8eebfbf`, and its tests):

| | Base | v6.1 (`0c89204`) | v8b (`8eebfbf`) |
|---|---|---|---|
| Time of `soup train`, s | 79.81 | 67.7 (68.28 and 67.17) | 63.3 (63.17 and 63.46) |
| Training loop, s | 55.00 | 44.6 (45.03 and 44.19) | 40.3 (40.09 and 40.47) |
| Step 1, with compilation, s | 1.78 | 5.8 (6.63 and 4.98) | 3.5 (3.55 and 3.51) |
| Peak GPU memory, MiB | 14,345 | 2,827 | 2,893 |
| s | 1.000 | 1.18 | 1.26 |
| m | 1.000 | 5.07 | 4.96 |

v8b is 4.4 s (6.5 %) shorter than v6.1 and uses 66 MiB more peak. Base time varies from session to session (79.8 to 90.2 s in our `--repeats 2` and head-to-head runs), so compare s inside a session.

Earlier versions, each against the base of its own session:

| Version | Base, s | Ours, s | Peak, MiB | s | m |
|---|---|---|---|---|---|
| K1 v1 + K4 (`a2c85c6`) | 84.96 | 82.55 | 10,383 | 1.029 | 1.382 |
| v1 = + K3 + K2 (`c2b1ff1`) | 78.22 | 72.54 | 5,845 | 1.078 | 2.454 |
| v2 = K1 v3s + U (`30e62a7`) | 77.06 | 61.22 | 5,025 | 1.259 | 2.855 |
| v4, S4 off (`c85343f`, `--repeats 2`) | 79.8 | 61.5 | 5,025 | 1.298 | 2.855 |
| v5f, S4 on (`47a279c`, `--repeats 2`) | 79.8 | 66.9 | 3,845 | 1.192 | 3.731 |
| v6 (`df2557b`) | 84.8 | out of memory at step 22 of 64 | 14,903 | n/a | n/a |
| v6.1 (`6fff364`, measured at `ebeec3a`, `--repeats 2`) | 90.2 | 74.5 | 2,827 | 1.211 | 5.074 |
| v7 (`cb7d630`, `--repeats 2`, not submitted) | 84.1 | 65.1 | 3,233 | 1.292 | 4.437 |
| v7 with S4 on all layers (`80c3606`, `--repeats 2`) | 80.6 | 65.6 | 2,893 | 1.230 | 4.958 |
| **v8b (`599c58a`, `--repeats 2`)** | 88.4 | 69.3 | 2,893 | 1.275 | 4.958 |

What we tried and dropped:
- Per-row causal SDPA (called B in our scouting notes, not the fused block B): SDPA on each row's real prefix. Attention GPU time fell from 56 to 26 ms per step, but about 1,700 small CPU-side operations per step were added, and wall time did not fall (62.93 s without it, 63.17 s with it, same session). One of its tests also failed. Dropped; `599c58a` has none of its code.
- I4, S4 off on some layers: 71.95 s with S4 on all layers, 72.39 and 70.65 s with 7 layers off, 71.70 s with 2 off (same-session ladder). No gain beyond noise; the peak was 3,233 MiB with 7 layers off against 2,893 MiB. Dropped to keep the memory.
- The v6 leak: both solution runs died at step 22 of 64 with "GPU ran out of memory", growing by about 0.5 GB per step. A1's forward builds a throw-away SDPA graph that never gets a backward, and D's pack hook returned saved tensors as they were, so SDPA's own outputs were stored together with the node that owns them: a reference cycle that only that backward breaks. v6.1 runs A1's inner calls under private hooks, makes D's hooks return `t.detach()`, and adds `solution/tests/test_v6_no_leak.py` (flat memory over 8 training steps). Confirmed on Kaggle: v6.1 ran all 64 steps at a flat 2,827 MiB peak.
- fp16 copy of the LM head (`STORE_HEAD_FP16`): about 445 MiB less static memory [estimate], never in a measured run. Kept off because we are not sure the rules allow it.

- [x] All 10 checks of `harness/check_rules.py` passed at v8b (`599c58a`, notebook version 11): 10 of 10 PASS.
- [x] `pytest solution -q` at v8b (`599c58a`): 397 passed, 0 failed.
- [x] README.md for our solution (what we did, how to run it, results, limits).

## Presentation covers
Which operations we sped up and why (NF4 dequantization, fp32 elementwise ops and norms, padding, the full-vocabulary logits, the fp32 MLP activations, and the per-layer attention and LoRA activations that dominate the rest of the peak), how K1 to K4, U, S4, B, A1, A2, D, C, I1 to I3, C1 and C2 work, the v6 memory leak and its fix, what we tried and dropped, results against the base (speed, memory, quality) and the limits.

## Limitations
- The 63.3 s and the 4.4 s gain over v6.1 come from a scouting run: one session, two runs per version, one base run. The two runs of each version differ by 1.1 s (v6.1) and 0.3 s (v8b). The jury-style numbers (notebook version 11) are in the first table: s = 1.275, m = 4.958, all 10 checks PASS, 397 tests passed. In that run our two repeats differ by 3.7 s (67.49 and 71.22 s) and the two base repeats by 1.95 s (89.39 and 87.44 s). The previous full run was `80c3606`: all 10 checks passed, s = 1.230, m = 4.958.
- The savings of B, A1, A2, D, C and I1 to I3 are estimates; only sums are measured. The 4.4 s are not split between I1 to I3, C1 and C2.
- C1 and C2 help the compile step only: after step 1 the steps were not faster in one same-session run (35.7 s against 36.6 s for steps 2 to 64, inside the noise). Step 1 is still 3.5 s against 1.8 s for the base. In the version 11 run the solution's `triton_cache` holds exactly one `__triton_launcher*.so` and 5 `*.cubin`, so C1 does give one launcher.
- C2 computes the same math as the starter kit's example, so the jury may not count it toward check 4; we do not count it, although the harness does (11.5 % with it). Check 3 in the version 11 output notes the bodies `soup_*_kernel` as "not launched in every solution run", because they run as inlined device functions inside the `*_usig` entries; the check still passes.
- A1 relies on torch's SDPA backward, which may not be deterministic on a T4. I1 may change real rows by fp16 ulps. K3's fp16-pointer variant changes the normalized output by a few fp32 ulp against v5f; the 2e-6 test tolerance is an estimate. C1 and C2 keep base values and are tested with `torch.equal`. Eval loss was within +0.001 % of the base for v6.1 and v7, -0.002 % for `80c3606` and +0.000 % for v8b (2.04964 against 2.04963).
- Of the 63.3 s of the same-session run, 40.3 s are the training loop; the other 23.0 s (base: 24.8 s) lie outside it. In scouting profiles the GPU was idle 100 to 180 ms of each 0.56 to 0.72 s step, mostly Soup and Trainer host work. The 7.4 s that the harness reports as `apply()` time are imports that Soup does anyway; `apply()` itself costs 0.02 s.
- The I3 cache assumes nobody writes the NF4 bytes in place. Nothing in Soup, peft or bitsandbytes does.
- Built and measured for Soup's T4 fp16 path and Qwen2 only. Guards check classes, dtypes and shapes, mostly not the GPU model; where Soup picks bf16, K1, K4, S4 and B fall back to the original code. Other GPUs are not tested.
- Next steps: split the per-kernel share of check 4 and measure I1 to I3 one by one.
