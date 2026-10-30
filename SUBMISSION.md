# Submission: Team Delta

## Links
- Repository, final commit on main: https://github.com/BAITC-Hacks/hack-8cddb4ae-delta/commit/[TODO final commit]
  (the submitted code is v6.1, commit `6fff364`, branch `v61-candidate`, which becomes `main`; the final commit on main adds our README and this file)
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
| Kernel (`@triton.jit` name) | File | What it replaces | Share of training time (check 4) |
|---|---|---|---|
| `soup_nf4_dequant_kernel` (K1) | `solution/kernels/nf4.py` | bitsandbytes NF4 dequantization on its dequantize + `F.linear` path (forward and backward re-dequantization), bit-exact | 3.27 s GPU per run |
| `soup_head_ce_kernel` (K2) | `solution/kernels/head_ce.py` (wiring: `solution/head_loss.py`) | `lm_head` + `ForCausalLMLoss` in `Qwen2ForCausalLM.forward`, and TRL's logits-based entropy and accuracy metrics; the [B, T, V] logits are never materialized | 0.98 s GPU per run |
| `soup_addnorm_fwd_kernel`, `soup_addnorm_bwd_kernel` (K3) | `solution/kernels/addnorm.py` (wiring: `solution/layers.py`, `solution/fused_block.py`) | Every `Qwen2RMSNorm` forward and backward, and the first residual add of each decoder layer fused with `post_attention_layernorm` | 0.66 s GPU per run (fwd 0.37 + bwd 0.29) |
| All four together | | | 10.6 % at v6.1 (`ebeec3a`, measured); threshold 5 % |

v6.1 adds no new Triton kernel. K3's two kernels got new runtime options for B: an fp16 pointer for the attention output, a flag that skips the store of the fp32 sum, and a flag that rebuilds it in backward. `nf4.py` is unchanged from v5f (`git diff --stat 47a279c 6fff364 -- solution/kernels/` lists only `addnorm.py` and `head_ce.py`).

Not counted as ours: the starter kit's example SwiGLU kernels, called by K4 (`solution/mlp.py`), by S4's backward and, in place, by C.

Non-kernel changes (no Triton kernel in any of them):
- U, `solution/unpad.py`: RMSNorms, residual adds and the MLP run on the real tokens only; attention keeps [B, T, C], so the LoRA dropout masks and RNG state match the base.
- K4, `solution/mlp.py`: fp16 MLP path; autograd keeps gate and up in fp16 instead of three fp32 tensors per layer.
- S4, `solution/mlp_recompute.py`, flag `USE_MLP_RECOMPUTE`: the MLP saves only its fp16 input and recomputes gate and up in backward, bit-identical to K4. Our own autograd recompute, not Soup's `gradient_checkpointing`, which stays off. Saves 1,180 MiB, costs about 5.4 s (v4 against v5f, different sessions).
- B, `solution/fused_block.py`, flag `USE_FUSED_BLOCK`: add + norm + S4 MLP + add as one autograd node saving (r, fp16 a, rstd). Estimated saving 213 MiB [estimate].
- A1, `solution/sdpa_recompute.py`, flag `USE_SDPA_RECOMPUTE`: attention saves fp16 q and un-repeated k and v and recomputes torch's SDPA in backward. Estimated saving 507 MiB [estimate].
- A2, `solution/lora_replay.py`, flag `USE_LORA_DROPOUT_REPLAY`: LoRA dropout saves the generator state, not the mask. Estimated saving 162 MiB [estimate].
- D, `solution/saved_compact.py`, flag `USE_LORA_ROW_COMPACT`: LoRA-A inputs keep the real rows only (public `saved_tensors_hooks`, nothing offloaded). Estimated saving 115 MiB [estimate].
- C, `solution/mlp_recompute.py`, flag `INPLACE_SWIGLU_BACKWARD`: the example backward kernel writes over its own inputs. Estimated saving 16 MiB [estimate].
- Switched off, not used in any measured run: `STORE_HEAD_FP16` and `USE_SDPA_KV_COMPACT`.

Ideas taken from other work (we wrote the code ourselves; none of the banned libraries is imported):
- bitsandbytes: NF4 format, double quantization layout, rounding order, tile idea of `kDequantizeBlockwise` (K1). https://github.com/bitsandbytes-foundation/bitsandbytes
- Soup issue #1693 and the `ChunkedStep` protocol in `soup_cli/utils/stream_loss_chunk.py` of soup-cli 0.77.0 (K2). Soup repository: https://github.com/MakazhanAlpamys/Soup. Issue: #1693 in that repository.
- Liger-Kernel: fused linear + cross entropy, fused add + RMSNorm (K2, K3, B), idea only. https://github.com/linkedin/Liger-Kernel
- Unsloth: fused add + RMSNorm (K3, B), idea only. https://github.com/unslothai/unsloth
- Apple Cut Cross-Entropy: cross entropy without full logits (K2), idea only. https://github.com/apple/ml-cross-entropy
- Online softmax, Milakov and Gimelshein 2018 (K2). https://arxiv.org/abs/1805.02867
- ByteDance Effective Transformer and the unpad path of FlashAttention's varlen BERT: removing padding (U). https://github.com/bytedance/effective_transformer
- Chen et al. 2016, "Training Deep Nets with Sublinear Memory Cost": activation recomputation (S4, B, A1). https://arxiv.org/abs/1604.06174
- FlashAttention, Dao et al. 2022: recompute attention intermediates in backward (A1), idea only. https://arxiv.org/abs/2205.14135
- Korthikanti et al. 2022, "Reducing Activation Recomputation in Large Transformer Models": selective recomputation (A1). https://arxiv.org/abs/2205.05198
- `torch.utils.checkpoint` (`preserve_rng_state`) and the Megatron-LM CUDA RNG tracker: restoring the generator state (A2), idea only.
- Starter kit example SwiGLU: called, not copied (K4, S4, C).

AI assistants: ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow. Every Triton kernel was run with `pytest solution -q` and measured on a Kaggle T4. K1 v3s went through a Kaggle microbenchmark before it entered `solution/`. Drafts of several components were written by both assistants in parallel and compared; the chosen version was reviewed against the library sources and measured on Kaggle.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
Jury-style run of v6.1 (`--repeats 2`: base, solution, solution, base; notebook version 3, `COMMIT = ebeec3a`). The submitted `6fff364` differs from `ebeec3a` in one test tolerance (the LoRA-A gradient bound in `test_v6_merged`); the training code is the same.

| | Base Soup | Ours (v6.1) | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 90.2 (91.62 and 88.83) | 74.5 (74.85 and 74.15) | s = 1.211: wall time 17.4 % shorter |
| Peak GPU memory, MiB | 14,345 | 2,827 | m = 5.074: 80 % less |
| Eval loss (public rows) | 2.04966 | 2.04968 | +0.001 % |
| Share of own kernels (check 4) | n/a | 10.6 % | threshold 5 % |

Earlier versions and one later one, each against the base of its own session (base times differ by session, compare s and m):

| Version | Base, s | Ours, s | Peak, MiB | s | m |
|---|---|---|---|---|---|
| Base Soup 0.77.0 | n/a | n/a | 14,345 | 1.000 | 1.000 |
| K1 v1 + K4 (`a2c85c6`) | 84.96 | 82.55 | 10,383 | 1.029 | 1.382 |
| v1 = + K3 + K2 (`c2b1ff1`) | 78.22 | 72.54 | 5,845 | 1.078 | 2.454 |
| v2 = K1 v3s + U (`30e62a7`) | 77.06 | 61.22 | 5,025 | 1.259 | 2.855 |
| v4, S4 off (`c85343f`, `--repeats 2`) | 79.8 | 61.5 | 5,025 | 1.298 | 2.855 |
| v5f, S4 on (`47a279c`, `--repeats 2`) | 79.8 | 66.9 | 3,845 | 1.192 | 3.731 |
| v6 (`df2557b`) | 84.8 | out of memory at step 22 of 64 | 14,903 | n/a | n/a |
| **v6.1 (`6fff364`, measured at `ebeec3a`, `--repeats 2`)** | 90.2 | 74.5 | 2,827 | 1.211 | 5.074 |
| v7 (`cb7d630`, `--repeats 2`, not submitted) | 84.1 | 65.1 | 3,233 | 1.292 | 4.437 |

What went wrong in v6: both solution runs died at step 22 of 64 with "GPU ran out of memory". Memory grew by about 0.5 GB per step. A1's forward builds a throw-away SDPA graph that never gets a backward, and D's pack hook returned saved tensors as they were, so SDPA's own outputs were stored together with the node that owns them: a reference cycle that only that backward breaks. v6.1 runs A1's inner calls under private hooks, makes D's hooks return `t.detach()`, and adds `solution/tests/test_v6_no_leak.py` (flat memory over 8 training steps). The fix was confirmed on Kaggle: v6.1 ran all 64 steps at a flat 2,827 MiB peak.

Tried after v6.1: v7 (`cb7d630`, branch `v7-candidate`) adds causal SDPA in A1 (I1), one dequantization in S4 backward (I2), a K1 host fast path with a permanent fp16 cache of the k and v weights (I3), and S4 off on 7 of 28 layers (I4). It measured s = 1.292 and m = 4.437 (3,233 MiB), and all 10 checks passed. We kept v6.1 for the lower peak.

- [x] All 10 checks of `harness/check_rules.py` passed at v6.1 (`ebeec3a`): check 4 share 10.6 %, check 6 0.00 to 0.01 %, check 7 eval loss +0.001 %, check 8 time x0.826 and memory x0.197 of the base, check 9 clean, check 10 79 passed.
- [x] README.md for our solution (what we did, how to run it, results, limits).

## Presentation covers
Which operations we sped up and why (NF4 dequantization, fp32 elementwise ops and norms, padding, the full-vocabulary logits, the fp32 MLP activations, and the per-layer attention and LoRA activations that dominate the rest of the peak), how K1 to K4, U, S4, B, A1, A2, D and C work, the v6 memory leak and its fix, results against the base (speed, memory, quality) and the limits.

## Limitations
- Measured with `--repeats 2` like the jury, but our base ran 77 to 92 s across sessions, so s is meaningful only against the base of the same session. v5f and v6.1 were not run in the same session.
- The run is from `ebeec3a`; the submitted `6fff364` differs in one test tolerance. We have not measured `6fff364` as such.
- The savings of B, A1, A2, D and C are estimates; only their sum is measured (the peak).
- `pytest solution -q` at `ebeec3a`: 285 passed, 1 failed (LoRA-A gradient tolerance in `test_v6_merged`, a wrong test bound; fixed in `6fff364` to the derived bound of 2e-2). Final count: [tests: TODO final count]. The training run passed all 10 checks.
- A1 relies on torch's SDPA backward, which may not be deterministic on a T4. K3's fp16-pointer variant changes the normalized output by a few fp32 ulp against v5f; the 2e-6 test tolerance is an estimate. Eval loss is within +0.001 % of the base.
- In the v6.1 profile about 180 ms of each 0.72 s step was GPU idle, mostly Soup and Trainer host work. `apply()` takes about 7.5 s and the first step, with Triton compilation, 4.9 s; the Triton cache is empty in every run.
- K4, S4 and C rely on the example kernels for SiLU(gate) * up and its backward; they are not counted as ours.
- Built and measured for Soup's T4 fp16 path and Qwen2 only. Guards check classes, dtypes and shapes, mostly not the GPU model; where Soup picks bf16, K1, K4, S4 and B fall back to the original code. Other GPUs are not tested.
- Next steps: record the final test count at `6fff364`; split the per-kernel share of check 4.
