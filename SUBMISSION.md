# Submission: Team Delta

## Links
- Repository, final commit on main: https://github.com/BAITC-Hacks/hack-8cddb4ae-delta/commit/[TODO final commit]
  (the submitted code is v5f, commit `47a279c`, branch `v5f-candidate`, which becomes `main`: the v4 code and docs (`4f83a07`) with `USE_MLP_RECOMPUTE = True`; the final commit on main adds our README and this file. The measured reference for the same kernels is v2, `30e62a7`, branch `v2-candidate`; the S4-off alternative is v4)
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
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `soup_nf4_dequant_kernel` (K1) | `solution/kernels/nf4.py` | bitsandbytes NF4 dequantization on its dequant + `F.linear` path (forward and backward re-dequant), bit-exact | [?: per-kernel share; measurements only have the combined 11.1 % at `30e62a7` and at `9fe3836`, and 10.9 % at v5f (`47a279c`), read the per-kernel breakdown of check 4 from `runs/checks.json`] |
| `soup_head_ce_kernel` (K2) | `solution/kernels/head_ce.py` (wiring: `solution/head_loss.py`) | `lm_head` + `ForCausalLMLoss` inside `Qwen2ForCausalLM.forward`, and TRL's logits-based entropy and token-accuracy metrics in `SFTTrainer.compute_loss`; the [B, T, V] logits are never materialized, the step is handed to Soup as a `ChunkedStep` | [?: per-kernel share; measurements only have the combined share of all four kernels] |
| `soup_addnorm_fwd_kernel`, `soup_addnorm_bwd_kernel` (K3) | `solution/kernels/addnorm.py`, wired in by `solution/layers.py` | Every `Qwen2RMSNorm` (`input_layernorm` and the final `model.norm`) forward and backward, and the first residual add of each decoder layer fused with `post_attention_layernorm`; the second residual add stays a torch add | [?: per-kernel share; measurements only have the combined share of all four kernels] |
| All four together | | | 10.9 % at v5f (`47a279c`, submitted); 11.1 % at v4 (`9fe3836`) and at v2 (`30e62a7`), 10.7 % in the v4 final run (`c85343f`); 11.3 % at v5 (`a0d23de`) (threshold 5 %) |

The four kernel files are identical in v2, v4, v5 and v5f (`git diff --stat 30e62a7 9fe3836` lists only `unpad.py`, `mlp_recompute.py` and test files; v5f differs from v4 only by the S4 flag).

Not counted as ours: the starter kit's example SwiGLU kernel, which `solution/mlp.py` (K4) calls inside an fp16 Qwen2 MLP. K4 saves memory by keeping gate and up in fp16 instead of fp32. The matmul results are the same, and the SiLU(gate) * up result is rounded to fp16 at the same point as in the base. The tests in `solution/tests/test_mlp.py` match the base `Qwen2MLP` within rtol = atol = 2e-3 but do not check bit identity.

Non-kernel changes (no Triton kernel in either):
- U, `solution/unpad.py`: the RMSNorms, the residual adds and the MLP run on the real tokens only; attention keeps [B, T, C], so the LoRA dropout masks and the RNG state match the base. On in v4 and v5f.
- S4, `solution/mlp_recompute.py`: controlled by one flag, `USE_MLP_RECOMPUTE` in `solution/__init__.py`, True in the submitted v5f. When on, the MLP saves only its fp16 input and recomputes gate and up in backward; by design the outputs and input gradients are bit-identical to K4 (the tests check `torch.equal`; at v4, `pytest solution -q` on the T4 reports no failure in that file: 195 passed, 7 failed, all 7 in `test_unpad.py`; at v5f, 203 passed). This is our own autograd recompute inside our MLP path, not Soup's `gradient_checkpointing` mode, which stays off. Measured with the flag on in the jury-style run as v5f (`47a279c`): 66.9 s and 3,845 MiB (s = 1.192, m = 3.731), all 10 checks passed. We submit v5f because it cuts peak memory further (3,845 MiB) while staying 1.19x faster than the base; S4 can be switched off with one flag (`USE_MLP_RECOMPUTE`) for the 1.30x / 5,025 MiB trade-off (v4).

Ideas taken from other kernels, with links (we wrote the code ourselves):
- bitsandbytes: NF4 format, double quantization layout and rounding order, and the tile idea of K1 v3s (one absmax per thread, one packed byte gives two outputs) after its `kDequantizeBlockwise` kernel; idea only, our own code (K1). https://github.com/bitsandbytes-foundation/bitsandbytes
- Soup issue #1693: the `ChunkedStep` protocol for a step without logits, defined in `soup_cli/utils/stream_loss_chunk.py` of soup-cli 0.77.0 (K2). Soup repository (from the starter kit README): https://github.com/MakazhanAlpamys/Soup. Issue link: [?: probably https://github.com/MakazhanAlpamys/Soup/issues/1693; page not opened]
- Liger-Kernel: fused linear + cross entropy and fused add + RMSNorm, idea only, no code and no import (K2, K3). https://github.com/linkedin/Liger-Kernel
- Unsloth: fused add + RMSNorm, idea only, no code and no import (K3). Link: [?: probably https://github.com/unslothai/unsloth; not in our files]
- Apple Cut Cross-Entropy: cross entropy without full logits, idea only, no code and no import (K2). https://github.com/apple/ml-cross-entropy
- Online softmax (Milakov & Gimelshein, 2018, "Online normalizer calculation for softmax"): running max and sum of exponentials over vocabulary tiles, idea only (K2). Link: [?: probably https://arxiv.org/abs/1805.02867; not in our files]
- ByteDance Effective Transformer and the "unpad" path of FlashAttention's varlen BERT: removing padding for token-wise ops, idea only, no code (U). Link: [?: probably https://github.com/bytedance/effective_transformer; not in our files]
- Chen et al. 2016, "Training Deep Nets with Sublinear Memory Cost": activation recomputation (S4). https://arxiv.org/abs/1604.06174
- Starter kit example SwiGLU: used as is inside K4 (`solution/swiglu.py` is the example file unchanged; K4 calls its function directly).

AI assistants: ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow. The four Triton kernels in v5f are the ones run with `pytest solution -q` and measured on a Kaggle T4 in v2, in v4 and again in v5f. K1 v3s went through a Kaggle microbenchmark (`672d725`) before it entered `solution/`.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
Final jury-style run of the submitted v5f (`47a279c`, `--repeats 2`: base, solution, solution, base): base 79.8 s (76.72 and 82.78) / 14,345 MiB, ours 66.9 s (65.38 and 68.39) / 3,845 MiB, s = 1.192, m = 3.731, all 10 checks pass, `pytest solution -q` 203 passed. v5f is the v4 code with S4 on. v4 reproduces v2 (`30e62a7`, same four Triton kernels and same K4): 61.22 to 60.76 s, the same peak. The v2 row stays in the ablation as history.

| | Base Soup | Ours (v5f) | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 79.8 | 66.9 | s = 1.192: 1.19x speedup, wall time 16.2 % shorter |
| Peak GPU memory, MiB | 14,345 | 3,845 | base/ours m = 3.731: 73 % less peak memory |
| Eval loss (public rows) | 2.04969 | 2.04962 | -0.003 % vs base |

The v4 alternative (S4 off, `--repeats 2` at `c85343f`): base 79.8 s, ours 61.5 s, s = 1.298, 5,025 MiB (m = 2.855), eval loss -0.001 %, all 10 checks pass, kernel share 10.7 %.

Ablation (against the base of the same session; one repeat each except the last two rows, which are `--repeats 2`):

| Version | Time, s | Peak, MiB | s | m |
|---|---|---|---|---|
| Example SwiGLU only | 83.05 (base 77.83) | 12,385 | 0.937 | 1.158 |
| K1 + K4 | 82.55 (base 84.96) | 10,383 | 1.029 | 1.382 |
| v1 = K1 + K4 + K3 + K2 | 72.54 (base 78.22) | 5,845 | 1.078 | 2.454 |
| v2 = + K1 v3s + unpadding (`30e62a7`) | 61.22 (base 77.06) | 5,025 | 1.259 | 2.855 |
| v4 = v2 + final U + test fixes, S4 off (`9fe3836`) | 60.76 (base 77.77) | 5,025 | 1.280 | 2.855 |
| v5 = v4 + S4 on (`a0d23de`, ablation) | 65.7 (base 77.6) | 3,845 | 1.182 | 3.731 |
| v4 final run, S4 off (`c85343f`, speed-oriented alternative) | 61.5 (base 79.8) | 5,025 | 1.298 | 2.855 |
| v5f = v4 + S4 on (`47a279c`, submitted) | 66.9 (base 79.8) | 3,845 | 1.192 | 3.731 |

- [x] All 10 checks of `harness/check_rules.py` passed at v5f (`47a279c`: check 4 share 10.9 %, check 6 0.00 to 0.01 %, check 7 eval loss -0.003 %, check 8 time x0.839 and memory x0.268 of the base, check 9 clean, check 10 79 passed). They also passed at v4 (`9fe3836`: check 4 share 11.1 %, check 6 0.01 % first step and 0.01 % mean, check 8 time x0.781 and memory x0.350; final run `c85343f`: check 4 10.7 %, check 8 time x0.771 and memory x0.350), at `30e62a7` (check 4 11.1 %, check 8 time x0.794 and memory x0.350) and at `a0d23de` (check 4 11.3 %, check 8 time x0.846 and memory x0.268).
- [x] README.md for our solution (what we did, how to run it, results, limits).

## Presentation covers
Which operations we sped up and why (NF4 dequantization, fp32 elementwise ops and norms, the full-vocabulary logits, and the fp32 MLP activations that dominate the peak), how K1 to K4, U and S4 (on in v5f, one flag switches it off) work, results against the base (speed, memory, quality) and the limits and next steps.

## Limitations
- The final run of the submitted v5f used `--repeats 2` like the jury: base 79.8 s, ours 66.9 s, s = 1.192, m = 3.731, eval loss -0.003 %, all 10 checks pass, kernel share 10.9 %. The v4 final run (`c85343f`, S4 off) also used `--repeats 2`: base 79.8 s, ours 61.5 s, s = 1.298, m = 2.855, eval loss -0.001 %, kernel share 10.7 %. Our base runs ranged from 77.06 to 84.96 s across sessions, so s is only meaningful against the base of the same session.
- Compile cost is part of the wall time because the Triton cache is empty in every run. We have not split compile time from kernel time yet [TODO].
- K4 relies on the example kernel for SiLU(gate) * up. It is not counted as our kernel.
- Our own test suite at `30e62a7`: 136 passed, 27 failed, 23 of them an import error in `test_unpad.py` caused by the Kaggle image's old torchao and PEFT 0.20, the rest 4 older test problems. They do not affect the training run (checks 5 to 10 pass). The 4 older ones were fixed in `2bef7a3`, the ImportError behind the 23 in `9fe3836` (v4). At `a0d23de` the output was not read to the end and at least one `test_unpad.py` gradient-tolerance failure was still visible (0.0038 against 0.002); `unpad.py` and its test are the same in v4. At v4 (`9fe3836`), `pytest solution -q` gave 195 passed and 7 failed, all 7 in `test_unpad.py`: six on the `layers.0.self_attn.q_proj.lora_A` gradient tolerance (0.0038 against 0.002) and `test_pack_unpack_roundtrip_and_gradients`. The 4 older failures and the torchao ImportError are fixed. The 7 were test-side problems and were fixed in `c85343f`: the pack/unpack test took its RNG snapshot before drawing its random inputs, and the LoRA-gradient comparisons ran without a loss scale, so fp16 subnormals dominated the relative error; the tests now scale the loss by 2**16 as GradScaler does (tolerance derived in a comment) and a negative test checks that misrouted row gradients are caught. At `c85343f`, `pytest solution -q` gave 203 passed, 0 failed; at the submitted v5f (`47a279c`) it gave 203 passed.
- Built and measured for Soup's T4 fp16 path only; Qwen2 only. K2 and U check the exact Qwen2 classes. S4 (on in v5f, off in v4) is installed only on `Qwen2MLP.forward` and checks K4's conditions (exact bitsandbytes `Linear4bit` layers with fp16 compute dtype), not the `Qwen2MLP` type. K1, K4 and S4 fall back unless the 4-bit layers compute in fp16. K2 falls back for other models and for dtypes other than fp16 autocast, but does not check the GPU model. K3 needs no particular GPU model: it runs on any CUDA GPU with an fp32 residual stream of width 1536 (Qwen2.5-1.5B) and frozen fp32 norm weights, and other dtypes, widths, CPU tensors and other models fall back. U checks neither the GPU model nor the dtype, so on another CUDA GPU with SDPA attention it still takes its fast path. Other GPUs are not tested.
- The submitted v5f was measured as a whole. Its code is v4's with S4 on, and v4's final U differs from the U in v2. v4 reproduces v2 (61.22 to 60.76 s, the same peak).
- Next steps: split compile time from kernel time.
