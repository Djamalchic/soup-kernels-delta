# Team Delta: Soup Kernel Challenge (Decentrathon 6.0)

Submitted version: v8b (code commit `599c58a`, branch `v8b-candidate`, which becomes `main`).

- Public Kaggle notebook: [TODO final link]
- Final commit on main: [TODO final commit] (the v8b code plus this README and SUBMISSION.md)

**In one paragraph.** On a Kaggle T4, in the final jury-style run (notebook version 11, `--repeats 2`), the fixed `soup train` task takes 69.3 s with our solution against 88.4 s for the base: s = 1.275, a wall time 21.6 % shorter. The peak is 2,893 MiB against 14,345 MiB (m = 4.958, 79.8 % less). Eval loss is 2.04964 against 2.04963 (+0.000 %). All 10 checks pass and `pytest solution -q` gives 397 passed. In a separate same-session run (notebook version 10) the task took 63.3 s with v8b, 67.7 s with our previous version v6.1 and 79.8 s with the base. Numbers without a measurement are marked `[estimate]`.

The same-session run used commit `8eebfbf`; `599c58a` is the clean branch. `git diff 8eebfbf 599c58a -- solution/` shows only the removed, switched-off per-row SDPA code (`USE_SDPA_PER_ROW` was `False` in `8eebfbf`) and its tests. The training path is the same.

## 1. What we did

### What is slow and what is big

The task is `soup train` for Qwen2.5-1.5B (NF4, LoRA r=16 on q and v, batch 4, up to 512 tokens, 64 steps) on one T4. The starter kit's time profile: matmul 36 %, elementwise ops and casts 44 %, cross entropy 7 %, attention 5 %, NF4 dequantization 5 %.

The base peak of 14,345 MiB has four big parts, removed in this order:

1. **fp32 MLP activations.** The residual stream is fp32, so autograd keeps gate, SiLU(gate) and up as three fp32 [tokens, 8960] tensors per layer. Fix: K4. Measured drop: 3,962 MiB.
2. **Logits over the 151,936-token vocabulary.** Fix: K2, added together with K3. Measured drop: 4,538 MiB, not splittable between them.
3. **Padding.** Fix: U, added together with K1 v3s. Measured: 11.3 s and 820 MiB less.
4. **Per-layer activations kept until backward** (residuals, SDPA tensors, LoRA dropout masks and inputs). Fix: S4, B, A1, A2, D, C. Together they took the peak from 3,845 MiB (v5f) to 2,827 MiB (v6.1).

We use no Soup mode: `gradient_checkpointing`, `activation_offloading`, `fast_lora`, `use_cut_ce` and `packing` stay off. Recomputation is our own autograd code. Every component is guarded; if a guard fails, the original code runs.

### Components (files under `solution/`)

Our Triton kernels: K1, K2, K3 and the SwiGLU kernel of C2. The rest are PyTorch-level changes.

**K1, NF4 dequantization** (`kernels/nf4.py`). Replaces bitsandbytes' dequantize + `F.linear` path through a guarded patch of `bitsandbytes.matmul_4bit`. A thread reads 4 packed bytes in one 32-bit load, each byte gives two fp16 outputs, and 16 contiguous bytes are stored. The NF4 table is a tree of 15 selects, not a memory lookup. Output is bit-exact against `dequantize_4bit` (`torch.equal`).

**K2, LM head + cross entropy without logits** (`kernels/head_ce.py`, `head_loss.py`). The vocabulary is walked in 10 tiles of at most 16,384 rows. cuBLAS does the head GEMM per tile; our kernel folds each tile into per-row online softmax statistics and, in backward, turns the recomputed tile into the logits gradient in place. The full [B, T, V] logits never exist. The step goes to Soup through its own `ChunkedStep` class.

**K3, fused residual add + RMSNorm** (`kernels/addnorm.py`, `layers.py`). One program per row does the first residual add and `post_attention_layernorm` in one pass; the same kernel serves the other RMSNorms. The backward kernel returns the norm input gradient and adds the incoming residual gradient in the same pass.

**K4, fp16 MLP path** (`mlp.py`). Soup computes 4-bit layers in fp16 on a T4, so casting the MLP input to fp16 first gives the same matmul results and lets autograd keep gate and up in fp16. SiLU(gate) * up runs in our `soup_swiglu_kernel` (C2 below).

**U, unpadding** (`unpad.py`). RMSNorms, residual adds and the MLP run on the real tokens only ([N, C]). Attention keeps the base shape [B, T, C], so LoRA dropout masks and the RNG stream match the base.

**S4, MLP recompute** (`mlp_recompute.py`). The MLP saves only its fp16 input and recomputes gate and up in backward with the same 4-bit GEMMs; outputs and input gradients are bit-identical to K4. Saves 1,180 MiB, costs about 5.4 s (v4 against v5f, different sessions).

**B, fused add + norm + MLP block** (`fused_block.py`). One autograd node covers the first residual add, post-attention norm, S4 MLP and second residual add. It saves (r, fp16 a, rstd), 6 bytes per element instead of 10.

**A1, SDPA recompute** (`sdpa_recompute.py`). Registered as the `"sdpa"` entry of transformers' `AttentionInterface`. It saves fp16 q and the un-repeated k and v instead of q, 12-head k and v, output, logsumexp and the float mask bias, and runs torch's own SDPA again in backward.

**A2, LoRA dropout replay** (`lora_replay.py`). The q and v LoRA dropout saves the 16-byte CUDA generator state instead of its bool mask and replays it in backward.

**D, LoRA-A inputs for real rows only** (`saved_compact.py`). Inside a packed layer's attention, `saved_tensors_hooks` keep only the real rows of the fp16 LoRA-A inputs. Padding rows are exact zeros, so unpacking is exact.

**C, in-place SwiGLU backward** (`mlp_recompute.py`). The SwiGLU backward writes its two gradients over its own inputs. Same bits as the out-of-place call.

### Changed since v6.1

Each change sits behind a flag in `solution/__init__.py`; all are on.

- **I1, causal SDPA on the packed path** (`USE_SDPA_CAUSAL_PADDED`). A1 calls SDPA with `is_causal=True` and no bool mask. Padding is on the right, so real rows never attend to padding. This may move real rows by fp16 ulps.
- **I2, one dequantization in S4 backward** (`S4_DEQUANT_REUSE`). Gate and up weights are dequantized once per backward MLP call. Transient cost about 26 MiB [estimate].
- **I3, K1 host fast path and k/v weight cache** (`USE_K1_FAST_HOST`, `USE_DENSE_CACHE`). Cheaper validation in K1's launcher, and a permanent fp16 copy of the `k_proj` and `v_proj` weights (about 42 MiB).
- **S4 on all 28 layers** (`S4_OFF_FRACTION = 0.0`). v7 switched S4 off on 7 layers (I4). Speed did not change and the peak was 340 MiB higher, so v8b keeps S4 everywhere. The I4 path stays in the code.
- **C1, one Triton launcher** (`USE_UNIFIED_SIG`). Triton builds a small C launcher for each distinct list of argument kinds; v7 had six (our reading of Triton's source, checked by `tests/test_compile_probe.py` on 3.6). Now K1, K2, K3 and C2 take the same twelve runtime arguments, so Triton builds one. K1 to K3 are launched through `soup_*_usig` entries that inline the v7 kernel unchanged (`torch.equal` tests). Step 1, where Triton compiles, fell from 5.1 s to 3.6 s in a same-session comparison.
- **C2, our own SwiGLU kernel** (`USE_SOUP_SWIGLU`). `soup_swiglu_kernel` does forward, backward and in-place backward in one jit and replaces the two example kernels in K4 and S4. **It computes the same SwiGLU math as the starter kit's example, operation for operation, with the same bits** (`tests/test_soup_swiglu.py`). The jury may treat it as the example and not count it toward check 4. We do not count it: our share is above 5 % without it.
- **Test fixes** for the three v7 test failures (tolerances and a memory estimate, not training).

Off in the submitted version and in every measured run: `STORE_HEAD_FP16` (fp16 copy of the LM head; unclear whether the rules count it as changing the model) and `USE_SDPA_KV_COMPACT` (fallback for A1).

### Where the ideas come from

We wrote all kernel code ourselves. Libraries banned by the rules (Liger-Kernel, Unsloth, Cut Cross-Entropy) were read for ideas only and are not imported.

| Source | What we took | Used in |
|---|---|---|
| bitsandbytes, https://github.com/bitsandbytes-foundation/bitsandbytes | NF4 table, nested absmax layout, rounding order; tile idea of `kDequantizeBlockwise` | K1 |
| Soup issue #1693; `ChunkedStep` in `soup_cli/utils/stream_loss_chunk.py` (soup-cli 0.77.0) | Contract for a step without logits | K2 |
| Liger-Kernel, https://github.com/linkedin/Liger-Kernel; Apple Cut Cross-Entropy, https://github.com/apple/ml-cross-entropy | Chunked linear + cross entropy, logits recomputed in backward | K2 |
| Liger-Kernel, Unsloth; the starter kit's example SwiGLU | Fused add + RMSNorm; fused SwiGLU (the example's math, reproduced) | K3, B, C2 |
| Milakov and Gimelshein 2018, online softmax | Running max and sum over vocabulary tiles | K2 |
| ByteDance Effective Transformer; FlashAttention varlen unpad path | Removing padding for token-wise ops | U |
| Chen et al. 2016, arXiv:1604.06174; Korthikanti et al. 2022, arXiv:2205.05198; Dao et al. 2022 (FlashAttention) | Activation recomputation; recomputing only the attention core | S4, B, A1 |
| `torch.utils.checkpoint` (`preserve_rng_state`); Megatron-LM RNG tracker | Restoring generator state to redo a dropout mask | A2 |
| PyTorch `saved_tensors_hooks`, `scaled_dot_product_attention` (public APIs) | Custom storage of saved tensors; causal SDPA | D, A1, I1 |

ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow. Drafts of several components were written by both assistants in parallel and compared; the chosen version was reviewed against the library sources and measured on Kaggle.

## 2. How to run

Pinned by the starter notebook: `soup-cli[train]==0.77.0`, `transformers==5.16.1`, `peft==0.20.0`, `trl==0.29.1`, `bitsandbytes==0.50.2`, `accelerate==1.14.0`. Triton and torch come with the Kaggle image.

1. Open the public notebook: [TODO final link].
2. Set the accelerator to GPU T4 x2 and switch Internet on.
3. In the first cell, check that `REPO_URL` is our repository and `COMMIT` is the final commit: [TODO final commit]. A public repository needs no secret.
4. Run, then Restart & clear cell outputs. Add no GPU cells of your own: a leftover cell once held 3 GB of GPU memory and failed check 9.
5. Click Run All. It takes about 10 minutes. The notebook (the starter kit's `templates/kaggle_notebook.ipynb`) installs the packages, clones the repository and runs:

```bash
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py            # base, then solution
python harness/check_rules.py         # check 10 runs Soup's own tests with our patches
python harness/score.py runs/result.json
```

For a jury-style measurement use `python harness/run_task.py --repeats 2` (base, solution, solution, base). To test only our side: `python harness/run_task.py --only solution --out runs_try`.

## 3. Results and limitations

### Against the base (Kaggle T4, `--repeats 2`, notebook version 11, `COMMIT = 599c58a`)

| | Base Soup 0.77.0 | Ours (v8b) | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 88.4 (89.39 and 87.44) | 69.3 (67.49 and 71.22) | s = 1.275 |
| Peak GPU memory (nvidia-smi), MiB | 14,345 | 2,893 | m = 4.958 |
| Eval loss (public rows) | 2.04963 | 2.04964 | +0.000 % |
| Share of own kernels (check 4) | n/a | 11.5 % | threshold 5 % |

The wall time is 21.6 % shorter and the peak 79.8 % lower. The 11.5 % includes `soup_swiglu_kernel`, which the harness counts. Without it the K1 to K3 share was 9.8 % at `80c3606`, so the 5 % threshold holds either way.

All 10 checks: PASS. Check 3 lists `soup_addnorm_bwd_usig`, `soup_addnorm_fwd_usig`, `soup_head_ce_usig`, `soup_nf4_dequant_usig` and `soup_swiglu_kernel`. Check 4: 11.5 %. Check 6: 0.00 to 0.01 %. Check 8: time x0.784, memory x0.202. Check 9: clean. Check 10: 79 passed. `pytest solution -q` at `599c58a`: 397 passed, 0 failed. The previous full run was v7 with S4 on all layers (`80c3606`): 10 of 10 checks, s = 1.230, m = 4.958.

### Same session: v6.1 against v8b (notebook version 10)

One base run, two runs of each version. The base time changes from session to session (79.8 to 90.2 s in our `--repeats 2` and head-to-head runs), so compare s and m inside one session, not seconds between sessions.

| | Base | v6.1 (`0c89204`) | v8b (`8eebfbf`) |
|---|---|---|---|
| Time of `soup train`, s | 79.81 | 67.7 (68.28 and 67.17) | 63.3 (63.17 and 63.46) |
| Training loop, s | 55.00 | 44.6 (45.03 and 44.19) | 40.3 (40.09 and 40.47) |
| Step 1, with compilation, s | 1.78 | 5.8 (6.63 and 4.98) | 3.5 (3.55 and 3.51) |
| Peak GPU memory, MiB | 14,345 | 2,827 | 2,893 |
| s | 1.000 | 1.18 | 1.26 |
| m | 1.000 | 5.07 | 4.96 |

v8b is 4.4 s (6.5 %) shorter than v6.1 and uses 66 MiB more peak, as expected from I1 to I3 (+42 MiB cache, +26 MiB transient, -2 MiB) [estimate]. Step 1 is 2.3 s shorter, the other steps about 2 s. We did not run the changes one by one, so the gain is not split.

### Ablation (each row against the base of its own session)

| Version | Base, s | Ours, s | Peak, MiB | s | m | Eval loss | All checks |
|---|---|---|---|---|---|---|---|
| v2 = K1 v3s + K4 + K3 + K2 + U | 77.06 | 61.22 | 5,025 | 1.259 | 2.855 | +0.000 % | yes |
| v4 = v2, final U, S4 off | 79.8 | 61.5 | 5,025 | 1.298 | 2.855 | -0.001 % | yes |
| v5f = v4 + S4 on | 79.8 | 66.9 | 3,845 | 1.192 | 3.731 | -0.003 % | yes |
| v6 = v5f + B, A1, A2, C, D | 84.8 | out of memory | 14,903 | n/a | n/a | n/a | no (1, 3, 4) |
| v6.1 = v6 + leak fix | 90.2 | 74.5 | 2,827 | 1.211 | 5.074 | +0.001 % | yes |
| v7 = v6.1 + I1 to I4 (not submitted) | 84.1 | 65.1 | 3,233 | 1.292 | 4.437 | +0.001 % | yes |
| v7 with S4 on all layers (`80c3606`) | 80.6 | 65.6 | 2,893 | 1.230 | 4.958 | -0.002 % | yes |
| **v8b = v6.1 + I1 to I3 + C1 + C2** | 88.4 | 69.3 | 2,893 | 1.275 | 4.958 | +0.000 % | yes |

v4, v5f, v6.1, the two v7 rows and v8b are `--repeats 2` runs; v2 is a single repeat. v6.1 measured s = 1.211 and 1.210 in two sessions and v7 with S4 on all layers 1.230. The gap is inside the spread between sessions, which is why we ran the same-session comparison. Earlier steps (K1 v1 + K4, v1) are in SUBMISSION.md.

### What we tried and dropped

- **Per-row causal SDPA** (called B in our scouting notes, not the fused block B above). SDPA on each row's real prefix. Attention GPU time fell from 56 to 26 ms per step, but about 1,700 small CPU-side operations per step were added and wall time did not fall (62.93 s without it, 63.17 s with it, same session). Dropped; `599c58a` has none of its code.
- **S4 off on some layers (I4).** Same-session ladder: 71.95 s with S4 on all layers, 72.39 and 70.65 s with 7 layers off, 71.70 s with 2 off. No gain beyond noise, and the peak was 3,233 MiB with 7 layers off against 2,893 MiB. Dropped to keep the memory.
- **The v6 out-of-memory leak**, fixed in v6.1 (last limitation below).
- **fp16 copy of the LM head** (`STORE_HEAD_FP16`). About 445 MiB less static memory [estimate], never in a measured run. Kept off because the rules may not allow it.

### Limitations

- **Estimates and noise.** Savings of B, A1, A2, D, C and I1 to I3 are estimates; only sums are measured. The task text gives 1 to 4 % spread between repeats. In the same-session comparison the two runs differ by 1.1 s (v6.1) and 0.3 s (v8b) against a gap of 4.4 s, but that is two runs and one base run. In the final run our two repeats differ by 3.7 s (67.49 and 71.22 s) and the two base repeats by 1.95 s (89.39 and 87.44 s).
- **C1 and C2 help the compile step only.** After step 1 the steps were not faster in a single same-session run (35.7 s against 36.6 s for steps 2 to 64, inside the noise). Step 1 is still 3.5 s against 1.8 s for the base. In the version 11 run the solution's `triton_cache` holds exactly one `__triton_launcher*.so` and 5 `*.cubin`, so C1 does give one launcher.
- **Example kernels.** C2 computes the same math as the starter kit's example, so we do not rely on it. In the version 11 output check 3 lists `soup_addnorm_bwd_usig`, `soup_addnorm_fwd_usig`, `soup_head_ce_usig`, `soup_nf4_dequant_usig` and `soup_swiglu_kernel`. It also notes the bodies `soup_*_kernel` as "not launched in every solution run", because they run as inlined device functions inside the `*_usig` entries; the check still passes.
- **Numerics.** K1, S4, B, A2, D, C, C1 and C2 keep base values by design (`torch.equal` tests). A1 relies on torch's SDPA backward, which may not be deterministic on a T4. I1 may change real rows by fp16 ulps. K3's fp16-pointer variant changes the normalized output by a few fp32 ulp against v5f. Eval loss was within +0.001 % of the base for v6.1 and v7, -0.002 % for `80c3606` and +0.000 % for v8b (2.04964 against 2.04963).
- **Part of the time is not ours.** Of the 63.3 s of the same-session run, 40.3 s are the training loop; the other 23.0 s (base: 24.8 s) lie outside it. In scouting profiles the GPU was idle 100 to 180 ms of each 0.56 to 0.72 s step, mostly Soup and Trainer host work. `apply()` itself costs 0.02 s; the 7.4 s the harness reports for it are imports Soup does anyway.
- **I3 cache** assumes nobody writes the NF4 bytes in place. Nothing in Soup, peft or bitsandbytes does.
- **Scope.** Fp16 on the T4, Qwen2 only. Guards check classes, dtypes and shapes, mostly not the GPU model. Where Soup picks bf16, K1, K4, S4 and B fall back to the original code. Other GPUs and models are not tested.
- **The v6 crash.** The first build of A1, A2, B, C and D ran out of memory at step 22 of 64, leaking about 0.5 GiB per step. A1's throw-away SDPA graph never got a backward, and D's pack hook stored SDPA's own outputs together with the node that owns them, a reference cycle that only that backward breaks. v6.1 runs A1's inner calls under private hooks and D's hooks return `t.detach()`; `test_v6_no_leak.py` checks flat memory over 8 steps. The mechanism is our reading of PyTorch autograd; v6.1 ran all 64 steps at a flat 2,827 MiB peak.
