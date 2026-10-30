# Team Delta: Soup Kernel Challenge (Decentrathon 6.0)

Submitted version: v6.1 (code commit `6fff364`, branch `v61-candidate`, which becomes `main`).

- Public Kaggle notebook: [TODO final link]
- Final commit on main: [TODO final commit] (the v6.1 code plus this README and SUBMISSION.md)

**In one paragraph.** On a Kaggle T4, in a jury-style run (`--repeats 2`, notebook version 3), our solution trains the fixed `soup train` task in 74.5 s instead of 90.2 s (s = 1.211, wall time 17.4 % shorter) with a peak of 2,827 MiB instead of 14,345 MiB (m = 5.074, 80 % less). Eval loss is 2.04968 against 2.04966 (+0.001 %). All 10 checks of `harness/check_rules.py` pass. Numbers without a measurement are marked `[estimate]`.

The measured run used commit `ebeec3a`. The submitted `6fff364` differs from it in one test tolerance (the LoRA-A gradient bound in `test_v6_merged`). The training code is the same.

## 1. What we did

### What is slow and what is big

The task is `soup train` for Qwen2.5-1.5B (NF4, LoRA r=16 on q and v, batch 4, up to 512 tokens, 64 steps) on one T4. The starter kit's time profile: matmul 36 %, elementwise ops and casts 44 %, cross entropy 7 %, attention 5 %, NF4 dequantization 5 %.

The base peak of 14,345 MiB has four big parts, removed in this order:

1. **fp32 MLP activations.** The residual stream is fp32, so autograd keeps gate, SiLU(gate) and up as three fp32 [tokens, 8960] tensors per layer. Fix: K4. Measured drop: 3,962 MiB.
2. **Logits over the 151,936-token vocabulary.** Fix: K2, added together with K3. Measured drop: 4,538 MiB, not splittable between them.
3. **Padding.** Fix: U, added together with K1 v3s. Measured: 11.3 s and 820 MiB less.
4. **Per-layer activations kept until backward** (residuals, SDPA tensors, LoRA dropout masks and inputs). Fix: S4, B, A1, A2, D, C. Together they took the peak from 3,845 MiB (v5f) to 2,827 MiB.

We use no Soup mode: `gradient_checkpointing`, `activation_offloading`, `fast_lora`, `use_cut_ce` and `packing` stay off. Recomputation is our own autograd code. Every component is guarded; if a guard fails, the original code runs.

### Components (files under `solution/`)

Our Triton kernels, as named in check 3: `soup_nf4_dequant_kernel` (K1), `soup_head_ce_kernel` (K2), `soup_addnorm_fwd_kernel` and `soup_addnorm_bwd_kernel` (K3). The rest are PyTorch-level changes.

**K1, NF4 dequantization** (`kernels/nf4.py`). Replaces bitsandbytes' dequantize + `F.linear` path through a guarded patch of `bitsandbytes.matmul_4bit`. A thread reads 4 packed bytes in one 32-bit load, each byte gives two fp16 outputs, and 16 contiguous bytes are stored. The NF4 table is a tree of 15 selects, not a memory lookup. Output is bit-exact against `dequantize_4bit` (`torch.equal` on bit patterns). Microbenchmark on the T4, weight [8960, 1536]: 0.166 ms per call against 0.300 ms for the bitsandbytes full path.

**K2, LM head + cross entropy without logits** (`kernels/head_ce.py`, `head_loss.py`). The vocabulary is walked in 10 tiles of at most 16,384 rows. cuBLAS does the head GEMM per tile; our kernel folds each tile into per-row online softmax statistics and, in backward, turns the recomputed tile into the logits gradient in place. The full [B, T, V] logits never exist. The step goes to Soup through its own `ChunkedStep` class.

**K3, fused residual add + RMSNorm** (`kernels/addnorm.py`, `layers.py`). One program per row does the first residual add and `post_attention_layernorm` in one pass; the same kernel serves the other RMSNorms. The backward kernel returns the norm input gradient and adds the incoming residual gradient in the same pass. For B, the kernels also read the attention output as fp16 and can rebuild the residual sum in backward.

**K4, fp16 MLP path** (`mlp.py`). Soup computes 4-bit layers in fp16 on a T4, so casting the MLP input to fp16 first gives the same matmul results and lets autograd keep gate and up in fp16. SiLU(gate) * up uses the organizers' example SwiGLU kernel, which is not counted as ours.

**U, unpadding** (`unpad.py`). RMSNorms, residual adds and the MLP run on the real tokens only ([N, C]). Attention still gets the base shape [B, T, C], so LoRA dropout masks and the RNG stream match the base.

**S4, MLP recompute** (`mlp_recompute.py`). The MLP saves only its fp16 input and recomputes gate and up in backward with the same 4-bit GEMMs; outputs and input gradients are bit-identical to K4 (`torch.equal`). Saves 1,180 MiB, costs about 5.4 s (v4 against v5f, different sessions).

**B, fused add + norm + MLP block** (`fused_block.py`). One autograd node covers the first residual add, post-attention norm, S4 MLP and second residual add. It saves (r, fp16 a, rstd), 6 bytes per element instead of 10, and rebuilds the rest in backward with the same kernels. Estimated saving 213 MiB [estimate].

**A1, SDPA recompute** (`sdpa_recompute.py`). Registered as the `"sdpa"` entry of transformers' `AttentionInterface`. It saves fp16 q and the un-repeated k and v instead of q, 12-head k and v, output, logsumexp and the float mask bias, and runs torch's own SDPA again in backward. Estimated saving 507 MiB [estimate].

**A2, LoRA dropout replay** (`lora_replay.py`). The q and v LoRA dropout saves the 16-byte CUDA generator state instead of its bool mask and replays the same dropout in backward. Estimated saving 162 MiB [estimate].

**D, LoRA-A inputs for real rows only** (`saved_compact.py`). Inside a packed layer's attention, `saved_tensors_hooks` keep only the real rows of the fp16 LoRA-A inputs. Padding rows are exact zeros, so unpacking is exact. Estimated saving 115 MiB [estimate].

**C, in-place SwiGLU backward** (`mlp_recompute.py`). The example backward kernel writes its outputs over its own inputs. Same kernel, same bits. Estimated saving 16 MiB [estimate].

Before the run we estimated the sum of B, A1, A2, D and C at about 1,013 MiB, so a peak of 2,830 to 2,960 MiB [estimate]. The measured peak is 2,827 MiB, 1,018 MiB below v5f (different sessions).

Off in the submitted version and in every measured run: `STORE_HEAD_FP16` (fp16 copy of the LM head; unclear whether the rules count it as changing the model) and `USE_SDPA_KV_COMPACT` (fallback for A1).

### Where the ideas come from

We wrote all kernel code ourselves. Libraries banned by the rules (Liger-Kernel, Unsloth, Cut Cross-Entropy) were read for ideas only and are not imported.

| Source | What we took | Used in |
|---|---|---|
| bitsandbytes, https://github.com/bitsandbytes-foundation/bitsandbytes | NF4 table, nested absmax layout, rounding order; tile idea of `kDequantizeBlockwise` | K1 |
| Soup issue #1693; `ChunkedStep` in `soup_cli/utils/stream_loss_chunk.py` (soup-cli 0.77.0) | Contract for a step without logits | K2 |
| Liger-Kernel, https://github.com/linkedin/Liger-Kernel; Apple Cut Cross-Entropy, https://github.com/apple/ml-cross-entropy | Chunked linear + cross entropy, logits recomputed in backward | K2 |
| Liger-Kernel, Unsloth | Fused add + RMSNorm | K3, B |
| Milakov and Gimelshein 2018, online softmax | Running max and sum over vocabulary tiles | K2 |
| ByteDance Effective Transformer; FlashAttention varlen unpad path | Removing padding for token-wise ops | U |
| Chen et al. 2016, arXiv:1604.06174; Korthikanti et al. 2022, arXiv:2205.05198; Dao et al. 2022 (FlashAttention) | Activation recomputation; recomputing only the attention core | S4, B, A1 |
| `torch.utils.checkpoint` (`preserve_rng_state`); Megatron-LM RNG tracker | Restoring generator state to redo a dropout mask | A2 |
| PyTorch `saved_tensors_hooks`, `scaled_dot_product_attention` (public APIs) | Custom storage of saved tensors; SDPA | D, A1 |

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

### Against the base (Kaggle T4, `--repeats 2`, notebook version 3, `COMMIT = ebeec3a`)

| | Base Soup 0.77.0 | Ours (v6.1) | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 90.2 (91.62 and 88.83) | 74.5 (74.85 and 74.15) | s = 1.211, 17.4 % shorter |
| Peak GPU memory (nvidia-smi), MiB | 14,345 | 2,827 | m = 5.074, 80 % less |
| Eval loss (public rows) | 2.04966 | 2.04968 | +0.001 % |
| Share of own kernels (check 4) | n/a | 10.6 % | threshold 5 % |

All 10 checks passed: check 6 (loss curve) 0.00 to 0.01 %, check 8 time x0.826 and memory x0.197 of the base, check 10 79 Soup tests passed. `pytest solution -q` at `ebeec3a`: 285 passed, 1 failed. The failure was a LoRA-A gradient tolerance in `test_v6_merged`, a wrong test bound, not a training failure. It is fixed in `6fff364` to the derived bound of 2e-2. Final count: [tests: TODO final count].

### Ablation (each row against the base of its own session)

Base times differ by session (77.1 to 90.2 s across the rows), so compare s and m, not seconds.

| Version | Base, s | Ours, s | Peak, MiB | s | m | Eval loss | All checks |
|---|---|---|---|---|---|---|---|
| Base Soup 0.77.0 | n/a | n/a | 14,345 | 1.000 | 1.000 | n/a | n/a |
| K1 v1 + K4 | 84.96 | 82.55 | 10,383 | 1.029 | 1.382 | -0.003 % | yes |
| v1 = + K3 + K2 | 78.22 | 72.54 | 5,845 | 1.078 | 2.454 | +0.001 % | yes |
| v2 = K1 v3s + U | 77.06 | 61.22 | 5,025 | 1.259 | 2.855 | +0.000 % | yes |
| v4 = v2, final U, S4 off | 79.8 | 61.5 | 5,025 | 1.298 | 2.855 | -0.001 % | yes |
| v5f = v4 + S4 on | 79.8 | 66.9 | 3,845 | 1.192 | 3.731 | -0.003 % | yes |
| v6 = v5f + B, A1, A2, C, D | 84.8 | out of memory | 14,903 | n/a | n/a | n/a | no (1, 3, 4) |
| **v6.1 = v6 + leak fix (submitted)** | 90.2 | 74.5 | 2,827 | 1.211 | 5.074 | +0.001 % | yes |
| v7 = v6.1 + I1 to I4 (not submitted) | 84.1 | 65.1 | 3,233 | 1.292 | 4.437 | +0.001 % | yes |

v4, v5f, v6.1 and v7 are `--repeats 2` runs; the others are single repeats. v6.1 against v5f: the peak falls by 1,018 MiB and s is 1.211 against 1.192, but the sessions differ.

### What we tried after v6.1

We built v7 on branch `v7-candidate` (`cb7d630`): causal SDPA in A1 (I1), one dequantization in S4 backward (I2), a K1 host fast path with a permanent fp16 cache of the k and v weights (I3), and S4 off on the first 7 of 28 layers (I4). In a `--repeats 2` run it measured s = 1.292 and m = 4.437 (3,233 MiB), and all 10 checks passed. We kept v6.1 for the lower peak.

### Limitations

- **Component savings are estimates.** For B, A1, A2, D and C only the sum is measured.
- **Noise.** Our base ran 77 to 92 s across sessions, and the task text gives 1 to 4 % spread between repeats. s is meaningful only inside one session. v5f and v6.1 were not run in the same session.
- **The measured commit.** The run is from `ebeec3a`; the submitted `6fff364` changes one test tolerance. We have not measured `6fff364` as such.
- **Numerics.** K1, S4, B, A2, D and C keep base values by design. A1 relies on torch's SDPA backward, which may not be deterministic on a T4. K3's fp16-pointer variant changes the normalized output by a few fp32 ulp against v5f (test tolerance 2e-6 is an estimate). A2 relies on the same dropout call drawing the same mask from the same generator state; it is guarded and tested on the T4. Measured eval loss is within +0.001 % of the base.
- **Part of the time is not ours.** In the v6.1 profile (16 steps) the GPU was busy 536 ms of a 0.72 s median step; the rest is mostly Soup and Trainer host work. `apply()` takes about 7.5 s and the first step, with Triton compilation, 4.9 s. The Triton cache is empty in every run.
- **Example kernels.** K4, S4 and C use the organizers' SwiGLU kernels, not counted as ours.
- **Scope.** Fp16 on the T4, Qwen2 only. Guards check classes, dtypes and shapes, mostly not the GPU model. Where Soup picks bf16, K1, K4, S4 and B fall back to the original code. Other GPUs and models are not tested.
- **The v6 crash.** The first build of A1, A2, B, C and D ran out of memory at step 22 of 64. A1's throw-away SDPA graph never got a backward, and D's pack hook stored SDPA's own outputs together with the node that owns them, a reference cycle that only that backward breaks. About 0.5 GiB leaked per step. v6.1 runs A1's inner calls under private hooks, D's hooks return `t.detach()`, and `test_v6_no_leak.py` checks that memory stays flat over 8 training steps. The mechanism is our reading of PyTorch autograd.
