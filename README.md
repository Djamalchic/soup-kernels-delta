# Team Delta: Soup Kernel Challenge (Decentrathon 6.0)

- Public Kaggle notebook (saved version with results): [TODO final link]
- Repository: https://github.com/BAITC-Hacks/hack-8cddb4ae-delta, final commit [TODO final commit]
- Submitted code: v9b, commit `91d70d3` (branch `v9b-candidate`). The final commit only adds this README and `SUBMISSION.md`.

**Result.** On a Kaggle T4, in the jury's order (base, ours, ours, base), `soup train` takes 67.9 s with our solution against 86.8 s for base Soup: s = 1.278, 21.8% less wall time. The GPU peak is 2,515 MiB against 14,345 MiB: m = 5.704, 82.5% less. Eval loss is 2.04964 against 2.04965. All 10 checks pass; `pytest solution -q` gives 485 passed, 0 failed.

## 1. What we did and why

### Where the time and memory go

The task trains Qwen2.5-1.5B in NF4 with LoRA r=16 on q and v, batch 4, up to 512 tokens, 64 steps, fp16 on one T4. The task's profile of a base step: matmul 36%, small elementwise ops and casts 44%, vocabulary softmax 7%, attention 5%, NF4 dequantization 5%.

The base peak of 14,345 MiB has four big parts. We removed them in this order, each with a measured drop:

1. **fp32 MLP activations**: three fp32 [tokens, 8960] tensors per layer. Fix: K4, 3,962 MiB less.
2. **Logits over 151,936 tokens.** Fix: K2 with K3, 4,538 MiB less.
3. **Padding.** Fix: U with a faster K1, 820 MiB and 11.3 s less.
4. **Per-layer tensors kept for backward.** Fix: S4, then B, A1, A2, C, D, then M2 and M3. Peak 5,025 MiB (v4), 3,845 (v5f), 2,827 (v6.1), 2,515 (v9b).

Time came from fused passes (K2, K3), a dequantization kernel faster than bitsandbytes (K1), unpadding (U) and one Triton launcher (C1). No Soup mode is on: `gradient_checkpointing`, `activation_offloading`, `fast_lora`, `use_cut_ce` and `packing` stay off, and all recomputation is our own autograd code. The components up to v6.1 are guarded: if a guard fails, the original code runs.

### Our Triton kernels

**K1, NF4 dequantization** (`solution/kernels/nf4.py`). A guarded patch of `bitsandbytes.matmul_4bit` replaces bitsandbytes' dequantization in forward and in the backward re-dequantization. A thread loads 4 packed bytes in one 32-bit read, each byte gives two fp16 values, and the NF4 table is a tree of 15 selects instead of a memory lookup. Output is bit-exact (`torch.equal` against `dequantize_4bit`); a [8960, 1536] weight takes 0.166 ms against 0.300 ms on the bitsandbytes path. A cheaper host check and a permanent fp16 copy of the small k and v weights (about 42 MiB) cut per-call overhead (I3).

**K2, LM head + cross entropy without logits** (`solution/kernels/head_ce.py`, wiring in `solution/head_loss.py`). cuBLAS does the head GEMM per vocabulary tile, and our kernel folds each tile into per-row online softmax statistics. In backward it turns the recomputed tile into the logits gradient in place. The [B, T, V] logits never exist; TRL's metrics and Soup's rewind log read the same numbers through Soup's own `ChunkedStep` class.

**K3, fused residual add + RMSNorm** (`solution/kernels/addnorm.py`, wiring in `solution/layers.py`). One program per row adds the fp16 attention output to the fp32 residual and applies `post_attention_layernorm` in one pass; the same kernel serves every other `Qwen2RMSNorm`. The backward kernel returns the norm input gradient and adds the incoming residual gradient in the same pass.

**C2, SwiGLU** (`solution/kernels/swiglu.py`). `soup_swiglu_kernel` does forward, backward and in-place backward in one jit and replaces the starter kit's two example kernels. It computes the example's math operation for operation, with the same bits (`solution/tests/test_soup_swiglu.py`). See the limitations for how we count it.

**C1, one launcher.** Triton builds a C launcher for each distinct runtime signature. K1, K2 and K3 take the same twelve runtime arguments through `soup_*_usig` entries that inline the original kernel bodies unchanged; C2's own kernel has the same signature, so Triton builds one. In one session C1 and C2 together cut step 1, where Triton compiles, from 5.1 s to 3.6 s.

### PyTorch-level changes

**K4, fp16 MLP** (`solution/mlp.py`). Soup computes 4-bit layers in fp16 on a T4, so casting the MLP input to fp16 first gives the same matmul results, and autograd keeps gate and up in fp16.

**U, unpadding** (`solution/unpad.py`). Norms, residual adds and the MLP run on each batch's real tokens only; rows of different batches are never mixed. Attention keeps the base [B, T, C] shape, so LoRA dropout masks and the RNG stream match the base.

**S4 + C, MLP recompute** (`solution/mlp_recompute.py`). The MLP saves only its fp16 input and recomputes gate and up in backward with the same 4-bit GEMMs, dequantizing them once per call (I2). The SwiGLU backward writes its gradients over its own inputs (C). Outputs and gradients are bit-identical to K4.

**B, fused block** (`solution/fused_block.py`). One autograd node covers the first residual add, post-attention norm, S4 MLP and second residual add. It saves the fp32 residual, the fp16 attention output and rstd (6 bytes per element instead of 10) and rebuilds the rest with K3.

**A1, attention recompute** (`solution/sdpa_recompute.py`). Registered as transformers' `"sdpa"` entry. It saves fp16 q and the un-repeated k and v instead of q, 12-head k and v, output, logsumexp and the float mask, and reruns torch's SDPA in backward. On the packed path it uses `is_causal=True` without a bool mask (I1); padding is on the right, so real rows never see it.

**A2, dropout replay** (`solution/lora_replay.py`). The q and v LoRA dropout saves the 16-byte CUDA generator state instead of its bool mask and replays the mask in backward.

**D, real rows only** (`solution/saved_compact.py`). Inside a packed layer's attention, `saved_tensors_hooks` keep only the real rows of the fp16 LoRA-A inputs. Padding rows are exact zeros, so unpacking is exact.

### New in v9b

**M2, LoRA input recompute** (`solution/lora_recompute.py`, `USE_LORA_INPUT_RECOMPUTE`). The fp16 inputs of the q and v LoRA-A are not saved. Backward rebuilds them from the fp32 residual that K3 and B already keep: K3 norm replay, `index_copy` of the real rows, dropout replay from the saved RNG state, fp16 cast. One rebuild serves both q and v.

**M3, attention tensors on real rows** (`solution/sdpa_recompute.py`; the flag `USE_SDPA_ROW_COMPACT` is in `solution/__init__.py` and is called `USE_ROW_COMPACT` in `sdpa_recompute.py`). A1 saves fp16 q, k and v on the real rows only and scatters them back into zeros in backward. Padded query rows get a zero gradient and padded keys are masked, so real rows are exact.

Together M2 and M3 took the peak from 2,893 MiB (v8b) to 2,515 MiB, 378 MiB less, with s at 1.275 (v8b) and 1.278 (v9b).

### In the code, switched off

- **M5, chunked backward** (`USE_BWD_CHUNKING = False`). Row-chunked S4 backward and a smaller K2 vocabulary tile. Full runs with and without it differ by 40 MiB of peak (2,475 against 2,515 MiB) for an estimated 0.5 s. It is the only one of the v9 changes (M2, M3, M5) that is not bit-exact: chunked cuBLAS GEMMs differ in the last bits at some row counts (for example 513 rows), because cuBLAS picks its kernel per shape.
- **fp16 LM head** (`STORE_HEAD_FP16 = False`). Stores the frozen tied embedding and head in fp16 after an on-device check that the cast is lossless. See the limitations.

### Where the ideas come from

We wrote all kernel code ourselves. Banned libraries were read for ideas only and are not imported.

| Source | What we took | Used in |
|---|---|---|
| bitsandbytes, https://github.com/bitsandbytes-foundation/bitsandbytes | NF4 table, nested absmax layout, rounding order, tile idea of `kDequantizeBlockwise` | K1 |
| Soup issue #1693; `ChunkedStep` in `soup_cli/utils/stream_loss_chunk.py` (soup-cli 0.77.0) | Contract for a step without logits | K2 |
| Liger-Kernel, https://github.com/linkedin/Liger-Kernel; Apple Cut Cross-Entropy, https://github.com/apple/ml-cross-entropy | Chunked linear + cross entropy, logits recomputed in backward | K2 |
| Liger-Kernel; Unsloth, https://github.com/unslothai/unsloth; the starter kit's example | Fused add + RMSNorm; fused SwiGLU | K3, B, C2 |
| Milakov and Gimelshein 2018, https://arxiv.org/abs/1805.02867 | Online softmax | K2 |
| ByteDance Effective Transformer, https://github.com/bytedance/effective_transformer; FlashAttention varlen unpad path | Removing padding for token-wise ops | U |
| Chen et al. 2016, https://arxiv.org/abs/1604.06174; Korthikanti et al. 2022, https://arxiv.org/abs/2205.05198; Dao et al. 2022, https://arxiv.org/abs/2205.14135 | Activation recomputation; recomputing only the attention core | S4, B, A1, M2 |
| `torch.utils.checkpoint` (`preserve_rng_state`); Megatron-LM RNG tracker | Restoring the generator state to redo a dropout mask | A2, M2 |
| PyTorch `saved_tensors_hooks`, `scaled_dot_product_attention` | Custom storage of saved tensors; causal SDPA | D, A1, M2 |

ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow. Every kernel was tested with `pytest solution -q` and measured on a Kaggle T4.

## 2. How to run

1. Open the public notebook: [TODO final link]. It is the starter kit's `templates/kaggle_notebook.ipynb`. Click Copy & Edit (a phone-verified Kaggle account is needed for GPU and Internet).
2. In the right panel set Accelerator to **GPU T4 x2** and switch **Internet on**.
3. Check the first code cell: `REPO_URL = "https://github.com/BAITC-Hacks/hack-8cddb4ae-delta.git"` and `COMMIT = "[TODO final commit]"`. If the repository is private, add a `GITHUB_TOKEN` secret.
4. Run > Restart & clear cell outputs. Add no GPU cells of your own: check 9 needs idle GPUs.
5. Run All. The notebook installs the pinned packages (`soup-cli[train]==0.77.0`, `transformers==5.16.1`, `peft==0.20.0`, `trl==0.29.1`, `bitsandbytes==0.50.2`, `accelerate==1.14.0`), clones the repository at `COMMIT` and runs:

```bash
python task/prepare_data.py
python -m pytest solution -q              # 485 passed
python harness/run_task.py --repeats 2    # jury order: base, ours, ours, base
python harness/check_rules.py             # 10 checks; check 10 runs Soup's 79 tests with our patches
python harness/score.py runs/result.json
```

The only edit to the template is in the `run_task.py` cell: `--repeats 2`. Run All takes about 12 minutes of notebook time (saved versions 16 and 17 took 10.7 and 11.5 minutes; Kaggle queueing adds to the wall clock). The last lines print `speedup`, `memory base/solution` and `eval loss vs base`; the numbers are saved in `runs/result.json` and the checks in `runs/checks.json`.

Quick look at our side only: `python harness/run_task.py --only solution --out runs_try` (checks that need the base then fail). To compare variants, flip a flag in `solution/__init__.py` (for example `USE_BWD_CHUNKING = True` for M5), push to a branch and put its commit in `COMMIT`.

## 3. Results and limitations

### Final run against the base

Kaggle T4, notebook version 16, `--repeats 2`, code `91d70d3`.

| | Base Soup 0.77.0 | Ours (v9b) | Ratio |
|---|---|---|---|
| Time of `soup train`, s (two repeats) | 86.8 (88.68, 84.87) | 67.9 (68.53, 67.23) | s = 1.278, 21.8% less |
| Peak GPU memory, MiB | 14,345 | 2,515 | m = 5.704, 82.5% less |
| Eval loss, public rows | 2.04965 | 2.04964 | -0.000% |
| Own kernels' share (check 4) | n/a | 11.6% | threshold 5% |

All 10 checks PASS. Check 3 lists `soup_addnorm_bwd_usig`, `soup_addnorm_fwd_usig`, `soup_head_ce_usig`, `soup_nf4_dequant_usig` and `soup_swiglu_kernel`; the bodies `soup_*_kernel` run as inlined device functions, so the harness reports them as "not launched" and the check passes. Check 6 (loss curve): 0.00 to 0.01%. Check 8: time x0.782, memory x0.175. Check 9: clean. Check 10: 79 passed. The run's `triton_cache` holds exactly one `__triton_launcher*.so` (C1).

### Ablation

Each row is measured against the base of its own session. Base time ranged from 78 to 90 s between sessions, so compare s and m, not seconds.

| Version | Adds | Base, s | Ours, s | Peak, MiB | s | m |
|---|---|---|---|---|---|---|
| Example only | starter kit SwiGLU | 77.83 | 83.05 | 12,385 | 0.937 | 1.158 |
| K1 + K4 | NF4 kernel, fp16 MLP | 84.96 | 82.55 | 10,383 | 1.029 | 1.382 |
| v1 | + K3, K2 | 78.22 | 72.54 | 5,845 | 1.078 | 2.454 |
| v2 | + faster K1, U | 77.06 | 61.22 | 5,025 | 1.259 | 2.855 |
| v4 | final U, tests | 79.8 | 61.5 | 5,025 | 1.298 | 2.855 |
| v5f | + S4 | 79.8 | 66.9 | 3,845 | 1.192 | 3.731 |
| v6 | + B, A1, A2, C, D | 84.8 | out of memory at step 22 | 14,903 | n/a | n/a |
| v6.1 | leak fixed | 90.2 | 74.5 | 2,827 | 1.211 | 5.074 |
| v7 | + I1 to I3, S4 off on 7 layers | 84.1 | 65.1 | 3,233 | 1.292 | 4.437 |
| v8b | S4 on all layers, C1, C2 | 88.4 | 69.3 | 2,893 | 1.275 | 4.958 |
| v9 | + M2, M3, M5 | 81.3 | 67.1 | 2,475 | 1.212 | 5.796 |
| **v9b** | **M5 off (submitted)** | **86.8** | **67.9** | **2,515** | **1.278** | **5.704** |

From v4 on, rows are `--repeats 2` runs. Two same-session comparisons back the closest calls: v6.1 against v8b code, 67.7 against 63.3 s (base 79.81 s); v8b, v9 and v9 with the fp16 head, 67.69, 69.85 and 69.10 s (base 87.89 s).

### Tried and dropped

- **Per-row causal SDPA.** Attention GPU time was cut by about 40% (56 to 26 ms per step raw, different sessions), but about 1,700 extra small CPU-side operations per step ate the gain: 63.17 s with it, 62.93 s without, same session. The step is CPU-bound there.
- **S4 off on the first layers.** 70.65 to 72.39 s with 2 or 7 layers off against 71.95 s with S4 everywhere, same session: no speed gain, and 340 MiB more peak with 7 off.
- **M5**, see above.
- **The v6 leak.** v6 ran out of memory at step 22: D's pack hook kept A1's inner SDPA graph alive. v6.1 runs A1's inner calls under private hooks, D returns `t.detach()`, and `solution/tests/test_v6_no_leak.py` checks flat memory over 8 steps.

### Limitations

- **Memory left on the table.** The fp16 head is off by team decision. In one session (order v8b, head, head, v8b) it ran 65.08 and 65.64 s against 65.43 and 66.69 s, at 2,453 against 2,893 MiB: 440 MiB less at no time cost. Section 8 of the task forbids changing the model, section 9 suggests repacking weights; we chose not to bet admission on the reading.
- **C2 is the example's math.** The jury may count `soup_swiglu_kernel` as the example. The 11.6% share includes it; without any SwiGLU kernel of ours the share was 9.8% at v7, so the 5% threshold holds either way.
- **Noise and estimates.** In the final run our repeats differ by 1.3 s and the base repeats by 3.8 s. v9b and v8b never ran in one session, so the equal s means "no visible cost", not "free". Only sums are measured: the split among B, A1, A2, C and D, or between M2 and M3, is not.
- **Memory bought with time.** S4, A1 and M2 redo work in backward. S4 alone cost about 5.4 s (v4 against v5f, different sessions). In one session v9 with M2, M3 and M5 ran 2.2 s slower than v8b.
- **Compilation.** C1 shortens step 1 but does not remove it: 3.5 s against 1.8 s for the base at v8b, same session.
- **Time that is not ours.** The GPU idles about 100 to 180 ms per step in Soup and TRL host code (metric logging, rewind, NaN checks, optimizer), which our kernels cannot reach.
- **Numerics.** K1, S4, A2, C, D, C1, C2, M2 and M3 are bit-exact by design (`torch.equal` tests). B gives the values of the unfused K3 path, not the base's: it repeats that path bit for bit, and K3 is not equal to the base. A1's SDPA backward may not be deterministic on a T4, causal SDPA may move real rows by fp16 ulps, and K3 reorders the norm's fp32 sum by a few ulps. Eval loss stayed within 0.003% of the base in every finished version.
- **Scope.** Built and measured for Soup's T4 fp16 path and Qwen2 only; where Soup picks bf16, K1, K4, S4 and B fall back to the original code. Other GPUs and models are untested.
