# Team Delta: Soup Kernel Challenge (Decentrathon 6.0)

> - **Task.** Make `soup train` (Qwen2.5-1.5B, NF4, LoRA, fp16) faster and smaller on a Kaggle T4 with our own Triton kernels, at the same loss.
> - **Result.** 64.1 s against 82.9 s (s = 1.293), peak 2,335 against 14,345 MiB (m = 6.144), eval loss 2.04965 against 2.04964, 10 of 10 checks passed.
> - **Kernels.** Five of ours: NF4 dequant, LM head + cross entropy, add + RMSNorm forward and backward, SwiGLU; one launcher.
> - **What is new.** No [B, T, V] logits, no padding in token-wise ops, backward tensors recomputed or kept on real rows, compile overlapped with model loading, q, k, v as one GEMM.
> - **Verify.** Copy the notebook [TODO final link] with `COMMIT = "[TODO final commit]"` and click Run All.
> - **Code.** https://github.com/BAITC-Hacks/hack-8cddb4ae-delta, code v11 = `f218f86`; the final commit [TODO final commit] changes only documentation.

## 1. What we did and why

### Where the time and memory go

In the task's profile of a base step, matmul takes 36%, small elementwise ops and casts 44%, vocabulary softmax 7%, attention 5% and NF4 dequantization 5%. The base peak of 14,345 MiB has four big parts, which we removed in this order: fp32 MLP activations, logits over 151,936 tokens, padding, and tensors saved for backward.

No Soup mode is on (`gradient_checkpointing`, `activation_offloading`, `fast_lora`, `use_cut_ce`, `packing`): all recomputation is our own autograd code. Components up to v6.1 fall back to the original code if a guard fails.

### Components

| Component | File in `solution/` | Replaces | Why it helps | Exactness |
|---|---|---|---|---|
| K1 NF4 dequant (Triton), I3 | `kernels/nf4.py` | bitsandbytes dequant, forward and backward | 0.166 against 0.300 ms per [8960, 1536] weight; fp16 k, v cache, 42 MiB | bit-exact |
| K2 head + cross entropy (Triton) | `kernels/head_ce.py`, `head_loss.py` | `lm_head`, cross entropy, TRL metrics | no [B, T, V] logits; with K3, 4,538 MiB less | within test tolerance |
| K3 add + RMSNorm (Triton) | `kernels/addnorm.py`, `layers.py` | every `Qwen2RMSNorm`, first residual add | one pass per row | fp32 sum off by a few ulps |
| C2 SwiGLU (Triton), C1 | `kernels/swiglu.py`, `kernels/__init__.py` | two example kernels; one launcher per kernel | step 1 from 5.1 to 3.6 s | same values |
| K4 fp16 MLP | `mlp.py` | fp32 MLP activations | 3,962 MiB less | same matmul results |
| U unpadding | `unpad.py` | padded rows in norms, adds, MLP | with faster K1, 820 MiB and 11.3 s less | attention, dropout, RNG as base |
| S4 + C MLP recompute, I2 | `mlp_recompute.py` | saved gate and up | S4: 1,180 MiB less; one dequant per backward; SwiGLU backward in place | bit-identical to K4 |
| B fused block | `fused_block.py` | separate nodes for adds, norm, MLP | saves residual, attention output, rstd | as K3 |
| A1 attention recompute, I1 | `sdpa_recompute.py` | SDPA's saved tensors | keeps fp16 q, un-repeated k, v; `is_causal=True` without bool mask | real rows may move by fp16 ulps |
| A2 dropout replay, D | `lora_replay.py`, `saved_compact.py` | dropout masks, padded LoRA-A inputs | 16-byte generator state; real rows only | bit-exact |
| M2 LoRA input recompute, M3 | `lora_recompute.py`, `sdpa_recompute.py` | saved LoRA-A inputs; A1's padded q, k, v | rebuilt from the residual; 2,893 to 2,515 MiB | bit-exact; padded rows get zero gradient |
| P2 elementwise fusion | `kernels/addnorm.py`, `fused_block.py`, `lora_recompute.py` | small torch casts, copies, adds | run inside K3 launches | bit-exact against torch |
| fp16 head | `head_loss.py` | fp32 tied embedding and LM head, 890 MiB | checked fp16 copy | bit-identical downstream |
| Compile overlap | `precompile.py` | compiling inside step 1 | step 1 from 4.0 to 1.1 s | same binaries |
| Q merged q, k, v GEMM | `merged_qkv.py`, `unpad.py`, `lora_recompute.py` | three base GEMMs, three dX GEMMs | one GEMM each way | within test tolerance |

### How the kernels work

**K1** patches `bitsandbytes.matmul_4bit` under a guard. Each thread reads 4 packed bytes at once and decodes NF4 with a tree of 15 selects instead of a table in memory.

**K2** lets cuBLAS compute the head GEMM per vocabulary tile and folds each tile into per-row online softmax statistics; in backward it turns the recomputed tile into the logits gradient in place. TRL's metrics and Soup's rewind log read the same numbers through `ChunkedStep`.

**K3** adds the fp16 attention output to the fp32 residual and applies the norm, one program per row. K1, K2 and K3 launch `soup_*_usig` entries that inline the original bodies and share C2's twelve-argument signature (C1).

**P2** (`USE_ELEMENTWISE_FUSION`) moves B's fp16 casts and the q and v LoRA epilogue into K3 launches, the epilogue only when its scaling is exact in fp16 (here 2.0), and skips peft's defensive clone of the base output. By our count P2 removes about 500 torch launches per step and adds 112 Triton launches; tests check that it gives v9b's bits.

**fp16 head** (`STORE_HEAD_FP16`). On a T4, `prepare_model_for_kbit_training` widens the frozen fp16 tied embedding and LM head to fp32, so their 890 MiB hold only fp16-exact values. After an on-device check of every element (`fp16(w).float() == w`, a NaN refuses) we store an fp16 copy on the same Parameter, keeping the tie and `requires_grad=False`. A forward hook widens the embedding output back to fp32, and K2 reads the fp16 head, as the base does under autocast.

**Compile overlap** (`USE_PRECOMPILE_OVERLAP`). One thread, started at the end of `apply()`, compiles our five kernels with `JITFunction.warmup` and the production specialization keys while Soup loads the model; the first `Qwen2ForCausalLM.forward` or any earlier launch of ours joins it, with a 60 s timeout. Nothing is launched on the GPU, no cache is shipped and no separate process starts.

**Q** (`USE_MERGED_QKV`, `MERGED_QKV_CACHE = False`) is not a kernel. On U's packed path, q, k and v run as one fp16 GEMM against W_qkv = [Wq; Wk; Wv] (2048 by 1536) with the three biases concatenated. W_qkv is assembled per call from K1's dequant (q) and I3's cached rows (k, v); no static memory is added (`MERGED_QKV_CACHE = True` would keep it: 126 MiB more, not measured). Backward computes dX as one GEMM on the concatenated gradient. The q and v LoRA branches run as before, with the same dropout draws in the same order, so the RNG stream is unchanged. Guards: exact classes, biases present and frozen, fp16 autocast, no hooks, shapes 1536, 256, 256; a miss runs the v10d path. Not bit-exact with v10d: cuBLAS may pick another kernel for N = 2048, and dX is rounded once instead of three times. Tests hold hidden states within 5e-3, loss 1e-4, LoRA gradients 2e-2.

**Rule notes.** On 31 October the organizers confirmed, through our team captain, that fp16 storage of fp32 weights is allowed as repacking when the fp32 to fp16 to fp32 round trip is exact and the weight tie and computation stay the same, and that such a compile thread is allowed when everything stays inside the measured run with an empty cache.

### Where the ideas come from

We wrote all kernel code ourselves; banned libraries were read for ideas only, never imported.

- K1: bitsandbytes (NF4 table, absmax layout, rounding order, tile idea), https://github.com/bitsandbytes-foundation/bitsandbytes
- K2: Soup issue #1693 and `ChunkedStep` in `soup_cli/utils/stream_loss_chunk.py`; Liger-Kernel, https://github.com/linkedin/Liger-Kernel; Apple Cut Cross-Entropy, https://github.com/apple/ml-cross-entropy; online softmax, Milakov and Gimelshein 2018, https://arxiv.org/abs/1805.02867
- K3, B, C2, P2: Liger-Kernel; Unsloth, https://github.com/unslothai/unsloth; the starter kit's example (fused add + RMSNorm, fused SwiGLU, LoRA scale and add in one pass)
- U: ByteDance Effective Transformer, https://github.com/bytedance/effective_transformer; FlashAttention's varlen path
- S4, B, A1, M2: activation recomputation, Chen et al. 2016, https://arxiv.org/abs/1604.06174; Korthikanti et al. 2022, https://arxiv.org/abs/2205.05198; Dao et al. 2022, https://arxiv.org/abs/2205.14135
- A2, D, M2: `torch.utils.checkpoint` (`preserve_rng_state`), the Megatron-LM RNG tracker, PyTorch `saved_tensors_hooks`
- Q: Megatron-LM's fused `query_key_value` linear; Unsloth's LoRA QKV path

ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow.

## 2. How to run

1. Open the public notebook [TODO final link] and click Copy & Edit. GPU and Internet need a phone-verified Kaggle account.
2. Set Accelerator to **GPU T4 x2** and switch **Internet on**.
3. In the first code cell, check `REPO_URL = "https://github.com/BAITC-Hacks/hack-8cddb4ae-delta.git"` and `COMMIT = "[TODO final commit]"`.
4. Click Run > Restart & clear cell outputs. Add no GPU cells of your own: check 9 needs idle GPUs.
5. Click Run All. The notebook is `templates/kaggle_notebook.ipynb` with one edit, `--repeats 2` for the jury's order base, ours, ours, base. It installs the pinned packages, clones the repository at `COMMIT` and runs:

   ```bash
   python task/prepare_data.py
   python -m pytest solution -q
   python harness/run_task.py --repeats 2
   python harness/check_rules.py
   python harness/score.py runs/result.json
   ```

6. Expected output, with our version 24 numbers; seconds vary by session:
   - `pytest` reports `623 passed`.
   - `run_task.py` ends with:
     ```
     === result ===
     base      time     82.9 s   peak  14345 MiB   eval loss 2.04964
     solution  time     64.1 s   peak   2335 MiB   eval loss 2.04965
     speedup 1.293x   memory base/solution 6.144x   eval loss vs base 0.0 %
     ```
   - `check_rules.py` prints ten `PASS` lines, then `ALL CHECKS PASSED`.
   - `score.py` ends with `all checks passed on this result: True`.
7. Run All took about 12 minutes for v10b, plus queueing.
8. Variants: in `solution/__init__.py`, `USE_MERGED_QKV = False` gives v10d, plus `USE_PRECOMPILE_OVERLAP = False` v10c, plus `STORE_HEAD_FP16 = False` v10b, plus `USE_ELEMENTWISE_FUSION = False` v9b. Push a branch and put its commit in `COMMIT`.

## 3. Results and limitations

### Final run against the base

Kaggle T4, notebook version 24, `--repeats 2`, code `f218f86`.

| | Base Soup 0.77.0 | Ours (v11) | Ratio |
|---|---|---|---|
| Time of `soup train`, s (repeats) | 82.9 (82.02, 83.83) | 64.1 (64.27, 63.97) | s = 1.293, wall 22.7% shorter |
| Peak GPU memory, MiB | 14,345 | 2,335 | m = 6.144, 83.7% less |
| Eval loss, public rows | 2.04964 | 2.04965 | +0.000% |
| Own kernels' share, check 4 | | 11.5% | threshold 5% |
| `check_rules.py` | | 10 of 10 passed | check 8: time x0.773, memory x0.163 |
| `pytest solution -q` | | 623 passed, 0 failed | 600 for v10d |

s = 1.293 is below v10d's 1.310 only because this session's base was faster (82.9 against 84.9 s); the same-session comparison below is the reliable one.

Q adds no Triton kernel. In the v11 run check 3 listed `soup_addnorm_bwd_v10`, `soup_addnorm_fwd_v10`, `soup_head_ce_usig`, `soup_nf4_dequant_usig` and `soup_swiglu_kernel`; check 6 held the loss curve within 0.00 to 0.01 % of the base; check 9 was clean; check 10 passed Soup's 79 tests.

### Ablation

Each row is against the base of its own session (77 to 90 s); rows from v4 on are `--repeats 2` runs. All runs below passed the 10 checks except the example only (checks 3 and 4) and v6.

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
| v9b, two sessions | M5 off | 86.8; 90.3 | 67.9; 72.6 | 2,515 | 1.278; 1.244 | 5.704 |
| v10b | + P2 | 89.8 | 71.6 | 2,495 | 1.254 | 5.750 |
| v10c | + fp16 head | 87.6 (89.11, 86.11) | 66.3 (65.56, 67.02) | 2,335 | 1.322 | 6.144 |
| v10d | + overlap (previous) | 84.9 | 64.8 | 2,335 | 1.310 | 6.144 |
| **v11** | **+ Q (submitted)** | **82.9** | **64.1** | **2,335** | **1.293** | **6.144** |

### Same-session comparisons

Each pair reads without, then with the change.

- **fp16 head, v8b code**, version 13, ABBA, base 78.16 s: 65.43 and 66.69 s against 65.08 and 65.64 s, within noise; peak 2,893 against 2,453 MiB.
- **fp16 head, v9 code** (`c6d756a`, M5 on), version 14, one run each: peak 2,475 against 2,335 MiB; wall 69.85 against 69.10 s.
- **P2 alone**, version 18, base 98.4 s, one run each: wall 73.29 against 72.79 s, inside the noise; median step 0.725 against 0.714 s; peak 2,515 against 2,495 MiB.
- **Compile overlap alone**, version 21, ABBA: v10c 66.23 and 67.08 s, v10d 63.97 and 65.11 s, 2.1 s less on the means against repeat spreads of 0.85 and 1.14 s; step 1 4.05 and 4.02 s against 1.10 s twice; `apply()` 8.79 and 9.02 s against 8.57 and 7.87 s; peak 2,335 MiB in all four.
- **Q alone**, version 23, ABBA: v10d 65.84 and 65.69 s, v11 65.24 and 65.19 s, 0.55 s less on the means; peak 2,335 MiB in both.

### Switched off or dropped

- **M5, chunked backward** (`USE_BWD_CHUNKING = False`): 40 MiB less peak for an estimated 0.5 s, and chunked cuBLAS GEMMs differ in the last bits at some row counts.
- **P1, compile pool joined inside `apply()`**: step 1 fell from 3.6 to 1.0 s, but `apply()` grew by 2.8 s and wall time did not move; the Triton frontend and `apply()`'s imports hold the GIL.
- **P3, merged LoRA** (`USE_LORA_MERGED`): about 0.15 s per run [estimate], below our bar of 5 ms per step.
- **Per-row causal SDPA**: about 40% less attention GPU time, lost to about 1,700 extra CPU operations per step, 63.17 against 62.93 s.
- **S4 off on 7 layers**: 340 MiB more peak, no speed gain.
- **v6 leak**: D's pack hook kept A1's inner SDPA graph alive, so v6 ran out of memory at step 22; v6.1 runs A1's inner calls under private hooks, and `test_v6_no_leak.py` checks flat memory over 8 steps.

### Limitations

- **C2 is the example's math**: without any SwiGLU kernel of ours, our share was 9.8% at v7, still above 5%.
- **Session variance**: the two full v9b runs gave s = 1.278 and 1.244, more than P2's gain or the head's time effect, so we do not credit the head with v10c's rise from 1.254 to 1.322; in the v10c, v10d and v11 runs our repeats differed by up to 1.46 s and the base's by up to 3.00 s, and per-component splits are not measured.
- **Q**: 0.55 s in one ABBA session, below the s swing between the two v9b sessions. Its single fp16 dX overflows where the sum of three terms exceeds 65,504; the window is narrow by the code's analysis, unmeasured.
- **fp16 head below paper**: the final peak is 160 MiB under v10b's from another session, our evidence that the store ran, against 445 MiB freed on paper; we think, untested, that the peak now sits in backward transients.
- **Memory bought with time**: S4, A1 and M2 redo work in backward; S4 alone cost about 5.4 s (v4 against v5f, different sessions), and v9 ran 2.2 s slower than v8b in one session.
- **Host code**: the GPU idles about 100 to 180 ms per step in Soup and TRL code (logging, rewind, NaN checks, optimizer), and at v8b our compile made step 1 take 3.5 s against 1.8 s, which v10d only overlaps with model loading.
- **Not bit-equal to the base**: K3, A1 and Q change last bits and A1's backward may be nondeterministic, yet eval loss stayed within 0.003% of the base in every full run.
- **Scope**: Soup's T4 fp16 path and Qwen2 only; under bf16, K1, K4, S4 and B fall back to the original code and the head stays fp32 unless fp16-exact; other GPUs and models are untested.

## 4. Repository map

`solution/` is our code: `__init__.py` (`apply()`, every flag), `kernels/` (Triton), `tests/`; `swiglu.py` and `test_kernels.py` are the starter kit's example, unchanged. `VERSIONS.md`: every version's branch, flags, component names. `harness/`, `tests/`, `task/`, `templates/`, `NOTICE` and `LICENSE` (Apache-2.0) are the starter kit, unchanged.
