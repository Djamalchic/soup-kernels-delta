# Team Delta: Soup Kernel Challenge (Decentrathon 6.0)

- Public Kaggle notebook (saved version with results): [TODO final link]
- Repository: https://github.com/BAITC-Hacks/hack-8cddb4ae-delta, final commit [TODO final commit]
- Submitted code: v10b, commit `69b0c95` (branch `v10b-candidate`) = v9b (`91d70d3`) plus one item, P2. The final commit only adds this README and `SUBMISSION.md`.

**Result.** On a Kaggle T4, in the jury's order (base, ours, ours, base), `soup train` takes 71.6 s with our solution against 89.8 s for base Soup: s = 1.254 (wall time 20.3% shorter). The GPU peak is 2,495 MiB against 14,345 MiB: m = 5.750 (82.6% less). Eval loss: 2.04965 against 2.04964 (+0.000%). Checks passed: 10 of 10. The value of s depends on the session, and this one was slow (base 89.8 s). The previous version, v9b, gave s = 1.244 in a comparable slow session (72.6 against 90.3 s) and s = 1.278 in a fast one (67.9 against 86.8 s; 2,515 MiB, m = 5.704). The gain of v10b over v9b was measured head to head in one session (73.29 to 72.79 s, 2,515 to 2,495 MiB).

## 1. What we did and why

### Where the time and memory go

The task trains Qwen2.5-1.5B in NF4 with LoRA r=16 on q and v, batch 4, up to 512 tokens, 64 steps, fp16 on one T4. The task's profile of a base step: matmul 36%, small elementwise ops and casts 44%, vocabulary softmax 7%, attention 5%, NF4 dequantization 5%.

The base peak of 14,345 MiB has four big parts. We removed them in this order, each with a measured drop:

1. **fp32 MLP activations**: three fp32 [tokens, 8960] tensors per layer. Fix: K4, 3,962 MiB less.
2. **Logits over 151,936 tokens.** Fix: K2 with K3, 4,538 MiB less.
3. **Padding.** Fix: U with a faster K1, 820 MiB and 11.3 s less.
4. **Per-layer tensors kept for backward.** Fix: S4, then B, A1, A2, C, D, then M2 and M3. Peak 5,025 MiB (v4), 3,845 (v5f), 2,827 (v6.1), 2,515 (v9b), 2,495 (v10b).

Time came from fused passes (K2, K3), a dequantization kernel faster than bitsandbytes (K1), unpadding (U), one Triton launcher (C1) and, in v10b, small casts moved into launches we already make (P2). No Soup mode is on: `gradient_checkpointing`, `activation_offloading`, `fast_lora`, `use_cut_ce` and `packing` stay off, and all recomputation is our own autograd code. The components up to v6.1 are guarded: if a guard fails, the original code runs.

### Our Triton kernels

**K1, NF4 dequantization** (`solution/kernels/nf4.py`). A guarded patch of `bitsandbytes.matmul_4bit` replaces its dequantization in forward and in backward. A thread loads 4 packed bytes in one 32-bit read, each byte gives two fp16 values, and the NF4 table is a tree of 15 selects instead of a memory lookup. Output is bit-exact (`torch.equal` against `dequantize_4bit`); a [8960, 1536] weight takes 0.166 ms against 0.300 ms. I3: a cheaper host check and a permanent fp16 copy of the small k and v weights (about 42 MiB).

**K2, LM head + cross entropy without logits** (`solution/kernels/head_ce.py`, wiring in `solution/head_loss.py`). cuBLAS does the head GEMM per vocabulary tile, and our kernel folds each tile into per-row online softmax statistics. In backward it turns the recomputed tile into the logits gradient in place. The [B, T, V] logits never exist; TRL's metrics and Soup's rewind log read the same numbers through Soup's `ChunkedStep`.

**K3, fused residual add + RMSNorm** (`solution/kernels/addnorm.py`, wiring in `solution/layers.py`). One program per row adds the fp16 attention output to the fp32 residual and applies `post_attention_layernorm`; the same kernel serves every other `Qwen2RMSNorm`. The backward kernel returns the norm input gradient and adds the incoming residual gradient in the same pass. With P2 both entries also do the casts around them.

**C2, SwiGLU** (`solution/kernels/swiglu.py`). `soup_swiglu_kernel` does forward, backward and in-place backward in one jit and replaces the starter kit's two example kernels. It computes the example's math operation for operation, with the same bits. See the limitations.

**C1, one launcher.** Triton builds a C launcher for each distinct runtime signature. K1, K2 and K3 take the same twelve runtime arguments through `soup_*_usig` entries that inline the original kernel bodies (K3 with P2: the two `_v10` entries); C2 has the same signature, so Triton builds one launcher. C1 and C2 together cut step 1, where Triton compiles, from 5.1 s to 3.6 s in one session.

### PyTorch-level changes

**K4, fp16 MLP** (`solution/mlp.py`). Soup computes 4-bit layers in fp16 on a T4, so casting the MLP input to fp16 first gives the same matmul results, and autograd keeps gate and up in fp16.

**U, unpadding** (`solution/unpad.py`). Norms, residual adds and the MLP run on each batch's real tokens only; rows of different batches are never mixed. Attention keeps the base shape, so LoRA dropout masks and the RNG stream match the base.

**S4 + C, MLP recompute** (`solution/mlp_recompute.py`). The MLP saves only its fp16 input and recomputes gate and up in backward with the same 4-bit GEMMs, dequantizing them once per call (I2). The SwiGLU backward writes its gradients over its own inputs (C). Bit-identical to K4.

**B, fused block** (`solution/fused_block.py`). One autograd node covers the first residual add, post-attention norm, S4 MLP and second residual add. It saves the fp32 residual, the fp16 attention output and rstd (6 bytes per element instead of 10) and rebuilds the rest with K3.

**A1, attention recompute** (`solution/sdpa_recompute.py`). Registered as transformers' `"sdpa"` entry. It saves fp16 q and the un-repeated k and v instead of q, 12-head k and v, output, logsumexp and mask, and reruns torch's SDPA in backward. On the packed path it uses `is_causal=True` without a bool mask (I1); padding is on the right, so real rows never see it.

**A2, dropout replay** (`solution/lora_replay.py`). The q and v LoRA dropout saves the 16-byte CUDA generator state instead of its mask and replays the mask in backward. **D, real rows only** (`solution/saved_compact.py`): `saved_tensors_hooks` keep only the real rows of the fp16 LoRA-A inputs; padding rows are exact zeros.

**M2, LoRA input recompute** (`solution/lora_recompute.py`, v9b). The fp16 inputs of the q and v LoRA-A are not saved. Backward rebuilds them from the fp32 residual that K3 and B already keep: norm replay, `index_copy` of the real rows, dropout replay from the saved RNG state, fp16 cast.

**M3, attention tensors on real rows** (`solution/sdpa_recompute.py`, v9b). A1 saves fp16 q, k and v on the real rows only and scatters them back into zeros in backward. Padded query rows get a zero gradient and padded keys are masked, so real rows are exact. M2 and M3 took the peak from 2,893 MiB (v8b) to 2,515 MiB.

### New in v10b

**P2, elementwise fusion** (`USE_ELEMENTWISE_FUSION` in `solution/__init__.py`, three sub-flags, all on; the master flag off gives v9b). Our own paths ran small casts, copies and adds as separate torch launches. P2 moves them into launches we already make. K3 gets two entries with the same twelve-slot signature, `soup_addnorm_fwd_v10` and `soup_addnorm_bwd_v10`. While P2 is on, every K3 launch goes to them, so a run still builds five cubins and one launcher (`solution/tests/test_ew_fusion.py` checks it).

- **`EW_K3_FP16`** (B, `solution/fused_block.py`). K3's forward writes the fp16 copy of the normalized output that the next GEMM needs, in forward and in the backward recompute, instead of a separate torch cast. The backward reads the fp16 gradient from the GEMMs and writes the fp16 gradient of the attention output back in place.
- **`EW_LORA_EPILOGUE`** (M2, `solution/lora_recompute.py`). The q and v LoRA epilogue (fp32 cast of the base output, scaling, add) and its backward run inside the K3 entry, one launch each way. Taken only when the scaling is exact in fp16 (here 32 / 16 = 2.0).
- **`EW_LORA_NO_CLONE`** (M2). peft's defensive clone of the base layer's output is skipped: an identity before an out-of-place add. With the epilogue running, it matters only where the epilogue is refused.

Counted from the code, P2 removes about 500 small torch launches per step and adds 112 Triton launches. The casts round to nearest even as torch's do; scaling and adds are the same fp32 operations on the same values, overflow to inf included.

### In the code, switched off

- **M5, chunked backward** (`USE_BWD_CHUNKING = False`): 40 MiB less peak for an estimated 0.5 s, and not bit-exact (chunked cuBLAS GEMMs differ in the last bits at some row counts, for example 513).
- **fp16 LM head** (`STORE_HEAD_FP16 = False`): the frozen tied embedding and head stored in fp16 after an on-device lossless check. See the limitations.

### Where the ideas come from

We wrote all kernel code ourselves. Banned libraries were read for ideas only and are not imported.

| Source | What we took | Used in |
|---|---|---|
| bitsandbytes, https://github.com/bitsandbytes-foundation/bitsandbytes | NF4 table, absmax layout, rounding order, tile idea | K1 |
| Soup issue #1693; `ChunkedStep` in `soup_cli/utils/stream_loss_chunk.py` (soup-cli 0.77.0) | Contract for a step without logits | K2 |
| Liger-Kernel, https://github.com/linkedin/Liger-Kernel; Apple Cut Cross-Entropy, https://github.com/apple/ml-cross-entropy; Milakov and Gimelshein 2018, https://arxiv.org/abs/1805.02867 | Chunked linear + cross entropy; online softmax | K2 |
| Liger-Kernel; Unsloth, https://github.com/unslothai/unsloth; the starter kit's example | Fused add + RMSNorm; fused SwiGLU; LoRA scale and add in one pass | K3, B, C2, P2 |
| ByteDance Effective Transformer, https://github.com/bytedance/effective_transformer; FlashAttention varlen path | Removing padding | U |
| Chen et al. 2016, https://arxiv.org/abs/1604.06174; Korthikanti et al. 2022, https://arxiv.org/abs/2205.05198; Dao et al. 2022, https://arxiv.org/abs/2205.14135 | Activation recomputation | S4, B, A1, M2 |
| `torch.utils.checkpoint` (`preserve_rng_state`); Megatron-LM RNG tracker; PyTorch `saved_tensors_hooks` | Dropout replay from the generator state; custom saved tensors | A2, D, M2 |

ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow.

## 2. How to run

1. Open the public notebook: [TODO final link], the starter kit's `templates/kaggle_notebook.ipynb`. Click Copy & Edit (a phone-verified Kaggle account is needed for GPU and Internet).
2. Set Accelerator to **GPU T4 x2** and switch **Internet on**.
3. Check the first code cell: `REPO_URL = "https://github.com/BAITC-Hacks/hack-8cddb4ae-delta.git"` and `COMMIT = "[TODO final commit]"`. For a private repository, add a `GITHUB_TOKEN` secret.
4. Run > Restart & clear cell outputs. Add no GPU cells of your own: check 9 needs idle GPUs.
5. Run All. The notebook installs the template's pinned packages, clones the repository at `COMMIT` and runs:

```bash
python task/prepare_data.py
python -m pytest solution -q              # 578 passed
python harness/run_task.py --repeats 2    # jury order: base, ours, ours, base
python harness/check_rules.py             # 10 checks; check 10 runs Soup's 79 tests with our patches
python harness/score.py runs/result.json
```

The only edit to the template is in the `run_task.py` cell: `--repeats 2`. Run All takes about 12 minutes of notebook time (v9b: 10.7 and 11.5 minutes), plus Kaggle queueing. To compare variants, flip a flag in `solution/__init__.py` (`USE_ELEMENTWISE_FUSION = False` gives v9b), push to a branch and put its commit in `COMMIT`.

## 3. Results and limitations

### Final run against the base

Kaggle T4, notebook version 19, `--repeats 2`, code `69b0c95`.

| | Base Soup 0.77.0 | Ours (v10b) | Ratio |
|---|---|---|---|
| Time of `soup train`, s (two repeats, mean) | 90.72 and 88.83 (89.8) | 71.42 and 71.83 (71.6) | s = 1.254 (wall 20.3% shorter) |
| Peak GPU memory, MiB | 14,345 | 2,495 | m = 5.750 (82.6% less) |
| Eval loss, public rows | 2.04964 | 2.04965 | +0.000% |
| Own kernels' share (check 4) | n/a | 12.8% | threshold 5% |

All 10 checks passed. Check 3 lists `soup_addnorm_bwd_v10`, `soup_addnorm_fwd_v10`, `soup_head_ce_usig`, `soup_nf4_dequant_usig` and `soup_swiglu_kernel`. Check 6: 0.00 to 0.01%. Check 8: time x0.798, memory x0.174. Check 9: clean. Check 10: 79 passed. `pytest solution -q`: 578 passed, 0 failed; the suite includes the v9b-bits comparison tests. The run's `triton_cache` holds one `__triton_launcher*.so` and 5 cubins. For comparison, v9b's final run (version 16) passed all 10 checks and 485 tests.

### P2 against v9b, one scouting session

Notebook version 18, a slow session (base 98.4 s), one run each, commit `1a859d5` (P2 without P1; the submitted `69b0c95` is v9b plus P2). Wall time 73.29 s for v9b against 72.79 s with P2; median step 0.725 against 0.714 s; peak 2,515 against 2,495 MiB. The 0.5 s is inside the noise; the 1.5% on the median step and the 20 MiB are steadier (every P2 run of that session peaked at 2,495 MiB). We did not measure where the 20 MiB comes from.

### Ablation

Each row is measured against the base of its own session. Base time ranged from 77 to 90 s between sessions, so compare s and m, not seconds.

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
| v9b | M5 off (previous version) | 86.8 | 67.9 | 2,515 | 1.278 | 5.704 |
| v9b, second run | same code, slower session (version 17) | 90.3 | 72.6 | 2,515 | 1.244 | 5.704 |
| **v10b** | **+ P2 (submitted)** | **89.8** | **71.6** | **2,495** | **1.254** | **5.750** |

From v4 on, rows are `--repeats 2` runs.

### Tried and dropped

- **P1, precompiling the Triton kernels in `apply()`** with a thread pool joined before `apply()` returns. Step 1 fell from 3.6 to 1.0 s, but `apply()` took 2.8 s longer, because Triton's Python compile frontend is serial. No net gain, dropped.
- **P3, merged LoRA** (`USE_LORA_MERGED`). About 0.15 s per run [estimate], below our bar of 5 ms per step. Dropped.
- **Per-row causal SDPA.** Attention GPU time fell by about 40%, but about 1,700 extra small CPU-side operations per step ate the gain: 63.17 s with it, 62.93 s without.
- **S4 off on the first layers.** No speed gain (70.65 to 72.39 s against 71.95 s with S4 everywhere) and 340 MiB more peak with 7 layers off.
- **The v6 leak.** v6 ran out of memory at step 22: D's pack hook kept A1's inner SDPA graph alive. v6.1 runs A1's inner calls under private hooks; `test_v6_no_leak.py` checks flat memory over 8 steps.

### Limitations

- **Memory left on the table.** The fp16 head is off by team decision. In one session it cost no time (65.08 and 65.64 s against 65.43 and 66.69 s for v8b) and saved 440 MiB (2,453 against 2,893). The task forbids changing the model (section 8) and suggests repacking weights (section 9); we did not bet admission on the reading.
- **C2 is the example's math.** The jury may count `soup_swiglu_kernel` as the example. Without any SwiGLU kernel of ours the share was 9.8% at v7, so the 5% threshold holds either way.
- **Noise and estimates.** The two full v9b runs gave s = 1.278 and 1.244, so session noise in s is larger than P2's gain, which comes from one pair of runs. In the final run our repeats differ by 0.41 s and the base repeats by 1.89 s. Only sums are measured: not the split among B, A1, A2, C and D, between M2 and M3, or among P2's three flags.
- **Memory bought with time.** S4, A1 and M2 redo work in backward. S4 alone cost about 5.4 s (v4 against v5f, different sessions); v9 (M2, M3, M5) cost 2.2 s against v8b in one session.
- **Time that is not ours.** The GPU idles about 100 to 180 ms per step in Soup and TRL host code (metric logging, rewind, NaN checks, optimizer). Step 1 still compiles our kernels: 3.5 s against 1.8 s for the base at v8b.
- **Numerics.** K1, S4, A2, C, D, C1, C2, M2 and M3 are bit-exact by design (`torch.equal` tests). P2 is bit-exact against the torch ops it replaces (`test_ew_fusion.py` compares the epilogue with peft's ops, B fused with unfused, and the M2 LoRA pair with peft's path) and is built to give v9b's bits. `test_v10_fwd_bits_match_v9b` and `test_v10_bwd_bits_match_v9b` compare the v10 K3 entries with v9b's fp32 row-sum order; the full suite, these two included, passed (578 passed, 0 failed). B gives the values of the unfused K3 path, and K3 is not equal to the base. A1's SDPA backward may not be deterministic on a T4, causal SDPA may move real rows by fp16 ulps, and K3 reorders the norm's fp32 sum by a few ulps. Eval loss stayed within 0.003% of the base in every full run, v10b included.
- **Scope.** Built and measured for Soup's T4 fp16 path and Qwen2 only; where Soup picks bf16, K1, K4, S4 and B fall back to the original code. Other GPUs and models are untested.
