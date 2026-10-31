# Team Delta: Soup Kernel Challenge (Decentrathon 6.0)

- Public Kaggle notebook (saved version with results): [TODO final link]
- Repository: https://github.com/BAITC-Hacks/hack-8cddb4ae-delta, final commit [TODO final commit]
- Submitted code: v10d, commit `6b56d31` (branch `v10d-candidate`) = v10c (`0b9e562`) plus the Triton compile overlap. The final commit only adds this README and `SUBMISSION.md`.

**Result.** On a Kaggle T4, in the jury's order (base, ours, ours, base), `soup train` takes 64.8 s with our solution against 84.9 s for base Soup: s = 1.310. The GPU peak is 2,335 MiB against 14,345 MiB: m = 6.144. Eval loss is 2.04963 against 2.04964. 10 of 10 checks passed. The score s varies between sessions (part 3).

## 1. What we did and why

### Where the time and memory go

The task's profile of a base step (Qwen2.5-1.5B in NF4, LoRA r=16 on q and v, batch 4, up to 512 tokens, fp16 on a T4): matmul 36%, small elementwise ops and casts 44%, vocabulary softmax 7%, attention 5%, NF4 dequantization 5%.

The base peak of 14,345 MiB has four big parts, removed in this order:

1. **fp32 MLP activations** (three fp32 [tokens, 8960] tensors per layer): K4, 3,962 MiB less.
2. **Logits over 151,936 tokens**: K2 with K3, 4,538 MiB less.
3. **Padding**: U with a faster K1, 820 MiB and 11.3 s less.
4. **Per-layer tensors kept for backward**: S4, then B, A1, A2, C, D, then M2 and M3.

The time savings came from K1, K2, K3, U, C1, P2 and the compile overlap (P1). No Soup mode is on (`gradient_checkpointing`, `activation_offloading`, `fast_lora`, `use_cut_ce`, `packing`); all recomputation is our own autograd code. Components up to v6.1 fall back to the original code if a guard fails.

### Our Triton kernels

**K1, NF4 dequantization** (`solution/kernels/nf4.py`). A guarded patch of `bitsandbytes.matmul_4bit` swaps in our dequantization, forward and backward. Each thread reads 4 packed bytes at once and decodes NF4 with a tree of 15 selects. Bit-exact; a [8960, 1536] weight takes 0.166 ms against 0.300 ms. I3 keeps a permanent fp16 copy of the small k and v weights (about 42 MiB).

**K2, LM head + cross entropy without logits** (`solution/kernels/head_ce.py`). cuBLAS computes the head GEMM per vocabulary tile. Our kernel folds each tile into per-row online softmax statistics and, in backward, turns the recomputed tile into the logits gradient in place. The [B, T, V] logits never exist. TRL's metrics and Soup's rewind log read the same numbers through `ChunkedStep`.

**K3, fused residual add + RMSNorm** (`solution/kernels/addnorm.py`). One program per row adds the fp16 attention output to the fp32 residual and applies the norm; the same kernel serves every `Qwen2RMSNorm`. Backward also adds the incoming residual gradient.

**C2, SwiGLU** (`solution/kernels/swiglu.py`). One jit, `soup_swiglu_kernel`, replaces the starter kit's two example kernels (forward, backward, in-place backward) with the same math.

**C1, one launcher.** K1, K2, K3 (through `soup_*_usig` entries that inline the original bodies) and C2 share one twelve-argument signature, so Triton builds one C launcher. With C2 this cut step 1 from 5.1 s to 3.6 s in one session.

### PyTorch-level changes

**K4, fp16 MLP** (`solution/mlp.py`). Soup runs 4-bit layers in fp16 on a T4, so an fp16 MLP input gives the same matmul results and keeps gate and up in fp16.

**U, unpadding** (`solution/unpad.py`). Norms, residual adds and the MLP see real tokens only. Attention keeps the base shape, so dropout masks and the RNG stream match the base.

**S4 + C, MLP recompute** (`solution/mlp_recompute.py`). Only the fp16 MLP input is saved. Backward recomputes gate and up with one dequantization per call (I2), and the SwiGLU backward overwrites its own inputs (C). Bit-identical to K4.

**B, fused block** (`solution/fused_block.py`). One autograd node covers both residual adds, the post-attention norm and the S4 MLP. It saves the fp32 residual, the fp16 attention output and rstd, and rebuilds the rest with K3.

**A1, attention recompute** (`solution/sdpa_recompute.py`). Our `"sdpa"` entry saves fp16 q and the un-repeated k and v and reruns SDPA in backward. On the packed path it uses `is_causal=True` and no bool mask (I1); right padding never reaches real rows.

**A2, dropout replay** (`solution/lora_replay.py`). The LoRA dropout saves the 16-byte CUDA generator state instead of its mask and replays the mask in backward. **D** (`solution/saved_compact.py`) saves the fp16 LoRA-A inputs on real rows only.

**M2 and M3** (v9b; `solution/lora_recompute.py`, `solution/sdpa_recompute.py`). M2 rebuilds the fp16 LoRA-A inputs in backward from the fp32 residual that K3 and B keep. M3 saves A1's q, k and v on real rows only; padded rows get zero gradient or are masked. Together they cut the peak from 2,893 MiB (v8b) to 2,515 MiB.

### New in v10b: P2, elementwise fusion

`USE_ELEMENTWISE_FUSION` (off gives v9b) moves small casts, copies and adds that ran as separate torch launches into launches we already make. K3 gets two entries with the same twelve-slot signature, `soup_addnorm_fwd_v10` and `soup_addnorm_bwd_v10`, so a run still builds five cubins and one launcher. Three sub-flags, all on:

- `EW_K3_FP16`: K3 also writes the fp16 copies around B's GEMMs, forward and backward.
- `EW_LORA_EPILOGUE`: the q and v LoRA epilogue (fp32 cast, scaling, add) and its backward run inside K3, when the scaling is exact in fp16 (here 2.0).
- `EW_LORA_NO_CLONE`: skips peft's defensive clone of the base output, an identity before an out-of-place add.

By our count of the code, P2 removes about 500 small torch launches per step and adds 112 Triton launches.

### New in v10c: fp16 head

`STORE_HEAD_FP16 = True` (`solution/head_loss.py`). On a T4, `prepare_model_for_kbit_training` widens the frozen fp16 tied embedding and LM head to fp32, so their 890 MiB hold only fp16-exact values. On the first fused step we check every element on the device (`fp16(w).float() == w`, a NaN refuses). Only if all pass do we store an fp16 copy on the same Parameter, so the tie and `requires_grad=False` stay. A forward hook widens the embedding output back to fp32, so everything downstream is bit-identical. K2 reads the fp16 head directly, as the base path casts it to fp16 under autocast anyway.

**Rule note.** The organizers confirmed to us (relayed by our team captain on 31 October) that storing fp32 weights in fp16 is allowed as storage repacking, when a round trip from fp32 to fp16 and back to fp32 preserves the values exactly and the weight tie and the computation stay the same.

### New in v10d: compile overlap

`USE_PRECOMPILE_OVERLAP = True` (`solution/precompile.py`). Every measured run starts with an empty Triton cache, so in v10c our five kernels compiled one after another inside step 1. Now one worker thread, started at the end of `apply()`, compiles them with `JITFunction.warmup` and the production specialization keys while Soup loads the model. The first `Qwen2ForCausalLM.forward`, or any earlier launch of ours, joins it (60 s timeout, then normal compilation). Nothing is launched on the GPU, no cache is shipped, and no separate process starts.

**Rule note.** The organizers confirmed to us (relayed by our team captain on 31 October) that such a thread is allowed when everything stays inside the measured run with an empty cache.

### Where the ideas come from

We wrote all kernel code ourselves; banned libraries were read for ideas only and are not imported.

- K1: bitsandbytes (NF4 table, absmax layout, rounding order, tile idea), https://github.com/bitsandbytes-foundation/bitsandbytes
- K2: Soup issue #1693 and `ChunkedStep` in `soup_cli/utils/stream_loss_chunk.py` (soup-cli 0.77.0); Liger-Kernel, https://github.com/linkedin/Liger-Kernel, and Apple Cut Cross-Entropy, https://github.com/apple/ml-cross-entropy (linear + cross entropy in chunks); online softmax, Milakov and Gimelshein 2018, https://arxiv.org/abs/1805.02867
- K3, B, C2, P2: Liger-Kernel; Unsloth, https://github.com/unslothai/unsloth; the starter kit's example (fused add + RMSNorm, fused SwiGLU, LoRA scale and add in one pass)
- U: ByteDance Effective Transformer, https://github.com/bytedance/effective_transformer; FlashAttention's varlen path
- S4, B, A1, M2: activation recomputation, Chen et al. 2016, https://arxiv.org/abs/1604.06174; Korthikanti et al. 2022, https://arxiv.org/abs/2205.05198; Dao et al. 2022, https://arxiv.org/abs/2205.14135
- A2, D, M2: `torch.utils.checkpoint` (`preserve_rng_state`), the Megatron-LM RNG tracker and PyTorch `saved_tensors_hooks`

ChatGPT (GPT-6-Astra) and Claude were used as coding assistants, as the rules allow.

## 2. How to run

1. Open the public notebook ([TODO final link], the starter kit's `templates/kaggle_notebook.ipynb`) and click Copy & Edit. GPU and Internet need a phone-verified Kaggle account.
2. Set Accelerator to **GPU T4 x2** and switch **Internet on**.
3. In the first code cell, check `REPO_URL = "https://github.com/BAITC-Hacks/hack-8cddb4ae-delta.git"` and `COMMIT = "[TODO final commit]"`.
4. Run > Restart & clear cell outputs. Add no GPU cells of your own: check 9 needs idle GPUs.
5. Run All. The notebook installs the template's pinned packages, clones the repository at `COMMIT` and runs:

```bash
python task/prepare_data.py
python -m pytest solution -q              # 600 passed
python harness/run_task.py --repeats 2    # jury order: base, ours, ours, base
python harness/check_rules.py             # 10 checks; check 10 runs Soup's 79 tests with our patches
python harness/score.py runs/result.json
```

The only edit to the template is `--repeats 2` in the `run_task.py` cell. Run All took about 12 minutes of notebook time for v10b, plus Kaggle queueing; later times are not in our notes. To compare variants, set `USE_PRECOMPILE_OVERLAP = False` in `solution/__init__.py` (v10c; with `STORE_HEAD_FP16 = False` as well, v10b; with `USE_ELEMENTWISE_FUSION = False` too, v9b), push a branch and put its commit in `COMMIT`.

## 3. Results and limitations

### Final run against the base

Kaggle T4, notebook version 22, `--repeats 2`, code `6b56d31`.

| | Base Soup 0.77.0 | Ours (v10d) | Ratio |
|---|---|---|---|
| Time of `soup train`, s (two repeats, mean) | 84.9 (86.24, 83.54) | 64.8 (65.43, 64.13) | s = 1.310 (wall 23.7% shorter) |
| Peak GPU memory, MiB | 14,345 | 2,335 | m = 6.144 (83.7% less) |
| Eval loss, public rows | 2.04964 | 2.04963 | −0.000% |
| Own kernels' share (check 4) | n/a | 11.3% | threshold 5% |

All 10 checks passed: check 3 lists `soup_addnorm_bwd_v10`, `soup_addnorm_fwd_v10`, `soup_head_ce_usig`, `soup_nf4_dequant_usig` and `soup_swiglu_kernel`; check 4 11.3%; check 6 0.00 to 0.01%; check 8 time x0.763, memory x0.163; check 9 clean; check 10 79 passed. `pytest solution -q`: 600 passed, 0 failed (600 in notebook version 21 too). The run's `triton_cache` holds one `__triton_launcher*.so` and 5 cubins.

Previous version, v10c (notebook version 20, code `0b9e562`): ours 65.56 and 67.02 s (66.3) against base 89.11 and 86.11 s (87.6), s = 1.322; peak 2,335 against 14,345 MiB, m = 6.144; all 10 checks and 578 tests passed.

### fp16 head: scouting and the final run

- **v8b code**, notebook version 13, ABBA order (v8b, head, head, v8b), base 78.16 s: wall 65.43 and 66.69 s without the head, 65.08 and 65.64 s with it, time unchanged within the noise; peak 2,893 against 2,453 MiB (440 MiB less).
- **v9 code** (commit `c6d756a`, M5 on), notebook version 14, one run each: peak 2,475 against 2,335 MiB (140 MiB less), wall 69.85 against 69.10 s.
- **v10c, final run**: 2,335 MiB against v10b's 2,495 MiB from another session, 160 MiB less. This drop, our evidence that the store ran, is well under the 445 MiB freed on paper. Our untested reading: on v8b the head set the peak, while in v10c the peak sits in backward transients.

### Ablation

Each row is against the base of its own session (77 to 90 s); compare s and m. Rows from v4 on are `--repeats 2` runs.

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
| v9b | M5 off | 86.8 | 67.9 | 2,515 | 1.278 | 5.704 |
| v9b, second run | same code (version 17) | 90.3 | 72.6 | 2,515 | 1.244 | 5.704 |
| v10b | + P2 | 89.8 | 71.6 | 2,495 | 1.254 | 5.750 |
| v10c | + fp16 head | 87.6 | 66.3 | 2,335 | 1.322 | 6.144 |
| **v10d** | **+ overlap (submitted)** | **84.9** | **64.8** | **2,335** | **1.310** | **6.144** |

P2 alone, in one slow session (notebook version 18, base 98.4 s, one run each), took v9b from 73.29 to 72.79 s (inside the noise), the median step from 0.725 to 0.714 s and the peak from 2,515 to 2,495 MiB.

The compile overlap alone, in one session (notebook version 21, ABBA order): v10c took 66.23 and 67.08 s, v10d 63.97 and 65.11 s (2.1 s less on the means; the repeats differ by 0.85 and 1.14 s). Step 1 fell from 4.0 to 1.1 s, `apply()` from 8.8 to 8.2 s, and the peak stayed at 2,335 MiB.

### Switched off or dropped

- **M5, chunked backward** (`USE_BWD_CHUNKING = False`): 40 MiB less peak for an estimated 0.5 s, and not bit-exact (chunked cuBLAS GEMMs differ in the last bits at some row counts).
- **P1, thread pool joined inside `apply()`.** Step 1 fell from 3.6 to 1.0 s, but `apply()` grew by 2.8 s and the wall time did not move (the Triton frontend and `apply()`'s imports hold the GIL). v10d ships the overlap variant (above).
- **P3, merged LoRA** (`USE_LORA_MERGED`): about 0.15 s per run [estimate], below our bar of 5 ms per step.
- **Per-row causal SDPA**: about 40% less attention GPU time, lost to about 1,700 extra CPU-side operations per step (63.17 s against 62.93 s without).
- **S4 off on the first 7 layers**: 340 MiB more peak, no speed gain.
- **The v6 leak.** v6 ran out of memory at step 22: D's pack hook kept A1's inner SDPA graph alive. v6.1 runs A1's inner calls under private hooks; `test_v6_no_leak.py` checks flat memory over 8 steps.

### Limitations

- **C2 is the example's math.** If the jury counts `soup_swiglu_kernel` as the example, our share without any SwiGLU kernel was 9.8% at v7, still above 5%.
- **Session variance.** The two full v9b runs gave s = 1.278 and 1.244, a spread larger than P2's gain or the head's time effect. So we do not credit the head with the rise from 1.254 (v10b) to 1.322. In the v10c run our repeats differed by 1.46 s and the base repeats by 3.00 s. Only sums are measured, not per-component splits.
- **Memory bought with time.** S4, A1 and M2 redo work in backward. S4 alone cost about 5.4 s (v4 against v5f, different sessions); v9 cost 2.2 s against v8b in one session.
- **GPU idle in Soup and TRL.** The GPU idles about 100 to 180 ms per step in their host code (logging, rewind, NaN checks, optimizer). Step 1 compiled our kernels (3.5 s against 1.8 s for the base at v8b); v10d overlaps that with model loading.
- **Numerics.** K1, S4, A2, C, D, C1, C2, M2 and M3 are bit-exact by design. P2 is bit-exact against the torch ops it replaces and is built to give v9b's bits; tests check both. K3 reorders the norm's fp32 sum by a few ulps, so it does not equal the base. A1's SDPA may move real rows by fp16 ulps, and its backward may be nondeterministic on a T4. Eval loss stayed within 0.003% of the base in every full run.
- **Scope.** Soup's T4 fp16 path and Qwen2 only. Under bf16, K1, K4, S4 and B fall back to the original code; the head stays fp32 unless every element is fp16-exact. Other GPUs and models are untested.
