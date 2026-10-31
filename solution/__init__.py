"""Team Delta's solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` installs guarded patches and changes no
setting. It never creates the CUDA context; with v10d (USE_PRECOMPILE_OVERLAP, below) and a context
the harness already created, its last step starts one background thread that compiles our five
kernels while Soup loads the model, joined before the first forward (solution/precompile.py).

- K1, `solution/kernels/nf4.py`: our NF4 dequantization kernel replaces bitsandbytes' dequant on
  its dequant+linear path, in the forward pass and in the backward re-dequant (bit-exact).
- K4, `solution/mlp.py`: the Qwen2 MLP runs on fp16 activations (bitsandbytes computes in fp16
  anyway), with a SwiGLU kernel in between (ours since v8 C2, the starter kit's example with
  USE_SOUP_SWIGLU off). S4, `solution/mlp_recompute.py` (flag below): the MLP saves only its fp16
  input and recomputes gate and up in backward; the SwiGLU backward kernel then writes its two
  gradients over its own fresh inputs (same kernel, same bits, two fewer [tokens, 8960] buffers at
  that moment).
- K3, `solution/kernels/addnorm.py` + `solution/layers.py`: Qwen2RMSNorm and the first residual
  add fused with post_attention_layernorm in our two kernels on the fp32 residual stream (the
  attention output is read as fp16 and widened inside the kernel).
- U, `solution/unpad.py` (flag below): inside each batch, token-wise ops (norms, residual adds,
  MLP) run on the real tokens only; the attention side keeps the original [B, T, C] shapes, so
  LoRA dropout masks and the RNG order are the same as in base. Rows of different batches are
  never mixed.
- K2, `solution/kernels/head_ce.py` + `solution/head_loss.py`: the LM head and the cross-entropy
  run in vocabulary tiles without the [B, T, V] logits; TRL's metrics and Soup's rewind log read
  the same numbers from a soup_cli ChunkedStep.

v6 (memory at the step's peak, every one guarded with a fallback to the v5 ops; same values as v5
except K3's fp16 delta pointer, which changes the norm's fp32 summation order by a few ulp, and
A1's SDPA backward, which is only as deterministic as torch's):
- B, `solution/fused_block.py`: on U's packed path, add + norm + S4 MLP + residual add as one
  autograd node that saves (r, fp16 a) instead of (r_mid, x16).
- A1, `solution/sdpa_recompute.py`: SDPA keeps fp16 q and the un-repeated k, v and recomputes its
  forward in backward, instead of keeping q, 12-head k and v, the output, lse and the float mask.
- A2, `solution/lora_replay.py`: the q/v LoRA dropout keeps the generator state instead of its
  bool mask and replays the mask in backward.
- D, `solution/saved_compact.py`: inside a packed layer's attention, LoRA-A's fp16 inputs keep only
  the real rows (they are exact zeros on padding rows).
- A, `solution/saved_compact.py` (fallback for A1, off): SDPA keeps one head per GQA group of its
  repeated key/value. Never on together with A1: A writes ALL_ATTENTION_FUNCTIONS' local mapping,
  which transformers reads before the global entry A1 registers (utils/generic.py:1110-1114, via
  get_interface, modeling_utils.py:5161-5173), so A1 would be bypassed. apply() enforces it
  (A1 first; A is not even patched while A1 is wanted, and A refuses a non-genuine "sdpa").
- I1 (v7), `solution/sdpa_recompute.py` USE_CAUSAL_PADDED + `solution/unpad.py` _causal_scope: on
  U's packed path A1 runs SDPA as is_causal=True without the bool mask (block skipping, no bias).
- K2 part 3, `solution/head_loss.py` STORE_HEAD_FP16 (off): lossless fp16 storage of the frozen
  tied embedding / lm_head after an element-by-element check on the device.

v7 I4, `solution/unpad.py` S4_OFF_FRACTION (flag below): the first int(layers * fraction) packed
layers skip B and S4 (K3 add_rmsnorm + K4 MLP keep gate/up), trading ~38 KiB per real token per
layer of the memory headroom for the gate/up recompute in backward. Bit-identical values. v8 sets
the fraction to 0.0 (every packed layer keeps S4, as v6.1); the path stays for tuning.
v7 I2, `solution/mlp_recompute.py` S4_DEQUANT_REUSE: S4's backward dequantizes W_gate / W_up once
each and reuses them for the recompute and dx (was twice). Bit-identical values.
v7 I3, `solution/kernels/nf4.py` USE_K1_FAST_HOST / USE_DENSE_CACHE: K1's validation cache accepts
aliases of the validated weight, the launch skips a same-device context, and k_proj / v_proj keep
their fp16 dequant for the run (42 MiB). Same values.
v8 C1, `solution/kernels/__init__.py` USE_UNIFIED_SIG (flag below): every kernel we launch takes
the same twelve runtime arguments (7 pointers, 4 int32, 1 fp32, no constexpr parameter; modes and
offsets in do_not_specialize), so Triton gcc-builds one launcher per run instead of one per kernel.
K1 / K2 / K3 launch `soup_*_usig` entries whose body is the v7 kernel, inlined. Same values.
v8 C2, `solution/kernels/swiglu.py` USE_SOUP_SWIGLU (flag below): K4 and S4 run our own
soup_swiglu_kernel (forward and backward in one jit by a runtime mode, in place in S4) instead of
the starter kit's two example kernels. Same values.
v9 M2, `solution/lora_recompute.py` USE_LORA_INPUT_RECOMPUTE (flag below): on U's packed path the
q/v LoRA-A fp16 inputs are not kept; backward rebuilds them bit for bit from the residual r that
input_layernorm and B keep anyway (K3 norm replay, zero-fill + index_copy, dropout replay from the
saved RNG state, fp16 cast). Same values.
v9 M3, `solution/sdpa_recompute.py` USE_ROW_COMPACT + `solution/unpad.py` _row_scope (flag
below): A1 keeps q16 / k16 / v16 of the real rows only and scatters them back into zeros in its
backward recompute. Same gradients (pad query rows have dO = 0, pad keys are masked).
The example's own class patch (`solution/swiglu.py: patch_qwen2_mlp`) stays off; with
USE_SOUP_SWIGLU off, K4 and S4 call the example kernels directly (v7).
v9 M5, USE_BWD_CHUNKING (flag below): the step's peak is the layer-27 MLP backward (S4 / B "both":
dh, g, u, W_gate, W_up alive). `solution/mlp_recompute.py` _gate_up_input_grad_rows keeps one
dense weight at a time and recomputes u in BWD_ROW_CHUNKS row chunks (3U + 2W -> 2U + W + U/k),
and `solution/kernels/head_ce.py` VOCAB_TILE goes 16384 -> K2_VOCAB_TILE so K2's tiles stay under
the new peak. Forward values unchanged; the S4 input gradient is bit-identical to "both" when
cuBLAS's u GEMM gives the same rows for M = chunk as for M = N (tests/test_bwd_chunking.py checks
it on the T4); K2's hidden gradient sums 19 tile partials instead of 10 (equal within K2's tests).
v10 P2 (USE_ELEMENTWISE_FUSION, flags below): element-wise casts / copies / adds of our own paths
run inside launches we already make: B's four fp32 <-> fp16 casts inside K3 (`solution/kernels/
addnorm.py` *_v10 entries, `solution/fused_block.py`), and M2's q/v LoRA output (bnb's `.to(fp32)`,
peft's clone, `* scaling`, the add, and their backward) as one K3-v10 launch each way
(`solution/lora_recompute.py`). Same IEEE operations on the same values; still five cubins.
v10d (USE_PRECOMPILE_OVERLAP, flag below; organisers' answer of 31.10: allowed), `solution/
precompile.py`: at the end of apply() one thread compiles the five kernels the run launches
(JITFunction.warmup with their production argument kinds: same cache keys, same binaries, same
values) while Soup builds the model; the first Qwen2ForCausalLM.forward, and every launch helper of
ours, joins it first. The compile time stays inside the measured run, with the same empty cache.
v11 Q (USE_MERGED_QKV, flags below), `solution/merged_qkv.py`: on U's packed path q_proj / k_proj /
v_proj base projections run as ONE fp16 GEMM against [W_q; W_k; W_v] ([2048, 1536], rows from K1's
bit-exact dequant), split into q / k / v inside one autograd node whose backward is ONE dX GEMM;
the q / v LoRA halves are M2's code unchanged (same dropout draws, same order). Not bit-identical:
cuBLAS's N = 2048 kernel may round an output element to the neighbouring fp16 value, and dX is
summed in fp32 inside the GEMM and rounded once (tests/test_merged_qkv.py tolerances).
"""

from . import fused_block, head_loss, lora_recompute, lora_replay, merged_qkv, mlp, mlp_recompute
from . import precompile, saved_compact, sdpa_recompute, unpad
from .head_loss import patch_head_loss
from .kernels.nf4 import patch_bnb_matmul_4bit
from .layers import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm
from .mlp import patch_qwen2_mlp_fp16

USE_ADDNORM = True        # K3
USE_UNPAD = True          # U
USE_MLP_RECOMPUTE = True  # S4
# v7 I2: S4 backward dequantizes W_gate and W_up once each and reuses them for the recompute and
# dx ("both"; "up" keeps only W_up, "off" is v6.1). Bit-identical; +26 MiB at the MLP backward.
S4_DEQUANT_REUSE = "both"
USE_HEAD_CE = True        # K2
USE_FUSED_BLOCK = True    # v6 B (needs K3, U and S4)
USE_SDPA_RECOMPUTE = True  # v6 A1
# v7 I1: on U's packed path A1 calls SDPA with is_causal=True instead of the causal+padding
# bool mask (same attention on real rows; pad rows are dropped by U). Needs A1 and U.
USE_SDPA_CAUSAL_PADDED = True
# v9 M3: on U's packed path A1 saves q16 / k16 / v16 for the real rows only ([N, heads, 128]) and
# rebuilds them with zero pad rows in backward; every gradient leaving the layer is unchanged
# (proof in solution/sdpa_recompute.py). ~-76 MiB at the peak [estimate]. Needs A1 and U; works
# with and without I1. False = v8b.
USE_SDPA_ROW_COMPACT = True
USE_LORA_DROPOUT_REPLAY = True  # v6 A2
USE_LORA_ROW_COMPACT = True  # v6 D (needs U)
# v6 A: fallback for A1 only. Taken only when USE_SDPA_RECOMPUTE is False (see the docstring).
USE_SDPA_KV_COMPACT = False
# K2 part 3: -445 MiB static. Off: "do not change the model" is an open rule question (PLAN Q2).
STORE_HEAD_FP16 = True  # fp16 copy of the tied embedding/lm_head, exact (every value checked on the device); team decision 31.10
# v7 I4 (needs U): the first int(num_hidden_layers * fraction) packed layers run K3 + K4 without
# S4's recompute (28 layers -> 7; 2-3-layer test models -> 0). Same values, ~+343 MiB at the peak
# batch [estimate], less backward work. Kaggle tuning: peak > 3 280 MiB -> 0.215 (6 layers),
# peak < 3 150 MiB -> 0.29 (8 layers). 0.0 is exactly v6.1.
# v8 F1: 0.0. The Kaggle scouting run measured the same speed at 0.0 as at v7's 0.25 and a
# 2 893 MiB peak instead of 3 233 (ADR-002 cap: <= 3 000 MiB).
S4_OFF_FRACTION = 0.0

# v7 I3 (solution/kernels/nf4.py). FAST_HOST: K1's validation cache also hits for D's / S4's
# aliases of a validated weight, and the launch skips a same-device torch.cuda.device context.
# DENSE_CACHE: k_proj / v_proj keep their fp16 dequant for the run (2 x 28 x 0.75 MiB = 42 MiB on
# Qwen2.5-1.5B), capped at DENSE_CACHE_MIB; DENSE_CACHE_QO adds q/o (+252 MiB, off).
USE_K1_FAST_HOST = True
USE_DENSE_CACHE = True
DENSE_CACHE_QO = False
DENSE_CACHE_MIB = 64

# v8 C1 (solution/kernels/__init__.py): K1 / K2 / K3 launch their *_usig entries, which share one
# runtime signature with C2's kernel: one gcc-built Triton launcher per run instead of six (step 1).
# Same compiled bodies, same values. False = the v7 launches.
USE_UNIFIED_SIG = True
# v8 C2 (solution/kernels/swiglu.py): K4 and S4 run our soup_swiglu_kernel instead of the starter
# kit's two example kernels (one compile instead of two). Same values. False = v7.
USE_SOUP_SWIGLU = True

# v9 M5 (solution/mlp_recompute.py, solution/kernels/head_ce.py). Per step at the peak batch
# (N = 1350 real rows) [estimate]: layer-27 MLP backward transients 121.7 -> 78.9 MiB, K2's forward
# tiles 90 -> 45 MiB, backward 102 -> 57 MiB; the step's peak drops ~43 MiB (next limit: the
# layer-27 MLP forward, ~6 MiB below). Cost ~+0.3-0.5 s per run: one more W_gate dequant per
# chunked MLP backward, the u GEMM in pieces, 9 more K2 tiles. False = v8b exactly.
USE_BWD_CHUNKING = False  # M5 measured ~-30 MiB for ~+0.5 s: not worth it, and the only non-bit-exact change in v9
BWD_ROW_CHUNKS = 4         # u recompute pieces (384-row chunks at N = 1350)
BWD_CHUNK_MIN_ROWS = 512   # smaller batches keep "both" (not the peak; saves the extra dequant)
K2_VOCAB_TILE = 8192

# v9 M2 (solution/lora_recompute.py; needs U and K3): on U's packed path the LoRA-A fp16 inputs of
# q_proj / v_proj are not kept; backward rebuilds them bit for bit from the residual r that
# input_layernorm and B keep anyway (norm replay, zero-fill + index_copy, dropout replay from the
# saved RNG state, fp16 cast). -2 x N x 1536 x 2 B per layer: ~221 MiB at N = 1350 over 28 layers
# [estimate]. Same values. False = v8b.
USE_LORA_INPUT_RECOMPUTE = True

# v10 P2, element-wise casts / copies / adds fused into launches we already make. Master switch
# (False = v9b exactly) and three sub-flags. No new compiled kernel: with EW_K3_FP16 every K3
# launch goes to the two *_v10 entries of solution/kernels/addnorm.py instead of the v8 pair.
# Per step at P = 2048, N = 1350 [estimate]:
# - EW_K3_FP16 (fused_block.py B): K3 writes B's x16 itself (forward and backward recompute), reads
#   dx16 as its fp16 dy and writes da16 back in place: -4 torch launches per layer (-112 / step),
#   ~-58 MB traffic per layer (~-6 ms GPU / step), no fp32 y / dy buffers.
# - EW_LORA_EPILOGUE (lora_recompute.py M2, needs EW_K3_FP16): q / v LoRA output = f32(base16) +
#   f32(fp16(lora * s)) in one K3-v10 launch; backward casts + mul in one more: -7 torch, +2 Triton
#   launches per LoRA call (-392 torch / +112 Triton per step), ~-10 ms GPU / step. Only when the
#   LoRA scaling is fp16-exact (task: 2.0).
# - EW_LORA_NO_CLONE (M2): peft's defensive clone of the base output is skipped (identity before
#   an out-of-place add): -56 launches / ~-5 ms GPU per step; covered by the epilogue when it runs.
# Same bits by construction (the same IEEE ops); tests/test_ew_fusion.py checks them on the T4,
# including whether the v10 K3 entries keep v9b's rstd summation order (else: fp32 ulps vs v9b,
# consistent inside the run).
USE_ELEMENTWISE_FUSION = True
EW_K3_FP16 = True
EW_LORA_EPILOGUE = True
EW_LORA_NO_CLONE = True

# v10d (solution/precompile.py): the last step of apply() starts ONE background thread that runs
# JITFunction.warmup for the five kernels the run launches (K1, the K3 pair that _ew_k3() selects,
# K2, K4; C1 signatures) and prebuilds their shared gcc launcher, while Soup goes on to load and
# quantize the model. Joined by the first Qwen2ForCausalLM.forward (and, as a safety net, by every
# launch helper of ours) before any of our kernels launches; JOIN_TIMEOUT_S guards a hang. Same
# cache keys -> same binaries -> same values; nothing launched or allocated on the GPU; only when
# the CUDA context already exists (harness/_train.py:120) and Triton is real. Each warmup adds 1 to
# the harness's triton_launches (never timed). v10 P1 did the same compiles inside apply() (Kaggle
# T4: step_s[0] -2.68 s, apply_s +2.74 s, wall +-0); the gain here is only the part that overlaps
# Soup's model loading [estimate -1...-2.5 s of wall]. False = v10c.
USE_PRECOMPILE_OVERLAP = True

# v11 Q (solution/merged_qkv.py; needs U, K1 and M2, which supplies the LoRA halves): q/k/v base
# projections as one GEMM per layer and one dX GEMM in backward; one x -> fp16 cast instead of
# three; x's gradient sums 3 contributions instead of 5. Per step at P = 2048 [estimate]: ~-4 ms
# forward (two casts, the N = 256 GEMMs' poor T4 fill) and ~-11 ms backward (two fp16 -> fp32
# casts and two fp32 adds of [P, 1536], one GEMM instead of three), minus ~2 ms for the q / v
# copies and the gradient concat. False = v10d exactly.
# MERGED_QKV_CACHE: keep [W_q; W_k; W_v] for the run (I3's k / v entries become views of it):
# +126 MiB static [28 x 1536 x 1536 x 2 B] for ~-1.7 ms / step (no q dequant). Off: the weight is
# built per call (q by K1 as today, k / v copied from I3's cache), no static memory.
USE_MERGED_QKV = True
MERGED_QKV_CACHE = False
MERGED_QKV_CACHE_MIB = 192


def apply():
    """Install the patches; with USE_PRECOMPILE_OVERLAP, then start the background compile."""
    _apply_patches()
    if USE_PRECOMPILE_OVERLAP:
        _start_precompile()


def _ew_k3():
    """v10 P2's K3 switch, as _apply_patches() writes it into kernels/addnorm.py USE_EW_K3; the K3
    pair the background compile warms comes from here too."""
    return bool(USE_ELEMENTWISE_FUSION) and bool(EW_K3_FP16)


def _start_precompile():
    """Start the background compile and, when it runs, the forward-side join. Never raises."""
    try:
        started = precompile.start_overlap(addnorm=bool(USE_ADDNORM), head_ce=bool(USE_HEAD_CE),
                                           swiglu=bool(USE_SOUP_SWIGLU),
                                           unified=bool(USE_UNIFIED_SIG), ew_k3=_ew_k3())
        if started:
            precompile.patch_forward_join()
        return started
    except Exception:  # noqa: BLE001 - every kernel then compiles at its first launch, as in v10c
        return False


def _apply_patches():
    # nf4 and head_ce are imported above (K1 patch, head_loss); addnorm loads here. Importing a
    # kernel module only defines Triton functions: no CUDA work, no compile.
    from .kernels import addnorm as _k3, head_ce as _k2, nf4 as _nf4

    _nf4.USE_K1_FAST_HOST = USE_K1_FAST_HOST  # I3
    _nf4.USE_DENSE_CACHE = USE_DENSE_CACHE
    _nf4.DENSE_CACHE_QO = DENSE_CACHE_QO
    _nf4.DENSE_CACHE_MIB = DENSE_CACHE_MIB
    for _kernels in (_nf4, _k3, _k2):  # C1, read at every launch
        _kernels.USE_UNIFIED_SIG = bool(USE_UNIFIED_SIG)
    # v10 P2, read at every call. The epilogue launches the K3-v10 forward entry, so it is only on
    # together with EW_K3_FP16 (otherwise a run would compile both K3 forward entries).
    _ew = bool(USE_ELEMENTWISE_FUSION)
    _k3.USE_EW_K3 = _ew_k3()  # = _ew and EW_K3_FP16; the K3 pair precompile warms follows it
    lora_recompute.USE_LORA_EPILOGUE = _k3.USE_EW_K3 and bool(EW_LORA_EPILOGUE)
    lora_recompute.USE_LORA_NO_CLONE = _ew and bool(EW_LORA_NO_CLONE)
    mlp.USE_SOUP_SWIGLU = bool(USE_SOUP_SWIGLU)  # C2, read by K4 and S4 at every call
    patch_bnb_matmul_4bit()        # K1
    patch_qwen2_mlp_fp16()         # K4
    mlp_recompute.USE_MLP_RECOMPUTE = USE_MLP_RECOMPUTE
    mlp_recompute.S4_DEQUANT_REUSE = S4_DEQUANT_REUSE  # v7 I2
    # v9 M5: read at every S4 / B backward and every K2 call; off = v8b's values.
    mlp_recompute.BWD_ROW_CHUNKS = int(BWD_ROW_CHUNKS) if USE_BWD_CHUNKING else 1
    mlp_recompute.BWD_CHUNK_MIN_ROWS = int(BWD_CHUNK_MIN_ROWS)
    _k2.VOCAB_TILE = int(K2_VOCAB_TILE) if USE_BWD_CHUNKING else _k2.VOCAB_TILE_V8
    if USE_MLP_RECOMPUTE:
        mlp_recompute.patch_qwen2_mlp_recompute()  # S4, on top of K4
    if USE_ADDNORM:                # K3
        patch_qwen2_rmsnorm()
        patch_qwen2_decoder_layer()
    unpad.USE_UNPAD = USE_UNPAD
    unpad.S4_OFF_FRACTION = S4_OFF_FRACTION if USE_UNPAD else 0.0  # I4 lives in U's packed path
    fused_block.USE_FUSED_BLOCK = False
    saved_compact.USE_LORA_ROW_COMPACT = False
    saved_compact.USE_SDPA_KV_COMPACT = False
    if USE_UNPAD:                  # U, after K3 so it can use K3's fused add+norm
        unpad.patch_qwen2_model_unpad()
        fused_block.USE_FUSED_BLOCK = USE_FUSED_BLOCK and USE_ADDNORM and USE_MLP_RECOMPUTE  # B
        saved_compact.USE_LORA_ROW_COMPACT = USE_LORA_ROW_COMPACT  # D
    # A1 and A are exclusive; A1 first. A's patch is not even called while A1 is wanted.
    sdpa_recompute.USE_SDPA_RECOMPUTE = USE_SDPA_RECOMPUTE
    a1 = USE_SDPA_RECOMPUTE and sdpa_recompute.patch_sdpa_recompute()
    sdpa_recompute.USE_CAUSAL_PADDED = bool(USE_SDPA_CAUSAL_PADDED and a1 and USE_UNPAD)  # I1
    sdpa_recompute.USE_ROW_COMPACT = bool(USE_SDPA_ROW_COMPACT and a1 and USE_UNPAD)  # v9 M3
    if not a1:
        sdpa_recompute.USE_SDPA_RECOMPUTE = False
        if USE_UNPAD and USE_SDPA_KV_COMPACT:  # A runs only inside U's packed-layer scope
            saved_compact.USE_SDPA_KV_COMPACT = saved_compact.patch_sdpa_kv_compact()
    lora_replay.USE_LORA_DROPOUT_REPLAY = USE_LORA_DROPOUT_REPLAY
    if USE_LORA_DROPOUT_REPLAY:    # A2
        lora_replay.patch_lora_dropout_replay()
    # v9 M2 after A2: its class patch must be the outer one (it hands every other call to A2's).
    lora_recompute.USE_LORA_INPUT_RECOMPUTE = False
    if USE_LORA_INPUT_RECOMPUTE and USE_UNPAD and USE_ADDNORM:
        lora_recompute.USE_LORA_INPUT_RECOMPUTE = lora_recompute.patch_lora_input_recompute()
    # v11 Q: no class patch; unpad's packed layer calls merged_qkv.attention_forward while the flag
    # is on. install() only captures the genuine classes (no CUDA work); needs M2's LoRA half.
    merged_qkv.USE_MERGED_QKV = False
    merged_qkv.MERGED_QKV_CACHE = bool(MERGED_QKV_CACHE)
    merged_qkv.MERGED_CACHE_MIB = int(MERGED_QKV_CACHE_MIB)
    if USE_MERGED_QKV and USE_UNPAD and lora_recompute.USE_LORA_INPUT_RECOMPUTE:
        merged_qkv.USE_MERGED_QKV = bool(merged_qkv.install())
    if USE_HEAD_CE:                # K2
        head_loss.STORE_HEAD_FP16 = STORE_HEAD_FP16
        patch_head_loss()
