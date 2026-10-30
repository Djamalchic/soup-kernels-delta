"""Team Delta's solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` only installs guarded patches: it does not
touch CUDA, compile kernels or change any setting.

- K1, `solution/kernels/nf4.py`: our NF4 dequantization kernel replaces bitsandbytes' dequant on
  its dequant+linear path, in the forward pass and in the backward re-dequant (bit-exact).
- K4, `solution/mlp.py`: the Qwen2 MLP runs on fp16 activations (bitsandbytes computes in fp16
  anyway), with the starter kit's example SwiGLU kernel in between. S4,
  `solution/mlp_recompute.py` (flag below): the MLP saves only its fp16 input and recomputes gate
  and up in backward; the example's SwiGLU backward kernel then writes its two gradients over its
  own fresh inputs (same kernel, same bits, two fewer [tokens, 8960] buffers at that moment).
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

v7 I4, `solution/unpad.py` S4_OFF_FRACTION (flag below): the first quarter of the packed layers
skip B and S4 (K3 add_rmsnorm + K4 MLP keep gate/up), trading ~38 KiB per real token per layer of
the memory headroom for the gate/up recompute in backward. Bit-identical values.
v7 I2, `solution/mlp_recompute.py` S4_DEQUANT_REUSE: S4's backward dequantizes W_gate / W_up once
each and reuses them for the recompute and dx (was twice). Bit-identical values.
v7 I3, `solution/kernels/nf4.py` USE_K1_FAST_HOST / USE_DENSE_CACHE: K1's validation cache accepts
aliases of the validated weight, the launch skips a same-device context, and k_proj / v_proj keep
their fp16 dequant for the run (42 MiB). Same values.
The example's own class patch (`solution/swiglu.py: patch_qwen2_mlp`) stays off: K4 calls the
example kernel directly.
"""

from . import fused_block, head_loss, lora_replay, mlp_recompute, saved_compact
from . import sdpa_recompute, unpad
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
USE_LORA_DROPOUT_REPLAY = True  # v6 A2
USE_LORA_ROW_COMPACT = True  # v6 D (needs U)
# v6 A: fallback for A1 only. Taken only when USE_SDPA_RECOMPUTE is False (see the docstring).
USE_SDPA_KV_COMPACT = False
# K2 part 3: -445 MiB static. Off: "do not change the model" is an open rule question (PLAN Q2).
STORE_HEAD_FP16 = False
# v7 I4 (needs U): the first int(num_hidden_layers * fraction) packed layers run K3 + K4 without
# S4's recompute (28 layers -> 7; 2-3-layer test models -> 0). Same values, ~+343 MiB at the peak
# batch [estimate], less backward work. Kaggle tuning: peak > 3 280 MiB -> 0.215 (6 layers),
# peak < 3 150 MiB -> 0.29 (8 layers). 0.0 is exactly v6.1.
S4_OFF_FRACTION = 0.25

# v7 I3 (solution/kernels/nf4.py). FAST_HOST: K1's validation cache also hits for D's / S4's
# aliases of a validated weight, and the launch skips a same-device torch.cuda.device context.
# DENSE_CACHE: k_proj / v_proj keep their fp16 dequant for the run (2 x 28 x 0.75 MiB = 42 MiB on
# Qwen2.5-1.5B), capped at DENSE_CACHE_MIB; DENSE_CACHE_QO adds q/o (+252 MiB, off).
USE_K1_FAST_HOST = True
USE_DENSE_CACHE = True
DENSE_CACHE_QO = False
DENSE_CACHE_MIB = 64


def apply():
    from .kernels import nf4 as _nf4  # already imported above (patch_bnb_matmul_4bit)

    _nf4.USE_K1_FAST_HOST = USE_K1_FAST_HOST  # I3
    _nf4.USE_DENSE_CACHE = USE_DENSE_CACHE
    _nf4.DENSE_CACHE_QO = DENSE_CACHE_QO
    _nf4.DENSE_CACHE_MIB = DENSE_CACHE_MIB
    patch_bnb_matmul_4bit()        # K1
    patch_qwen2_mlp_fp16()         # K4
    mlp_recompute.USE_MLP_RECOMPUTE = USE_MLP_RECOMPUTE
    mlp_recompute.S4_DEQUANT_REUSE = S4_DEQUANT_REUSE  # v7 I2
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
    if not a1:
        sdpa_recompute.USE_SDPA_RECOMPUTE = False
        if USE_UNPAD and USE_SDPA_KV_COMPACT:  # A runs only inside U's packed-layer scope
            saved_compact.USE_SDPA_KV_COMPACT = saved_compact.patch_sdpa_kv_compact()
    lora_replay.USE_LORA_DROPOUT_REPLAY = USE_LORA_DROPOUT_REPLAY
    if USE_LORA_DROPOUT_REPLAY:    # A2
        lora_replay.patch_lora_dropout_replay()
    if USE_HEAD_CE:                # K2
        head_loss.STORE_HEAD_FP16 = STORE_HEAD_FP16
        patch_head_loss()
