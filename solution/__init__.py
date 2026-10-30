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

v6 (memory at the step's peak, every one bit-identical to v5 or to base, every one guarded with
a fallback to the v5 ops):
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
- K2 part 3, `solution/head_loss.py` STORE_HEAD_FP16 (off): lossless fp16 storage of the frozen
  tied embedding / lm_head after an element-by-element check on the device.
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
USE_HEAD_CE = True        # K2
USE_FUSED_BLOCK = True    # v6 B (needs K3, U and S4)
USE_SDPA_RECOMPUTE = True  # v6 A1
USE_LORA_DROPOUT_REPLAY = True  # v6 A2
USE_LORA_ROW_COMPACT = True  # v6 D (needs U)
# v6 A: fallback for A1 only. Taken only when USE_SDPA_RECOMPUTE is False (see the docstring).
USE_SDPA_KV_COMPACT = False
# K2 part 3: -445 MiB static. Off: "do not change the model" is an open rule question (PLAN Q2).
STORE_HEAD_FP16 = False


def apply():
    patch_bnb_matmul_4bit()        # K1
    patch_qwen2_mlp_fp16()         # K4
    mlp_recompute.USE_MLP_RECOMPUTE = USE_MLP_RECOMPUTE
    if USE_MLP_RECOMPUTE:
        mlp_recompute.patch_qwen2_mlp_recompute()  # S4, on top of K4
    if USE_ADDNORM:                # K3
        patch_qwen2_rmsnorm()
        patch_qwen2_decoder_layer()
    unpad.USE_UNPAD = USE_UNPAD
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
