"""Team Delta's solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` only installs guarded patches: it does not
touch CUDA, compile kernels or change any setting.

- K1, `solution/kernels/nf4.py`: our NF4 dequantization kernel replaces bitsandbytes' dequant on
  its dequant+linear path, in the forward pass and in the backward re-dequant (bit-exact).
- K4, `solution/mlp.py`: the Qwen2 MLP runs on fp16 activations (bitsandbytes computes in fp16
  anyway), with the starter kit's example SwiGLU kernel in between. S4,
  `solution/mlp_recompute.py` (flag below): the MLP saves only its fp16 input and recomputes gate
  and up in backward.
- K3, `solution/kernels/addnorm.py` + `solution/layers.py`: Qwen2RMSNorm and the first residual
  add fused with post_attention_layernorm in our two kernels on the fp32 residual stream.
- U, `solution/unpad.py` (flag below): inside each batch, token-wise ops (norms, residual adds,
  MLP) run on the real tokens only; the attention side keeps the original [B, T, C] shapes, so
  LoRA dropout masks and the RNG order are the same as in base. Rows of different batches are
  never mixed.
- K2, `solution/kernels/head_ce.py` + `solution/head_loss.py`: the LM head and the cross-entropy
  run in vocabulary tiles without the [B, T, V] logits; TRL's metrics and Soup's rewind log read
  the same numbers from a soup_cli ChunkedStep.
The example's own class patch (`solution/swiglu.py: patch_qwen2_mlp`) stays off: K4 calls the
example kernel directly.
"""

from . import mlp_recompute, unpad
from .head_loss import patch_head_loss
from .kernels.nf4 import patch_bnb_matmul_4bit
from .layers import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm
from .mlp import patch_qwen2_mlp_fp16

USE_ADDNORM = True        # K3
USE_UNPAD = True          # U
USE_MLP_RECOMPUTE = False  # S4: about -1.7 GiB peak for about +95-145 ms per step
USE_HEAD_CE = True        # K2


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
    if USE_UNPAD:                  # U, after K3 so it can use K3's fused add+norm
        unpad.patch_qwen2_model_unpad()
    if USE_HEAD_CE:                # K2
        patch_head_loss()
