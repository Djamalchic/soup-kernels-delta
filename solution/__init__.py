"""Team Delta's solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` only installs guarded patches: it does not
touch CUDA, compile kernels or change any setting.

- K1, `solution/kernels/nf4.py`: our NF4 dequantization kernel replaces bitsandbytes' dequant on
  its dequant+linear path, in the forward pass and in the backward re-dequant (bit-exact).
- K3, `solution/kernels/addnorm.py` + `solution/layers.py`: Qwen2RMSNorm (input_layernorm and the
  final model.norm) and the first residual add fused with post_attention_layernorm run in our two
  kernels soup_addnorm_fwd_kernel / soup_addnorm_bwd_kernel on the fp32 residual stream.
- K2, `solution/kernels/head_ce.py` + `solution/head_loss.py`: the LM head and the cross-entropy
  run in vocabulary tiles without the [B, T, V] logits and without a persistent fp16 copy of the
  head, in one kernel `soup_head_ce_kernel` (weight-tile cast, softmax statistics, logits
  gradient); TRL's metrics and Soup's rewind log read the same numbers from a soup_cli ChunkedStep.
- K4, `solution/mlp.py`: the Qwen2 MLP runs on fp16 activations (bitsandbytes computes in fp16
  anyway), with the starter kit's example SwiGLU kernel in between; gate and up are kept in fp16
  for backward instead of fp32 (memory).
The example's own class patch (`solution/swiglu.py: patch_qwen2_mlp`) stays off: K4 calls the
example kernel directly.
"""

from .head_loss import patch_head_loss
from .kernels.nf4 import patch_bnb_matmul_4bit
from .layers import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm
from .mlp import patch_qwen2_mlp_fp16

USE_EXAMPLE_SWIGLU_PATCH = False  # the example's fp32 class patch; superseded by K4
USE_ADDNORM = True  # K3; False leaves the base Qwen2RMSNorm and Qwen2DecoderLayer untouched
USE_HEAD_CE = True  # K2; False keeps the base LM head and loss


def apply():
    patch_bnb_matmul_4bit()        # K1
    patch_qwen2_mlp_fp16()         # K4
    if USE_ADDNORM:                # K3: Qwen2RMSNorm first, then the decoder layer that fuses
        patch_qwen2_rmsnorm()      #     the first residual add with post_attention_layernorm
        patch_qwen2_decoder_layer()
    if USE_HEAD_CE:
        patch_head_loss()          # K2: Qwen2ForCausalLM.forward + trl.SFTTrainer.compute_loss
    if USE_EXAMPLE_SWIGLU_PATCH:
        from .swiglu import patch_qwen2_mlp

        patch_qwen2_mlp()
