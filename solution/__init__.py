"""Team Delta's solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` only installs guarded patches: it does not
touch CUDA, compile kernels or change any setting.

Stage v0 (in progress):
- K1, `solution/kernels/nf4.py`: our NF4 dequantization kernel replaces bitsandbytes' dequant on
  its dequant+linear path, in the forward pass and in the backward re-dequant (bit-exact).
- K4, `solution/mlp.py`: the Qwen2 MLP runs on fp16 activations (bitsandbytes computes in fp16
  anyway), with the starter kit's example SwiGLU kernel in between; gate and up are kept in fp16
  for backward instead of fp32 (memory).
The example's own class patch (`solution/swiglu.py: patch_qwen2_mlp`) stays off: K4 calls the
example kernel directly.
"""

from .kernels.nf4 import patch_bnb_matmul_4bit
from .mlp import patch_qwen2_mlp_fp16

USE_EXAMPLE_SWIGLU_PATCH = False  # the example's fp32 class patch; superseded by K4


def apply():
    patch_bnb_matmul_4bit()
    patch_qwen2_mlp_fp16()
    if USE_EXAMPLE_SWIGLU_PATCH:
        from .swiglu import patch_qwen2_mlp

        patch_qwen2_mlp()
