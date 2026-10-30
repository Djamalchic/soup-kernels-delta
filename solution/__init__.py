"""Team Delta's solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` only installs guarded patches: it does not
touch CUDA, compile kernels or change any setting.

Stage v0-K1 (this commit): our NF4 dequantization kernel (`solution/kernels/nf4.py`) replaces
bitsandbytes' dequant on its dequant+linear path, in the forward pass and in the backward
re-dequant. Everything else runs as in base Soup. The example SwiGLU patch is kept in
`solution/swiglu.py` but switched off for this measurement, so the run changes one thing at a time.
"""

from .kernels.nf4 import patch_bnb_matmul_4bit

USE_EXAMPLE_SWIGLU = False  # the example: does not count as our kernel; measured +5.2 s on T4


def apply():
    patch_bnb_matmul_4bit()
    if USE_EXAMPLE_SWIGLU:
        from .swiglu import patch_qwen2_mlp

        patch_qwen2_mlp()
