"""K4: the Qwen2 MLP on fp16 activations.

Why this is exact: on a T4 Soup builds every 4-bit linear with bnb_4bit_compute_dtype=float16
(soup_cli/utils/gpu.py get_compute_dtype), and bitsandbytes' Linear4bit casts its input to that
dtype before the matmul and casts the output back to the input dtype afterwards. In base Soup the
residual stream is fp32, so gate_proj and up_proj compute in fp16 and hand back fp32 copies of fp16
values, and SiLU(gate) * up runs in fp32 only to be cast to fp16 again by down_proj.

Feeding the MLP an fp16 input produces the same fp16 matmul results, but autograd now keeps gate
and up in fp16 instead of three fp32 tensors (gate, SiLU(gate), up) of shape [tokens, 8960]:
about 140 MiB less per layer at 2048 tokens. SiLU(gate) * up runs in the starter kit's example
SwiGLU kernel (fp32 math inside, fp16 in and out), which rounds to fp16 exactly where down_proj
rounded before. The example kernel does not count as our own kernel; this patch is about memory.

The patch is guarded: any module or input that is not the expected shape (real bitsandbytes
Linear4bit layers with an fp16 compute dtype, a CUDA input, SiLU activation) runs the original
forward unchanged.
"""

import torch

_PATCH_MARK = "_soup_delta_fp16_mlp"


def _fp16_path_ok(module, x):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return False
    if x.dtype not in (torch.float32, torch.float16) or x.shape[-1] != getattr(module, "hidden_size", -1):
        return False
    act = getattr(getattr(module, "config", None), "hidden_act", None)
    if act != "silu":
        return False
    try:
        from bitsandbytes.nn import Linear4bit
    except Exception:  # noqa: BLE001 - without bitsandbytes there is nothing to do
        return False
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(module, name, None)
        if type(layer) is not Linear4bit:  # exact type: LoRA wrappers or other layers keep the original path
            return False
        if getattr(layer, "compute_dtype", None) is not torch.float16:
            return False
    return True


def patch_qwen2_mlp_fp16():
    """Send Qwen2MLP.forward through the fp16 path. Idempotent; no CUDA work at call time."""
    from transformers.models.qwen2 import modeling_qwen2

    from .swiglu import swiglu

    current = modeling_qwen2.Qwen2MLP.forward
    if getattr(current, _PATCH_MARK, False):
        return
    original = current

    def forward(self, x):
        if not _fp16_path_ok(self, x):
            return original(self, x)
        x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
        hidden = swiglu(self.gate_proj(x16), self.up_proj(x16))
        out = self.down_proj(hidden)
        return out if out.dtype == x.dtype else out.to(x.dtype)

    setattr(forward, _PATCH_MARK, True)
    forward.__wrapped__ = original
    modeling_qwen2.Qwen2MLP.forward = forward
