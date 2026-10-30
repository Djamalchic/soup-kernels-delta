"""S4: the Qwen2 MLP keeps only its fp16 input for backward and recomputes gate/up there.

K4 (`solution/mlp.py`) already runs the MLP on fp16 activations; autograd then keeps the fp16
gate and up outputs for the SwiGLU backward: 2 x [tokens, 8960] fp16 = 70 MiB per layer at 2048
tokens. Nothing else in the MLP saves activations: bitsandbytes' MatMul4Bit saves only the packed
weight (`bitsandbytes/autograd/_functions.py:357-360`, `ctx.tensors = (None, B)`), and so does
K1's `SoupNF4Matmul` (`solution/kernels/nf4.py:224`).

`SoupMLPRecompute` wraps the whole MLP in one autograd node that saves only x16 (6 MiB per layer at
2048 tokens) and, in backward, replays exactly the ops autograd would have run for K4:

  1. dh  = grad_out @ dequant(W_down)                  (MatMul4Bit.backward, `_functions.py:384`)
  2. g, u = gate_proj(x16), up_proj(x16)               (the same Linear4bit calls as the forward,
                                                        so bnb.matmul_4bit -> K1 when K1 is on)
  3. dg, du = SwiGLU backward(dh, g, u)                (the example's own backward, see below)
  4. dx16 = dg @ dequant(W_gate) + du @ dequant(W_up)  (two MatMul4Bit.backward results, summed in
                                                        fp16 the way autograd sums two gradients
                                                        flowing into one tensor)

With the default settings every one of these ops gets the same inputs as in K4, so outputs and
input gradients are bit-identical to K4 (the tests check `torch.equal`).

SwiGLU backward. K4's autograd runs the starter kit's example backward kernel. Calling that same
backward here (`SWIGLU_BACKWARD = "example"`, default) keeps the result bit-exact and costs one
fused pass (~10 bytes per element). Nothing of the example is copied or rewritten: we call its
`SwiGLU.backward` with the recomputed tensors, like K4's autograd does. `SWIGLU_BACKWARD = "torch"`
is the same formula in plain torch fp32 math, op for op in the example's order; it is ~10x more
memory traffic (~110 bytes per element, about +7 ms per layer at 2048 tokens on a T4 [estimate])
and differs from the Triton kernel in the last fp32 bits (tl.sigmoid, FMA contraction), so a few
fp16 results move by one ulp. It is also the automatic fallback if the example's API changes.

Idea credit: activation recomputation, Chen et al. 2016, "Training Deep Nets with Sublinear Memory
Cost" (arXiv:1604.06174). Code is our own.

The patch is guarded: unless USE_MLP_RECOMPUTE is on, grad is being recorded, x requires grad, K4's
own guard passes (CUDA input, exact bnb Linear4bit layers with fp16 compute dtype, SiLU), the
layers have no bias, frozen weights, a quant state and no module hooks, the forward is the K4 path
(which itself falls back to the original Qwen2MLP.forward). Importing this module does no CUDA
work and does not import Triton.
"""

import torch

from . import mlp as _k4

# Off by default; apply() (or the team) flips it after a Kaggle run confirms the trade.
USE_MLP_RECOMPUTE = False
# True: recompute gate and up (saves 64 MiB per layer at 2048 tokens).
# False: keep the fp16 gate output, recompute only up (saves 29 MiB per layer, half the extra GEMMs).
RECOMPUTE_GATE = True
# "example": the starter kit's SwiGLU backward (bit-exact with K4); "torch": plain torch fp32 math.
SWIGLU_BACKWARD = "example"

_PATCH_MARK = "_soup_delta_mlp_recompute"
_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")

_swiglu_forward = None  # set by the patch: solution.swiglu.swiglu
_swiglu_example_backward = None  # set by the patch: solution.swiglu.SwiGLU.backward, if present
_nf4_module = None  # solution.kernels.nf4, imported lazily on the first backward


class _SavedPair:
    """The one attribute the example's SwiGLU.backward reads from its ctx (`swiglu.py:54`)."""

    __slots__ = ("saved_tensors",)

    def __init__(self, gate, up):
        self.saved_tensors = (gate, up)


def _swiglu_backward_torch(dh, gate, up):
    """dgate, dup of silu(gate) * up in fp32 math, rounded to the input dtype once at the end.

    Same expressions and multiplication order as the example kernel (`swiglu.py:34-36`):
      dup   = (grad * gate) * sig
      dgate = ((grad * up) * sig) * (1 + gate * (1 - sig))
    Mixed fp32 x fp16 ops promote to fp32, so the fp16 operands enter exactly. At most three fp32
    [tokens, intermediate] buffers are alive at once.
    """
    out_dtype = gate.dtype
    d = dh.float()
    sig = torch.sigmoid(gate.float())
    du = torch.mul(d, gate)
    du.mul_(sig)
    du_out = du.to(out_dtype)
    del du
    t = torch.rsub(sig, 1.0)  # 1 - sig
    t.mul_(gate).add_(1.0)  # 1 + gate * (1 - sig)
    d.mul_(up).mul_(sig).mul_(t)
    del sig, t
    return d.to(out_dtype), du_out


def _swiglu_backward(dh, gate, up):
    if SWIGLU_BACKWARD == "example" and _swiglu_example_backward is not None:
        try:
            # Mirrors the example's forward, which saves gate.contiguous() and
            # up.to(gate.dtype).contiguous() (`swiglu.py:45-50`); both are no-ops here.
            return _swiglu_example_backward(
                _SavedPair(gate.contiguous(), up.to(gate.dtype).contiguous()), dh
            )
        except (AttributeError, TypeError, ValueError):
            pass  # the example's API changed: same math in torch below
    return _swiglu_backward_torch(dh, gate, up)


def _active_nf4():
    """K1's module when K1's bnb.matmul_4bit replacement is installed, else None."""
    global _nf4_module
    try:
        import bitsandbytes as bnb

        if _nf4_module is None:
            from .kernels import nf4 as module

            _nf4_module = module
    except Exception:  # noqa: BLE001 - no K1 in this tree: bitsandbytes' dequant is used
        return None
    if getattr(bnb, "matmul_4bit", None) is getattr(_nf4_module, "soup_matmul_4bit", object()):
        return _nf4_module
    return None


def _dequantize_weight(layer, dtype):
    """dequantize(W).to(dtype) as bitsandbytes' MatMul4Bit.backward builds it (`_functions.py:384`).

    With K1 installed, K1's NF4 kernel produces the same fp16 bits as bnb's dequantize_4bit (K1's
    tests and the Kaggle micro-benchmark check torch.equal), so it is used here as well.
    """
    weight = layer.weight
    quant_state = weight.quant_state
    packed = weight.view(-1, 1)  # canonical [(N*K+1)//2, 1] form, as in `_functions.py:321`
    nf4 = _active_nf4()
    if nf4 is not None:
        try:
            dense = nf4.nf4_dequantize(packed, quant_state)
        except (TypeError, ValueError):  # not nf4/blocksize 64/nested: K1 refuses, bnb handles it
            dense = None
        if dense is not None:
            return dense if dense.dtype == dtype else dense.to(dtype)
    from bitsandbytes import functional as bnb_functional

    return bnb_functional.dequantize_4bit(packed, quant_state).to(dtype)


class SoupMLPRecompute(torch.autograd.Function):
    """down_proj(silu(gate_proj(x16)) * up_proj(x16)) that saves only x16 (and gate, if asked)."""

    @staticmethod
    def forward(ctx, x16, gate_proj, up_proj, down_proj, recompute_gate):
        # Autograd records nothing inside forward, so g, u and h die with this frame.
        g = gate_proj(x16)
        u = up_proj(x16)
        h = _swiglu_forward(g, u)
        del u
        if recompute_gate:
            del g
            ctx.save_for_backward(x16)
        else:
            ctx.save_for_backward(x16, g)
            del g
        out = down_proj(h)
        del h
        ctx.layers = (gate_proj, up_proj, down_proj)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        saved = ctx.saved_tensors
        x16 = saved[0]
        gate_proj, up_proj, down_proj = ctx.layers

        # 1. down_proj input gradient, freeing the dense weight before the recompute.
        weight = _dequantize_weight(down_proj, grad_out.dtype)
        dh = torch.matmul(grad_out, weight)
        del weight

        # 2. Recompute through the layers: same calls, same inputs, same bits as the forward.
        g = saved[1] if len(saved) > 1 else gate_proj(x16)
        u = up_proj(x16)
        del saved

        # 3. SwiGLU backward.
        dg, du = _swiglu_backward(dh, g, u)
        del dh, g, u

        # 4. gate_proj and up_proj input gradients, summed in fp16 like autograd's accumulation of
        #    two gradients into x16 (a + b is commutative in IEEE arithmetic, so order is moot).
        weight = _dequantize_weight(gate_proj, dg.dtype)
        dx = torch.matmul(dg, weight)
        del weight, dg
        weight = _dequantize_weight(up_proj, du.dtype)
        dx.add_(torch.matmul(du, weight))
        return dx, None, None, None, None


def _recompute_ok(module, x):
    if not isinstance(x, torch.Tensor) or not torch.is_grad_enabled() or not x.requires_grad:
        return False
    if not _k4._fp16_path_ok(module, x):
        return False
    for name in _PROJ_NAMES:
        layer = getattr(module, name)
        if getattr(layer, "bias", None) is not None:
            return False
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.requires_grad:
            return False
        if getattr(weight, "quant_state", None) is None:
            return False
        # Recompute calls the layers a second time: hooks with side effects would fire twice.
        if getattr(layer, "_forward_hooks", None) or getattr(layer, "_forward_pre_hooks", None):
            return False
    return True


def patch_qwen2_mlp_recompute():
    """Wrap the K4 forward with the S4 path. Idempotent; no CUDA work at call time.

    Installs K4 first if needed, and marks the S4 forward with K4's mark too, so calling
    patch_qwen2_mlp_fp16() after this is a no-op instead of wrapping S4 inside K4.
    """
    global _swiglu_forward, _swiglu_example_backward
    from transformers.models.qwen2 import modeling_qwen2

    from . import swiglu as example

    _k4.patch_qwen2_mlp_fp16()
    current = modeling_qwen2.Qwen2MLP.forward
    if getattr(current, _PATCH_MARK, False):
        return
    previous = current  # the K4 forward

    _swiglu_forward = example.swiglu
    backward = getattr(getattr(example, "SwiGLU", None), "backward", None)
    _swiglu_example_backward = backward if callable(backward) else None

    def forward(self, x):
        if USE_MLP_RECOMPUTE and _recompute_ok(self, x):
            x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
            out = SoupMLPRecompute.apply(
                x16, self.gate_proj, self.up_proj, self.down_proj, bool(RECOMPUTE_GATE)
            )
            return out if out.dtype == x.dtype else out.to(x.dtype)
        return previous(self, x)

    setattr(forward, _PATCH_MARK, True)
    setattr(forward, _k4._PATCH_MARK, True)
    forward.__wrapped__ = previous
    modeling_qwen2.Qwen2MLP.forward = forward
