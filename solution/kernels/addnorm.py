"""K3: fused residual add + Qwen2 RMSNorm, forward and backward (two Triton kernels).

Reference math, transformers 5.16.1 models/qwen2/modeling_qwen2.py:
- Qwen2RMSNorm.forward (lines 247-252): u = f32(x); r = rsqrt(mean(u^2) + eps); y = w * cast_in(u * r)
- Qwen2DecoderLayer.forward (lines 291, 294-295): residual = residual + attn_out;
  post_attention_layernorm(residual)

Scope (checked by the callers in solution/layers.py; anything else stays on the original path):
the residual stream and the norm weights are fp32 (base Soup keeps the residual stream in fp32, see
solution/mlp.py docstring), so cast_in() is the identity and y = w * (u * r) in fp32, the base
output dtype promote(fp32, fp32). Under accelerate's fp16 autocast the attention output (delta)
arrives in fp16: bitsandbytes Linear4bit returns its input dtype (bitsandbytes/nn/modules.py:626,
637) and peft's LoRA adds an fp16 update to it (peft/tuners/lora/bnb.py:527-552). add_rmsnorm casts
the delta to fp32 in Python before the launch: exact, the same value base's type-promoting add
uses, and its gradient is rounded back to fp16 by autograd exactly like base AddBackward. So every
pointer is fp32 and each kernel compiles once (PLAN.md K3 budget: 2 specializations).

Kernels:
- soup_addnorm_fwd_kernel: optional add (runtime flag has_delta), stores r (fp32) only when it
  added, y, rstd.
- soup_addnorm_bwd_kernel: dx = rstd * (g - x * rstd^2 * mean(g * x)) with g = w * dy, plus the
  incoming residual gradient dr (runtime flag has_dr). No dW: the norm weights are frozen.
Saved for backward: x (or r) fp32, rstd fp32 [rows] and the weight (save_for_backward keeps a
reference to the parameter, no copy): the same activations base autograd keeps (input and rsqrt).

Idea credit: fused add + RMSNorm as in Liger-Kernel and Unsloth; code written from scratch from the
formula above. tl.rsqrt and the needs_input_grad gating follow the reviewed GPT-6-Astra K3 draft.
"""

import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 1536  # Qwen2.5-1.5B hidden size; the only width this file accepts
BLOCK = 2048  # next power of two >= HIDDEN_SIZE
NUM_WARPS = 4  # 128 threads x 16 elements per row; 8 is the alternative to time on T4


@triton.jit(do_not_specialize=["has_delta", "eps"])
def soup_addnorm_fwd_kernel(
    X, D, W, Y, R, RSTD,
    has_delta, eps,
    H: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)  # grid = (rows,), one row per program
    col = tl.arange(0, BLOCK)
    cmask = col < H
    offs = row * H + col
    add = cmask & (has_delta != 0)

    x = tl.load(X + offs, mask=cmask, other=0.0).to(tl.float32)
    d = tl.load(D + offs, mask=add, other=0.0).to(tl.float32)
    x = x + d  # residual + f32(delta): the same single fp32 add as base (line 291)
    tl.store(R + offs, x, mask=add)  # plain norm: nothing stored, R is a dummy pointer

    w = tl.load(W + col, mask=cmask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = tl.rsqrt(var + eps)  # torch.rsqrt in base (line 251)
    y = w * (x * rstd)  # base order: hs * rsqrt, then weight * hs (lines 251-252)
    tl.store(Y + offs, y, mask=cmask)
    tl.store(RSTD + row, rstd)


@triton.jit(do_not_specialize=["has_dr"])
def soup_addnorm_bwd_kernel(
    DY, DR, X, W, RSTD, DX,
    has_dr,
    H: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    cmask = col < H
    offs = row * H + col

    x = tl.load(X + offs, mask=cmask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=cmask, other=0.0).to(tl.float32)
    w = tl.load(W + col, mask=cmask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)

    # y_j = w_j x_j r, dr/dx_i = -r^3 x_i / H  =>  dx_i = r (g_i - x_i r^2 mean(g x)), g = w dy.
    # fp32 input: the cast boundary of line 252 is the identity, no rounding of g.
    g = dy * w
    mean_gx = tl.sum(g * x, axis=0) / H
    dx = rstd * (g - x * (rstd * rstd * mean_gx))
    dr = tl.load(DR + offs, mask=cmask & (has_dr != 0), other=0.0).to(tl.float32)
    tl.store(DX + offs, dx + dr, mask=cmask)


def _launch_fwd(x, delta, weight, eps):
    rows = x.shape[0]
    y = torch.empty_like(x)
    r = torch.empty_like(x) if delta is not None else y  # dummy pointer, store masked off
    rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
    if rows:
        with torch.cuda.device(x.device):
            soup_addnorm_fwd_kernel[(rows,)](
                x, delta if delta is not None else x, weight, y, r, rstd,
                1 if delta is not None else 0, eps,
                H=HIDDEN_SIZE, BLOCK=BLOCK, num_warps=NUM_WARPS,
            )
    return (r if delta is not None else None), y, rstd


def _launch_bwd(dy, dr, x, weight, rstd):
    rows = x.shape[0]
    dx = torch.empty_like(x)
    if rows:
        with torch.cuda.device(x.device):
            soup_addnorm_bwd_kernel[(rows,)](
                dy, dr if dr is not None else dy, x, weight, rstd, dx,
                1 if dr is not None else 0,
                H=HIDDEN_SIZE, BLOCK=BLOCK, num_warps=NUM_WARPS,
            )
    return dx


def _grad(t):
    # Our outputs are fp32, so autograd hands fp32 grads; only contiguity can differ (e.g. a
    # grad produced by expand/transpose downstream).
    return t if t is None or t.is_contiguous() else t.contiguous()


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        _, y, rstd = _launch_fwd(x, None, weight, eps)
        ctx.save_for_backward(x, rstd, weight)
        return y

    @staticmethod
    def backward(ctx, dy):
        # Only reached when x requires grad: the weight is frozen and eps is a float.
        x, rstd, weight = ctx.saved_tensors
        return _launch_bwd(_grad(dy), None, x, weight, rstd), None, None


class _AddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, weight, eps):
        r, y, rstd = _launch_fwd(x, delta, weight, eps)
        ctx.save_for_backward(r, rstd, weight)
        ctx.set_materialize_grads(False)
        return r, y

    @staticmethod
    def backward(ctx, dr, dy):
        r, rstd, weight = ctx.saved_tensors
        if dy is None:
            d = _grad(dr)
        else:
            d = _launch_bwd(_grad(dy), _grad(dr), r, weight, rstd)
        # d(residual + delta) = (1, 1): base AddBackward also hands the same grad to both inputs.
        need_x, need_delta = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        return (d if need_x else None), (d if need_delta else None), None, None


def norm_ok(x, weight):
    """True when (x, weight) fit the kernels: CUDA fp32 [..., 1536] input, frozen fp32 weight."""
    return (
        isinstance(x, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.float32
        and x.ndim >= 1
        and x.shape[-1] == HIDDEN_SIZE
        and weight.dtype == torch.float32
        and weight.ndim == 1
        and weight.numel() == HIDDEN_SIZE
        and weight.device == x.device
        and not weight.requires_grad
        and weight.is_contiguous()
    )


def rmsnorm(x, weight, eps):
    """Qwen2RMSNorm on an fp32 residual stream. Caller checks norm_ok(x, weight)."""
    shape = x.shape
    x2 = (x if x.is_contiguous() else x.contiguous()).view(-1, HIDDEN_SIZE)
    return _RMSNorm.apply(x2, weight, float(eps)).view(shape)


def add_rmsnorm(residual, delta, weight, eps):
    """(residual + delta, RMSNorm(residual + delta)) in one kernel. Caller checks norm_ok and that
    delta is a CUDA fp16/fp32 tensor of the same shape and device as residual."""
    shape = residual.shape
    if delta.dtype != torch.float32:
        delta = delta.to(torch.float32)  # exact; autograd rounds its grad back like base AddBackward
    x2 = (residual if residual.is_contiguous() else residual.contiguous()).view(-1, HIDDEN_SIZE)
    d2 = (delta if delta.is_contiguous() else delta.contiguous()).view(-1, HIDDEN_SIZE)
    r, y = _AddRMSNorm.apply(x2, d2, weight, float(eps))
    return r.view(shape), y.view(shape)
