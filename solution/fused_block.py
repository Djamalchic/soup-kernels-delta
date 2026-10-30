"""B (v6 peak-timing): the second half of a packed decoder layer as one autograd node.

On U's packed path (`solution/unpad.py` `_layer_packed`) v5 runs, per layer, on N real rows:

    r_mid, y = K3 add_rmsnorm(r, a)        saves r_mid  (fp32 [N, 1536], 4 B/elem)
    out      = S4 MLP(y)                   saves x16    (fp16 [N, 1536], 2 B/elem)
    r_next   = r_mid + f32(out)

while the layer input r (fp32) is already saved by input_layernorm (K3 rmsnorm), and a (the fp16
attention output, packed) is saved by nothing. So per layer autograd keeps r, r_mid and x16:
10 bytes per element of [N, 1536].

`SoupAddNormMLP` replaces those three ops with one node that saves (r, a, rstd): r is the very
tensor input_layernorm already keeps (no new memory), a costs 2 bytes per element, and in backward
it rebuilds everything else with the same kernels and the same operands:

  1. y   = K3 fwd(r, a)       store_r = 0; r + f32(a) is the forward's single fp32 add, bit for bit
  2. x16 = fp16(y)            S4's cast (`mlp_recompute.py`, the `x.to(torch.float16)` of forward)
  3. dx16 = S4 backward(fp16(d r_next), x16)   (`mlp_recompute._down_input_grad`,
                                               `mlp_recompute._gate_up_input_grad`)
  4. d r_mid = K3 bwd(f32(dx16), dr = d r_next, X = r, D = a, has_delta = 1)
  5. grads: d r_mid for r, fp16(d r_mid) for a   (base: AddBackward hands the same fp32 grad to
     both addends, ToCopyBackward rounds the fp16 one; v5's `_AddRMSNorm.backward` does the same)

Saved per layer: 10 -> 6 bytes per element of [N, 1536] (-4 B: about -7.9 MiB per layer and
-221 MiB over 28 layers at N = 1350, the N the v4 -> v5 peak difference implies [estimate]).
Extra work per layer in backward: one K3 fwd launch over [N, 1536] and one fp32 -> fp16 cast.

Every value is produced by the same op on the same operands as v5's path, so forward outputs and
all gradients are bit-identical to it (tests/test_fused_block.py checks torch.equal). The
chain-rule bookkeeping that autograd did between the three v5 nodes is reproduced explicitly:
  r_next = r_mid + f32(out16)      ->  d r_mid += d r_next,  d out16 = fp16(d r_next)
  x16 = fp16(y)                    ->  dy = f32(dx16)
  r_mid = r + f32(a)               ->  dr = d r_mid,  da = fp16(d r_mid)

Guards (the caller, `unpad._layer_packed`, checks `fused_ok`): USE_FUSED_BLOCK; S4 is installed
and on with RECOMPUTE_GATE (so v5 would have taken exactly S4's path) and S4's own guard passes;
the MLP module is an exact, hook-free Qwen2MLP; post_attention_layernorm is K3-eligible; r is a
contiguous CUDA fp32 [N, 1536] and a a contiguous fp16 tensor of the same shape on the same device.
Anything else takes v5's ops unchanged. No Triton kernel lives in this file and nothing touches
CUDA at import time.

Idea credit: activation recomputation (Chen et al. 2016, arXiv:1604.06174); saving the residual
inputs of an add+norm instead of its output follows the "recompute the cheap, keep the small"
pattern of fused-norm kernels in Liger-Kernel and Unsloth. Code written from scratch.
"""

import torch

from . import mlp_recompute as _s4

USE_FUSED_BLOCK = False  # read by unpad._layer_packed; solution.apply() switches it on

_MARK = "_soup_delta_fused_block"


class SoupAddNormMLP(torch.autograd.Function):
    """r_next = r + f32(a) + f32(MLP(fp16(RMSNorm(r + f32(a))))) saving only (r, a, rstd)."""

    @staticmethod
    def forward(ctx, r, a16, norm_weight, eps, gate_proj, up_proj, down_proj):
        from .kernels import addnorm as k3

        ctx.set_materialize_grads(False)
        # Grad mode is off in here: nothing below is recorded.
        r_mid, y, rstd = k3._launch_fwd(r, a16, norm_weight, eps)  # v5: add_rmsnorm
        x16 = y.to(torch.float16)  # v5: S4 forward's x.to(torch.float16)
        del y
        out16 = _s4.mlp_forward_nograd(x16, gate_proj, up_proj, down_proj)  # v5: S4 forward
        del x16
        # v5: S4 returns out.to(fp32) and the layer adds r_mid + that. The mixed-dtype in-place add
        # widens out16 exactly and adds in fp32: the same single rounding. r_mid is ours alone.
        r_mid.add_(out16)
        del out16
        ctx.save_for_backward(r, a16, rstd, norm_weight)
        ctx.eps = eps
        ctx.layers = (gate_proj, up_proj, down_proj)
        return r_mid

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_next):
        from .kernels import addnorm as k3

        n_in = 7
        if d_next is None or not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return (None,) * n_in
        r, a16, rstd, norm_weight = ctx.saved_tensors
        gate_proj, up_proj, down_proj = ctx.layers
        d_next = k3._grad(d_next)

        # d out16 = fp16(d r_next) (v5: ToCopyBackward of out.to(fp32)).
        d_out16 = d_next.to(torch.float16)
        # 1-2. x16 = fp16(K3(r + f32(a))): the forward's kernel on the forward's operands.
        _, y, _ = k3._launch_fwd(r, a16, norm_weight, ctx.eps, store_r=False)
        x16 = y.to(torch.float16)
        del y
        # 3. S4 backward (v5: SoupMLPRecompute.backward with the same helpers).
        dh_box = [_s4._down_input_grad(d_out16, down_proj)]
        del d_out16
        dx16 = _s4._gate_up_input_grad(dh_box, x16, gate_proj, up_proj)
        del x16
        # 4. dy = f32(dx16) (v5: ToCopyBackward of x.to(fp16)); K3 bwd rebuilds r_mid = r + f32(a).
        dy = dx16.float()
        del dx16
        d_mid = k3._launch_bwd(dy, d_next, r, norm_weight, rstd, delta=a16)
        del dy
        # 5. The same fp32 grad to both addends; fp16 rounding for a.
        dr = d_mid if ctx.needs_input_grad[0] else None
        da = d_mid.to(a16.dtype) if ctx.needs_input_grad[1] else None
        return dr, da, None, None, None, None, None


def _hookless(module):
    for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
        if getattr(module, name, None):
            return False
    return "forward" not in module.__dict__


def fused_ok(layer, r, a, mlp_cls, norm_cls):
    """True when SoupAddNormMLP reproduces v5's add_rmsnorm + S4 MLP + residual add for this layer.

    Reads only. `mlp_cls` / `norm_cls` are the real Qwen2MLP / Qwen2RMSNorm classes the caller
    captured at patch time (Soup's tests may swap sys.modules entries later).
    """
    if not USE_FUSED_BLOCK or not _s4.USE_MLP_RECOMPUTE or not _s4.RECOMPUTE_GATE:
        return False
    if _s4._swiglu_forward is None:  # S4 never installed
        return False
    mlp = getattr(layer, "mlp", None)
    ln2 = getattr(layer, "post_attention_layernorm", None)
    if type(mlp) is not mlp_cls or not _hookless(mlp):
        return False
    # v5 would call layer.mlp(y) -> Qwen2MLP.forward; only when that is S4's forward is S4's path
    # (and so this node) what v5 runs.
    if not getattr(mlp_cls.forward, _s4._PATCH_MARK, False):
        return False
    if type(ln2) is not norm_cls or not _hookless(ln2):
        return False
    eps = getattr(ln2, "variance_epsilon", None)
    if not isinstance(eps, float):
        return False
    from .kernels import addnorm as k3

    if not k3.norm_ok(r, getattr(ln2, "weight", None)):
        return False
    if not (isinstance(a, torch.Tensor) and a.dtype == torch.float16 and a.is_cuda
            and a.device == r.device and a.shape == r.shape and r.dim() == 2):
        return False
    if not (r.is_contiguous() and a.is_contiguous()):
        return False
    if not torch.is_grad_enabled() or not (a.requires_grad or r.requires_grad):
        return False
    # S4's own guard (K4 device/width/activation/layer types, autocast, global and layer hooks,
    # frozen nf4 weights without bias). y would be fp32 and require grad exactly when r or a does;
    # the guard reads only its device, width, dtype class and requires_grad, which a shares.
    probe = a if a.requires_grad else r
    return _s4._reject_reason(mlp, probe) is None


def add_norm_mlp(r, a, layer):
    """The fused node for one layer; the caller has checked fused_ok."""
    ln2 = layer.post_attention_layernorm
    mlp = layer.mlp
    return SoupAddNormMLP.apply(r, a, ln2.weight, float(ln2.variance_epsilon),
                                mlp.gate_proj, mlp.up_proj, mlp.down_proj)
